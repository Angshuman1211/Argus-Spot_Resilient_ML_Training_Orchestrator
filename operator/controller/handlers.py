"""
Argus Operator — kopf event handlers.

Week 4: stubs only.
Week 5: full reconcile loop — risk polling, S3 checkpoint trigger, cordon, reschedule, SQS.

Checkpoint trigger mechanism (Person B's design):
  Operator writes _FLUSH_TRIGGER marker to S3 → training pod polls every 100 batches →
  detects marker → saves model.pt → deletes marker.
  This keeps the operator decoupled from the training process internals.
"""

import logging
import os
import time

import boto3
import httpx
import kopf
from botocore.exceptions import ClientError

from controller import scheduler, sqs_publisher
from controller.metrics import (
    checkpoint_count,
    checkpoint_wait_timeouts,
    predict_failures,
    risk_score_gauge,
    job_completion_count,
)

logger = logging.getLogger(__name__)

PREDICT_SERVICE_URL = os.environ.get("PREDICT_SERVICE_URL", "http://localhost:8000")
RISK_THRESHOLD_DEFAULT = float(os.environ.get("RISK_THRESHOLD", "0.05"))
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL_SECONDS", "60"))
MAX_FAILURE_RETRIES = int(os.environ.get("MAX_FAILURE_RETRIES", "3"))
UNCORDON_AFTER_SECONDS = int(os.environ.get("UNCORDON_AFTER_SECONDS", "600"))
DEFAULT_AZ = os.environ.get("DEFAULT_AZ", "eu-north-1a")
CHECKPOINT_WAIT_SECONDS = float(os.environ.get("CHECKPOINT_WAIT_SECONDS", "60"))
CHECKPOINT_WAIT_POLL_SECONDS = float(os.environ.get("CHECKPOINT_WAIT_POLL_SECONDS", "2"))

# When AWS_ENDPOINT_URL is set (local dev via .env.local), boto3 hits LocalStack.
# On EKS (Week 6) the var is unset and boto3 hits real AWS via IRSA credentials.
_AWS_ENDPOINT = os.environ.get("AWS_ENDPOINT_URL")


def _boto3_client(service: str):
    kwargs = {"region_name": os.environ.get("AWS_DEFAULT_REGION", "eu-north-1")}
    if _AWS_ENDPOINT:
        kwargs["endpoint_url"] = _AWS_ENDPOINT
    return boto3.client(service, **kwargs)


def trigger_s3_checkpoint(job_name: str, checkpoint_path: str) -> None:
    """
    Writes a _FLUSH_TRIGGER marker to S3 so the training pod saves a checkpoint.

    Trigger key: {prefix}/_FLUSH_TRIGGER
    where prefix is derived from checkpointPath: s3://{bucket}/{prefix}

    The training pod (train.py) polls for this key every 100 batches,
    flushes model.pt on detection, then deletes the marker.
    """
    path_clean = checkpoint_path.replace("s3://", "")
    bucket, prefix = path_clean.split("/", 1)
    trigger_key = f"{prefix}/_FLUSH_TRIGGER"

    s3 = _boto3_client("s3")
    try:
        s3.put_object(Bucket=bucket, Key=trigger_key, Body=b"1")
        logger.info(f"[CHECKPOINT] Trigger written → s3://{bucket}/{trigger_key}")
    except Exception as e:
        logger.error(f"[CHECKPOINT] Failed to write trigger for '{job_name}': {e}")
        raise


def wait_for_checkpoint_flush(checkpoint_path: str, timeout: float = CHECKPOINT_WAIT_SECONDS) -> bool:
    """
    Blocks until the training pod has consumed the _FLUSH_TRIGGER marker.

    The pod deletes the marker only after save_checkpoint() finishes, so a missing
    marker means the checkpoint is on S3. Returns False on timeout, in which case
    the caller must not assume a checkpoint exists.
    """
    path_clean = checkpoint_path.replace("s3://", "")
    bucket, prefix = path_clean.split("/", 1)
    trigger_key = f"{prefix}/_FLUSH_TRIGGER"

    s3 = _boto3_client("s3")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            s3.head_object(Bucket=bucket, Key=trigger_key)
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return True
            logger.warning(f"[CHECKPOINT] head_object failed while waiting: {e}")
        time.sleep(CHECKPOINT_WAIT_POLL_SECONDS)
    return False


@kopf.on.create("argus.io", "v1", "spotresilientjobs")
def on_create(spec, name, namespace, patch, **kwargs):
    logger.info(f"[CREATE] SpotResilientJob '{name}' in namespace '{namespace}'")
    logger.info(f"  image:            {spec['image']}")
    logger.info(f"  checkpointPath:   {spec['checkpointPath']}")
    logger.info(f"  riskThreshold:    {spec.get('riskThreshold', RISK_THRESHOLD_DEFAULT)}")
    logger.info(f"  instanceFallback: {spec['instanceFallback']}")

    patch.status["phase"] = "Running"
    patch.status["lastRiskScore"] = 0.0
    patch.status["lastCheckpointStep"] = 0


@kopf.on.update("argus.io", "v1", "spotresilientjobs")
def on_update(spec, name, namespace, old, new, diff, **kwargs):
    logger.info(f"[UPDATE] SpotResilientJob '{name}' updated")
    for op, field, old_val, new_val in diff:
        logger.info(f"  {op} {'.'.join(str(f) for f in field)}: {old_val!r} → {new_val!r}")


@kopf.on.delete("argus.io", "v1", "spotresilientjobs")
def on_delete(spec, name, namespace, **kwargs):
    logger.info(f"[DELETE] SpotResilientJob '{name}' deleted")


def _maybe_uncordon(name, status, patch, pods):
    """
    Releases a node this operator cordoned during an earlier migration.

    Without this, a cordon is permanent: on a single-node cluster the replacement pod
    can never schedule, and on EKS the cordoned node is lost to the job forever.
    Uncordon once a replacement pod is Running on another node, or after a timeout.
    """
    node = status.get("cordonedNode")
    if not node:
        return
    replacement_ok = any(
        p.status.phase == "Running" and p.spec.node_name != node for p in pods
    )
    timed_out = time.time() - float(status.get("cordonedAt", 0)) > UNCORDON_AFTER_SECONDS
    if not (replacement_ok or timed_out):
        return
    try:
        scheduler.uncordon_node(node)
    except Exception as e:
        logger.warning(f"[RECONCILE] '{name}' uncordon of '{node}' failed: {e}")
        return
    patch.status["cordonedNode"] = None
    patch.status["cordonedAt"] = None


@kopf.timer("argus.io", "v1", "spotresilientjobs", interval=POLL_INTERVAL)
def reconcile(spec, name, namespace, status, patch, **kwargs):
    current_phase = status.get("phase", "Running")
    risk_threshold = spec.get("riskThreshold", RISK_THRESHOLD_DEFAULT)
    fallback_types = spec["instanceFallback"]
    checkpoint_path = spec["checkpointPath"]
    last_step = status.get("lastCheckpointStep", 0)
    instance_type = fallback_types[0]
    az = spec.get("availabilityZone", DEFAULT_AZ)

    if current_phase == "Failed":
        retries = status.get("retryCount", 0)
        if retries >= MAX_FAILURE_RETRIES:
            logger.error(f"[RECONCILE] '{name}' stays Failed after {retries} retries")
            return
        patch.status["phase"] = "Running"
        patch.status["retryCount"] = retries + 1
        logger.warning(f"[RECONCILE] '{name}' Failed - retry {retries + 1}/{MAX_FAILURE_RETRIES}")
        return

    if current_phase in ("Checkpointing", "Migrating", "Succeeded"):
        logger.info(f"[RECONCILE] '{name}' skipped - already in phase '{current_phase}'")
        return

    # Check for job completion and release any node we cordoned earlier
    pods = []
    try:
        pods = scheduler.list_job_pods(name, namespace)
    except Exception as e:
        logger.warning(f"[RECONCILE] Failed to list pods for '{name}': {e}")
    if any(p.status.phase == "Succeeded" for p in pods):
        patch.status["phase"] = "Succeeded"
        job_completion_count.labels(job_name=name).inc()
        logger.info(f"[RECONCILE] '{name}' job completed successfully")
        return
    _maybe_uncordon(name, status, patch, pods)

    # 1. Poll prediction service
    try:
        response = httpx.get(
            f"{PREDICT_SERVICE_URL}/predict",
            params={"instance_type": instance_type, "az": az},
            timeout=10.0,
        )
        response.raise_for_status()
        risk = response.json()
    except Exception as e:
        predict_failures.labels(job_name=name).inc()
        logger.warning(f"[RECONCILE] '{name}' - predict service unreachable: {e}")
        return

    risk_score = risk["risk_score"]
    patch.status["lastRiskScore"] = risk_score
    risk_score_gauge.labels(job_name=name, instance_type=instance_type, az=az).set(risk_score)

    logger.info(
        f"[RECONCILE] '{name}' | risk={risk_score:.4f} threshold={risk_threshold} | phase={current_phase}"
    )

    if risk_score <= risk_threshold:
        return

    # 2. Risk exceeded: checkpoint, confirm the flush, then migrate
    logger.warning(f"[RECONCILE] '{name}' risk {risk_score:.4f} > {risk_threshold} - triggering migration")

    try:
        trigger_s3_checkpoint(name, checkpoint_path)
        patch.status["lastCheckpointStep"] = last_step + 1
        checkpoint_count.labels(job_name=name).inc()
    except Exception as e:
        logger.error(f"[RECONCILE] '{name}' checkpoint trigger failed: {e}")
        patch.status["phase"] = "Failed"
        return

    # Do not kill the pod until it has actually flushed. The trigger is only polled
    # every 100 batches, so deleting immediately races the flush and loses work.
    if wait_for_checkpoint_flush(checkpoint_path):
        logger.info(f"[RECONCILE] '{name}' checkpoint confirmed on S3")
    else:
        checkpoint_wait_timeouts.labels(job_name=name).inc()
        logger.warning(
            f"[RECONCILE] '{name}' checkpoint not confirmed after {CHECKPOINT_WAIT_SECONDS}s "
            f"- migrating anyway, relying on the pod's SIGTERM handler"
        )

    # 3. Publish SQS risk event
    try:
        sqs_publisher.publish_risk_event(
            job_name=name,
            risk_score=risk_score,
            instance_type=instance_type,
            az=az,
            action="checkpoint_and_migrate",
        )
    except Exception as e:
        logger.warning(f"[RECONCILE] '{name}' SQS publish failed (non-fatal): {e}")

    # 4. Cordon node + reschedule pod
    try:
        node_name = scheduler.get_pod_node(name, namespace)
        if node_name:
            scheduler.cordon_node(node_name)
            patch.status["cordonedNode"] = node_name
            patch.status["cordonedAt"] = time.time()
        scheduler.reschedule_pod(name, namespace, fallback_types)
    except Exception as e:
        logger.error(f"[RECONCILE] '{name}' reschedule failed: {e}")
        patch.status["phase"] = "Failed"
        return

    patch.status["phase"] = "Running"
    patch.status["retryCount"] = 0
    logger.info(f"[RECONCILE] '{name}' migration complete - back to Running")
