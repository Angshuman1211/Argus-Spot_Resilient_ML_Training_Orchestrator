"""
Argus Operator — kopf event handlers.

Reconcile loop: poll the risk score, and above the threshold ask the pod to flush a checkpoint, wait for
its acknowledgement, cordon the node, delete the pod and recreate it from a snapshot, publish an SQS event.

Checkpoint handshake (see trigger_s3_checkpoint): the operator writes a fresh id to _FLUSH_TRIGGER, the
pod saves model.pt and writes the same id to _FLUSH_ACK. This keeps the operator decoupled from the
training process internals and needs no s3:DeleteObject.

Gang jobs (spec.gang: true, see _reconcile_gang) move all pods of a synchronous multi-node job together.
"""

import logging
import os
import time
import uuid

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
# After a migration the pool's risk is usually still high (the score describes the instance pool, not the pod),
# so without a cooldown the operator would migrate again on the very next poll. 900 s is the prediction horizon.
MIGRATION_COOLDOWN_SECONDS = int(os.environ.get("MIGRATION_COOLDOWN_SECONDS", "900"))
MAX_FAILURE_RETRIES = int(os.environ.get("MAX_FAILURE_RETRIES", "3"))
UNCORDON_AFTER_SECONDS = int(os.environ.get("UNCORDON_AFTER_SECONDS", "600"))
DEFAULT_AZ = os.environ.get("DEFAULT_AZ", "eu-north-1a")
CHECKPOINT_WAIT_SECONDS = float(os.environ.get("CHECKPOINT_WAIT_SECONDS", "60"))
CHECKPOINT_WAIT_POLL_SECONDS = float(os.environ.get("CHECKPOINT_WAIT_POLL_SECONDS", "2"))
# How long a migration waits for the old pod to terminate before recreating it; if it is still going,
# the next reconcile recreates it instead.
RECREATE_WAIT_SECONDS = float(os.environ.get("RECREATE_WAIT_SECONDS", "120"))  # pod grace period is 90 s

# When AWS_ENDPOINT_URL is set (local dev via .env.local), boto3 hits LocalStack.
# On EKS (Week 6) the var is unset and boto3 hits real AWS via IRSA credentials.
_AWS_ENDPOINT = os.environ.get("AWS_ENDPOINT_URL")


def _boto3_client(service: str):
    kwargs = {"region_name": os.environ.get("AWS_DEFAULT_REGION", "eu-north-1")}
    if _AWS_ENDPOINT:
        kwargs["endpoint_url"] = _AWS_ENDPOINT
    return boto3.client(service, **kwargs)


def _checkpoint_keys(checkpoint_path: str):
    path_clean = checkpoint_path.replace("s3://", "")
    bucket, prefix = path_clean.split("/", 1)
    return bucket, f"{prefix}/_FLUSH_TRIGGER", f"{prefix}/_FLUSH_ACK"


def trigger_s3_checkpoint(job_name: str, checkpoint_path: str) -> str:
    """
    Asks the training pod to flush a checkpoint and returns the request's unique id.

    Protocol (needs only PutObject and GetObject, because the pod role cannot delete objects):
      1. operator writes {prefix}/_FLUSH_TRIGGER containing a fresh id
      2. the pod notices an id it has not handled, saves model.pt, then writes the same id to
         {prefix}/_FLUSH_ACK
      3. the operator waits until _FLUSH_ACK holds its id (wait_for_checkpoint_flush)
    """
    bucket, trigger_key, _ = _checkpoint_keys(checkpoint_path)
    trigger_id = uuid.uuid4().hex
    s3 = _boto3_client("s3")
    try:
        s3.put_object(Bucket=bucket, Key=trigger_key, Body=trigger_id.encode())
        logger.info(f"[CHECKPOINT] Trigger {trigger_id[:8]} written -> s3://{bucket}/{trigger_key}")
    except Exception as e:
        logger.error(f"[CHECKPOINT] Failed to write trigger for '{job_name}': {e}")
        raise
    return trigger_id


def wait_for_checkpoint_flush(checkpoint_path: str, trigger_id: str, timeout: float = CHECKPOINT_WAIT_SECONDS) -> bool:
    """
    Blocks until the training pod acknowledges this trigger by writing its id to _FLUSH_ACK, which it does
    only after save_checkpoint() succeeded. Returns False on timeout; the caller must then not assume a
    checkpoint exists.
    """
    bucket, _, ack_key = _checkpoint_keys(checkpoint_path)
    s3 = _boto3_client("s3")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            body = s3.get_object(Bucket=bucket, Key=ack_key)["Body"].read().decode().strip()
            if body == trigger_id:
                return True
        except ClientError as e:
            if e.response["Error"]["Code"] not in ("404", "NoSuchKey", "NotFound"):
                logger.warning(f"[CHECKPOINT] reading ack failed while waiting: {e}")
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
    if spec.get("gang"):
        _reconcile_gang(spec, name, namespace, status, patch, pods)
        return
    _maybe_uncordon(name, status, patch, pods)

    # A migration deleted the pod but could not recreate it yet (old pod still terminating)
    snapshot = status.get("podSnapshot")
    if snapshot and not pods:
        try:
            if scheduler.recreate_pod(snapshot, namespace):
                logger.info(f"[RECONCILE] '{name}' recreated its training pod")
        except Exception as e:
            logger.error(f"[RECONCILE] '{name}' could not recreate its training pod: {e}")
        return

    # 1. Poll prediction service
    signal = spec.get("riskSignal", "price")
    try:
        if signal == "capacity":
            node = scheduler.get_pod_node(name, namespace)
            if not node:
                logger.info(f"[RECONCILE] '{name}' pod has no node yet - no capacity risk to read")
                return
            info = scheduler.get_node_info(node)
            instance_type = info["instance_type"] or instance_type
            az = info["zone"] or az
            if info["age_min"] is None:
                raise ValueError(f"node '{node}' has no creation time")
            response = httpx.get(
                f"{PREDICT_SERVICE_URL}/reclaim_risk",
                params={"instance_type": instance_type, "az": az, "instance_age_min": round(info["age_min"], 2)},
                timeout=10.0,
            )
        else:
            response = httpx.get(
                f"{PREDICT_SERVICE_URL}/predict",
                params={"instance_type": instance_type, "az": az},
                timeout=10.0,
            )
        response.raise_for_status()
        risk = response.json()
    except Exception as e:
        predict_failures.labels(job_name=name).inc()
        logger.warning(f"[RECONCILE] '{name}' - {signal} risk unavailable: {e}")
        return

    risk_score = risk["reclaim_risk"] if signal == "capacity" else risk["risk_score"]
    patch.status["lastRiskScore"] = risk_score
    risk_score_gauge.labels(job_name=name, instance_type=instance_type, az=az).set(risk_score)

    logger.info(
        f"[RECONCILE] '{name}' | risk={risk_score:.4f} threshold={risk_threshold} | phase={current_phase}"
    )

    if risk_score <= risk_threshold:
        return

    since = time.time() - float(status.get("lastMigrationAt", 0))
    if since < MIGRATION_COOLDOWN_SECONDS:
        logger.info(f"[RECONCILE] '{name}' risk above threshold but migrated {since:.0f}s ago - cooling down")
        return

    # 2. Risk exceeded: checkpoint, confirm the flush, then migrate
    logger.warning(f"[RECONCILE] '{name}' risk {risk_score:.4f} > {risk_threshold} - triggering migration")

    try:
        trigger_id = trigger_s3_checkpoint(name, checkpoint_path)
        patch.status["lastCheckpointStep"] = last_step + 1
        checkpoint_count.labels(job_name=name).inc()
    except Exception as e:
        logger.error(f"[RECONCILE] '{name}' checkpoint trigger failed: {e}")
        patch.status["phase"] = "Failed"
        return

    # Do not kill the pod until it has actually flushed. The trigger is only polled
    # every 100 batches, so deleting immediately races the flush and loses work.
    if wait_for_checkpoint_flush(checkpoint_path, trigger_id):
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

    # 4. Cordon node, delete the pod and recreate it from a snapshot (it is a bare Pod, so nothing else would)
    try:
        snapshot = scheduler.snapshot_pod(name, namespace) or status.get("podSnapshot")
        if snapshot:
            patch.status["podSnapshot"] = snapshot
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
    if snapshot:
        try:
            scheduler.recreate_pod(snapshot, namespace, wait_seconds=RECREATE_WAIT_SECONDS)
        except Exception as e:  # the next reconcile retries from status.podSnapshot
            logger.warning(f"[RECONCILE] '{name}' recreate failed, will retry: {e}")
    else:
        logger.warning(f"[RECONCILE] '{name}' had no pod to snapshot - it will not be recreated")

    patch.status["phase"] = "Running"
    patch.status["retryCount"] = 0
    patch.status["lastMigrationAt"] = time.time()
    logger.info(f"[RECONCILE] '{name}' migration complete - back to Running")


# ------------------------------------------------------------------------------------------- gang jobs
# A synchronous multi-node job (spec.gang: true) is one SpotResilientJob with several pods labelled
# argus.io/job=<name>. A reclaim of any node stops all of them, so they are moved together: one flush,
# then every pod is deleted and recreated and the whole job resumes from the same checkpoint. Only the
# nodes whose risk crossed the threshold are cordoned; the rest stay usable for the replacements.


def _node_risk(signal, node, spec):
    """Risk score of one node from the prediction service, or None if it cannot be read."""
    info = scheduler.get_node_info(node)
    instance_type = info["instance_type"] or spec["instanceFallback"][0]
    az = info["zone"] or spec.get("availabilityZone", DEFAULT_AZ)
    if signal == "capacity":
        if info["age_min"] is None:
            raise ValueError(f"node '{node}' has no creation time")
        r = httpx.get(f"{PREDICT_SERVICE_URL}/reclaim_risk",
                      params={"instance_type": instance_type, "az": az, "instance_age_min": round(info["age_min"], 2)},
                      timeout=10.0)
        r.raise_for_status()
        return r.json()["reclaim_risk"], instance_type, az
    r = httpx.get(f"{PREDICT_SERVICE_URL}/predict", params={"instance_type": instance_type, "az": az}, timeout=10.0)
    r.raise_for_status()
    return r.json()["risk_score"], instance_type, az


def _gang_uncordon(name, status, patch, pods):
    nodes = status.get("cordonedNodes") or []
    if not nodes:
        return
    expected = len(status.get("podSnapshots") or []) or 1
    running_elsewhere = [p for p in pods if p.status.phase == "Running" and p.spec.node_name not in nodes]
    timed_out = time.time() - float(status.get("cordonedAt", 0)) > UNCORDON_AFTER_SECONDS
    if len(running_elsewhere) < expected and not timed_out:
        return
    left = []
    for node in nodes:
        try:
            scheduler.uncordon_node(node)
        except Exception as e:
            logger.warning(f"[GANG] '{name}' uncordon of '{node}' failed: {e}")
            left.append(node)
    patch.status["cordonedNodes"] = left or None
    if not left:
        patch.status["cordonedAt"] = None


def _reconcile_gang(spec, name, namespace, status, patch, pods):
    _gang_uncordon(name, status, patch, pods)

    # recreate members a migration deleted but could not recreate yet
    snapshots = status.get("podSnapshots") or []
    present = {p.metadata.name for p in pods}
    missing = [s for s in snapshots if s["metadata"]["name"] not in present]
    if missing:
        for s in missing:
            try:
                scheduler.recreate_pod(s, namespace)
            except Exception as e:
                logger.error(f"[GANG] '{name}' could not recreate '{s['metadata']['name']}': {e}")
        return

    threshold = spec.get("riskThreshold", RISK_THRESHOLD_DEFAULT)
    signal = spec.get("riskSignal", "price")
    risks = {}
    for p in pods:
        node = p.spec.node_name
        if not node or node in risks:
            continue
        try:
            risks[node] = _node_risk(signal, node, spec)
        except Exception as e:
            predict_failures.labels(job_name=name).inc()
            logger.warning(f"[GANG] '{name}' - {signal} risk of node '{node}' unavailable: {e}")
    if not risks:
        return
    worst = max(risks, key=lambda n: risks[n][0])
    score, instance_type, az = risks[worst]
    patch.status["lastRiskScore"] = score
    risk_score_gauge.labels(job_name=name, instance_type=instance_type, az=az).set(score)
    risky = sorted(n for n, (s, _, _) in risks.items() if s > threshold)
    logger.info(f"[GANG] '{name}' | {len(pods)} pods on {len(risks)} nodes | max risk={score:.4f} "
                f"threshold={threshold} | risky nodes={risky}")
    if not risky:
        return
    since = time.time() - float(status.get("lastMigrationAt", 0))
    if since < MIGRATION_COOLDOWN_SECONDS:
        logger.info(f"[GANG] '{name}' risk above threshold but migrated {since:.0f}s ago - cooling down")
        return

    try:
        trigger_id = trigger_s3_checkpoint(name, spec["checkpointPath"])
        patch.status["lastCheckpointStep"] = status.get("lastCheckpointStep", 0) + 1
        checkpoint_count.labels(job_name=name).inc()
    except Exception as e:
        logger.error(f"[GANG] '{name}' checkpoint trigger failed: {e}")
        patch.status["phase"] = "Failed"
        return
    if wait_for_checkpoint_flush(spec["checkpointPath"], trigger_id):
        logger.info(f"[GANG] '{name}' checkpoint confirmed on S3")
    else:
        checkpoint_wait_timeouts.labels(job_name=name).inc()
        logger.warning(f"[GANG] '{name}' checkpoint not confirmed - moving anyway, relying on SIGTERM handlers")
    try:
        sqs_publisher.publish_risk_event(job_name=name, risk_score=score, instance_type=instance_type, az=az,
                                         action="checkpoint_and_migrate_gang")
    except Exception as e:
        logger.warning(f"[GANG] '{name}' SQS publish failed (non-fatal): {e}")

    try:
        snaps = [s for s in (scheduler.snapshot_pod(p.metadata.name, namespace) for p in pods) if s] or snapshots
        patch.status["podSnapshots"] = snaps
        for node in risky:
            scheduler.cordon_node(node)
        patch.status["cordonedNodes"] = sorted(set(status.get("cordonedNodes") or []) | set(risky))
        patch.status["cordonedAt"] = time.time()
        for p in pods:  # every member restarts, so the job resumes from one consistent checkpoint
            scheduler.reschedule_pod(p.metadata.name, namespace, spec["instanceFallback"])
    except Exception as e:
        logger.error(f"[GANG] '{name}' move failed: {e}")
        patch.status["phase"] = "Failed"
        return
    for s in snaps:
        try:
            scheduler.recreate_pod(s, namespace, wait_seconds=RECREATE_WAIT_SECONDS)
        except Exception as e:  # the next reconcile recreates the missing members from status.podSnapshots
            logger.warning(f"[GANG] '{name}' recreate of '{s['metadata']['name']}' failed, will retry: {e}")
    patch.status["phase"] = "Running"
    patch.status["retryCount"] = 0
    patch.status["lastMigrationAt"] = time.time()
    logger.info(f"[GANG] '{name}' moved all {len(pods)} pods, cordoned {risky}")
