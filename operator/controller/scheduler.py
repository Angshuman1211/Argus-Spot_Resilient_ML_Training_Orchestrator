"""
Kubernetes scheduler operations — cordon risky nodes and reschedule pods.

Uses the kubernetes Python client which is already configured by kopf
to talk to whichever cluster is active (Minikube locally, EKS in prod).
"""

import copy
import logging
import time

from kubernetes import client

logger = logging.getLogger(__name__)

# Fields the API server or admission webhooks add to a pod. Copying them into a new pod either fails
# (spec.priority, duplicate volumes) or pins it to the node we are leaving (spec.nodeName).
_INJECTED_VOLUME_PREFIXES = ("kube-api-access-", "aws-iam-token")
_INJECTED_ENV = {"AWS_ROLE_ARN", "AWS_WEB_IDENTITY_TOKEN_FILE", "AWS_STS_REGIONAL_ENDPOINTS"}
_DROPPED_SPEC_FIELDS = ("nodeName", "priority")


def cordon_node(node_name: str) -> None:
    """
    Marks the node as unschedulable so no new pods land on it.
    Existing pods continue running until evicted or deleted.
    """
    v1 = client.CoreV1Api()
    body = {"spec": {"unschedulable": True}}
    v1.patch_node(node_name, body)
    logger.info(f"[SCHEDULER] Cordoned node '{node_name}'")


def uncordon_node(node_name: str) -> None:
    """Reverses a cordon — used if migration completes without interruption."""
    v1 = client.CoreV1Api()
    body = {"spec": {"unschedulable": False}}
    v1.patch_node(node_name, body)
    logger.info(f"[SCHEDULER] Uncordoned node '{node_name}'")


def get_pod_node(pod_name: str, namespace: str = "default") -> str | None:
    """Returns the node name the pod is currently running on."""
    v1 = client.CoreV1Api()
    try:
        pod = v1.read_namespaced_pod(name=pod_name, namespace=namespace)
        return pod.spec.node_name
    except client.exceptions.ApiException:
        return None


def reschedule_pod(
    pod_name: str,
    namespace: str,
    fallback_instance_types: list[str],
) -> None:
    """
    Deletes the pod. The training pod is a bare Pod named after the job, so nothing else recreates it:
    the operator does that from a snapshot (snapshot_pod / recreate_pod), and the cordon keeps the new
    pod off the node being left. On a single-node cluster (Minikube) the new pod stays Pending until the
    operator uncordons the node after UNCORDON_AFTER_SECONDS.
    """
    v1 = client.CoreV1Api()

    logger.info(
        f"[SCHEDULER] Deleting pod '{pod_name}' in '{namespace}' "
        f"— will reschedule to fallback: {fallback_instance_types}"
    )

    try:
        v1.delete_namespaced_pod(
            name=pod_name,
            namespace=namespace,
            # No grace override: use the pod's terminationGracePeriodSeconds so the
            # SIGTERM handler can flush a checkpoint if the S3 trigger was not consumed.
            body=client.V1DeleteOptions(),
        )
        logger.info(f"[SCHEDULER] Pod '{pod_name}' deleted — Kubernetes will reschedule it")
    except client.exceptions.ApiException as e:
        if e.status == 404:
            logger.warning(f"[SCHEDULER] Pod '{pod_name}' not found — already gone")
        else:
            raise


def get_node_info(node_name: str) -> dict:
    """
    Instance type, zone and age (minutes) of a node, from its well-known labels and creation time. A Spot
    node registers a minute or two after its instance launches, so the age is a slight underestimate.
    """
    v1 = client.CoreV1Api()
    node = v1.read_node(node_name)
    labels = node.metadata.labels or {}
    created = node.metadata.creation_timestamp
    return {
        "instance_type": labels.get("node.kubernetes.io/instance-type"),
        "zone": labels.get("topology.kubernetes.io/zone"),
        "age_min": (time.time() - created.timestamp()) / 60 if created else None,
    }


def clean_pod_manifest(pod: dict) -> dict:
    """
    Turns a live pod (as a dict) into a manifest that can be created again: identity, labels and spec
    only, minus everything the cluster filled in. Pure function, so it is unit-testable.
    """
    meta = pod.get("metadata", {})
    spec = copy.deepcopy(pod.get("spec", {}))
    for field in _DROPPED_SPEC_FIELDS:
        spec.pop(field, None)

    def injected(name):
        return any(name.startswith(p) for p in _INJECTED_VOLUME_PREFIXES)

    spec["volumes"] = [v for v in spec.get("volumes") or [] if not injected(v.get("name", ""))]
    if not spec["volumes"]:
        spec.pop("volumes")
    for key in ("initContainers", "containers"):
        for c in spec.get(key) or []:
            if c.get("volumeMounts") is not None:
                c["volumeMounts"] = [m for m in c["volumeMounts"] if not injected(m.get("name", ""))]
            if c.get("env") is not None:
                c["env"] = [e for e in c["env"] if e.get("name") not in _INJECTED_ENV]

    out_meta = {"name": meta["name"], "namespace": meta.get("namespace", "default")}
    for key in ("labels", "annotations"):
        if meta.get(key):
            out_meta[key] = dict(meta[key])
    return {"apiVersion": "v1", "kind": "Pod", "metadata": out_meta, "spec": spec}


def snapshot_pod(pod_name: str, namespace: str = "default") -> dict | None:
    """Returns a re-creatable manifest of the job's pod, or None if it does not exist."""
    v1 = client.CoreV1Api()
    try:
        pod = v1.read_namespaced_pod(name=pod_name, namespace=namespace)
    except client.exceptions.ApiException as e:
        if e.status == 404:
            return None
        raise
    return clean_pod_manifest(client.ApiClient().sanitize_for_serialization(pod))


def recreate_pod(manifest: dict, namespace: str, wait_seconds: float = 0, poll_seconds: float = 2) -> bool:
    """
    Creates the pod from a snapshot once the old pod of the same name is gone. Returns False if the old
    pod is still terminating after wait_seconds; the next reconcile then tries again.
    """
    v1 = client.CoreV1Api()
    name = manifest["metadata"]["name"]
    deadline = time.time() + wait_seconds
    while True:
        try:
            v1.read_namespaced_pod(name=name, namespace=namespace)
        except client.exceptions.ApiException as e:
            if e.status != 404:
                raise
            break
        if time.time() >= deadline:
            logger.info(f"[SCHEDULER] Pod '{name}' still terminating - will recreate on a later poll")
            return False
        time.sleep(poll_seconds)
    try:
        v1.create_namespaced_pod(namespace=namespace, body=manifest)
    except client.exceptions.ApiException as e:
        if e.status == 409:  # someone (or a previous poll) already recreated it
            return True
        raise
    logger.info(f"[SCHEDULER] Recreated pod '{name}' - it resumes from the latest checkpoint")
    return True


def list_job_pods(job_name: str, namespace: str = "default") -> list:
    """Returns pods associated with a SpotResilientJob by label selector."""
    v1 = client.CoreV1Api()
    pods = v1.list_namespaced_pod(
        namespace=namespace,
        label_selector=f"argus.io/job={job_name}",
    )
    return pods.items
