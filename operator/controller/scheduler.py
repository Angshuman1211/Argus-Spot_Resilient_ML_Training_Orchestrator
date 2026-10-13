"""
Kubernetes scheduler operations — cordon risky nodes and reschedule pods.

Uses the kubernetes Python client which is already configured by kopf
to talk to whichever cluster is active (Minikube locally, EKS in prod).
"""

import logging

from kubernetes import client

logger = logging.getLogger(__name__)


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
    Deletes the pod so Kubernetes reschedules it.

    The new pod will land on a non-cordoned node. On real EKS (Week 6),
    the Spot node group's instance types match the fallback list so K8s
    naturally picks a healthy instance.

    On Minikube there's only one node — the pod reschedules back to the
    same node, which is expected behaviour for local testing.
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


def list_job_pods(job_name: str, namespace: str = "default") -> list:
    """Returns pods associated with a SpotResilientJob by label selector."""
    v1 = client.CoreV1Api()
    pods = v1.list_namespaced_pod(
        namespace=namespace,
        label_selector=f"argus.io/job={job_name}",
    )
    return pods.items
