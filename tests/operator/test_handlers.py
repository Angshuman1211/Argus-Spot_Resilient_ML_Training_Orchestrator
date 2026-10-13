"""Unit tests for the operator reconcile loop. No cluster, AWS or network needed."""
import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
from botocore.exceptions import ClientError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "operator"))

from controller import handlers  # noqa: E402

CKPT = "s3://bucket/checkpoints/job"


def _spec(threshold=0.05):
    return {
        "image": "img",
        "checkpointPath": CKPT,
        "riskThreshold": threshold,
        "instanceFallback": ["m5.xlarge", "c5.xlarge"],
    }


def _pod(phase, node):
    return SimpleNamespace(status=SimpleNamespace(phase=phase), spec=SimpleNamespace(node_name=node))


def _patch():
    return SimpleNamespace(status={})


def _risk(score):
    resp = mock.Mock()
    resp.json.return_value = {"risk_score": score}
    resp.raise_for_status.return_value = None
    return resp


@pytest.fixture
def env():
    """Patches every external call and records their order on one manager."""
    mgr = mock.Mock()
    with mock.patch.object(handlers.scheduler, "list_job_pods", mgr.list_job_pods), \
         mock.patch.object(handlers.scheduler, "get_pod_node", mgr.get_pod_node), \
         mock.patch.object(handlers.scheduler, "cordon_node", mgr.cordon_node), \
         mock.patch.object(handlers.scheduler, "uncordon_node", mgr.uncordon_node), \
         mock.patch.object(handlers.scheduler, "reschedule_pod", mgr.reschedule_pod), \
         mock.patch.object(handlers.sqs_publisher, "publish_risk_event", mgr.publish), \
         mock.patch.object(handlers, "trigger_s3_checkpoint", mgr.trigger), \
         mock.patch.object(handlers, "wait_for_checkpoint_flush", mgr.wait), \
         mock.patch.object(handlers.httpx, "get", mgr.http_get):
        mgr.list_job_pods.return_value = [_pod("Running", "node-a")]
        mgr.get_pod_node.return_value = "node-a"
        mgr.wait.return_value = True
        mgr.http_get.return_value = _risk(0.01)
        yield mgr


def _call_names(mgr):
    return [c[0] for c in mgr.mock_calls if c[0] in
            {"trigger", "wait", "cordon_node", "reschedule_pod", "publish"}]


def test_below_threshold_does_nothing(env):
    env.http_get.return_value = _risk(0.0001)
    patch = _patch()
    handlers.reconcile(spec=_spec(), name="job", namespace="default", status={}, patch=patch)
    assert _call_names(env) == []
    assert patch.status["lastRiskScore"] == 0.0001


def test_checkpoint_confirmed_before_pod_deleted(env):
    patch = _patch()
    handlers.reconcile(spec=_spec(), name="job", namespace="default", status={}, patch=patch)
    order = _call_names(env)
    assert order.index("trigger") < order.index("wait") < order.index("cordon_node")
    assert order.index("wait") < order.index("reschedule_pod")
    assert patch.status["phase"] == "Running"
    assert patch.status["cordonedNode"] == "node-a"


def test_unconfirmed_flush_is_counted_and_still_migrates(env):
    env.wait.return_value = False
    patch = _patch()
    with mock.patch.object(handlers.checkpoint_wait_timeouts, "labels") as labels:
        handlers.reconcile(spec=_spec(), name="job", namespace="default", status={}, patch=patch)
    labels.assert_called_once_with(job_name="job")
    labels.return_value.inc.assert_called_once()
    env.reschedule_pod.assert_called_once()


def test_trigger_failure_marks_failed_and_does_not_kill_pod(env):
    env.trigger.side_effect = RuntimeError("s3 down")
    patch = _patch()
    handlers.reconcile(spec=_spec(), name="job", namespace="default", status={}, patch=patch)
    assert patch.status["phase"] == "Failed"
    env.reschedule_pod.assert_not_called()
    env.cordon_node.assert_not_called()


def test_predict_service_down_is_a_noop(env):
    env.http_get.side_effect = RuntimeError("down")
    patch = _patch()
    handlers.reconcile(spec=_spec(), name="job", namespace="default", status={}, patch=patch)
    assert _call_names(env) == []


def test_availability_zone_comes_from_spec(env):
    spec = _spec()
    spec["availabilityZone"] = "eu-north-1c"
    handlers.reconcile(spec=spec, name="job", namespace="default", status={}, patch=_patch())
    assert env.http_get.call_args.kwargs["params"]["az"] == "eu-north-1c"


def test_failed_job_retries_then_gives_up(env):
    patch = _patch()
    handlers.reconcile(spec=_spec(), name="job", namespace="default",
                       status={"phase": "Failed", "retryCount": 0}, patch=patch)
    assert patch.status["phase"] == "Running" and patch.status["retryCount"] == 1

    patch = _patch()
    handlers.reconcile(spec=_spec(), name="job", namespace="default",
                       status={"phase": "Failed", "retryCount": handlers.MAX_FAILURE_RETRIES}, patch=patch)
    assert patch.status == {}


def test_uncordons_when_replacement_runs_elsewhere(env):
    env.list_job_pods.return_value = [_pod("Running", "node-b")]
    env.http_get.return_value = _risk(0.0001)
    patch = _patch()
    status = {"cordonedNode": "node-a", "cordonedAt": 9999999999}
    handlers.reconcile(spec=_spec(), name="job", namespace="default", status=status, patch=patch)
    env.uncordon_node.assert_called_once_with("node-a")
    assert patch.status["cordonedNode"] is None


def test_keeps_cordon_while_pod_still_on_cordoned_node(env):
    env.list_job_pods.return_value = [_pod("Running", "node-a")]
    env.http_get.return_value = _risk(0.0001)
    status = {"cordonedNode": "node-a", "cordonedAt": 9999999999}
    handlers.reconcile(spec=_spec(), name="job", namespace="default", status=status, patch=_patch())
    env.uncordon_node.assert_not_called()


def test_succeeded_pod_marks_job_succeeded(env):
    env.list_job_pods.return_value = [_pod("Succeeded", "node-a")]
    patch = _patch()
    handlers.reconcile(spec=_spec(), name="job", namespace="default", status={}, patch=patch)
    assert patch.status["phase"] == "Succeeded"
    env.http_get.assert_not_called()


def _client_error(code):
    return ClientError({"Error": {"Code": code, "Message": "x"}}, "HeadObject")


def test_wait_returns_true_once_trigger_is_consumed():
    s3 = mock.Mock()
    s3.head_object.side_effect = [{}, {}, _client_error("404")]
    with mock.patch.object(handlers, "_boto3_client", return_value=s3), \
         mock.patch.object(handlers, "CHECKPOINT_WAIT_POLL_SECONDS", 0.001):
        assert handlers.wait_for_checkpoint_flush(CKPT, timeout=5) is True
    assert s3.head_object.call_args.kwargs["Key"] == "checkpoints/job/_FLUSH_TRIGGER"


def test_wait_returns_false_on_timeout():
    s3 = mock.Mock()
    s3.head_object.return_value = {}
    with mock.patch.object(handlers, "_boto3_client", return_value=s3), \
         mock.patch.object(handlers, "CHECKPOINT_WAIT_POLL_SECONDS", 0.001):
        assert handlers.wait_for_checkpoint_flush(CKPT, timeout=0.05) is False
