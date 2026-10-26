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
SNAPSHOT = {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "job", "namespace": "default"},
            "spec": {"containers": [{"name": "train", "image": "img"}]}}


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
         mock.patch.object(handlers.scheduler, "snapshot_pod", mgr.snapshot_pod), \
         mock.patch.object(handlers.scheduler, "get_node_info", mgr.get_node_info), \
         mock.patch.object(handlers.scheduler, "recreate_pod", mgr.recreate_pod), \
         mock.patch.object(handlers.sqs_publisher, "publish_risk_event", mgr.publish), \
         mock.patch.object(handlers, "trigger_s3_checkpoint", mgr.trigger), \
         mock.patch.object(handlers, "wait_for_checkpoint_flush", mgr.wait), \
         mock.patch.object(handlers.httpx, "get", mgr.http_get):
        mgr.list_job_pods.return_value = [_pod("Running", "node-a")]
        mgr.get_pod_node.return_value = "node-a"
        mgr.snapshot_pod.return_value = SNAPSHOT
        mgr.recreate_pod.return_value = True
        mgr.get_node_info.return_value = {"instance_type": "c5.large", "zone": "eu-north-1b", "age_min": 7.5}
        mgr.wait.return_value = True
        mgr.trigger.return_value = 'trigger-id-1'
        mgr.http_get.return_value = _risk(0.5)
        yield mgr


def _call_names(mgr):
    return [c[0] for c in mgr.mock_calls if c[0] in
            {"trigger", "wait", "cordon_node", "reschedule_pod", "publish", "snapshot_pod", "recreate_pod"}]


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


def test_no_second_migration_during_the_cooldown(env):
    # found on the real cluster: the pool's risk stays high after a migration, so without a cooldown the
    # operator deleted the replacement pod on every poll and it never came up
    patch = _patch()
    status = {"lastMigrationAt": __import__("time").time() - 60}
    handlers.reconcile(spec=_spec(), name="job", namespace="default", status=status, patch=patch)
    assert _call_names(env) == []
    assert "phase" not in patch.status


def test_migrates_again_once_the_cooldown_has_passed(env):
    patch = _patch()
    status = {"lastMigrationAt": __import__("time").time() - handlers.MIGRATION_COOLDOWN_SECONDS - 5}
    handlers.reconcile(spec=_spec(), name="job", namespace="default", status=status, patch=patch)
    assert "reschedule_pod" in _call_names(env)
    assert patch.status["lastMigrationAt"] > status["lastMigrationAt"]


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
    return ClientError({"Error": {"Code": code, "Message": "x"}}, "GetObject")


def _body(text):
    return {"Body": mock.Mock(read=lambda: text.encode())}


def test_trigger_writes_a_fresh_id_to_the_trigger_key():
    s3 = mock.Mock()
    with mock.patch.object(handlers, "_boto3_client", return_value=s3):
        a = handlers.trigger_s3_checkpoint("job", CKPT)
        b = handlers.trigger_s3_checkpoint("job", CKPT)
    assert a != b and len(a) >= 16
    kwargs = s3.put_object.call_args_list[0].kwargs
    assert kwargs["Key"] == "checkpoints/job/_FLUSH_TRIGGER" and kwargs["Body"] == a.encode()


def test_wait_returns_true_once_the_ack_carries_our_id():
    s3 = mock.Mock()
    s3.get_object.side_effect = [_client_error("NoSuchKey"), _body("someone-elses-id"), _body("mine")]
    with mock.patch.object(handlers, "_boto3_client", return_value=s3), \
         mock.patch.object(handlers, "CHECKPOINT_WAIT_POLL_SECONDS", 0.001):
        assert handlers.wait_for_checkpoint_flush(CKPT, "mine", timeout=5) is True
    assert s3.get_object.call_args.kwargs["Key"] == "checkpoints/job/_FLUSH_ACK"


def test_wait_returns_false_on_timeout_and_never_accepts_a_stale_ack():
    s3 = mock.Mock()
    s3.get_object.return_value = _body("an-older-request")
    with mock.patch.object(handlers, "_boto3_client", return_value=s3), \
         mock.patch.object(handlers, "CHECKPOINT_WAIT_POLL_SECONDS", 0.001):
        assert handlers.wait_for_checkpoint_flush(CKPT, "mine", timeout=0.05) is False


def test_migration_recreates_the_deleted_pod_from_a_snapshot(env):
    # the training pod is a bare Pod, so before this fix nothing brought it back after the operator deleted it
    patch = _patch()
    handlers.reconcile(spec=_spec(), name="job", namespace="default", status={}, patch=patch)
    order = _call_names(env)
    assert order.index("snapshot_pod") < order.index("reschedule_pod") < order.index("recreate_pod")
    assert env.recreate_pod.call_args.args[0] == SNAPSHOT
    assert patch.status["podSnapshot"] == SNAPSHOT


def test_pod_still_terminating_is_recreated_on_a_later_poll(env):
    env.list_job_pods.return_value = []          # old pod finally gone, replacement not created yet
    patch = _patch()
    status = {"podSnapshot": SNAPSHOT, "lastMigrationAt": __import__("time").time() - 60}
    handlers.reconcile(spec=_spec(), name="job", namespace="default", status=status, patch=patch)
    env.recreate_pod.assert_called_once_with(SNAPSHOT, "default")
    env.http_get.assert_not_called()


def test_no_recreate_while_a_job_pod_exists(env):
    env.http_get.return_value = _risk(0.0001)
    handlers.reconcile(spec=_spec(), name="job", namespace="default", status={"podSnapshot": SNAPSHOT},
                       patch=_patch())
    env.recreate_pod.assert_not_called()


def test_recreate_error_does_not_fail_the_migration(env):
    env.recreate_pod.side_effect = RuntimeError("api down")
    patch = _patch()
    handlers.reconcile(spec=_spec(), name="job", namespace="default", status={}, patch=patch)
    assert patch.status["phase"] == "Running"
    assert patch.status["podSnapshot"] == SNAPSHOT


def test_clean_pod_manifest_drops_what_the_cluster_injected():
    live = {
        "apiVersion": "v1", "kind": "Pod",
        "metadata": {"name": "job", "namespace": "ml", "uid": "u-1", "resourceVersion": "42",
                     "creationTimestamp": "2026-10-14T09:00:00Z", "labels": {"argus.io/job": "job"},
                     "ownerReferences": [{"kind": "X"}]},
        "spec": {
            "nodeName": "node-a", "priority": 0, "serviceAccountName": "trainer",
            "volumes": [{"name": "dataset", "emptyDir": {}}, {"name": "kube-api-access-abcde", "projected": {}},
                        {"name": "aws-iam-token", "projected": {}}],
            "initContainers": [{"name": "fetch", "image": "aws",
                                "volumeMounts": [{"name": "dataset", "mountPath": "/d"},
                                                 {"name": "kube-api-access-abcde", "mountPath": "/var/run/secrets"}]}],
            "containers": [{"name": "train", "image": "img",
                            "env": [{"name": "EPOCHS", "value": "50"}, {"name": "AWS_ROLE_ARN", "value": "arn"},
                                    {"name": "AWS_WEB_IDENTITY_TOKEN_FILE", "value": "/t"}],
                            "volumeMounts": [{"name": "aws-iam-token", "mountPath": "/t"}]}],
        },
        "status": {"phase": "Running"},
    }
    out = handlers.scheduler.clean_pod_manifest(live)
    assert out["metadata"] == {"name": "job", "namespace": "ml", "labels": {"argus.io/job": "job"}}
    assert "status" not in out and "nodeName" not in out["spec"] and "priority" not in out["spec"]
    assert [v["name"] for v in out["spec"]["volumes"]] == ["dataset"]
    assert out["spec"]["initContainers"][0]["volumeMounts"] == [{"name": "dataset", "mountPath": "/d"}]
    assert out["spec"]["containers"][0]["env"] == [{"name": "EPOCHS", "value": "50"}]
    assert out["spec"]["containers"][0]["volumeMounts"] == []
    assert out["spec"]["serviceAccountName"] == "trainer"
    assert live["spec"]["nodeName"] == "node-a"   # input not modified


def _reclaim(score):
    resp = mock.Mock()
    resp.json.return_value = {"reclaim_risk": score}
    resp.raise_for_status.return_value = None
    return resp


def test_capacity_signal_asks_reclaim_risk_with_the_real_node(env):
    spec = _spec(threshold=0.16)
    spec["riskSignal"] = "capacity"
    env.http_get.return_value = _reclaim(0.05)
    patch = _patch()
    handlers.reconcile(spec=spec, name="job", namespace="default", status={}, patch=patch)
    url, params = env.http_get.call_args.args[0], env.http_get.call_args.kwargs["params"]
    assert url.endswith("/reclaim_risk")
    # the node's own labels win over the spec's guesses
    assert params == {"instance_type": "c5.large", "az": "eu-north-1b", "instance_age_min": 7.5}
    assert patch.status["lastRiskScore"] == 0.05
    assert _call_names(env) == []


def test_capacity_signal_above_threshold_migrates(env):
    spec = _spec(threshold=0.16)
    spec["riskSignal"] = "capacity"
    env.http_get.return_value = _reclaim(0.4)
    handlers.reconcile(spec=spec, name="job", namespace="default", status={}, patch=_patch())
    assert "reschedule_pod" in _call_names(env)


def test_capacity_signal_waits_while_the_pod_has_no_node(env):
    spec = _spec()
    spec["riskSignal"] = "capacity"
    env.get_pod_node.return_value = None
    handlers.reconcile(spec=spec, name="job", namespace="default", status={}, patch=_patch())
    env.http_get.assert_not_called()


def test_price_signal_is_the_default_and_uses_predict(env):
    env.http_get.return_value = _risk(0.0001)
    handlers.reconcile(spec=_spec(), name="job", namespace="default", status={}, patch=_patch())
    assert env.http_get.call_args.args[0].endswith("/predict")
    env.get_node_info.assert_not_called()


# ------------------------------------------------------------------------------------------- gang jobs

def _member(name, node, phase="Running"):
    return SimpleNamespace(metadata=SimpleNamespace(name=name), status=SimpleNamespace(phase=phase),
                           spec=SimpleNamespace(node_name=node))


def _gang_spec(threshold=0.05):
    spec = _spec(threshold)
    spec["gang"] = True
    return spec


def _snap(name):
    return {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": name, "namespace": "default"}, "spec": {}}


@pytest.fixture
def gang(env):
    env.list_job_pods.return_value = [_member("job-0", "node-a"), _member("job-1", "node-b"), _member("job-2", "node-c")]
    env.snapshot_pod.side_effect = lambda n, ns: _snap(n)
    # only node-b is risky
    env.get_node_info.side_effect = lambda node: {"instance_type": f"type-{node}", "zone": "eu-north-1a", "age_min": 5.0}
    env.http_get.side_effect = lambda url, params, timeout: _risk(0.5 if params["instance_type"] == "type-node-b" else 0.001)
    return env


def test_gang_reads_every_member_node(gang):
    patch = _patch()
    handlers.reconcile(spec=_gang_spec(threshold=0.9), name="job", namespace="default", status={}, patch=patch)
    asked = sorted(c.kwargs["params"]["instance_type"] for c in gang.http_get.call_args_list)
    assert asked == ["type-node-a", "type-node-b", "type-node-c"]
    assert patch.status["lastRiskScore"] == 0.5
    assert _call_names(gang) == []


def test_gang_moves_every_member_after_one_flush(gang):
    patch = _patch()
    handlers.reconcile(spec=_gang_spec(), name="job", namespace="default", status={}, patch=patch)
    order = _call_names(gang)
    assert order.count("trigger") == 1 and order.count("wait") == 1
    assert order.index("wait") < order.index("reschedule_pod")
    deleted = [c.args[0] for c in gang.reschedule_pod.call_args_list]
    recreated = [c.args[0]["metadata"]["name"] for c in gang.recreate_pod.call_args_list]
    assert deleted == recreated == ["job-0", "job-1", "job-2"]
    assert order.index("reschedule_pod") < order.index("recreate_pod")  # all deleted before any comes back


def test_gang_cordons_only_the_risky_node(gang):
    patch = _patch()
    handlers.reconcile(spec=_gang_spec(), name="job", namespace="default", status={}, patch=patch)
    assert [c.args[0] for c in gang.cordon_node.call_args_list] == ["node-b"]
    assert patch.status["cordonedNodes"] == ["node-b"]
    assert len(patch.status["podSnapshots"]) == 3


def test_gang_respects_the_cooldown(gang):
    status = {"lastMigrationAt": __import__("time").time() - 60}
    handlers.reconcile(spec=_gang_spec(), name="job", namespace="default", status=status, patch=_patch())
    assert _call_names(gang) == []


def test_gang_recreates_missing_members_on_a_later_poll(gang):
    gang.list_job_pods.return_value = [_member("job-0", "node-d")]
    status = {"podSnapshots": [_snap("job-0"), _snap("job-1"), _snap("job-2")]}
    handlers.reconcile(spec=_gang_spec(), name="job", namespace="default", status=status, patch=_patch())
    assert sorted(c.args[0]["metadata"]["name"] for c in gang.recreate_pod.call_args_list) == ["job-1", "job-2"]
    gang.trigger.assert_not_called()  # no new migration while the gang is incomplete


def test_gang_uncordons_once_every_member_runs_elsewhere(gang):
    gang.list_job_pods.return_value = [_member("job-0", "node-d"), _member("job-1", "node-e"), _member("job-2", "node-a")]
    gang.http_get.side_effect = lambda url, params, timeout: _risk(0.001)
    status = {"cordonedNodes": ["node-b"], "cordonedAt": __import__("time").time(),
              "podSnapshots": [_snap("job-0"), _snap("job-1"), _snap("job-2")]}
    patch = _patch()
    handlers.reconcile(spec=_gang_spec(), name="job", namespace="default", status=status, patch=patch)
    gang.uncordon_node.assert_called_once_with("node-b")
    assert patch.status["cordonedNodes"] is None


def test_gang_keeps_cordon_while_a_member_is_not_back(gang):
    gang.list_job_pods.return_value = [_member("job-0", "node-d"), _member("job-1", "node-e"),
                                       _member("job-2", None, phase="Pending")]
    gang.http_get.side_effect = lambda url, params, timeout: _risk(0.001)
    status = {"cordonedNodes": ["node-b"], "cordonedAt": __import__("time").time(),
              "podSnapshots": [_snap("job-0"), _snap("job-1"), _snap("job-2")]}
    handlers.reconcile(spec=_gang_spec(), name="job", namespace="default", status=status, patch=_patch())
    gang.uncordon_node.assert_not_called()


def test_gang_capacity_signal_uses_reclaim_risk(gang):
    spec = _gang_spec(threshold=0.16)
    spec["riskSignal"] = "capacity"
    gang.http_get.side_effect = lambda url, params, timeout: _reclaim(0.01)
    handlers.reconcile(spec=spec, name="job", namespace="default", status={}, patch=_patch())
    assert all(c.args[0].endswith("/reclaim_risk") for c in gang.http_get.call_args_list)
    assert len(gang.http_get.call_args_list) == 3
