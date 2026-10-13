"""
The operator and the training job must agree on the flush handshake under the REAL IAM limits:
the pod role can put and get objects but cannot delete them. This test wires the actual operator
functions and the actual training-job functions to one fake S3 whose delete is denied.
"""
import importlib.util
import io
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("kopf")
from botocore.exceptions import ClientError  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "operator"))
from controller import handlers  # noqa: E402

CKPT_PATH = "s3://bucket/checkpoints/job"


class RoleLimitedS3:
    """Put and get work, delete is denied: the same limits as argus-operator-irsa."""

    def __init__(self):
        self.objects = {}
        self.deletes_attempted = 0

    def put_object(self, Bucket, Key, Body):
        self.objects[(Bucket, Key)] = Body

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "x"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def delete_object(self, Bucket, Key):
        self.deletes_attempted += 1
        raise ClientError({"Error": {"Code": "AccessDenied", "Message": "no s3:DeleteObject"}}, "DeleteObject")

    def upload_file(self, path, bucket, key):
        with open(path, "rb") as f:
            self.objects[(bucket, key)] = f.read()

    def copy_object(self, Bucket, Key, CopySource):
        self.objects[(Bucket, Key)] = self.objects[(CopySource["Bucket"], CopySource["Key"])]

    def download_file(self, bucket, key, path):
        with open(path, "wb") as f:
            f.write(self.objects[(bucket, key)])


@pytest.fixture
def world(tmp_path, monkeypatch):
    tv = types.ModuleType("torchvision")
    tv.transforms, tv.datasets = types.ModuleType("t"), types.ModuleType("d")
    monkeypatch.setitem(sys.modules, "torchvision", tv)
    monkeypatch.setitem(sys.modules, "torchvision.transforms", tv.transforms)
    monkeypatch.setitem(sys.modules, "torchvision.datasets", tv.datasets)
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location("job_train", os.path.join(ROOT, "ml", "cifar10_job", "train.py"))
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    s3 = RoleLimitedS3()
    monkeypatch.setattr(job, "s3_client", s3)
    monkeypatch.setattr(job.time, "sleep", lambda s: None)
    monkeypatch.setattr(handlers, "_boto3_client", lambda service: s3)
    monkeypatch.setattr(handlers, "CHECKPOINT_WAIT_POLL_SECONDS", 0.001)
    job.S3_BUCKET, job.S3_PREFIX = "bucket", "checkpoints/job/latest_checkpoint.pt"
    job.model = job.SimpleCNN()
    job.optimizer = torch.optim.SGD(job.model.parameters(), lr=0.001)
    job._last_trigger_id = None
    return types.SimpleNamespace(job=job, s3=s3)


def test_operator_request_is_flushed_and_acknowledged_without_any_delete(world):
    trigger_id = handlers.trigger_s3_checkpoint("job", CKPT_PATH)
    assert handlers.wait_for_checkpoint_flush(CKPT_PATH, trigger_id, timeout=0.05) is False  # nobody flushed yet

    world.job.check_flush_trigger(epoch=3)  # the pod's next poll

    assert handlers.wait_for_checkpoint_flush(CKPT_PATH, trigger_id, timeout=1) is True
    assert ("bucket", "checkpoints/job/latest_checkpoint.pt") in world.s3.objects
    assert world.s3.deletes_attempted == 0


def test_two_successive_requests_are_each_flushed_and_confirmed_separately(world):
    first = handlers.trigger_s3_checkpoint("job", CKPT_PATH)
    world.job.check_flush_trigger(epoch=1)
    assert handlers.wait_for_checkpoint_flush(CKPT_PATH, first, timeout=1)

    second = handlers.trigger_s3_checkpoint("job", CKPT_PATH)
    assert second != first
    assert handlers.wait_for_checkpoint_flush(CKPT_PATH, second, timeout=0.05) is False  # old ack is not enough
    world.job.check_flush_trigger(epoch=2)
    assert handlers.wait_for_checkpoint_flush(CKPT_PATH, second, timeout=1)


def test_a_failed_flush_is_never_reported_as_confirmed(world, monkeypatch):
    trigger_id = handlers.trigger_s3_checkpoint("job", CKPT_PATH)
    monkeypatch.setattr(world.s3, "upload_file", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("s3 down")))
    world.job.check_flush_trigger(epoch=1)
    assert handlers.wait_for_checkpoint_flush(CKPT_PATH, trigger_id, timeout=0.05) is False
