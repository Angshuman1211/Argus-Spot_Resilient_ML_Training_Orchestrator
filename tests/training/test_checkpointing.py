"""Checkpoint/resume semantics of ml/cifar10_job/train.py, with S3 faked in memory."""
import importlib
import importlib.util
import os
import sys
import types
from unittest import mock

import pytest

torch = pytest.importorskip("torch")

JOB_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "ml", "cifar10_job")


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.fail_uploads = 0

    def upload_file(self, path, bucket, key):
        if self.fail_uploads > 0:
            self.fail_uploads -= 1
            raise RuntimeError("upload failed")
        with open(path, "rb") as f:
            self.objects[(bucket, key)] = f.read()

    def copy_object(self, Bucket, Key, CopySource):
        self.objects[(Bucket, Key)] = self.objects[(CopySource["Bucket"], CopySource["Key"])]

    def download_file(self, bucket, key, path):
        if (bucket, key) not in self.objects:
            raise FileNotFoundError(key)
        with open(path, "wb") as f:
            f.write(self.objects[(bucket, key)])

    def get_object(self, Bucket, Key):
        import io
        if (Bucket, Key) not in self.objects:
            raise KeyError(Key)
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def put_object(self, Bucket, Key, Body):
        self.objects[(Bucket, Key)] = Body

    def delete_object(self, Bucket, Key):
        self.objects.pop((Bucket, Key), None)


@pytest.fixture
def job(tmp_path, monkeypatch):
    tv = types.ModuleType("torchvision")
    tv.transforms = types.ModuleType("torchvision.transforms")
    tv.datasets = types.ModuleType("torchvision.datasets")
    monkeypatch.setitem(sys.modules, "torchvision", tv)
    monkeypatch.setitem(sys.modules, "torchvision.transforms", tv.transforms)
    monkeypatch.setitem(sys.modules, "torchvision.datasets", tv.datasets)
    monkeypatch.chdir(tmp_path)
    # Load by path under a unique name: ml/model/train.py is also called "train".
    spec = importlib.util.spec_from_file_location(
        "cifar10_job_train", os.path.join(os.path.abspath(JOB_DIR), "train.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fake = FakeS3()
    monkeypatch.setattr(module, "s3_client", fake)
    monkeypatch.setattr(module.time, "sleep", lambda s: None)
    module.fake_s3 = fake
    yield module


def _fresh(job):
    model = job.SimpleCNN()
    return model, torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def test_completed_checkpoint_resumes_at_next_epoch(job):
    model, opt = _fresh(job)
    assert job.save_checkpoint(model, opt, 3, completed=True)
    m2, o2 = _fresh(job)
    assert job.load_checkpoint(m2, o2) == 4


def test_mid_epoch_checkpoint_resumes_same_epoch(job):
    model, opt = _fresh(job)
    assert job.save_checkpoint(model, opt, 3, completed=False)
    m2, o2 = _fresh(job)
    assert job.load_checkpoint(m2, o2) == 3


def test_resume_restores_weights(job):
    model, opt = _fresh(job)
    job.save_checkpoint(model, opt, 0, completed=True)
    m2, o2 = _fresh(job)
    job.load_checkpoint(m2, o2)
    for a, b in zip(model.state_dict().values(), m2.state_dict().values()):
        assert torch.equal(a, b)


def test_no_checkpoint_starts_fresh(job):
    model, opt = _fresh(job)
    assert job.load_checkpoint(model, opt) == 0


def test_upload_retries_then_succeeds(job):
    job.fake_s3.fail_uploads = 2
    model, opt = _fresh(job)
    assert job.save_checkpoint(model, opt, 1, completed=True) is True


def test_upload_failure_returns_false(job):
    job.fake_s3.fail_uploads = 99
    model, opt = _fresh(job)
    assert job.save_checkpoint(model, opt, 1, completed=True) is False
    assert (job.S3_BUCKET, job.S3_PREFIX) not in job.fake_s3.objects


def test_failed_upload_never_leaves_a_truncated_final_key(job):
    model, opt = _fresh(job)
    job.save_checkpoint(model, opt, 1, completed=True)
    good = job.fake_s3.objects[(job.S3_BUCKET, job.S3_PREFIX)]
    job.fake_s3.fail_uploads = 99
    job.save_checkpoint(model, opt, 2, completed=True)
    assert job.fake_s3.objects[(job.S3_BUCKET, job.S3_PREFIX)] == good


def _keys(job):
    prefix_dir = job.S3_PREFIX.rsplit("/", 1)[0]
    return (job.S3_BUCKET, prefix_dir + "/_FLUSH_TRIGGER"), (job.S3_BUCKET, prefix_dir + "/_FLUSH_ACK")


def test_ack_is_written_only_after_the_checkpoint_is_on_s3(job):
    model, opt = _fresh(job)
    job.model, job.optimizer, job._last_trigger_id = model, opt, None
    trigger, ack = _keys(job)
    job.fake_s3.objects[trigger] = b"req-1"

    job.fake_s3.fail_uploads = 99
    job.check_flush_trigger(2)
    assert ack not in job.fake_s3.objects  # a failed flush must not look confirmed

    job.fake_s3.fail_uploads = 0
    job.check_flush_trigger(2)
    assert job.fake_s3.objects[ack] == b"req-1"
    assert (job.S3_BUCKET, job.S3_PREFIX) in job.fake_s3.objects


def test_the_same_request_is_not_flushed_twice_and_a_new_one_is(job):
    model, opt = _fresh(job)
    job.model, job.optimizer, job._last_trigger_id = model, opt, None
    trigger, ack = _keys(job)
    job.fake_s3.objects[trigger] = b"req-1"
    job.check_flush_trigger(0)
    first = job.fake_s3.objects[(job.S3_BUCKET, job.S3_PREFIX)]

    job.fake_s3.objects[(job.S3_BUCKET, job.S3_PREFIX)] = b"sentinel"   # would be overwritten by a second flush
    job.check_flush_trigger(1)
    assert job.fake_s3.objects[(job.S3_BUCKET, job.S3_PREFIX)] == b"sentinel"

    job.fake_s3.objects[trigger] = b"req-2"
    job.check_flush_trigger(1)
    assert job.fake_s3.objects[ack] == b"req-2"
    assert job.fake_s3.objects[(job.S3_BUCKET, job.S3_PREFIX)] != b"sentinel"
    assert first != b"sentinel"


def test_a_pod_started_after_the_ack_does_not_repeat_the_flush(job):
    trigger, ack = _keys(job)
    job.fake_s3.objects[trigger] = b"req-1"
    job.fake_s3.objects[ack] = b"req-1"
    assert job._read_marker(ack.__getitem__(1)) == "req-1"
    job._last_trigger_id = job._read_marker(ack[1])  # what main() does at startup
    model, opt = _fresh(job)
    job.model, job.optimizer = model, opt
    job.check_flush_trigger(0)
    assert (job.S3_BUCKET, job.S3_PREFIX) not in job.fake_s3.objects  # nothing was saved


def test_sigterm_handler_only_sets_flag(job):
    with mock.patch.object(job, "save_checkpoint") as save:
        job.handle_sigterm()
    assert job.shutdown_requested is True
    save.assert_not_called()
