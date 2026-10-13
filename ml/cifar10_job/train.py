import os
import sys
import signal
import time

import boto3
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# S3_BUCKET and S3_PREFIX must align with the CRD's checkpointPath.
# checkpointPath: s3://argus-checkpoints-844641713781/checkpoints/cifar10-test
# → S3_BUCKET = argus-checkpoints-844641713781
# → S3_PREFIX = checkpoints/cifar10-test/latest_checkpoint.pt
# → trigger key = checkpoints/cifar10-test/_FLUSH_TRIGGER  (written by operator)
S3_BUCKET = os.environ.get("S3_BUCKET", "argus-checkpoints-844641713781")
S3_PREFIX = os.environ.get("S3_PREFIX", "checkpoints/cifar10-test/latest_checkpoint.pt")
LOCAL_CHECKPOINT_PATH = "checkpoint.pt"

s3_client = boto3.client(
    "s3",
    endpoint_url=os.environ.get("AWS_ENDPOINT_URL"),  # LocalStack in dev, unset on EKS
    region_name=os.environ.get("AWS_DEFAULT_REGION", "eu-north-1"),
)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 16 * 16, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 16 * 16)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


UPLOAD_RETRIES = int(os.environ.get("UPLOAD_RETRIES", "3"))


def save_checkpoint(model, optimizer, epoch, completed):
    """
    Saves a checkpoint locally and uploads it to S3. Returns True only if the upload
    succeeded, so callers never report a flush that did not happen.

    completed=True means `epoch` finished, so a resume starts at epoch + 1.
    completed=False is a mid-epoch save (flush trigger or SIGTERM): the weights have
    advanced, but the epoch is unfinished, so a resume re-runs `epoch` instead of
    skipping its remaining batches.

    The upload goes to a temporary key and is then copied over the real one, so a kill
    during the upload cannot leave a truncated latest_checkpoint.pt behind.
    """
    print(f"Saving checkpoint (epoch {epoch}, completed={completed}) -> s3://{S3_BUCKET}/{S3_PREFIX}")
    torch.save(
        {
            "epoch": epoch,
            "completed": completed,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        LOCAL_CHECKPOINT_PATH,
    )
    tmp_key = S3_PREFIX + ".tmp"
    size_mb = os.path.getsize(LOCAL_CHECKPOINT_PATH) / 1e6
    for attempt in range(1, UPLOAD_RETRIES + 1):
        try:
            t0 = time.time()
            s3_client.upload_file(LOCAL_CHECKPOINT_PATH, S3_BUCKET, tmp_key)
            s3_client.copy_object(
                Bucket=S3_BUCKET,
                Key=S3_PREFIX,
                CopySource={"Bucket": S3_BUCKET, "Key": tmp_key},
            )
            print(f"Checkpoint uploaded: {size_mb:.1f} MB in {time.time() - t0:.1f} s "
                  f"({size_mb / max(time.time() - t0, 1e-6):.1f} MB/s)", flush=True)
            return True
        except Exception as e:
            print(f"Checkpoint upload failed (attempt {attempt}/{UPLOAD_RETRIES}): {e}")
            time.sleep(min(2 ** attempt, 10))
    return False


def load_checkpoint(model, optimizer):
    print(f"Loading checkpoint from s3://{S3_BUCKET}/{S3_PREFIX}")
    try:
        s3_client.download_file(S3_BUCKET, S3_PREFIX, LOCAL_CHECKPOINT_PATH)
    except Exception as e:
        print(f"No checkpoint found - starting fresh. ({e})")
        return 0
    ckpt = torch.load(LOCAL_CHECKPOINT_PATH, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    # Checkpoints written before the `completed` flag existed were always end-of-epoch.
    start = ckpt["epoch"] + 1 if ckpt.get("completed", True) else ckpt["epoch"]
    print(f"Resuming from epoch {start}")
    return start


# EXTRA_PARAMS_MB pads the model with a large block of parameters so the checkpoint reaches a chosen size
# (weights plus SGD momentum, so the file is about twice this). Used to measure real flush times for large
# models on the real cluster; 0 (default) leaves the small CIFAR-10 CNN unchanged.
EXTRA_PARAMS_MB = int(os.environ.get("EXTRA_PARAMS_MB", "0"))

model = SimpleCNN()
if EXTRA_PARAMS_MB > 0:
    model.extra = nn.Parameter(torch.zeros(EXTRA_PARAMS_MB * 262144))  # 4 bytes per float32
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
current_epoch = 0


shutdown_requested = False


def handle_sigterm(*args):
    """Only sets a flag. Uploading to S3 from inside a signal handler can interrupt
    boto3 mid-call; the training loop checks the flag every batch and flushes there."""
    global shutdown_requested
    shutdown_requested = True
    print("SIGTERM received - will flush a checkpoint and exit at the next batch")


signal.signal(signal.SIGTERM, handle_sigterm)


def check_flush_trigger(epoch):
    """
    Checks if the operator dropped a _FLUSH_TRIGGER marker in S3.
    Trigger key mirrors the operator's trigger_s3_checkpoint() path:
        {prefix_dir}/_FLUSH_TRIGGER

    The marker is deleted only after a successful upload: the operator waits for the
    marker to disappear as its confirmation that the checkpoint is on S3.
    """
    prefix_dir = S3_PREFIX.rsplit("/", 1)[0]
    trigger_key = f"{prefix_dir}/_FLUSH_TRIGGER"
    try:
        s3_client.head_object(Bucket=S3_BUCKET, Key=trigger_key)
    except Exception:
        return
    print("Operator flush trigger detected - saving checkpoint early")
    if save_checkpoint(model, optimizer, epoch, completed=False):
        s3_client.delete_object(Bucket=S3_BUCKET, Key=trigger_key)


def main():
    global current_epoch

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

    start_epoch = load_checkpoint(model, optimizer)
    epochs = int(os.environ.get("EPOCHS", "5"))

    for epoch in range(start_epoch, epochs):
        current_epoch = epoch
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if EXTRA_PARAMS_MB > 0:
                loss = loss + 1e-12 * model.extra[:1024].sum()  # keeps the block in the graph and optimizer state
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:
                print(f"[epoch {epoch+1}, batch {i+1}] loss: {running_loss / 200:.3f}")
                running_loss = 0.0

            if shutdown_requested:
                save_checkpoint(model, optimizer, epoch, completed=False)
                sys.exit(0)

            if i % 100 == 0:
                check_flush_trigger(epoch)

        save_checkpoint(model, optimizer, epoch, completed=True)

    print("Training complete.")


if __name__ == "__main__":
    main()
