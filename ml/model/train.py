import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import random
import joblib
import numpy as np
import mlflow
import mlflow.pytorch
from sklearn.metrics import average_precision_score
from dataset import create_dataloaders
from transformer import SpotInterruptionPredictor
from losses import FocalLoss  # noqa: F401  (re-exported: tests and tune_focal_loss.py import it from here)
from feature_config import FEATURE_COLUMNS, SEQ_LENGTH, PREDICTION_HORIZON

def train_model(seed: int = None, output_dir: str = None, max_epochs: int = 25, run_name: str = None,
                 prediction_horizon: int = None, spike_threshold: float = 1.01, oversample: bool = False):
    """
    Executes a local training run of the Spot Predictor on the downloaded AWS data.

    seed: if set, seeds torch/numpy/random for a reproducible run (needed for
        multi_seed_eval.py to report a real mean/std instead of one noisy draw).
    output_dir: where to write the checkpoint/scaler/metadata. Defaults to this
        file's directory (the "shipped" location); multi_seed_eval.py points this
        at a scratch dir per seed so sweep runs don't clobber the shipped model.
    prediction_horizon/spike_threshold/oversample: label/sampling config, tuned by
        tune_data_config.py. Defaults (None horizon -> feature_config's SEQ_LENGTH-linked
        default, threshold=1.01, oversample=False) reproduce the originally shipped config.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    features_csv = os.path.join(base_dir, "../data/features.csv")
    output_dir = output_dir or base_dir

    if not os.path.exists(features_csv):
        print(f"Error: {features_csv} not found. Run dataset generation first.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing Training Loop on {device} (seed={seed})... This might take a bit.")

    # 1. Build DataLoader
    batch_size = 128
    horizon = prediction_horizon if prediction_horizon is not None else PREDICTION_HORIZON
    train_loader, val_loader, input_features, scaler = create_dataloaders(
        csv_path=features_csv,
        batch_size=batch_size,
        train_split=0.8,
        prediction_horizon=horizon,
        spike_threshold=spike_threshold,
        oversample=oversample,
    )

    # 2. Define Model (from transformer.py)
    # d_model=128, nhead=2, num_layers=2, lr=5e-4 — the config hyperparameter_tune.py's
    # grid search found best (see docs/B3_model_training.md), and ~2x cheaper per batch
    # than the 4-head/4-layer config this script used to hardcode.
    d_model, nhead, num_layers, lr = 128, 2, 2, 5e-4
    model = SpotInterruptionPredictor(num_features=input_features, d_model=d_model, nhead=nhead, num_layers=num_layers)
    model.to(device)

    # 3. Handle Imbalance with Focal Loss
    # alpha=0.75, gamma=1.0 — from tune_focal_loss.py's grid search (after fixing the
    # alpha bug: it used to be applied as a flat scalar to every sample regardless of
    # label, which is a no-op for class balance).
    focal_alpha, focal_gamma = 0.75, 1.0
    criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # 4. Training Loop — real run, not a smoke test. Early-stops on val PR-AUC, not val
    # loss: on a ~0.08% positive rate, val loss barely moves epoch to epoch (dominated
    # by the 99.9% negatives it already gets right), so it's a poor signal for whether
    # the model is actually getting better at the thing we care about — ranking the
    # rare positives higher.
    patience = 5

    # SQLite backend, not file:// — the raw path used to crash on Windows path
    # resolution (same issue documented for hyperparameter_tune.py, see
    # docs/B3_model_training.md).
    db_path = os.path.join(base_dir, "mlruns.db").replace("\\", "/")
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("Spot-Interruption-Predictor")

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "spot_transformer.pt")
    scaler_path = os.path.join(output_dir, "spot_scaler.joblib")
    metadata_path = os.path.join(output_dir, "model_metadata.json")

    best_val_pr_auc = -1.0
    epochs_without_improvement = 0

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("seed", seed)
        mlflow.log_param("max_epochs", max_epochs)
        mlflow.log_param("patience", patience)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("focal_alpha", focal_alpha)
        mlflow.log_param("focal_gamma", focal_gamma)
        mlflow.log_param("d_model", d_model)
        mlflow.log_param("nhead", nhead)
        mlflow.log_param("num_layers", num_layers)

        for epoch in range(max_epochs):
            model.train()
            running_loss = 0.0

            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()

                raw_logits = model(x)
                loss = criterion(raw_logits, y)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if batch_idx % 200 == 0:
                    print(f"Epoch {epoch+1}/{max_epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

            avg_train_loss = running_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

            # Validation — this loader existed before but was never actually used.
            model.eval()
            val_running_loss = 0.0
            val_probs, val_labels = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    raw_logits = model(x)
                    val_running_loss += criterion(raw_logits, y).item()
                    val_probs.extend(torch.sigmoid(raw_logits).cpu().numpy().flatten())
                    val_labels.extend(y.cpu().numpy().flatten())

            avg_val_loss = val_running_loss / len(val_loader)
            val_pr_auc = average_precision_score(val_labels, val_probs)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_pr_auc", val_pr_auc, step=epoch)

            print(f"==> Epoch {epoch+1} Complete. Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val PR-AUC: {val_pr_auc:.4f}\n")

            if val_pr_auc > best_val_pr_auc:
                best_val_pr_auc = val_pr_auc
                epochs_without_improvement = 0

                # 5. Save the best checkpoint + the exact preprocessing needed to serve it.
                torch.save(model.state_dict(), checkpoint_path)
                joblib.dump(scaler, scaler_path)
                with open(metadata_path, "w") as f:
                    json.dump({
                        "feature_columns": FEATURE_COLUMNS,
                        "seq_length": SEQ_LENGTH,
                        "prediction_horizon": horizon,
                        "spike_threshold": spike_threshold,
                        "oversample": oversample,
                        "num_features": input_features,
                        "d_model": d_model,
                        "nhead": nhead,
                        "num_layers": num_layers,
                        "focal_alpha": focal_alpha,
                        "focal_gamma": focal_gamma,
                        "best_epoch": epoch + 1,
                        "val_loss": avg_val_loss,
                        "val_pr_auc": val_pr_auc,
                    }, f, indent=2)
                print(f"    New best val_pr_auc {val_pr_auc:.4f} — saved checkpoint, scaler, and metadata.")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"No val_pr_auc improvement for {patience} epochs — stopping early at epoch {epoch+1}.")
                    break

        mlflow.log_metric("best_val_pr_auc", best_val_pr_auc)
        mlflow.pytorch.log_model(model, "model")
        print(f"Best model saved to {checkpoint_path} (scaler: {scaler_path}, metadata: {metadata_path})")

    return {
        "checkpoint_path": checkpoint_path,
        "scaler_path": scaler_path,
        "metadata_path": metadata_path,
        "best_val_pr_auc": best_val_pr_auc,
    }

if __name__ == "__main__":
    train_model()