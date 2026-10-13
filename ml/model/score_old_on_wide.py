"""
Scores the SHIPPED (old) model on the wide dataset's test windows.

The shipped model was trained on Jan-Apr 2026 data from five instance types in one region.
The wide test slice is the last ~9 days of the Jul-Oct 2026 history across 768 series, so this
is a fully out-of-time (and, for most series, out-of-distribution) evaluation. It is the fair
reference for the retrained models: same windows, same labels, same metrics.

    python ml/model/score_old_on_wide.py
Output: ml/model/wide_results/old_model_on_wide.json
"""
import json
import os

import joblib
import numpy as np
import torch

import wide_experiments as we
from transformer import SpotInterruptionPredictor

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = we.WideData(dev)
    with open(os.path.join(HERE, "model_metadata.json")) as f:
        meta = json.load(f)
    assert meta["feature_columns"] == we.BASE, "shipped model expects the 13 base features"
    scaler = joblib.load(os.path.join(HERE, "spot_scaler.joblib"))
    model = SpotInterruptionPredictor(meta["num_features"], meta["d_model"], meta["nhead"], meta["num_layers"])
    model.load_state_dict(torch.load(os.path.join(HERE, "spot_transformer.pt"), map_location="cpu"))
    model.to(dev)

    Xg = data.to_device(we.BASE, scaler.mean_, scaler.scale_)
    lt = we.predict_logits(model, data, Xg, data.starts["test"])
    yt = data.labels["test"]
    out = {"all": we.metrics(yt, lt)}
    for name, m in (("eu-north-1", data.region_mask("test", "eu-north-1")),
                    ("eu-north-1_old5", data.region_mask("test", "eu-north-1", we.OLD_TYPES))):
        if yt[m].sum() > 0:
            out[name] = we.metrics(yt[m], lt[m])
    os.makedirs(we.RESULTS, exist_ok=True)
    with open(os.path.join(we.RESULTS, "old_model_on_wide.json"), "w") as f:
        json.dump(out, f, indent=2)
    for k, v in out.items():
        print(f"{k:16s} positives {v['positives']:>5}  AUPRC {v['auprc']:.4f}  lift {v['lift']:.2f}x  AUROC {v['auroc']:.3f}")


if __name__ == "__main__":
    main()
