"""Baseline 1 — parameter -> descriptor regression (Task 4, spec §5).

Sanity-check baseline only — descriptors are a near-deterministic function of
params, so high R^2 here is partly re-learning the AudioCommons model. Useful
to validate the encoding + splits end-to-end and to show which descriptors
carry the most learnable signal.

Model ladder:
  1. mean predictor — train-mean descriptor vector (floor),
  2. ridge regression — linear-recoverable signal,
  3. MLP — 256 -> 256 -> 128 ReLU + dropout + BN, Adam, early-stop on val MAE.

Outputs:
  baselines/artifacts/results/regression_metrics.csv
  baselines/artifacts/figures/regression_mae_per_descriptor.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge

from baselines.common.data import build_split_matrices, per_descriptor_metrics
from baselines.common.io import TIMBRAL_KEYS


_BASELINES_ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = _BASELINES_ROOT / "artifacts" / "results" / "regression_metrics.csv"
FIG_PNG = _BASELINES_ROOT / "artifacts" / "figures" / "regression_mae_per_descriptor.png"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cpu",
                    help="'cpu' (default) | 'cuda' | 'mps' if available")
    ap.add_argument("--drop-ambiguous", action="store_true",
                    help="exclude flagged-ambiguous random presets")
    return ap.parse_args()


class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden=(256, 256, 128), dropout: float = 0.15):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _impute_train_mean(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replace NaN per-column with the column mean (excluding NaNs). Return (Y_filled, train_means)."""
    means = np.nanmean(Y, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    out = Y.copy()
    for j in range(Y.shape[1]):
        col = out[:, j]
        m = ~np.isfinite(col)
        col[m] = means[j]
    return out, means


def train_mlp(X_tr, Y_tr_std, X_va, Y_va_std, mean_std_y, args) -> tuple[nn.Module, list[dict]]:
    """Standard MLP training with val-MAE early stopping. Y_*_std is z-scored."""
    device = torch.device(args.device)
    in_dim = X_tr.shape[1]
    out_dim = Y_tr_std.shape[1]
    model = MLPRegressor(in_dim, out_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.SmoothL1Loss()

    Xtr_t = torch.from_numpy(X_tr).float()
    Ytr_t = torch.from_numpy(Y_tr_std).float()
    Xva_t = torch.from_numpy(X_va).float().to(device)
    Yva_t = torch.from_numpy(Y_va_std).float().to(device)
    n_tr = Xtr_t.shape[0]
    best_val = float("inf")
    best_state = None
    stale = 0
    history = []
    bsz = args.batch_size

    for epoch in range(1, args.epochs + 1):
        model.train()
        idx = torch.randperm(n_tr)
        total = 0.0
        for s in range(0, n_tr, bsz):
            j = idx[s:s + bsz]
            xb = Xtr_t[j].to(device)
            yb = Ytr_t[j].to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.size(0)
        tr_loss = total / n_tr
        model.eval()
        with torch.no_grad():
            preds_va = model(Xva_t).cpu().numpy()
        # MAE on the descriptor scale (de-standardise)
        mean_y, std_y = mean_std_y
        preds_va_native = preds_va * std_y + mean_y
        Yva_native = Y_va_std * std_y + mean_y
        val_mae = float(np.mean(np.abs(preds_va_native - Yva_native)))
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_mae": val_mae})

        if val_mae < best_val - 1e-4:
            best_val = val_mae
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  MLP best val MAE = {best_val:.3f} (epoch {history[-1-stale]['epoch']}/{epoch}, stopped after {stale} stale)")
    return model, history


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    FIG_PNG.parent.mkdir(parents=True, exist_ok=True)

    print("Loading split matrices...")
    splits, _codec = build_split_matrices(drop_ambiguous=args.drop_ambiguous)
    X_tr, Y_tr, _ = splits["train"]
    X_va, Y_va, _ = splits["val"]
    X_te, Y_te, _ = splits["test"]
    print(f"  train={X_tr.shape}, val={X_va.shape}, test={X_te.shape}")
    print(f"  encoded_dim={X_tr.shape[1]}, descriptors={Y_tr.shape[1]}")

    # Impute NaN descriptors with train mean (per spec, they should all be present,
    # but be defensive).
    Y_tr_imp, train_means = _impute_train_mean(Y_tr)
    Y_va_imp, _ = _impute_train_mean(Y_va)
    Y_te_imp, _ = _impute_train_mean(Y_te)

    # Mean-predictor baseline.
    pred_mean_te = np.tile(train_means, (Y_te_imp.shape[0], 1))
    mean_metrics = per_descriptor_metrics(Y_te_imp, pred_mean_te)
    print(f"\nMean predictor (test): macro MAE={mean_metrics['macro']['mae']:.3f}")

    # Ridge baseline (per-descriptor).
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr, Y_tr_imp)
    pred_ridge_te = ridge.predict(X_te)
    ridge_metrics = per_descriptor_metrics(Y_te_imp, pred_ridge_te)
    print(f"Ridge (test):          macro MAE={ridge_metrics['macro']['mae']:.3f}")

    # MLP — standardise targets on TRAIN only.
    mean_y = Y_tr_imp.mean(axis=0)
    std_y = Y_tr_imp.std(axis=0)
    std_y = np.where(std_y > 1e-6, std_y, 1.0)
    Y_tr_std = (Y_tr_imp - mean_y) / std_y
    Y_va_std = (Y_va_imp - mean_y) / std_y
    model, hist = train_mlp(X_tr.astype(np.float32), Y_tr_std.astype(np.float32),
                            X_va.astype(np.float32), Y_va_std.astype(np.float32),
                            (mean_y, std_y), args)
    model.eval()
    with torch.no_grad():
        pred_mlp_te_std = model(torch.from_numpy(X_te.astype(np.float32))).cpu().numpy()
    pred_mlp_te = pred_mlp_te_std * std_y + mean_y
    mlp_metrics = per_descriptor_metrics(Y_te_imp, pred_mlp_te)
    print(f"MLP (test):            macro MAE={mlp_metrics['macro']['mae']:.3f}")

    # Save CSV.
    rows = []
    for method, M in (("mean", mean_metrics), ("ridge", ridge_metrics), ("mlp", mlp_metrics)):
        for k in TIMBRAL_KEYS + ("macro",):
            r = {"method": method, "descriptor": k, **M[k]}
            rows.append(r)
    with open(RESULTS_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "descriptor", "mae", "rmse", "r2", "n"])
        w.writeheader()
        for r in rows:
            r.setdefault("n", "")
            w.writerow(r)
    print(f"\nWrote {RESULTS_CSV}")

    # Grouped bar chart: MAE per descriptor per method.
    width = 0.27
    x = np.arange(len(TIMBRAL_KEYS))
    plt.figure(figsize=(10, 5))
    plt.bar(x - width, [mean_metrics[k]["mae"] for k in TIMBRAL_KEYS], width, label="mean")
    plt.bar(x,         [ridge_metrics[k]["mae"] for k in TIMBRAL_KEYS], width, label="ridge")
    plt.bar(x + width, [mlp_metrics[k]["mae"] for k in TIMBRAL_KEYS], width, label="MLP")
    plt.xticks(x, TIMBRAL_KEYS, rotation=20)
    plt.ylabel("Test MAE (0-100 units)")
    plt.title("Parameter -> descriptor regression: per-descriptor test MAE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_PNG, dpi=150)
    plt.close()
    print(f"Wrote {FIG_PNG}")

    # Save training history alongside results for reproducibility.
    history_path = RESULTS_CSV.parent / "regression_mlp_train_history.json"
    with open(history_path, "w") as fh:
        json.dump({"seed": args.seed, "history": hist}, fh, indent=2)
    print(f"Wrote {history_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
