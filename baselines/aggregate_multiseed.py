"""Aggregate multi-seed baseline metrics into mean +/- std tables.

Inputs:
  regression_metrics_seed{0..3}.csv          (mean/ridge deterministic; MLP varies)
  inversion_metrics_full.csv                 (seed-0 run: nn + cvae_mean + cvae_sample)
  inversion_metrics_full_seed{1..3}.csv      (cvae_mean + cvae_sample only)

Outputs:
  regression_metrics_multiseed.csv, inversion_metrics_multiseed.csv
  and LaTeX-ready table rows printed to stdout.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent / "artifacts" / "results"
SEEDS = [0, 1, 2, 3]
DESCRIPTORS = ["brightness", "depth", "hardness", "roughness",
               "warmth", "sharpness", "boominess", "macro"]


def load(path: Path) -> dict:
    """{(method, descriptor): row-dict}"""
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out[(r["method"], r["descriptor"])] = r
    return out


def collect(files: dict[int, Path], methods: list[str]) -> dict:
    """{method: {descriptor: [mae per seed]}} plus rates if present."""
    per = {m: {d: [] for d in DESCRIPTORS} for m in methods}
    rates = {m: {"audibility_rate": [], "validity_rate": []} for m in methods}
    for seed, path in sorted(files.items()):
        rows = load(path)
        for m in methods:
            for d in DESCRIPTORS:
                if (m, d) in rows:
                    per[m][d].append(float(rows[(m, d)]["mae"]))
            key = (m, "macro")
            if key in rows and rows[key].get("audibility_rate"):
                rates[m]["audibility_rate"].append(float(rows[key]["audibility_rate"]))
                rates[m]["validity_rate"].append(float(rows[key]["validity_rate"]))
    return per, rates


def fmt(vals: list[float], det: bool) -> str:
    """LaTeX cell: 'm' for deterministic methods, 'm ± s' otherwise."""
    a = np.asarray(vals, dtype=float)
    if det or len(a) <= 1:
        return f"{a.mean():.2f}"
    return f"{a.mean():.2f} $\\pm$ {a.std(ddof=1):.2f}"


def write_csv(path: Path, per: dict, det_methods: set):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "descriptor", "n_seeds", "mae_mean", "mae_std"])
        for m, dd in per.items():
            for d, vals in dd.items():
                if not vals:
                    continue
                a = np.asarray(vals, dtype=float)
                std = 0.0 if (m in det_methods or len(a) <= 1) else float(a.std(ddof=1))
                w.writerow([m, d, len(a), f"{a.mean():.4f}", f"{std:.4f}"])
    print(f"Wrote {path}")


def main() -> int:
    # --- Regression ---
    reg_files = {s: ROOT / f"regression_metrics_seed{s}.csv" for s in SEEDS}
    missing = [p for p in reg_files.values() if not p.exists()]
    if missing:
        print(f"[regression] missing: {[str(m) for m in missing]}")
    else:
        per, _ = collect(reg_files, ["mean", "ridge", "mlp"])
        write_csv(ROOT / "regression_metrics_multiseed.csv", per, {"mean", "ridge"})
        print("\n--- Table 4 rows (regression, test MAE) ---")
        for m, label in (("mean", "Mean predictor"), ("ridge", "Ridge"), ("mlp", "MLP")):
            det = m in {"mean", "ridge"}
            cells = [fmt(per[m][d], det) for d in DESCRIPTORS]
            print(f"{label} & " + " & ".join(cells[:-1]) + f" & {cells[-1]} \\\\")

    # --- Inversion ---
    inv_files = {0: ROOT / "inversion_metrics_full.csv"}
    for s in [1, 2, 3]:
        inv_files[s] = ROOT / f"inversion_metrics_full_seed{s}.csv"
    missing = [p for p in inv_files.values() if not p.exists()]
    if missing:
        print(f"\n[inversion] missing: {[str(m) for m in missing]}")
        return 0
    per, rates = collect(inv_files, ["nn", "cvae_mean", "cvae_sample"])
    write_csv(ROOT / "inversion_metrics_multiseed.csv", per, {"nn"})
    print("\n--- Table 5 rows (inversion, round-trip MAE) ---")
    for m, label in (("nn", "NN retrieval"),
                     ("cvae_sample", "CVAE ($z\\!\\sim\\!\\mathcal{N}(0,I)$)"),
                     ("cvae_mean", "CVAE ($z\\!=\\!0$)")):
        det = m == "nn"
        cells = [fmt(per[m][d], det) for d in DESCRIPTORS]
        aud = np.mean(rates[m]["audibility_rate"]) * 100
        val = np.mean(rates[m]["validity_rate"]) * 100
        print(f"{label} & " + " & ".join(cells[:-1]) +
              f" & {cells[-1]} & {aud:.1f}\\% & {val:.0f}\\% \\\\")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
