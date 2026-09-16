"""Analyse multi-pitch sensitivity.

Per-descriptor sensitivity = mean absolute deviation of the descriptor's
value at each pitch from its value at A4 (the dataset's reference pitch).
Also per-descriptor Pearson and Spearman correlation across pitch pairs to
quantify pitch invariance.

Also produces per-preset signed-deviation boxplots (a length-encoded
distribution view) and extended distributional statistics
(std, median absolute deviation, signed range) in the sensitivity CSV.

Output:
  baselines/artifacts/results/multipitch_sensitivity.csv
  baselines/artifacts/figures/multipitch_heatmap.png  (descriptor x pitch shift)
  baselines/artifacts/figures/multipitch_boxplots.png (per-preset signed deviations)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baselines.common.io import TIMBRAL_KEYS

_BASELINES_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = _BASELINES_ROOT / "artifacts" / "results" / "multipitch_renders.csv"
DEFAULT_CSV_OUT = _BASELINES_ROOT / "artifacts" / "results" / "multipitch_sensitivity.csv"
DEFAULT_FIG_OUT = _BASELINES_ROOT / "artifacts" / "figures" / "multipitch_heatmap.png"
DEFAULT_BOX_OUT = _BASELINES_ROOT / "artifacts" / "figures" / "multipitch_boxplots.png"


PITCH_LABELS = ["A2", "A3", "A4", "A5", "A6"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="input_csv", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=DEFAULT_CSV_OUT)
    ap.add_argument("--fig", type=Path, default=DEFAULT_FIG_OUT)
    ap.add_argument("--boxfig", type=Path, default=DEFAULT_BOX_OUT)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    if df.empty:
        print(f"No rows in {args.input_csv}")
        return 1

    # Pivot so columns are pitches per descriptor.
    out_rows = []
    fig_grid = np.zeros((len(TIMBRAL_KEYS), len(PITCH_LABELS)), dtype=float)
    fig_grid[:] = np.nan

    # signed per-preset deviations, keyed (descriptor, pitch) - for boxplots
    signed_devs: dict[tuple[str, str], list[float]] = {}

    for di, descr in enumerate(TIMBRAL_KEYS):
        col = f"d_{descr}"
        # Per-preset A4 reference
        a4_df = df[df["note_label"] == "A4"][["id", col]].rename(columns={col: "ref"})
        a4_map = dict(zip(a4_df["id"], a4_df["ref"]))
        for pi, pitch in enumerate(PITCH_LABELS):
            sub = df[df["note_label"] == pitch][["id", col]].dropna()
            diffs = []
            signed = []
            for _, r in sub.iterrows():
                ref = a4_map.get(r["id"])
                if ref is None or not np.isfinite(ref) or not np.isfinite(r[col]):
                    continue
                diffs.append(abs(r[col] - ref))
                signed.append(r[col] - ref)
            if not diffs:
                continue
            signed_devs[(descr, pitch)] = signed
            mae = float(np.mean(diffs))
            fig_grid[di, pi] = mae
            # Pearson r against A4 values across the same presets
            vals = []; refs = []
            for _, r in sub.iterrows():
                ref = a4_map.get(r["id"])
                if ref is not None and np.isfinite(ref) and np.isfinite(r[col]):
                    vals.append(r[col]); refs.append(ref)
            if len(vals) >= 5:
                pearson = float(np.corrcoef(vals, refs)[0, 1])
            else:
                pearson = float("nan")
            sarr = np.asarray(signed, dtype=float)
            out_rows.append({
                "descriptor": descr,
                "pitch": pitch,
                "n": len(diffs),
                "mean_abs_dev_from_a4": mae,
                "std_abs_dev": float(np.std(diffs)),
                "median_abs_dev": float(np.median(diffs)),
                "signed_dev_min": float(sarr.min()),
                "signed_dev_max": float(sarr.max()),
                "pearson_r_to_a4": pearson,
            })

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["descriptor", "pitch", "n",
                                            "mean_abs_dev_from_a4",
                                            "std_abs_dev", "median_abs_dev",
                                            "signed_dev_min", "signed_dev_max",
                                            "pearson_r_to_a4"])
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"Wrote {args.out}")

    # Heatmap descriptor x pitch (mean abs deviation from A4).
    # Colour scale spans the observed min/max (explicit vmin/vmax), large
    # cell-annotation fonts, no in-figure title.
    plt.rcParams.update({
        "font.size": 15,
        "axes.labelsize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "figure.dpi": 150,
    })
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    vmin = float(np.nanmin(fig_grid))
    vmax = float(np.nanmax(fig_grid))
    im = ax.imshow(fig_grid, aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(PITCH_LABELS))); ax.set_xticklabels(PITCH_LABELS)
    ax.set_yticks(range(len(TIMBRAL_KEYS))); ax.set_yticklabels(TIMBRAL_KEYS)
    ax.set_xlabel("Render pitch")
    ax.set_ylabel("AudioCommons descriptor")
    # Annotate cells with the numeric value
    for i in range(fig_grid.shape[0]):
        for j in range(fig_grid.shape[1]):
            v = fig_grid[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}",
                        ha="center", va="center",
                        color="white" if v < vmin + 0.55 * (vmax - vmin) else "black",
                        fontsize=14)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("MAE vs A4 (0-100 units)")
    plt.tight_layout()
    args.fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.fig, dpi=150)
    plt.close()
    print(f"Wrote {args.fig}")

    # Boxplots of per-preset SIGNED deviations from A4, one panel per
    # descriptor (length-encoded distribution view with range, spread, and
    # median visible; boxes show median/IQR, whiskers 5th-95th percentile).
    box_pitches = [p for p in PITCH_LABELS if p != "A4"]
    fig, axes = plt.subplots(1, len(TIMBRAL_KEYS), figsize=(3.1 * len(TIMBRAL_KEYS), 4.6),
                             sharey=True)
    for ax, descr in zip(np.atleast_1d(axes), TIMBRAL_KEYS):
        data = [signed_devs.get((descr, p), []) for p in box_pitches]
        ax.boxplot(data, tick_labels=box_pitches, whis=(5, 95), showfliers=False,
                   medianprops={"color": "#d62728", "linewidth": 1.6})
        ax.axhline(0.0, color="k", lw=0.7, ls="--")
        ax.set_title(descr)
        ax.set_xlabel("Render pitch")
    np.atleast_1d(axes)[0].set_ylabel("Deviation from A4 value (0-100 units)")
    fig.tight_layout()
    args.boxfig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.boxfig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.boxfig}")

    # Console summary
    print("\nDescriptor sensitivity to pitch (mean |dev from A4|, 0-100 units):")
    for descr in TIMBRAL_KEYS:
        per_pitch = {r["pitch"]: r["mean_abs_dev_from_a4"]
                     for r in out_rows if r["descriptor"] == descr}
        seq = "  ".join(f"{p}={per_pitch.get(p,0):5.2f}" for p in PITCH_LABELS)
        print(f"  {descr:>10s}: {seq}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
