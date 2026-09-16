"""Consolidated multi-panel parameter-distribution figure (Figure 4 of the paper).

One figure, seven panels: six numeric parameters as factory-vs-random overlay
histograms plus the categorical oscillator waveform as side-by-side proportion
bars (spanning two grid slots so the 2x4 grid has no empty space).

Encoding:
- Random-preset histograms are drawn as blue-hatched translucent overlays on
  top of the solid factory bars, so bins where the random density is LOWER
  than the factory density remain visible.
- Panels grouped by function (filter | envelope || oscillator | effects |
  categorical), no suptitle (captions carry the description), larger fonts.

Usage:
    python -m baselines.sensitivity.param_distributions_consolidated \\
        --in FINAL_timbral_dataset_audiocommons.json \\
        --out baselines/artifacts/figures/param_distributions_consolidated.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 13,
    "figure.dpi": 150,
    "hatch.linewidth": 0.9,
})

# Grouped by function: filtering, then envelope, then oscillator/effects.
# The categorical waveform panel is drawn last, spanning two grid slots.
DEFAULT_NUMERIC = [
    "filter_a_cutoff",
    "filterctl_cutoff",
    "ampenv_a_attack",
    "ampenv_a_decay",
    "osc_a1_detune",
    "reverb_dry_wet",
]
DEFAULT_CATEGORICAL = [
    "osc_a1_waveform",
]

FACTORY_COLOR = "#ff7f0e"
RANDOM_COLOR = "#1f77b4"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="input_json", type=Path, required=True)
    ap.add_argument("--out", dest="output_png", type=Path, required=True)
    ap.add_argument("--numeric", nargs="+", default=DEFAULT_NUMERIC,
                    help="numeric parameter keys (overlay histograms)")
    ap.add_argument("--categorical", nargs="+", default=DEFAULT_CATEGORICAL,
                    help="categorical parameter keys (side-by-side proportion bars)")
    ap.add_argument("--bins", type=int, default=40)
    return ap.parse_args()


def _kind(entry: dict) -> str:
    return "random" if (entry.get("name") or "").strip() == "Preset" else "factory"


def main() -> int:
    args = parse_args()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    with open(args.input_json, "r") as fh:
        data = json.load(fh)

    numeric: dict[str, dict[str, list[float]]] = {
        p: {"factory": [], "random": []} for p in args.numeric
    }
    categorical: dict[str, dict[str, list[str]]] = {
        p: {"factory": [], "random": []} for p in args.categorical
    }
    for entry in data:
        kind = _kind(entry)
        params = entry.get("params") or {}
        for p in args.numeric:
            v = params.get(p)
            try:
                fv = float(v)
                if np.isfinite(fv):
                    numeric[p][kind].append(fv)
            except (TypeError, ValueError):
                pass
        for p in args.categorical:
            v = params.get(p)
            if v is None:
                continue
            categorical[p][kind].append(str(v))

    # Fail loudly on a missing/empty parameter rather than shipping a
    # blank panel.
    for p in args.numeric:
        if not (numeric[p]["factory"] or numeric[p]["random"]):
            raise SystemExit(f"No numeric values found for parameter {p!r} - "
                             f"check the key name against the dataset.")

    # 2x4 grid: six numeric panels, categorical panel spans the last two slots.
    n_cols = 4
    n_rows = 2
    fig = plt.figure(figsize=(5.0 * n_cols, 4.0 * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols)
    slots = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)]

    random_face = mcolors.to_rgba(RANDOM_COLOR, alpha=0.30)

    for (row, col), p in zip(slots, args.numeric):
        ax = fig.add_subplot(gs[row, col])
        f = np.asarray(numeric[p]["factory"], dtype=float)
        r = np.asarray(numeric[p]["random"], dtype=float)
        both = np.concatenate([f, r])
        bins = np.linspace(float(np.nanmin(both)), float(np.nanmax(both)),
                           args.bins + 1)
        # Factory: solid orange bars underneath.
        ax.hist(f, bins=bins, density=True, color=FACTORY_COLOR, alpha=0.85,
                zorder=1)
        # Random: translucent blue fill with blue diagonal hatching on top, so
        # bins where random < factory show as slashes over the orange bar.
        ax.hist(r, bins=bins, density=True, histtype="stepfilled",
                facecolor=random_face, edgecolor=RANDOM_COLOR, hatch="///",
                linewidth=1.0, zorder=2)
        ax.set_title(p)
        ax.set_xlabel("value")
        ax.set_ylabel("density")

    # Categorical panel: side-by-side proportion bars, spanning two slots.
    for p in args.categorical:
        ax = fig.add_subplot(gs[1, 2:4])
        all_cats = set(categorical[p]["factory"]) | set(categorical[p]["random"])
        f_counts = {c: categorical[p]["factory"].count(c) for c in all_cats}
        r_counts = {c: categorical[p]["random"].count(c) for c in all_cats}
        cats = sorted(all_cats,
                      key=lambda c: -(r_counts.get(c, 0) + f_counts.get(c, 0)))
        f_total = max(1, sum(f_counts.values()))
        r_total = max(1, sum(r_counts.values()))
        f_prop = [f_counts.get(c, 0) / f_total for c in cats]
        r_prop = [r_counts.get(c, 0) / r_total for c in cats]
        x = np.arange(len(cats))
        w = 0.4
        ax.bar(x - w / 2, f_prop, w, color=FACTORY_COLOR, alpha=0.85)
        ax.bar(x + w / 2, r_prop, w, facecolor=random_face,
               edgecolor=RANDOM_COLOR, hatch="///", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=30, ha="right")
        ax.set_title(p)
        ax.set_ylabel("proportion")

    legend_handles = [
        Patch(facecolor=FACTORY_COLOR, alpha=0.85, label="factory (n=292)"),
        Patch(facecolor=random_face, edgecolor=RANDOM_COLOR, hatch="///",
              label="random (n=10,000)"),
    ]
    fig.axes[0].legend(handles=legend_handles, loc="upper right")

    fig.tight_layout()
    fig.savefig(args.output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
