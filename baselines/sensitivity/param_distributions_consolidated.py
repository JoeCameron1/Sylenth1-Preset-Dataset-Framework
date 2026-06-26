"""Consolidated multi-panel parameter-distribution figure.

Replaces the seven near-identical param-distribution plots Reviewer 3 flagged
(Figs 4-10 in the original draft) with ONE figure. Each panel overlays the
factory and random distributions for one parameter on shared axes (R3's
specific ask for "comparative overlay plots ... that would make these
patterns visually accessible" missing from the original figures).

Defaults match the six numeric parameters used in the original Figs 4-9 plus
the categorical waveform from Fig 10 (rendered as side-by-side proportion
bars). Pass ``--numeric``/``--categorical`` to override.

Usage:
    python -m baselines.sensitivity.param_distributions_consolidated \\
        --in FINAL_timbral_dataset_audiocommons.json \\
        --out baselines/artifacts/figures/param_distributions_consolidated.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
})

# The six numeric parameters used in the original Figs 4-9, and the
# categorical waveform from Fig 10. Reviewer 3 asked for "a few
# representative params"; we keep the exact set the reviewers were looking
# at, just consolidated into one multi-panel figure with overlays.
DEFAULT_NUMERIC = [
    "filter_a_cutoff",
    "filterctl_cutoff",
    "ampenv_a_decay",
    "ampenv_a_attack",
    "osc_a1_detune",
    "reverb_drywet",
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

    total = len(args.numeric) + len(args.categorical)
    n_cols = 3
    n_rows = (total + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.0 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    panel = 0
    # Numeric panels: overlayed density histograms (factory in orange, random in blue),
    # sharing a common x-axis range per panel so the two distributions are directly comparable.
    for p in args.numeric:
        ax = axes[panel]; panel += 1
        f = np.asarray(numeric[p]["factory"], dtype=float)
        r = np.asarray(numeric[p]["random"], dtype=float)
        if f.size == 0 and r.size == 0:
            ax.set_title(f"{p}\n(no numeric values)"); ax.axis("off"); continue
        x_min = float(np.nanmin(np.concatenate([f, r]))) if f.size + r.size else 0.0
        x_max = float(np.nanmax(np.concatenate([f, r]))) if f.size + r.size else 1.0
        bins = np.linspace(x_min, x_max, args.bins + 1)
        if r.size:
            ax.hist(r, bins=bins, density=True, alpha=0.55, color=RANDOM_COLOR, label="random")
        if f.size:
            ax.hist(f, bins=bins, density=True, alpha=0.65, color=FACTORY_COLOR, label="factory")
        ax.set_title(p)
        ax.set_xlabel("value"); ax.set_ylabel("density")
        if panel == 1:
            ax.legend(loc="best", fontsize=9)

    # Categorical panels: side-by-side proportion bars per category.
    for p in args.categorical:
        ax = axes[panel]; panel += 1
        f_counts = {c: categorical[p]["factory"].count(c)
                    for c in set(categorical[p]["factory"]) | set(categorical[p]["random"])}
        r_counts = {c: categorical[p]["random"].count(c)
                    for c in set(categorical[p]["factory"]) | set(categorical[p]["random"])}
        cats = sorted(f_counts, key=lambda c: -(r_counts.get(c, 0) + f_counts.get(c, 0)))
        f_total = max(1, sum(f_counts.values()))
        r_total = max(1, sum(r_counts.values()))
        f_prop = [f_counts.get(c, 0) / f_total for c in cats]
        r_prop = [r_counts.get(c, 0) / r_total for c in cats]
        x = np.arange(len(cats))
        w = 0.4
        ax.bar(x - w / 2, r_prop, w, color=RANDOM_COLOR, alpha=0.85, label="random")
        ax.bar(x + w / 2, f_prop, w, color=FACTORY_COLOR, alpha=0.85, label="factory")
        ax.set_xticks(x); ax.set_xticklabels(cats, rotation=35, ha="right", fontsize=9)
        ax.set_title(p); ax.set_ylabel("proportion")
        if not args.numeric:
            ax.legend(loc="best", fontsize=9)

    for j in range(panel, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Factory vs random distributions for representative parameters",
                 y=1.0, fontsize=15)
    fig.tight_layout()
    fig.savefig(args.output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
