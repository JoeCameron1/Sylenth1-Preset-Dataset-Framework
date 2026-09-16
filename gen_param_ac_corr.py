"""Pearson correlation heatmap: Sylenth1 numeric parameters vs AudioCommons
descriptors (`pearson_correlations.png`, Figure 5 of the paper).

Columns are indexed numerically; the full index -> parameter-name key is
written to `pearson_correlations_column_key.csv`, and the strongest
correlates are named in the paper text. The colour scale is normalized to
the observed min/max of the matrix (deep red = observed maximum, deep blue =
observed minimum) rather than the theoretical [-1, +1]. No in-figure title,
large fonts.

Usage:
    python gen_param_ac_corr.py
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 13,
    "figure.dpi": 150,
})

JSON_PATH = Path("FINAL_timbral_dataset_audiocommons.json")
OUT_PNG = Path("pearson_correlations.png")
OUT_CSV = Path("pearson_correlations_column_key.csv")

TIMBRAL_KEYS = [
    "brightness", "warmth", "hardness", "roughness",
    "depth", "sharpness", "boominess",
]


def main() -> int:
    with JSON_PATH.open() as fh:
        data = json.load(fh)

    # Numeric-parseable parameters under the same selection rule as the PCA
    # analysis: keep a parameter if every present value parses as a finite
    # float. Factory-only keys are excluded by requiring presence in a random
    # entry's schema (random entries carry exactly the canonical 179 keys).
    random_entry = next(e for e in data
                        if (e.get("name") or "").strip() == "Preset")
    candidate_keys = sorted(random_entry["params"].keys())

    def parse(v):
        try:
            f = float(v)
            return f if np.isfinite(f) else None
        except (TypeError, ValueError):
            return None

    numeric_keys = []
    for k in candidate_keys:
        ok = True
        for e in data:
            v = e.get("params", {}).get(k)
            if v is None:
                continue
            if parse(v) is None:
                ok = False
                break
        if ok:
            numeric_keys.append(k)

    # Drop constant parameters (zero variance -> undefined correlation).
    cols = {}
    for k in numeric_keys:
        vals = np.array([parse(e["params"].get(k)) for e in data], dtype=float)
        if np.nanstd(vals) > 0:
            cols[k] = vals
    params = sorted(cols.keys())

    desc = {k: np.array([float(e["models"][k]) for e in data]) for k in TIMBRAL_KEYS}

    corr = np.zeros((len(TIMBRAL_KEYS), len(params)))
    for i, dk in enumerate(TIMBRAL_KEYS):
        for j, pk in enumerate(params):
            x, y = cols[pk], desc[dk]
            mask = np.isfinite(x) & np.isfinite(y)
            corr[i, j] = np.corrcoef(x[mask], y[mask])[0, 1]

    # Column key CSV (index -> parameter name), released in the repository.
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["column_index", "parameter"])
        for j, pk in enumerate(params, start=1):
            w.writerow([j, pk])

    # Report the strongest correlations for the paper text.
    flat = [(corr[i, j], TIMBRAL_KEYS[i], params[j])
            for i in range(len(TIMBRAL_KEYS)) for j in range(len(params))]
    flat.sort(key=lambda t: t[0])
    print(f"{len(params)} numeric parameter columns")
    print("Most negative correlations:")
    for r, d, p in flat[:6]:
        print(f"  {d:>10s} vs {p:<28s} r={r:+.3f}")
    print("Most positive correlations:")
    for r, d, p in flat[-6:][::-1]:
        print(f"  {d:>10s} vs {p:<28s} r={r:+.3f}")

    # Heatmap: colour scale symmetric about zero at the OBSERVED max |r|
    # (deep red = observed max, deep blue = observed min), not [-1, +1].
    vmax = float(np.nanmax(np.abs(corr)))
    fig, ax = plt.subplots(figsize=(20, 4.4))
    im = ax.imshow(corr, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   interpolation="nearest")
    ax.set_yticks(range(len(TIMBRAL_KEYS)))
    ax.set_yticklabels(TIMBRAL_KEYS)
    tick_step = 10
    ticks = np.arange(tick_step - 1, len(params), tick_step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t + 1) for t in ticks])
    ax.set_xlabel("Parameter index (see released column key)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Pearson's $r$")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PNG} and {OUT_CSV} (vmax={vmax:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
