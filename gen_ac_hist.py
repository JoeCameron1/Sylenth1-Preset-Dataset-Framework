"""Histogram grid of the seven AudioCommons descriptors over all presets.

Writes AC_Hist.png (Figure 3 of the paper): no in-figure title (the paper
caption carries the description), large fonts for full-width inclusion.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "figure.dpi": 150,
})

# -----------------------------
# Config
# -----------------------------
JSON_PATH = Path("FINAL_timbral_dataset_audiocommons.json")  # Change to the relevant file
OUT_PATH = Path("AC_Hist.png")
TIMBRAL_KEYS = [
    "brightness",
    "warmth",
    "hardness",
    "roughness",
    "depth",
    "sharpness",
    "boominess",
]

# -----------------------------
# Load dataset
# -----------------------------
def load_dataset(path: Path):
    with path.open("r") as f:
        data = json.load(f)
    # your file is a list of entries
    assert isinstance(data, list)
    return data

entries = load_dataset(JSON_PATH)

# -----------------------------
# Extract AudioCommons values
# -----------------------------
values = {k: [] for k in TIMBRAL_KEYS}

for e in entries:
    models = e.get("models", {}) or {}
    for k in TIMBRAL_KEYS:
        v = models.get(k, None)
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            values[k].append(v)

# Convert to numpy arrays
for k in TIMBRAL_KEYS:
    values[k] = np.asarray(values[k], dtype=float)

# -----------------------------
# Plot composite histogram grid
# -----------------------------
n_rows, n_cols = 2, 4   # 7 plots + 1 empty
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 9))
axes = axes.flatten()

for i, key in enumerate(TIMBRAL_KEYS):
    ax = axes[i]
    vals = values[key]
    ax.hist(vals, bins=30)
    ax.set_title(key.capitalize())
    ax.set_xlim(0, 100)
    ax.set_xlabel("Value (0–100)")
    ax.set_ylabel("Count")

# Hide the unused last axis
axes[-1].axis("off")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Wrote {OUT_PATH}")
