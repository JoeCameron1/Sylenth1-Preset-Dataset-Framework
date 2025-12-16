import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
JSON_PATH = Path("FINAL_timbral_dataset_audiocommons.json")  # Change to the relevant file
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
n_feats = len(TIMBRAL_KEYS)
n_rows, n_cols = 2, 4   # 7 plots + 1 empty
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6))
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

fig.suptitle("Distributions of AudioCommons Timbral Models", y=0.99)
plt.tight_layout()
plt.show()
