import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
TIMBRAL_JSON = Path("FINAL_timbral_dataset_audiocommons.json")  # Change to the relevant file
N_COMPONENTS = 20
OUT_SCORES_CSV = Path("pc_scores_factory_random.csv")

# -------------------------------------------------------
# 1. Load dataset + build DataFrame of parameters
# -------------------------------------------------------
print(f"Loading dataset from {TIMBRAL_JSON}...")
with TIMBRAL_JSON.open("r") as f:
    data = json.load(f)

rows = []
for item in data:
    params = item.get("params", {}) or {}
    numeric_params = {k: v for k, v in params.items()
                      if isinstance(v, (int, float, np.number))}

    # NEW RULE:
    # - If item["name"] == "Preset" → random
    # - Else → factory
    preset_name = (item.get("name") or "").strip()
    if preset_name == "Preset":
        src_label = "random"
    else:
        src_label = "factory"

    row = {
        "id": item.get("id"),
        "name": preset_name,
        "source_norm": src_label
    }
    row.update(numeric_params)
    rows.append(row)

df = pd.DataFrame(rows)
print(f"Loaded {len(df)} presets.")

# Extract numeric feature matrix
meta_cols = ["id", "name", "source_norm"]
feature_cols = [c for c in df.columns if c not in meta_cols]
X = df[feature_cols].values.astype(float)

# Handle NaNs
nan_mask = ~np.isfinite(X)
if nan_mask.any():
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])
    print("Warning: NaNs found; filled with column means.")

# -------------------------------------------------------
# 2. Standardize + PCA
# -------------------------------------------------------
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

pca = PCA(n_components=N_COMPONENTS, random_state=0)
Z = pca.fit_transform(X_std)

# Save scores
scores_df = df[["id", "name", "source_norm"]].copy()
for i in range(N_COMPONENTS):
    scores_df[f"PC{i+1}"] = Z[:, i]

scores_df.to_csv(OUT_SCORES_CSV, index=False)
print(f"Saved PCA scores → {OUT_SCORES_CSV.resolve()}")

# -------------------------------------------------------
# 3. Plotting (factory vs random)
# -------------------------------------------------------
pc1 = scores_df["PC1"].values
pc2 = scores_df["PC2"].values
src = scores_df["source_norm"].values

factory_mask = (src == "factory")
random_mask  = (src == "random")

pc1_factory = pc1[factory_mask]
pc2_factory = pc2[factory_mask]
pc1_random  = pc1[random_mask]
pc2_random  = pc2[random_mask]

print(f"Factory count: {factory_mask.sum()}")
print(f"Random count:  {random_mask.sum()}")

# Shared axis limits
all_pc1 = np.concatenate([pc1_factory, pc1_random])
all_pc2 = np.concatenate([pc2_factory, pc2_random])
pad_x = 0.05 * (all_pc1.max() - all_pc1.min())
pad_y = 0.05 * (all_pc2.max() - all_pc2.min())
xlim = (all_pc1.min() - pad_x, all_pc1.max() + pad_x)
ylim = (all_pc2.min() - pad_y, all_pc2.max() + pad_y)

# -------------------------------------------------------
# 4A. Overlay Scatter Plot
# -------------------------------------------------------
plt.figure(figsize=(6, 5))
plt.scatter(pc1_random, pc2_random, s=5, alpha=0.25,
            label="Random presets", color="tab:blue")
plt.scatter(pc1_factory, pc2_factory, s=18, alpha=0.9,
            label="Factory presets", color="black")

plt.title("Factory vs Random Presets in PCA Space (PC1–PC2)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.xlim(xlim)
plt.ylim(ylim)
plt.legend()
plt.tight_layout()
plt.savefig("fig_pca_factory_random_overlay.png", dpi=300)

# -------------------------------------------------------
# 4B. Density Triptych
# -------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)

# Factory density
ax = axes[0]
ax.hexbin(pc1_factory, pc2_factory, gridsize=40, cmap="Greys", mincnt=1)
ax.set_title("Factory presets")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_xlim(xlim)
ax.set_ylim(ylim)

# Random density
ax = axes[1]
ax.hexbin(pc1_random, pc2_random, gridsize=40, cmap="Blues", mincnt=1)
ax.set_title("Random presets")
ax.set_xlabel("PC1")

# Combined
ax = axes[2]
ax.scatter(pc1_random, pc2_random, s=4, alpha=0.15, color="tab:blue")
ax.scatter(pc1_factory, pc2_factory, s=16, alpha=0.9, color="black")
ax.set_title("Combined")
ax.set_xlabel("PC1")

plt.tight_layout()
plt.savefig("fig_pca_factory_random_density_triptych.png", dpi=300)

plt.show()
print("Figures saved.")
