#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PCA ↔ AudioCommons analysis for Sylenth1 preset datasets.

- Loads a JSON list of presets with:
    { "models": {...}, "params": {...}, ... }
- Extracts numeric Sylenth1 parameters and AudioCommons descriptors.
- Standardizes numeric params and runs PCA.
- Correlates PCs with each AudioCommons model (Pearson r).
- Saves:
    - CSVs: PC scores, loadings, explained variance ratio, PC↔model correlations, best PC per model
    - Figures (matplotlib): Scree plot, correlation heatmap, per-model PC-corr bars, and top loadings bars.

Usage:
    python pca_audiocommons_analysis.py --in data.json --outdir pca_audiocommons_analysis

Dependencies:
    pip install pandas numpy scikit-learn matplotlib
"""

import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def load_and_flatten(json_path: Path) -> pd.DataFrame:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    rows = []
    for item in data:
        base = {
            "id": item.get("id"),
            "name": item.get("name"),
            "sound_type": item.get("sound_type"),
            "note_name": item.get("note_name"),
            "note_midi": item.get("note_midi"),
        }
        models = item.get("models", {}) or {}
        params = item.get("params", {}) or {}
        flat = {**base}
        flat.update({f"model_{k}": v for k, v in models.items()})
        flat.update({f"param_{k}": v for k, v in params.items()})
        rows.append(flat)
    df = pd.DataFrame(rows)

    # Coerce numeric params where possible
    for c in [c for c in df.columns if c.startswith("param_")]:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="ignore")
    return df


def split_columns(df: pd.DataFrame):
    model_cols = sorted([c for c in df.columns if c.startswith("model_")])
    param_cols_all = [c for c in df.columns if c.startswith("param_")]
    param_numeric_cols = sorted([c for c in param_cols_all if pd.api.types.is_numeric_dtype(df[c])])
    return model_cols, param_numeric_cols


def run_pca(X: pd.DataFrame, n_max: int = 20):
    X_filled = X.fillna(X.median(numeric_only=True))
    Xs = StandardScaler().fit_transform(X_filled)
    n_components = min(n_max, Xs.shape[0], Xs.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(Xs)
    pc_names = [f"PC{i+1}" for i in range(scores.shape[1])]
    scores_df = pd.DataFrame(scores, columns=pc_names, index=X.index)
    loadings_df = pd.DataFrame(pca.components_.T, index=X.columns, columns=pc_names)
    explained = pd.Series(pca.explained_variance_ratio_, index=pc_names)
    return scores_df, loadings_df, explained


def compute_pc_model_corr(scores_df: pd.DataFrame, models_df: pd.DataFrame) -> pd.DataFrame:
    corr = pd.DataFrame(index=scores_df.columns,
                        columns=[c.replace("model_", "") for c in models_df.columns],
                        dtype=float)
    for m in models_df.columns:
        y = models_df[m].astype(float)
        for pc in scores_df.columns:
            corr.loc[pc, m.replace("model_", "")] = scores_df[pc].corr(y)
    return corr


def plot_scree(explained: pd.Series, out: Path):
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(explained) + 1), explained.values, marker="o")
    plt.title("PCA Scree Plot (Explained Variance Ratio)")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_corr_heatmap(corr: pd.DataFrame, out: Path):
    plt.figure(figsize=(max(6, 0.6 * len(corr.columns)), max(5, 0.35 * len(corr.index))))
    im = plt.imshow(corr.values, aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(ticks=np.arange(len(corr.columns)), labels=corr.columns, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(len(corr.index)), labels=corr.index)
    plt.title("Correlation: PCs vs AudioCommons Models")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_pc_corr_bars_for_model(corr: pd.DataFrame, model_short: str, out_dir: Path) -> str:
    vals = corr[model_short].astype(float)
    order = vals.abs().sort_values(ascending=False).index.tolist()
    plt.figure(figsize=(8, 4.5))
    plt.bar(range(len(vals)), vals[order].values)
    plt.xticks(range(len(vals)), order, rotation=45, ha="right")
    plt.ylabel("Pearson r")
    plt.title(f"PC Correlations with {model_short}")
    out = out_dir / f"fig_pc_corr_{model_short}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return str(out), order[0]  # best PC name


def plot_top_loadings(loadings_df: pd.DataFrame, pc_name: str, label: str, out_dir: Path, top_n: int = 20) -> str:
    pc_loads = loadings_df[pc_name]
    topN = min(top_n, len(pc_loads))
    top_params = pc_loads.reindex(pc_loads.abs().sort_values(ascending=False).index[:topN])
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(top_params)), top_params.values)
    plt.yticks(range(len(top_params)), [p.replace("param_", "") for p in top_params.index])
    plt.gca().invert_yaxis()
    plt.xlabel("Loading")
    plt.title(f"Top {topN} Parameter Loadings — {pc_name}{' ('+label+')' if label else ''}")
    out = out_dir / f"fig_top_loadings_{label}_{pc_name}.png" if label else out_dir / f"fig_top_loadings_{pc_name}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input_json", required=True, help="Path to timbral dataset JSON")
    ap.add_argument("--outdir", dest="out_dir", required=True, help="Output directory for figures + CSVs")
    args = ap.parse_args()

    in_path = Path(args.input_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_flatten(in_path)
    model_cols, param_numeric_cols = split_columns(df)
    if not model_cols or not param_numeric_cols:
        raise ValueError("No model columns or numeric parameter columns found.")

    X = df[param_numeric_cols]
    Y = df[model_cols]
    scores_df, loadings_df, explained = run_pca(X, n_max=20)

    # Save CSVs
    scores_df.assign(name=df["name"].values, sound_type=df["sound_type"].values).to_csv(out_dir / "pca_scores.csv", index=False)
    loadings_df.to_csv(out_dir / "pca_loadings.csv")
    explained.to_csv(out_dir / "pca_explained_variance_ratio.csv", header=["explained_variance_ratio"])

    # Correlations PCs ↔ models
    corr = compute_pc_model_corr(scores_df, Y)
    corr.to_csv(out_dir / "pc_vs_audiocommons_corr.csv")

    # Plots
    plot_scree(explained, out_dir / "fig_scree_plot.png")
    plot_corr_heatmap(corr, out_dir / "fig_pc_vs_models_corr_heatmap.png")

    # Per-model PC correlation bars + top loadings of best PC
    best_map = {}
    for model_short in corr.columns:
        bar_path, best_pc = plot_pc_corr_bars_for_model(corr, model_short, out_dir)
        best_map[model_short] = best_pc
        _ = plot_top_loadings(loadings_df, best_pc, f"{model_short}_bestPC", out_dir, top_n=20)

    # Also provide top loadings for PC1 and PC2 (often the most interpretable)
    for pc in scores_df.columns[:2]:
        _ = plot_top_loadings(loadings_df, pc, "", out_dir, top_n=25)

    # Summary CSV: best PC per model
    pd.DataFrame.from_dict(best_map, orient="index", columns=["best_PC_for_model"])\
      .reset_index().rename(columns={"index":"model"})\
      .to_csv(out_dir / "best_pc_per_model.csv", index=False)

    print(f"Done. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()