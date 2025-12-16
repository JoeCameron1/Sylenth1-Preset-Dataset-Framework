#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a Sylenth1 × AudioCommons interactive dashboard from a JSON dataset.

Usage:
    python build_sylenth1_timbre_dashboard.py --in timbral_dataset.json --out dashboard.html

Inputs:
    - JSON file with items shaped like:
        {
          "id": ...,
          "name": ...,
          "sound_type": ...,
          "note_name": ...,
          "note_midi": ...,
          "models": { "brightness": float, "warmth": float, ... },
          "params": { "filter_a_cutoff": float, "osc_a1_waveform": "Saw", ... }
        }
Outputs:
    - A single self-contained HTML file with 5 interactive Plotly views:
        1) Numeric parameter distributions (dropdown)
        1b) Categorical parameter counts (dropdown)
        2) Correlation heatmap (numeric params × timbral models)
        3) PCA of parameter space (color by chosen timbral model)
        4) Scatter Explorer (pick any numeric param vs any model, labels update)

Dependencies:
    pip install pandas numpy plotly scikit-learn
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def load_and_flatten(json_path: Path) -> pd.DataFrame:
    with open(json_path, "r") as f:
        data = json.load(f)

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

    # Coerce param numerics where possible
    param_cols_all = [c for c in df.columns if c.startswith("param_")]
    for c in param_cols_all:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="ignore")

    return df


def split_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    model_cols = sorted([c for c in df.columns if c.startswith("model_")])
    param_cols_all = [c for c in df.columns if c.startswith("param_")]
    param_numeric_cols = sorted([c for c in param_cols_all if pd.api.types.is_numeric_dtype(df[c])])
    param_categorical_cols = sorted(list(set(param_cols_all) - set(param_numeric_cols)))
    return model_cols, param_numeric_cols, param_categorical_cols


def build_numeric_histograms(df: pd.DataFrame, param_numeric_cols: List[str]) -> go.Figure:
    fig = go.Figure()
    if not param_numeric_cols:
        return fig
    for i, col in enumerate(param_numeric_cols):
        fig.add_trace(go.Histogram(x=df[col], name=col, visible=(i == 0)))
    buttons = []
    for i, col in enumerate(param_numeric_cols):
        vis = [False] * len(param_numeric_cols)
        vis[i] = True
        buttons.append(dict(method="update",
                            label=col.replace("param_", ""),
                            args=[{"visible": vis},
                                  {"title": f"Distribution of {col.replace('param_', '')}"}]))
    fig.update_layout(
        title=f"Distribution of {param_numeric_cols[0].replace('param_', '')}",
        xaxis_title="Value",
        yaxis_title="Count",
        updatemenus=[dict(buttons=buttons, direction="down", showactive=True, x=1.02, y=1,
                          xanchor="left", yanchor="top", active=0)]
    )
    return fig


def build_categorical_bars(df: pd.DataFrame, param_categorical_cols: List[str]) -> go.Figure:
    fig = go.Figure()
    if not param_categorical_cols:
        return fig
    for i, col in enumerate(param_categorical_cols):
        counts = df[col].fillna("None").astype(str).value_counts().sort_values(ascending=False)
        fig.add_trace(go.Bar(x=counts.index.tolist(), y=counts.values.tolist(), name=col, visible=(i == 0)))
    buttons = []
    for i, col in enumerate(param_categorical_cols):
        vis = [False] * len(param_categorical_cols)
        vis[i] = True
        buttons.append(dict(method="update",
                            label=col.replace("param_", ""),
                            args=[{"visible": vis},
                                  {"title": f"Counts of {col.replace('param_', '')}"}]))
    fig.update_layout(
        title=f"Counts of {param_categorical_cols[0].replace('param_', '')}",
        xaxis_title="Category",
        yaxis_title="Count",
        updatemenus=[dict(buttons=buttons, direction="down", showactive=True, x=1.02, y=1,
                          xanchor="left", yanchor="top", active=0)]
    )
    return fig


def build_corr_heatmap(df: pd.DataFrame, model_cols: List[str], param_numeric_cols: List[str]) -> go.Figure:
    fig = go.Figure()
    if not (model_cols and param_numeric_cols):
        return fig
    corr_matrix = pd.DataFrame(index=model_cols, columns=param_numeric_cols, dtype=float)
    for m in model_cols:
        for p in param_numeric_cols:
            x = df[p]; y = df[m]
            corr_matrix.loc[m, p] = x.corr(y) if (x.notna().sum() > 2 and y.notna().sum() > 2) else np.nan
    z = corr_matrix.values.astype(float)
    xlabels = [c.replace("param_", "") for c in corr_matrix.columns]
    ylabels = [c.replace("model_", "") for c in corr_matrix.index]
    fig.add_trace(go.Heatmap(z=z, x=xlabels, y=ylabels, zmin=-1, zmax=1, colorscale="RdBu"))
    fig.update_layout(title="Correlation: Sylenth1 Parameters vs. AudioCommons Timbral Models",
                      xaxis_title="Parameters", yaxis_title="Timbral Models")
    return fig


def build_pca(df: pd.DataFrame, model_cols: List[str], param_numeric_cols: List[str]) -> go.Figure:
    fig = go.Figure()
    if not (param_numeric_cols and len(df) >= 2):
        return fig
    X = df[param_numeric_cols].fillna(df[param_numeric_cols].median())
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    Xp = pca.fit_transform(Xs)
    df_embed = pd.DataFrame({"PC1": Xp[:, 0], "PC2": Xp[:, 1], "name": df["name"], "sound_type": df["sound_type"]})
    selector_models = list(model_cols)
    for m in selector_models:
        df_embed[m] = df[m]
    for i, m in enumerate(selector_models):
        fig.add_trace(go.Scatter(
            x=df_embed["PC1"], y=df_embed["PC2"], mode="markers",
            marker=dict(size=10, color=df_embed[m], colorbar=dict(title=m.replace("model_", "")), colorscale="Viridis"),
            name=m.replace("model_", ""), text=df_embed["name"],
            hovertemplate="<b>%{text}</b><br>PC1=%{x:.2f}, PC2=%{y:.2f}<br>" + m.replace("model_", "") + "=%{marker.color:.2f}<extra></extra>",
            visible=(i == 0)
        ))
    buttons = []
    for i, m in enumerate(selector_models):
        vis = [False] * len(selector_models); vis[i] = True
        buttons.append(dict(method="update", label=m.replace("model_", ""),
                            args=[{"visible": vis},
                                  {"title": f"PCA of Parameters — colored by {m.replace('model_', '')}",
                                   "xaxis_title": "PC1", "yaxis_title": "PC2"}]))
    first_color = selector_models[0] if selector_models else ""
    fig.update_layout(
        title=f"PCA of Parameters — colored by {first_color.replace('model_', '')}",
        xaxis_title="PC1", yaxis_title="PC2",
        updatemenus=[dict(buttons=buttons, direction="down", showactive=True, x=1.02, y=1,
                          xanchor="left", yanchor="top", active=0)]
    )
    return fig


def build_scatter_explorer(df: pd.DataFrame, model_cols: List[str], param_numeric_cols: List[str]) -> go.Figure:
    fig = go.Figure()
    if not (param_numeric_cols and model_cols):
        return fig

    # Defaults tied to the *first* items in each dropdown so labels, data, and menu show consistent state
    default_x = param_numeric_cols[0]
    default_y = model_cols[0]

    fig.add_trace(go.Scatter(
        x=df[default_x], y=df[default_y], mode="markers", text=df["name"],
        hovertemplate="<b>%{text}</b><br>X=%{x:.3g}<br>Y=%{y:.3g}<extra></extra>"
    ))
    fig.update_layout(
        title=f"Scatter Explorer — {default_x.replace('param_', '')} vs {default_y.replace('model_', '')}",
        xaxis_title=default_x.replace("param_", ""), yaxis_title=default_y.replace("model_", "")
    )

    x_series = {col: df[col] for col in param_numeric_cols}
    y_series = {col: df[col] for col in model_cols}

    x_buttons = [dict(
        method="update",
        label=xcol.replace("param_", ""),
        args=[{"x": [x_series[xcol]]},
              {"xaxis": {"title": {"text": xcol.replace("param_", "")}},
               "title": f"Scatter Explorer — {xcol.replace('param_', '')} vs {fig.layout.yaxis.title.text}"}]
    ) for xcol in param_numeric_cols]

    y_buttons = [dict(
        method="update",
        label=ycol.replace("model_", ""),
        args=[{"y": [y_series[ycol]]},
              {"yaxis": {"title": {"text": ycol.replace("model_", "")}},
               "title": f"Scatter Explorer — {fig.layout.xaxis.title.text} vs {ycol.replace('model_', '')}"}]
    ) for ycol in model_cols]

    fig.update_layout(
        updatemenus=[
            dict(buttons=x_buttons, direction="down", x=0.15, y=1.15, xanchor="left", yanchor="top", showactive=True, active=0),
            dict(buttons=y_buttons, direction="down", x=0.55, y=1.15, xanchor="left", yanchor="top", showactive=True, active=0)
        ],
        annotations=[
            dict(x=0.08, y=1.17, xref="paper", yref="paper", text="X param:", showarrow=False, align="left"),
            dict(x=0.49, y=1.17, xref="paper", yref="paper", text="Y model:", showarrow=False, align="left"),
        ]
    )
    return fig


def build_parallel_coords(df: pd.DataFrame) -> go.Figure:
    par_cols = [c for c in [
        "param_filter_a_cutoff", "param_filter_a_reso", "param_filter_a_drive",
        "param_osc_a1_voices", "param_osc_a1_detune",
        "param_mix_a", "param_distort_amount", "param_eq_treble_db",
        "model_brightness", "model_warmth", "model_hardness"
    ] if c in df.columns]

    fig = go.Figure()
    if not par_cols:
        return fig
    dims = []
    for c in par_cols:
        s = df[c].astype(float)
        dims.append(dict(label=c.replace("param_", "").replace("model_", "").title().replace("_", " "),
                         values=s))
    color_vals = df["model_brightness"] if "model_brightness" in df else np.arange(len(df))
    fig.add_trace(go.Parcoords(line=dict(color=color_vals, colorscale="Viridis"),
                               dimensions=dims, labelfont=dict(size=12)))
    fig.update_layout(title="Parallel Coordinates — Selected Params and Timbral Models")
    return fig


def to_div(fig: go.Figure) -> str:
    return plot(fig, include_plotlyjs=False, output_type="div")


def build_dashboard_html(df: pd.DataFrame, model_cols: List[str],
                         param_numeric_cols: List[str], param_categorical_cols: List[str]) -> str:
    # Build figures
    hist_fig = build_numeric_histograms(df, param_numeric_cols)
    cat_fig = build_categorical_bars(df, param_categorical_cols)
    corr_fig = build_corr_heatmap(df, model_cols, param_numeric_cols)
    pca_fig = build_pca(df, model_cols, param_numeric_cols)
    scatter_fig = build_scatter_explorer(df, model_cols, param_numeric_cols)
    par_fig = build_parallel_coords(df)

    # Stats
    n_presets = len(df)
    n_params_num = len(param_numeric_cols)
    n_params_cat = len(param_categorical_cols)

    sections = []
    sections.append("<h1 style='font-family:Inter,system-ui,sans-serif;'>Sylenth1 × AudioCommons — Interactive Dashboard</h1>")
    sections.append("<p>This dashboard explores relationships between <b>Sylenth1 parameters</b> and <b>AudioCommons timbral model</b> values across your presets.</p>")
    sections.append(f"<p><b>{n_presets}</b> presets • <b>{n_params_num}</b> numeric params • <b>{n_params_cat}</b> categorical params • <b>{len(model_cols)}</b> timbral models.</p>")

    if len(param_numeric_cols):
        sections.append("<hr><h2>1) Parameter Distributions — Numeric</h2><p>Use the dropdown to switch parameters.</p>")
        sections.append(to_div(hist_fig))

    if len(param_categorical_cols):
        sections.append("<hr><h2>1b) Parameter Distributions — Categorical</h2><p>Use the dropdown to view category counts (e.g., oscillator waveforms, filter types).</p>")
        sections.append(to_div(cat_fig))

    if len(model_cols) and len(param_numeric_cols):
        sections.append("<hr><h2>2) Correlation Heatmap</h2><p>Pearson correlation between numeric parameters (columns) and timbral models (rows).</p>")
        sections.append(to_div(corr_fig))

    if len(model_cols) and len(param_numeric_cols):
        sections.append("<hr><h2>3) PCA of Parameter Space</h2><p>Points = presets. Use the dropdown to color by a timbral descriptor.</p>")
        sections.append(to_div(pca_fig))

    if len(model_cols) and len(param_numeric_cols):
        sections.append("<hr><h2>4) Scatter Explorer</h2><p>Pick any numeric parameter for X and any timbral model for Y.</p>")
        sections.append(to_div(scatter_fig))

    if any([t.startswith("param_") for t in df.columns]):
        sections.append("<hr><h2>5) Parallel Coordinates</h2><p>Compare selected parameters and timbral descriptors per preset.</p>")
        sections.append(to_div(par_fig))

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Sylenth1 × AudioCommons Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
body{{font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; margin:24px; line-height:1.45;}}
h1{{margin-bottom:0.3rem;}}
h2{{margin-top:2rem;}}
hr{{margin:2rem 0; border:none; border-top:1px solid #eee;}}
</style>
</head>
<body>
{''.join(sections)}
</body>
</html>
"""
    return html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input_json", required=True, help="Path to timbral dataset JSON")
    ap.add_argument("--out", dest="output_html", required=True, help="Path to output HTML dashboard")
    args = ap.parse_args()

    in_path = Path(args.input_json)
    out_path = Path(args.output_html)

    df = load_and_flatten(in_path)
    model_cols, param_numeric_cols, param_categorical_cols = split_columns(df)
    html = build_dashboard_html(df, model_cols, param_numeric_cols, param_categorical_cols)

    out_path.write_text(html, encoding="utf-8")
    print(f"Dashboard written to: {out_path}")


if __name__ == "__main__":
    main()
