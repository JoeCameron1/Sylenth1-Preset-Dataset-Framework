# Sylenth1-Preset-Dataset-Framework

## Dataset & Code Repository for the paper "A Dataset and Analysis Framework for Large-Scale Synthesizer Preset Generation"

This repository contains a dataset of 10,292 presets from the Sylenth1 synthesizer (292 factory + 10,000 randomly generated) along with the seven AudioCommons timbral model values for each preset, the code used to build the dataset, and the baseline experiments reported in the paper (parameter→descriptor regression, descriptor→parameter inversion via nearest-neighbour retrieval and a CVAE, multi-pitch sensitivity, convergent validity against librosa acoustic features, and a 44-WAV listening-audit set).

Below is an explanation of each file and folder in this repository.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_VARIABLES=True pip install -r requirements.txt
```

> **Note on the `SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_VARIABLES=True` prefix:** `timbral_models 0.4.0` declares its scikit-learn dependency under the deprecated `sklearn` PyPI alias, which PyPI now refuses to install unless this opt-in flag is set. The flag only affects the alias shim; `scikit-learn==1.7.2` is installed normally from `requirements.txt`. Without the flag the install fails partway through.

The custom Sylenth1 controller app and any code that re-renders presets require macOS with Sylenth1 installed at `/Library/Audio/Plug-Ins/VST3/Sylenth1.vst3`. The dataset JSONs, all analysis scripts, and the baseline training / evaluation scripts (regression, NN retrieval, CVAE training) work on any platform; only round-trip evaluation, multi-pitch re-rendering, and the parity-check gate need the plugin.

## Dataset files

- ***[Sylenth1_Full_Factory_Presets_AC_A4.json](Sylenth1_Full_Factory_Presets_AC_A4.json)*** — the 292 factory presets of Sylenth1 with their corresponding AudioCommons timbral model values.
- ***[Full_Random_Presets.json](Full_Random_Presets.json)*** — the 10,000 randomly generated presets with their corresponding AudioCommons timbral model values.
- ***[FINAL_timbral_dataset_audiocommons.json](FINAL_timbral_dataset_audiocommons.json)*** — the combined 10,292-entry dataset (292 factory + 10,000 random) used throughout the paper.
- ***[random_preset_audio_snippets/](random_preset_audio_snippets/)*** — the rendered audio snippets for the 10,000 random presets used to compute their AudioCommons timbral model values.
- ***[factory_preset_audio_snippets/](factory_preset_audio_snippets/)*** — the rendered audio snippets for the 292 factory presets.

## Parameter / preset specification files

- ***[sylenth1_params.json](sylenth1_params.json)*** — canonical 179-parameter specification (106 float, 55 enum, 18 bool) with value ranges, enum options, and descriptions. This is the single source of truth for parameter validity used by the controller app and the baseline codec.
- ***[sylenth_defaults.json](sylenth_defaults.json)*** — default parameter values for Sylenth1's initial (INIT) preset on startup.
- ***[user_presets.json](user_presets.json)*** — saved user presets from the custom controller app.

## Original analysis & dataset-building scripts

- ***[sylenth1_preset_custom_app.py](sylenth1_preset_custom_app.py)*** — custom GUI app used to interact with and control Sylenth1; presets were created and added to the dataset via this app. *Usage*: `python sylenth1_preset_custom_app.py`.
- ***[build_sylenth1_timbre_dashboard.py](build_sylenth1_timbre_dashboard.py)*** — builds the Plotly dashboards for analysing the contents and distributions of the dataset JSONs. The distribution and correlation figures in Sections 5.1 and 5.2 are produced by this script. *Usage*: `python build_sylenth1_timbre_dashboard.py --in [TIMBRAL_DATASET].json --out [DASHBOARD].html`.
- ***[pca_audiocommons_analysis.py](pca_audiocommons_analysis.py)*** — runs the PCA analysis on the dataset; the PCA figures in Section 5.3 are produced by this script (the revised figures use a diverging RdBu colormap and consolidated PC bars). *Usage*: `python pca_audiocommons_analysis.py --in [TIMBRAL_DATASET].json --outdir [OUT_DIR]`.
- ***[gen_ac_hist.py](gen_ac_hist.py)*** — produces the AudioCommons descriptor histograms (Section 5.1). *Usage*: `python gen_ac_hist.py` (input path set inside the script).
- ***[gen_pca_rand_vs_fact.py](gen_pca_rand_vs_fact.py)*** — produces the factory-vs-random PCA overlay (Section 5.4). *Usage*: `python gen_pca_rand_vs_fact.py`.
- ***[Sylenth1_Full_FactoryPresets_Timbre_Dashboard.html](Sylenth1_Full_FactoryPresets_Timbre_Dashboard.html)*** — the Plotly dashboard for the 292 factory presets.
- ***[Full_Random_Presets_Dashboard.html](Full_Random_Presets_Dashboard.html)*** — the Plotly dashboard for the 10,000 random presets.
- ***[FINAL_Dataset_Dashboard.html](FINAL_Dataset_Dashboard.html)*** — the Plotly dashboard for the combined 10,292-entry dataset.

## Baseline experiments (added in revision)

The `baselines/` directory contains the implementations and released artefacts for everything reported in the revised Sections 5.5 (convergent validity), 5.6 (pitch sensitivity), and 6 (baseline implementations). The engineering map and headline results table live in [README_baselines.md](README_baselines.md).

**Code:**

- ***[baselines/common/](baselines/common/)*** — shared infrastructure: headless renderer + AudioCommons descriptor chain (`render.py`), render-parity gate (`parity_check.py`), recovery-based grouped train/val/test splits (`splits.py`), mixed-head parameter codec (`encoding.py`), dataset loading (`io.py`, `data.py`).
- ***[baselines/regression/train_regression.py](baselines/regression/train_regression.py)*** — parameter→descriptor regression baselines (mean predictor, ridge, MLP); reproduces the Section 6 regression table.
- ***[baselines/inversion/](baselines/inversion/)*** — descriptor→parameter inversion baselines: CVAE definition and training (`model_cvae.py`, `train_cvae.py`), nearest-neighbour retrieval (`baselines_nn.py`), and round-trip evaluation through the renderer (`eval_inversion.py`).
- ***[baselines/sensitivity/](baselines/sensitivity/)*** — convergent-validity analysis against librosa acoustic features (`convergent_validity.py`), multi-pitch re-rendering and sensitivity heatmap (`multipitch_render.py`, `multipitch_analyze.py`), the consolidated parameter distributions figure that replaces the prior Figs 4–10 (`param_distributions_consolidated.py`), and the pathology-audit candidate selector (`pathology_candidates.py`).

**Released artefacts (`baselines/artifacts/`):**

- ***[baselines/artifacts/splits.json](baselines/artifacts/splits.json)*** — the 70/15/15 recovery-based grouped split assignments (each random preset inherits its nearest-factory parent's split).
- ***[baselines/artifacts/encoding_codec.json](baselines/artifacts/encoding_codec.json)*** — parameter codec fitted on the training split only.
- ***[baselines/artifacts/cvae_checkpoint.pt](baselines/artifacts/cvae_checkpoint.pt)*** — trained CVAE weights.
- ***[baselines/artifacts/param_reconciliation.md](baselines/artifacts/param_reconciliation.md)*** — the parameter accounting (246 host params → 179 timbre-relevant → 121 PCA-numeric features) reported in Table 1.
- ***[baselines/artifacts/pathology_audit/](baselines/artifacts/pathology_audit/)*** — 44-WAV listening-audit set algorithmically selected by 11 complementary pathology criteria, provided to support qualitative inspection of edge-case random presets (see the folder's README for the criterion-to-file mapping).
- ***[baselines/artifacts/results/](baselines/artifacts/results/)*** — per-task CSVs and notes: render-parity check, regression metrics, inversion metrics (NN and CVAE, partial and full test set), multipitch renders and per-descriptor sensitivity, convergent-validity Spearman correlations (overall and per preset), pathology-audit candidates, and accompanying `*_notes.md` files with discussion.
- ***[baselines/artifacts/figures/](baselines/artifacts/figures/)*** — all new figures referenced in the revised paper: `param_distributions_consolidated.png`, `convergent_validity_heatmap.png`, `multipitch_heatmap.png`, `regression_mae_per_descriptor.png`, `inversion_mae_per_descriptor.png` (and `_full.png`).

End-to-end reproduction commands are listed in [README_baselines.md](README_baselines.md#reproducing-end-to-end).

## Other files

- ***[requirements.txt](requirements.txt)*** — pip dependencies for the custom controller app and the baseline experiments. Install with the `SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_VARIABLES=True` prefix described in the *Installation* section above.
- ***[README_baselines.md](README_baselines.md)*** — engineering map for `baselines/` with reproduction commands and headline results tables.
