# Sylenth1-10KPresets — baseline experiments

Engineering map for the baseline experiments reported in Sections 5.4
(convergent validity), 5.5 (pitch sensitivity), and 6 (baseline
implementations) of the paper: layout, reproduction commands, and headline
results.

All artefacts are written under `baselines/artifacts/` and ship with the
dataset release.

## Layout

```
baselines/
  common/
    io.py              dataset loading; factory/random by name=='Preset'
    render.py          headless renderer + AC descriptor chain
    parity_check.py    render-parity verification gate
    encoding.py        ParamCodec (754-dim mixed-head encoding)
    splits.py          recovery-based grouped train/val/test splits
    data.py            (X, Y) materialisation per split; metrics
  regression/
    train_regression.py     Baseline 1 (mean / ridge / MLP), per-seed outputs
  inversion/
    model_cvae.py           CVAE with mixed Gaussian/softmax/BCE heads
    train_cvae.py           training loop with beta-anneal
    baselines_nn.py         descriptor->train-preset NN retrieval
    eval_inversion.py       round-trip eval through the renderer
  sensitivity/
    multipitch_render.py    re-render stratified subset at A2..A6
    multipitch_analyze.py   pitch-sensitivity heatmap + signed-deviation
                            boxplots + extended statistics CSV
    param_distributions_consolidated.py   factory-vs-random distributions (Fig 4)
    convergent_validity.py  AC descriptors vs librosa features (Fig 10);
                            --replot regenerates the heatmap from the CSV
    pathology_candidates.py 44-WAV listening-audit set selector
    render_factory_wavs.py  renders the 292 factory-preset WAVs
  phase3_multiseed.sh       multi-seed runs (regression seeds 0-3; CVAE 1-3)
  aggregate_multiseed.py    mean ± std tables from the per-seed CSVs
  artifacts/
    splits.json
    encoding_codec.json
    cvae_checkpoint.pt              (seed 0)
    cvae_checkpoint_seed{1..3}.pt
    param_reconciliation.md
    pathology_audit/                44 curated WAVs + README
    results/
      render_parity_check.csv           + render_parity_notes.md
      regression_metrics_seed{0..3}.csv + per-seed MLP training histories
      regression_metrics_multiseed.csv
      inversion_metrics.csv             (n=200 smoke run) + inversion_notes.md
      inversion_metrics_full.csv        (full test, seed 0)
      inversion_metrics_full_seed{1..3}.csv
      inversion_metrics_multiseed.csv
      multipitch_renders.csv
      multipitch_sensitivity.csv        (mean/std/median abs dev, signed range)
                                        + multipitch_notes.md
      convergent_validity_correlations.csv
      convergent_validity_per_preset.csv + convergent_validity_notes.md
      pathology_candidates.csv
    figures/
      param_distributions_consolidated.png
      convergent_validity_heatmap.png
      multipitch_heatmap.png
      multipitch_boxplots.png
      pca_audiocommons_outputs/        (regenerated PCA figures + CSVs)
      per-seed regression/inversion MAE bar charts
```

## Reproducing end-to-end

```bash
source .venv/bin/activate

# render-parity gate (~21s for n=20; --noise-floor doubles renders)
python -m baselines.common.parity_check --n 20 --seed 0 --noise-floor

# grouped splits (~1 min)
python -m baselines.common.splits --seed 0

# fit codec on TRAIN
python -m baselines.common.encoding

# regression baselines (~1 min CPU per seed)
python -m baselines.regression.train_regression --seed 0

# train CVAE (~3 min CPU) then round-trip eval (~10 min for n=200)
python -m baselines.inversion.train_cvae --seed 0
python -m baselines.inversion.eval_inversion --n-test 200 --seed 0
# full test set:
# python -m baselines.inversion.eval_inversion --n-test 1481 --seed 0   # ~25 min

# multi-seed protocol behind the paper's mean ± std tables
zsh baselines/phase3_multiseed.sh        # regression seeds 0-3; CVAE train +
                                         # full-test eval seeds 1-3 (seed 0 =
                                         # cvae_checkpoint.pt / inversion_metrics_full.csv)
python baselines/aggregate_multiseed.py  # mean ± std tables from per-seed CSVs

# multipitch sensitivity (~17 min for n=200); heatmap + boxplots + stats CSV
python -m baselines.sensitivity.multipitch_render --n 200 --seed 0
python -m baselines.sensitivity.multipitch_analyze

# convergent validity (recomputes librosa features over all WAVs; use
# --replot to only re-render the heatmap from the released CSV)
python -m baselines.sensitivity.convergent_validity

# paper figures
python pca_audiocommons_analysis.py --in FINAL_timbral_dataset_audiocommons.json \
       --outdir baselines/artifacts/figures/pca_audiocommons_outputs
python -m baselines.sensitivity.param_distributions_consolidated \
       --in FINAL_timbral_dataset_audiocommons.json \
       --out baselines/artifacts/figures/param_distributions_consolidated.png
python gen_ac_hist.py          # AC descriptor histograms (Fig 3)
python gen_param_ac_corr.py    # parameter-descriptor correlation heatmap (Fig 5)
```

## Headline results

Non-deterministic methods (MLP, CVAE) are reported as mean ± standard
deviation over four training seeds (0–3); the mean predictor, ridge
regression, and NN retrieval are deterministic (single runs).

### Render parity (n=20)

For 4 of 7 descriptors the recomputed-vs-stored MAE is at or below the
pipeline noise floor (Sylenth1's own per-render variance). The chain is
reproducing the dataset within plugin nondeterminism. Full table in
`baselines/artifacts/results/render_parity_notes.md`.

### Parameter -> descriptor regression

Test macro MAE over 7 descriptors (lower is better; 0-100 units):

| Method | Macro MAE | Worst (roughness) | Best (warmth) |
|---|---:|---:|---:|
| mean predictor | 15.78 | 16.24 | 14.31 |
| ridge          | 12.10 | 12.89 | 10.36 |
| MLP            | **10.78 ± 0.12** | 12.05 ± 0.28 | 8.91 ± 0.08 |

The MLP beats ridge on every descriptor at every seed. Roughness is the
hardest descriptor to predict, warmth the easiest.

### Descriptor -> parameter inversion + round-trip (full test, n=1481)

| Method | Macro MAE | Audibility | Validity |
|---|---:|---:|---:|
| `nn` retrieval                | **2.94** | 100% | 100% |
| `cvae_sample` (z ~ N(0,I))    | 13.85 ± 0.32 | 100% | 100% |
| `cvae_mean` (z = 0)           | 16.94 ± 2.33 | 90.6% (67.1–100% per seed) | 100% |

NN retrieval sits at the pipeline noise floor (~1-2 per descriptor; NN is
within 2-4 on every descriptor); neither CVAE decoding mode approaches it
on this single-sample metric at any seed. The z=0 mode is markedly
seed-sensitive in both accuracy and audibility — two of the four seeds
produce decoders that emit near-silent patches for a substantial fraction
of targets (discussed in Section 6.4 of the paper).

### Multipitch sensitivity (n=200 presets x 5 pitches)

Mean |deviation from A4| per descriptor and pitch (0-100 units):

| descriptor | A2 | A3 | A4 | A5 | A6 |
|---|---:|---:|---:|---:|---:|
| brightness |  9.46 |  5.78 | 0.00 |  6.06 | 11.71 |
| depth      | 16.44 | 10.73 | 0.00 |  9.96 | 16.51 |
| hardness   |  8.13 |  5.96 | 0.00 |  6.07 |  8.79 |
| roughness  | 11.19 |  7.70 | 0.00 |  6.97 | 10.18 |
| warmth     |  9.97 |  6.84 | 0.00 |  7.13 | 11.28 |
| sharpness  |  8.93 |  6.25 | 0.00 |  7.13 | 13.63 |
| boominess  | 14.14 | 10.36 | 0.00 | 10.42 | 16.10 |

Depth and boominess are most pitch-sensitive (~16 units at extremes —
substantial on a 0-100 scale); hardness is most invariant (~8). The shifts
are directional (brightness/sharpness read lower at A2 and higher at A6;
depth/boominess/warmth the opposite) — see the signed-deviation boxplots at
`baselines/artifacts/figures/multipitch_boxplots.png` and the extended
statistics columns (std, median absolute deviation, signed range) in
`multipitch_sensitivity.csv`. Discussion in
`baselines/artifacts/results/multipitch_notes.md`.

### AudioCommons label validity on synth audio

Convergent-validity Spearman r across all 9,998 random presets with librosa
acoustic features computed from the source WAVs:

| AC descriptor | expected predictor | r |
|---|---|---:|
| brightness | spectral_centroid    | **+0.87** |
| depth      | low_band_ratio       | **+0.88** |
| boominess  | low_band_ratio       | **+0.88** |
| warmth     | high_band_ratio      | **-0.78** |
| sharpness  | spectral_centroid    | **+0.77** |
| hardness   | zero_crossing_rate   | **+0.69** |
| roughness  | (no dedicated psychoacoustic predictor; max +0.72 with spectral_bandwidth, a generic brightness-family feature) | — |

Five of seven descriptors clear r > 0.7 with their expected predictor —
strong evidence the AC labels are doing something sensible on Sylenth1 A4
output. Roughness is the one descriptor without a roughness-specific
predictor in the feature set (nothing measures amplitude-modulation cues);
its residual correlations ride on generic spectral features, so it is
flagged in the paper as the descriptor requiring the most caution. Notes in
`baselines/artifacts/results/convergent_validity_notes.md`; heatmap at
`baselines/artifacts/figures/convergent_validity_heatmap.png`. Companion
listening-audit set of 44 curated WAVs in
`baselines/artifacts/pathology_audit/`.

### Parameter reconciliation

`baselines/artifacts/param_reconciliation.md` — authoritative accounting of
raw plugin (246) -> spec (179 = 106 float + 55 enum + 18 bool) -> PCA-numeric
(121) counts.
