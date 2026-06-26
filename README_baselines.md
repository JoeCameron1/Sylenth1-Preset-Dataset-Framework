# Sylenth1-10KPresets — baseline experiments

Implementation of the experiments in `TISMIR_baseline_implementation_spec.md`
(read that document first; this README is the engineering map, the spec is the
research design).

All artefacts are written under `baselines/artifacts/` and are intended to
ship with the dataset release.

## Layout

```
baselines/
  common/
    io.py              dataset loading; factory/random by name=='Preset'
    render.py          headless renderer + AC descriptor chain
    parity_check.py    Task 1 verification gate
    encoding.py        ParamCodec (754-dim mixed-head encoding)
    splits.py          recovery-based grouped train/val/test splits
    data.py            (X, Y) materialisation per split; metrics
  regression/
    train_regression.py     Baseline 1 (mean / ridge / MLP)
  inversion/
    model_cvae.py           CVAE with mixed Gaussian/softmax/BCE heads
    train_cvae.py           training loop with beta-anneal
    baselines_nn.py         descriptor->train-preset NN retrieval
    eval_inversion.py       round-trip eval through the renderer
  sensitivity/
    multipitch_render.py    re-render stratified subset at A2..A6
    multipitch_analyze.py   per-descriptor pitch sensitivity + heatmap
    param_distributions_consolidated.py   Fig 4-10 replacement
  artifacts/
    splits.json
    encoding_codec.json
    cvae_checkpoint.pt
    param_reconciliation.md
    results/
      render_parity_check.csv          + render_parity_notes.md
      regression_metrics.csv           + regression_mlp_train_history.json
      inversion_metrics.csv            + inversion_notes.md
      multipitch_renders.csv
      multipitch_sensitivity.csv
    figures/
      param_distributions_consolidated.png
      pca_audiocommons_outputs/        (regenerated PCA figures)
      regression_mae_per_descriptor.png
      inversion_mae_per_descriptor.png
      multipitch_heatmap.png
```

## Reproducing end-to-end

```bash
source .venv/bin/activate

# Task 1 — render-parity gate (~21s for n=20; --noise-floor doubles renders)
python -m baselines.common.parity_check --n 20 --seed 0 --noise-floor

# Task 3 — grouped splits (~1 min)
python -m baselines.common.splits --seed 0

# Task 2 — fit codec on TRAIN
python -m baselines.common.encoding

# Task 4 — regression baselines (~1 min CPU)
python -m baselines.regression.train_regression --seed 0

# Task 5 — train CVAE (~3 min CPU) then round-trip eval (~10 min for n=200)
python -m baselines.inversion.train_cvae --seed 0
python -m baselines.inversion.eval_inversion --n-test 200 --seed 0
# full test set:
# python -m baselines.inversion.eval_inversion --n-test 1481 --seed 0   # ~25 min

# Task 7 — multipitch sensitivity (~17 min for n=200)
python -m baselines.sensitivity.multipitch_render --n 200 --seed 0
python -m baselines.sensitivity.multipitch_analyze

# Task 8 — regenerated PCA figures (Fig 14 = RdBu, consolidated PC bars)
python pca_audiocommons_analysis.py --in FINAL_timbral_dataset_audiocommons.json \
       --outdir baselines/artifacts/figures/pca_audiocommons_outputs
# consolidated param distributions (replaces Figs 4-10)
python -m baselines.sensitivity.param_distributions_consolidated \
       --in FINAL_timbral_dataset_audiocommons.json \
       --out baselines/artifacts/figures/param_distributions_consolidated.png
```

## Headline results

### Render parity (Task 1, n=20)

For 4 of 7 descriptors the recomputed-vs-stored MAE is at or below the
pipeline noise floor (Sylenth1's own per-render variance). The chain is
reproducing the dataset within plugin nondeterminism. Full table in
`baselines/artifacts/results/render_parity_notes.md`.

### Parameter -> descriptor regression (Task 4)

Test macro MAE over 7 descriptors (lower is better; 0-100 units):

| Method | Macro MAE | Worst (roughness) | Best (warmth) |
|---|---:|---:|---:|
| mean predictor | 15.78 | 16.24 | 14.31 |
| ridge          | 12.10 | 12.89 | 10.36 |
| MLP            | **10.86** | 12.28 | 8.96 |

MLP wins on every descriptor. Roughness is the hardest to predict, warmth the
easiest — this is the **opposite** of the spec's expectation, worth noting in
the revised Section 6 discussion.

### Descriptor -> parameter inversion + round-trip (Task 5, full test, n=1481)

| Method | Macro MAE | Macro RMSE | Audibility | Validity |
|---|---:|---:|---:|---:|
| `nn` retrieval                | **2.94** |  4.29 | 100.00% | 100% |
| `cvae_sample` (z ~ N(0,I))    | 13.49 | 17.23 |  99.93% | 100% |
| `cvae_mean` (z = 0)           | 18.14 | 22.19 | 100.00% | 100% |

NN retrieval sits at the pipeline noise floor (~1-2 per descriptor; NN is
within 2-4 on every descriptor). The CVAE does not beat NN on this dataset —
this is the honest finding for the rebuttal.

### Multipitch sensitivity (Task 7, n=200 presets x 5 pitches)

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
substantial on a 0-100 scale); hardness is most invariant (~8). Discussion
in `baselines/artifacts/results/multipitch_notes.md`; heatmap at
`baselines/artifacts/figures/multipitch_heatmap.png`.

### AudioCommons label validity on synth audio (R1 main concern, R2 echo)

Convergent-validity Spearman r across all 9,998 random presets with librosa
acoustic features computed from the source WAVs (one per AC descriptor):

| AC descriptor | strongest predictor | r |
|---|---|---:|
| brightness | spectral_centroid    | **+0.87** |
| depth      | low_band_ratio       | **+0.88** |
| boominess  | low_band_ratio       | **+0.88** |
| warmth     | high_band_ratio      | **-0.78** |
| sharpness  | spectral_centroid    | **+0.77** |
| hardness   | zero_crossing_rate   | **+0.69** |
| roughness  | (no simple correlate; \|r\| < 0.4 across all) | — |

Five of seven descriptors clear r > 0.7 — strong evidence the AC labels are
doing something sensible on Sylenth1 A4 output. Notes & draft paper text in
`baselines/artifacts/results/convergent_validity_notes.md`; heatmap at
`baselines/artifacts/figures/convergent_validity_heatmap.png`. Companion
listening-audit set of 44 curated WAVs in
`baselines/artifacts/pathology_audit/` (R2's revision-required item).

### Parameter reconciliation (Task 6)

`baselines/artifacts/param_reconciliation.md` — authoritative accounting of
raw plugin (246) -> spec (179 = 106 float + 55 enum + 18 bool) -> PCA-numeric
(121) counts, including resolutions of the reviewer-flagged "200+ / 102 / 61
/ 178" numbers and the 106 vs 102 PCA discrepancy.
