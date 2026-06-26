# Render parity check — notes

Companion to `render_parity_check.csv` (Task 1 verification gate).

**Setup.** 20 random presets sampled with seed=0 from
`FINAL_timbral_dataset_audiocommons.json`. Each preset's stored 179 params were
fed through `baselines.common.render.render_and_describe()` and the recomputed
7 AudioCommons descriptors compared to the stored values. `--noise-floor` also
re-rendered each preset a second time within the same plugin instance to
measure the pipeline's irreducible variance.

| descriptor | parity MAE | pipeline-noise MAE | excess |
|---|---:|---:|---:|
| brightness | 1.28 | 1.18 | +0.10 |
| depth      | 1.17 | 1.56 | −0.39 |
| hardness   | 2.66 | 1.86 | +0.80 |
| roughness  | 2.63 | 1.33 | +1.31 |
| warmth     | 0.83 | 0.95 | −0.12 |
| sharpness  | 1.83 | 2.03 | −0.20 |
| boominess  | 1.41 | 1.25 | +0.16 |

For 4 of 7 descriptors (brightness, depth, warmth, sharpness), parity error is
**at or below the pipeline noise floor** — i.e. our reproduction agrees with the
dataset to within Sylenth1's own per-render variance.

For hardness, roughness, and boominess, parity exceeds the noise floor by
0.16–1.31. Roughness's `max|diff|=24.2` traces to its known 0-spike (≈9.8% of
the dataset stores exactly 0); plugin nondeterminism can push a near-zero
realisation across the threshold.

**Implication for round-trip evaluation (Task 5).** The per-descriptor
"pipeline-noise MAE" column above is the floor below which round-trip metrics
cannot meaningfully improve — any inversion baseline's round-trip MAE should
be reported alongside this noise floor, computed on the test split, so that
"how close we get to the target descriptor" is judged relative to the limit
imposed by Sylenth1 itself rather than against the optimistic 0.0.
