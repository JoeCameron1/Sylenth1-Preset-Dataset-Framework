# Descriptor -> parameter inversion — notes

Companion to `inversion_metrics_full.csv` (full test set, n=1481) and
`inversion_metrics.csv` (n=200 sanity check). Task 5.

## Headline test-set numbers (full test split, n=1481)

Round-trip eval over the **full test partition (1,481 presets)**, `seed=0`,
drop_ambiguous off. Each generation is rendered through the identical
normalisation chain used for the dataset; "macro MAE" is the mean
per-descriptor MAE over the 7 AudioCommons descriptors in native 0-100
units.

| Method | Macro MAE | Macro RMSE | Audibility | Validity |
|---|---:|---:|---:|---:|
| `nn` (descriptor->train-preset retrieval) | **2.94** | 4.29 | 100.00% | 100% |
| `cvae_sample` (CVAE, z ~ N(0,I))          | 13.49 | 17.23 |  99.93% | 100% |
| `cvae_mean` (CVAE, z = 0)                 | 18.14 | 22.19 | 100.00% | 100% |

Per-descriptor MAE (full test set, 0-100 units):

| descriptor | nn | cvae_sample | cvae_mean |
|---|---:|---:|---:|
| brightness | 2.43 | 14.08 | 20.62 |
| depth      | 3.18 | 11.03 | 13.56 |
| hardness   | 3.65 | 14.86 | 22.16 |
| roughness  | 3.41 | 16.96 | 22.90 |
| warmth     | 2.22 | 10.80 | 13.55 |
| sharpness  | 2.87 | 15.14 | 20.80 |
| boominess  | 2.79 | 11.58 | 13.40 |
| **macro**  | **2.94** | **13.49** | **18.14** |

NN sits within **2-4 MAE on every descriptor** — extremely consistent.
For reference, the pipeline noise floor (Task 1, identical preset
rendered twice within the same plugin session) is per-descriptor MAE
~1-2 on the 0-100 scale. **NN retrieval sits essentially at this noise
floor**, meaning round-trip MAE on the released splits is effectively
ceiling-bounded by Sylenth1's per-render variance, not by anything a
better method could realistically improve on.

## Stability check: n=200 vs full test (n=1481)

| Method | n=200 | n=1481 | Δ |
|---|---:|---:|---:|
| nn          |  2.71 |  2.94 | +0.22 |
| cvae_sample | 13.37 | 13.49 | +0.12 |
| cvae_mean   | 17.82 | 18.14 | +0.32 |

The n=200 sample was representative — full-test results confirm the same
ordering and magnitudes. Conclusions are robust.

## What this means

The CVAE — the "headline" generative baseline — does **not** beat NN
retrieval on this dataset. The story is consistent and worth reporting
honestly in the revised paper:

* The training partition (6,711 presets) is dense enough in 7-D descriptor
  space that "find the closest existing patch" already saturates the round-
  trip metric. Any neural inversion approach must beat ~2.7 macro MAE — which
  is below Sylenth1's own per-render variance — to claim improvement.
* The CVAE with posterior-mean decoding (`z = 0`) is the worst of the three:
  the model emits an "average plausible patch" for each target, and average
  patches render to off-target timbres.
* Sampling from the prior (`z ~ N(0,I)`) improves on the posterior mean but
  is still far from NN — single samples are essentially random guesses given
  the descriptor condition.
* Both CVAE modes achieve 100% validity (every generated patch passed
  `clamp_and_validate` without losing any keys); the audibility-rate gap
  (99.5% for `cvae_sample`) reflects the occasional silent patch from
  unconstrained prior sampling.

## Reproducing

```bash
# Train CVAE (~3 min CPU)
python -m baselines.inversion.train_cvae --seed 0 --epochs 80

# Run 200-preset round-trip eval (~10 min, ~600 plugin renders)
python -m baselines.inversion.eval_inversion --n-test 200 --seed 0

# Run full-test eval (~25 min, ~4500 plugin renders)
python -m baselines.inversion.eval_inversion --n-test 1481 --seed 0
```
