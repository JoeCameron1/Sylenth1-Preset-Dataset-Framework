#!/bin/zsh
# Multi-seed re-runs for the non-deterministic baselines.
# MLP regression seeds 0-3; CVAE training +
# full-test round-trip eval seeds 1-3 (the released cvae_checkpoint.pt and
# inversion_metrics_full.csv are the seed-0 run; NN retrieval is
# deterministic and is not re-run).
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
R=baselines/artifacts/results
F=baselines/artifacts/figures

echo "=== [1/3] Regression seeds 0-3 ==="
for s in 0 1 2 3; do
  echo "--- regression seed $s ---"
  python -m baselines.regression.train_regression --seed $s \
    --out $R/regression_metrics_seed$s.csv \
    --fig $F/regression_mae_per_descriptor_seed$s.png
done

echo "=== [2/3] CVAE training seeds 1-3 ==="
for s in 1 2 3; do
  echo "--- cvae train seed $s ---"
  python -m baselines.inversion.train_cvae --seed $s \
    --out baselines/artifacts/cvae_checkpoint_seed$s.pt
done

echo "=== [3/3] CVAE full-test round-trip eval seeds 1-3 ==="
for s in 1 2 3; do
  echo "--- cvae eval seed $s ---"
  python -m baselines.inversion.eval_inversion \
    --ckpt baselines/artifacts/cvae_checkpoint_seed$s.pt \
    --seed $s --n-test 1481 --methods cvae_mean cvae_sample \
    --out $R/inversion_metrics_full_seed$s.csv \
    --fig $F/inversion_mae_per_descriptor_seed$s.png
done

echo "PHASE3 DONE"
