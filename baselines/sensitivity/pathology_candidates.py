"""Algorithmic pathology candidate selection for author listening (R2).

R2 asked for a listening-test pass on "samples that strongly deviate the
most from the preset". R1 cited a specific concerning example
(``0029e854.wav``). Both ultimately want a small, curated audit set the
author can listen to and decide whether the AC descriptors agree with
human perception, OR whether the random sampler produced unmusical patches.

This script does NOT do the listening (that requires humans). It builds the
curated candidate set algorithmically using five complementary criteria, each
contributing a handful of presets:

  (A) Convergent-validity outliers — presets whose AC descriptor disagrees
      strongly with a simple acoustic predictor (e.g. high AC brightness but
      low spectral_centroid). These are the strongest "is the AC label even
      sensible here?" candidates. Needs convergent_validity_per_preset.csv.
  (B) Contradictory descriptor pairs — high brightness + high boominess,
      etc. Pairs that should normally trade off.
  (C) Roughness edge cases — the dataset has a roughness=0 spike (~9.8%).
      We pick small-non-zero values (just above the floor) and the high
      tail; the former are likely the "should this really be zero?" cases.
  (D) Z-score outliers — presets with any AC descriptor more than 3 stddev
      from the per-descriptor mean. Statistical anomalies.
  (E) Maximally distant from any factory — high-tail of assign_distance
      from splits.json (R1: "presets near (0,12.5)" / "those filled with
      blue points due to chosen interpolation strategy"). These are the
      presets most likely to be musically suspicious.

Output:
  baselines/artifacts/results/pathology_candidates.csv
  baselines/artifacts/pathology_audit/{id}.wav   — symlinks to source WAVs
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from baselines.common.io import (
    DATASET_PATH_DEFAULT,
    TIMBRAL_KEYS,
    descriptors_of,
    load_dataset,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BASELINES_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PER_PRESET = _BASELINES_ROOT / "artifacts" / "results" / "convergent_validity_per_preset.csv"
DEFAULT_SPLITS = _BASELINES_ROOT / "artifacts" / "splits.json"
DEFAULT_OUT_CSV = _BASELINES_ROOT / "artifacts" / "results" / "pathology_candidates.csv"
DEFAULT_AUDIT_DIR = _BASELINES_ROOT / "artifacts" / "pathology_audit"
RANDOM_WAV_DIR = _REPO_ROOT / "random_preset_audio_snippets"
FACTORY_WAV_DIR = _REPO_ROOT / "factory_preset_audio_snippets"

# Per-criterion expected-correlation pairs (descriptor, feature, expected sign).
# A "disagreement" is a preset whose actual z-score gap |z_descr - sign*z_feat| is largest.
CONVERGENT_PAIRS = (
    ("brightness", "spectral_centroid", +1),
    ("depth",      "low_band_ratio",    +1),
    ("boominess",  "low_band_ratio",    +1),
    ("warmth",     "high_band_ratio",   -1),
)

# (a, b) descriptor pairs that should normally trade off.
CONTRADICTORY_PAIRS = (
    ("brightness", "boominess"),  # bright = hifreq energy, boomy = lowfreq energy
    ("depth",      "sharpness"),  # depth = low/dark, sharpness = high/cutting
    ("warmth",     "sharpness"),
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-preset", type=Path, default=DEFAULT_PER_PRESET)
    ap.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--dataset", type=Path, default=DATASET_PATH_DEFAULT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_CSV)
    ap.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    ap.add_argument("--per-criterion", type=int, default=4,
                    help="presets to pick per criterion (5 criteria -> ~20 total)")
    return ap.parse_args()


def _zscore(series: pd.Series) -> pd.Series:
    m, s = series.mean(skipna=True), series.std(skipna=True)
    return (series - m) / s if s > 1e-9 else series * 0


def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.per_preset)
    print(f"Loaded per-preset stats: {len(df)} rows")

    # Z-score descriptors and features for fair criterion comparisons.
    for k in TIMBRAL_KEYS:
        df[f"z_ac_{k}"] = _zscore(df[f"ac_{k}"])
    for f in df.columns:
        if f.startswith("feat_"):
            df[f"z_{f}"] = _zscore(df[f])

    # Load splits for criterion (E).
    with open(args.splits, "r") as fh:
        splits_meta = json.load(fh)["presets"]
    df["assign_distance"] = df["id"].map(lambda i: splits_meta.get(i, {}).get("assign_distance", np.nan))

    picks: list[dict] = []
    seen: set[str] = set()

    def _take(rows: pd.DataFrame, reason: str, k: int) -> None:
        for _, r in rows.iterrows():
            if r["id"] in seen:
                continue
            picks.append({
                "id": r["id"],
                "name": r.get("name", ""),
                "kind": r.get("kind", ""),
                "reason": reason,
                **{f"ac_{x}": r.get(f"ac_{x}") for x in TIMBRAL_KEYS},
            })
            seen.add(r["id"])
            if sum(1 for p in picks if p["reason"] == reason) >= k:
                return

    # (A) Convergent-validity outliers.
    for descr, feat, sign in CONVERGENT_PAIRS:
        gap = (df[f"z_ac_{descr}"] - sign * df[f"z_feat_{feat}"]).abs()
        candidates = df.assign(_gap=gap).sort_values("_gap", ascending=False)
        _take(candidates, f"convergent_outlier:{descr}~{feat}", args.per_criterion)

    # (B) Contradictory pairs — sum of z-scores high in both.
    for a, b in CONTRADICTORY_PAIRS:
        score = df[f"z_ac_{a}"].clip(lower=0) + df[f"z_ac_{b}"].clip(lower=0)
        candidates = df.assign(_s=score).sort_values("_s", ascending=False)
        _take(candidates, f"contradictory_pair:{a}+{b}", args.per_criterion)

    # (C) Roughness edge cases:
    #   - small non-zero (likely "should be zero?" cases) — sort by ascending positive roughness
    nonzero_low = df[df["ac_roughness"] > 0].sort_values("ac_roughness", ascending=True)
    _take(nonzero_low, "roughness:small_nonzero", args.per_criterion)
    #   - very high roughness — top of the distribution
    rough_top = df.sort_values("ac_roughness", ascending=False)
    _take(rough_top, "roughness:very_high", args.per_criterion)

    # (D) Z-score outliers — any AC descriptor > 3 stddev.
    z_max = df[[f"z_ac_{k}" for k in TIMBRAL_KEYS]].abs().max(axis=1)
    z_outliers = df.assign(_z=z_max).sort_values("_z", ascending=False)
    _take(z_outliers, "extreme_zscore", args.per_criterion)

    # (E) Maximally distant from any factory parent (random presets only).
    far = df[(df["kind"] == "random") & df["assign_distance"].notna()] \
        .sort_values("assign_distance", ascending=False)
    _take(far, "max_distance_from_factory", args.per_criterion)

    # Write CSV.
    fields = ["id", "name", "kind", "reason"] + [f"ac_{k}" for k in TIMBRAL_KEYS]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for p in picks:
            w.writerow(p)
    print(f"Wrote {len(picks)} candidates -> {args.out}")

    # Copy/symlink WAVs into the audit directory (overwrite existing).
    n_copied = 0
    for p in picks:
        is_random = (p["name"] or "").strip() == "Preset"
        src = (RANDOM_WAV_DIR if is_random else FACTORY_WAV_DIR) / f"{p['id']}.wav"
        if not src.exists():
            continue
        # Make filename self-describing: kind_reason_id.wav
        safe_reason = p["reason"].replace(":", "_").replace("+", "and").replace("~", "_vs_")
        dst = args.audit_dir / f"{p['kind']}_{safe_reason}_{p['id']}.wav"
        if dst.exists():
            dst.unlink()
        shutil.copy2(src, dst)
        n_copied += 1
    print(f"Copied {n_copied} WAVs into {args.audit_dir}/")

    # Console summary by reason
    from collections import Counter
    by_reason = Counter(p["reason"] for p in picks)
    print("\nCandidates per criterion:")
    for r, n in sorted(by_reason.items()):
        print(f"  {n:>2}  {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
