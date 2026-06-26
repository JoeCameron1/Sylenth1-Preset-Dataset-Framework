"""Recovery-based grouped train/val/test splits (Task 3, spec §4).

Random presets are 1-to-k parameter mutations of factory presets, but the
originating factory ID was never logged. Naive row-level splitting would put
near-duplicate mutant siblings in both train and test, so we:

  1. Recover the parent factory by nearest-factory assignment in the canonical
     179-key parameter space, using a mixed distance:
       - floats and ordered-numeric enums: L1, per-feature-normalized by the
         training-IQR (or range fallback) so no wide-range param dominates,
       - categorical enums and bools: Hamming.
  2. Split the 292 factory progenitors 70/15/15 seeded.
  3. Each random preset inherits its assigned parent's split.

Output: ``baselines/artifacts/splits.json`` mapping ``preset_id -> {split,
parent_factory_id, assign_distance, ambiguous}``. Released alongside the
dataset.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from .io import (
    DATASET_PATH_DEFAULT,
    FACTORY_ONLY_KEYS,
    canonical_param_keys,
    kind_of,
    load_dataset,
    load_param_spec,
)

_BASELINES_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = _BASELINES_ROOT / "artifacts" / "splits.json"

# Enums whose string options actually represent ordered scalar values; we treat
# them as numerics for the distance computation so that "1/4 -> 1/8" is closer
# than "1/4 -> 1/64". The rest are unordered categorical enums (Hamming).
ORDERED_NUMERIC_ENUM_HINTS = (
    "_rate", "comp_ratio", "delay_time_left", "delay_time_right",
    "_note", "_octave", "_voices", "polyphony",
    "filter_a_db_db", "filter_b_db_db",
)


def _is_ordered_numeric_enum(key: str, spec_entry: dict) -> bool:
    if any(h in key for h in ORDERED_NUMERIC_ENUM_HINTS):
        return True
    # If every option parses to a float (or `<num>/<num>[TD]` musical fraction),
    # treat as ordered numeric.
    opts = spec_entry.get("options") or []
    if not opts:
        return False
    parsed = [_parse_enum_scalar(o) for o in opts]
    return all(p is not None for p in parsed)


def _parse_enum_scalar(s) -> float | None:
    """Parse an ordered-numeric enum option to a scalar.

    Handles:
      - plain numbers ("3", "-7", "1.5")
      - ratio form ("2.000:1")
      - musical fractions ("1/4", "1/8T" -> *2/3, "1/4D" -> *1.5)
    """
    try:
        x = str(s).strip()
        suffix_mul = 1.0
        if x.endswith("T"):
            x = x[:-1]; suffix_mul = 2 / 3
        elif x.endswith("D"):
            x = x[:-1]; suffix_mul = 1.5
        if x.endswith(":1"):
            x = x[:-2]
        if "/" in x:
            num, den = x.split("/", 1)
            return (float(num) / float(den)) * suffix_mul
        return float(x) * suffix_mul
    except Exception:
        return None


def _vectorize_preset(params: dict, spec: dict, spec_keys: list[str]):
    """Return (num_vec, cat_vec) where num_vec is float scalars (NaN if missing)
    and cat_vec is string tokens (with '<MISSING>' fallback). The vectorization
    treats ordered-numeric enums as numeric.
    """
    num = []
    cat = []
    for k in spec_keys:
        v = params.get(k)
        typ = (spec[k].get("type") or "float").lower()
        if typ == "float":
            try:
                num.append(float(v))
            except (TypeError, ValueError):
                num.append(np.nan)
        elif typ == "bool":
            if isinstance(v, bool):
                cat.append("true" if v else "false")
            else:
                cat.append(str(v).strip().lower())
        else:  # enum
            if _is_ordered_numeric_enum(k, spec[k]):
                if v is None:
                    num.append(np.nan)
                else:
                    parsed = _parse_enum_scalar(v)
                    num.append(parsed if parsed is not None else np.nan)
            else:
                cat.append("<MISSING>" if v is None else str(v))
    return np.asarray(num, dtype=float), np.asarray(cat, dtype=object)


def _build_feature_axis(spec: dict, spec_keys: list[str]) -> tuple[list[str], list[str]]:
    """Return (numeric_keys_in_order, categorical_keys_in_order) for inspection."""
    num_keys, cat_keys = [], []
    for k in spec_keys:
        typ = (spec[k].get("type") or "float").lower()
        if typ == "float":
            num_keys.append(k)
        elif typ == "bool":
            cat_keys.append(k)
        else:
            if _is_ordered_numeric_enum(k, spec[k]):
                num_keys.append(k)
            else:
                cat_keys.append(k)
    return num_keys, cat_keys


def _norm_scales(num_matrix: np.ndarray) -> np.ndarray:
    """Per-column IQR (fall back to range, then to 1.0). NaNs ignored."""
    scales = np.ones(num_matrix.shape[1], dtype=float)
    for j in range(num_matrix.shape[1]):
        col = num_matrix[:, j]
        col = col[np.isfinite(col)]
        if col.size < 2:
            scales[j] = 1.0
            continue
        q1, q3 = np.quantile(col, [0.25, 0.75])
        iqr = q3 - q1
        if iqr > 0:
            scales[j] = float(iqr)
        else:
            rng = float(col.max() - col.min())
            scales[j] = rng if rng > 0 else 1.0
    return scales


def _mixed_distance_one_to_many(num_q: np.ndarray, cat_q: np.ndarray,
                                num_F: np.ndarray, cat_F: np.ndarray,
                                num_scales: np.ndarray) -> np.ndarray:
    """Distance from query (num_q, cat_q) to each of F factory presets.

    Numeric L1 normalized by per-feature scale (NaNs treated as zero diff —
    a missing value contributes no penalty either direction, which matches the
    "drop the 3 factory-only keys" canonical schema rule).

    Hamming on categoricals: 1 if different, 0 if same, '<MISSING>' compared
    on either side contributes 0 (same logic).
    """
    # numeric block: |a-b| / scale
    diffs = np.abs(num_F - num_q[None, :]) / num_scales[None, :]
    diffs = np.where(np.isfinite(diffs), diffs, 0.0)
    num_d = diffs.sum(axis=1)

    # categorical: vectorized broadcast comparison
    eq = (cat_F == cat_q[None, :])
    missing = (cat_F == "<MISSING>") | (cat_q[None, :] == "<MISSING>")
    diff_mask = (~eq) & (~missing)
    cat_d = diff_mask.sum(axis=1).astype(float)

    return num_d + cat_d


def _stratified_70_15_15(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    n_train = int(round(n * 0.70))
    n_val = int(round(n * 0.15))
    splits = [""] * n
    for i, j in enumerate(idx):
        if i < n_train:
            splits[j] = "train"
        elif i < n_train + n_val:
            splits[j] = "val"
        else:
            splits[j] = "test"
    return splits


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=DATASET_PATH_DEFAULT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ambiguous-percentile", type=float, default=95.0,
                    help="random presets above this percentile of assignment "
                         "distance get ambiguous=True (default 95)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset}")
    entries = load_dataset(args.dataset)
    spec = load_param_spec()
    spec_keys = canonical_param_keys(spec)

    factory = [e for e in entries if kind_of(e) == "factory"]
    random_ = [e for e in entries if kind_of(e) == "random"]
    print(f"  factory: {len(factory)}, random: {len(random_)}")

    # Sanity: confirm factory entries are 292 and random are 10000.
    if len(factory) != 292 or len(random_) != 10000:
        print(f"  WARN: expected 292/10000, got {len(factory)}/{len(random_)}")

    num_keys, cat_keys = _build_feature_axis(spec, spec_keys)
    print(f"  numeric features (floats + ordered-numeric enums): {len(num_keys)}")
    print(f"  categorical features (enums + bools): {len(cat_keys)}")

    # Vectorize factory matrix.
    F_num = np.zeros((len(factory), len(num_keys)), dtype=float)
    F_cat = np.empty((len(factory), len(cat_keys)), dtype=object)
    for i, e in enumerate(factory):
        params = {k: v for k, v in (e.get("params") or {}).items()
                  if k not in FACTORY_ONLY_KEYS}
        num_vec, cat_vec = _vectorize_preset(params, spec, spec_keys)
        F_num[i] = num_vec
        F_cat[i] = cat_vec

    # Per-feature scales fit on FACTORY ONLY — we don't want any random
    # information leaking into the assignment metric.
    scales = _norm_scales(F_num)

    # Factory 70/15/15 split, seeded.
    factory_splits = _stratified_70_15_15(len(factory), args.seed)
    counts = Counter(factory_splits)
    print(f"  factory split: train={counts['train']} val={counts['val']} test={counts['test']}")

    out: dict[str, dict] = {}
    factory_ids = [e.get("id") for e in factory]
    for e, sp in zip(factory, factory_splits):
        out[e["id"]] = {
            "split": sp,
            "parent_factory_id": e["id"],
            "assign_distance": 0.0,
            "ambiguous": False,
            "kind": "factory",
        }

    # Assign each random preset to nearest factory.
    distances: list[float] = []
    runner_up_gap: list[float] = []
    parent_for: list[str] = []
    for i, e in enumerate(random_):
        params = e.get("params") or {}
        num_vec, cat_vec = _vectorize_preset(params, spec, spec_keys)
        d = _mixed_distance_one_to_many(num_vec, cat_vec, F_num, F_cat, scales)
        best = int(np.argmin(d))
        # 2nd nearest for ambiguity check
        d2 = d.copy(); d2[best] = np.inf
        runner_idx = int(np.argmin(d2))
        gap = float(d2[runner_idx] - d[best])
        distances.append(float(d[best]))
        runner_up_gap.append(gap)
        parent_for.append(factory_ids[best])
        if (i + 1) % 1000 == 0:
            print(f"  assigned {i+1}/{len(random_)}")

    # Ambiguity: top-K by assignment distance (tail) OR tiny runner-up gap.
    cutoff = float(np.percentile(distances, args.ambiguous_percentile))
    gap_eps = float(np.median(np.asarray(runner_up_gap)) * 0.05)  # 5% of median gap
    ambig_count = 0
    for e, parent_id, d_best, gap in zip(random_, parent_for, distances, runner_up_gap):
        ambiguous = bool((d_best >= cutoff) or (gap < gap_eps))
        out[e["id"]] = {
            "split": out[parent_id]["split"],
            "parent_factory_id": parent_id,
            "assign_distance": float(d_best),
            "runner_up_gap": float(gap),
            "ambiguous": ambiguous,
            "kind": "random",
        }
        if ambiguous:
            ambig_count += 1

    # Final per-split row counts.
    per_split = Counter(v["split"] for v in out.values())
    per_split_kind = Counter((v["split"], v["kind"]) for v in out.values())
    print()
    print(f"  total presets in splits: {sum(per_split.values())}")
    for sp in ("train", "val", "test"):
        f = per_split_kind.get((sp, "factory"), 0)
        r = per_split_kind.get((sp, "random"), 0)
        print(f"    {sp}: {f + r} (factory={f}, random={r})")
    print(f"  ambiguous random presets (distance >= P{args.ambiguous_percentile:.0f} or "
          f"runner-up gap < 5% of median gap): {ambig_count}")
    print(f"  assignment-distance: min={min(distances):.3f} median={np.median(distances):.3f} "
          f"P95={cutoff:.3f} max={max(distances):.3f}")

    payload = {
        "meta": {
            "seed": args.seed,
            "n_factory": len(factory),
            "n_random": len(random_),
            "ambiguous_percentile": args.ambiguous_percentile,
            "distance_p95_cutoff": cutoff,
            "runner_up_gap_eps": gap_eps,
            "n_numeric_features": len(num_keys),
            "n_categorical_features": len(cat_keys),
            "split_counts": {
                sp: {"factory": per_split_kind.get((sp, "factory"), 0),
                     "random": per_split_kind.get((sp, "random"), 0)}
                for sp in ("train", "val", "test")
            },
        },
        "presets": out,
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
