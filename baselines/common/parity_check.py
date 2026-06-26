"""Render-parity gate for Task 1.

Sample N presets from FINAL_timbral_dataset_audiocommons.json, feed each one's
stored ``params`` through ``render_and_describe()``, and compare the 7
recomputed AudioCommons descriptors to the values shipped in the dataset.

Pass criterion (per spec §2): per-descriptor MAE far below 1.0 on the 0-100
scale (ideally well below). Write the per-preset diffs to
``baselines/artifacts/results/render_parity_check.csv`` and print a summary.

Usage:
    python -m baselines.common.parity_check --n 20 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

from .io import (
    DATASET_PATH_DEFAULT,
    TIMBRAL_KEYS,
    descriptors_of,
    load_dataset,
    load_param_spec,
)
from .render import Sylenth1Controller, SYLENTH1_PATH_DEFAULT, render_and_describe


_BASELINES_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = _BASELINES_ROOT / "artifacts" / "results" / "render_parity_check.csv"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20, help="number of presets to check (default 20)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for preset selection")
    ap.add_argument("--dataset", type=Path, default=DATASET_PATH_DEFAULT)
    ap.add_argument("--plugin", type=str, default=SYLENTH1_PATH_DEFAULT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--kind", choices=("random", "factory", "any"), default="random",
                    help="which subset to sample from (spec asks for 20 random presets)")
    ap.add_argument("--noise-floor", action="store_true",
                    help="ALSO render each preset twice and report within-pipeline variance "
                         "(the irreducible noise from plugin nondeterminism).")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset}")
    entries = load_dataset(args.dataset)
    spec = load_param_spec()

    if args.kind == "random":
        pool = [e for e in entries if (e.get("name") or "").strip() == "Preset"]
    elif args.kind == "factory":
        pool = [e for e in entries if (e.get("name") or "").strip() != "Preset"]
    else:
        pool = list(entries)

    rng = random.Random(args.seed)
    sampled = rng.sample(pool, min(args.n, len(pool)))
    print(f"Sampled {len(sampled)} {args.kind} presets from a pool of {len(pool)}.")

    print(f"Loading Sylenth1 plugin: {args.plugin}")
    controller = Sylenth1Controller(args.plugin)

    rows = []
    skipped = 0
    t0 = time.time()
    for i, entry in enumerate(sampled, 1):
        stored = descriptors_of(entry)
        params = entry.get("params", {}) or {}
        try:
            recomputed = render_and_describe(controller, params, param_limits=spec)
            recomputed2 = (render_and_describe(controller, params, param_limits=spec)
                           if args.noise_floor else None)
        except Exception as e:
            print(f"  [{i}/{len(sampled)}] {entry.get('id','?')} render error: {e}")
            recomputed = None
            recomputed2 = None
        if recomputed is None or any(recomputed.get(k) is None for k in TIMBRAL_KEYS):
            skipped += 1
            print(f"  [{i}/{len(sampled)}] {entry.get('id','?')} skipped (silent/invalid)")
            continue
        row = {
            "id": entry.get("id"),
            "name": entry.get("name"),
            "seed_index": i,
        }
        for k in TIMBRAL_KEYS:
            s = stored.get(k)
            r = recomputed.get(k)
            row[f"stored_{k}"] = s
            row[f"recomputed_{k}"] = r
            row[f"abs_diff_{k}"] = abs(r - s) if (s is not None and r is not None) else None
            if args.noise_floor and recomputed2 is not None:
                r2 = recomputed2.get(k)
                row[f"recomputed2_{k}"] = r2
                row[f"noise_diff_{k}"] = abs(r - r2) if (r is not None and r2 is not None) else None
        rows.append(row)
        if i % 5 == 0:
            dt = time.time() - t0
            print(f"  [{i}/{len(sampled)}] elapsed {dt:.1f}s ({dt/i:.2f}s/preset)")

    if not rows:
        print("ERROR: no presets produced valid recomputed descriptors; cannot judge parity.")
        return 2

    print("\nRender parity summary (lower is better; spec target MAE << 1.0 on 0-100 scale,")
    print("but plugin nondeterminism imposes a noise floor — use --noise-floor to measure it).")
    print(f"  presets checked: {len(rows)}, skipped: {skipped}")
    summary = {}
    for k in TIMBRAL_KEYS:
        diffs = [r[f"abs_diff_{k}"] for r in rows if r[f"abs_diff_{k}"] is not None]
        if not diffs:
            print(f"  {k:>10s}: no valid diffs")
            continue
        mae = sum(diffs) / len(diffs)
        mx = max(diffs)
        summary[k] = (mae, mx)
        extra = ""
        if args.noise_floor:
            nd = [r[f"noise_diff_{k}"] for r in rows if r.get(f"noise_diff_{k}") is not None]
            if nd:
                extra = f"   pipeline-noise MAE={sum(nd)/len(nd):6.3f} (max={max(nd):6.3f})"
        print(f"  {k:>10s}: MAE={mae:6.3f}   max|diff|={mx:6.3f}   (n={len(diffs)}){extra}")

    fieldnames = ["id", "name", "seed_index"]
    for k in TIMBRAL_KEYS:
        fieldnames += [f"stored_{k}", f"recomputed_{k}", f"abs_diff_{k}"]
        if args.noise_floor:
            fieldnames += [f"recomputed2_{k}", f"noise_diff_{k}"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
