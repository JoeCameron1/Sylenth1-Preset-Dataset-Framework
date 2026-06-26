"""Multi-pitch sensitivity render pass (Task 7, spec §8).

Stratified subset of presets re-rendered at MIDI 45, 57, 69, 81, 93
(A2, A3, A4, A5, A6) through the IDENTICAL normalisation chain used for
the A4 dataset. Saves per-(preset, pitch) descriptors to a single CSV for
the analyse step.

Scope deliberately small: the spec's whole point is to *measure* descriptor
shifts across register so the A4-only baselines stand on cited evidence, not
to extend the baselines themselves to multipitch.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from baselines.common.io import (
    DATASET_PATH_DEFAULT,
    TIMBRAL_KEYS,
    canonical_param_keys,
    descriptors_of,
    kind_of,
    load_dataset,
    load_param_spec,
    project_params_to_spec,
)
from baselines.common.render import (
    SYLENTH1_PATH_DEFAULT,
    Sylenth1Controller,
    render_and_describe,
)


_BASELINES_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = _BASELINES_ROOT / "artifacts" / "results" / "multipitch_renders.csv"
PITCHES = (45, 57, 69, 81, 93)   # A2 A3 A4 A5 A6


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=200, help="number of presets in the stratified subset (default 200)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", type=Path, default=DATASET_PATH_DEFAULT)
    ap.add_argument("--plugin", type=str, default=SYLENTH1_PATH_DEFAULT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--stratify-by", default="brightness",
                    choices=TIMBRAL_KEYS,
                    help="stratify the subset by quintiles of this descriptor (default brightness)")
    return ap.parse_args()


def _stratified_sample(entries: list[dict], n: int, by: str, seed: int) -> list[dict]:
    """Sample roughly n/5 from each quintile of the chosen descriptor (mix kind too)."""
    rng = np.random.RandomState(seed)
    vals = np.asarray([(e.get("models") or {}).get(by) for e in entries], dtype=float)
    finite = np.isfinite(vals)
    quintiles = np.quantile(vals[finite], [0.2, 0.4, 0.6, 0.8])
    bins = np.digitize(vals, quintiles)
    per_bin = max(1, n // 5)
    picked: list[int] = []
    for b in range(5):
        idx = np.flatnonzero((bins == b) & finite)
        if len(idx) == 0:
            continue
        take = min(per_bin, len(idx))
        picked.extend(rng.choice(idx, take, replace=False).tolist())
    rng.shuffle(picked)
    return [entries[i] for i in picked[:n]]


def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset}")
    entries = load_dataset(args.dataset)
    sample = _stratified_sample(entries, args.n, args.stratify_by, args.seed)
    print(f"  stratified subset: {len(sample)} presets (by quintiles of {args.stratify_by!r})")
    n_factory = sum(1 for e in sample if kind_of(e) == "factory")
    n_random = sum(1 for e in sample if kind_of(e) == "random")
    print(f"    factory={n_factory}, random={n_random}")

    spec = load_param_spec()
    spec_keys = canonical_param_keys(spec)

    print(f"Loading Sylenth1: {args.plugin}")
    controller = Sylenth1Controller(args.plugin)

    fieldnames = ["id", "name", "kind", "midi_note", "note_label"]
    for k in TIMBRAL_KEYS:
        fieldnames += [f"d_{k}"]
    fieldnames += [f"stored_a4_{k}" for k in TIMBRAL_KEYS]

    note_label = {45: "A2", 57: "A3", 69: "A4", 81: "A5", 93: "A6"}
    total = len(sample) * len(PITCHES)
    rows = []
    skipped = 0
    t0 = time.time()
    for i, entry in enumerate(sample):
        params = project_params_to_spec(entry.get("params") or {}, spec_keys)
        stored_a4 = descriptors_of(entry)
        for pitch in PITCHES:
            try:
                d = render_and_describe(controller, params, param_limits=spec, midi_note=pitch)
            except Exception as e:
                d = None
                print(f"    err preset {entry.get('id','?')} pitch {pitch}: {e}")
            if d is None:
                skipped += 1
                continue
            row = {
                "id": entry.get("id"),
                "name": entry.get("name"),
                "kind": kind_of(entry),
                "midi_note": pitch,
                "note_label": note_label[pitch],
            }
            for k in TIMBRAL_KEYS:
                row[f"d_{k}"] = d.get(k)
                row[f"stored_a4_{k}"] = stored_a4.get(k)
            rows.append(row)
        if (i + 1) % 20 == 0:
            done = (i + 1) * len(PITCHES)
            dt = time.time() - t0
            print(f"  preset {i+1}/{len(sample)} - {done}/{total} renders - {dt:.1f}s ({dt/done:.2f}s/render)")

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {args.out}")
    print(f"  rows: {len(rows)} ({skipped} skipped silent/invalid)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
