"""Render the 292 factory preset WAVs to ship with the repo (R1).

Each factory preset's stored params are applied to a freshly-loaded plugin,
rendered at A4 (the dataset's reference pitch) through the IDENTICAL
normalisation chain used to build the dataset (peak-normalise -> loudness
normalise to -23 LUFS), and written as 16-bit mono WAV to
``factory_preset_audio_snippets/{id}.wav``. Matches the existing
``random_preset_audio_snippets/`` folder structure.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from baselines.common.io import (
    DATASET_PATH_DEFAULT,
    canonical_param_keys,
    kind_of,
    load_dataset,
    load_param_spec,
    project_params_to_spec,
)
from baselines.common.render import (
    SYLENTH1_PATH_DEFAULT,
    Sylenth1Controller,
    clamp_and_validate_params,
    loudness_normalize_audio,
    render_snippet,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = _REPO_ROOT / "factory_preset_audio_snippets"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=DATASET_PATH_DEFAULT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--plugin", type=str, default=SYLENTH1_PATH_DEFAULT)
    ap.add_argument("--midi-note", type=int, default=69)
    ap.add_argument("--velocity", type=int, default=100)
    ap.add_argument("--duration", type=float, default=0.5)
    ap.add_argument("--sample-rate", type=int, default=44100)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset}")
    entries = load_dataset(args.dataset)
    factory = [e for e in entries if kind_of(e) == "factory"]
    print(f"  factory presets: {len(factory)}")

    spec = load_param_spec()
    spec_keys = canonical_param_keys(spec)

    print(f"Loading Sylenth1: {args.plugin}")
    controller = Sylenth1Controller(args.plugin)

    written = 0
    skipped = 0
    t0 = time.time()
    for i, e in enumerate(factory, 1):
        pid = e.get("id", f"factory{i:03d}")
        out_path = args.out / f"{pid}.wav"
        if out_path.exists():
            continue
        params = project_params_to_spec(e.get("params") or {}, spec_keys)
        valid = clamp_and_validate_params(params, spec)
        controller.set_params(valid)
        y, sr = render_snippet(controller, midi_note=args.midi_note,
                               velocity=args.velocity, duration=args.duration,
                               sample_rate=args.sample_rate)
        if y is None or not y.size:
            skipped += 1
            continue
        y = loudness_normalize_audio(y, fs=sr, target_lufs=-23.0)
        sf.write(out_path, y, sr, subtype="PCM_16")
        written += 1
        if i % 50 == 0:
            dt = time.time() - t0
            print(f"  rendered {i}/{len(factory)} in {dt:.1f}s ({dt/i:.2f}s/preset)")

    print(f"\nWrote {written} WAVs to {args.out}/ (skipped {skipped})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
