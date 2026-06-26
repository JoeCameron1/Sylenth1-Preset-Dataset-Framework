"""Convergent-validity check for AudioCommons descriptors on Sylenth1 A4 audio.

Addresses Reviewer 1's main concern (and Reviewer 2's echo): the AudioCommons
models were developed and validated on real-world / musical-instrument audio,
not synthesised sounds. The published dataset should demonstrate that the AC
labels are at least correlated with simple, well-understood acoustic features
on Sylenth1 output — i.e. that the labels are doing *something* sensible, not
just outputting noise on out-of-domain audio.

For each random preset (10,000 WAVs), compute simple librosa features:
  - spectral_centroid (Hz)         predicts: brightness, sharpness
  - spectral_rolloff_85 (Hz)       predicts: brightness, depth (inverse)
  - spectral_bandwidth (Hz)
  - spectral_flatness (0..1)
  - low_band_energy (0-250 Hz / total) predicts: depth, boominess, warmth
  - mid_band_energy (250-2k Hz / total) predicts: warmth
  - high_band_energy (>4k Hz / total)   predicts: sharpness, brightness (inv depth)
  - zero_crossing_rate            predicts: hardness, sharpness
  - rms                           sanity
  - spectral_centroid_log         log-transformed for monotonic ordering

Pair with stored AC descriptors and report Pearson + Spearman correlation
matrices. Highlight the expected diagonal (e.g. brightness vs centroid).

Output:
  baselines/artifacts/results/convergent_validity_correlations.csv
  baselines/artifacts/figures/convergent_validity_heatmap.png
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import stats

from baselines.common.io import (
    DATASET_PATH_DEFAULT,
    TIMBRAL_KEYS,
    descriptors_of,
    load_dataset,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BASELINES_ROOT = Path(__file__).resolve().parents[1]
RANDOM_WAV_DIR = _REPO_ROOT / "random_preset_audio_snippets"
FACTORY_WAV_DIR = _REPO_ROOT / "factory_preset_audio_snippets"
DEFAULT_CSV = _BASELINES_ROOT / "artifacts" / "results" / "convergent_validity_correlations.csv"
DEFAULT_FIG = _BASELINES_ROOT / "artifacts" / "figures" / "convergent_validity_heatmap.png"
DEFAULT_PER_PRESET = _BASELINES_ROOT / "artifacts" / "results" / "convergent_validity_per_preset.csv"

ACOUSTIC_FEATURES = (
    "spectral_centroid",
    "spectral_rolloff_85",
    "spectral_bandwidth",
    "spectral_flatness",
    "low_band_ratio",
    "mid_band_ratio",
    "high_band_ratio",
    "zero_crossing_rate",
    "rms",
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=DATASET_PATH_DEFAULT)
    ap.add_argument("--random-wav-dir", type=Path, default=RANDOM_WAV_DIR)
    ap.add_argument("--factory-wav-dir", type=Path, default=FACTORY_WAV_DIR)
    ap.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--per-preset-out", type=Path, default=DEFAULT_PER_PRESET)
    ap.add_argument("--fig-out", type=Path, default=DEFAULT_FIG)
    ap.add_argument("--limit", type=int, default=None,
                    help="optional cap on number of presets analysed (for smoke testing)")
    return ap.parse_args()


def _acoustic_features(y: np.ndarray, sr: int) -> dict:
    """Compute simple, well-understood acoustic features over a mono signal."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1) if y.shape[1] > 1 else y[:, 0]
    if not y.size or float(np.max(np.abs(y))) < 1e-9:
        return {f: np.nan for f in ACOUSTIC_FEATURES}

    # Use a small FFT consistent with 0.5s renders.
    n_fft = 1024
    hop = 256
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Per-frame spectrum -> aggregate by mean across time.
    sc = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr, n_fft=n_fft)))
    sro = float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr, n_fft=n_fft, roll_percent=0.85)))
    sb = float(np.mean(librosa.feature.spectral_bandwidth(S=S, sr=sr, n_fft=n_fft)))
    sf_flat = float(np.mean(librosa.feature.spectral_flatness(S=S)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop)))
    rms = float(np.sqrt(np.mean(y * y)))

    # Band-energy ratios via the power spectrum.
    total = float(S.sum()) + 1e-12
    low_mask = freqs < 250.0
    mid_mask = (freqs >= 250.0) & (freqs < 2000.0)
    high_mask = freqs >= 4000.0
    low = float(S[low_mask, :].sum()) / total
    mid = float(S[mid_mask, :].sum()) / total
    high = float(S[high_mask, :].sum()) / total

    return {
        "spectral_centroid": sc,
        "spectral_rolloff_85": sro,
        "spectral_bandwidth": sb,
        "spectral_flatness": sf_flat,
        "low_band_ratio": low,
        "mid_band_ratio": mid,
        "high_band_ratio": high,
        "zero_crossing_rate": zcr,
        "rms": rms,
    }


def _resolve_wav(entry: dict, random_dir: Path, factory_dir: Path) -> Path | None:
    pid = entry.get("id")
    if not pid:
        return None
    is_random = (entry.get("name") or "").strip() == "Preset"
    candidate = (random_dir if is_random else factory_dir) / f"{pid}.wav"
    return candidate if candidate.exists() else None


def main() -> int:
    args = parse_args()
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    args.per_preset_out.parent.mkdir(parents=True, exist_ok=True)
    args.fig_out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset}")
    entries = load_dataset(args.dataset)
    if args.limit:
        entries = entries[:args.limit]

    rows = []
    descriptor_vals: dict[str, list[float]] = {k: [] for k in TIMBRAL_KEYS}
    feature_vals: dict[str, list[float]] = {f: [] for f in ACOUSTIC_FEATURES}

    skipped_missing = 0
    skipped_silent = 0
    t0 = time.time()
    for i, e in enumerate(entries, 1):
        wav_path = _resolve_wav(e, args.random_wav_dir, args.factory_wav_dir)
        if wav_path is None:
            skipped_missing += 1
            continue
        try:
            y, sr = sf.read(wav_path)
        except Exception:
            skipped_missing += 1
            continue
        feats = _acoustic_features(y, sr)
        if any(not np.isfinite(v) for v in feats.values()):
            skipped_silent += 1
            continue
        d = descriptors_of(e)
        row = {"id": e["id"], "name": e.get("name"), "kind": "random" if (e.get("name") or "").strip() == "Preset" else "factory"}
        for k in TIMBRAL_KEYS:
            row[f"ac_{k}"] = d[k]
            descriptor_vals[k].append(d[k] if d[k] is not None else np.nan)
        for f, v in feats.items():
            row[f"feat_{f}"] = v
            feature_vals[f].append(v)
        rows.append(row)
        if i % 1000 == 0:
            dt = time.time() - t0
            print(f"  processed {i}/{len(entries)} ({dt:.1f}s, {dt/i*1000:.1f} ms/preset)")

    print(f"  total processed: {len(rows)}; missing wav: {skipped_missing}; silent/invalid: {skipped_silent}")

    # Pearson + Spearman correlation matrices (rows = AC descriptors, cols = features).
    desc_matrix = np.asarray([descriptor_vals[k] for k in TIMBRAL_KEYS], dtype=float)
    feat_matrix = np.asarray([feature_vals[f] for f in ACOUSTIC_FEATURES], dtype=float)
    pearson = np.full((len(TIMBRAL_KEYS), len(ACOUSTIC_FEATURES)), np.nan)
    spearman = np.full((len(TIMBRAL_KEYS), len(ACOUSTIC_FEATURES)), np.nan)
    for i, k in enumerate(TIMBRAL_KEYS):
        for j, f in enumerate(ACOUSTIC_FEATURES):
            x, y = desc_matrix[i], feat_matrix[j]
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 5:
                continue
            pearson[i, j] = float(np.corrcoef(x[mask], y[mask])[0, 1])
            spearman[i, j] = float(stats.spearmanr(x[mask], y[mask]).correlation)

    # Save correlation tables.
    with open(args.csv_out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "ac_descriptor"] + list(ACOUSTIC_FEATURES))
        for i, k in enumerate(TIMBRAL_KEYS):
            w.writerow(["pearson", k] + [f"{v:.4f}" if np.isfinite(v) else "" for v in pearson[i]])
        for i, k in enumerate(TIMBRAL_KEYS):
            w.writerow(["spearman", k] + [f"{v:.4f}" if np.isfinite(v) else "" for v in spearman[i]])
    print(f"Wrote {args.csv_out}")

    # Per-preset CSV (used by pathology selection).
    with open(args.per_preset_out, "w", newline="") as fh:
        fields = (["id", "name", "kind"]
                  + [f"ac_{k}" for k in TIMBRAL_KEYS]
                  + [f"feat_{f}" for f in ACOUSTIC_FEATURES])
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {args.per_preset_out}")

    # Heatmap of Spearman correlations (rank-based — robust to nonlinear ties).
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "figure.dpi": 150})
    fig, ax = plt.subplots(figsize=(11, 5.5))
    im = ax.imshow(spearman, aspect="auto", vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(ACOUSTIC_FEATURES)))
    ax.set_xticklabels(ACOUSTIC_FEATURES, rotation=35, ha="right")
    ax.set_yticks(range(len(TIMBRAL_KEYS)))
    ax.set_yticklabels(TIMBRAL_KEYS)
    ax.set_title("Spearman r: AudioCommons descriptors vs simple acoustic features")
    for i in range(spearman.shape[0]):
        for j in range(spearman.shape[1]):
            v = spearman[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        color="white" if abs(v) > 0.55 else "black", fontsize=9)
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Spearman r")
    plt.tight_layout()
    plt.savefig(args.fig_out, dpi=150)
    plt.close()
    print(f"Wrote {args.fig_out}")

    # Console summary — the expected "diagonal" correlations.
    print("\nKey convergent-validity correlations (Spearman r):")
    pairs = [
        ("brightness", "spectral_centroid"),
        ("brightness", "high_band_ratio"),
        ("sharpness", "spectral_centroid"),
        ("sharpness", "zero_crossing_rate"),
        ("depth", "low_band_ratio"),
        ("depth", "spectral_rolloff_85"),  # expect negative
        ("warmth", "mid_band_ratio"),
        ("warmth", "high_band_ratio"),  # expect negative
        ("boominess", "low_band_ratio"),
        ("hardness", "zero_crossing_rate"),
        ("hardness", "spectral_flatness"),
    ]
    feat_idx = {f: i for i, f in enumerate(ACOUSTIC_FEATURES)}
    descr_idx = {k: i for i, k in enumerate(TIMBRAL_KEYS)}
    for d, f in pairs:
        v = spearman[descr_idx[d], feat_idx[f]]
        print(f"  {d:>11s} <-> {f:<22s}: r={v:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
