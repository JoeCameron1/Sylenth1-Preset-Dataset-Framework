"""Shared helpers to materialise (X, Y) matrices per split.

X = encoded 179-key parameter vector (codec.encoded_dim columns).
Y = 7 AudioCommons descriptors in native 0-100 units.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .encoding import ParamCodec
from .io import (
    DATASET_PATH_DEFAULT,
    TIMBRAL_KEYS,
    canonical_param_keys,
    descriptors_of,
    load_dataset,
    load_param_spec,
    project_params_to_spec,
)


_BASELINES_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS = _BASELINES_ROOT / "artifacts" / "splits.json"
DEFAULT_CODEC = _BASELINES_ROOT / "artifacts" / "encoding_codec.json"


def load_splits(path: Path | str = DEFAULT_SPLITS) -> dict:
    with open(path, "r") as fh:
        return json.load(fh)


def build_split_matrices(splits_path: Path | str = DEFAULT_SPLITS,
                         dataset_path: Path | str = DATASET_PATH_DEFAULT,
                         codec_path: Path | str = DEFAULT_CODEC,
                         splits: Iterable[str] = ("train", "val", "test"),
                         drop_ambiguous: bool = False):
    """Return dict[split_name] -> (X, Y, ids).

    Y is the raw descriptor matrix (0-100 units, NaN where missing). Use the
    train mean to impute on a per-descriptor basis when training.
    """
    splits_data = load_splits(splits_path)["presets"]
    spec = load_param_spec()
    keys = canonical_param_keys(spec)
    codec = ParamCodec.load(codec_path, spec)
    entries = load_dataset(dataset_path)
    by_id = {e["id"]: e for e in entries}

    out: dict[str, tuple] = {}
    for sp in splits:
        ids, Xs, Ys = [], [], []
        for pid, meta in splits_data.items():
            if meta["split"] != sp:
                continue
            if drop_ambiguous and meta.get("ambiguous"):
                continue
            entry = by_id.get(pid)
            if entry is None:
                continue
            params = project_params_to_spec(entry.get("params") or {}, keys)
            x = codec.encode(params)
            d = descriptors_of(entry)
            y = np.asarray([d[k] if d[k] is not None else np.nan for k in TIMBRAL_KEYS],
                           dtype=np.float32)
            ids.append(pid)
            Xs.append(x)
            Ys.append(y)
        out[sp] = (np.stack(Xs, 0), np.stack(Ys, 0), ids)
    return out, codec


def per_descriptor_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-descriptor MAE, RMSE, R^2 (with NaNs ignored), plus macro means."""
    out = {}
    for i, k in enumerate(TIMBRAL_KEYS):
        t = y_true[:, i]
        p = y_pred[:, i]
        m = np.isfinite(t) & np.isfinite(p)
        if not m.any():
            out[k] = {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "n": 0}
            continue
        ti, pi = t[m], p[m]
        mae = float(np.mean(np.abs(ti - pi)))
        rmse = float(np.sqrt(np.mean((ti - pi) ** 2)))
        ss_res = float(np.sum((ti - pi) ** 2))
        ss_tot = float(np.sum((ti - ti.mean()) ** 2)) or 1e-12
        r2 = 1.0 - ss_res / ss_tot
        out[k] = {"mae": mae, "rmse": rmse, "r2": r2, "n": int(m.sum())}
    out["macro"] = {
        "mae": float(np.mean([out[k]["mae"] for k in TIMBRAL_KEYS])),
        "rmse": float(np.mean([out[k]["rmse"] for k in TIMBRAL_KEYS])),
        "r2": float(np.mean([out[k]["r2"] for k in TIMBRAL_KEYS])),
    }
    return out
