"""Dataset loading + factory/random partition rule.

Single source of truth for:
  - loading FINAL_timbral_dataset_audiocommons.json into a DataFrame-friendly
    list of (id, name, kind, params, descriptors) records,
  - applying the name-based factory/random rule (every entry has
    source=="user", so `source` is NOT a usable label),
  - pulling out the 7 AudioCommons descriptors per entry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH_DEFAULT = _REPO_ROOT / "FINAL_timbral_dataset_audiocommons.json"
PARAMS_SPEC_PATH = _REPO_ROOT / "sylenth1_params.json"

TIMBRAL_KEYS = ("brightness", "depth", "hardness", "roughness", "warmth", "sharpness", "boominess")

# Factory entries carry these three extra keys; drop them so factory and
# random rows share an identical 179-key schema.
FACTORY_ONLY_KEYS = ("lfo_1_free", "lfo_2_free", "solo")


def load_param_spec(path: Path | str = PARAMS_SPEC_PATH) -> dict:
    with open(path, "r") as fh:
        return json.load(fh)


def load_dataset(path: Path | str = DATASET_PATH_DEFAULT) -> list[dict]:
    """Load the dataset. Each entry retains its raw shape from JSON."""
    with open(path, "r") as fh:
        return json.load(fh)


def kind_of(entry: dict) -> str:
    """'random' if entry['name'].strip() == 'Preset' else 'factory'.

    NOTE: do NOT use entry['source']; every entry is source=='user'.
    """
    name = (entry.get("name") or "").strip()
    return "random" if name == "Preset" else "factory"


def canonical_param_keys(spec: dict | None = None) -> list[str]:
    """The 179 spec keys (everything in sylenth1_params.json)."""
    if spec is None:
        spec = load_param_spec()
    return sorted(spec.keys())


def project_params_to_spec(params: dict, spec_keys: Iterable[str]) -> dict:
    """Drop factory-only keys (lfo_1_free, lfo_2_free, solo); keep only spec keys."""
    keys = set(spec_keys)
    return {k: v for k, v in params.items() if k in keys}


def descriptors_of(entry: dict) -> dict:
    """Return {key: float|None} for the 7 timbral descriptors."""
    models = entry.get("models", {}) or {}
    out = {}
    for k in TIMBRAL_KEYS:
        v = models.get(k)
        try:
            v = float(v)
            if not np.isfinite(v):
                v = None
        except (TypeError, ValueError):
            v = None
        out[k] = v
    return out
