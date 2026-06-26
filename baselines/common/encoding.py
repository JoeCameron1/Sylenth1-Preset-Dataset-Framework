"""Canonical, reversible parameter encoding (Task 2, spec §3).

One codec shared by both baselines and the CVAE round-trip decoder. Fit on
the TRAIN split only; persisted as JSON so eval reuses identical statistics
(avoids the most common subtle leak after row splitting).

Feature set = the 179 spec keys (factory-only ``lfo_1_free``, ``lfo_2_free``,
``solo`` are dropped).

Encoding rules:
  * **float** (106): z-score using train mean/std.
  * **bool** (18): {True: 1.0, False: 0.0}.
  * **enum** (55):
      - ordered-numeric enums (synced rates, comp_ratio, oscillator
        note/octave/voices, polyphony, filter db_db): parse to scalar
        and z-score as a numeric.
      - the rest: one-hot with an extra ``<UNK>`` slot so unseen test
        categories don't crash.

Decode mirror:
  * numeric heads -> inverse z-score -> clamp to spec [min, max] (for floats)
    or snap to nearest valid enum option (for ordered-numeric enums).
  * one-hot heads -> argmax -> category string (with ``<UNK>`` ignored).
  * bool heads -> threshold at 0.5.

The codec exposes ``encoded_dim`` and slot indices so the CVAE decoder can
emit Gaussian heads for the numeric block and softmax heads per categorical.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .io import (
    FACTORY_ONLY_KEYS,
    canonical_param_keys,
    load_param_spec,
)
from .splits import ORDERED_NUMERIC_ENUM_HINTS, _is_ordered_numeric_enum, _parse_enum_scalar

UNK_TOKEN = "<UNK>"


@dataclass
class CodecBlock:
    """One numeric or categorical block in the encoded vector.

    ``slot`` is the column-range within the encoded vector occupied by this
    block: a single column for numeric/bool blocks, ``len(categories)+1``
    columns for one-hot blocks (the +1 is the UNK slot).
    """
    key: str
    kind: str  # 'float', 'numeric_enum', 'bool', 'onehot_enum'
    mean: float | None = None
    std: float | None = None
    spec_min: float | None = None  # for clamping on decode (floats only)
    spec_max: float | None = None
    enum_options: list[str] = field(default_factory=list)
    enum_scalars: list[float] = field(default_factory=list)  # for numeric_enum (parsed values)
    start: int = -1
    width: int = -1

    @property
    def end(self) -> int:
        return self.start + self.width


class ParamCodec:
    """Reversible encoder/decoder fitted on a TRAIN-split DataFrame-like view."""

    def __init__(self, spec: dict, spec_keys: list[str], blocks: list[CodecBlock]):
        self.spec = spec
        self.spec_keys = list(spec_keys)
        self.blocks = blocks
        self.encoded_dim = sum(b.width for b in blocks)

    # ----- fitting ---------------------------------------------------------

    @classmethod
    def fit(cls, train_params: list[dict], spec: dict | None = None) -> "ParamCodec":
        """Fit the codec on a list of TRAIN-split param dicts (179 keys each)."""
        spec = spec or load_param_spec()
        spec_keys = canonical_param_keys(spec)

        blocks: list[CodecBlock] = []
        for k in spec_keys:
            entry = spec[k]
            typ = (entry.get("type") or "float").lower()
            if typ == "float":
                vals = []
                for p in train_params:
                    v = p.get(k)
                    try:
                        fv = float(v)
                        if np.isfinite(fv):
                            vals.append(fv)
                    except (TypeError, ValueError):
                        pass
                arr = np.asarray(vals, dtype=float) if vals else np.zeros(1)
                mean = float(arr.mean())
                std = float(arr.std())
                blocks.append(CodecBlock(
                    key=k, kind="float", mean=mean, std=std if std > 1e-9 else 1.0,
                    spec_min=float(entry["min"]), spec_max=float(entry["max"]),
                    width=1,
                ))
            elif typ == "bool":
                blocks.append(CodecBlock(key=k, kind="bool", width=1))
            else:  # enum
                if _is_ordered_numeric_enum(k, entry):
                    options = [str(o) for o in (entry.get("options") or [])]
                    scalars = [_parse_enum_scalar(o) for o in options]
                    # All should parse if _is_ordered_numeric_enum returned True.
                    pairs = [(s, o) for s, o in zip(scalars, options) if s is not None]
                    if not pairs:
                        # Fall back to one-hot to be safe.
                        blocks.append(_fit_onehot_block(k, entry, train_params))
                        continue
                    parsed_scalars, parsed_options = zip(*pairs)
                    vals = []
                    for p in train_params:
                        v = p.get(k)
                        s = _parse_enum_scalar(v)
                        if s is not None and np.isfinite(s):
                            vals.append(s)
                    arr = np.asarray(vals, dtype=float) if vals else np.asarray(parsed_scalars, dtype=float)
                    mean = float(arr.mean())
                    std = float(arr.std())
                    blocks.append(CodecBlock(
                        key=k, kind="numeric_enum",
                        mean=mean, std=std if std > 1e-9 else 1.0,
                        enum_options=list(parsed_options),
                        enum_scalars=list(parsed_scalars),
                        width=1,
                    ))
                else:
                    blocks.append(_fit_onehot_block(k, entry, train_params))

        # Assign start/end positions in the encoded vector.
        pos = 0
        for b in blocks:
            b.start = pos
            pos += b.width
        return cls(spec, spec_keys, blocks)

    # ----- encode / decode -------------------------------------------------

    def encode(self, params: dict) -> np.ndarray:
        """Project a single 179-key param dict into the encoded vector space."""
        x = np.zeros(self.encoded_dim, dtype=np.float32)
        for b in self.blocks:
            v = params.get(b.key)
            if b.kind == "float":
                try:
                    fv = float(v)
                    if not np.isfinite(fv):
                        fv = b.mean
                except (TypeError, ValueError):
                    fv = b.mean
                x[b.start] = (fv - b.mean) / b.std
            elif b.kind == "bool":
                if isinstance(v, bool):
                    x[b.start] = 1.0 if v else 0.0
                else:
                    s = str(v).strip().lower()
                    x[b.start] = 1.0 if s in ("true", "1", "yes", "on") else 0.0
            elif b.kind == "numeric_enum":
                s = _parse_enum_scalar(v)
                if s is None or not np.isfinite(s):
                    s = b.mean
                x[b.start] = (s - b.mean) / b.std
            elif b.kind == "onehot_enum":
                sval = "<MISSING>" if v is None else str(v)
                # The UNK slot is at the end of the block.
                slot = b.enum_options.index(sval) if sval in b.enum_options else (b.width - 1)
                x[b.start + slot] = 1.0
        return x

    def encode_batch(self, params_list: list[dict]) -> np.ndarray:
        return np.stack([self.encode(p) for p in params_list], axis=0)

    def decode(self, x: np.ndarray) -> dict:
        """Map an encoded vector back to a plugin-valid param dict.

        Numeric heads -> inverse z-score -> clamp/snap; categorical -> argmax.
        """
        if x.shape[0] != self.encoded_dim:
            raise ValueError(f"expected dim {self.encoded_dim}, got {x.shape[0]}")
        out: dict[str, Any] = {}
        for b in self.blocks:
            if b.kind == "float":
                fv = float(x[b.start]) * b.std + b.mean
                fv = max(b.spec_min, min(b.spec_max, fv))
                out[b.key] = fv
            elif b.kind == "bool":
                out[b.key] = bool(float(x[b.start]) >= 0.5)
            elif b.kind == "numeric_enum":
                fv = float(x[b.start]) * b.std + b.mean
                # Snap to nearest valid option scalar.
                if b.enum_scalars:
                    idx = int(np.argmin([abs(s - fv) for s in b.enum_scalars]))
                    out[b.key] = b.enum_options[idx]
                else:
                    out[b.key] = fv
            elif b.kind == "onehot_enum":
                logits = x[b.start:b.end]
                # The last slot is UNK; choose only among real options.
                idx = int(np.argmax(logits[:-1]))
                out[b.key] = b.enum_options[idx]
        return out

    # ----- persistence -----------------------------------------------------

    def to_json(self) -> dict:
        return {
            "encoded_dim": self.encoded_dim,
            "spec_keys": self.spec_keys,
            "blocks": [
                {
                    "key": b.key, "kind": b.kind,
                    "mean": b.mean, "std": b.std,
                    "spec_min": b.spec_min, "spec_max": b.spec_max,
                    "enum_options": b.enum_options,
                    "enum_scalars": b.enum_scalars,
                    "start": b.start, "width": b.width,
                }
                for b in self.blocks
            ],
        }

    def save(self, path: Path | str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_json(), fh, indent=2)

    @classmethod
    def load(cls, path: Path | str, spec: dict | None = None) -> "ParamCodec":
        with open(path, "r") as fh:
            payload = json.load(fh)
        spec = spec or load_param_spec()
        blocks = [CodecBlock(**b) for b in payload["blocks"]]
        return cls(spec, payload["spec_keys"], blocks)


def _fit_onehot_block(k: str, spec_entry: dict, train_params: list[dict]) -> CodecBlock:
    options = [str(o) for o in (spec_entry.get("options") or [])]
    # Always reserve an UNK slot at the end so unseen/val/test categories work.
    width = len(options) + 1
    return CodecBlock(
        key=k, kind="onehot_enum",
        enum_options=options, width=width,
    )


# ---------------------------------------------------------------------------
# Top-level CLI helper: fit codec on TRAIN split and save it next to splits.
# ---------------------------------------------------------------------------

def fit_from_splits(splits_path: Path, dataset_path: Path, out_path: Path) -> ParamCodec:
    from .io import load_dataset, project_params_to_spec

    with open(splits_path, "r") as fh:
        splits = json.load(fh)
    train_ids = {pid for pid, v in splits["presets"].items() if v["split"] == "train"}

    entries = load_dataset(dataset_path)
    spec = load_param_spec()
    spec_keys = canonical_param_keys(spec)

    train_params = []
    for e in entries:
        if e.get("id") not in train_ids:
            continue
        p = project_params_to_spec(e.get("params") or {}, spec_keys)
        train_params.append(p)
    print(f"  fitting codec on {len(train_params)} TRAIN-split presets...")

    codec = ParamCodec.fit(train_params, spec)
    codec.save(out_path)
    print(f"  encoded_dim = {codec.encoded_dim}")
    counts = {
        "float": sum(1 for b in codec.blocks if b.kind == "float"),
        "bool": sum(1 for b in codec.blocks if b.kind == "bool"),
        "numeric_enum": sum(1 for b in codec.blocks if b.kind == "numeric_enum"),
        "onehot_enum": sum(1 for b in codec.blocks if b.kind == "onehot_enum"),
    }
    print(f"  blocks: {counts}")
    onehot_width = sum(b.width for b in codec.blocks if b.kind == "onehot_enum")
    numeric_width = sum(b.width for b in codec.blocks if b.kind != "onehot_enum")
    print(f"  numeric block width: {numeric_width}, onehot block width: {onehot_width}")
    print(f"  wrote {out_path}")
    return codec


def main() -> int:
    import argparse
    _BASELINES_ROOT = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="Fit ParamCodec on the TRAIN split and save it.")
    ap.add_argument("--splits", type=Path, default=_BASELINES_ROOT / "artifacts" / "splits.json")
    ap.add_argument("--dataset", type=Path,
                    default=Path(__file__).resolve().parents[2] / "FINAL_timbral_dataset_audiocommons.json")
    ap.add_argument("--out", type=Path, default=_BASELINES_ROOT / "artifacts" / "encoding_codec.json")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fit_from_splits(args.splits, args.dataset, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
