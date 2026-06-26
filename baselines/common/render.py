"""Headless Sylenth1 renderer + AudioCommons descriptor computation.

Lifted (logic-preserving) out of ``sylenth1_preset_custom_app.py`` so that
round-trip and sensitivity experiments can call the dataset's exact rendering
chain in a loop without a GUI.

The dataset was built as:
    set_params(clamp_and_validate(params))
    y, sr = render_snippet(midi_note=69, velocity=100, duration=0.5, sr=44100)
        # render_snippet internally peak-normalizes to 0.95
    y     = loudness_normalize(y, sr, target_lufs=-23.0)
    descr = compute_timbral_models(y, sr)

Any re-render that does not reproduce this chain exactly will absorb a
normalization artifact into the round-trip error.
"""

from __future__ import annotations

import json
from functools import wraps
from pathlib import Path
from typing import Optional

import numpy as np

# --------------------------------------------------------------------------
# Compat shims so timbral_models 0.4.0 works against current NumPy / librosa.
# These must be installed BEFORE timbral_models is imported.
# --------------------------------------------------------------------------
if not hasattr(np, "lib") or not hasattr(np.lib, "pad"):
    if not hasattr(np, "lib"):
        class _Lib:  # noqa: D401 - tiny shim type
            pass
        np.lib = _Lib()
    np.lib.pad = np.pad  # type: ignore[attr-defined]

try:
    import librosa as _librosa

    def _is_number(x) -> bool:
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    if hasattr(_librosa, "onset"):
        if hasattr(_librosa.onset, "onset_detect"):
            _orig_detect = _librosa.onset.onset_detect

            @wraps(_orig_detect)
            def _detect_compat(*args, **kwargs):
                if args:
                    kwargs.setdefault("y", args[0])
                    sr = args[1] if len(args) > 1 else kwargs.get("sr", 22050)
                    kwargs.setdefault("sr", sr if sr is not None else 22050)
                    return _orig_detect(**kwargs)
                return _orig_detect(**kwargs)

            _librosa.onset.onset_detect = _detect_compat

        if hasattr(_librosa.onset, "onset_strength"):
            _orig_strength = _librosa.onset.onset_strength

            @wraps(_orig_strength)
            def _strength_compat(*args, **kwargs):
                if args:
                    kwargs.setdefault("y", args[0])
                    if len(args) >= 2 and "sr" not in kwargs and (_is_number(args[1]) or args[1] is None):
                        kwargs["sr"] = 22050 if args[1] is None else args[1]
                    if len(args) >= 3 and "hop_length" not in kwargs and _is_number(args[2]):
                        kwargs["hop_length"] = int(args[2])
                    return _orig_strength(**kwargs)
                return _orig_strength(**kwargs)

            _librosa.onset.onset_strength = _strength_compat
except Exception:
    pass

try:
    import pyloudnorm as pyln
    _PYLN_AVAILABLE = True
except Exception:
    pyln = None
    _PYLN_AVAILABLE = False

try:
    from timbral_models import (
        timbral_brightness,
        timbral_depth,
        timbral_hardness,
        timbral_roughness,
        timbral_warmth,
        timbral_sharpness,
        timbral_booming,
    )
    _TIMBRAL_AVAILABLE = True
except Exception:
    _TIMBRAL_AVAILABLE = False

from pedalboard import load_plugin


SYLENTH1_PATH_DEFAULT = "/Library/Audio/Plug-Ins/VST3/Sylenth1.vst3"

# Path to the canonical 179-param spec; resolved relative to repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
SYLENTH1_PARAMS_JSON = _REPO_ROOT / "sylenth1_params.json"


def load_param_spec(path: Path | str = SYLENTH1_PARAMS_JSON) -> dict:
    """Load the 179-key param spec (type, min/max, options) used to validate params."""
    with open(path, "r") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Sylenth1-specific quirk: modulation-destination enums apply the *previous*
# option when you select anything beyond the first 4 entries. We pre-shift
# FORWARD by one to land on the intended choice. Load-bearing; do not remove.
# ---------------------------------------------------------------------------
BUGGY_ENUM_SHIFT_PARAMS = {
    "ymodmisc1a_dest1", "ymodmisc1a_dest2", "ymodmisc1b_dest1", "ymodmisc1b_dest2",
    "ymodmisc2a_dest1", "ymodmisc2a_dest2", "ymodmisc2b_dest1", "ymodmisc2b_dest2",
    "ymodlfo1_dest1", "ymodlfo1_dest2", "ymodlfo2_dest1", "ymodlfo2_dest2",
    "ymodenv1_dest1", "ymodenv1_dest2", "ymodenv2_dest1", "ymodenv2_dest2",
}

# Common abbreviations the app accepts when set_param sees a string enum.
_SYNTH_ALIASES = {
    "lpf": "Lowpass", "lowpass": "Lowpass",
    "bpf": "Bandpass", "bandpass": "Bandpass",
    "hpf": "Highpass", "highpass": "Highpass",
    "bypass": "Bypass",
    "Sawtooth": "Saw", "SawTooth": "Saw",
}

TIMBRAL_KEYS = ("brightness", "depth", "hardness", "roughness", "warmth", "sharpness", "boominess")


def _enum_value_variants(v) -> list[str]:
    """Return candidate string forms to match against spec enum options.

    The dataset stores some "ordered numeric" enum values as floats
    (e.g. ``osc_a1_note=3.0`` while spec options are the strings ``['-7'..'7']``).
    Try the raw string AND its int-truncated form when the float is integral.
    """
    s = str(v)
    out = [s]
    try:
        fv = float(v)
        if np.isfinite(fv) and float(int(fv)) == fv:
            out.append(str(int(fv)))
    except Exception:
        pass
    # Trailing ".0" form -> bare-int form
    if isinstance(s, str) and s.endswith(".0"):
        out.append(s[:-2])
    return out


def clamp_and_validate_params(param_changes: dict, param_limits: dict) -> dict:
    """Return a copy of ``param_changes`` restricted to spec-valid values.

    - floats clamped to [min, max]
    - enums matched case-insensitively, with int/float normalization for
      ordered-numeric enums (e.g. ``osc_a1_note=3.0`` -> option ``"3"``)
    - bools accept Python bool, "true"/"false", "1"/"0", "yes"/"no", "on"/"off"
    Keys not in ``param_limits`` are dropped.
    """
    valid: dict = {}
    for k, v in param_changes.items():
        if k not in param_limits:
            continue
        spec = param_limits[k]
        typ = (spec.get("type") or "float").lower()
        if typ == "float":
            try:
                fv = float(v)
                valid[k] = max(spec["min"], min(fv, spec["max"]))
            except Exception:
                continue
        elif typ == "enum":
            options = [str(o) for o in spec.get("options", [])]
            options_lower = [o.lower() for o in options]
            matched = None
            # Empty / missing values: prefer "None" if it's a valid option
            # (matches the implicit behaviour of mod-dest enums in the dataset).
            if v is None or (isinstance(v, str) and v.strip() == ""):
                if "None" in options:
                    matched = "None"
            if matched is None:
                for cand in _enum_value_variants(v):
                    if cand in options:
                        matched = cand
                        break
                    cl = cand.lower()
                    if cl in options_lower:
                        matched = options[options_lower.index(cl)]
                        break
            if matched is None:
                # Nearest-numeric-snap for enums whose options are scalar-parseable
                # (covers comp_ratio="5.704:1", synced rates, ratio strings).
                def _parse_num(x):
                    try:
                        s = str(x).strip()
                        if s.endswith(":1"):
                            s = s[:-2]
                        return float(s)
                    except Exception:
                        return None
                numerics = []
                for o in options:
                    nv = _parse_num(o)
                    if nv is not None:
                        numerics.append((nv, o))
                candidate = _parse_num(v)
                if numerics and candidate is not None:
                    _, matched = min(numerics, key=lambda p: abs(p[0] - candidate))
            if matched is not None:
                valid[k] = matched
        elif typ == "bool":
            if isinstance(v, bool):
                valid[k] = v
            else:
                sval = str(v).strip().lower()
                if sval in ("true", "1", "yes", "on"):
                    valid[k] = True
                elif sval in ("false", "0", "no", "off"):
                    valid[k] = False
                else:
                    opts = [str(o).lower() for o in spec.get("options", [])]
                    if sval in opts:
                        valid[k] = (sval == "true")
    return valid


class Sylenth1Controller:
    """Thin pedalboard wrapper for the Sylenth1 VST3 with the enum-shift workaround."""

    def __init__(self, plugin_path: str = SYLENTH1_PATH_DEFAULT):
        self.synth = load_plugin(plugin_path)
        self.param_names = list(self.synth.parameters.keys())
        print(f"Loaded Sylenth1 with {len(self.param_names)} parameters.")

    def get_all_params(self) -> dict:
        params: dict = {}
        for name in self.param_names:
            try:
                val = getattr(self.synth, name)
                try:
                    params[name] = float(val)
                except (TypeError, ValueError):
                    params[name] = str(val)
            except Exception:
                params[name] = None
        return params

    def set_param(self, param: str, value, apply_shim: bool = False) -> None:
        """Set a single plugin parameter.

        ``apply_shim`` (default False): pre-shift modulation-destination enums
        forward by one to compensate for Sylenth1's off-by-one selection bug.
        Use ``apply_shim=True`` ONLY when ``value`` represents the user's
        semantic intent (e.g. a dropdown choice in the GUI). The dataset's
        stored params are post-shim setattr values (because
        ``_snapshot_current_params`` reads them back from pedalboard, which
        echoes the setattr'd value, not the plugin's effectively-active option),
        so for REPLAY of stored or model-generated values, leave the shim off
        — that is the convention the dataset descriptors were computed under.
        """
        if param not in self.synth.parameters:
            return
        param_obj = self.synth.parameters[param]
        try:
            min_val = param_obj.min_value
            max_val = param_obj.max_value
            num_value = float(value)
            clamped = max(min_val, min(max_val, num_value))
            setattr(self.synth, param, clamped)
            return
        except (AttributeError, ValueError, TypeError):
            pass

        if isinstance(value, bool):
            setattr(self.synth, param, value)
            return

        valid_values = getattr(param_obj, "valid_values", None)
        if valid_values:
            val_str = str(value).lower()
            valid_lower = [str(v).lower() for v in valid_values]
            if val_str in valid_lower:
                corrected = valid_values[valid_lower.index(val_str)]
            else:
                corrected = _SYNTH_ALIASES.get(val_str)
                if corrected not in valid_values:
                    # Numeric-snap: enums like "2.001:1" -> nearest valid "X.XXX:1"
                    def _parse_num(x):
                        try:
                            s = str(x).strip()
                            if s.endswith(":1"):
                                s = s[:-2]
                            return float(s)
                        except Exception:
                            return None

                    numerics = []
                    for vv in valid_values:
                        nv = _parse_num(vv)
                        if nv is None:
                            numerics = None
                            break
                        numerics.append((nv, vv))
                    candidate = _parse_num(value)
                    if numerics is not None and candidate is not None and len(numerics):
                        _, corrected = min(numerics, key=lambda p: abs(p[0] - candidate))
                    else:
                        return

            if apply_shim and param in BUGGY_ENUM_SHIFT_PARAMS:
                opts = [str(v) for v in valid_values]
                if corrected in opts:
                    idx = opts.index(corrected)
                    if idx >= 4 and idx + 1 < len(opts):
                        corrected = opts[idx + 1]
            setattr(self.synth, param, corrected)
        else:
            try:
                setattr(self.synth, param, value)
            except Exception:
                return

    def set_params(self, param_dict: dict, apply_shim: bool = False) -> None:
        """Apply many params at once. ``apply_shim`` defaults to False (replay mode)."""
        for k, v in param_dict.items():
            self.set_param(k, v, apply_shim=apply_shim)


# ---------------------------------------------------------------------------
# Rendering & descriptor computation (free functions; no UI dependencies).
# ---------------------------------------------------------------------------

def render_snippet(controller: Sylenth1Controller,
                   midi_note: int = 69,
                   velocity: int = 100,
                   duration: float = 0.5,
                   sample_rate: int = 44100) -> tuple[Optional[np.ndarray], Optional[int]]:
    """Offline render of the current plugin state. Returns (y_mono_float32, sr) or (None, None).

    Matches the dataset-building call exactly:
      - reset=True  (clean voice state per render)
      - mono mixdown
      - DC removal
      - peak-normalize to 0.95
    """
    try:
        note_on = (bytes([0x90, int(midi_note) & 0x7F, int(velocity) & 0x7F]), 0.0)
        note_off = (bytes([0x80, int(midi_note) & 0x7F, int(velocity) & 0x7F]), max(0.05, duration * 0.8))
        y = controller.synth([note_on, note_off],
                             duration=duration,
                             sample_rate=sample_rate,
                             reset=True)
        y = np.asarray(y)
        if y.ndim == 2:
            if y.shape[0] in (1, 2) and y.shape[1] > 2:
                y = y.T
            y = y.mean(axis=1) if y.shape[1] > 1 else y.reshape(-1)
        elif y.ndim > 2:
            y = y.reshape(-1)
        y = y.astype(np.float32, copy=False)
        if y.size:
            y = y - float(np.mean(y))
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if not np.isfinite(peak) or peak < 1e-6:
            return y, sample_rate
        y = 0.95 * (y / peak)
        return y, sample_rate
    except Exception as e:
        print(f"[render] render error: {e}")
        return None, None


def peak_normalize_audio(audio: np.ndarray, peak_target: float = 0.99) -> np.ndarray:
    """Mono peak-normalize to +/- peak_target; fallback when pyloudnorm is unavailable."""
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    elif x.ndim > 2:
        x = x.reshape(-1)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0:
        x = (peak_target / peak) * x
    return np.clip(x, -1.0, 1.0).astype(np.float32, copy=False)


def loudness_normalize_audio(audio: np.ndarray,
                             fs: int = 44100,
                             target_lufs: float = -23.0) -> np.ndarray:
    """Integrated-loudness normalize to ``target_lufs`` LUFS; fall back to peak normalize."""
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    elif x.ndim > 2:
        x = x.reshape(-1)
    if _PYLN_AVAILABLE:
        try:
            meter = pyln.Meter(fs)
            loud = meter.integrated_loudness(x)
            x = pyln.normalize.loudness(x, loud, target_lufs)
            return np.clip(x, -1.0, 1.0).astype(np.float32, copy=False)
        except Exception:
            pass
    return peak_normalize_audio(x)


def compute_timbral_models(y: Optional[np.ndarray], sr: Optional[int]) -> dict:
    """Compute the 7 AudioCommons timbral descriptors. Returns {} on silent input."""
    if y is None or sr is None:
        return {}
    if not y.size or not np.isfinite(y).all() or float(np.max(np.abs(y))) < 1e-6:
        return {}
    if not _TIMBRAL_AVAILABLE:
        raise RuntimeError("timbral_models is not installed; run `pip install timbral_models`.")

    def _safe(func):
        try:
            v = float(func(y, fs=sr))
            return v if np.isfinite(v) else None
        except Exception:
            return None

    return {
        "brightness": _safe(timbral_brightness),
        "depth": _safe(timbral_depth),
        "hardness": _safe(timbral_hardness),
        "roughness": _safe(timbral_roughness),
        "warmth": _safe(timbral_warmth),
        "sharpness": _safe(timbral_sharpness),
        "boominess": _safe(timbral_booming),
    }


def is_audio_audible(y: Optional[np.ndarray], threshold: float = 1e-2) -> bool:
    """Mirror of the app's audibility check: peak >= threshold AND rms >= threshold/4."""
    if y is None:
        return False
    y = np.asarray(y)
    if y.ndim == 2:
        y = y[:, 0]
    if y.size == 0 or not np.isfinite(y).any():
        return False
    y = y[np.isfinite(y)]
    if y.size == 0:
        return False
    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y * y)))
    return (peak >= threshold) and (rms >= threshold * 0.25)


def render_and_describe(controller: Sylenth1Controller,
                        params: dict,
                        param_limits: Optional[dict] = None,
                        midi_note: int = 69,
                        velocity: int = 100,
                        duration: float = 0.5,
                        sr: int = 44100,
                        target_lufs: float = -23.0) -> Optional[dict]:
    """Apply ``params`` to the plugin, render at ``midi_note``, normalize, return 7 descriptors.

    Returns None when the render is silent (post-loudness-normalize) so callers can
    track an audibility-failure rate without crashing.
    """
    if param_limits is None:
        param_limits = load_param_spec()
    valid = clamp_and_validate_params(params, param_limits)
    controller.set_params(valid)
    y, _sr = render_snippet(controller, midi_note=midi_note, velocity=velocity,
                            duration=duration, sample_rate=sr)
    if y is None or _sr is None or not y.size:
        return None
    y = loudness_normalize_audio(y, fs=_sr, target_lufs=target_lufs)
    if not is_audio_audible(y):
        return None
    models = compute_timbral_models(y, _sr)
    if not models or all(v is None for v in models.values()):
        return None
    return models
