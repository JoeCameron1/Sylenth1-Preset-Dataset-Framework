
import sys
import json
import requests
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton,
    QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView, QHBoxLayout,
    QListWidget, QListWidgetItem, QMessageBox, QAbstractItemView, QSlider, QFrame,
    QDialog, QDialogButtonBox, QFormLayout, QCheckBox, QComboBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize
from PyQt5.QtWidgets import QInputDialog
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor

from pedalboard import Plugin, Pedalboard, load_plugin

import numpy as np
import sounddevice as sd
import threading
import re
import os
import random

# Optional loudness normalization (recommended for stable timbral features)
try:
    import pyloudnorm as pyln
    _PYLND_AVAILABLE = True
except Exception:
    pyln = None
    _PYLND_AVAILABLE = False

# Optional factory sampler (Gaussian/Empirical/KDE + categorical temperature)
try:
    import factory_sampler as _FS
    FACTORY_SAMPLER_AVAILABLE = True
except Exception:
    _FS = None
    FACTORY_SAMPLER_AVAILABLE = False
    
# --- Factory Randomizer integration ---
try:
    from factory_randomizer import FactoryRandomizer
    FACTORY_RANDOMIZER_AVAILABLE = True
except Exception:
    FactoryRandomizer = None
    FACTORY_RANDOMIZER_AVAILABLE = False

from queue import Queue

from datetime import datetime, timezone

# --- Compatibility shims so timbral_models works with current NumPy/Librosa ---
# Provide numpy.lib.pad if missing (older code paths expect it)
import numpy as _np
if not hasattr(_np, "lib") or not hasattr(_np.lib, "pad"):
    if not hasattr(_np, "lib"):
        class _Lib: ...
        _np.lib = _Lib()
    _np.lib.pad = _np.pad

# Wrap librosa onset APIs to accept older positional call patterns used by timbral_models
try:
    import librosa as _librosa
    from functools import wraps as _wraps

    def _numeric(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    if hasattr(_librosa, "onset"):
        # onset_detect shim
        if hasattr(_librosa.onset, "onset_detect"):
            _orig_detect = _librosa.onset.onset_detect

            @_wraps(_orig_detect)
            def _detect_compat(*args, **kwargs):
                # Accept old-style onset_detect(y, sr, ...)
                if args:
                    y = args[0]
                    sr = args[1] if len(args) > 1 else kwargs.get("sr", 22050)
                    kwargs.setdefault("y", y)
                    kwargs.setdefault("sr", sr if sr is not None else 22050)
                    return _orig_detect(**kwargs)
                return _orig_detect(**kwargs)

            _librosa.onset.onset_detect = _detect_compat

        # onset_strength shim
        if hasattr(_librosa.onset, "onset_strength"):
            _orig_strength = _librosa.onset.onset_strength

            @_wraps(_orig_strength)
            def _strength_compat(*args, **kwargs):
                # Accept old-style onset_strength(y, sr, hop_length, ...)
                if args:
                    y = args[0]
                    kwargs.setdefault("y", y)
                    if len(args) >= 2 and "sr" not in kwargs and (_numeric(args[1]) or args[1] is None):
                        kwargs["sr"] = 22050 if args[1] is None else args[1]
                    if len(args) >= 3 and "hop_length" not in kwargs and _numeric(args[2]):
                        kwargs["hop_length"] = int(args[2])
                    return _orig_strength(**kwargs)
                return _orig_strength(**kwargs)

            _librosa.onset.onset_strength = _strength_compat
except Exception:
    # If librosa is not installed yet, timbral_models import will raise; we'll handle that later
    pass

# Optional AudioCommons timbral models
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
    TIMBRAL_AVAILABLE = True
except Exception:
    TIMBRAL_AVAILABLE = False

SYLENTH1_PATH = "/Library/Audio/Plug-Ins/VST3/Sylenth1.vst3"  # update if needed
OLLAMA_URL = "http://localhost:11434/api/generate"  # default Ollama API

# Load the Sylenth1 parameter limits/spec from the JSON file
with open("sylenth1_params.json", "r") as f:
    SYLENTH1_PARAM_LIMITS = json.load(f)

# Path to AudioCommons timbral dataset file
TIMBRAL_DATASET_FILE = "timbral_dataset_audiocommons.json"

# Mapping of effect params to switch param names
EFFECT_SWITCHES = {
    "distort": "sw_distonoff",
    "eq": "sw_eqonoff",
    "phaser": "sw_phaseronoff",
    "chorus": "sw_chorusonoff",
    "comp": "sw_componoff",
    "delay": "sw_delayonoff",
    "reverb": "sw_reverbonoff"
}

# Buggy Enum Parameter Workaround
BUGGY_ENUM_SHIFT_PARAMS = {
    'ymodmisc1a_dest1', 'ymodmisc1a_dest2', 'ymodmisc1b_dest1', 'ymodmisc1b_dest2', 'ymodmisc2a_dest1', 'ymodmisc2a_dest2', 'ymodmisc2b_dest1', 'ymodmisc2b_dest2',
    'ymodlfo1_dest1', 'ymodlfo1_dest2', 'ymodlfo2_dest1', 'ymodlfo2_dest2',
    'ymodenv1_dest1', 'ymodenv1_dest2', 'ymodenv2_dest1', 'ymodenv2_dest2',
}

def build_param_table(param_limits):
    lines = []
    for name, spec in param_limits.items():
        desc = spec.get("description", "")
        typ = spec.get("type", "float")
        if typ == "float":
            min_ = spec.get("min", "unknown")
            max_ = spec.get("max", "unknown")
            lines.append(f"{name}: float [{min_}–{max_}] - {desc}")
        elif typ == "enum":
            options = ", ".join(spec.get("options", []))
            lines.append(f"{name}: enum {{{options}}} - {desc}")
    return "\n".join(lines)

def clamp_and_validate_params(param_changes, param_limits):
    valid = {}
    for k, v in param_changes.items():
        if k not in param_limits:
            continue
        spec = param_limits[k]
        typ = (spec.get("type") or "float").lower()
        if typ == "float":
            try:
                v = float(v)
                v = max(spec["min"], min(v, spec["max"]))
                valid[k] = v
            except Exception:
                continue
        elif typ == "enum":
            valid_options = [str(opt) for opt in spec.get("options", [])]
            sval = str(v)
            if sval in valid_options:
                valid[k] = sval
            else:
                # Case-insensitive match
                match = [opt for opt in valid_options if sval.lower() == str(opt).lower()]
                if match:
                    valid[k] = match[0]
        elif typ == "bool":
            # Accept real bools, and common string/number forms
            if isinstance(v, bool):
                valid[k] = v
            else:
                sval = str(v).strip().lower()
                if sval in ("true", "1", "yes", "on"):
                    valid[k] = True
                elif sval in ("false", "0", "no", "off"):
                    valid[k] = False
                else:
                    # If options are provided as strings ["True","False"], map them
                    opts = [str(opt).lower() for opt in spec.get("options", [])]
                    if sval in opts:
                        valid[k] = (sval == "true")
    return valid

class Sylenth1Controller:
    def __init__(self, plugin_path):
        self.synth = load_plugin(plugin_path)
        self.param_names = list(self.synth.parameters.keys())
        self.param_values = self.get_all_params()
        print(f"Loaded Sylenth1 with {len(self.param_names)} parameters.")
        
    def get_all_params(self):
        params = {}
        for name in self.param_names:
            try:
                val = getattr(self.synth, name)
                try:
                    params[name] = float(val)
                except (TypeError, ValueError):
                    params[name] = str(val)
            except Exception as e:
                params[name] = None
                print(f"Error reading parameter '{name}': {e}")
        return params

    # NOTE: Sylenth1 workaround — for certain modulation destination enums, the plugin
    # applies the *previous* option when you select anything beyond the first 4.
    # To land on the intended choice, we pre-shift FORWARD by one for the params
    # listed in BUGGY_ENUM_SHIFT_PARAMS.
    def set_param(self, param, value):
        if param in self.synth.parameters:
            param_obj = self.synth.parameters[param]
            try:
                # Try to clamp numeric values if possible
                min_val = param_obj.min_value
                max_val = param_obj.max_value
                num_value = float(value)
                clamped = max(min_val, min(max_val, num_value))
                if num_value != clamped:
                    print(
                        f"[AI clamp warning] '{param}' value {num_value} was out of range "
                        f"[{min_val}, {max_val}]; clamped to {clamped}."
                    )
                setattr(self.synth, param, clamped)
            except (AttributeError, ValueError, TypeError):
                # Special handling for bool values: assign directly and return
                if isinstance(value, bool):
                    setattr(self.synth, param, value)
                    return
                # Handle enumerated (string) parameters
                # Try to match value to valid choices
                valid_values = getattr(param_obj, "valid_values", None)
                if valid_values:
                    # Case-insensitive match, allow common synth abbreviations
                    val = str(value).lower()
                    valid_lower = [str(v).lower() for v in valid_values]
                    # Try direct match
                    if val in valid_lower:
                        corrected_value = valid_values[valid_lower.index(val)]
                    else:
                        # Common synth abbreviations
                        synth_aliases = {
                            "lpf": "Lowpass",
                            "lowpass": "Lowpass",
                            "bpf": "Bandpass",
                            "bandpass": "Bandpass",
                            "hpf": "Highpass",
                            "highpass": "Highpass",
                            "bypass": "Bypass",
                            "Sawtooth": "Saw",
                            "SawTooth": "Saw"
                        }
                        corrected_value = synth_aliases.get(val)
                        if corrected_value in valid_values:
                            pass  # Found!
                        else:
                            # Try to smart-snap numeric-like enums (e.g., "2.001:1" -> nearest available "X.XXX:1")
                            def _parse_num(x):
                                try:
                                    s = str(x).strip()
                                    # Accept raw number, numeric string, or ratio-form like '2.001:1'
                                    if s.endswith(":1"):
                                        s = s[:-2]
                                    return float(s)
                                except Exception:
                                    return None

                            # Only attempt snapping if the enum choices themselves look numeric-ish
                            numerics = []
                            for vv in valid_values:
                                nv = _parse_num(vv)
                                if nv is None:
                                    numerics = None
                                    break
                                numerics.append((nv, vv))  # (numeric_value, original_enum_string)

                            candidate = _parse_num(val)
                            if numerics is not None and candidate is not None and len(numerics):
                                # choose nearest by absolute difference
                                nearest_num, nearest_str = min(numerics, key=lambda p: abs(p[0] - candidate))
                                corrected_value = nearest_str
                                # Proceed with snapped value instead of skipping
                            else:
                                print(
                                    f"[AI value error] '{param}' value '{value}' not in valid choices {valid_values}. Skipping."
                                )
                                return
                    # Apply Sylenth1 enum off-by-one workaround for certain params
                    # When selecting any option beyond the first 4 entries, the plugin applies the *previous* item;
                    # to land on the intended value, pre-shift FORWARD by one (idx >= 4 -> opts[idx+1]).
                    if param in BUGGY_ENUM_SHIFT_PARAMS:
                        try:
                            opts = [str(v) for v in valid_values]
                            if corrected_value in opts:
                                idx = opts.index(corrected_value)
                                if idx >= 4 and idx + 1 < len(opts):
                                    corrected_value = opts[idx + 1]
                        except Exception:
                            pass
                    setattr(self.synth, param, corrected_value)
                else:
                    # If no valid_values, just try to set as-is
                    setattr(self.synth, param, value)

    def set_params(self, param_dict):
        for param, value in param_dict.items():
            self.set_param(param, value)

    def refresh_params(self):
        self.param_values = self.get_all_params()


class Llama3Ollama:
    def __init__(self, model_name="llama3:8b"):
        self.model_name = model_name
        
    # --- Robust JSON helpers ---
    @staticmethod
    def _robust_extract_json_objects(text: str):
        """Return a list of top-level JSON object substrings by brace balancing."""
        out, stack, start = [], [], None
        for i, ch in enumerate(text):
            if ch == '{':
                if not stack:
                    start = i
                stack.append('{')
            elif ch == '}':
                if stack:
                    stack.pop()
                    if not stack and start is not None:
                        out.append(text[start:i+1])
                        start = None
        return out

    @staticmethod
    def _robust_json_fixups(s: str) -> str:
        """Fix common LLM JSON issues: unquoted fractions, Python bools, trailing commas, NaN/Inf."""
        import re
        # Wrap bare fractions like 1/4, 2/1 or 1/4T after a colon
        s = re.sub(r'(:\\s*)(\\d+\\/\\d+T?)', r'\\1"\\2"', s)
        # Python True/False -> JSON true/false when unquoted
        s = re.sub(r'(:\\s*)(True|False)(\\s*[,}\\]])',
                   lambda m: m.group(1) + m.group(2).lower() + m.group(3), s)
        # Remove trailing commas before object/array close
        s = re.sub(r',\\s*([}\\]])', r'\\1', s)
        # Replace NaN/Infinity with null
        s = re.sub(r'(:\\s*)(NaN|Infinity|-Infinity)(\\s*[,}\\]])', r'\\1null\\3', s)
        return s

class RealtimeSynthEngine:
    """Low-latency, real-time playback engine using a callback stream.
    Collects MIDI events and renders small audio blocks from the Sylenth plugin
    in the stream callback, avoiding any queued/latent offline renders.
    """
    def __init__(self, sylenth_controller, sample_rate=44100, blocksize=2048, channels=2):
        self.sylenth = sylenth_controller
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.channels = channels
        self.stream = None
        self.events = Queue()  # tuples of (status_byte, note, velocity)
        self.muted = False  # when True, callback outputs silence

    def start(self):
        if self.stream is not None:
            return
        def callback(outdata, frames, time, status):
            # Fast path: output silence while muted to avoid leaks/xruns during offline renders
            if getattr(self, "muted", False):
                try:
                    while True:
                        self.events.get_nowait()
                except Exception:
                    pass
                outdata[:] = np.zeros((frames, self.channels), dtype=np.float32)
                return
            # Drain all pending events for this block
            midi_events = []
            try:
                while True:
                    status_byte, note, vel = self.events.get_nowait()
                    midi_events.append((bytes([status_byte, note, vel]), 0.0))
            except Exception:
                pass
            # Render exactly one block worth of audio
            try:
                audio = self.sylenth.synth(
                    midi_events,
                    duration=frames / float(self.sample_rate),
                    sample_rate=self.sample_rate,
                    reset=False
                )
                audio = self._normalize_audio(audio, frames)
            except Exception:
                # On error, output silence for this block
                audio = np.zeros((frames, self.channels), dtype=np.float32)
            outdata[:] = audio
        try:
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.blocksize,
                dtype='float32',
                callback=callback
            )
            self.stream.start()
        except Exception as e:
            print(f"Error starting realtime audio stream: {e}")
            self.stream = None

    def stop(self):
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass
        finally:
            self.stream = None

    def note_on(self, note, velocity=100):
        self.events.put((0x90, int(note), int(velocity)))

    def note_off(self, note, velocity=100):
        self.events.put((0x80, int(note), int(velocity)))
        
    def set_muted(self, muted: bool):
        self.muted = bool(muted)

    def _normalize_audio(self, output, frames_expected):
        output = np.asarray(output)
        # Convert to (frames, channels)
        if output.ndim == 1:
            output = np.column_stack([output, output])
        elif output.ndim == 2:
            frames, chans = output.shape
            # If channels-first (1,N) or (2,N) style, transpose
            if frames in (1, 2) and chans > 2:
                output = output.T
                frames, chans = output.shape
            if chans == 1:
                output = np.repeat(output, 2, axis=1)
            elif chans > 2:
                output = output[:, :2]
        else:
            output = output.reshape(-1)
            output = np.column_stack([output, output])
        # Ensure correct frame count (pad or trim)
        if output.shape[0] < frames_expected:
            pad = np.zeros((frames_expected - output.shape[0], output.shape[1]), dtype=output.dtype)
            output = np.vstack([output, pad])
        elif output.shape[0] > frames_expected:
            output = output[:frames_expected, :]
        return output.astype('float32', copy=False)

class PianoKeyboard(QFrame):
    noteOn = pyqtSignal(int)   # MIDI note number
    noteOff = pyqtSignal(int)

    def __init__(self, base_note=60, num_keys=25, parent=None):
        super().__init__(parent)
        self.base_note = base_note  # C4 default
        self.num_keys = num_keys    # 25 keys ≈ 2 octaves
        self.setMinimumHeight(120)
        self.setMouseTracking(True)
        self._pressed_note = None
        self._white_key_indices = []
        self._black_key_indices = []
        # Precompute which semitone indices are white/black (relative to C)
        self._black_semitones = {1, 3, 6, 8, 10}  # C# D# F# G# A#
    
    def sizeHint(self):
        return QSize(700, 120)

    def set_base_note(self, base_note):
        self.base_note = base_note
        self.update()

    def _is_black(self, midi_note: int) -> bool:
        semitone = midi_note % 12
        return semitone in {1, 3, 6, 8, 10}

    def _note_name(self, midi_note: int) -> str:
        semitone = midi_note % 12
        white_names = {
            0: 'C', 2: 'D', 4: 'E', 5: 'F', 7: 'G', 9: 'A', 11: 'B'
        }
        black_names = {
            1: 'C#', 3: 'Eb', 6: 'F#', 8: 'Ab', 10: 'Bb'
        }
        return white_names.get(semitone) or black_names.get(semitone, '')

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        w = self.width()
        h = self.height()

        # Build ordered list of notes to display
        notes = [self.base_note + i for i in range(self.num_keys)]
        whites = [n for n in notes if not self._is_black(n)]
        blacks = [n for n in notes if self._is_black(n)]

        # Compute white key geometry
        num_white = max(1, len(whites))
        white_w = max(1, int(w / num_white))
        self._white_key_indices = []
        self._black_key_indices = []

        # Map white note -> sequential white index
        white_index_by_note = {}
        xi = 0
        for n in notes:
            if not self._is_black(n):
                white_index_by_note[n] = xi
                xi += 1

        white_left = {i: i * white_w for i in range(num_white)}

        # --- Draw white keys ---
        for n in whites:
            iwhite = white_index_by_note[n]
            x = white_left[iwhite]
            rect = (x, 0, white_w, h)
            self._white_key_indices.append((rect, n))
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(QPen(QColor(0, 0, 0)))
            painter.drawRect(*rect)
            if n == self._pressed_note:
                painter.fillRect(x+2, 2, white_w-4, h-4, QBrush(QColor(200, 220, 255)))
            # Label centered near bottom of white key
            name = self._note_name(n)
            if name:
                painter.setPen(QPen(QColor(0, 0, 0)))
                painter.drawText(x, h - 6, white_w, 14, Qt.AlignHCenter | Qt.AlignBottom, name)

        # --- Draw black keys on top ---
        black_w = int(white_w * 0.6)
        black_h = int(h * 0.6)

        # For positioning, each black belongs between two whites (after C, D, F, G, A)
        for n in blacks:
            # find immediate preceding white note present in our range
            base = None
            search = n - 1
            while search >= self.base_note:
                if not self._is_black(search):
                    base = search
                    break
                search -= 1
            if base is None or base not in white_index_by_note:
                continue
            iwhite = white_index_by_note[base]
            x_left = white_left[iwhite]
            next_left = white_left.get(iwhite + 1, x_left + white_w)
            center = (x_left + next_left + white_w) / 2.0
            bx = int(center - black_w / 2)
            rect = (bx, 0, black_w, black_h)
            self._black_key_indices.append((rect, n))
            painter.setBrush(QBrush(QColor(0, 0, 0)))
            painter.setPen(QPen(QColor(0, 0, 0)))
            painter.drawRect(*rect)
            if n == self._pressed_note:
                painter.fillRect(bx+2, 2, black_w-4, black_h-4, QBrush(QColor(80, 120, 180)))
            # White text label for contrast on black keys
            name = self._note_name(n)
            if name:
                painter.setPen(QPen(QColor(255, 255, 255)))
                painter.drawText(bx, black_h - 6, black_w, 14, Qt.AlignHCenter | Qt.AlignBottom, name)

        painter.end()

    def _note_at_pos(self, x, y):
        # Black keys first (they sit on top)
        for (rx, ry, rw, rh), note in self._black_key_indices:
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                return note
        # Then white keys
        for (rx, ry, rw, rh), note in self._white_key_indices:
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                return note
        return None

    def mousePressEvent(self, event):
        note = self._note_at_pos(event.x(), event.y())
        if note is not None:
            self._pressed_note = note
            self.noteOn.emit(note)
            self.update()

    def mouseMoveEvent(self, event):
        if self._pressed_note is None:
            return
        note = self._note_at_pos(event.x(), event.y())
        if note is None or note == self._pressed_note:
            return
        # slide to new note: send off for old, on for new
        old = self._pressed_note
        self._pressed_note = note
        self.noteOff.emit(old)
        self.noteOn.emit(note)
        self.update()

    def mouseReleaseEvent(self, event):
        if self._pressed_note is not None:
            note = self._pressed_note
            self._pressed_note = None
            self.noteOff.emit(note)
            self.update()


# Optional parameter-level randomizer (factory stats)
try:
    from param_randomizer import FactoryParamRandomizer
    _PARAM_RANDOMIZER_AVAILABLE = True
except Exception:
    FactoryParamRandomizer = None
    _PARAM_RANDOMIZER_AVAILABLE = False

class TimbralDatasetDialog(QDialog):
    def __init__(self, parent=None, default_type: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Save to Timbral Dataset")
        form = QFormLayout(self)

        # Sound type as a dropdown with blank default
        self.type_combo = QComboBox()
        # First entry blank to allow optional selection
        self.type_combo.addItem("")
        for opt in ["Lead", "Bass", "Pad", "Pluck", "Keys", "Arp", "FX", "Poly", "Mono"]:
            self.type_combo.addItem(opt)
        if default_type and default_type in [self.type_combo.itemText(i) for i in range(self.type_combo.count())]:
            self.type_combo.setCurrentText(default_type)
        else:
            self.type_combo.setCurrentIndex(0)  # blank

        self.tags_edit = QLineEdit()
        self.desc_edit = QTextEdit()
        self.desc_edit.setFixedHeight(80)
        form.addRow("Sound Type:", self.type_combo)
        form.addRow("Extra Tags (comma-separated):", self.tags_edit)
        form.addRow("Description:", self.desc_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def get_values(self):
        sound_type = self.type_combo.currentText().strip()
        tags = [t.strip() for t in self.tags_edit.text().split(',') if t.strip()]
        desc = self.desc_edit.toPlainText().strip()
        return sound_type, tags, desc

# --- Randomize Settings Dialog ---
class RandomizeSettingsDialog(QDialog):
    def __init__(self, parent=None, current_mode="uniform", current_temp=1.0, current_seed="", use_factory=True, factory_available=True):
        super().__init__(parent)
        self.setWindowTitle("Randomize Settings")
        form = QFormLayout(self)

        # Numeric mode
        self.mode_combo = QComboBox()
        for m in ["uniform", "gaussian", "empirical", "kde"]:
            self.mode_combo.addItem(m)
        idx = max(0, self.mode_combo.findText(str(current_mode)))
        self.mode_combo.setCurrentIndex(idx)

        # Categorical temperature
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.1, 5.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setDecimals(2)
        self.temp_spin.setValue(float(current_temp))

        # Seed (optional)
        self.seed_edit = QLineEdit(str(current_seed) if current_seed is not None else "")
        self.seed_edit.setPlaceholderText("Leave blank for random")
        
        # Sparsity and Repair toggles
        self.sparsity_chk = QCheckBox("Use sparsity (learn active controls)")
        self.sparsity_chk.setChecked(True)
        self.repair_chk = QCheckBox("Repair for audibility/playability")
        self.repair_chk.setChecked(True)

        # Use factory sampler checkbox
        self.use_factory_chk = QCheckBox("Use factory sampler when available")
        self.use_factory_chk.setChecked(bool(use_factory and factory_available))
        self.use_factory_chk.setEnabled(factory_available)

        form.addRow("Numeric mode:", self.mode_combo)
        form.addRow("Categorical temperature:", self.temp_spin)
        form.addRow("Seed:", self.seed_edit)
        form.addRow(self.use_factory_chk)
        form.addRow(self.sparsity_chk)
        form.addRow(self.repair_chk)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def get_values(self):
        mode = self.mode_combo.currentText()
        temp = float(self.temp_spin.value())
        seed_text = self.seed_edit.text().strip()
        seed = seed_text if seed_text != "" else None
        use_factory = bool(self.use_factory_chk.isChecked())
        sparsity = bool(self.sparsity_chk.isChecked())
        repair = bool(self.repair_chk.isChecked())
        return mode, temp, seed, use_factory, sparsity, repair

class MainWindow(QWidget):
    def _refresh_param_table(self, highlight_keys=None, highlight_color=Qt.white):
        """Refresh the parameter table from current controller values.
        Optionally highlight a subset of keys.
        """
        try:
            self.sylenth.refresh_params()
        except Exception:
            pass
        hi = set(highlight_keys) if highlight_keys else set()
        for i, name in enumerate(self.sylenth.param_names):
            value = self.sylenth.param_values.get(name)
            self.param_table.setItem(i, 1, QTableWidgetItem(str(value)))
            if name in hi:
                self.param_table.item(i, 0).setBackground(highlight_color)
                self.param_table.item(i, 1).setBackground(highlight_color)
            else:
                self.param_table.item(i, 0).setBackground(Qt.white)
                self.param_table.item(i, 1).setBackground(Qt.white)
    def _ac_descriptor_text_from_spins(self) -> str:
        """Return a canonical descriptor string like 'Brightness=50, Warmth=20, ...'
        using the current values of the 7 AudioCommons spinboxes. If spinboxes
        are not present, returns an empty string.
        """
        if not hasattr(self, "ac_spins") or not self.ac_spins:
            return ""
        order = ["brightness","warmth","hardness","roughness","depth","sharpness","boominess"]
        parts = []
        for k in order:
            sp = self.ac_spins.get(k)
            if sp is None:
                continue
            parts.append(f"{k.title()}={int(sp.value())}")
        return ", ".join(parts)
    def _autoflip_effect_switches(self, params: dict):
        """If effect parameters suggest audible use, ensure their sw_* switch is True.
        This only mutates the provided params dict (before applying to the synth)."""
        def any_gt(keys, thresh):
            for k in keys:
                val = params.get(k)
                try:
                    if float(val) > thresh:
                        return True
                except Exception:
                    continue
            return False

        # Reverb
        if any_gt(["reverb_dry_wet", "reverb_size", "reverb_width"], 5.0):
            if "sw_reverbonoff" in SYLENTH1_PARAM_LIMITS:
                params["sw_reverbonoff"] = True
        # Chorus
        if any_gt(["chorus_dry_wet", "chorus_depth", "chorus_width"], 5.0):
            if "sw_chorusonoff" in SYLENTH1_PARAM_LIMITS:
                params["sw_chorusonoff"] = True
        # Delay
        if any_gt(["delay_dry_wet", "delay_feedback", "delay_spread", "delay_width"], 5.0):
            if "sw_delayonoff" in SYLENTH1_PARAM_LIMITS:
                params["sw_delayonoff"] = True
        # Distortion
        if any_gt(["distort_drywet", "distort_amount"], 5.0):
            if "sw_distonoff" in SYLENTH1_PARAM_LIMITS:
                params["sw_distonoff"] = True
        # Compressor (enable if threshold clearly below 0 dB or ratio not 1:1)
        ratio = params.get("comp_ratio")
        thr = params.get("comp_threshold_db")
        try:
            thr_ok = (float(thr) < -1.0)
        except Exception:
            thr_ok = False
        ratio_ok = isinstance(ratio, str) and ratio.strip() != "1.000:1"
        if (thr_ok or ratio_ok) and "sw_componoff" in SYLENTH1_PARAM_LIMITS:
            params["sw_componoff"] = True
        # EQ
        if any_gt(["eq_bass_db", "eq_treble_db"], 0.5):
            if "sw_eqonoff" in SYLENTH1_PARAM_LIMITS:
                params["sw_eqonoff"] = True
        # Phaser
        if any_gt(["phaser_dry_wet", "phaser_width", "phaser_lfo_gain"], 5.0):
            if "sw_phaseronoff" in SYLENTH1_PARAM_LIMITS:
                params["sw_phaseronoff"] = True
    
    
    # LOUDNESS NORMALISATION
    def _peak_normalize_audio(self, audio: np.ndarray, peak_target: float = 0.99) -> np.ndarray:
        """Simple peak normalization to +/- peak_target if pyloudnorm is not available."""
        try:
            x = np.asarray(audio, dtype=np.float32)
            if x.ndim == 2:
                x = np.mean(x, axis=1) if x.shape[1] > 1 else x[:, 0]
            elif x.ndim > 2:
                x = x.reshape(-1)
            peak = float(np.max(np.abs(x))) if x.size else 0.0
            if peak > 0:
                x = (peak_target / peak) * x
            return np.clip(x, -1.0, 1.0).astype(np.float32, copy=False)
        except Exception:
            return np.asarray(audio, dtype=np.float32).reshape(-1)

    def _loudness_normalize_audio(self, audio: np.ndarray, fs: int = 44100, target_lufs: float = -23.0) -> np.ndarray:
        """
        Prefer integrated-loudness normalisation to a target LUFS; fall back to peak normalisation.
        Returns a MONO float32 array suitable for timbral_models.
        """
        x = np.asarray(audio, dtype=np.float32)
        if x.ndim == 2:
            x = np.mean(x, axis=1) if x.shape[1] > 1 else x[:, 0]
        elif x.ndim > 2:
            x = x.reshape(-1)

        if _PYLND_AVAILABLE:
            try:
                meter = pyln.Meter(fs)
                loud = meter.integrated_loudness(x)
                x = pyln.normalize.loudness(x, loud, target_lufs)
                return np.clip(x, -1.0, 1.0).astype(np.float32, copy=False)
            except Exception:
                pass
        return self._peak_normalize_audio(x)
    
    def _nudge_to_mid_if_extreme(self, key: str, params: dict, low_frac: float = 0.05, high_frac: float = 0.95) -> None:
        """If a float param is too close to min or max, push it to the midpoint."""
        spec = SYLENTH1_PARAM_LIMITS.get(key, {})
        if (spec.get("type") or "").lower() != "float":
            return
        if key not in params:
            return
        try:
            val = float(params[key])
            pmin = float(spec.get("min", 0.0))
            pmax = float(spec.get("max", 1.0))
            if not np.isfinite(val) or pmax <= pmin:
                return
            lo = pmin + low_frac * (pmax - pmin)
            hi = pmin + high_frac * (pmax - pmin)
            if val <= lo or val >= hi:
                params[key] = float(pmin + 0.5 * (pmax - pmin))
        except Exception:
            return

    def _last_chance_make_audible(self, params: dict) -> dict:
        """
        After 3 silent retries, gently bias a few controls toward audibility:
        - Ensure filter inputs aren't 'None'
        - Force filter cutoffs into a midband (40%–60% of their range)
        - Ensure mix/volume aren't below midpoint
        Returns a modified dict (same keys).
        """
        out = dict(params)
        #print(out)
        changed = False

        # 1) Filter inputs: avoid 'None' if options exist
        for k in ("filter_a_input", "filter_b_input"):
            spec = SYLENTH1_PARAM_LIMITS.get(k, {})
            if (spec.get("type") or "").lower() == "enum":
                cur = out.get(k)
                if isinstance(cur, str) and cur.strip().lower() in {"none", "-"}:
                    opts = [str(o) for o in (spec.get("options") or [])]
                    cand = next((o for o in opts if o not in {"None", "-"}), None)
                    if cand is not None:
                        out[k] = cand
                        changed = True

        # 2) Cutoffs: force to random 40%–60% of allowed range
        rng = getattr(self, "_rng", random)
        for k in ("filter_a_cutoff", "filter_b_cutoff", "filterctl_cutoff"):
            spec = SYLENTH1_PARAM_LIMITS.get(k, {})
            if (spec.get("type") or "").lower() == "float":
                try:
                    pmin = float(spec.get("min", 0.0))
                    pmax = float(spec.get("max", 1.0))
                    if pmax > pmin:
                        target = pmin + (0.4 + 0.2 * rng.random()) * (pmax - pmin)  # 40%..60%
                        out[k] = float(target)
                        changed = True
                except Exception:
                    pass

        # 3) Mix/volume: keep at least midpoint
        for k in ("mix_a", "mix_b"):
            spec = SYLENTH1_PARAM_LIMITS.get(k, {})
            if (spec.get("type") or "").lower() == "float":
                try:
                    v = float(out.get(k, 0.0))
                    pmin = float(spec.get("min", 0.0))
                    pmax = float(spec.get("max", 1.0))
                    mid = pmin + 0.5 * (pmax - pmin)
                    if not np.isfinite(v) or v < mid:
                        out[k] = float(mid)
                        changed = True
                except Exception:
                    pass

        return out if changed else out
    
    #-------
    
    def __init__(self, sylenth, llama3):
        super().__init__()
        self.sylenth = sylenth
        self.llama3 = llama3
        self.current_preset_name = ""
        # --- Preset management ---
        self.preset_file = "user_presets.json"
        self.presets = {}  # name -> param dict
        self.load_all_presets()

        # --- Randomization settings ---
        self.rand_mode = "uniform"          # "uniform", "gaussian", "empirical", "kde"
        self.rand_temp = 1.0               # categorical temperature
        self.rand_seed = None              # None => unseeded
        # Minimal factory-sampler integration (lazy init; KDE + sparsity + repairs)
        self.fs_json_path = "Sylenth1_Full_Factory_Presets_AC_A4.json"
        # New factory-based randomisation (pick a factory preset, then mutate a few active controls)
        self._factory_bank = self._load_factory_presets(self.fs_json_path)
        self._fs_sampler = None  # created on first use

        # --- Param randomizer (factory stats, if available) ---
        self.param_randomizer = None
        if _PARAM_RANDOMIZER_AVAILABLE:
            try:
                self.param_randomizer = FactoryParamRandomizer(
                    limits_json="sylenth1_params.json",
                    factory_json="Sylenth1_Full_Factory_Presets_AC_A4.json",
                    variance_scale=1.0,
                )
            except Exception as _e:
                self.param_randomizer = None
                try:
                    self.feedback_box.append(f"Param Randomizer init failed: {_e}")
                except Exception:
                    pass

        # Parameters we intentionally do NOT touch during randomisation
        self._excluded_randomize_keys = {"mix_a", "mix_b"}

        # Per-instance cryptographic RNG (unseeded) for real randomness
        import random as _py_random
        self._rng = _py_random.SystemRandom()

        # Build the UI now (avoid blank window)
        self.init_ui()

        # Track external changes from Sylenth1 GUI
        self.last_param_values = self.sylenth.get_all_params()
        self.last_program_value = self.sylenth.get_all_params().get("program")  # Detect when a preset is loaded
        self.sync_timer = QTimer()
        self.sync_timer.timeout.connect(self.check_external_changes)
        self.sync_timer.start(1000)  # Check every 1s

        # Real-time audio engine
        self.rt_engine = RealtimeSynthEngine(self.sylenth, sample_rate=44100, blocksize=2048, channels=2)
        self.rt_engine.start()
    def _load_factory_presets(self, path):
        """Load factory preset bank from JSON path. Returns a list of {'params': {...}, ...} entries."""
        try:
            if not os.path.isfile(path):
                return []
            with open(path, "r") as f:
                data = json.load(f)
            # Accept either list of entries (each has 'params') or dict name->entry
            presets = []
            if isinstance(data, list):
                for e in data:
                    if isinstance(e, dict) and isinstance(e.get("params"), dict):
                        presets.append(e)
            elif isinstance(data, dict):
                for name_key, e in data.items():
                    if isinstance(e, dict) and isinstance(e.get("params"), dict):
                        e2 = dict(e)
                        # Preserve a friendly name if not already present
                        e2.setdefault("name", str(name_key))
                        presets.append(e2)
            return presets
        except Exception as e:
            try:
                self.feedback_box.append(f"Randomizer: failed to load factory presets: {e}")
            except Exception:
                pass
            return []

    def _neutral_default_value(self, key, spec):
        """Heuristic neutral default per parameter spec."""
        ptype = (spec.get("type") or "").lower()
        if ptype == "bool" or str(key).startswith("sw_"):
            return False
        if ptype == "enum":
            opts = spec.get("options", []) or []
            # Prefer explicit neutral options if present
            for neutral in ("-", "None", ""):
                if neutral in opts:
                    return neutral
            return opts[0] if opts else ""
        # float (or unknown): assume 0.0 as neutral
        return 0.0

    def _baseline_default_dict(self):
        """Build a neutral baseline dict across the JSON spec."""
        base = {}
        for k, spec in SYLENTH1_PARAM_LIMITS.items():
            base[k] = self._neutral_default_value(k, spec)
        return base

    def _non_default_keys(self, params, baseline):
        """Return keys whose params differ from baseline (i.e., 'active' controls)."""
        active = []
        for k, spec in SYLENTH1_PARAM_LIMITS.items():
            if k not in params:
                continue
            # Never consider excluded keys as candidates for mutation
            if k in getattr(self, "_excluded_randomize_keys", set()):
                continue
            v = params[k]
            b = baseline.get(k)
            ptype = (spec.get("type") or "").lower()
            try:
                if ptype == "float":
                    # consider different if not almost-equal (scale 0..10 often used)
                    if not np.isfinite(float(v)) or not np.isfinite(float(b)):
                        continue
                    if abs(float(v) - float(b)) > 1e-3:
                        active.append(k)
                else:
                    if v != b:
                        active.append(k)
            except Exception:
                if v != b:
                    active.append(k)
        return active

    def _load_defaults_json(self, path: str) -> dict:
        """Load a JSON of default (neutral) parameter values. Returns {} on failure."""
        try:
            if not path or not os.path.isfile(path):
                return {}
            with open(path, "r") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _active_keys_against_defaults(self, params: dict, defaults: dict) -> list:
        """Return keys whose values in `params` differ from `defaults`.
        Falls back to JSON spec neutral defaults when a key is absent in `defaults`.
        """
        active = []
        for k, spec in SYLENTH1_PARAM_LIMITS.items():
            if k not in params:
                continue
            if k in getattr(self, "_excluded_randomize_keys", set()):
                continue
            v = params.get(k)
            if k in defaults:
                d = defaults.get(k)
            else:
                d = self._neutral_default_value(k, spec)
            ptype = (spec.get("type") or "").lower()
            try:
                if ptype == "float":
                    if np.isfinite(float(v)) and np.isfinite(float(d)):
                        if abs(float(v) - float(d)) > 1e-3:
                            active.append(k)
                else:
                    if v != d:
                        active.append(k)
            except Exception:
                if v != d:
                    active.append(k)
        return active

    def _choose_different_value(self, key, spec, current):
        """Sample a valid value different from 'current' according to the spec."""
        ptype = (spec.get("type") or "").lower()
        r = self._rng
        if ptype == "bool" or str(key).startswith("sw_"):
            return not bool(current)
        if ptype == "enum":
            opts = spec.get("options", []) or []
            usable = [o for o in opts if str(o).strip() != "-"] or opts
            usable = [o for o in usable if o != current]
            if not usable:
                return current
            return r.choice(usable)
        # float
        pmin = float(spec.get("min", 0.0))
        pmax = float(spec.get("max", 1.0))
        if pmax <= pmin:
            return current
        # try a few times to be "meaningfully different"
        for _ in range(8):
            val = r.uniform(pmin, pmax)
            if not np.isfinite(val):
                continue
            if abs(val - float(current)) >= 0.1 * (pmax - pmin):
                return float(val)
        return float(r.uniform(pmin, pmax))

    
    def _randomize_from_factory_active_using_module(self):
        """Implements randomized mutation of a factory preset's *active* parameters:
        1) choose random factory preset
        2) compute active params vs provided defaults.json (or spec-neutral baseline)
        3) draw k ~ Uniform{1..n}
        4) choose k active params uniformly
        5) for each, sample a new value via self.param_randomizer.generate_random_value(param)
        Returns (patch_dict, base_name, mutated_keys).
        """
        # Ensure factory bank is loaded
        if not getattr(self, "_factory_bank", None):
            self._factory_bank = self._load_factory_presets(self.fs_json_path)
        if not self._factory_bank:
            return {}, "", []

        # 1) choose a factory preset
        entry = self._rng.choice(self._factory_bank)
        base_params = dict(entry.get("params", {}))
        base_params = {k: base_params[k] for k in base_params.keys() if k in SYLENTH1_PARAM_LIMITS}

        # --- Load the chosen factory preset into Sylenth1 for consistency (like load_preset) ---
        try:
            self.sylenth.set_params(base_params)
            self.sylenth.refresh_params()
            # Update parameter table to reflect the loaded factory preset
            for i, pname in enumerate(self.sylenth.param_names):
                value = self.sylenth.param_values[pname]
                self.param_table.setItem(i, 1, QTableWidgetItem(str(value)))
                # Reset highlight to neutral; mutations below will recolor changed keys
                self.param_table.item(i, 0).setBackground(Qt.white)
                self.param_table.item(i, 1).setBackground(Qt.white)
            self.feedback_box.append(
                f"Loaded factory preset “{entry.get('name') or entry.get('program') or '(unnamed)'}” before randomization."
            )
        except Exception as e:
            self.feedback_box.append(f"Warning: failed to load factory preset cleanly: {e}")
        # Re-snapshot from the live synth to ensure we use exactly what the plugin accepted
        try:
            base_params_live = self._snapshot_current_params()
            if isinstance(base_params_live, dict) and base_params_live:
                base_params = {k: base_params_live[k] for k in base_params_live if k in SYLENTH1_PARAM_LIMITS}
        except Exception:
            pass

        # Exclusions (e.g., mix_a/mix_b, program, etc.)
        excl = set(getattr(self, "_excluded_randomize_keys", set()))
        for k in ("program", "preset_name", "mix_a", "mix_b"):
            excl.add(k)
        for k in tuple(excl):
            base_params.pop(k, None)

        if not base_params:
            return {}, "", []

        # 2) compute active vs defaults.json (if available) or spec-neutral baseline
        defaults_path = getattr(self, "defaults_json_path", None)
        defaults = self._load_defaults_json(defaults_path) if defaults_path else {}
        active = self._active_keys_against_defaults(base_params, defaults)
        if not active:
            # Degenerate case: treat all keys in base_params as candidates
            active = list(base_params.keys())
        # Sanity: remove exclusions again
        active = [k for k in active if k not in excl]
        if not active:
            return {}, entry.get("name") or entry.get("program") or "", []

        # 3) pick k in [1, n]
        n = len(active)
        k = self._rng.randint(1, n)

        # 4) choose k active params uniformly
        mutate_keys = self._rng.sample(active, k=k)

        # 5) sample new values via module randomizer (fallback to spec if unavailable)
        mutated = {}
        for kkey in mutate_keys:
            spec = SYLENTH1_PARAM_LIMITS.get(kkey, {})
            cur = base_params.get(kkey)
            new_val = None
            if getattr(self, "param_randomizer", None) is not None:
                try:
                    new_val = self.param_randomizer.generate_random_value(kkey)
                except Exception:
                    new_val = None
            if new_val is None:
                # fallback: different value using spec
                new_val = self._choose_different_value(kkey, spec, cur)
            # Avoid no-op (equal value); if equal, try one more draw from spec
            if new_val == cur:
                try:
                    alt = self._choose_different_value(kkey, spec, cur)
                    if alt != cur:
                        new_val = alt
                except Exception:
                    pass
            mutated[kkey] = new_val

        # Merge with base (only the k mutated keys change)
        out = dict(base_params)
        out.update(mutated)
        base_name = entry.get("name") or entry.get("program") or ""
        return out, base_name, mutate_keys
    def _factory_random_params(self) -> dict:
        """Use FactoryRandomizer (KDE + sparsity + repairs) to create a random audible preset,
        with extra app-side constraints to (a) keep categorical params sparse and
        (b) avoid overly long attacks which were uncommon in the factory set.
        Returns {} if FactoryRandomizer not available or on failure.
        """
        if not getattr(self, "factory_randomizer", None):
            return {}
        try:
            # Ask the randomizer for a relatively conservative sample
            patch = self.factory_randomizer.sample_preset(
                numeric_mode="kde",
                variance_scale=0.35,        # tighter variance to avoid tails (e.g., huge attacks)
                cat_temperature=1.2,        # closer to factory frequencies
                frac_active_numeric=0.10,   # fewer numeric params active on average
                num_active_categorical=6,   # keep categorical selection sparse
                repair=True,
            )
            patch = dict(patch)  # ensure plain dict

            # --- App-side post processing / guardrails ---

            # (1) Enforce categorical sparsity cap: drop extras beyond MAX_ENUMS, except "essential" ones
            MAX_ENUMS = 6
            essential_enums = {
                # Osc waveform/voices/mix and filter types are usually core to timbre
                "osc_a1_waveform","osc_a2_waveform","osc_b1_waveform","osc_b2_waveform",
                "osc_a1_voices","osc_a2_voices","osc_b1_voices","osc_b2_voices",
                "filter_a_type","filter_b_type"
            }
            enum_keys = [k for k,v in patch.items()
                         if k in SYLENTH1_PARAM_LIMITS and (SYLENTH1_PARAM_LIMITS[k].get("type","").lower()=="enum")]
            # Keep essentials first
            kept = [k for k in enum_keys if k in essential_enums]
            rest = [k for k in enum_keys if k not in essential_enums]
            # If still over budget, randomly drop from the rest
            rng = getattr(self, "_rng", random)
            if len(kept) + len(rest) > MAX_ENUMS:
                need_to_drop = len(kept) + len(rest) - MAX_ENUMS
                # don't drop below kept essentials
                drop_from = rest.copy()
                rng.shuffle(drop_from)
                for k in drop_from[:need_to_drop]:
                    patch.pop(k, None)

            # (2) Cap overly long ADSR attacks (factory sets tend to be short to moderate)
            for atk_key in ("ampenv_a_attack", "ampenv_b_attack"):
                if atk_key in patch and isinstance(patch[atk_key], (int, float)):
                    try:
                        val = float(patch[atk_key])
                        if not np.isfinite(val):
                            raise ValueError
                        # If attack > 4.0 (outlier for many factory patches), pull it down into a musical range
                        if val > 4.0:
                            # sample a smaller attack in [0.03, 3.5]
                            lo, hi = 0.03, 3.5
                            patch[atk_key] = float(rng.uniform(lo, hi))
                    except Exception:
                        # if it's not numeric, drop it
                        patch.pop(atk_key, None)

            # (3) Optional: guard against excessive "mix" imbalance rendering silence (keep some presence)
            for mix_key in ("mix_a","mix_b"):
                if mix_key in patch and isinstance(patch[mix_key], (int,float)):
                    try:
                        v = float(patch[mix_key])
                        if v < 0.2:
                            patch[mix_key] = 0.2
                    except Exception:
                        pass

            # Done
            return patch

        except Exception as e:
            try:
                self.feedback_box.append(f"FactoryRandomizer error: {e}")
            except Exception:
                pass
            return {}

    def init_ui(self):
        self.setWindowTitle("Sylenth1 Preset Custom Controller")
        layout = QVBoxLayout()

        # Random preset controls
        rnd_row = QHBoxLayout()
        self.random_preset_btn = QPushButton("Randomize Preset")
        self.random_preset_btn.setToolTip("Generate a random preset (respecting ranges/options) and apply it")
        self.random_preset_btn.clicked.connect(self.on_random_preset)
        rnd_row.addWidget(self.random_preset_btn)
        layout.addLayout(rnd_row)

        # Feedback display
        self.feedback_box = QTextEdit()
        self.feedback_box.setReadOnly(True)
        layout.addWidget(QLabel("Feedback:"))
        layout.addWidget(self.feedback_box)

        # Parameter Table
        layout.addWidget(QLabel("All Sylenth1 Parameters (highlighted if changed):"))
        self.param_table = QTableWidget()
        self.param_table.setRowCount(len(self.sylenth.param_names))
        self.param_table.setColumnCount(2)
        self.param_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.param_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for i, name in enumerate(self.sylenth.param_names):
            self.param_table.setItem(i, 0, QTableWidgetItem(name))
            self.param_table.setItem(i, 1, QTableWidgetItem(str(self.sylenth.param_values[name])))
        layout.addWidget(self.param_table)

        # --- Preset management UI ---
        layout.addWidget(QLabel("User Presets:"))
        preset_hbox = QHBoxLayout()
        self.preset_list = QListWidget()
        self.preset_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.refresh_preset_list()
        preset_hbox.addWidget(self.preset_list)
        preset_btns = QVBoxLayout()
        self.save_preset_btn = QPushButton("Save Preset")
        self.save_preset_btn.clicked.connect(self.on_save_preset)
        preset_btns.addWidget(self.save_preset_btn)
        self.load_preset_btn = QPushButton("Load Preset")
        self.load_preset_btn.clicked.connect(self.on_load_preset)
        preset_btns.addWidget(self.load_preset_btn)
        self.delete_preset_btn = QPushButton("Delete Preset")
        self.delete_preset_btn.clicked.connect(self.on_delete_preset)
        preset_btns.addWidget(self.delete_preset_btn)
        preset_hbox.addLayout(preset_btns)
        layout.addLayout(preset_hbox)

        # Show Sylenth1 GUI
        self.show_sylenth1_gui_button = QPushButton("Show Sylenth1 GUI")
        self.show_sylenth1_gui_button.clicked.connect(self.show_sylenth1_gui)
        layout.addWidget(self.show_sylenth1_gui_button)

        # Timbral dataset save
        self.btn_save_timbral = QPushButton("Save to Timbral Dataset…")
        self.btn_save_timbral.setToolTip("Compute AudioCommons timbral attributes for A4 (440 Hz) and save with current params")
        self.btn_save_timbral.clicked.connect(self.on_save_to_timbral_dataset)
        layout.addWidget(self.btn_save_timbral)

        # --- On-screen Piano Keyboard ---
        kb_panel = QHBoxLayout()
        self.oct_down_btn = QPushButton("Octave –")
        self.oct_up_btn = QPushButton("Octave +")
        self.velocity_slider = QSlider(Qt.Horizontal)
        self.velocity_slider.setRange(1, 127)
        self.velocity_slider.setValue(100)
        kb_panel.addWidget(QLabel("Velocity"))
        kb_panel.addWidget(self.velocity_slider)
        kb_panel.addWidget(self.oct_down_btn)
        kb_panel.addWidget(self.oct_up_btn)
        layout.addLayout(kb_panel)

        self.keyboard_base_note = 60  # C4
        self.keyboard = PianoKeyboard(base_note=self.keyboard_base_note, num_keys=25)
        self.keyboard.setFrameStyle(QFrame.Panel | QFrame.Raised)
        layout.addWidget(self.keyboard)

        # Connect keyboard controls
        self.oct_down_btn.clicked.connect(lambda: self._change_keyboard_octave(-1))
        self.oct_up_btn.clicked.connect(lambda: self._change_keyboard_octave(1))
        self.keyboard.noteOn.connect(self.on_keyboard_note_on)
        self.keyboard.noteOff.connect(self.on_keyboard_note_off)

        # Finalize layout
        self.setLayout(layout)
        self.resize(900, 900)

    def generate_random_params(self) -> dict:
        """
        Generate random values for all adjustable parameters using either the factory-driven
        param randomizer (if available), or fallback to uniform random sampling.
        """
        params = {}

        # If we have the factory-driven param randomizer, use it directly
        if self.param_randomizer is not None:
            try:
                # Get all defined parameter names from the limits JSON
                for param_name in SYLENTH1_PARAM_LIMITS.keys():
                    # Skip excluded parameters (e.g. program name)
                    if param_name.lower() in ("program", "preset_name"):
                        continue
                    # Get a random value for this parameter from the factory-based model
                    val = self.param_randomizer.generate_random_value(param_name)
                    params[param_name] = val
                self.feedback_box.append("[Randomizer] Generated full preset from factory distributions.")
                return params
            except Exception as e:
                self.feedback_box.append(f"[Randomizer error] Falling back to uniform random: {e}")

        # Legacy fallback — uniform sampling if factory randomizer unavailable
        r = self._rng
        for key, spec in SYLENTH1_PARAM_LIMITS.items():
            if key.lower() in ("program", "preset_name"):
                continue
            ptype = (spec.get("type") or "").lower()
            try:
                if ptype == "float":
                    pmin = float(spec.get("min", 0.0))
                    pmax = float(spec.get("max", 1.0))
                    params[key] = float(r.uniform(pmin, pmax))
                elif ptype == "enum":
                    opts = spec.get("options", [])
                    usable = [o for o in opts if str(o).strip() != "-"] or opts
                    if usable:
                        params[key] = r.choice(usable)
                elif ptype == "bool" or str(key).startswith("sw_"):
                    params[key] = bool(r.getrandbits(1))
            except Exception:
                continue

        self.feedback_box.append("[Randomizer] Fallback uniform randomization applied.")
        return params

    def on_random_preset(self):
        """Randomize by mutating k active parameters of a random factory preset (module-driven values)."""
        try:
            used_factory_flow = False
            rnd, base_name, mutate_keys = {}, "", []

            # Prefer the module-driven factory-active mutation if available
            if getattr(self, "param_randomizer", None) is not None and getattr(self, "_factory_bank", None) is not None or self.fs_json_path:
                rnd, base_name, mutate_keys = self._randomize_from_factory_active_using_module()
                used_factory_flow = bool(rnd)

            # Retry loop to ensure we end up with an audible patch
            MAX_RETRIES = 3
            attempt = 0
            while attempt < MAX_RETRIES:
                if not rnd:
                    # First, try a fresh factory-based mutation each attempt
                    used_factory_flow = False
                    try:
                        cand, cand_base, cand_keys = self._randomize_from_factory_active_using_module()
                    except Exception:
                        cand, cand_base, cand_keys = {}, "", []
                    if cand:
                        rnd = cand
                        base_name = cand_base
                        mutate_keys = cand_keys
                        used_factory_flow = True
                    else:
                        # Fallback to uniform for this attempt
                        rnd = self.generate_random_params()
                        base_name, mutate_keys = "", list(rnd.keys())

                # Check audibility with a silent render (no UI mutation)
                if self._is_patch_audible(rnd, threshold=1e-2):
                    break
                else:
                    self.feedback_box.append(f"[Randomizer] Silent patch detected (attempt {attempt + 1}); retrying…")
                    rnd = {}
                    attempt += 1

            if not rnd:
                # --- Last-chance fallback after 3 silent retries: nudge obviously-silent settings ---
                try:
                    base_for_fallback = self._snapshot_current_params()
                except Exception:
                    base_for_fallback = {}
                if isinstance(base_for_fallback, dict) and base_for_fallback:
                    fixed = self._last_chance_make_audible(base_for_fallback)
                    if fixed:
                        try:
                            # Try the fixed params silently first
                            if self._is_patch_audible(fixed, threshold=1e-2):
                                self.sylenth.set_params(fixed)
                                self.sylenth.refresh_params()
                                # Highlight changed keys in the table
                                try:
                                    after_params = self._snapshot_current_params()
                                except Exception:
                                    after_params = {}
                                changed_keys = [k for k in fixed.keys() if str(after_params.get(k)) == str(fixed.get(k))]
                                if not changed_keys:
                                    changed_keys = list(fixed.keys())
                                self._refresh_param_table(highlight_keys=changed_keys, highlight_color=Qt.yellow)
                                self.feedback_box.append("[Randomizer] Fallback adjustments made the patch audible; applied.")
                                return
                            else:
                                self.feedback_box.append("[Randomizer] Fallback adjustments still silent.")
                        except Exception:
                            # Even if application fails, continue to final message
                            pass
                # If we get here, nothing worked
                self.feedback_box.append("[Randomizer] All attempts resulted in silence; keeping last preset.")
                return

            try:
                safe = clamp_and_validate_params(rnd, SYLENTH1_PARAM_LIMITS)
            except Exception:
                safe = rnd

            # Diff-based highlight: snapshot BEFORE applying, then apply once, then snapshot AFTER
            try:
                before_params = self._snapshot_current_params()
            except Exception:
                before_params = {}

            self.sylenth.set_params(safe)
            self.sylenth.refresh_params()

            try:
                after_params = self._snapshot_current_params()
            except Exception:
                after_params = {}

            changed_keys = []
            for k in SYLENTH1_PARAM_LIMITS.keys():
                if (k in before_params) or (k in after_params):
                    if str(before_params.get(k)) != str(after_params.get(k)):
                        changed_keys.append(k)
            if not changed_keys:
                changed_keys = list(safe.keys())

            self._refresh_param_table(highlight_keys=changed_keys, highlight_color=Qt.yellow)

            # Reset slider highlight
            if getattr(self, "active_slider_widget", None):
                self.active_slider_widget.setStyleSheet("")
                self.active_slider_widget = None
            self.active_slider_label.setText("No slider engaged.")

            # Feedback
            try:
                changed_preview = []
                for k in changed_keys[:12]:
                    v = safe.get(k, rnd.get(k))
                    changed_preview.append(f"{k}={v}")
                more = "" if len(changed_keys) <= 12 else f" … (+{len(changed_keys) - 12} more)"
                if used_factory_flow:
                    base_disp = base_name or "(unnamed factory preset)"
                    self.feedback_box.append(
                        f"Randomize: based on factory preset “{base_disp}”. Altered {len(changed_keys)} parameter(s): "
                        + ", ".join(changed_preview) + more
                    )
                else:
                    self.feedback_box.append("Randomize: factory bank unavailable; used uniform randomization.")
            except Exception as e:
                # make sure feedback never disappears silently
                try:
                    self.feedback_box.append(f"[Randomizer] (feedback error) {e}")
                except Exception:
                    pass
        except Exception as e:
            self.feedback_box.append(f"Randomize failed: {e}")

    def _change_keyboard_octave(self, delta):
        # Change by octaves (12 semitones)
        self.keyboard_base_note = max(0, min(108, self.keyboard_base_note + delta * 12))
        self.keyboard.set_base_note(self.keyboard_base_note)
        self.feedback_box.append(f"Keyboard base note set to MIDI {self.keyboard_base_note}.")

    def on_keyboard_note_on(self, midi_note):
        velocity = int(self.velocity_slider.value())
        self.rt_engine.note_on(midi_note, velocity)

    def on_keyboard_note_off(self, midi_note):
        velocity = int(self.velocity_slider.value())
        self.rt_engine.note_off(midi_note, velocity)


    def closeEvent(self, event):
        try:
            if hasattr(self, 'rt_engine') and self.rt_engine:
                self.rt_engine.stop()
        except Exception:
            pass
        super().closeEvent(event)

    def _play_midi_note_async(self, midi_note, velocity=100, duration_seconds=0.6, sample_rate=44100):
        def worker():
            try:
                note_on = (bytes([0x90, midi_note, velocity]), 0.0)
                note_off = (bytes([0x80, midi_note, velocity]), max(0.05, duration_seconds * 0.8))
                output = self.sylenth.synth(
                    [note_on, note_off],
                    duration=duration_seconds,
                    sample_rate=sample_rate,
                    reset=False
                )
                # --- Normalize to shape (frames, channels) for sounddevice ---
                output = np.asarray(output)
                if output.ndim == 1:
                    # mono -> stereo
                    output = np.column_stack([output, output])
                elif output.ndim == 2:
                    frames, chans = output.shape
                    # If channels-first (1,N) or (2,N), transpose to (N,C)
                    if frames in (1, 2) and chans > 2:
                        output = output.T
                        frames, chans = output.shape
                    # Ensure stereo
                    if chans == 1:
                        output = np.repeat(output, 2, axis=1)
                    elif chans > 2:
                        output = output[:, :2]
                else:
                    # Fallback: flatten to mono then stereo
                    output = output.reshape(-1)
                    output = np.column_stack([output, output])

                try:
                    sd.play(output, sample_rate)
                    sd.wait()
                except Exception as e:
                    # Fallback to mono if device/channel mismatch
                    try:
                        mono = output[:, 0] if output.ndim == 2 else output
                        sd.play(mono, sample_rate)
                        sd.wait()
                    except Exception:
                        raise
            except Exception as e:
                self.feedback_box.append(f"Error playing note {midi_note}: {e}")
        threading.Thread(target=worker, daemon=True).start()

    def _snapshot_current_params(self) -> dict:
        """Return current Sylenth params limited to the JSON spec keys."""
        all_params = self.sylenth.get_all_params()
        return {k: all_params[k] for k in SYLENTH1_PARAM_LIMITS if k in all_params}
        
    def _render_snippet_audibility(self, cand_params: dict, midi_note=69, velocity=100, duration=0.30, sample_rate=44100):
        """Side-effect-free render used ONLY for audibility checks.
        - Snapshots current plugin params
        - Applies candidate params
        - Mutes realtime engine (if available)
        - Renders a short offline snippet with reset=False
        - Restores previous params and realtime mute state
        Returns (y, sr) where y is mono float32, or (None, None) on error.
        """
        prev_params = None
        rt = getattr(self, "rt_engine", None)
        prev_muted = None
        try:
            try:
                prev_params = self._snapshot_current_params()
            except Exception:
                prev_params = None
            try:
                self.sylenth.set_params(dict(cand_params or {}))
            except Exception:
                return None, None
            if rt is not None and hasattr(rt, "set_muted"):
                try:
                    prev_muted = getattr(rt, "_muted", None)
                    rt.set_muted(True)
                except Exception:
                    pass
            note_on = (bytes([0x90, int(midi_note) & 0x7F, int(velocity) & 0x7F]), 0.0)
            note_off = (bytes([0x80, int(midi_note) & 0x7F, int(velocity) & 0x7F]), max(0.05, duration * 0.8))
            y = self.sylenth.synth(
                [note_on, note_off],
                duration=duration,
                sample_rate=sample_rate,
                reset=False,
            )
            y = np.asarray(y)
            if y.ndim == 2:
                if y.shape[0] in (1, 2) and y.shape[1] > 2:
                    y = y.T
                if y.shape[1] > 1:
                    y = y.mean(axis=1)
                else:
                    y = y.reshape(-1)
            elif y.ndim > 2:
                y = y.reshape(-1)
            y = y.astype(np.float32, copy=False)
            if y.size:
                y = y - float(np.mean(y))
            peak = float(np.max(np.abs(y))) if y.size else 0.0
#            if not np.isfinite(peak) or peak < 1e-9:
#                return y, sample_rate
#            y = 0.95 * (y / peak)
            return y, sample_rate
        except Exception:
            return None, None
        finally:
            if rt is not None and hasattr(rt, "set_muted"):
                try:
                    if isinstance(prev_muted, bool):
                        rt.set_muted(prev_muted)
                    else:
                        rt.set_muted(False)
                except Exception:
                    pass
            if prev_params:
                try:
                    self.sylenth.set_params(prev_params)
                except Exception:
                    pass
    
    # --- AudioCommons timbral dataset helpers ---
    def _render_snippet(self, midi_note=69, velocity=100, duration=0.5, sample_rate=44100):
        """Render a short mono snippet from the current patch without playing it.
        Returns (y, sr) where y is float32 mono. Handles DC removal and safe normalization.
        """
        try:
            note_on = (bytes([0x90, midi_note, velocity]), 0.0)
            note_off = (bytes([0x80, midi_note, velocity]), max(0.05, duration * 0.8))
            y = self.sylenth.synth(
                [note_on, note_off],
                duration=duration,
                sample_rate=sample_rate,
                reset=True,
            )
            y = np.asarray(y)
            # To mono
            if y.ndim == 2:
                if y.shape[0] in (1, 2) and y.shape[1] > 2:
                    y = y.T
                if y.shape[1] > 1:
                    y = y.mean(axis=1)
                else:
                    y = y.reshape(-1)
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
            self.feedback_box.append(f"Timbral: render error: {e}")
            return None, None
            
    def _is_patch_audible(self, params, threshold=1e-2) -> bool:
        try:
            y, sr = self._render_snippet_audibility(
                params, midi_note=69, velocity=100, duration=0.30, sample_rate=44100
            )
            if y is None:
                return False
            y = np.asarray(y)
            if y.ndim == 2:
                y = y[:, 0]
            if y.size == 0 or not np.isfinite(y).any():
                return False
            # Compute peak and RMS on the raw buffer
            finite = np.isfinite(y)
            if not finite.any():
                return False
            y = y[finite]
            peak = float(np.nanmax(np.abs(y))) if y.size else 0.0
            rms = float(np.sqrt(np.nanmean(y * y))) if y.size else 0.0
            # Require both some peak and some sustained energy.
            if not np.isfinite(peak) or not np.isfinite(rms):
                return False
            return (peak >= float(threshold)) and (rms >= float(threshold) * 0.25)
        except Exception:
            return False

    def _compute_timbral_models(self, y, sr):
        """Compute AudioCommons timbral model values; returns dict of 7 attributes or {} if unavailable."""
        results = {}
        if y is None or sr is None:
            return results
        if not y.size or not np.isfinite(y).all() or float(np.max(np.abs(y))) < 1e-6:
            self.feedback_box.append("Timbral: audio snippet is silent/near-silent; skipping model computation.")
            return results
        if not TIMBRAL_AVAILABLE:
            self.feedback_box.append("Timbral: timbral_models not installed. Run `pip install timbral_models`.")
            return results

        def safe_call(func):
            try:
                val = func(y, fs=sr)
                valf = float(val)
                if not np.isfinite(valf):
                    return None
                return valf
            except Exception:
                return None

        results["brightness"] = safe_call(timbral_brightness)
        results["depth"] = safe_call(timbral_depth)
        results["hardness"] = safe_call(timbral_hardness)
        results["roughness"] = safe_call(timbral_roughness)
        results["warmth"] = safe_call(timbral_warmth)
        results["sharpness"] = safe_call(timbral_sharpness)
        results["boominess"] = safe_call(timbral_booming)
        return results
        
    def _midi_to_note_name(self, midi_note: int) -> str:
        names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
        n = int(midi_note)
        pitch = names[n % 12]
        octave = (n // 12) - 1
        return f"{pitch}{octave}"

    def _sanitize_program_name(self, program_value) -> str:
        # Convert program parameter to a compact identifier (letters/digits only)
        prog = str(program_value) if program_value is not None else ""
        parts = re.findall(r"[A-Za-z0-9]+", prog)
        return "".join(parts) or "Preset"

    def _timbral_dataset_load(self):
        if not os.path.isfile(TIMBRAL_DATASET_FILE):
            return []
        try:
            with open(TIMBRAL_DATASET_FILE, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return [data]
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _timbral_dataset_save(self, entries: list):
        tmp = TIMBRAL_DATASET_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(entries, f, indent=2)
        os.replace(tmp, TIMBRAL_DATASET_FILE)

    def _tokenize_simple(self, text: str):
        return set(re.findall(r"[a-zA-Z0-9#+-]+", (text or "").lower()))

    def _parse_sound_type(self, text: str):
        tokens = self._tokenize_simple(text)
        for t in ("lead","pad","pluck","bass","keys","fx","arp","poly","mono"):
            if t in tokens:
                return t
        return None

    # --- Preset logic ---
    def save_current_preset(self, name):
        """Save the current Sylenth1 parameters as a preset with the given name."""
        # Only save parameters in SYLENTH1_PARAM_LIMITS
        all_params = self.sylenth.get_all_params()
        preset_params = {k: all_params[k] for k in SYLENTH1_PARAM_LIMITS if k in all_params}
        self.presets[name] = preset_params
        self.save_all_presets()
        self.refresh_preset_list()
        self.feedback_box.append(f"Preset '{name}' saved.")

    def load_preset(self, name):
        """Load the named preset into Sylenth1."""
        if name not in self.presets:
            self.feedback_box.append(f"Preset '{name}' not found.")
            return
        preset_params = self.presets[name]
        self.sylenth.set_params(preset_params)
        self.sylenth.refresh_params()
        # Update param table
        for i, pname in enumerate(self.sylenth.param_names):
            value = self.sylenth.param_values[pname]
            self.param_table.setItem(i, 1, QTableWidgetItem(str(value)))
            if pname in preset_params:
                self.param_table.item(i, 0).setBackground(Qt.yellow)
                self.param_table.item(i, 1).setBackground(Qt.yellow)
            else:
                self.param_table.item(i, 0).setBackground(Qt.white)
                self.param_table.item(i, 1).setBackground(Qt.white)
        self.feedback_box.append(f"Preset '{name}' loaded.")
        self.current_preset_name = name

    def save_all_presets(self):
        """Write the preset dictionary to a JSON file."""
        try:
            with open(self.preset_file, "w") as f:
                json.dump(self.presets, f, indent=2)
        except Exception as e:
            self.feedback_box.append(f"Error saving presets: {e}")

    def load_all_presets(self):
        """Read the preset dictionary from a JSON file."""
        if os.path.isfile(self.preset_file):
            try:
                with open(self.preset_file, "r") as f:
                    self.presets = json.load(f)
            except Exception as e:
                self.presets = {}
                print(f"Failed to load presets: {e}")
        else:
            self.presets = {}

    def get_all_presets(self):
        """Return a list of preset names."""
        return list(self.presets.keys())

    def refresh_preset_list(self):
        """Refresh the QListWidget to show all preset names."""
        self.preset_list.clear()
        for name in sorted(self.get_all_presets()):
            self.preset_list.addItem(QListWidgetItem(name))

    # --- Preset UI handlers ---
    def on_save_preset(self):
        name, ok = QInputDialog.getText(self, "Save Preset", "Enter a name for this preset:")
        if not ok or not name:
            return
        if name in self.presets:
            reply = QMessageBox.question(
                self, "Overwrite Preset", f"Preset '{name}' already exists. Overwrite?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        self.save_current_preset(name)

    def on_load_preset(self):
        selected = self.preset_list.selectedItems()
        if not selected:
            QMessageBox.information(self, "No Preset Selected", "Please select a preset to load.")
            return
        name = selected[0].text()
        self.load_preset(name)
        if self.active_slider_widget:
            self.active_slider_widget.setStyleSheet("")
            self.active_slider_widget = None
        self.active_slider_label.setText("No slider engaged.")
        self.current_preset_name = name
        
    def on_delete_preset(self):
        selected = self.preset_list.selectedItems()
        if not selected:
            QMessageBox.information(self, "No Preset Selected", "Please select at least one preset to delete.")
            return
        names = [item.text() for item in selected]
        reply = QMessageBox.question(
            self, "Delete Preset(s)",
            f"Are you sure you want to delete these preset(s)?\n\n" + "\n".join(names),
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        for name in names:
            if name in self.presets:
                del self.presets[name]
        self.save_all_presets()
        self.refresh_preset_list()
        self.feedback_box.append(f"Deleted preset(s): {', '.join(names)}.")

    def check_external_changes(self):
        # Get current params
        current = self.sylenth.get_all_params()
        program_now = current.get("program")

        # Detect preset change
        preset_changed = (program_now != self.last_program_value)
        if preset_changed:
            self.feedback_box.append(f"Preset/program changed in Sylenth1! Syncing full plugin state...")
            # Save and restore raw_state to ensure full sync
            raw = self.sylenth.synth.raw_state
            self.sylenth.synth.raw_state = raw
            self.sylenth.refresh_params()
            # Update param table with new values
            for i, name in enumerate(self.sylenth.param_names):
                value = self.sylenth.param_values[name]
                self.param_table.setItem(i, 1, QTableWidgetItem(str(value)))
                self.param_table.item(i, 0).setBackground(Qt.green)
                self.param_table.item(i, 1).setBackground(Qt.green)
            self.last_program_value = program_now
        else:
            # Standard update for parameter changes (your original code)
            updates = {}
            for k, v in current.items():
                last_val = self.last_param_values[k]
                try:
                    if abs(float(v) - float(last_val)) > 1e-3:
                        updates[k] = v
                except (TypeError, ValueError):
                    if v != last_val:
                        updates[k] = v
            if updates:
                self.feedback_box.append("Detected changes in Sylenth1 GUI:")
                for k, v in updates.items():
                    self.feedback_box.append(f"{k}: {v}")
                for i, name in enumerate(self.sylenth.param_names):
                    value = current[name]
                    self.param_table.setItem(i, 1, QTableWidgetItem(str(value)))
                    if name in updates:
                        self.param_table.item(i, 0).setBackground(Qt.cyan)
                        self.param_table.item(i, 1).setBackground(Qt.cyan)
                    else:
                        self.param_table.item(i, 0).setBackground(Qt.white)
                        self.param_table.item(i, 1).setBackground(Qt.white)
            self.last_param_values = current

    def manual_sync(self):
        self.sylenth.refresh_params()
        for i, name in enumerate(self.sylenth.param_names):
            value = self.sylenth.param_values[name]
            self.param_table.setItem(i, 1, QTableWidgetItem(str(value)))
            self.param_table.item(i, 0).setBackground(Qt.white)
            self.param_table.item(i, 1).setBackground(Qt.white)
        self.feedback_box.append("Manually synced parameters from Sylenth1.")
        
    def show_sylenth1_gui(self):
        self.sylenth.synth.show_editor()
        
    def play_test_note(self):
        note = 60
        velocity = 100
        self.rt_engine.note_on(note, velocity)
        QTimer.singleShot(600, lambda: self.rt_engine.note_off(note, velocity))

    def on_save_to_timbral_dataset(self):
        dlg = TimbralDatasetDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return
        sound_type, tags, desc = dlg.get_values()

        # Snapshot params from live synth (only spec keys)
        params = self._snapshot_current_params()
        if not params:
            self.feedback_box.append("Timbral: could not read current parameters.")
            return

        # Prefer the app's current preset name; fall back to Sylenth 'program' value if empty
        preset_name = (getattr(self, "current_preset_name", "") or "").strip()
        program_val = params.get("program")
        if not preset_name:
            preset_name = self._sanitize_program_name(program_val)
        program_name = preset_name or "Preset"

        ds = self._timbral_dataset_load()
        saved = 0
        skipped = 0
        import uuid
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

        # Mute realtime engine during offline renders to prevent audible leaks and crackles
        if hasattr(self, "rt_engine") and self.rt_engine:
            self.rt_engine.set_muted(True)
        try:
            for midi_note in [69]:  # A4 only
                y, sr = self._render_snippet(
                    midi_note=midi_note,
                    velocity=int(self.velocity_slider.value()) if hasattr(self, "velocity_slider") else 127,
                    duration=0.5,
                    sample_rate=44100,
                )
                # Loudness normalization before computing AudioCommons values
                if y is not None and y.size:
                    y = self._loudness_normalize_audio(y, fs=sr, target_lufs=-23.0)
                models = self._compute_timbral_models(y, sr)
                if not models or all(v is None for v in models.values()):
                    skipped += 1
                    continue
                note_name = self._midi_to_note_name(midi_note)
                entry_id = uuid.uuid4().hex[:8]
                entry = {
                    "id": entry_id,
                    "name": program_name,
                    "sound_type": sound_type or "",
                    "tags": tags,
                    "description": desc,
                    "models": models,
                    "params": params,
                    "note_midi": int(midi_note),
                    "note_name": note_name,
                    "program": program_val,
                    "created_at": ts,
                    "source": "user",
                }
                ds.append(entry)
                saved += 1
        finally:
            if hasattr(self, "rt_engine") and self.rt_engine:
                self.rt_engine.set_muted(False)

        if saved == 0:
            self.feedback_box.append("Timbral: no valid timbral entries computed (audio likely silent). Nothing saved.")
            return

        try:
            self._timbral_dataset_save(ds)
            # Rebuild timbral index so the new entries participate in retrieval immediately
            self.feedback_box.append(
                f"Timbral: saved {saved} entry(ies) for {program_name} at A4 (MIDI 69); skipped {skipped} due to invalid audio."
            )
        except Exception as e:
            self.feedback_box.append(f"Timbral: failed to save entries: {e}")

if __name__ == "__main__":
    sylenth = Sylenth1Controller(SYLENTH1_PATH)
    llama3 = Llama3Ollama(model_name="llama3:8b")
    app = QApplication(sys.argv)
    win = MainWindow(sylenth, llama3)
    win.show()
    sys.exit(app.exec_())

    # --- Optionally ensure we exclude mix_a/mix_b by default ---
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ... (other attribute initializations)
        # Ensure exclusions for randomization
        if not hasattr(self, "_excluded_randomize_keys"):
            self._excluded_randomize_keys = set()
        self._excluded_randomize_keys.update({"mix_a", "mix_b", "program", "preset_name"})
