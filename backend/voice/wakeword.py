"""Wake word detection with openWakeWord (runs fully on-device).

Feed it 80 ms frames (1280 samples of 16 kHz int16 audio).
Supports ONNX models and TFLite (.tflite) models natively.
"""
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .. import config

log = logging.getLogger("millie.wakeword")
FRAME_SAMPLES = 1280
COOLDOWN_S = 2.0


def _ensure_tflite_runtime():
    """Expose LiteRT as tflite_runtime so openwakeword can load .tflite models."""
    try:
        import ai_edge_litert.interpreter
        import types
        if "tflite_runtime" not in sys.modules:
            mod = types.ModuleType("tflite_runtime")
            mod.interpreter = ai_edge_litert.interpreter
            sys.modules["tflite_runtime"] = mod
            sys.modules["tflite_runtime.interpreter"] = ai_edge_litert.interpreter
    except ImportError:
        pass


class WakeWord:
    def __init__(self):
        self.model = None
        self.name = "unavailable"
        self._last = 0.0
        try:
            self._load()
        except Exception as e:
            log.warning("Wake word disabled (%s). Use the push-to-talk button or type instead.", e)

    def _load(self):
        _ensure_tflite_runtime()
        import openwakeword
        from openwakeword.model import Model

        target = str(config.WAKEWORD_MODEL).strip() if config.WAKEWORD_MODEL else ""
        if not target:
            # Check default custom model in data/
            default_tflite = config.DATA_DIR / "hey_millie.tflite"
            default_onnx = config.DATA_DIR / "hey_millie.onnx"
            if default_tflite.exists():
                target = str(default_tflite)
            elif default_onnx.exists():
                target = str(default_onnx)
            else:
                target = "hey_jarvis"

        target_path = Path(target)
        if target_path.exists():
            framework = "tflite" if target.endswith(".tflite") else "onnx"
            log.info("Loading custom wake word model '%s' via %s...", target_path.name, framework)
            self.model = Model(wakeword_models=[str(target_path.resolve())], inference_framework=framework)
            self.name = target_path.stem
        else:
            try:
                self.model = Model(wakeword_models=[target], inference_framework="onnx")
            except Exception:
                log.info("Downloading openWakeWord base models (one time)...")
                openwakeword.utils.download_models(model_names=[target])
                self.model = Model(wakeword_models=[target], inference_framework="onnx")
            self.name = target

        log.info("Wake word ready: '%s' (threshold: %.2f)", self.name, config.WAKEWORD_THRESHOLD)

    @property
    def available(self) -> bool:
        return self.model is not None

    def reset(self) -> None:
        if self.model is not None:
            try:
                self.model.reset()
            except Exception:
                pass

    def process(self, frame: np.ndarray) -> bool:
        """frame: int16 array of FRAME_SAMPLES. Returns True when the wake word is heard."""
        if self.model is None:
            return False
        scores = self.model.predict(frame)
        score = max(scores.values()) if scores else 0.0
        if score > 0.15:
            log.info("Wake word '%s' score: %.3f (threshold: %.2f)", self.name, score, config.WAKEWORD_THRESHOLD)
        if score >= config.WAKEWORD_THRESHOLD and time.time() - self._last > COOLDOWN_S:
            log.info("Wake word '%s' TRIGGERED (score: %.3f)", self.name, score)
            self._last = time.time()
            self.reset()
            return True
        return False
