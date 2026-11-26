"""Speech-to-text. mlx-whisper on Apple Silicon, faster-whisper elsewhere.

Whisper is not a streaming model, so we run it once on the whole finished command
(the VAD tells us when you stopped). For short commands that's fast and accurate.
"""
import logging
import re

import numpy as np

from .. import config

log = logging.getLogger("millie.stt")

WAKE_PREFIX = re.compile(r"^\s*(?:(?:hey|hi|hello|okay|ok)[,.!\s]+)?(?:millie|milly|mili|mille|jarvis)\b[,.!?\s]*", re.I)
HALLUCINATIONS = {"", "you", "thank you.", "thank you", "thanks for watching.", "thanks for watching!", "bye."}


def strip_wake_phrase(text: str) -> str:
    return WAKE_PREFIX.sub("", text, count=1).strip()


class STT:
    def __init__(self):
        self.backend = None
        self._fw = None

    def load(self) -> None:
        if self.backend:
            return
        try:
            import mlx_whisper  # noqa: F401
            self.backend = "mlx"
        except ImportError:
            try:
                from faster_whisper import WhisperModel
                self._fw = WhisperModel("small", compute_type="int8")
                self.backend = "faster-whisper"
            except ImportError:
                raise RuntimeError("No STT engine installed. pip install mlx-whisper (Mac) or faster-whisper.")
        log.info("STT backend: %s", self.backend)

    def transcribe(self, audio: np.ndarray) -> str:
        """audio: float32 mono 16 kHz in [-1, 1]."""
        self.load()
        if self.backend == "mlx":
            import mlx_whisper
            out = mlx_whisper.transcribe(
                audio, path_or_hf_repo=config.STT_MODEL, language=config.STT_LANGUAGE or None,
                condition_on_previous_text=False, verbose=None)
            text = out.get("text", "")
        else:
            segs, _ = self._fw.transcribe(audio, language=config.STT_LANGUAGE or None, vad_filter=False)
            text = " ".join(s.text for s in segs)
        text = text.strip()
        return "" if text.lower() in HALLUCINATIONS else text
