"""Voice activity detection: is someone speaking in this 32 ms frame?

Uses Silero VAD when installed, otherwise a crude energy gate so the app still works.
"""
import logging

import numpy as np

log = logging.getLogger("millie.vad")
FRAME_SAMPLES = 512  # 32 ms at 16 kHz


class VAD:
    def __init__(self):
        self.model = None
        self.kind = "energy"
        try:
            import torch
            from silero_vad import load_silero_vad
            self._torch = torch
            self.model = load_silero_vad()
            self.kind = "silero"
        except Exception as e:
            log.warning("Silero VAD unavailable (%s); using a simple energy gate.", e)

    def reset(self) -> None:
        if self.model is not None:
            try:
                self.model.reset_states()
            except Exception:
                pass

    def speech_prob(self, frame: np.ndarray) -> float:
        """frame: int16 array of FRAME_SAMPLES -> probability of speech (0..1)."""
        if self.model is not None:
            x = self._torch.from_numpy(frame.astype(np.float32) / 32768.0)
            return float(self.model(x, 16000).item())
        rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
        return 1.0 if rms > 500 else 0.0
