"""Text-to-speech. Returns WAV bytes, or None to let the browser speak it instead.

Backends: say (macOS built-in, free), kokoro (nicer local voice), browser (fallback).
"""
import asyncio
import io
import logging
import os
import tempfile
import wave

import numpy as np

from .. import config

log = logging.getLogger("millie.tts")


def _wav_bytes(samples: np.ndarray, rate: int) -> bytes:
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class TTS:
    def __init__(self):
        self.backend = config.TTS_BACKEND
        self._kokoro = None

    async def synth(self, text: str) -> bytes | None:
        try:
            if self.backend == "say":
                return await self._say(text)
            if self.backend == "kokoro":
                return await asyncio.to_thread(self._kokoro_synth, text)
        except Exception as e:
            log.warning("TTS backend '%s' failed (%s); falling back to the browser voice.", self.backend, e)
            self.backend = "browser"
        return None

    async def _say(self, text: str) -> bytes:
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            proc = await asyncio.create_subprocess_exec(
                "say", "-v", config.TTS_VOICE, "-o", path, "--data-format=LEI16@22050", "--", text)
            await proc.wait()
            with open(path, "rb") as f:
                data = f.read()
            if not data:
                raise RuntimeError("say produced no audio")
            return data
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def _kokoro_synth(self, text: str) -> bytes:
        if self._kokoro is None:
            from kokoro_onnx import Kokoro
            self._kokoro = Kokoro(config.KOKORO_MODEL, config.KOKORO_VOICES)
        samples, rate = self._kokoro.create(text, voice=config.KOKORO_VOICE, speed=1.0, lang="en-us")
        return _wav_bytes(np.asarray(samples), rate)
