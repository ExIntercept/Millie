"""One live voice conversation over a WebSocket.

State machine:
    idle      -> only the wake word detector is listening
    listening -> VAD collects your command until you stop talking
    thinking  -> STT + LLM (+ tools) are working
    speaking  -> reply is being synthesised and streamed to the browser

Audio in:  16 kHz mono int16 PCM frames (binary WebSocket messages).
Audio out: WAV chunks (binary), one per sentence, so speech starts before the
           LLM has finished writing the whole reply.
Barge-in:  saying the wake word while Millie is thinking/speaking cancels her.
"""
import asyncio
import collections
import logging
import re
import time
from typing import Optional

import numpy as np

from .. import config
from ..assistant import Assistant
from .stt import STT, strip_wake_phrase
from .tts import TTS
from .vad import VAD
from .wakeword import WakeWord

log = logging.getLogger("millie.voice")

WAKE_FRAME_BYTES = 1280 * 2
VAD_FRAME_BYTES = 512 * 2
FRAME_MS = 32
PREROLL_FRAMES = 8          # ~250 ms kept before speech starts
RING_FRAMES = 38            # ~1.2 s of recent audio, so "Hey Millie, <command>" in one breath works
MAX_EMPTY_RETRIES = 2


class Engine:
    """Heavy models, loaded once and shared (single-user local app)."""

    def __init__(self):
        self.wake = WakeWord()
        self.vad = VAD()
        self.stt = STT()
        self.tts = TTS()

    def capabilities(self) -> dict:
        return {"type": "capabilities", "wake": self.wake.available, "wake_name": self.wake.name,
                "vad": self.vad.kind, "tts": self.tts.backend, "name": config.ASSISTANT_NAME}


class SentenceSplitter:
    """Cuts a stream of text deltas into speakable sentences."""
    _END = re.compile(r"(?<=[.!?])\s+")

    def __init__(self):
        self.buf = ""

    def feed(self, delta: str) -> list[str]:
        self.buf += delta
        parts = self._END.split(self.buf)
        if len(parts) == 1:
            if len(self.buf) > 160 and "," in self.buf:  # long run-on: speak up to the last comma
                i = self.buf.rfind(",")
                out, self.buf = self.buf[: i + 1], self.buf[i + 1:].lstrip()
                return [out]
            return []
        self.buf = parts[-1]
        return [p.strip() for p in parts[:-1] if p.strip()]

    def flush(self) -> list[str]:
        s, self.buf = self.buf.strip(), ""
        return [s] if s else []


def clean_for_speech(text: str) -> str:
    cleaned = re.sub(r"[*_`#>]+", "", text).strip()
    return re.sub(r"^[-•\s]+", "", cleaned).strip()


class VoiceSession:
    def __init__(self, ws, engine: Engine, assistant: Assistant):
        self.ws, self.e, self.assistant = ws, engine, assistant
        self.state = "idle"
        self.buf_wake = bytearray()
        self.buf_vad = bytearray()
        self.ring: collections.deque = collections.deque(maxlen=RING_FRAMES)
        self.utt: list[bytes] = []
        self.speech_started = False
        self.silence = 0
        self.t_listen = 0.0
        self.retries = 0
        self.task: Optional[asyncio.Task] = None
        self.pending = 0                       # audio chunks sent but not yet finished playing
        self.idle_evt = asyncio.Event()
        self.idle_evt.set()
        self.e.wake.reset()
        self.e.vad.reset()

    # ---- helpers -------------------------------------------------------
    async def send(self, obj: dict) -> None:
        try:
            await self.ws.send_json(obj)
        except Exception:
            pass

    async def set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            await self.send({"type": "state", "state": state})

    def audio_ended(self) -> None:
        self.pending = max(0, self.pending - 1)
        if self.pending == 0:
            self.idle_evt.set()

    async def close(self) -> None:
        if self.task and not self.task.done():
            self.task.cancel()

    # ---- incoming audio -------------------------------------------------
    async def on_audio(self, data: bytes) -> None:
        self.buf_wake += data
        self.buf_vad += data
        while len(self.buf_wake) >= WAKE_FRAME_BYTES:
            frame = np.frombuffer(bytes(self.buf_wake[:WAKE_FRAME_BYTES]), dtype=np.int16)
            del self.buf_wake[:WAKE_FRAME_BYTES]
            if self.state != "listening" and self.e.wake.process(frame):
                await self.wake()
        while len(self.buf_vad) >= VAD_FRAME_BYTES:
            frame = np.frombuffer(bytes(self.buf_vad[:VAD_FRAME_BYTES]), dtype=np.int16)
            del self.buf_vad[:VAD_FRAME_BYTES]
            await self._vad_frame(frame)

    async def _vad_frame(self, frame: np.ndarray) -> None:
        raw = frame.tobytes()
        self.ring.append(raw)
        if self.state != "listening":
            return
        speaking = self.e.vad.speech_prob(frame) > 0.5
        if speaking:
            if not self.speech_started:
                self.speech_started = True
                self.utt = list(self.ring)[-PREROLL_FRAMES:]
            else:
                self.utt.append(raw)
            self.silence = 0
        elif self.speech_started:
            self.utt.append(raw)
            self.silence += 1
        elif time.perf_counter() - self.t_listen > config.NO_SPEECH_TIMEOUT_S:
            await self.set_state("idle")
            return
        too_long = len(self.utt) * FRAME_MS / 1000 > config.MAX_UTTERANCE_S
        if self.speech_started and (self.silence * FRAME_MS >= config.END_SILENCE_MS or too_long):
            await self._finalize()

    # ---- state transitions ------------------------------------------------
    async def wake(self) -> None:
        await self.interrupt()
        await self.start_listening(preroll=True)
        await self.send({"type": "wake"})

    async def push_to_talk(self) -> None:
        await self.interrupt()
        self.retries = 0
        await self.start_listening(preroll=False)
        await self.send({"type": "wake"})

    async def start_listening(self, preroll: bool) -> None:
        self.e.vad.reset()
        self.utt = list(self.ring) if preroll else []
        self.speech_started = False
        self.silence = 0
        self.t_listen = time.perf_counter()
        await self.set_state("listening")

    async def interrupt(self) -> None:
        if self.task and not self.task.done() and self.task is not asyncio.current_task():
            self.task.cancel()
            try:
                await self.task
            except BaseException:
                pass
        self.pending = 0
        self.idle_evt.set()
        await self.send({"type": "stop_audio"})

    async def stop(self) -> None:
        """User pressed stop: cancel whatever is happening and go quiet."""
        await self.interrupt()
        self.utt, self.speech_started, self.silence = [], False, 0
        await self.set_state("idle")

    async def _finalize(self) -> None:
        audio = np.frombuffer(b"".join(self.utt), dtype=np.int16).astype(np.float32) / 32768.0
        endpoint_ms = self.silence * FRAME_MS
        self.utt, self.speech_started, self.silence = [], False, 0
        await self.set_state("thinking")
        self.task = asyncio.create_task(self._guard(self._handle_audio(audio, endpoint_ms)))

    async def _guard(self, coro) -> None:
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("voice turn failed")
            await self.send({"type": "error", "message": str(e)})
            await self.set_state("idle")

    # ---- a turn -----------------------------------------------------------
    async def _handle_audio(self, audio: np.ndarray, endpoint_ms: int) -> None:
        t0 = time.perf_counter()
        text = await asyncio.to_thread(self.e.stt.transcribe, audio)
        t_stt = time.perf_counter()
        text = strip_wake_phrase(text)
        log.info("STT recognized: %r", text)
        if not text:
            # Just "Hey Millie" (or noise): keep listening for the actual command.
            self.retries += 1
            if self.retries > MAX_EMPTY_RETRIES:
                self.retries = 0
                await self.set_state("idle")
            else:
                await self.start_listening(preroll=False)
            return
        self.retries = 0
        await self.send({"type": "transcript", "role": "user", "text": text})
        await self._respond(text, {"t0": t0, "t_stt": t_stt, "endpoint_ms": endpoint_ms})

    async def handle_text(self, text: str) -> None:
        """Typed input: same brain, no audio in."""
        await self.interrupt()
        await self.set_state("thinking")
        await self.send({"type": "transcript", "role": "user", "text": text})
        self.task = asyncio.create_task(self._guard(self._respond(text, {"t0": time.perf_counter()})))

    async def _respond(self, text: str, m: dict) -> None:
        splitter, queue = SentenceSplitter(), asyncio.Queue()
        speaker = asyncio.create_task(self._speak_loop(queue, m))

        async def on_text(delta: str) -> None:
            for sentence in splitter.feed(delta):
                await queue.put(sentence)

        def on_first_token() -> None:
            m["first_token"] = time.perf_counter()

        try:
            reply = await self.assistant.turn(text, on_text, on_first_token)
            for sentence in splitter.flush():
                await queue.put(sentence)
            await queue.put(None)
            await speaker
        finally:
            if not speaker.done():
                speaker.cancel()
        await self.send({"type": "transcript", "role": "assistant", "text": reply})
        await self._report_latency(m)

        if reply.rstrip().endswith("?"):      # Millie asked a question: listen without needing the wake word
            try:
                await asyncio.wait_for(self.idle_evt.wait(), timeout=25)
            except asyncio.TimeoutError:
                pass
            await self.start_listening(preroll=False)
        else:
            await self.set_state("idle")

    async def _speak_loop(self, queue: asyncio.Queue, m: dict) -> None:
        while True:
            sentence = await queue.get()
            if sentence is None:
                return
            sentence = clean_for_speech(sentence)
            if not sentence:
                continue
            m.setdefault("first_sentence", time.perf_counter())
            wav = await self.e.tts.synth(sentence)
            if "first_audio" not in m:
                m["first_audio"] = time.perf_counter()
                await self.set_state("speaking")
            self.pending += 1
            self.idle_evt.clear()
            if wav:
                await self.ws.send_bytes(wav)
            else:
                await self.send({"type": "speak", "text": sentence})

    async def _report_latency(self, m: dict) -> None:
        def ms(a, b):
            return round((a - b) * 1000) if a is not None and b is not None else None

        t0, t_stt = m["t0"], m.get("t_stt")
        base = t_stt or t0
        await self.send({
            "type": "latency",
            "endpoint_ms": m.get("endpoint_ms"),
            "stt_ms": ms(t_stt, t0),
            "llm_first_token_ms": ms(m.get("first_token"), base),
            "tts_first_byte_ms": ms(m.get("first_audio"), m.get("first_sentence")),
            "to_first_audio_ms": ms(m.get("first_audio"), t0),
        })
