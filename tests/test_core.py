"""Run with:  python -m unittest discover -s tests -v
No network, models, or API keys needed: the LLM, STT, TTS, VAD and WebSocket are faked.
"""
import asyncio
import json
import sys
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config, files as filelib, research as research_mod            # noqa: E402
from backend.assistant import Assistant, music_command                                            # noqa: E402
from backend.db import DB                                                          # noqa: E402
from backend.hub import Hub                                                        # noqa: E402
from backend.research import ResearchManager                                       # noqa: E402
from backend.tools import ToolBox                                                  # noqa: E402
from backend.voice.session import Engine, SentenceSplitter, VoiceSession           # noqa: E402
from backend.voice.stt import strip_wake_phrase                                    # noqa: E402


def future_iso(days=2, hour=18, minute=30) -> str:
    d = datetime.now(ZoneInfo(config.TIMEZONE)) + timedelta(days=days)
    return d.replace(hour=hour, minute=minute, second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M:%S")


class FakeLLM:
    """Plays back scripted turns. Each turn = (text, [(tool_name, args_dict), ...])."""
    def __init__(self, script):
        self.script, self.calls_seen = list(script), []

    async def stream(self, messages, tools=None):
        self.calls_seen.append(messages)
        text, calls = self.script.pop(0)
        out = [{"id": f"c{i}", "name": n, "arguments": json.dumps(a)} for i, (n, a) in enumerate(calls)]
        for word in text.split(" "):
            if word:
                yield ("text", word + " ")
        yield ("done", text, out)

    async def run(self, messages, tools=None):
        text, calls = "", []
        async for ev in self.stream(messages, tools):
            if ev[0] == "done":
                text, calls = ev[1], ev[2]
        return text, calls


class DBTests(unittest.TestCase):
    def test_insight_search_across_tabs_and_delete(self):
        db = DB()
        a, b = db.create_tab("capstone 1"), db.create_tab("capstone 2")
        db.add_insight(a, "Transformers for ASR", "Whisper uses an encoder-decoder transformer", ["http://x"])
        db.add_insight(b, "Diffusion", "Latent diffusion models for images")
        hits = db.search_insights("whisper transformer")
        self.assertEqual([h["tab_name"] for h in hits], ["capstone 1"])
        self.assertEqual(hits[0]["sources"], ["http://x"])
        self.assertEqual(db.search_insights("diffusion", [a]), [])
        db.delete_tab(a)
        self.assertEqual(db.search_insights("whisper"), [])


class ToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.db, self.hub, self.started = DB(), Hub(), []
        self.tb = ToolBox(self.db, self.hub, lambda tid, ins: self.started.append((tid, ins)) or True)

    async def call(self, _tool, **args):
        return json.loads(await self.tb.call(_tool, json.dumps(args)))

    async def test_reminder_future_ok_past_rejected(self):
        r = await self.call("create_reminder", text="Deloitte interview", when=future_iso())
        self.assertTrue(r["ok"], r)
        self.assertEqual(len(self.db.list_reminders()), 1)
        past = await self.call("create_reminder", text="x", when="2020-01-01T10:00:00")
        self.assertFalse(past["ok"])
        self.assertIn("past", past["error"])

    async def test_capstone_flow_links_and_starts_research(self):
        self.db.create_tab("capstone 1")
        r = await self.call("create_research_tab", name="capstone 2", link_to=["capstone 1"],
                            instructions="Continue the lit survey per the uploaded ppt")
        self.assertTrue(r["ok"], r)
        tab = [t for t in self.db.list_tabs() if t["name"] == "capstone 2"][0]
        self.assertEqual(tab["links"], [1])
        self.assertEqual(self.started, [(tab["id"], "Continue the lit survey per the uploaded ppt")])

    async def test_unknown_link_target_creates_nothing_and_ambiguity_is_reported(self):
        self.db.create_tab("capstone 1"); self.db.create_tab("capstone 2")
        r = await self.call("create_research_tab", name="new", link_to=["capstone"])
        self.assertFalse(r["ok"])
        self.assertIn("Ask the user", r["error"])
        self.assertEqual(len(self.db.list_tabs()), 2)

    async def test_stopwatch_and_bad_json_and_unknown_tool(self):
        await self.call("start_stopwatch", label="study")
        s = await self.call("stop_stopwatch")
        self.assertTrue(s["ok"])
        self.assertFalse((await self.call("stop_stopwatch"))["ok"])
        self.assertFalse(json.loads(await self.tb.call("nope", "{}"))["ok"])
        self.assertFalse(json.loads(await self.tb.call("create_note", "{bad"))["ok"])
        self.assertFalse((await self.call("create_note"))["ok"])  # missing required arg


class AssistantTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_loop_sets_reminder_and_replies(self):
        db, hub = DB(), Hub()
        tb = ToolBox(db, hub, lambda *a: True)
        llm = FakeLLM([
            ("Sure.", [("create_reminder", {"text": "Deloitte interview", "when": future_iso()})]),
            ("Reminder set for six thirty in two days.", []),
        ])
        spoken = []

        async def on_text(d): spoken.append(d)

        reply = await Assistant(db, llm, tb).turn("remind me in 2 days", on_text)
        self.assertEqual(len(db.list_reminders()), 1)
        self.assertIn("Reminder set", reply)
        self.assertTrue("".join(spoken).startswith("Sure."))
        tool_msgs = [m for m in llm.calls_seen[1] if m["role"] == "tool"]
        self.assertTrue(json.loads(tool_msgs[0]["content"])["ok"])
        self.assertIn("Current date and time", llm.calls_seen[0][0]["content"])


class ResearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_uses_linked_context_and_saves_findings(self):
        db, hub = DB(), Hub()
        c1, c2 = db.create_tab("capstone 1"), db.create_tab("capstone 2")
        db.add_insight(c1, "Prior finding", "Wav2vec2 pretraining reduces labeled data needs")
        db.link_tabs(c2, c1)
        db.add_file(c2, "survey.pptx", "/tmp/x", "text", False, "[slide 1]\nLiterature survey outline: ASR robustness")
        research_mod._search_sync = lambda q, n=6: [{"title": "Paper", "href": "http://p", "body": "snippet"}]
        llm = FakeLLM([
            ("", [("web_search", {"query": "ASR robustness survey"})]),
            ("", [("save_insight", {"title": "Noise robustness", "content": "Findings...", "sources": ["http://p"]})]),
            ("", [("finish", {"summary": "Covered noise robustness."})]),
        ])
        events = []
        hub.clients.add(type("C", (), {"send_json": staticmethod(lambda m: _async(events.append(m)))})())
        mgr = ResearchManager(db, hub, llm)
        self.assertTrue(mgr.start(c2, "continue the lit survey"))
        self.assertFalse(mgr.start(c2, "again"))          # busy tabs refuse a second run
        await mgr.tasks[c2]
        system = llm.calls_seen[0][0]["content"]
        self.assertIn("Wav2vec2", system)                  # linked tab's findings reached the agent
        self.assertIn("Literature survey outline", system) # uploaded file text reached the agent
        kinds = [(i["kind"], i["title"]) for i in db.get_insights(c2)]
        self.assertIn(("finding", "Noise robustness"), kinds)
        self.assertIn(("summary", "Summary of this run"), kinds)
        self.assertEqual(db.get_tab(c2)["status"], "done")
        self.assertTrue(any(e.get("type") == "research_done" for e in events))
        # reuse across tabs: a third tab can find tab 2's work
        self.assertTrue(db.search_insights("noise robustness"))


async def _async(x):
    return x


class SmallPieces(unittest.TestCase):
    def test_sentence_splitter(self):
        sp = SentenceSplitter()
        out = []
        for d in ["Okay, I set it for 6", ".30 pm. ", "Anything el", "se?"]:
            out += sp.feed(d)
        out += sp.flush()
        self.assertEqual(out, ["Okay, I set it for 6.30 pm.", "Anything else?"])

    def test_music_fast_path_only_for_music_commands(self):
        self.assertEqual(music_command("pause"), {"action": "pause"})
        self.assertEqual(music_command("play bohemian rhapsody"), {"query": "bohemian rhapsody"})
        # "listen to" in the middle of a sentence is not a music command
        self.assertIsNone(music_command("remind me to listen to the podcast tomorrow"))
        # catalog aliases only match the whole utterance
        db = DB()
        self.assertIsNotNone(db.find_in_music_catalog("study", exact=True))
        self.assertIsNone(db.find_in_music_catalog("start a stopwatch called study", exact=True))

    def test_strip_wake_phrase(self):
        self.assertEqual(strip_wake_phrase("Hey Millie, set a reminder"), "set a reminder")
        self.assertEqual(strip_wake_phrase("Hey Millie."), "")
        self.assertEqual(strip_wake_phrase("remind me about Millie's birthday"), "remind me about Millie's birthday")

    def test_file_extract_text_and_image_without_ocr(self):
        tmp = Path(__file__).parent / "_tmp.txt"
        tmp.write_text("hello world")
        try:
            self.assertEqual(filelib.extract(tmp, False), ("text", "hello world"))
        finally:
            tmp.unlink()
        img = Path(__file__).parent / "_tmp.png"
        img.write_bytes(b"\x89PNG")
        try:
            self.assertEqual(filelib.extract(img, False), ("image", ""))   # OCR off -> nothing stored
        finally:
            img.unlink()


# ---- voice session with everything faked ------------------------------------
class FakeWS:
    def __init__(self): self.json, self.bytes = [], []
    async def send_json(self, o): self.json.append(o)
    async def send_bytes(self, b): self.bytes.append(b)


class FakeWake:
    available, name = True, "fake"
    def __init__(self): self.n, self.fire_at = 0, 3
    def reset(self): pass
    def process(self, frame):
        self.n += 1
        return self.n == self.fire_at


class FakeVAD:
    kind = "fake"
    def reset(self): pass
    def speech_prob(self, frame): return 1.0 if np.abs(frame).mean() > 1000 else 0.0


class FakeSTT:
    backend = "fake"
    def __init__(self, text): self.text = text
    def transcribe(self, audio): return self.text


class FakeTTS:
    backend = "fake"
    async def synth(self, text): return b"RIFFfake:" + text.encode()


class FakeAssistant:
    def __init__(self, reply): self.reply, self.heard = reply, []
    async def turn(self, text, on_text=None, on_first_token=None):
        self.heard.append(text)
        if on_first_token: on_first_token()
        for w in self.reply.split(" "):
            await on_text(w + " ")
        return self.reply


def pcm(loud: bool, ms: int) -> bytes:
    n = 16 * ms
    return (np.full(n, 5000 if loud else 0, dtype=np.int16)).tobytes()


class VoiceTests(unittest.IsolatedAsyncioTestCase):
    def make(self, stt_text, reply):
        eng = Engine.__new__(Engine)
        eng.wake, eng.vad, eng.stt, eng.tts = FakeWake(), FakeVAD(), FakeSTT(stt_text), FakeTTS()
        ws, asst = FakeWS(), FakeAssistant(reply)
        return VoiceSession(ws, eng, asst), ws, asst

    async def feed_until_done(self, s, chunks):
        for c in chunks:
            await s.on_audio(c)
            await asyncio.sleep(0)
        for _ in range(100):
            await asyncio.sleep(0.01)
            if s.task and s.task.done():
                break

    async def test_wake_listen_think_speak_idle(self):
        s, ws, asst = self.make("Hey Millie, set a note to buy milk", "Saved your note. Anything else.")
        # silence (wake fires on 3rd 80ms frame), then speech, then 1 s of silence to end the utterance
        await self.feed_until_done(s, [pcm(False, 80)] * 3 + [pcm(True, 640)] + [pcm(False, 1000)])
        states = [m["state"] for m in ws.json if m["type"] == "state"]
        self.assertEqual(states[:3], ["listening", "thinking", "speaking"])
        self.assertEqual(states[-1], "idle")
        self.assertEqual(asst.heard, ["set a note to buy milk"])         # wake phrase stripped
        self.assertEqual(len(ws.bytes), 2)                               # one WAV chunk per sentence
        self.assertTrue(any(m["type"] == "wake" for m in ws.json))
        lat = [m for m in ws.json if m["type"] == "latency"][0]
        self.assertIsNotNone(lat["to_first_audio_ms"])
        self.assertEqual(s.pending, 2)

    async def test_question_reopens_mic_after_playback(self):
        s, ws, _ = self.make("Hey Millie remind me about the interview", "What time should I set it for?")
        await self.feed_until_done(s, [pcm(False, 80)] * 3 + [pcm(True, 640)] + [pcm(False, 1000)])
        await asyncio.sleep(0.05)
        self.assertEqual(s.state, "speaking")      # still waiting for the browser to finish playing
        s.audio_ended()
        await asyncio.sleep(0.05)
        self.assertEqual(s.state, "listening")     # follow-up without needing the wake word

    async def test_bare_wake_phrase_keeps_listening_then_times_out(self):
        s, ws, asst = self.make("Hey Millie.", "unused")
        await self.feed_until_done(s, [pcm(False, 80)] * 3 + [pcm(True, 400)] + [pcm(False, 1000)])
        await asyncio.sleep(0.05)
        self.assertEqual(asst.heard, [])
        self.assertEqual(s.state, "listening")

    async def test_barge_in_cancels_speech(self):
        s, ws, _ = self.make("x", "unused")
        s.state = "speaking"
        s.task = asyncio.create_task(asyncio.sleep(10))
        s.pending = 3
        s.e.wake.n, s.e.wake.fire_at = 0, 1
        await s.on_audio(pcm(False, 80))
        await asyncio.sleep(0.01)
        self.assertTrue(s.task.cancelled())
        self.assertTrue(any(m["type"] == "stop_audio" for m in ws.json))
        self.assertEqual(s.state, "listening")
        self.assertEqual(s.pending, 0)

    async def test_typed_input_uses_same_brain(self):
        s, ws, asst = self.make("", "Done.")
        await s.handle_text("start a stopwatch")
        await asyncio.wait_for(s.task, 2)
        self.assertEqual(asst.heard, ["start a stopwatch"])
        self.assertEqual(len(ws.bytes), 1)


if __name__ == "__main__":
    unittest.main()
