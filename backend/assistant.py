"""The assistant's brain for one conversation: prompt + LLM + tool loop."""
import json
import logging
import re
from typing import Awaitable, Callable, Optional

from . import config, files
from .db import DB
from .llm import LLM, assistant_tool_message, tool_result_message
from .tools import ToolBox, now_local

log = logging.getLogger("millie.assistant")

MAX_TOOL_ROUNDS = 5
HISTORY_MESSAGES = 6


def system_prompt(db: DB) -> str:
    now = now_local()
    tabs = ", ".join(t["name"] for t in db.list_tabs()[:5]) or "(none)"
    
    # Knowledge base context (compact to save TPM on Groq)
    k_items = db.list_knowledge(limit=4)
    if k_items:
        kb_text = "\n".join(f"- [{k['category'].upper()}] {k['title']}: {k['content'][:140]}" for k in k_items)
    else:
        kb_text = "(empty)"

    # Workspace files overview (compact)
    try:
        ws_info = files.list_workspace()
        ws_files = [f["rel_path"] for f in ws_info.get("files", []) if not f.get("is_dir")]
        files_text = ", ".join(ws_files[:8]) if ws_files else "(ready)"
    except Exception:
        files_text = "(ready)"

    return f"""You are {config.ASSISTANT_NAME}, a voice assistant living inside the user's personal workspace app.
Current date and time: {now.strftime('%A, %d %B %Y, %I:%M:%S %p')} ({config.TIMEZONE}).
For relative countdowns (e.g. "10 second timer", "5 minutes"), pass `in_seconds` directly to create_alarm or create_reminder.

Capabilities:
1. Files & Suite: Create files in workspace (Word .docx, .pdf, .md, .txt), convert formats, and merge multiple documents/images.
2. Knowledge Base: Remember facts, user preferences, and project context across sessions via save_knowledge.
3. Spotify & Music: Auto-play any song, artist, album, or playlist on Spotify via `play_spotify(query='song title')`.
4. Browser Lookup: Open any topic, search query, or website in a new browser tab via `browse_lookup(query=...)`.
5. App Opener: Launch macOS apps or open URLs with open_app.

Workspace files: {files_text}
Knowledge Base:
{kb_text}
Tabs: {tabs}

How to behave:
- Your replies are SPOKEN aloud. Keep them to one or two short sentences. No markdown, no lists, no emojis.
- ALWAYS execute tools before speaking. Do the task with tools instead of just describing it in text.
- MUSIC & SPOTIFY RULE: To control Spotify playback like the dashboard buttons:
  * To pause: call `play_spotify(action="pause")`.
  * To resume / unpause: call `play_spotify(action="play")`.
  * To skip / next track: call `play_spotify(action="next")`.
  * To go back / previous track: call `play_spotify(action="previous")`.
  * To play a specific song: call `play_spotify(query="<song name>")`.
  Leave query empty when pausing, resuming, or skipping. Reply ONLY with "Done."
- KNOWLEDGE BASE RULE: Whenever the user tells you to remember, save, or add something to the knowledge base, you MUST call `save_knowledge(title=..., content=..., category=...)`. NEVER say you remembered or saved something unless you actually called `save_knowledge`.
- FILES & PDF RULE: When asked to create documents, call `create_workspace_file`. For `.docx` it builds Word documents; for `.pdf` it builds valid, readable PDFs.
- BROWSER RULE: When the user asks to look up or search for something in a browser, call `browse_lookup(query=...)`."""



_POLITE = r"^(?:please\s+|can you\s+|could you\s+)?"
_PAUSE = re.compile(_POLITE + r"(?:pause|stop)(?:\s+(?:it|the\s+music|music|playing|spotify|song))?$", re.I)
_RESUME = re.compile(_POLITE + r"(?:resume|unpause|continue)(?:\s+(?:it|the\s+music|music|playing|spotify|song))?$", re.I)
_NEXT = re.compile(_POLITE + r"(?:next|skip|play\s+next)(?:\s+(?:one|this|song|track|it))?$", re.I)
_PREV = re.compile(_POLITE + r"(?:previous|prev|go\s+back|last\s+song)(?:\s+(?:one|song|track|it))?$", re.I)
_PLAY = re.compile(r"^(?:can you\s+|could you\s+|please\s+|i want to\s+|i'd like to\s+)?"
                   r"(?:play\s+me|play\s+the\s+song|play\s+some|play|put\s+on|listen\s+to|hear)\b\s*(.*)$", re.I)
_NOT_MUSIC = ("app", "file", "tab", "note", "video", "youtube")


def music_command(u: str) -> Optional[dict]:
    """Return play_spotify arguments if `u` is a plain music command, else None."""
    if _PAUSE.match(u):
        return {"action": "pause"}
    if _RESUME.match(u) or u.lower() in {"play", "play music", "start music"}:
        return {"action": "play"}
    if _NEXT.match(u):
        return {"action": "next"}
    if _PREV.match(u):
        return {"action": "previous"}
    m = _PLAY.match(u)   # only when the sentence STARTS with play / put on / listen to / hear
    if m:
        target = m.group(1).strip()
        if target.lower() in {"music", "song", "songs", "something", "spotify", ""}:
            return {"action": "play"}
        if not target.lower().startswith(_NOT_MUSIC):
            return {"query": target}
    return None


class Assistant:
    def __init__(self, db: DB, llm: LLM, toolbox: ToolBox):
        self.db, self.llm, self.toolbox = db, llm, toolbox
        self.history: list[dict] = []

    async def _quick_reply(self, user_text: str, reply: str,
                           on_text: Optional[Callable[[str], Awaitable[None]]]) -> str:
        if on_text:
            await on_text(reply)
        self.history += [{"role": "user", "content": user_text}, {"role": "assistant", "content": reply}]
        return reply

    async def turn(self, user_text: str,
                   on_text: Optional[Callable[[str], Awaitable[None]]] = None,
                   on_first_token: Optional[Callable[[], None]] = None) -> str:
        # Fast path for simple music commands: run them directly, skip the LLM, reply "Done."
        u = re.sub(r"[\s.?!]+$", "", user_text.strip())
        u = re.sub(r"\s+(?:on|in|from)\s+spotify\s*$", "", u, flags=re.I)
        u = re.sub(r"^(?:hey\s+)?(?:jarvis|millie)[,\s]*", "", u, flags=re.I).strip()
        args = music_command(u)
        if args is None:
            # A bare song title the user has played before ("bohemian rhapsody")
            if self.db.find_in_music_catalog(u, exact=True):
                args = {"query": u}
        if args is not None:
            log.info("Music fast path: %s", args)
            await self.toolbox.call("play_spotify", json.dumps(args))
            return await self._quick_reply(user_text, "Done.", on_text)

        messages = [{"role": "system", "content": system_prompt(self.db)},
                    *self.history[-HISTORY_MESSAGES:],
                    {"role": "user", "content": user_text}]
        replies: list[str] = []
        first = True
        spotify_called = False
        for _ in range(MAX_TOOL_ROUNDS):
            text, calls = "", []
            async for ev in self.llm.stream(messages, self.toolbox.schemas()):
                if ev[0] == "text":
                    if first:
                        first = False
                        if on_first_token:
                            on_first_token()
                    if on_text:
                        await on_text(ev[1])
                else:
                    text, calls = ev[1], ev[2]
            if text.strip() and not spotify_called:
                replies.append(text.strip())
            if not calls:
                break
            messages.append(assistant_tool_message(text, calls))
            for c in calls:
                tool_res = await self.toolbox.call(c["name"], c["arguments"])
                messages.append(tool_result_message(c["id"], tool_res))
                if c["name"] == "play_spotify":
                    spotify_called = True

            # If Spotify was executed, finish immediately with "Done." — no extra thinking or talking over the song
            if spotify_called:
                return await self._quick_reply(user_text, "Done.", on_text)

        reply = " ".join(replies) or "Done."
        if not replies and on_text:
            await on_text(reply)
        self.history += [{"role": "user", "content": user_text}, {"role": "assistant", "content": reply}]
        return reply
