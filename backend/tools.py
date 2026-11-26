"""The actions Millie can take. The LLM picks a tool; this code does the work reliably.

Every tool returns a JSON string: {"ok": true, ...} or {"ok": false, "error": "..."}.
Errors are written so the LLM can turn them into a useful spoken follow-up question.
"""
import difflib
import json
import logging
from datetime import datetime
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from . import apps, config, files
from .db import DB
from .hub import Hub

log = logging.getLogger("millie.tools")


class ToolError(Exception):
    pass


def now_local() -> datetime:
    return datetime.now(ZoneInfo(config.TIMEZONE))


def parse_when(value: str) -> float:
    """ISO-8601 local time -> unix timestamp. Naive times are read in TIMEZONE."""
    try:
        dt = datetime.fromisoformat(value.strip())
    except ValueError:
        raise ToolError(f"Could not read the time '{value}'. Use ISO format like 2026-11-28T18:30:00.")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(config.TIMEZONE))
    ts = dt.timestamp()
    if ts < now_local().timestamp() - 30:
        raise ToolError(f"{value} is in the past. Ask the user what time they meant.")
    return ts


def pretty_time(ts: float) -> str:
    return datetime.fromtimestamp(ts, ZoneInfo(config.TIMEZONE)).strftime("%a %d %b %Y, %I:%M %p")


def _fn(name: str, description: str, props: dict, required: Optional[list] = None) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": props, "required": required or []}}}


S = {"type": "string"}
OPT_S = {"type": ["string", "null"]}
OPT_INT = {"type": ["integer", "null"]}
ARR_S = {"type": "array", "items": S}
OPT_ARR_S = {"type": ["array", "null"], "items": S}

SCHEMAS = [
    _fn("create_note", "Save a quick note.",
        {"content": {**S, "description": "The note text"}, "title": {**OPT_S, "description": "Optional short title"}},
        ["content"]),
    _fn("create_reminder", "Schedule a reminder that notifies the user at a time.",
        {"text": {**S, "description": "What to remind about"},
         "when": {**OPT_S, "description": "Local time, ISO 8601, e.g. 2026-11-28T18:30:00 (omit if using in_seconds)"},
         "in_seconds": {**OPT_INT, "description": "Relative duration in seconds from now, e.g. 10 for a 10s timer, 300 for 5 min"}},
        ["text"]),
    _fn("create_alarm", "Set an alarm or timer that rings.",
        {"label": {**OPT_S, "description": "Alarm/timer label"},
         "when": {**OPT_S, "description": "Local time, ISO 8601, e.g. 2026-11-28T06:30:00 (omit if using in_seconds)"},
         "in_seconds": {**OPT_INT, "description": "Relative duration in seconds from now, e.g. 10 for a 10s timer, 60 for 1 min"}}),
    _fn("start_stopwatch", "Start a stopwatch.", {"label": {**OPT_S, "description": "Optional label"}}),
    _fn("stop_stopwatch", "Stop a running stopwatch (the most recent one unless a label is given).",
        {"label": {**OPT_S, "description": "Optional label of the stopwatch to stop (or null/empty)"}}),
    _fn("create_research_tab",
        "Create a new research tab. Optionally link it to existing tabs (it will inherit their findings) "
        "and start researching right away.",
        {"name": {**S, "description": "Tab name"},
         "link_to": {"type": "array", "items": S, "description": "Names of existing tabs to connect"},
         "instructions": {**OPT_S, "description": "What to research. Leave empty to only create the tab."}}, ["name"]),
    _fn("link_tabs", "Connect a research tab to other tabs so it can use their findings.",
        {"tab": S, "link_to": {"type": "array", "items": S}}, ["tab", "link_to"]),
    _fn("research_in_tab", "Run (or continue) research in an existing tab, in the background.",
        {"tab": {**S, "description": "Tab name"}, "instructions": S}, ["tab", "instructions"]),
    _fn("get_research", "Read everything a tab has found so far.", {"tab": S}, ["tab"]),
    _fn("search_research", "Search saved findings across tabs. Use to reuse earlier research.",
        {"query": S, "tabs": {"type": "array", "items": S, "description": "Limit to these tab names (optional)"}},
        ["query"]),
    _fn("list_tabs", "List all research tabs with status and links.", {}),
    _fn("list_upcoming", "List upcoming reminders and alarms, and running stopwatches.", {}),
    # File Suite & Workspace tools
    _fn("create_workspace_file", "Create or fill a file in the workspace (e.g. Word .docx, markdown .md, .txt, .json). For Word (.docx), formatted markdown is automatically converted into headings, bullet lists, and tables.",
        {"path": {**S, "description": "Relative file path, e.g. 'summary.docx', 'notes.md', 'reports/q1.docx'"},
         "content": {**S, "description": "Text or markdown content to write into the file"}},
        ["path", "content"]),
    _fn("read_workspace_file", "Read text content and metadata of a file in the workspace.",
        {"path": {**S, "description": "Relative path of file to read"}},
        ["path"]),
    _fn("list_workspace_files", "List files and folders in the workspace directory tree.",
        {"subdir": {**OPT_S, "description": "Optional subfolder to list"}}),
    _fn("convert_workspace_file", "Convert an existing workspace file to another format (.docx, .md, .txt, .html, .pdf, .png, .jpg).",
        {"src_path": {**S, "description": "Relative path of source file"},
         "target_ext": {**S, "description": "Target extension, e.g. '.docx', '.pdf', '.md', '.html', '.txt'"},
         "out_name": {**OPT_S, "description": "Optional custom output filename"}},
        ["src_path", "target_ext"]),
    _fn("merge_workspace_files", "Merge multiple workspace files (multiple PDFs, Word documents, images, or text/markdown) into one file.",
        {"src_paths": {**ARR_S, "description": "List of relative paths of files to merge"},
         "out_name": {**S, "description": "Filename for merged output, e.g. 'combined.pdf' or 'report.docx'"}},
        ["src_paths", "out_name"]),
    # Knowledge Base tools
    _fn("save_knowledge", "Save a learned fact, user preference, instruction, or background detail into the persistent knowledge base.",
        {"title": {**S, "description": "Short title or concept name"},
         "content": {**S, "description": "The fact, detail, or preference to remember"},
         "category": {**OPT_S, "description": "Category: preference, project, personal, work, or general"},
         "tags": {**OPT_ARR_S, "description": "Optional list of tags"}},
        ["title", "content"]),
    _fn("search_knowledge", "Search the persistent knowledge base for facts, preferences, and learned context.",
        {"query": {**S, "description": "Search query keywords"}},
        ["query"]),
    # App opener & Media / Web lookup
    _fn("open_app", "Launch an application or URL on the user's Mac (e.g. Spotify, Chrome, Safari, VS Code, Finder, Notes, Terminal).",
        {"app_name": {**S, "description": "Application name or alias (e.g. 'Spotify', 'Chrome', 'VS Code', 'Finder', 'Notes') or URL"},
         "path_or_url": {**OPT_S, "description": "Optional file path or URL to open with the app"}},
        ["app_name"]),
    _fn("play_spotify", "Control Spotify playback (play, pause, next, previous) or search and play specific songs.",
        {"action": {**OPT_S, "description": "Playback control action: 'play' (to resume/unpause), 'pause' (to pause), 'next' (to skip to next song), 'previous' (to go back)",
                    "enum": ["play", "pause", "next", "previous"]},
         "query": {**OPT_S, "description": "Specific song title, artist, or playlist to search and play. Leave EMPTY when pausing, resuming, or skipping."}}),
    _fn("browse_lookup", "Look up a search query, topic, question, or URL in a new browser tab.",
        {"query": {**S, "description": "Search query or URL (e.g. 'quantum computing advances', 'https://github.com')"},
         "engine": {**OPT_S, "description": "Search engine: 'google' (default), 'youtube', 'duckduckgo', or 'github'"},
         "browser": {**OPT_S, "description": "Optional browser: 'Google Chrome', 'Safari', etc."}},
        ["query"]),
]




class ToolBox:
    def __init__(self, db: DB, hub: Hub, start_research: Callable[[int, str], bool]):
        self.db, self.hub, self.start_research = db, hub, start_research

    def schemas(self) -> list[dict]:
        return SCHEMAS

    # ---- dispatch ------------------------------------------------------
    async def call(self, name: str, raw_args: str) -> str:
        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            return json.dumps({"ok": False, "error": "Arguments were not valid JSON."})
        fn = getattr(self, f"t_{name}", None)
        if fn is None:
            return json.dumps({"ok": False, "error": f"Unknown tool {name}."})
        try:
            result = await fn(**args)
            log.info("tool %s(%s) -> ok", name, args)
            return json.dumps({"ok": True, **result}, default=str)
        except ToolError as e:
            log.info("tool %s(%s) -> %s", name, args, e)
            return json.dumps({"ok": False, "error": str(e)})
        except TypeError as e:  # wrong/missing arguments from the model
            return json.dumps({"ok": False, "error": f"Bad arguments for {name}: {e}"})
        except Exception as e:
            log.exception("tool %s crashed", name)
            return json.dumps({"ok": False, "error": f"{name} failed: {e}"})

    # ---- helpers -------------------------------------------------------
    def resolve_tab(self, ref) -> dict:
        tabs = self.db.list_tabs()
        if not tabs:
            raise ToolError("There are no research tabs yet.")
        s = str(ref).strip().lower()
        names = ", ".join(f"'{t['name']}'" for t in tabs)
        exact = [t for t in tabs if t["name"].lower() == s or str(t["id"]) == s]
        if len(exact) == 1:
            return exact[0]
        loose = [t for t in tabs if s in t["name"].lower() or t["name"].lower() in s]
        if len(loose) == 1:
            return loose[0]
        close = difflib.get_close_matches(s, [t["name"].lower() for t in tabs], n=2, cutoff=0.6)
        if len(close) == 1:
            return next(t for t in tabs if t["name"].lower() == close[0])
        raise ToolError(f"No single tab matches '{ref}'. Existing tabs: {names}. Ask the user which one.")

    # ---- tools ---------------------------------------------------------
    async def t_create_note(self, content: str, title: Optional[str] = "") -> dict:
        nid = self.db.add_note(content, title or "")
        await self.hub.changed("notes")
        return {"note_id": nid}

    async def t_create_reminder(self, text: str, when: Optional[str] = None, in_seconds: Optional[int] = None) -> dict:
        if in_seconds is not None and in_seconds > 0:
            ts = now_local().timestamp() + in_seconds
        elif when:
            ts = parse_when(when)
        else:
            raise ToolError("Please specify either when or in_seconds for the reminder.")
        rid = self.db.add_reminder(text, ts, "reminder")
        await self.hub.changed("reminders")
        return {"reminder_id": rid, "due": pretty_time(ts)}

    async def t_create_alarm(self, when: Optional[str] = None, label: Optional[str] = "Alarm", in_seconds: Optional[int] = None) -> dict:
        if in_seconds is not None and in_seconds > 0:
            ts = now_local().timestamp() + in_seconds
        elif when:
            ts = parse_when(when)
        else:
            raise ToolError("Please specify either when or in_seconds for the alarm.")
        rid = self.db.add_reminder(label or "Alarm", ts, "alarm")
        await self.hub.changed("reminders")
        return {"alarm_id": rid, "rings": pretty_time(ts)}

    async def t_start_stopwatch(self, label: Optional[str] = "Stopwatch") -> dict:
        sid = self.db.start_stopwatch(label or "Stopwatch")
        await self.hub.changed("stopwatches")
        return {"stopwatch_id": sid}

    async def t_stop_stopwatch(self, label: Optional[str] = None) -> dict:
        running = [s for s in self.db.list_stopwatches() if s["stopped_at"] is None]
        if label and label.strip():
            running = [s for s in running if label.strip().lower() in s["label"].lower()] or running
        if not running:
            raise ToolError("No stopwatch is running.")
        sw = running[0]
        self.db.stop_stopwatch(sw["id"])
        await self.hub.changed("stopwatches")
        import time
        return {"label": sw["label"], "elapsed_seconds": round(time.time() - sw["started_at"], 1)}

    async def t_create_research_tab(self, name: str, link_to: Optional[list] = None, instructions: str = "") -> dict:
        targets = [self.resolve_tab(x) for x in (link_to or [])]   # fail early, before creating anything
        tab_id = self.db.create_tab(name)
        for t in targets:
            self.db.link_tabs(tab_id, t["id"])
        await self.hub.changed("tabs", tab_id=tab_id, open=True)
        started = False
        if instructions.strip():
            started = self.start_research(tab_id, instructions.strip())
        return {"tab_id": tab_id, "name": name, "linked_to": [t["name"] for t in targets],
                "research_started": started}

    async def t_link_tabs(self, tab: str, link_to: list) -> dict:
        src = self.resolve_tab(tab)
        targets = [self.resolve_tab(x) for x in link_to]
        for t in targets:
            self.db.link_tabs(src["id"], t["id"])
        await self.hub.changed("tabs", tab_id=src["id"])
        return {"tab": src["name"], "linked_to": [t["name"] for t in targets]}

    async def t_research_in_tab(self, tab: str, instructions: str) -> dict:
        t = self.resolve_tab(tab)
        if not self.start_research(t["id"], instructions):
            raise ToolError(f"Tab '{t['name']}' is already researching. Wait for it to finish.")
        return {"tab": t["name"], "research_started": True}

    async def t_get_research(self, tab: str) -> dict:
        t = self.resolve_tab(tab)
        items = self.db.get_insights(t["id"])
        return {"tab": t["name"], "status": t["status"],
                "insights": [{"title": i["title"], "kind": i["kind"], "content": i["content"][:1500]} for i in items]}

    async def t_search_research(self, query: str, tabs: Optional[list] = None) -> dict:
        ids = [self.resolve_tab(x)["id"] for x in tabs] if tabs else None
        hits = self.db.search_insights(query, ids)
        return {"results": [{"tab": h["tab_name"], "title": h["title"], "content": h["content"][:1200],
                             "sources": h["sources"]} for h in hits]}

    async def t_list_tabs(self) -> dict:
        names = {t["id"]: t["name"] for t in self.db.list_tabs()}
        return {"tabs": [{"name": t["name"], "status": t["status"],
                          "linked_to": [names.get(i, "?") for i in t["links"]]} for t in self.db.list_tabs()]}

    async def t_list_upcoming(self) -> dict:
        rem = [r for r in self.db.list_reminders() if not r["fired"]]
        sws = [s for s in self.db.list_stopwatches() if s["stopped_at"] is None]
        return {"upcoming": [{"kind": r["kind"], "text": r["text"], "due": pretty_time(r["due_at"])} for r in rem],
                "running_stopwatches": [s["label"] for s in sws]}

    # Workspace tools
    async def t_create_workspace_file(self, path: str, content: str) -> dict:
        try:
            target = files.create_workspace_file(path, content)
            await self.hub.changed("workspace")
            rel = str(target.relative_to(files.get_workspace_dir()))
            return {"file": rel, "size_bytes": target.stat().st_size, "message": f"Created file {rel} successfully."}
        except Exception as e:
            raise ToolError(f"Failed to create workspace file: {e}")

    async def t_read_workspace_file(self, path: str) -> dict:
        try:
            return files.read_workspace_file(path)
        except Exception as e:
            raise ToolError(f"Failed to read file '{path}': {e}")

    async def t_list_workspace_files(self, subdir: Optional[str] = "") -> dict:
        try:
            res = files.list_workspace(subdir or "")
            return {"files": [f["rel_path"] for f in res.get("files", [])]}
        except Exception as e:
            raise ToolError(f"Failed to list files: {e}")

    async def t_convert_workspace_file(self, src_path: str, target_ext: str, out_name: Optional[str] = None) -> dict:
        try:
            dest = files.convert_file(src_path, target_ext, out_name or "")
            await self.hub.changed("workspace")
            rel = str(dest.relative_to(files.get_workspace_dir()))
            return {"converted_file": rel, "size_bytes": dest.stat().st_size}
        except Exception as e:
            raise ToolError(f"Failed to convert file: {e}")

    async def t_merge_workspace_files(self, src_paths: list, out_name: str) -> dict:
        try:
            dest = files.merge_files(src_paths, out_name)
            await self.hub.changed("workspace")
            rel = str(dest.relative_to(files.get_workspace_dir()))
            return {"merged_file": rel, "size_bytes": dest.stat().st_size}
        except Exception as e:
            raise ToolError(f"Failed to merge files: {e}")

    # Knowledge Base tools
    async def t_save_knowledge(self, title: str, content: str, category: Optional[str] = "general", tags: Optional[list] = None) -> dict:
        try:
            kid = self.db.add_knowledge(title, content, category or "general", tags or [])
            await self.hub.changed("knowledge")
            return {"knowledge_id": kid, "title": title, "message": f"Saved '{title}' to knowledge base."}
        except Exception as e:
            raise ToolError(f"Failed to save knowledge: {e}")

    async def t_search_knowledge(self, query: str) -> dict:
        try:
            hits = self.db.search_knowledge(query)
            return {"query": query, "results": hits}
        except Exception as e:
            raise ToolError(f"Failed to search knowledge base: {e}")

    # App Opener
    async def t_open_app(self, app_name: str, path_or_url: Optional[str] = None) -> dict:
        try:
            res = apps.open_application(app_name, path_or_url)
            return {"status": "opened", "app": res["app"], "target": res.get("target")}
        except Exception as e:
            raise ToolError(f"Could not open application: {e}")

    # Media & Browser Tools
    async def t_play_spotify(self, query: Optional[str] = None, action: Optional[str] = "play") -> dict:
        try:
            return apps.play_spotify(query, action or "play")
        except Exception as e:
            raise ToolError(f"Spotify command failed: {e}")

    async def t_browse_lookup(self, query: str, engine: Optional[str] = "google", browser: Optional[str] = None) -> dict:
        try:
            return apps.browse_lookup(query, engine or "google", browser)
        except Exception as e:
            raise ToolError(f"Browser lookup failed: {e}")


