"""Research tabs: each tab runs its own background agent with its own context.

Context for a run = the tab's uploaded files (as text) + findings from linked tabs
+ this tab's own earlier findings. Findings are saved as insights (searchable with
FTS5), which is how research is reused across tabs later.
"""
import asyncio
import json
import logging
import re

from . import config
from .db import DB
from .hub import Hub
from .llm import LLM, assistant_tool_message, tool_result_message
from .tools import _fn, S

log = logging.getLogger("millie.research")

FINISH = _fn("finish", "Call when research is complete, with a concise summary of what was learned.",
             {"summary": {**S, "description": "5-10 sentence summary of the key conclusions"}}, ["summary"])
TOOLS = [
    _fn("web_search", "Search the web. Returns titles, URLs and snippets.", {"query": S}, ["query"]),
    _fn("fetch_page", "Read the text of a web page (first ~6000 characters).", {"url": S}, ["url"]),
    _fn("save_insight", "Save one distinct finding to this tab's notebook.",
        {"title": S, "content": {**S, "description": "The finding, with specifics (names, years, numbers)"},
         "sources": {"type": "array", "items": S, "description": "URLs backing this finding"}},
        ["title", "content"]),
    FINISH,
]

FILE_CHARS = 8000
LINKED_CHARS = 6000


def _search_sync(query: str, n: int = 6) -> list[dict]:
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS  # older package name
    with DDGS() as d:
        return list(d.text(query, max_results=n))


async def fetch_page_text(url: str, limit: int = 6000) -> str:
    import httpx
    async with httpx.AsyncClient(follow_redirects=True, timeout=15,
                                 headers={"User-Agent": "Mozilla/5.0 (MillieResearch)"}) as client:
        r = await client.get(url)
        r.raise_for_status()
    html = r.text
    html = re.sub(r"(?is)<(script|style|noscript|svg|nav|footer|header)[^>]*>.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = re.sub(r"&nbsp;|&amp;|&quot;|&#39;", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:limit]


def build_context(db: DB, tab_id: int) -> str:
    tab = db.get_tab(tab_id)
    parts = []
    for f in db.get_files(tab_id):
        if f["extracted_text"].strip():
            parts.append(f"### Uploaded file: {f['filename']}\n{f['extracted_text'][:FILE_CHARS]}")
    for lid in tab["links"]:
        lt = db.get_tab(lid)
        if not lt:
            continue
        ins = sorted(db.get_insights(lid), key=lambda i: i["kind"] != "summary")  # summaries first
        body = "\n".join(f"- {i['title']}: {i['content']}" for i in ins)[:LINKED_CHARS]
        if body:
            parts.append(f"### Findings from linked tab '{lt['name']}'\n{body}")
    own = db.get_insights(tab_id)
    if own:
        parts.append("### Already saved in this tab (do not repeat)\n" +
                     "\n".join(f"- {i['title']}" for i in own)[:LINKED_CHARS])
    return "\n\n".join(parts) or "(no prior context)"


class ResearchManager:
    def __init__(self, db: DB, hub: Hub, llm: LLM):
        self.db, self.hub, self.llm = db, hub, llm
        self.tasks: dict[int, asyncio.Task] = {}

    def is_running(self, tab_id: int) -> bool:
        t = self.tasks.get(tab_id)
        return bool(t and not t.done())

    def start(self, tab_id: int, instructions: str) -> bool:
        """Begin a background run. Returns False if this tab is already busy."""
        if self.is_running(tab_id):
            return False
        self.db.add_message(tab_id, "user", instructions)
        self.db.set_tab_status(tab_id, "running")
        self.tasks[tab_id] = asyncio.get_running_loop().create_task(self._run(tab_id, instructions))
        return True

    async def _log(self, tab_id: int, text: str, role: str = "log") -> None:
        self.db.add_message(tab_id, role, text)
        await self.hub.changed("tab", tab_id=tab_id)

    async def _run(self, tab_id: int, instructions: str) -> None:
        tab = self.db.get_tab(tab_id)
        await self.hub.changed("tabs", tab_id=tab_id)
        system = (
            f"You are a careful research agent working in the tab '{tab['name']}'. "
            "Available tools: web_search (to find URLs), fetch_page (to read page text), save_insight (to save key findings), and finish (when done). "
            "Search the web for good sources, read the promising ones with fetch_page, "
            "and save each distinct finding with save_insight, including specifics (names, years, numbers) and source URLs. "
            "Do not repeat findings that are already saved. For literature surveys, organise by theme and note gaps. "
            "When done, call finish with a concise summary. Be efficient: you have a limited number of steps.\n\n"
            f"CONTEXT\n{build_context(self.db, tab_id)}")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": instructions}]
        summary = None
        try:
            for step in range(config.RESEARCH_MAX_STEPS):
                last = step == config.RESEARCH_MAX_STEPS - 1
                if last:
                    messages.append({"role": "user", "content": "You are out of steps. Call finish now."})
                text, calls = await self.llm.run(messages, [FINISH] if last else TOOLS)
                if not calls:
                    summary = text or "(no summary produced)"
                    break
                messages.append(assistant_tool_message(text, calls))
                for c in calls:
                    result, final = await self._exec(tab_id, c)
                    messages.append(tool_result_message(c["id"], result))
                    summary = final or summary
                if summary:
                    break
            self.db.add_insight(tab_id, "Summary of this run", summary or "(no summary produced)", kind="summary")
            self.db.set_tab_status(tab_id, "done")
            await self._log(tab_id, summary or "(no summary produced)", role="agent")
        except asyncio.CancelledError:
            self.db.set_tab_status(tab_id, "idle")
            raise
        except Exception as e:
            log.exception("research failed in tab %s", tab_id)
            self.db.set_tab_status(tab_id, "error")
            await self._log(tab_id, f"Research stopped: {e}")
        finally:
            await self.hub.changed("tabs", tab_id=tab_id)
            await self.hub.broadcast({"type": "research_done", "tab_id": tab_id, "name": tab["name"]})

    async def _exec(self, tab_id: int, call: dict) -> tuple[str, str | None]:
        """Run one research tool. Returns (result_text, final_summary_or_None)."""
        try:
            args = json.loads(call["arguments"] or "{}")
        except json.JSONDecodeError:
            return "Invalid JSON arguments.", None
        name = call["name"]
        # Some models call these tools by slightly different names; accept both.
        try:
            if name in ("web_search", "search_web"):
                q = args.get("query") or args.get("q") or ""
                await self._log(tab_id, f"Searching: {q}")
                rows = await asyncio.to_thread(_search_sync, q)
                return "\n".join(f"{r.get('title')}\n{r.get('href')}\n{r.get('body')}\n" for r in rows) or "No results.", None
            if name in ("fetch_page", "web_fetch_page"):
                u = args.get("url") or args.get("link") or ""
                await self._log(tab_id, f"Reading: {u}")
                return await fetch_page_text(u), None
            if name == "save_insight":
                self.db.add_insight(tab_id, args["title"], args["content"], args.get("sources", []))
                await self._log(tab_id, f"Saved finding: {args['title']}")
                await self.hub.changed("insights", tab_id=tab_id)
                return "Saved.", None
            if name == "finish":
                return "Finished.", args.get("summary", "")
            return f"Unknown tool {name}.", None
        except Exception as e:  # a bad page or a search hiccup should not kill the run
            return f"Tool error: {e}", None
