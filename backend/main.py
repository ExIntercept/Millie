"""FastAPI app: WebSocket for voice + events, REST for the workspace UI.

Run:  uvicorn backend.main:app --host 127.0.0.1 --port 8000
"""
import asyncio
import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import apps, config, files as filelib
from .assistant import Assistant
from .db import DB
from .hub import Hub
from .llm import LLM
from .research import ResearchManager
from .scheduler import Scheduler
from .tools import ToolBox
from .voice.session import Engine, VoiceSession

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("millie")

db = DB(config.DB_PATH)
hub = Hub()
llm = LLM()
research = ResearchManager(db, hub, llm)
toolbox = ToolBox(db, hub, research.start)
assistant = Assistant(db, llm, toolbox)
engine: Engine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    log.info("Loading voice models...")
    engine = await asyncio.to_thread(Engine)
    try:
        await asyncio.to_thread(engine.stt.load)
    except Exception as e:
        log.warning("STT not ready: %s (typed input and push-to-talk text still work)", e)
    scheduler = asyncio.create_task(Scheduler(db, hub).run())
    log.info("Ready: http://127.0.0.1:8000  (wake word: %s)", engine.wake.name)
    yield
    scheduler.cancel()


app = FastAPI(title="Millie", lifespan=lifespan)


# ---- WebSocket: audio in, events + speech out -----------------------------
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    hub.clients.add(ws)
    session = VoiceSession(ws, engine, assistant)
    await ws.send_json(engine.capabilities())
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes"):
                await session.on_audio(msg["bytes"])
            elif msg.get("text"):
                data = json.loads(msg["text"])
                kind = data.get("type")
                if kind == "text" and data.get("text", "").strip():
                    await session.handle_text(data["text"].strip())
                elif kind == "ptt":
                    await session.push_to_talk()
                elif kind == "stop":
                    await session.stop()
                elif kind == "audio_ended":
                    session.audio_ended()
    except Exception as e:
        log.info("websocket closed: %s", e)
    finally:
        hub.clients.discard(ws)
        await session.close()


# ---- REST ------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {**(engine.capabilities() if engine else {}), "llm_configured": bool(config.LLM_API_KEY),
            "llm_model": config.LLM_MODEL}


@app.get("/api/dashboard")
def dashboard():
    return {
        "now": time.time(),
        "tabs": db.list_tabs(),
        "notes": db.list_notes(),
        "reminders": db.list_reminders(),
        "stopwatches": db.list_stopwatches(),
        "knowledge_count": len(db.list_knowledge(limit=1000)),
        "workspace": filelib.list_workspace(),
    }


class NewTab(BaseModel):
    name: str
    link_to: list[int] = []


class Run(BaseModel):
    instructions: str


class NewNote(BaseModel):
    content: str
    title: str = ""


class NewStopwatch(BaseModel):
    label: str = "Stopwatch"


class Link(BaseModel):
    linked_tab_id: int


def _tab_or_404(tab_id: int) -> dict:
    tab = db.get_tab(tab_id)
    if not tab:
        raise HTTPException(404, "Tab not found")
    return tab


@app.post("/api/tabs")
async def create_tab(body: NewTab):
    if not body.name.strip():
        raise HTTPException(400, "Give the tab a name")
    tab_id = db.create_tab(body.name)
    for lid in body.link_to:
        if db.get_tab(lid):
            db.link_tabs(tab_id, lid)
    await hub.changed("tabs", tab_id=tab_id)
    return db.get_tab(tab_id)


@app.get("/api/tabs/{tab_id}")
def get_tab(tab_id: int):
    tab = _tab_or_404(tab_id)
    names = {t["id"]: t["name"] for t in db.list_tabs()}
    return {"tab": tab, "links": [{"id": i, "name": names.get(i, "?")} for i in tab["links"]],
            "messages": db.get_messages(tab_id), "files": db.get_files(tab_id),
            "insights": db.get_insights(tab_id), "running": research.is_running(tab_id)}


@app.delete("/api/tabs/{tab_id}")
async def delete_tab(tab_id: int):
    _tab_or_404(tab_id)
    task = research.tasks.get(tab_id)
    if task and not task.done():
        task.cancel()
    db.delete_tab(tab_id)
    await hub.changed("tabs")
    return {"ok": True}


@app.post("/api/tabs/{tab_id}/run")
async def run_research(tab_id: int, body: Run):
    _tab_or_404(tab_id)
    if not body.instructions.strip():
        raise HTTPException(400, "Tell the tab what to research")
    if not research.start(tab_id, body.instructions.strip()):
        raise HTTPException(409, "This tab is already researching")
    return {"ok": True}


@app.post("/api/tabs/{tab_id}/link")
async def link_tab(tab_id: int, body: Link):
    _tab_or_404(tab_id)
    _tab_or_404(body.linked_tab_id)
    db.link_tabs(tab_id, body.linked_tab_id)
    await hub.changed("tabs", tab_id=tab_id)
    return {"ok": True}


@app.post("/api/tabs/{tab_id}/upload")
async def upload(tab_id: int, file: UploadFile = File(...), ocr: bool = Form(False)):
    _tab_or_404(tab_id)
    safe = re.sub(r"[^\w.\-]+", "_", Path(file.filename or "upload").name)
    dest = config.UPLOAD_DIR / f"{tab_id}_{uuid.uuid4().hex[:8]}_{safe}"
    dest.write_bytes(await file.read())
    kind, text = await asyncio.to_thread(filelib.extract, dest, ocr)
    fid = db.add_file(tab_id, safe, str(dest), kind, ocr, text)
    await hub.changed("tab", tab_id=tab_id)
    return {"id": fid, "filename": safe, "kind": kind, "chars": len(text)}


@app.post("/api/notes")
async def add_note(body: NewNote):
    nid = db.add_note(body.content, body.title)
    await hub.changed("notes")
    return {"id": nid}


@app.delete("/api/notes/{nid}")
async def del_note(nid: int):
    db.delete_note(nid)
    await hub.changed("notes")
    return {"ok": True}


@app.delete("/api/reminders/{rid}")
async def del_reminder(rid: int):
    db.delete_reminder(rid)
    await hub.changed("reminders")
    return {"ok": True}


@app.post("/api/stopwatches")
async def add_stopwatch(body: NewStopwatch):
    sid = db.start_stopwatch(body.label)
    await hub.changed("stopwatches")
    return {"id": sid}


@app.post("/api/stopwatches/{sid}/stop")
async def stop_stopwatch(sid: int):
    db.stop_stopwatch(sid)
    await hub.changed("stopwatches")
    return {"ok": True}


@app.delete("/api/stopwatches/{sid}")
async def del_stopwatch(sid: int):
    db.delete_stopwatch(sid)
    await hub.changed("stopwatches")
    return {"ok": True}


# ---- Workspace Suite Endpoints ---------------------------------------------
class NewWorkspaceFile(BaseModel):
    path: str
    content: str


class ConvertFileRequest(BaseModel):
    src_path: str
    target_ext: str
    out_name: str = ""


class MergeFilesRequest(BaseModel):
    src_paths: list[str]
    out_name: str


class NewKnowledge(BaseModel):
    title: str
    content: str
    category: str = "general"
    tags: list[str] = []


class UpdateKnowledge(BaseModel):
    title: str | None = None
    content: str | None = None
    category: str | None = None
    tags: list[str] | None = None


class OpenAppRequest(BaseModel):
    app_name: str
    target: str | None = None


@app.get("/api/workspace/files")
def list_workspace_files(subdir: str = ""):
    return filelib.list_workspace(subdir)


@app.post("/api/workspace/files")
async def create_workspace_file_endpoint(body: NewWorkspaceFile):
    try:
        target = filelib.create_workspace_file(body.path, body.content)
        await hub.changed("workspace")
        rel = str(target.relative_to(filelib.get_workspace_dir()))
        return {"ok": True, "rel_path": rel, "size_bytes": target.stat().st_size}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/api/workspace/files/content")
def read_workspace_file_endpoint(path: str):
    try:
        return filelib.read_workspace_file(path)
    except FileNotFoundError:
        raise HTTPException(404, "File not found")
    except Exception as e:
        raise HTTPException(400, str(e))


@app.delete("/api/workspace/files")
async def delete_workspace_file_endpoint(path: str):
    try:
        ok = filelib.delete_workspace_file(path)
        if not ok:
            raise HTTPException(404, "File not found")
        await hub.changed("workspace")
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/workspace/upload")
async def upload_workspace_file(file: UploadFile = File(...), rel_dir: str = Form("")):
    try:
        raw = Path(file.filename or "upload").name
        safe_name = re.sub(r"[^\w.\-]+", "_", raw).strip(" .")
        if not safe_name:
            safe_name = f"upload_{uuid.uuid4().hex[:6]}"
        target_dir = filelib._safe_rel(rel_dir) if rel_dir else filelib.get_workspace_dir()
        target_dir.mkdir(parents=True, exist_ok=True)
        dest = target_dir / safe_name
        dest.write_bytes(await file.read())
        await hub.changed("workspace")
        rel = str(dest.relative_to(filelib.get_workspace_dir()))
        return {"ok": True, "rel_path": rel, "size_bytes": dest.stat().st_size}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/workspace/convert")
async def convert_workspace_file_endpoint(body: ConvertFileRequest):
    try:
        dest = filelib.convert_file(body.src_path, body.target_ext, body.out_name)
        await hub.changed("workspace")
        rel = str(dest.relative_to(filelib.get_workspace_dir()))
        return {"ok": True, "rel_path": rel, "size_bytes": dest.stat().st_size}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/workspace/merge")
async def merge_workspace_files_endpoint(body: MergeFilesRequest):
    try:
        dest = filelib.merge_files(body.src_paths, body.out_name)
        await hub.changed("workspace")
        rel = str(dest.relative_to(filelib.get_workspace_dir()))
        return {"ok": True, "rel_path": rel, "size_bytes": dest.stat().st_size}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/api/workspace/download/{rel_path:path}")
def download_workspace_file(rel_path: str):
    try:
        target = filelib._safe_rel(rel_path)
        if not target.exists() or target.is_dir():
            raise HTTPException(404, "File not found")
        return FileResponse(path=str(target), filename=target.name)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, str(e))


# ---- Knowledge Base Endpoints ----------------------------------------------
@app.get("/api/knowledge")
def list_knowledge(query: str = "", category: str = "", limit: int = 50):
    if query.strip():
        return {"items": db.search_knowledge(query.strip(), limit=limit)}
    return {"items": db.list_knowledge(category=category.strip() or None, limit=limit)}


@app.post("/api/knowledge")
async def add_knowledge_endpoint(body: NewKnowledge):
    if not body.title.strip() or not body.content.strip():
        raise HTTPException(400, "Title and content are required")
    kid = db.add_knowledge(body.title, body.content, body.category, body.tags)
    await hub.changed("knowledge")
    return {"ok": True, "id": kid}


@app.put("/api/knowledge/{kid}")
async def update_knowledge_endpoint(kid: int, body: UpdateKnowledge):
    k = db.get_knowledge(kid)
    if not k:
        raise HTTPException(404, "Knowledge entry not found")
    title = body.title if body.title is not None else k["title"]
    content = body.content if body.content is not None else k["content"]
    category = body.category if body.category is not None else k["category"]
    tags = body.tags if body.tags is not None else k["tags"]
    db.update_knowledge(kid, title, content, category, tags)
    await hub.changed("knowledge")
    return {"ok": True, "id": kid}


@app.delete("/api/knowledge/{kid}")
async def delete_knowledge_endpoint(kid: int):
    ok = db.delete_knowledge(kid)
    if not ok:
        raise HTTPException(404, "Knowledge entry not found")
    await hub.changed("knowledge")
    return {"ok": True}


# ---- Application Launcher Endpoint -----------------------------------------
@app.post("/api/apps/open")
def open_app_endpoint(body: OpenAppRequest):
    try:
        res = apps.open_application(body.app_name, body.target)
        return {"ok": True, **res}
    except Exception as e:
        raise HTTPException(400, str(e))


# ---- Spotify & Browser Lookup Endpoints ------------------------------------
class SpotifyRequest(BaseModel):
    query: str | None = None
    action: str = "play"


class BrowserLookupRequest(BaseModel):
    query: str
    engine: str = "google"
    browser: str | None = None


@app.post("/api/spotify/control")
def spotify_control_endpoint(body: SpotifyRequest):
    try:
        res = apps.play_spotify(body.query, body.action)
        return {"ok": True, **res}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/browser/open")
def browser_open_endpoint(body: BrowserLookupRequest):
    try:
        res = apps.browse_lookup(body.query, body.engine, body.browser)
        return {"ok": True, **res}
    except Exception as e:
        raise HTTPException(400, str(e))


class MusicCatalogItem(BaseModel):
    title: str
    artist: str = ""
    spotify_uri: str
    aliases: list[str] = []


@app.get("/api/music/catalog")
def get_music_catalog():
    return {"catalog": db.list_music_catalog()}


@app.post("/api/music/catalog")
def add_music_catalog(body: MusicCatalogItem):
    try:
        mid = db.add_to_music_catalog(body.title, body.artist, body.spotify_uri, body.aliases)
        return {"ok": True, "id": mid}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.delete("/api/music/catalog/{mid}")
def delete_music_catalog(mid: int):
    ok = db.delete_from_music_catalog(mid)
    return {"ok": ok}


# Frontend is served last so it never shadows /api or /ws
app.mount("/", StaticFiles(directory=str(config.FRONTEND_DIR), html=True), name="frontend")
