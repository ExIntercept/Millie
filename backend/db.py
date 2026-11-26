"""SQLite storage. One local file, FTS5 for searching research across tabs.

Everything is synchronous and guarded by a lock; operations are tiny, so this is
simpler and more reliable than async drivers for a single-user local app.
"""
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS tabs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'idle',      -- idle | running | done | error
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS tab_links (
    tab_id INTEGER NOT NULL,
    linked_tab_id INTEGER NOT NULL,
    PRIMARY KEY (tab_id, linked_tab_id)
);
CREATE TABLE IF NOT EXISTS messages (          -- per-tab log: user instructions, agent steps
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tab_id INTEGER NOT NULL,
    role TEXT NOT NULL,                        -- user | agent | log
    content TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tab_id INTEGER NOT NULL,
    filename TEXT NOT NULL,
    path TEXT NOT NULL,
    kind TEXT NOT NULL,                        -- text | image | other
    ocr INTEGER NOT NULL DEFAULT 0,
    extracted_text TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tab_id INTEGER NOT NULL,
    kind TEXT NOT NULL DEFAULT 'finding',      -- finding | summary
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    sources TEXT NOT NULL DEFAULT '[]',        -- JSON list of URLs
    created_at REAL NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS insights_fts USING fts5(
    title, content, content='insights', content_rowid='id'
);
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS reminders (         -- also used for alarms (kind='alarm')
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL DEFAULT 'reminder',
    text TEXT NOT NULL,
    due_at REAL NOT NULL,                      -- unix timestamp
    fired INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS stopwatches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT NOT NULL,
    started_at REAL NOT NULL,
    stopped_at REAL
);
CREATE TABLE IF NOT EXISTS knowledge_base (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'general',
    tags TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
    title, content, category, content='knowledge_base', content_rowid='id'
);
CREATE TABLE IF NOT EXISTS music_catalog (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    artist TEXT NOT NULL DEFAULT '',
    spotify_uri TEXT NOT NULL,
    aliases TEXT NOT NULL DEFAULT '[]',
    play_count INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_music_title ON music_catalog(title);
"""


class DB:
    def __init__(self, path: Path | str = ":memory:"):
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()
            cnt = self._one("SELECT count(*) as c FROM music_catalog")
            if cnt and cnt["c"] == 0:
                self.seed_default_music_catalog()

    # ---- helpers -------------------------------------------------------
    def _exec(self, sql: str, args: tuple = ()) -> int:
        with self.lock:
            cur = self.conn.execute(sql, args)
            self.conn.commit()
            return cur.lastrowid

    def _all(self, sql: str, args: tuple = ()) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def _one(self, sql: str, args: tuple = ()) -> Optional[dict]:
        rows = self._all(sql, args)
        return rows[0] if rows else None

    # ---- tabs ----------------------------------------------------------
    def create_tab(self, name: str) -> int:
        return self._exec("INSERT INTO tabs(name, created_at) VALUES (?,?)", (name.strip(), time.time()))

    def get_tab(self, tab_id: int) -> Optional[dict]:
        t = self._one("SELECT * FROM tabs WHERE id=?", (tab_id,))
        if t:
            t["links"] = self.get_links(tab_id)
        return t

    def list_tabs(self) -> list[dict]:
        tabs = self._all("SELECT * FROM tabs ORDER BY id")
        for t in tabs:
            t["links"] = self.get_links(t["id"])
        return tabs

    def set_tab_status(self, tab_id: int, status: str) -> None:
        self._exec("UPDATE tabs SET status=? WHERE id=?", (status, tab_id))

    def rename_tab(self, tab_id: int, name: str) -> None:
        self._exec("UPDATE tabs SET name=? WHERE id=?", (name.strip(), tab_id))

    def delete_tab(self, tab_id: int) -> None:
        for r in self._all("SELECT id, title, content FROM insights WHERE tab_id=?", (tab_id,)):
            self._fts_delete(r["id"], r["title"], r["content"])
        for table in ("messages", "files", "insights"):
            self._exec(f"DELETE FROM {table} WHERE tab_id=?", (tab_id,))
        self._exec("DELETE FROM tab_links WHERE tab_id=? OR linked_tab_id=?", (tab_id, tab_id))
        self._exec("DELETE FROM tabs WHERE id=?", (tab_id,))

    def link_tabs(self, tab_id: int, linked_tab_id: int) -> None:
        if tab_id == linked_tab_id:
            return
        self._exec("INSERT OR IGNORE INTO tab_links(tab_id, linked_tab_id) VALUES (?,?)", (tab_id, linked_tab_id))

    def get_links(self, tab_id: int) -> list[int]:
        return [r["linked_tab_id"] for r in self._all(
            "SELECT linked_tab_id FROM tab_links WHERE tab_id=? ORDER BY linked_tab_id", (tab_id,))]

    # ---- messages ------------------------------------------------------
    def add_message(self, tab_id: int, role: str, content: str) -> int:
        return self._exec("INSERT INTO messages(tab_id, role, content, created_at) VALUES (?,?,?,?)",
                          (tab_id, role, content, time.time()))

    def get_messages(self, tab_id: int, limit: int = 200) -> list[dict]:
        return self._all("SELECT * FROM (SELECT * FROM messages WHERE tab_id=? ORDER BY id DESC LIMIT ?) ORDER BY id",
                         (tab_id, limit))

    # ---- files ---------------------------------------------------------
    def add_file(self, tab_id: int, filename: str, path: str, kind: str, ocr: bool, text: str) -> int:
        return self._exec(
            "INSERT INTO files(tab_id, filename, path, kind, ocr, extracted_text, created_at) VALUES (?,?,?,?,?,?,?)",
            (tab_id, filename, path, kind, int(ocr), text, time.time()))

    def get_files(self, tab_id: int) -> list[dict]:
        return self._all("SELECT * FROM files WHERE tab_id=? ORDER BY id", (tab_id,))

    # ---- insights (with full-text search) -------------------------------
    def add_insight(self, tab_id: int, title: str, content: str, sources: Optional[list] = None,
                    kind: str = "finding") -> int:
        rid = self._exec(
            "INSERT INTO insights(tab_id, kind, title, content, sources, created_at) VALUES (?,?,?,?,?,?)",
            (tab_id, kind, title, content, json.dumps(sources or []), time.time()))
        self._exec("INSERT INTO insights_fts(rowid, title, content) VALUES (?,?,?)", (rid, title, content))
        return rid

    def _fts_delete(self, rid: int, title: str, content: str) -> None:
        self._exec("INSERT INTO insights_fts(insights_fts, rowid, title, content) VALUES('delete',?,?,?)",
                   (rid, title, content))

    def get_insights(self, tab_id: int) -> list[dict]:
        rows = self._all("SELECT * FROM insights WHERE tab_id=? ORDER BY id", (tab_id,))
        for r in rows:
            r["sources"] = json.loads(r["sources"])
        return rows

    def search_insights(self, query: str, tab_ids: Optional[list[int]] = None, limit: int = 8) -> list[dict]:
        terms = [w for w in "".join(c if c.isalnum() else " " for c in query).split() if len(w) > 1]
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms)
        sql = ("SELECT i.*, t.name AS tab_name FROM insights_fts f "
               "JOIN insights i ON i.id = f.rowid JOIN tabs t ON t.id = i.tab_id "
               "WHERE insights_fts MATCH ?")
        args: list[Any] = [match]
        if tab_ids:
            sql += f" AND i.tab_id IN ({','.join('?' * len(tab_ids))})"
            args += tab_ids
        sql += " ORDER BY rank LIMIT ?"
        args.append(limit)
        rows = self._all(sql, tuple(args))
        for r in rows:
            r["sources"] = json.loads(r["sources"])
        return rows

    # ---- notes ---------------------------------------------------------
    def add_note(self, content: str, title: str = "") -> int:
        return self._exec("INSERT INTO notes(title, content, created_at) VALUES (?,?,?)",
                          (title, content, time.time()))

    def list_notes(self) -> list[dict]:
        return self._all("SELECT * FROM notes ORDER BY id DESC")

    def delete_note(self, note_id: int) -> None:
        self._exec("DELETE FROM notes WHERE id=?", (note_id,))

    # ---- reminders / alarms -------------------------------------------
    def add_reminder(self, text: str, due_at: float, kind: str = "reminder") -> int:
        return self._exec("INSERT INTO reminders(kind, text, due_at, created_at) VALUES (?,?,?,?)",
                          (kind, text, due_at, time.time()))

    def list_reminders(self) -> list[dict]:
        return self._all("SELECT * FROM reminders ORDER BY fired, due_at")

    def due_reminders(self, now: float) -> list[dict]:
        return self._all("SELECT * FROM reminders WHERE fired=0 AND due_at<=?", (now,))

    def mark_fired(self, rid: int) -> None:
        self._exec("UPDATE reminders SET fired=1 WHERE id=?", (rid,))

    def delete_reminder(self, rid: int) -> None:
        self._exec("DELETE FROM reminders WHERE id=?", (rid,))

    # ---- stopwatches ---------------------------------------------------
    def start_stopwatch(self, label: str) -> int:
        return self._exec("INSERT INTO stopwatches(label, started_at) VALUES (?,?)", (label, time.time()))

    def stop_stopwatch(self, sid: int) -> None:
        self._exec("UPDATE stopwatches SET stopped_at=? WHERE id=? AND stopped_at IS NULL", (time.time(), sid))

    def list_stopwatches(self) -> list[dict]:
        return self._all("SELECT * FROM stopwatches ORDER BY id DESC LIMIT 20")

    def delete_stopwatch(self, sid: int) -> None:
        self._exec("DELETE FROM stopwatches WHERE id=?", (sid,))

    # ---- knowledge base (persistent memory) ----------------------------
    def add_knowledge(self, title: str, content: str, category: str = "general",
                      tags: Optional[list] = None) -> int:
        now = time.time()
        kid = self._exec(
            "INSERT INTO knowledge_base(title, content, category, tags, created_at, updated_at) VALUES (?,?,?,?,?,?)",
            (title.strip(), content.strip(), (category or "general").strip().lower(),
             json.dumps(tags or []), now, now))
        self._exec("INSERT INTO knowledge_fts(rowid, title, content, category) VALUES (?,?,?,?)",
                   (kid, title.strip(), content.strip(), (category or "general").strip().lower()))
        return kid

    def update_knowledge(self, kid: int, title: Optional[str] = None, content: Optional[str] = None,
                         category: Optional[str] = None, tags: Optional[list] = None) -> None:
        existing = self.get_knowledge(kid)
        if not existing:
            return
        t = title.strip() if title is not None else existing["title"]
        c = content.strip() if content is not None else existing["content"]
        cat = (category or existing["category"]).strip().lower()
        t_list = tags if tags is not None else existing["tags"]
        now = time.time()
        self._exec(
            "UPDATE knowledge_base SET title=?, content=?, category=?, tags=?, updated_at=? WHERE id=?",
            (t, c, cat, json.dumps(t_list), now, kid))
        self._exec("INSERT INTO knowledge_fts(knowledge_fts, rowid, title, content, category) VALUES('delete',?,?,?,?)",
                   (kid, existing["title"], existing["content"], existing["category"]))
        self._exec("INSERT INTO knowledge_fts(rowid, title, content, category) VALUES (?,?,?,?)",
                   (kid, t, c, cat))

    def delete_knowledge(self, kid: int) -> bool:
        existing = self.get_knowledge(kid)
        if existing:
            self._exec("INSERT INTO knowledge_fts(knowledge_fts, rowid, title, content, category) VALUES('delete',?,?,?,?)",
                       (kid, existing["title"], existing["content"], existing["category"]))
            self._exec("DELETE FROM knowledge_base WHERE id=?", (kid,))
            return True
        return False

    def get_knowledge(self, kid: int) -> Optional[dict]:
        k = self._one("SELECT * FROM knowledge_base WHERE id=?", (kid,))
        if k:
            k["tags"] = json.loads(k["tags"])
        return k

    def list_knowledge(self, category: Optional[str] = None, limit: int = 100) -> list[dict]:
        sql = "SELECT * FROM knowledge_base"
        args: list = []
        if category and category.strip():
            sql += " WHERE category=?"
            args.append(category.strip().lower())
        sql += " ORDER BY updated_at DESC LIMIT ?"
        args.append(limit)
        rows = self._all(sql, tuple(args))
        for r in rows:
            r["tags"] = json.loads(r["tags"])
        return rows

    def search_knowledge(self, query: str, limit: int = 8) -> list[dict]:
        terms = [w for w in "".join(c if c.isalnum() else " " for c in query).split() if len(w) > 1]
        if not terms:
            return self.list_knowledge(limit=limit)
        match = " OR ".join(f'"{t}"' for t in terms)
        sql = ("SELECT k.* FROM knowledge_fts f "
               "JOIN knowledge_base k ON k.id = f.rowid "
               "WHERE knowledge_fts MATCH ? ORDER BY rank LIMIT ?")
        rows = self._all(sql, (match, limit))
        for r in rows:
            r["tags"] = json.loads(r["tags"])
        return rows

    # ---- music catalog --------------------------------------------------
    def add_to_music_catalog(self, title: str, artist: str, spotify_uri: str, aliases: Optional[list[str]] = None) -> int:
        now = time.time()
        aliases_json = json.dumps([a.lower().strip() for a in (aliases or []) if a.strip()])
        # Check if already exists by URI or title
        existing = self._one("SELECT id, play_count, aliases FROM music_catalog WHERE spotify_uri=? OR lower(title)=lower(?)",
                             (spotify_uri.strip(), title.strip()))
        if existing:
            mid = existing["id"]
            count = existing["play_count"] + 1
            # Merge aliases
            curr_aliases = set(json.loads(existing["aliases"] or "[]"))
            curr_aliases.update([a.lower().strip() for a in (aliases or []) if a.strip()])
            curr_aliases.add(title.lower().strip())
            if artist:
                curr_aliases.add(f"{title} {artist}".lower().strip())
                curr_aliases.add(f"{title} by {artist}".lower().strip())
            self._exec("UPDATE music_catalog SET play_count=?, updated_at=?, aliases=? WHERE id=?",
                       (count, now, json.dumps(list(curr_aliases)), mid))
            return mid

        # Create new
        all_aliases = set([a.lower().strip() for a in (aliases or []) if a.strip()])
        all_aliases.add(title.lower().strip())
        if artist:
            all_aliases.add(f"{title} {artist}".lower().strip())
            all_aliases.add(f"{title} by {artist}".lower().strip())
        return self._exec(
            "INSERT INTO music_catalog (title, artist, spotify_uri, aliases, play_count, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 1, ?, ?)",
            (title.strip(), (artist or "").strip(), spotify_uri.strip(), json.dumps(list(all_aliases)), now, now)
        )

    def find_in_music_catalog(self, query: str, exact: bool = False) -> Optional[dict]:
        """Find a song in the catalog by title, artist or alias.

        exact=True only accepts a whole-string match. Use it when checking free speech,
        so "start a stopwatch called study" doesn't match the "study" playlist alias.
        """
        clean = (query or "").lower().strip()
        if not clean:
            return None
        rows = self.list_music_catalog(limit=300)
        if exact:
            for r in rows:
                if clean == r["title"].lower() or clean in r.get("aliases", []):
                    return r
            return None
        # 1. Exact match on title or artist
        for r in rows:
            if clean == r["title"].lower() or (r["artist"] and clean == r["artist"].lower()):
                self._exec("UPDATE music_catalog SET play_count=play_count+1, updated_at=? WHERE id=?", (time.time(), r["id"]))
                return r

        # 2. Match in aliases
        for r in rows:
            aliases = r.get("aliases", [])
            for a in aliases:
                if a == clean or (len(a) > 3 and a in clean) or (len(clean) > 3 and clean in a):
                    self._exec("UPDATE music_catalog SET play_count=play_count+1, updated_at=? WHERE id=?", (time.time(), r["id"]))
                    return r

        # 3. Substring match on title
        for r in rows:
            t = r["title"].lower()
            if (len(clean) > 3 and clean in t) or (len(t) > 3 and t in clean):
                self._exec("UPDATE music_catalog SET play_count=play_count+1, updated_at=? WHERE id=?", (time.time(), r["id"]))
                return r

        return None

    def list_music_catalog(self, limit: int = 100) -> list[dict]:
        rows = self._all("SELECT * FROM music_catalog ORDER BY play_count DESC, updated_at DESC LIMIT ?", (limit,))
        for r in rows:
            r["aliases"] = json.loads(r["aliases"] or "[]")
        return rows

    def delete_from_music_catalog(self, mid: int) -> bool:
        return self._exec("DELETE FROM music_catalog WHERE id=?", (mid,)) > 0

    def seed_default_music_catalog(self) -> None:
        """Seed a few starter tracks and playlists so music commands work on first run."""
        defaults = [
            ("Sweet Child O' Mine", "Guns N' Roses", "spotify:track:4MhTFsyqIJnjsOweVcU8ug",
             ["sweet child of mine", "sweet child o mine", "guns and roses", "guns n roses"]),
            ("Bohemian Rhapsody", "Queen", "spotify:track:3z8h0TU7ReDPLIbEnYhWZb",
             ["bohemian rhapsody", "bohemian raphsody", "queen", "mama just killed a man"]),
            ("Marea (we've lost dancing)", "Fred again..", "spotify:track:5Z8Dj3LtbyCMiwE86rhg2f",
             ["marea", "maria", "we've lost dancing", "fred again", "ferd again", "marea fred again"]),
            ("Delilah (pull me out of this)", "Fred again..", "spotify:track:0Ftrkz2waaHcjKb4qYvLmz",
             ["delilah", "pull me out of this", "fred again delilah", "delilah fred again", "delilah by fred again", "delilah by fred"]),
            ("Hotel California", "Eagles", "spotify:track:40riOy7x9W7GXjyGp4pjAv",
             ["hotel california", "eagles"]),
            ("Starboy", "The Weeknd", "spotify:track:7MXVkk9YM5IZxh0WSlVIH0",
             ["starboy", "weeknd", "the weeknd"]),
            ("Lo-Fi Beats", "Spotify", "spotify:playlist:37i9dQZF1DXdLEN7aqioXM",
             ["lofi", "lo-fi", "chill", "chill beats", "study", "study beats"]),
            ("Today's Top Hits", "Spotify", "spotify:playlist:37i9dQZF1DXcBWIGoYBM5M",
             ["top hits", "hits", "today's hits", "popular", "chart"]),
            ("Rock Classics", "Spotify", "spotify:playlist:37i9dQZF1DX1rVvRgjXZwF",
             ["rock", "classic rock", "rock hits"]),
        ]
        for title, artist, uri, aliases in defaults:
            self.add_to_music_catalog(title, artist, uri, aliases)
