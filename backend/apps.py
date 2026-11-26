"""Helper for launching applications, controlling Spotify, and opening browser lookups on macOS."""
import json
import logging
import os
import re
import subprocess
import time
import urllib.parse
import urllib.request
from typing import Any, Optional

log = logging.getLogger("millie.apps")

APP_ALIASES = {
    "chrome": "Google Chrome",
    "google chrome": "Google Chrome",
    "safari": "Safari",
    "code": "Visual Studio Code",
    "vscode": "Visual Studio Code",
    "vs code": "Visual Studio Code",
    "spotify": "Spotify",
    "finder": "Finder",
    "notes": "Notes",
    "terminal": "Terminal",
    "iterm": "iTerm",
    "iterm2": "iTerm",
    "calendar": "Calendar",
    "reminders": "Reminders",
    "mail": "Mail",
    "music": "Music",
    "apple music": "Music",
    "settings": "System Settings",
    "system settings": "System Settings",
    "system preferences": "System Settings",
    "slack": "Slack",
    "discord": "Discord",
    "messages": "Messages",
    "preview": "Preview",
    "calculator": "Calculator",
    "photos": "Photos",
}


def open_application(app_name: str, target: Optional[str] = None) -> dict:
    """Launch an application or URL on macOS.
    
    If app_name is an alias or name, resolve it.
    If target is a URL or file path, open it within the app or default handler.
    """
    clean_name = app_name.strip()
    low = clean_name.lower()

    # Route Spotify music playback requests
    if low == "spotify" and target and not target.startswith(("/", "file:")):
        return play_spotify(query=target)

    # Route Browser search queries
    if low in {"browser", "chrome", "google chrome", "safari"} and target and not re.match(r"^https?://", target, re.IGNORECASE) and not target.startswith("/"):
        return browse_lookup(query=target, browser=APP_ALIASES.get(low))

    # If user provided a URL as app_name (e.g. "open https://google.com")
    if re.match(r"^https?://", clean_name, re.IGNORECASE):
        cmd = ["open", clean_name]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(res.stderr.strip() or f"Failed to open URL '{clean_name}'")
        return {"app": "Default Browser", "target": clean_name}

    resolved_app = APP_ALIASES.get(low, clean_name)

    cmd = ["open", "-a", resolved_app]
    if target and target.strip():
        cmd.append(target.strip())

    log.info("Executing: %s", cmd)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        err = res.stderr.strip()
        log.warning("App launch failed for %s: %s", resolved_app, err)
        raise RuntimeError(f"Could not open '{resolved_app}': {err}")

    return {"app": resolved_app, "target": target}


SPOTIFY_PLAYLISTS = {
    "top hits": "spotify:playlist:37i9dQZF1DXcBWIGoYBM5M",
    "hits": "spotify:playlist:37i9dQZF1DXcBWIGoYBM5M",
    "music": "spotify:playlist:37i9dQZF1DXcBWIGoYBM5M",
    "song": "spotify:playlist:37i9dQZF1DXcBWIGoYBM5M",
    "songs": "spotify:playlist:37i9dQZF1DXcBWIGoYBM5M",
    "today's top hits": "spotify:playlist:37i9dQZF1DXcBWIGoYBM5M",
    "lofi": "spotify:playlist:37i9dQZF1DXdLEN7aqioXM",
    "lo-fi": "spotify:playlist:37i9dQZF1DXdLEN7aqioXM",
    "chill": "spotify:playlist:37i9dQZF1DXdLEN7aqioXM",
    "study": "spotify:playlist:37i9dQZF1DXdLEN7aqioXM",
    "rap": "spotify:playlist:37i9dQZF1DX0XUsuxWHRQd",
    "hip hop": "spotify:playlist:37i9dQZF1DX0XUsuxWHRQd",
    "rock": "spotify:playlist:37i9dQZF1DX1rVvRgjXZwF",
    "classic rock": "spotify:playlist:37i9dQZF1DX1rVvRgjXZwF",
    "pop": "spotify:playlist:37i9dQZF1DX1ngEVM09JJU",
    "workout": "spotify:playlist:37i9dQZF1DX76Wlfdnj7AP",
    "gym": "spotify:playlist:37i9dQZF1DX76Wlfdnj7AP",
    "jazz": "spotify:playlist:37i9dQZF1DXbITWG1ZJKYt",
    "dance": "spotify:playlist:37i9dQZF1DX4dyzvuaRJ0n",
    "edm": "spotify:playlist:37i9dQZF1DX4dyzvuaRJ0n",
    "piano": "spotify:playlist:37i9dQZF1DX4sWSpwq3LiO",
}


def _get_spotify_state() -> str:
    res = subprocess.run(["osascript", "-e", 'tell application "Spotify" to return player state as string'],
                         capture_output=True, text=True)
    return res.stdout.strip().lower() if res.returncode == 0 else "unknown"


def _get_current_track_info() -> str:
    res = subprocess.run(
        ["osascript", "-e", 'tell application "Spotify" to return (name of current track) & " by " & (artist of current track)'],
        capture_output=True, text=True
    )
    return res.stdout.strip() if res.returncode == 0 else ""


def clean_spotify_query(q: str) -> str:
    """Strip trailing punctuation and 'on/in spotify' phrases."""
    clean = (q or "").strip()
    clean = re.sub(r"[\s.?!]+$", "", clean).strip()
    clean = re.sub(r"\s+(?:on|in|from)\s+spotify\s*$", "", clean, flags=re.I).strip()
    return clean


def _search_spotify_track_online(query: str) -> Optional[str]:
    """Resolve a freeform song/artist query to a Spotify track URI."""
    q_clean = clean_spotify_query(query)
    if not q_clean:
        return None

    client_id = os.getenv("SPOTIFY_CLIENT_ID", "").strip()
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET", "").strip()

    # 1. Try official Spotify API if user configured credentials
    if client_id and client_secret:
        try:
            import base64
            auth_str = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
            token_req = urllib.request.Request(
                "https://accounts.spotify.com/api/token",
                data=b"grant_type=client_credentials",
                headers={"Authorization": f"Basic {auth_str}", "Content-Type": "application/x-www-form-urlencoded"}
            )
            with urllib.request.urlopen(token_req, timeout=4) as t_resp:
                t_data = json.loads(t_resp.read().decode())
                access_token = t_data.get("access_token")

            if access_token:
                search_url = f"https://api.spotify.com/v1/search?q={urllib.parse.quote(q_clean)}&type=track&limit=1"
                s_req = urllib.request.Request(search_url, headers={"Authorization": f"Bearer {access_token}"})
                with urllib.request.urlopen(s_req, timeout=4) as s_resp:
                    s_data = json.loads(s_resp.read().decode())
                    items = s_data.get("tracks", {}).get("items", [])
                    if items:
                        return items[0].get("uri")
        except Exception as e:
            log.warning("Spotify Web API lookup error: %s", e)

    # 2. Free DuckDuckGo Spotify track resolution (no credentials needed)
    for q_pattern in [f"site:open.spotify.com/track {q_clean}", f"site:open.spotify.com {q_clean}"]:
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(q_pattern, max_results=3))
                for r in results:
                    m = re.search(r"open\.spotify\.com/(track|album|playlist)/([a-zA-Z0-9]+)", r.get("href", ""))
                    if m:
                        return f"spotify:{m.group(1)}:{m.group(2)}"
        except Exception as e:
            log.warning("DDGS Spotify lookup pattern '%s' error: %s", q_pattern, e)

    return None


def _get_db():
    try:
        from . import config
        from .db import DB
        return DB(config.DB_PATH)
    except Exception:
        return None


def play_spotify(query: Optional[str] = None, action: str = "play", db: Optional[Any] = None) -> dict:
    """Control Spotify and auto-play tracks, albums, artists, or playlists on macOS."""
    clean_action = (action or "play").lower().strip()

    # Catch when an action keyword was mistakenly passed into query instead of action
    if query:
        q_raw = clean_spotify_query(query).lower()
        if q_raw in {"pause", "stop", "halt", "pause music", "stop music", "pause it"}:
            clean_action = "pause"
            query = None
        elif q_raw in {"next", "skip", "next song", "next track", "skip song", "skip this"}:
            clean_action = "next"
            query = None
        elif q_raw in {"prev", "previous", "back", "previous song", "last song", "go back"}:
            clean_action = "prev"
            query = None
        elif q_raw in {"play", "resume", "unpause", "continue", "play music", "resume music"}:
            clean_action = "play"
            query = None

    # Ensure Spotify is active
    subprocess.run(["open", "-a", "Spotify"], capture_output=True)

    if clean_action in {"pause", "stop"}:
        subprocess.run(["osascript", "-e", 'tell application "Spotify" to pause'], capture_output=True)
        return {"action": "paused", "status": "Spotify playback paused."}

    if clean_action in {"next", "skip"}:
        subprocess.run(["osascript", "-e", 'tell application "Spotify" to next track'], capture_output=True)
        time.sleep(0.3)
        info = _get_current_track_info()
        return {"action": "next", "status": f"Skipped to next track{' (' + info + ')' if info else ''}."}

    if clean_action in {"prev", "previous", "back"}:
        subprocess.run(["osascript", "-e", 'tell application "Spotify" to previous track'], capture_output=True)
        time.sleep(0.3)
        info = _get_current_track_info()
        return {"action": "previous", "status": f"Went back to previous track{' (' + info + ')' if info else ''}."}

    if query and query.strip():
        q = clean_spotify_query(query)
        low = q.lower()

        # 1. Direct Spotify URI (e.g. spotify:track:... or spotify:playlist:...)
        m_uri = re.match(r"spotify:(track|playlist|album):[a-zA-Z0-9]+", q)
        m_url = re.search(r"open\.spotify\.com/(track|playlist|album)/([a-zA-Z0-9]+)", q)
        target_uri = None
        if m_uri:
            target_uri = q
        elif m_url:
            target_uri = f"spotify:{m_url.group(1)}:{m_url.group(2)}"
        else:
            # 2. Genre / mood playlists: "lofi", "some jazz", "rock music". Whole-phrase match only,
            #    so "songs by arijit singh" isn't mistaken for the generic "songs" playlist.
            genre = re.sub(r"^(?:some|a)\s+|\s+(?:music|playlist|songs|beats|mix)$", "", low).strip()
            target_uri = SPOTIFY_PLAYLISTS.get(low) or SPOTIFY_PLAYLISTS.get(genre)

        # 3. Check persistent music catalog FIRST (0ms, 100% accurate, no rate limits)
        db_inst = db or _get_db()
        cat_match = None
        if not target_uri and db_inst:
            cat_match = db_inst.find_in_music_catalog(q)
            if cat_match:
                target_uri = cat_match["spotify_uri"]
                log.info("Found in music catalog: '%s' -> %s (%s)", q, target_uri, cat_match.get("title"))

        # 4. Resolve specific song name or artist online to actual track URI
        if not target_uri:
            log.info("Searching online Spotify track URI for: %s", q)
            target_uri = _search_spotify_track_online(q)

        if target_uri:
            script = f'tell application "Spotify" to play track "{target_uri}"'
            res = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
            log.info("Played Spotify track %s -> exit %s", target_uri, res.returncode)
            time.sleep(0.4)
            # Ensure it is actively playing and not paused
            state = _get_spotify_state()
            if state != "playing":
                subprocess.run(["osascript", "-e", 'tell application "Spotify" to play'], capture_output=True)
            info = _get_current_track_info()

            # Auto-cache newly resolved song to music catalog so it's instant next time!
            if db_inst and not cat_match:
                try:
                    title_guess = q.title()
                    artist_guess = ""
                    if info and " by " in info:
                        p = info.split(" by ", 1)
                        title_guess, artist_guess = p[0].strip(), p[1].strip()
                    db_inst.add_to_music_catalog(title_guess, artist_guess, target_uri, aliases=[q.lower()])
                    log.info("Auto-cached to music catalog: %s by %s", title_guess, artist_guess)
                except Exception as e:
                    log.warning("Could not auto-cache song: %s", e)

            return {"action": "playing", "query": q, "status": f"Now playing '{info or q}' on Spotify."}

        # 5. Fallback if resolution failed: open search and trigger playback
        encoded = urllib.parse.quote(q)
        subprocess.run(["open", f"spotify:search:{encoded}"])
        time.sleep(0.5)
        subprocess.run(["osascript", "-e", 'tell application "Spotify" to play'], capture_output=True)
        return {
            "action": "playing",
            "query": q,
            "status": f"Opened '{q}' on Spotify and started playback."
        }

    # 4. Resume current playback or start top hits if empty
    state = _get_spotify_state()
    if state == "stopped":
        # Load top hits playlist so there's music in queue
        subprocess.run(
            ["osascript", "-e", 'tell application "Spotify" to play track "spotify:playlist:37i9dQZF1DXcBWIGoYBM5M"'],
            capture_output=True
        )
    else:
        subprocess.run(["osascript", "-e", 'tell application "Spotify" to play'], capture_output=True)

    info = _get_current_track_info()
    return {"action": "playing", "status": f"Playing on Spotify{' (' + info + ')' if info else ''}."}



def browse_lookup(query: str, engine: str = "google", browser: Optional[str] = None) -> dict:
    """Look up a search query or URL in a new browser tab on macOS."""
    clean_q = query.strip()
    clean_engine = (engine or "google").lower().strip()

    # Direct URL detection
    if re.match(r"^https?://", clean_q, re.IGNORECASE):
        url = clean_q
    elif re.match(r"^[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}(/.*)?$", clean_q) and " " not in clean_q:
        url = f"https://{clean_q}"
    else:
        encoded = urllib.parse.quote_plus(clean_q)
        if clean_engine == "youtube":
            url = f"https://www.youtube.com/results?search_query={encoded}"
        elif clean_engine in {"duckduckgo", "ddg"}:
            url = f"https://duckduckgo.com/?q={encoded}"
        elif clean_engine == "github":
            url = f"https://github.com/search?q={encoded}"
        else:
            url = f"https://www.google.com/search?q={encoded}"

    cmd = ["open"]
    if browser and browser.strip():
        resolved_browser = APP_ALIASES.get(browser.strip().lower(), browser.strip())
        cmd += ["-a", resolved_browser]
    cmd.append(url)

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip() or f"Failed to open browser tab for '{url}'")

    return {"url": url, "query": clean_q, "engine": clean_engine, "status": f"Opened '{clean_q}' in browser tab."}
