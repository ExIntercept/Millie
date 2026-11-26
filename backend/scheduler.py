"""Fires reminders and alarms on time, even if the browser tab is closed."""
import asyncio
import logging
import platform
import time

from . import config
from .db import DB
from .hub import Hub

log = logging.getLogger("millie.scheduler")


async def mac_notify(title: str, text: str) -> None:
    if platform.system() != "Darwin":
        return
    esc = lambda s: s.replace("\\", "\\\\").replace('"', '\\"')
    script = f'display notification "{esc(text)}" with title "{esc(title)}" sound name "Glass"'
    try:
        proc = await asyncio.create_subprocess_exec("osascript", "-e", script)
        await proc.wait()
    except Exception as e:
        log.warning("notification failed: %s", e)


class Scheduler:
    def __init__(self, db: DB, hub: Hub):
        self.db, self.hub = db, hub

    async def run(self) -> None:
        while True:
            try:
                for r in self.db.due_reminders(time.time()):
                    self.db.mark_fired(r["id"])
                    label = "Alarm" if r["kind"] == "alarm" else "Reminder"
                    log.info("%s fired: %s", label, r["text"])
                    await self.hub.broadcast({"type": "reminder_fired", "kind": r["kind"], "text": r["text"]})
                    await self.hub.changed("reminders")
                    await mac_notify(f"{config.ASSISTANT_NAME}: {label}", r["text"])
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("scheduler tick failed")
            await asyncio.sleep(1)
