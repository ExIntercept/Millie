"""Tiny broadcast hub: pushes JSON events to every connected browser."""
import logging

log = logging.getLogger("millie.hub")


class Hub:
    def __init__(self):
        self.clients: set = set()

    async def broadcast(self, message: dict) -> None:
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_json(message)
            except Exception:  # client went away
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    async def changed(self, what: str, **extra) -> None:
        """Tell the UI that some data changed so it can refetch."""
        await self.broadcast({"type": "changed", "what": what, **extra})
