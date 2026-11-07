"""Pushes live data to every open dashboard over WebSockets."""
from __future__ import annotations

import asyncio
import json
import logging

from fastapi import WebSocket

log = logging.getLogger(__name__)


class Hub:
    def __init__(self):
        self.clients: set[WebSocket] = set()

    async def connect(self, ws: WebSocket, hello: dict) -> None:
        await ws.accept()
        await ws.send_text(json.dumps(hello))
        self.clients.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self.clients.discard(ws)

    async def broadcast(self, msg: dict) -> None:
        if not self.clients:
            return
        text = json.dumps(msg, default=str)

        async def send(ws: WebSocket):
            try:
                await asyncio.wait_for(ws.send_text(text), 2)
            except Exception:
                self.disconnect(ws)

        await asyncio.gather(*(send(ws) for ws in list(self.clients)))
