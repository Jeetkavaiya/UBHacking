"""Mission Control AI backend.

Run:  uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import FlightDirector
from .config import settings as S
from .data import load_channels
from .hub import Hub
from .incidents import Manager
from .replay import FAULT_TYPES, Busy, Engine
from .storage import Storage
from .voice import Voice

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("mission-control")


class App:
    """Holds every running part, built once at startup."""


A = App()


# ---------- voice: one ordered queue, high priority drops pending low ----------
class VoiceQueue:
    def __init__(self, voice: Voice, hub: Hub):
        self.voice, self.hub = voice, hub
        self.q: asyncio.Queue = asyncio.Queue()
        self.ids = itertools.count(1)
        self.recent: list[dict] = []

    async def say(self, text: str, priority: str = "high", incident_id: str | None = None) -> None:
        if priority == "high":  # an alert never waits behind an all-clear
            kept = [m for m in self._drain() if m["priority"] != "low"]
            for m in kept:
                self.q.put_nowait(m)
        self.q.put_nowait({"id": next(self.ids), "text": text, "priority": priority,
                           "incident_id": incident_id, "ts": time.time()})

    def _drain(self) -> list[dict]:
        out = []
        while not self.q.empty():
            out.append(self.q.get_nowait())
        return out

    async def worker(self) -> None:
        while True:
            m = await self.q.get()
            try:
                m["audio_url"] = await asyncio.wait_for(self.voice.audio_for(m["text"]), 8)
            except Exception:
                m["audio_url"] = None
            self.recent = (self.recent + [m])[-20:]
            await self.hub.broadcast({"type": "voice", **m})


async def lookup(cid: str, seconds: float) -> list[dict]:
    """The tool Gemini calls: live stats for related channels, plus Tiger Data aggregates."""
    if cid not in A.engine.states:
        return [{"error": f"unknown channel {cid}"}]
    rel = A.engine.related(cid, seconds)
    db = await A.storage.recent_stats([r["channel"] for r in rel] + [cid], seconds)
    for r in rel:
        r.update(db.get(r["channel"], {}))
    return rel


async def on_batch(batch, flags) -> None:
    now = time.time()
    await A.hub.broadcast({"type": "readings", "t": now, "speed": A.engine.speed, "data": batch})
    if A.storage.pool:
        k = max(1, S.store_every)
        ts = datetime.fromtimestamp(now, timezone.utc)
        rows = [(ts, cid, i, v, s, bool(x))
                for cid, b in batch.items()
                for j, (i, v, s, x) in enumerate(zip(b["i"], b["v"], b["s"], b["x"])) if j % k == 0]
        A.storage.add_readings(rows)
    await A.manager.on_batch(flags)


@asynccontextmanager
async def lifespan(app: FastAPI):
    S.cache_dir.mkdir(parents=True, exist_ok=True)
    channels, A.data_mode = load_channels(S.data_dir, S.channels, S.max_channels)
    A.channels = {c.id: c for c in channels}
    A.hub = Hub()
    A.storage = Storage(S.database_url)
    await A.storage.init()
    A.engine = Engine(channels, S)
    A.agent = FlightDirector(S.gemini_api_key, S.gemini_model, S.gemini_rpm, S.gemini_timeout, S.cache_dir, lookup,
                             sources=S.gemini_sources, daily_limit=S.gemini_daily_limit)
    A.voice = Voice(S.elevenlabs_api_key, S.elevenlabs_voice_id, S.elevenlabs_model, S.cache_dir / "audio")
    A.vq = VoiceQueue(A.voice, A.hub)
    A.manager = Manager(S, A.engine, A.agent, A.storage, A.hub.broadcast, A.vq.say)
    A.engine.on_batch = on_batch
    app.mount("/audio", StaticFiles(directory=S.cache_dir / "audio"), name="audio")
    tasks = [asyncio.create_task(c) for c in (A.engine.run(), A.vq.worker(), A.manager.heartbeat_loop(),
                                               A.storage.flush_loop(), A.voice.warm())]
    await A.manager.event("system", f"Monitoring started on {len(channels)} channels ({A.data_mode} data).")
    yield
    for t in tasks:
        t.cancel()
    await A.voice.close()
    await A.storage.close()


app = FastAPI(title="Mission Control AI", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------- REST ----------
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "uptime_s": round(time.time() - A.engine.started, 1),
        "data_mode": A.data_mode,
        "channels": len(A.channels),
        "speed": A.engine.speed,
        "open_incidents": len(A.manager.open_incidents()),
        "gemini": {"enabled": A.agent.enabled, "model": S.gemini_model if A.agent.enabled else None,
                   **A.agent.stats, **A.agent.quota()},
        "voice": {"enabled": A.voice.enabled, **A.voice.stats},
        "storage": {"mode": A.storage.mode, "timescale": A.storage.timescale, **A.storage.stats},
    }


@app.get("/api/channels")
async def channels():
    return [{**c.meta(), "detector": A.engine.states[c.id].det.info(),
             "surprise": round(A.engine.surprise(c.id), 2), "fault": bool(A.engine.states[c.id].fault)}
            for c in A.channels.values()]


@app.get("/api/channels/{cid}/history")
async def history(cid: str, n: int = 300):
    if cid not in A.channels:
        raise HTTPException(404, f"unknown channel {cid}")
    return A.engine.history(cid, min(max(n, 1), 3000))


class SpeedIn(BaseModel):
    speed: float = Field(ge=0.25, le=100)


@app.post("/api/control/speed")
async def set_speed(body: SpeedIn):
    A.engine.set_speed(body.speed)
    return {"speed": A.engine.speed}


class FaultIn(BaseModel):
    channel: str
    type: str = Field(description=f"one of {FAULT_TYPES}")


@app.post("/api/faults")
async def inject(body: FaultIn):
    if body.channel not in A.channels:
        raise HTTPException(404, f"unknown channel {body.channel}")
    if body.channel in A.manager.by_channel:
        raise HTTPException(409, f"{body.channel} already has an open incident")
    try:
        f = A.engine.inject(body.channel, body.type, "synthetic")
    except Busy as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))
    await A.manager.event("fault_injected", f"Injected a synthetic {f.type} on {body.channel}.",
                          tag="Synthetic fault")
    return {"channel": body.channel, "type": f.type, "synthetic": True}


@app.post("/api/replay")
async def replay(channel: str | None = None, which: int = 0):
    if channel and channel not in A.channels:
        raise HTTPException(404, f"unknown channel {channel}")
    if channel and channel in A.manager.by_channel:
        raise HTTPException(409, f"{channel} already has an open incident")
    try:
        r = A.engine.replay_labeled(channel, which)
    except Busy as e:
        raise HTTPException(409, str(e))
    real = A.data_mode == "real"
    await A.manager.event("replay_started",
                          f"Replaying {'a real NASA' if real else 'a simulated'} labeled anomaly on {r['channel']}.",
                          tag="Sample replay")
    return {**r, "real_data": real}


@app.get("/api/incidents")
async def incidents(open_only: bool = False, limit: int = 50):
    items = [i.view() for i in A.manager.incidents.values() if i.open or not open_only]
    return sorted(items, key=lambda x: x["opened_at"], reverse=True)[:limit]


@app.post("/api/incidents/{inc_id}/{decision}")
async def decide(inc_id: str, decision: str):
    if decision not in ("approve", "decline", "cancel"):
        raise HTTPException(404, "decision must be approve, decline or cancel")
    try:
        return A.manager.decide(inc_id, decision).view()
    except KeyError:
        raise HTTPException(404, f"unknown incident {inc_id}")
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.get("/api/log")
async def get_log(limit: int = 100):
    return list(A.manager.log)[:limit]


@app.post("/api/heartbeat")
async def heartbeat_now():
    """Trigger the all-clear call now (handy for demos)."""
    await A.manager.heartbeat()
    return {"ok": True}


# ---------- WebSocket ----------
@app.websocket("/ws")
async def ws(websocket: WebSocket):
    hello = {
        "type": "hello",
        "data_mode": A.data_mode,
        "speed": A.engine.speed,
        "base_rate": S.base_rate,
        "threshold": S.z_threshold,
        "veto_seconds": S.veto_seconds,
        "channels": [c.meta() for c in A.channels.values()],
        "history": {cid: A.engine.history(cid, 300) for cid in A.channels},
        "incidents": [i.view() for i in A.manager.incidents.values() if i.open],
        "log": list(A.manager.log)[:50],
    }
    await A.hub.connect(websocket, hello)
    try:
        while True:
            await websocket.receive_text()  # dashboard may send pings; actions go through REST
    except WebSocketDisconnect:
        A.hub.disconnect(websocket)
    except Exception:
        A.hub.disconnect(websocket)


# Serve the dashboard from the same server (published claude.ai pages can't call other sites).
if S.frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=S.frontend_dir, html=True), name="frontend")
