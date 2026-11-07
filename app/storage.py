"""Saves every reading, incident and event to Tiger Data (PostgreSQL + TimescaleDB).
Without DATABASE_URL, the app keeps recent data in memory only."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

log = logging.getLogger(__name__)

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS telemetry (
        ts TIMESTAMPTZ NOT NULL,
        channel TEXT NOT NULL,
        idx INTEGER NOT NULL,
        value DOUBLE PRECISION NOT NULL,
        surprise DOUBLE PRECISION NOT NULL,
        injected BOOLEAN NOT NULL DEFAULT FALSE)""",
    "CREATE INDEX IF NOT EXISTS telemetry_channel_ts ON telemetry (channel, ts DESC)",
    """CREATE TABLE IF NOT EXISTS incidents (
        id TEXT PRIMARY KEY,
        channel TEXT NOT NULL,
        opened_at TIMESTAMPTZ NOT NULL,
        closed_at TIMESTAMPTZ,
        status TEXT NOT NULL,
        source TEXT,
        shape TEXT,
        severity TEXT,
        action TEXT,
        tier INTEGER,
        peak_surprise DOUBLE PRECISION,
        data JSONB)""",
    """CREATE TABLE IF NOT EXISTS events (
        ts TIMESTAMPTZ NOT NULL,
        incident_id TEXT,
        kind TEXT NOT NULL,
        message TEXT NOT NULL,
        tag TEXT)""",
]


class Storage:
    def __init__(self, url: str):
        self.url = url
        self.pool = None
        self.buf: list[tuple] = []
        self.mode = "memory"
        self.timescale = False
        self.stats = {"rows_written": 0, "errors": 0}

    async def init(self) -> None:
        if not self.url:
            log.warning("DATABASE_URL not set: in-memory only.")
            return
        import asyncpg
        try:
            self.pool = await asyncpg.create_pool(self.url, min_size=1, max_size=4, command_timeout=15)
            async with self.pool.acquire() as con:
                for q in SCHEMA:
                    await con.execute(q)
                try:
                    await con.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
                    await con.execute("SELECT create_hypertable('telemetry', 'ts', if_not_exists => TRUE)")
                    await con.execute("SELECT create_hypertable('events', 'ts', if_not_exists => TRUE)")
                    self.timescale = True
                except Exception as e:
                    log.warning("TimescaleDB hypertables not available (%s); using plain tables.", e)
            self.mode = "tigerdata" if self.timescale else "postgres"
            log.info("Storage ready (%s).", self.mode)
        except Exception as e:
            log.error("Database unavailable (%s): in-memory only.", e)
            self.pool = None

    # ---------- writes ----------
    def add_readings(self, rows: list[tuple]) -> None:
        if self.pool:
            self.buf.extend(rows)

    async def flush_loop(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            if not self.pool or not self.buf:
                continue
            rows, self.buf = self.buf, []
            try:
                async with self.pool.acquire() as con:
                    await con.copy_records_to_table(
                        "telemetry", records=rows,
                        columns=["ts", "channel", "idx", "value", "surprise", "injected"])
                self.stats["rows_written"] += len(rows)
            except Exception as e:
                self.stats["errors"] += 1
                log.warning("Telemetry flush failed (%s); dropped %d rows.", e, len(rows))

    async def save_incident(self, inc: dict) -> None:
        if not self.pool:
            return
        d = inc.get("diagnosis") or {}
        try:
            async with self.pool.acquire() as con:
                await con.execute(
                    """INSERT INTO incidents (id, channel, opened_at, closed_at, status, source, shape, severity,
                                             action, tier, peak_surprise, data)
                       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
                       ON CONFLICT (id) DO UPDATE SET closed_at=$4, status=$5, shape=$7, severity=$8,
                         action=$9, tier=$10, peak_surprise=$11, data=$12""",
                    inc["id"], inc["channel"], _ts(inc["opened_at"]), _ts(inc.get("closed_at")), inc["status"],
                    inc["source"], inc.get("shape"), d.get("severity"), d.get("recommended_action"),
                    inc.get("tier"), inc.get("peak_surprise"), json.dumps(inc))
        except Exception as e:
            self.stats["errors"] += 1
            log.warning("save_incident failed: %s", e)

    async def save_event(self, ev: dict) -> None:
        if not self.pool:
            return
        try:
            async with self.pool.acquire() as con:
                await con.execute("INSERT INTO events (ts, incident_id, kind, message, tag) VALUES ($1,$2,$3,$4,$5)",
                                  _ts(ev["ts"]), ev.get("incident_id"), ev["kind"], ev["message"], ev.get("tag"))
        except Exception as e:
            self.stats["errors"] += 1
            log.warning("save_event failed: %s", e)

    # ---------- reads (used by the flight director's lookup tool) ----------
    async def recent_stats(self, channels: list[str], seconds: float) -> dict[str, dict]:
        if not self.pool:
            return {}
        try:
            async with self.pool.acquire() as con:
                rows = await con.fetch(
                    """SELECT channel, avg(value) AS mean, stddev_pop(value) AS std, max(surprise) AS max_surprise,
                              count(*) AS n
                       FROM telemetry WHERE channel = ANY($1) AND ts > now() - make_interval(secs => $2)
                       GROUP BY channel""", channels, float(seconds))
            return {r["channel"]: {"db_mean": round(r["mean"] or 0, 4), "db_std": round(r["std"] or 0, 4),
                                   "db_max_surprise": round(r["max_surprise"] or 0, 2), "db_rows": r["n"]}
                    for r in rows}
        except Exception as e:
            log.warning("recent_stats failed: %s", e)
            return {}

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()


def _ts(x):
    if x is None:
        return None
    return datetime.fromtimestamp(x, tz=timezone.utc) if isinstance(x, (int, float)) else x
