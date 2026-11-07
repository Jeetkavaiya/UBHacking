"""The flight director. Gemini diagnoses each incident; rules take over if Gemini
is missing, slow, rate-limited or returns something invalid.

Each diagnosis costs 2 Gemini requests:
  1) Gemini may call get_related_channels (reads live stats / Tiger Data)
  2) Gemini returns the diagnosis as strict JSON
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Awaitable, Callable, Iterable

log = logging.getLogger(__name__)

# Gemini's free-tier daily quota resets at midnight Pacific. Fixed UTC-8 (PST) is close enough.
QUOTA_TZ = timezone(timedelta(hours=-8))
DEFAULT_SOURCES = ("synthetic", "sample-replay")


class QuotaExhausted(Exception):
    pass

ACTIONS = ["monitor", "power_cycle_sensor", "switch_to_backup", "safe_mode", "call_engineer"]
LEVELS = ["low", "medium", "high"]

SCHEMA = {
    "type": "object",
    "properties": {
        "subsystem": {"type": "string"},
        "severity": {"type": "string", "enum": LEVELS},
        "likely_cause": {"type": "string"},
        "recommended_action": {"type": "string", "enum": ACTIONS},
        "confidence": {"type": "string", "enum": LEVELS},
        "voice_summary": {"type": "string"},
    },
    "required": ["subsystem", "severity", "likely_cause", "recommended_action", "confidence", "voice_summary"],
}

SYSTEM = """You are the flight director AI in a spacecraft mission-control room.
You receive one anomaly at a time from an automatic detector watching NASA telemetry.

Hard rules:
- The data is anonymized and scaled to -1..1. Channel names are codes; only the first
  letter hints at the group (P = power, R = radiation; other letters are unknown groups).
  Diagnose at the subsystem/group level only. Never name specific hardware parts.
- Choose recommended_action ONLY from: monitor, power_cycle_sensor, switch_to_backup,
  safe_mode, call_engineer.
  - monitor: short glitch that is already settling.
  - power_cycle_sensor: the reading froze (flatline).
  - switch_to_backup: a unit is drifting away from normal.
  - safe_mode: only for HIGH severity that threatens the whole spacecraft (e.g. power).
  - call_engineer: you cannot tell what is happening.
- The "shape" field is measured by the detector (spike = brief jump, drift = sustained move
  out of the normal range, flatline = frozen reading). Treat it as fact and describe the event
  with that word. Pick the action that matches the shape unless related channels clearly show
  something bigger.
- If unsure, say so and lower your confidence. Do not invent numbers.
- Call get_related_channels before deciding if other channels could explain the event.
- likely_cause: one or two plain sentences a non-engineer understands.
- voice_summary: one short sentence to be read aloud, under 25 words."""

PAST = {
    "monitor": "Started close monitoring",
    "power_cycle_sensor": "Power cycled the sensor",
    "switch_to_backup": "Switched to the backup unit",
    "safe_mode": "Entered safe mode",
    "call_engineer": "Paged the on-call engineer",
}
LABEL = {
    "monitor": "Keep monitoring",
    "power_cycle_sensor": "Power cycle the sensor",
    "switch_to_backup": "Switch to backup unit",
    "safe_mode": "Enter safe mode",
    "call_engineer": "Call the on-call engineer",
}

Lookup = Callable[[str, float], Awaitable[list[dict]]]


class RateLimiter:
    """Sliding one-minute window, so we stay under the free-tier RPM."""

    def __init__(self, rpm: int):
        self.rpm = max(1, rpm)
        self.calls: deque[float] = deque()
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self.lock:
            while True:
                now = time.monotonic()
                while self.calls and now - self.calls[0] > 60:
                    self.calls.popleft()
                if len(self.calls) < self.rpm:
                    self.calls.append(now)
                    return
                await asyncio.sleep(60 - (now - self.calls[0]) + 0.05)


def rule_diagnosis(ctx: dict, related: list[dict]) -> dict:
    """Deterministic fallback, also used when no Gemini key is set."""
    shape, peak, z = ctx["shape"], ctx["peak_surprise"], ctx["threshold"]
    sev = "low" if peak < 2 * z else "medium" if peak < 4 * z else "high"
    cid, sub = ctx["channel"], ctx["subsystem"]
    moving = [r["channel"] for r in related if r.get("surprise_now", 0) > z * 0.5]
    with_others = (f" {', '.join(moving[:2])} moved at the same time, so the wider {sub.lower()} area may be involved."
                   if moving else " Related channels look normal, so this seems local to one unit.")
    if shape == "flatline":
        action, cause = "power_cycle_sensor", f"{cid} stopped changing when it normally moves. The sensor or its data link has likely frozen."
    elif shape == "drift":
        action, cause = "switch_to_backup", f"{cid} moved outside the range it normally stays in." + with_others
    else:
        action, cause = "monitor", f"A short burst on {cid}." + with_others
    if sev == "high" and ctx["group"] == "P" and shape != "spike":
        action = "safe_mode"
    return {
        "subsystem": sub,
        "severity": sev,
        "likely_cause": cause,
        "recommended_action": action,
        "confidence": "medium" if moving or shape != "spike" else "low",
        "voice_summary": f"{sev.capitalize()} severity on {cid}. Recommended: {LABEL[action].lower()}.",
    }


def _valid(d: dict) -> bool:
    return (isinstance(d, dict) and all(k in d for k in SCHEMA["required"])
            and d["recommended_action"] in ACTIONS and d["severity"] in LEVELS and d["confidence"] in LEVELS)


class FlightDirector:
    def __init__(self, api_key: str, model: str, rpm: int, timeout: float, cache_dir: Path, lookup: Lookup,
                 sources: Iterable[str] = DEFAULT_SOURCES, daily_limit: int = 18):
        self.model, self.timeout, self.lookup = model, timeout, lookup
        self.limiter = RateLimiter(rpm)
        self.queue_lock = asyncio.Lock()  # one diagnosis at a time
        self.cache_path = cache_dir / "diagnosis_cache.json"
        self.cache: dict[str, dict] = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        # Quota guard: only these incident sources use Gemini, and at most daily_limit requests per day.
        self.sources = {s.strip() for s in sources if s.strip()}
        self.daily_limit = max(0, daily_limit)
        self.usage_path = cache_dir / "gemini_usage.json"
        self.usage = self._load_usage()
        self.client = None
        self.stats = {"requests": 0, "fallbacks": 0, "cache_hits": 0, "rate_limited": 0,
                      "skipped_source": 0, "quota_blocked": 0}
        if api_key:
            from google import genai
            from google.genai import types
            # Function calling needs v1beta; some SDK versions default to v1.
            self.client = genai.Client(api_key=api_key,
                                       http_options=types.HttpOptions(api_version="v1beta"))
        else:
            log.warning("GEMINI_API_KEY not set: using rule-based diagnoses.")

    @property
    def enabled(self) -> bool:
        return self.client is not None

    # ---------- daily quota ----------
    @staticmethod
    def _today() -> str:
        return datetime.now(QUOTA_TZ).date().isoformat()

    def _load_usage(self) -> dict:
        try:
            u = json.loads(self.usage_path.read_text())
            if u.get("day") == self._today():
                return {"day": u["day"], "used": int(u.get("used", 0))}
        except (OSError, ValueError, TypeError, KeyError):
            pass
        return {"day": self._today(), "used": 0}

    def _used_today(self) -> int:
        if self.usage["day"] != self._today():  # quota reset at midnight Pacific
            self.usage = {"day": self._today(), "used": 0}
        return self.usage["used"]

    def _count_request(self) -> None:
        self._used_today()
        self.usage["used"] += 1
        try:
            self.usage_path.parent.mkdir(parents=True, exist_ok=True)
            self.usage_path.write_text(json.dumps(self.usage))
        except OSError as e:
            log.warning("Could not save Gemini usage (%s).", e)

    def _has_budget(self, need: int) -> bool:
        return self.daily_limit == 0 or self._used_today() + need <= self.daily_limit

    def uses_gemini_for(self, source: str) -> bool:
        return "all" in self.sources or source in self.sources

    def quota(self) -> dict:
        return {"sources": sorted(self.sources), "daily_used": self._used_today(),
                "daily_limit": self.daily_limit or None}

    async def diagnose(self, ctx: dict) -> dict:
        key = f"{ctx['channel']}|{ctx['shape']}|{ctx['source']}"
        if ctx["source"] == "sample-replay" and key in self.cache:
            self.stats["cache_hits"] += 1
            return {**self.cache[key], "source": "gemini-cached"}
        related = await self.lookup(ctx["channel"], 30)
        if not self.client:
            return {**rule_diagnosis(ctx, related), "source": "rules"}
        if not self.uses_gemini_for(ctx["source"]):  # natural incidents: save quota for the demo
            self.stats["skipped_source"] += 1
            return {**rule_diagnosis(ctx, related), "source": "rules"}
        async with self.queue_lock:
            if not self._has_budget(2):  # each diagnosis needs 2 requests
                self.stats["quota_blocked"] += 1
                log.warning("Gemini daily limit reached (%s/%s); using rules.", self._used_today(), self.daily_limit)
                return {**rule_diagnosis(ctx, related), "source": "rules-quota"}
            try:
                d = await self._gemini(ctx)
                if ctx["source"] == "sample-replay":
                    self.cache[key] = d
                    self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                    self.cache_path.write_text(json.dumps(self.cache, indent=1))
                return {**d, "source": "gemini"}
            except QuotaExhausted:  # retries used up the budget mid-diagnosis
                self.stats["quota_blocked"] += 1
                return {**rule_diagnosis(ctx, related), "source": "rules-quota"}
            except Exception as e:  # anything goes wrong -> rules, demo keeps going
                log.warning("Gemini failed (%s); using rules.", e)
                self.stats["fallbacks"] += 1
                return {**rule_diagnosis(ctx, related), "source": "rules-fallback"}

    async def _call(self, contents, config):
        from google.genai import errors
        for attempt in range(4):
            if not self._has_budget(1):
                raise QuotaExhausted()
            await self.limiter.acquire()
            self.stats["requests"] += 1
            self._count_request()  # failed and retried requests may count against the quota too
            try:
                return await asyncio.wait_for(
                    self.client.aio.models.generate_content(model=self.model, contents=contents, config=config),
                    self.timeout)
            except errors.APIError as e:
                if getattr(e, "code", None) == 429 and attempt < 3:
                    self.stats["rate_limited"] += 1
                    await asyncio.sleep(2 ** (attempt + 1))
                    continue
                raise

    async def _gemini(self, ctx: dict) -> dict:
        from google.genai import types
        decl = types.FunctionDeclaration(
            name="get_related_channels",
            description="Live statistics for channels in the same group or on the same spacecraft "
                        "over a recent window (means before/after, spread, surprise scores).",
            parameters_json_schema={"type": "object", "properties": {
                "channel_id": {"type": "string"},
                "window_seconds": {"type": "number", "description": "10 to 120"}},
                "required": ["channel_id"]},
        )
        prompt = "New anomaly from the detector:\n" + json.dumps(ctx, indent=1)
        contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]

        # Request 1: let Gemini look up related channels.
        r1 = await self._call(contents, types.GenerateContentConfig(
            system_instruction=SYSTEM, temperature=0.2,
            tools=[types.Tool(function_declarations=[decl])],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)))
        calls = r1.function_calls or []
        if calls and r1.candidates:
            contents.append(r1.candidates[0].content)
            parts = []
            for c in calls[:2]:
                args = dict(c.args or {})
                win = float(args.get("window_seconds", 30) or 30)
                data = await self.lookup(str(args.get("channel_id", ctx["channel"])), max(10, min(win, 120)))
                parts.append(types.Part.from_function_response(name=c.name, response={"result": data}))
            contents.append(types.Content(role="user", parts=parts))
        else:
            related = await self.lookup(ctx["channel"], 30)
            contents.append(types.Content(role="user", parts=[types.Part(
                text="Related channels: " + json.dumps(related))]))
        contents.append(types.Content(role="user", parts=[types.Part(text="Now give your diagnosis as JSON.")]))

        # Request 2: strict JSON answer.
        r2 = await self._call(contents, types.GenerateContentConfig(
            system_instruction=SYSTEM, temperature=0.2,
            response_mime_type="application/json", response_json_schema=SCHEMA))
        d = json.loads(r2.text or "{}")
        if not _valid(d):
            raise ValueError(f"invalid diagnosis: {d}")
        return d