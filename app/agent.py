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
DEFAULT_SOURCES = ("synthetic", "sample-replay", "sim-scenario")


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
    "shed_load": "Switched off the payload to save power",
}
LABEL = {
    "monitor": "Keep monitoring",
    "power_cycle_sensor": "Power cycle the sensor",
    "switch_to_backup": "Switch to backup unit",
    "safe_mode": "Enter safe mode",
    "call_engineer": "Call the on-call engineer",
    "shed_load": "Switch off the payload to save power",
}
# SIM-1 has one more action: it has a payload that can be switched off.
SIM_ACTIONS = ACTIONS + ["shed_load"]


def _sim_parts() -> list[str]:
    from .sim import PARTS
    return list(PARTS)


SIM_SCHEMA = {
    "type": "object",
    "properties": {
        "subsystem": {"type": "string"},
        "severity": {"type": "string", "enum": LEVELS},
        "likely_cause": {"type": "string", "description": "What happened and why, 2-3 plain sentences."},
        "root_cause": {"type": "string", "enum": _sim_parts()},
        "evidence": {"type": "array", "items": {"type": "string"}, "description": "2-4 facts from the data."},
        "recommended_action": {"type": "string", "enum": SIM_ACTIONS},
        "target": {"type": "string", "enum": _sim_parts()},
        "fallback_action": {"type": "string", "enum": SIM_ACTIONS},
        "fallback_target": {"type": "string", "enum": _sim_parts()},
        "confidence": {"type": "string", "enum": LEVELS},
        "voice_summary": {"type": "string"},
    },
    "required": ["subsystem", "severity", "likely_cause", "root_cause", "evidence", "recommended_action",
                 "target", "fallback_action", "fallback_target", "confidence", "voice_summary"],
}

# The spacecraft handbook: how SIM-1's parts work and connect. It deliberately does NOT list
# fault signatures or answers; the flight director has to reason from the data.
SIM_SYSTEM = """You are the flight director AI for SIM-1, a SIMULATED small Earth-orbiting satellite.
You get one incident at a time: the first alarm, every linked symptom with timing, a snapshot of
all sensors with their normal ranges, the spacecraft status (orbit, eclipse, ops plan, active
units, heater state, expected solar current) and past incidents with what worked.

SIM-1 handbook (how the parts work):
- Orbit: 95 minutes, about 35 of them in Earth's shadow (eclipse). The solar array only makes
  current in sunlight. solar_current_expected_now_a is what a healthy array gives right now.
- Power: the array charges the battery in sunlight; in eclipse the battery alone powers every
  load (bus, radio, payload camera, heater, wheel motor). More load or less solar input means a
  deeper discharge. The ops plan may run extra loads, such as an imaging pass.
- Thermal: a thermostat switches the battery heater on below 8 C and off above 12 C, reading the
  ACTIVE battery temperature sensor. The heater controller has a backup (B). Outside temperatures
  swing with sunlight and shadow.
- Attitude: the active reaction wheel points the spacecraft; its motor current is the torque it
  needs, and work turns into wheel heat. Pointing error grows when the wheel cannot deliver.
- Comms: radio signal margin and transmitter temperature.
- Spares: one backup (unit B) each for reaction_wheel, battery_temp_sensor, heater and radio.
  The battery, solar array, payload and star tracker have no backup.

Actions (pick ONE now, plus a fallback to try if the first does not work):
- monitor: nothing is broken, or the behavior is expected.
- power_cycle_sensor + target: restart that unit's electronics.
- switch_to_backup + target: swap that part to unit B.
- shed_load: switch the payload off to save power.
- safe_mode: payload off and array to the sun. Only if the battery is in real danger; needs human approval.
- call_engineer: a hardware problem commands cannot fix.

Rules:
- root_cause is the part that actually failed, or "none" if nothing failed. It is often NOT the
  part the first alarming sensor describes. target is the part your action is sent to.
- Use only part names from the list. Quote numbers from the data; never invent them.
- Check past_incidents: do not repeat a fix that already failed on the same part.
- likely_cause: 2-3 plain sentences a non-engineer understands: what happened and why.
- evidence: 2-4 short facts from the data that support the root cause.
- voice_summary: one sentence under 25 words, read aloud to the control room.
- If unsure, say so and lower your confidence."""


def label(action: str, target: str | None = None) -> str:
    """Action label naming the part for SIM-1 (NASA parts are unknown, so no target there)."""
    from .sim import PARTS
    part = PARTS.get(target or "", None)
    if part and target not in ("none", "spacecraft"):
        if action == "power_cycle_sensor":
            return f"Power cycle the {part}"
        if action == "switch_to_backup":
            return f"Switch the {part} to its backup"
    return LABEL[action]


def past(action: str, target: str | None = None) -> str:
    from .sim import PARTS
    part = PARTS.get(target or "", None)
    if part and target not in ("none", "spacecraft"):
        if action == "power_cycle_sensor":
            return f"Power cycled the {part}"
        if action == "switch_to_backup":
            return f"Switched the {part} to its backup"
    return PAST[action]

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
    area = sub if sub.startswith("Group") else sub.lower()  # "Group T" stays a name; "Power" -> "power"
    moving = [r["channel"] for r in related if r.get("surprise_now", 0) > z * 0.5]
    with_others = (f" {', '.join(moving[:2])} moved at the same time, so the wider {area} area may be involved."
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


def rule_diagnosis_sim(ctx: dict) -> dict:
    """Naive SIM-1 rules: blame the part the FIRST alarming sensor describes, act on its shape.
    They know nothing about the orbit, the ops plan or how parts affect each other. This is
    the baseline the AI flight director is graded against."""
    shape, peak, z = ctx["shape"], ctx["peak_surprise"], ctx["threshold"]
    sev = "low" if peak < 2 * z else "medium" if peak < 4 * z else "high"
    name, part = ctx["sensor_name"], ctx["part"]
    lo, hi = ctx["normal_range"]
    units = ctx.get("units", "")
    from .sim import BACKUPS
    fb, fb_target = "call_engineer", "spacecraft"
    if shape == "spike":
        root, action, target = "none", "monitor", part
        cause = f"A brief jump on {name.lower()}. It looks like a one-off glitch."
    elif shape == "flatline":
        root = target = ctx.get("sensor_part") or part
        action = "power_cycle_sensor"
        cause = f"{name} stopped changing. The sensor has likely frozen."
        if target in BACKUPS:
            fb, fb_target = "switch_to_backup", target
    else:
        root, action, target = part, "switch_to_backup", part
        cause = f"{name} moved outside its normal range ({lo:g} to {hi:g} {units})."
        if sev == "high" and ctx["subsystem"] == "Power":
            action, target = "safe_mode", "spacecraft"
    now = (ctx.get("channel_stats") or {}).get("value_now")
    evidence = [f"{name} is {now:g} {units}; normal is {lo:g} to {hi:g}." if now is not None
                else f"{name} normal range is {lo:g} to {hi:g} {units}.",
                f"Detector saw a {shape} (peak surprise {peak:.1f}, threshold {z:g})."]
    return {
        "subsystem": ctx["subsystem"],
        "severity": sev,
        "likely_cause": cause,
        "root_cause": root,
        "evidence": evidence,
        "recommended_action": action,
        "target": target,
        "fallback_action": fb,
        "fallback_target": fb_target,
        "confidence": "medium" if shape != "spike" else "low",
        "voice_summary": f"{sev.capitalize()} severity on {name.lower()}. Recommended: {label(action, target).lower()}.",
    }


def _valid_sim(d: dict) -> bool:
    parts = set(_sim_parts())
    return (isinstance(d, dict) and all(k in d for k in SIM_SCHEMA["required"])
            and d["recommended_action"] in SIM_ACTIONS and d["fallback_action"] in SIM_ACTIONS
            and d["root_cause"] in parts and d["target"] in parts and d["fallback_target"] in parts
            and d["severity"] in LEVELS and d["confidence"] in LEVELS and isinstance(d["evidence"], list))


def _valid(d: dict) -> bool:
    return (isinstance(d, dict) and all(k in d for k in SCHEMA["required"])
            and d["recommended_action"] in ACTIONS and d["severity"] in LEVELS and d["confidence"] in LEVELS)


class FlightDirector:
    def __init__(self, api_key: str, model: str, rpm: int, timeout: float, cache_dir: Path, lookup: Lookup,
                 sources: Iterable[str] = DEFAULT_SOURCES, daily_limit: int = 18,
                 sim_lookup: Callable[[list[str], float], Awaitable[list[dict]]] | None = None):
        self.model, self.timeout, self.lookup, self.sim_lookup = model, timeout, lookup, sim_lookup
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
        if ctx.get("kind") == "sim":
            return await self._diagnose_sim(ctx)
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

    async def _diagnose_sim(self, ctx: dict) -> dict:
        """SIM-1: Gemini reasons over named parts (2 requests); naive rules otherwise."""
        if not self.client:
            return {**rule_diagnosis_sim(ctx), "source": "rules"}
        if not self.uses_gemini_for(ctx["source"]):
            self.stats["skipped_source"] += 1
            return {**rule_diagnosis_sim(ctx), "source": "rules"}
        async with self.queue_lock:
            if not self._has_budget(2):
                self.stats["quota_blocked"] += 1
                return {**rule_diagnosis_sim(ctx), "source": "rules-quota"}
            try:
                return {**await self._gemini_sim(ctx), "source": "gemini"}
            except QuotaExhausted:
                self.stats["quota_blocked"] += 1
                return {**rule_diagnosis_sim(ctx), "source": "rules-quota"}
            except Exception as e:
                log.warning("Gemini (SIM-1) failed (%s); using rules.", e)
                self.stats["fallbacks"] += 1
                return {**rule_diagnosis_sim(ctx), "source": "rules-fallback"}

    async def _gemini_sim(self, ctx: dict) -> dict:
        from google.genai import types
        decl = types.FunctionDeclaration(
            name="get_sensor_history",
            description="Summary of SIM-1 sensors over the last N mission minutes: min, max, mean, "
                        "first and latest value, and the normal range. Use it to see what changed first "
                        "or to compare sunlit and eclipse behavior.",
            parameters_json_schema={"type": "object", "properties": {
                "sensor_ids": {"type": "array", "items": {"type": "string"}},
                "minutes": {"type": "number", "description": "5 to 150 mission minutes"}},
                "required": ["sensor_ids"]},
        )
        prompt = "New SIM-1 incident:\n" + json.dumps(ctx, indent=1, default=str)
        contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]

        # Request 1: Gemini may pull sensor history before deciding.
        r1 = await self._call(contents, types.GenerateContentConfig(
            system_instruction=SIM_SYSTEM, temperature=0.2,
            tools=[types.Tool(function_declarations=[decl])],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)))
        calls = r1.function_calls or []
        if calls and r1.candidates and self.sim_lookup:
            contents.append(r1.candidates[0].content)
            parts = []
            for c in calls[:3]:
                args = dict(c.args or {})
                ids = [str(x) for x in (args.get("sensor_ids") or [ctx["channel"]])][:11]
                mins = max(5.0, min(float(args.get("minutes", 30) or 30), 150.0))
                parts.append(types.Part.from_function_response(
                    name=c.name, response={"result": await self.sim_lookup(ids, mins)}))
            contents.append(types.Content(role="user", parts=parts))
            self.stats["tool_calls"] = self.stats.get("tool_calls", 0) + len(parts)
        contents.append(types.Content(role="user", parts=[types.Part(text="Now give your decision as JSON.")]))

        # Request 2: strict JSON decision.
        r2 = await self._call(contents, types.GenerateContentConfig(
            system_instruction=SIM_SYSTEM, temperature=0.2,
            response_mime_type="application/json", response_json_schema=SIM_SCHEMA))
        d = json.loads(r2.text or "{}")
        if not _valid_sim(d):
            raise ValueError(f"invalid SIM-1 decision: {d}")
        d["evidence"] = [str(e) for e in d["evidence"]][:4]
        return d

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