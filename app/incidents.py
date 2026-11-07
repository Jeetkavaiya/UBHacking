"""Incident manager: turns detector flags into incidents and runs each one through

    detected -> analyzing -> diagnosed -> (veto window | awaiting approval) -> acting
             -> monitoring -> closed      (or escalated if it doesn't recover)

Bounded autonomy:
  Tier 1  automatic, nothing sent to the spacecraft  (monitor, call_engineer)
  Tier 2  automatic after a veto window             (power_cycle_sensor, switch_to_backup)
  Tier 3  needs operator approval                   (safe_mode)
  Tier 4  never: the AI cannot change limits, alarms, its own permissions,
          or send anything outside the action menu (enforced by code, not prompt)
Low confidence moves Tier 2 up to Tier 3.
All "commands" are simulated against the replay stream.
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import time
from collections import Counter, deque
from dataclasses import dataclass, field, fields
from typing import Awaitable, Callable

from .agent import LABEL, PAST, FlightDirector
from .config import Settings
from .detector import SHAPES, Result
from .replay import Engine
from .storage import Storage

log = logging.getLogger(__name__)

TIER = {"monitor": 1, "call_engineer": 1, "power_cycle_sensor": 2, "switch_to_backup": 2, "safe_mode": 3}
COMMANDS = {"power_cycle_sensor", "switch_to_backup", "safe_mode"}
TAGS = {"synthetic": "Synthetic fault", "sample-replay": "Sample replay (real NASA anomaly)",
        "labeled-anomaly": "Real NASA labeled anomaly", "detected": ""}

Broadcast = Callable[[dict], Awaitable[None]]


@dataclass
class Incident:
    id: str
    channel: str
    spacecraft: str
    subsystem: str
    group: str
    source: str
    opened_at: float
    status: str = "detected"
    peak_surprise: float = 0.0
    components: Counter = field(default_factory=Counter)
    onset: str | None = None            # detector check that fired first; decides the shape
    shape: str | None = None
    diagnosis: dict | None = None
    tier: int | None = None
    action: str | None = None
    deadline: float | None = None       # veto window end
    decision: str | None = None         # approve | decline | cancel | recovered
    acted_at: float | None = None
    calm_since: float | None = None
    closed_at: float | None = None
    outcome: str | None = None
    _wake: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    @property
    def open(self) -> bool:
        return self.status != "closed"

    def view(self) -> dict:
        d = {f.name: getattr(self, f.name) for f in fields(self) if not f.name.startswith("_")}
        d["components"] = dict(self.components)
        d["tag"] = TAGS.get(self.source, "")
        d["action_label"] = LABEL.get(self.action or "", None)
        return d


class Manager:
    def __init__(self, s: Settings, engine: Engine, agent: FlightDirector, storage: Storage,
                 broadcast: Broadcast, say: Callable[..., Awaitable[None]]):
        self.s, self.engine, self.agent, self.storage = s, engine, agent, storage
        self.broadcast, self.say = broadcast, say
        self.incidents: dict[str, Incident] = {}
        self.by_channel: dict[str, Incident] = {}
        self.cooldown_until: dict[str, float] = {}
        self.last_command: dict[str, float] = {}
        self.log: deque[dict] = deque(maxlen=500)
        self._ids = itertools.count(1)
        self.tasks: set[asyncio.Task] = set()

    # ---------- helpers ----------
    def _spawn(self, coro) -> None:
        t = asyncio.create_task(coro)
        self.tasks.add(t)
        t.add_done_callback(self.tasks.discard)

    async def event(self, kind: str, message: str, inc: Incident | None = None, tag: str | None = None) -> None:
        ev = {"ts": time.time(), "kind": kind, "message": message,
              "incident_id": inc.id if inc else None, "tag": tag if tag is not None else (inc and TAGS.get(inc.source))}
        self.log.appendleft(ev)
        await self.broadcast({"type": "event", "event": ev, "incident": inc.view() if inc else None})
        self._spawn(self.storage.save_event(ev))
        if inc:
            self._spawn(self.storage.save_incident(inc.view()))

    def open_incidents(self) -> list[Incident]:
        return [i for i in self.incidents.values() if i.open]

    # ---------- called by the engine every batch ----------
    async def on_batch(self, flags: dict[str, Result]) -> None:
        now = time.time()
        for cid, r in flags.items():
            inc = self.by_channel.get(cid)
            if inc:
                inc.components[r.component] += 1
            elif now >= self.cooldown_until.get(cid, 0):
                self._open(cid, r)
        for inc in list(self.by_channel.values()):
            sur = self.engine.surprise(inc.channel)
            inc.peak_surprise = max(inc.peak_surprise, sur)
            calm = sur < self.s.z_threshold * 0.5
            inc.calm_since = (inc.calm_since or now) if calm else None
            waiting = inc.status in ("veto_window", "awaiting_approval")
            if waiting and inc.calm_since and now - inc.calm_since >= self.s.self_recover_seconds:
                inc.decision = "recovered"
                inc._wake.set()
            if inc.status in ("monitoring", "escalated") and inc.calm_since and \
                    now - inc.calm_since >= self.s.recover_seconds:
                self._spawn(self._close(inc, "recovered"))

    def _open(self, cid: str, r: Result) -> None:
        st = self.engine.states[cid]
        inc = Incident(id=f"INC-{next(self._ids):04d}", channel=cid, spacecraft=st.ch.spacecraft,
                       subsystem=st.ch.subsystem, group=st.ch.group, source=self.engine.source_of(cid),
                       opened_at=time.time(), peak_surprise=r.surprise)
        inc.components[r.component] += 1
        inc.onset = r.component
        self.incidents[inc.id] = inc
        self.by_channel[cid] = inc
        self._spawn(self._run(inc))

    # ---------- the incident pipeline ----------
    async def _run(self, inc: Incident) -> None:
        try:
            await self.event("anomaly_detected", f"Anomaly detected on {inc.channel} ({inc.subsystem}).", inc)
            name = inc.subsystem if inc.subsystem.startswith("Group") else inc.subsystem.lower()
            await self.say(f"Alert. Anomaly on {inc.channel}, {name}.", "high", inc.id)

            await asyncio.sleep(1.0)  # let the event build up a little before diagnosing
            inc.status = "analyzing"
            await self.event("agent_analyzing", f"Flight director analyzing {inc.channel} and related channels.", inc)
            await self.say("Flight director analyzing. Checking related channels.", "high", inc.id)

            # Shape = how the event STARTED. A spike that stays out of range fires the envelope
            # check on later readings, so "most common check" would wrongly call it drift.
            comp = inc.onset or (inc.components.most_common(1)[0][0] if inc.components else "jump")
            inc.shape = SHAPES.get(comp, "spike")
            ctx = {"channel": inc.channel, "spacecraft": inc.spacecraft, "group": inc.group,
                   "subsystem": inc.subsystem, "shape": inc.shape, "detector_checks": dict(inc.components),
                   "peak_surprise": round(inc.peak_surprise, 2), "threshold": self.s.z_threshold,
                   "seconds_open": round(time.time() - inc.opened_at, 1), "source": inc.source,
                   "channel_stats": self.engine.stats(inc.channel)}
            d = await self.agent.diagnose(ctx)
            if not inc.open:
                return
            inc.diagnosis, inc.action = d, d["recommended_action"]
            tier = TIER[inc.action]
            if tier == 2 and d["confidence"] == "low":
                tier = 3
            inc.tier, inc.status = tier, "diagnosed"
            await self.event("diagnosis", f"{d['severity'].capitalize()} severity {inc.shape} on {inc.channel}. "
                                          f"Recommended: {LABEL[inc.action].lower()} (tier {tier}).", inc)
            await self.say(d["voice_summary"], "high", inc.id)

            if tier == 1:
                await self._execute(inc)
            elif tier == 2:
                await self._veto_window(inc)
            else:
                await self._await_approval(inc)
        except Exception:
            log.exception("incident %s failed", inc.id)
            await self._escalate(inc, "Internal error while handling the incident.")

    async def _veto_window(self, inc: Incident) -> None:
        secs = int(self.s.veto_seconds)
        inc.status, inc.deadline = "veto_window", time.time() + secs
        await self.event("veto_window", f"{LABEL[inc.action]} on {inc.channel} in {secs} s unless cancelled.", inc)
        await self.say(f"{LABEL[inc.action]} on {inc.channel} in {secs} seconds unless cancelled.", "high", inc.id)
        inc._wake.clear()
        try:
            await asyncio.wait_for(inc._wake.wait(), secs)
        except asyncio.TimeoutError:
            inc.decision = "timeout"
        inc.deadline = None
        if inc.decision == "cancel":
            await self.event("cancelled", f"Operator cancelled the action on {inc.channel}.", inc)
            await self._escalate(inc, "Action cancelled by operator.")
        elif inc.decision == "recovered":
            await self._close(inc, "self-recovered")
        else:
            await self._execute(inc)

    async def _await_approval(self, inc: Incident) -> None:
        inc.status = "awaiting_approval"
        await self.event("approval_needed", f"{LABEL[inc.action]} on {inc.channel} needs operator approval.", inc)
        await self.say(f"Awaiting operator approval to {LABEL[inc.action].lower()} on {inc.channel}.", "high", inc.id)
        inc._wake.clear()
        while True:
            try:
                await asyncio.wait_for(inc._wake.wait(), self.s.remind_seconds)
                break
            except asyncio.TimeoutError:
                await self.event("reminder", f"Still waiting for approval on {inc.channel}.", inc)
                await self.say(f"Reminder. {inc.channel} is still waiting for operator approval.", "high", inc.id)
        if inc.decision == "approve":
            await self._execute(inc, approved=True)
        elif inc.decision == "recovered":
            await self._close(inc, "self-recovered")
        else:
            await self.event("declined", f"Operator declined: {LABEL[inc.action].lower()} on {inc.channel}.", inc)
            await self._escalate(inc, "Declined by operator.")

    async def _execute(self, inc: Incident, approved: bool = False) -> None:
        act, now = inc.action, time.time()
        if act in COMMANDS:
            if now - self.last_command.get(inc.channel, 0) < self.s.command_cooldown:
                await self._escalate(inc, "Command rate limit reached for this channel.")
                return
            self.last_command[inc.channel] = now
            self.engine.clear_fault(inc.channel)  # simulated command takes effect on the stream
        inc.status, inc.acted_at = "monitoring", now
        how = "Approved by operator" if approved else "Automatic" if inc.tier == 1 else "Automatic after veto window"
        await self.event("action_taken", f"Action taken: {PAST[act].lower()} on {inc.channel} (simulated).", inc, tag=how)
        if act == "monitor":
            await self.say(f"No command needed. Watching {inc.channel} closely.", "high", inc.id)
        else:
            await self.say(f"Action taken. {PAST[act]} on {inc.channel}. Monitoring for recovery.", "high", inc.id)
        if act == "call_engineer":
            inc.status = "escalated"
        self._spawn(self._verify(inc))

    async def _verify(self, inc: Incident) -> None:
        await asyncio.sleep(self.s.verify_timeout)
        if inc.open and inc.status == "monitoring":
            await self._escalate(inc, f"{inc.channel} has not recovered after the action.")

    async def _escalate(self, inc: Incident, why: str) -> None:
        if not inc.open:
            return
        inc.status = "escalated"
        await self.event("escalated", f"{why} Escalated to the on-call engineer.", inc)
        await self.say(f"{inc.channel} escalated to the on-call engineer.", "high", inc.id)

    async def _close(self, inc: Incident, outcome: str) -> None:
        if not inc.open:
            return
        inc.status, inc.outcome, inc.closed_at = "closed", outcome, time.time()
        self.by_channel.pop(inc.channel, None)
        self.cooldown_until[inc.channel] = time.time() + self.s.incident_cooldown
        inc._wake.set()
        msg = "recovered on its own" if outcome == "self-recovered" else "back to nominal"
        await self.event("closed", f"{inc.channel} {msg}. Incident closed.", inc)
        await self.say(f"{inc.channel} {msg}. Incident closed.", "high", inc.id)

    # ---------- operator decisions ----------
    def decide(self, inc_id: str, decision: str) -> Incident:
        inc = self.incidents.get(inc_id)
        if not inc:
            raise KeyError(inc_id)
        ok = {"approve": "awaiting_approval", "decline": "awaiting_approval", "cancel": "veto_window"}
        if inc.status != ok[decision]:
            raise ValueError(f"{inc_id} is '{inc.status}', cannot {decision}")
        inc.decision = decision
        inc._wake.set()
        return inc

    # ---------- 5-minute all-clear ----------
    async def heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(self.s.heartbeat_seconds)
            await self.heartbeat()

    async def heartbeat(self) -> None:
        if self.open_incidents():
            await self.event("heartbeat_skipped", "All-clear call skipped while an incident is open.")
            return
        since = time.time() - self.s.heartbeat_seconds
        n = sum(1 for i in self.incidents.values() if i.opened_at >= since)
        mins = round(self.s.heartbeat_seconds / 60)
        total = len(self.engine.states)
        line = (f"Mission Control. All {total} channels nominal. "
                + (f"{n} {'incident' if n == 1 else 'incidents'} in the last {mins} minutes, all closed."
                   if n else f"No anomalies in the last {mins} minutes."))
        await self.event("heartbeat", "All-clear call: all channels nominal.")
        await self.say(line, "low", None)