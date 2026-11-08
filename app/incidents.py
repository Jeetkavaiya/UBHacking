"""Incident manager: turns detector flags into incidents and runs each one through

    detected -> analyzing -> diagnosed -> (veto window | awaiting approval) -> acting
             -> monitoring -> closed      (or escalated if it doesn't recover)

Bounded autonomy:
  Tier 1  automatic, nothing sent to the spacecraft  (monitor, call_engineer)
  Tier 2  automatic after a veto window             (power_cycle_sensor, switch_to_backup, shed_load)
  Tier 3  needs operator approval                   (safe_mode)
  Tier 4  never: the AI cannot change limits, alarms, its own permissions,
          or send anything outside the action menu (enforced by code, not prompt)
Low confidence moves Tier 2 up to Tier 3.

NASA replay: one incident per channel; commands are simulated (they only clear injected faults).
SIM-1: one incident per fault, collecting every sensor that alarms while it is open;
commands really change the simulated satellite, and each incident is graded against the
scenario's answer key (hidden from the AI, and from the incident view until it closes).
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import time
from collections import Counter, deque
from dataclasses import dataclass, field, fields
from typing import Awaitable, Callable

from .agent import PAST, FlightDirector, label, past
from .config import Settings
from .detector import SHAPES, Result
from .replay import Engine
from .sim import DT, PARTS
from .storage import Storage

log = logging.getLogger(__name__)

TIER = {"monitor": 1, "call_engineer": 1, "power_cycle_sensor": 2, "switch_to_backup": 2, "shed_load": 2,
        "safe_mode": 3}
COMMANDS = {"power_cycle_sensor", "switch_to_backup", "safe_mode", "shed_load"}
TAGS = {"synthetic": "Synthetic fault", "sample-replay": "Sample replay (real NASA anomaly)",
        "labeled-anomaly": "Real NASA labeled anomaly", "detected": "",
        "sim-scenario": "Simulated satellite scenario", "sim-nominal": "Simulated satellite (no scenario running)"}
FALSE_ALARM = {"scenario": None, "root_cause": "none", "ok_actions": ["monitor"],
               "summary": "No scenario was running, so this was a false alarm."}

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
    # SIM-1 only
    kind: str = "nasa"
    name: str | None = None             # sensor name, e.g. "Battery voltage"
    symptoms: list = field(default_factory=list)   # every sensor that alarmed, in order
    target: str | None = None           # part the action is sent to
    commands: list = field(default_factory=list)   # commands sent and what they did
    truth: dict | None = None           # answer key: grading only, never sent to the AI
    grade: dict | None = None
    acted_step: int | None = None
    calm_since_step: int | None = None  # SIM-1 recovery is counted in readings (mission time)
    attempt: int = 1                    # 2 = plan B is running
    report: dict | None = None          # SIM-1 after-action report
    _wake: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    @property
    def open(self) -> bool:
        return self.status != "closed"

    @property
    def channels(self) -> list[str]:
        return [s["channel"] for s in self.symptoms] or [self.channel]

    def view(self) -> dict:
        d = {f.name: getattr(self, f.name) for f in fields(self) if not f.name.startswith("_")}
        d["components"] = dict(self.components)
        d["tag"] = TAGS.get(self.source, "")
        d["action_label"] = label(self.action, self.target) if self.action else None
        d["truth"] = self.truth if not self.open else None  # revealed once the incident is over
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

    def sim_open(self) -> Incident | None:
        return next((i for i in self.by_channel.values() if i.kind == "sim"), None)

    @staticmethod
    def where(inc: Incident) -> str:
        return f"{inc.name.lower()} on {inc.spacecraft}" if inc.kind == "sim" else inc.channel

    def _symptom(self, cid: str, r: Result, opened_at: float) -> dict:
        ch = self.engine.states[cid].ch
        return {"channel": cid, "name": ch.name, "check": r.component, "shape": SHAPES.get(r.component or "", None),
                "seconds_after_first": round(time.time() - opened_at, 1),
                "mission_time": self.engine.sim.clock() if self.engine.sim else None}

    # ---------- called by the engine every batch ----------
    async def on_batch(self, flags: dict[str, Result]) -> None:
        now = time.time()
        for cid, r in flags.items():
            inc = self.by_channel.get(cid)
            if inc is None and self.engine.states[cid].ch.kind == "sim":
                inc = self.sim_open()  # one SIM-1 fault, one incident: link the new symptom
                if inc:
                    inc.symptoms.append(self._symptom(cid, r, inc.opened_at))
                    self.by_channel[cid] = inc
                    await self.event("symptom", f"{inc.symptoms[-1]['name']} also out of its normal range "
                                                f"(linked to {inc.id}).", inc)
            if inc:
                inc.components[r.component] += 1
            elif now >= self.cooldown_until.get(cid, 0):
                self._open(cid, r)
        for inc in {id(i): i for i in self.by_channel.values()}.values():
            sur = max(self.engine.surprise(c) for c in inc.channels)
            inc.peak_surprise = max(inc.peak_surprise, sur)
            if inc.kind == "sim" and self.engine.sim:
                # SIM-1: calm = every symptom back below the alarm level, counted in readings so
                # it means the same at 1x and 20x (wall-clock streaks get broken by noise at speed).
                step = self.engine.sim.step
                calm = sur < self.s.z_threshold
                if calm and inc.calm_since_step is None:
                    inc.calm_since_step = step
                elif not calm:
                    inc.calm_since_step = None
                calm_for = step - inc.calm_since_step if calm else -1
                recovered = calm_for >= self.s.recover_seconds * self.s.base_rate
                self_recovered = calm_for >= self.s.self_recover_seconds * self.s.base_rate
                inc.calm_since = (inc.calm_since or now) if calm else None
            else:
                calm = sur < self.s.z_threshold * 0.5
                inc.calm_since = (inc.calm_since or now) if calm else None
                recovered = bool(inc.calm_since) and now - inc.calm_since >= self.s.recover_seconds
                self_recovered = bool(inc.calm_since) and now - inc.calm_since >= self.s.self_recover_seconds
            waiting = inc.status in ("veto_window", "awaiting_approval")
            if waiting and self_recovered:
                inc.decision = "recovered"
                inc._wake.set()
            if inc.status in ("monitoring", "escalated") and recovered:
                self._spawn(self._close(inc, "recovered"))

    def _open(self, cid: str, r: Result) -> None:
        st = self.engine.states[cid]
        inc = Incident(id=f"INC-{next(self._ids):04d}", channel=cid, spacecraft=st.ch.spacecraft,
                       subsystem=st.ch.subsystem, group=st.ch.group, source=self.engine.source_of(cid),
                       opened_at=time.time(), peak_surprise=r.surprise, kind=st.ch.kind, name=st.ch.name or None)
        inc.components[r.component] += 1
        inc.onset = r.component
        if inc.kind == "sim":
            inc.symptoms.append(self._symptom(cid, r, inc.opened_at))
            if inc.source == "sim-scenario":
                inc.truth = self.engine.sim.truth()
                self.engine.sim.scenario["incident_id"] = inc.id
            elif inc.source == "sim-nominal":
                inc.truth = dict(FALSE_ALARM)
        self.incidents[inc.id] = inc
        self.by_channel[cid] = inc
        self._spawn(self._run(inc))

    # ---------- the incident pipeline ----------
    def _nasa_ctx(self, inc: Incident) -> dict:
        return {"channel": inc.channel, "spacecraft": inc.spacecraft, "group": inc.group,
                "subsystem": inc.subsystem, "shape": inc.shape, "detector_checks": dict(inc.components),
                "peak_surprise": round(inc.peak_surprise, 2), "threshold": self.s.z_threshold,
                "seconds_open": round(time.time() - inc.opened_at, 1), "source": inc.source,
                "channel_stats": self.engine.stats(inc.channel)}

    def _sim_ctx(self, inc: Incident) -> dict:
        """What the flight director may see about a SIM-1 incident. No answer key in here."""
        ch = self.engine.states[inc.channel].ch
        return {"kind": "sim", "channel": inc.channel, "sensor_name": ch.name, "units": ch.units,
                "part": ch.part, "sensor_part": ch.sensor_part, "normal_range": ch.normal_range(),
                "spacecraft": inc.spacecraft, "subsystem": inc.subsystem, "group": inc.group,
                "shape": inc.shape, "detector_checks": dict(inc.components),
                "peak_surprise": round(inc.peak_surprise, 2), "threshold": self.s.z_threshold,
                "seconds_open": round(time.time() - inc.opened_at, 1), "source": inc.source,
                "symptoms": [dict(s) for s in inc.symptoms],
                "channel_stats": self.engine.stats(inc.channel),
                "all_sensors": self._sim_snapshot(),
                "spacecraft_status": self.engine.sim.context(),
                "past_incidents": self._sim_history(exclude=inc.id)}

    def _sim_snapshot(self) -> list[dict]:
        """Every SIM-1 sensor right now vs its normal range: the whole picture in one glance."""
        out = []
        for cid in self.engine.sim_ids():
            st = self.engine.states[cid]
            if not st.recent:
                continue
            lo, hi = st.ch.normal_range()
            v = st.recent[-1][1]
            out.append({"sensor": cid, "name": st.ch.name, "units": st.ch.units, "now": round(v, 3),
                        "normal_range": [lo, hi], "in_range": lo <= v <= hi,
                        "alarming": cid in self.by_channel, "anomaly_checked": st.ch.monitored})
        return out

    def _sim_history(self, exclude: str | None = None, n: int = 5) -> list[dict]:
        """Memory: the last finished SIM-1 incidents and what worked. Never includes the answer key."""
        done = [i for i in self.incidents.values()
                if i.kind == "sim" and i.id != exclude and i.status in ("closed", "escalated") and i.diagnosis]
        out = []
        for i in sorted(done, key=lambda x: x.opened_at)[-n:]:
            recovered = i.outcome == "recovered"
            out.append({"incident": i.id, "first_alarm": i.name, "symptoms": [s["name"] for s in i.symptoms],
                        "root_cause_said": i.diagnosis.get("root_cause"),
                        "commands": [{"action": c["action"], "target": c["target"], "accepted": c["ok"],
                                      "fixed_it": recovered and k == len(i.commands) - 1}
                                     for k, c in enumerate(i.commands)],
                        "outcome": i.outcome or i.status})
        return out

    def _make_report(self, inc: Incident) -> dict:
        """After-action report, built from the incident record (no extra Gemini requests)."""
        d = inc.diagnosis or {}
        root = d.get("root_cause")
        steps = [f"{c['mission_time']} {past(c['action'], c['target'])}: {c['result']}" for c in inc.commands]
        outcome = {"recovered": "Recovered after the action.", "self-recovered": "Recovered on its own.",
                   "reset": "Closed by a simulator reset."}.get(inc.outcome or "", "Escalated to the on-call engineer.")
        return {
            "headline": f"{inc.name} alarm on {inc.spacecraft}: "
                        + ("no part failed" if root == "none" else f"{PARTS.get(root, root)} at fault"),
            "symptoms": [f"+{s['seconds_after_first']}s {s['name']} ({s['shape'] or s['check']})" for s in inc.symptoms],
            "diagnosis": d.get("likely_cause"),
            "evidence": d.get("evidence", []),
            "diagnosed_by": d.get("source"),
            "actions": steps or ["No command sent."],
            "plan_b_used": inc.attempt > 1,
            "outcome": outcome,
        }

    def _grade(self, inc: Incident) -> dict | None:
        t, d = inc.truth, inc.diagnosis or {}
        if not t:
            return None
        ok = t["ok_actions"]
        return {"diagnosed_by": "gemini" if str(d.get("source", "")).startswith("gemini") else "rules",
                "root_cause_said": d.get("root_cause"), "root_cause_true": t["root_cause"],
                "root_cause_ok": d.get("root_cause") == t["root_cause"],
                "action_ok": inc.action in ok or f"{inc.action}:{inc.target}" in ok}

    async def _run(self, inc: Incident) -> None:
        try:
            where = self.where(inc)
            await self.event("anomaly_detected", f"Anomaly detected on {where} ({inc.subsystem}).", inc)
            if inc.kind == "sim":
                await self.say(f"Alert. {inc.name} out of its normal range on {inc.spacecraft}.", "high", inc.id)
            else:
                name = inc.subsystem if inc.subsystem.startswith("Group") else inc.subsystem.lower()
                await self.say(f"Alert. Anomaly on {inc.channel}, {name}.", "high", inc.id)

            await asyncio.sleep(1.0)  # let the event build up a little before diagnosing
            if not inc.open:
                return
            inc.status = "analyzing"
            await self.event("agent_analyzing", f"Flight director analyzing {where} and related channels.", inc)
            await self.say("Flight director analyzing. Checking related channels.", "high", inc.id)

            # Shape = how the event STARTED. A spike that stays out of range fires the envelope
            # check on later readings, so "most common check" would wrongly call it drift.
            comp = inc.onset or (inc.components.most_common(1)[0][0] if inc.components else "jump")
            inc.shape = SHAPES.get(comp, "spike")
            ctx = self._sim_ctx(inc) if inc.kind == "sim" else self._nasa_ctx(inc)
            d = await self.agent.diagnose(ctx)
            if not inc.open:
                return
            inc.diagnosis, inc.action = d, d["recommended_action"]
            inc.target = d.get("target") if inc.kind == "sim" else None
            inc.grade = self._grade(inc)
            inc.tier, inc.status = self._tier(inc), "diagnosed"
            if inc.kind == "sim" and d.get("root_cause"):
                root = "no part failed" if d["root_cause"] == "none" else f"root cause: {PARTS.get(d['root_cause'])}"
                head = f"{d['severity'].capitalize()} severity on {where}, {root}."
            else:
                head = f"{d['severity'].capitalize()} severity {inc.shape} on {where}."
            await self.event("diagnosis", f"{head} Recommended: {label(inc.action, inc.target).lower()} "
                                          f"(tier {inc.tier}).", inc)
            await self.say(d["voice_summary"], "high", inc.id)
            await self._dispatch(inc)
        except Exception:
            log.exception("incident %s failed", inc.id)
            await self._escalate(inc, "Internal error while handling the incident.")

    def _tier(self, inc: Incident) -> int:
        tier = TIER[inc.action]
        if tier == 2 and (inc.diagnosis or {}).get("confidence") == "low":
            tier = 3
        return tier

    async def _dispatch(self, inc: Incident) -> None:
        if inc.tier == 1:
            await self._execute(inc)
        elif inc.tier == 2:
            await self._veto_window(inc)
        else:
            await self._await_approval(inc)

    async def _plan_b(self, inc: Incident, why: str) -> bool:
        """The first fix failed: try the fallback the flight director gave with its diagnosis
        (no extra Gemini requests). Same autonomy tiers apply. At most one plan B."""
        d = inc.diagnosis or {}
        fa, ft = d.get("fallback_action"), d.get("fallback_target")
        if inc.kind != "sim" or inc.attempt > 1 or fa not in TIER or fa == "monitor" \
                or (fa, ft) == (inc.action, inc.target):
            return False
        inc.attempt += 1
        inc.action, inc.target = fa, ft
        inc.tier, inc.status, inc.decision = self._tier(inc), "diagnosed", None
        await self.event("plan_b", f"{why} Plan B: {label(fa, ft).lower()} (tier {inc.tier}).", inc)
        await self.say(f"First fix did not work. Plan B: {label(fa, ft).lower()}.", "high", inc.id)
        await self._dispatch(inc)
        return True

    def _act_phrase(self, inc: Incident) -> str:
        what = label(inc.action, inc.target)
        return what if inc.kind == "sim" else f"{what} on {inc.channel}"

    async def _veto_window(self, inc: Incident) -> None:
        secs = int(self.s.veto_seconds)
        inc.status, inc.deadline = "veto_window", time.time() + secs
        await self.event("veto_window", f"{self._act_phrase(inc)} in {secs} s unless cancelled.", inc)
        await self.say(f"{self._act_phrase(inc)} in {secs} seconds unless cancelled.", "high", inc.id)
        inc._wake.clear()
        try:
            await asyncio.wait_for(inc._wake.wait(), secs)
        except asyncio.TimeoutError:
            inc.decision = "timeout"
        inc.deadline = None
        if not inc.open:  # closed meanwhile (e.g. simulator reset)
            return
        if inc.decision == "cancel":
            await self.event("cancelled", f"Operator cancelled the action on {self.where(inc)}.", inc)
            await self._escalate(inc, "Action cancelled by operator.")
        elif inc.decision == "recovered":
            await self._close(inc, "self-recovered")
        else:
            await self._execute(inc)

    async def _await_approval(self, inc: Incident) -> None:
        inc.status = "awaiting_approval"
        await self.event("approval_needed", f"{self._act_phrase(inc)} needs operator approval.", inc)
        await self.say(f"Awaiting operator approval to {self._act_phrase(inc).lower()}.", "high", inc.id)
        inc._wake.clear()
        while True:
            try:
                await asyncio.wait_for(inc._wake.wait(), self.s.remind_seconds)
                break
            except asyncio.TimeoutError:
                await self.event("reminder", f"Still waiting for approval on {self.where(inc)}.", inc)
                await self.say(f"Reminder. {self.where(inc)} is still waiting for operator approval.", "high", inc.id)
        if not inc.open:
            return
        if inc.decision == "approve":
            await self._execute(inc, approved=True)
        elif inc.decision == "recovered":
            await self._close(inc, "self-recovered")
        else:
            await self.event("declined", f"Operator declined: {self._act_phrase(inc).lower()}.", inc)
            await self._escalate(inc, "Declined by operator.")

    async def _execute(self, inc: Incident, approved: bool = False) -> None:
        act, now = inc.action, time.time()
        res = None
        if act in COMMANDS:
            key = f"sim:{inc.target}:{act}" if inc.kind == "sim" else inc.channel
            if now - self.last_command.get(key, 0) < self.s.command_cooldown:
                await self._escalate(inc, "Command rate limit reached for this unit.")
                return
            self.last_command[key] = now
            if inc.kind == "sim":  # really changes the simulated satellite
                res = self.engine.sim_command(act, inc.target)
                inc.commands.append({"action": act, "target": inc.target, "ok": res["ok"],
                                     "result": res["message"], "mission_time": self.engine.sim.clock()})
            else:
                self.engine.clear_fault(inc.channel)  # simulated command takes effect on the stream
        inc.status, inc.acted_at = "monitoring", now
        if inc.kind == "sim":
            inc.acted_step = self.engine.sim.step
        how = "Approved by operator" if approved else "Automatic" if inc.tier == 1 else "Automatic after veto window"
        if inc.kind == "sim":
            extra = f" {res['message']}" if res else ""
            await self.event("action_taken", f"Action taken on {inc.spacecraft}: {past(act, inc.target).lower()}.{extra}",
                             inc, tag=how)
        else:
            await self.event("action_taken", f"Action taken: {PAST[act].lower()} on {inc.channel} (simulated).", inc, tag=how)
        if res is not None and not res["ok"]:
            why = f"The command could not help: {res['message']}"
            if not await self._plan_b(inc, why):
                await self._escalate(inc, why)
            return
        if act == "monitor":
            await self.say(f"No command needed. Watching {self.where(inc)} closely.", "high", inc.id)
        else:
            await self.say(f"Action taken. {past(act, inc.target)}. Monitoring for recovery.", "high", inc.id)
        if act == "call_engineer":
            inc.status = "escalated"
            if inc.kind == "sim":
                inc.report = self._make_report(inc)
        self._spawn(self._verify(inc))

    async def _verify(self, inc: Incident) -> None:
        attempt = inc.attempt
        if inc.kind == "sim" and self.engine.sim:
            # SIM-1 physics runs on mission time: wait SIM_VERIFY_MINUTES of it (capped in real time).
            end_step = (inc.acted_step or self.engine.sim.step) + int(self.s.sim_verify_minutes * 60 / DT)
            cap = time.time() + max(self.s.verify_timeout, 900)
            while inc.open and inc.status == "monitoring" and self.engine.sim.step < end_step and time.time() < cap:
                await asyncio.sleep(0.25)
        else:
            await asyncio.sleep(self.s.verify_timeout)
        if inc.open and inc.status == "monitoring" and inc.attempt == attempt:
            why = f"{self.where(inc)} has not recovered after the action."
            if not await self._plan_b(inc, why):
                await self._escalate(inc, why)

    async def _escalate(self, inc: Incident, why: str) -> None:
        if not inc.open:
            return
        inc.status = "escalated"
        if inc.kind == "sim":
            inc.report = self._make_report(inc)
        await self.event("escalated", f"{why} Escalated to the on-call engineer.", inc)
        await self.say(f"{self.where(inc)} escalated to the on-call engineer.", "high", inc.id)

    async def _close(self, inc: Incident, outcome: str) -> None:
        if self._mark_closed(inc, outcome):
            await self._announce_closed(inc, outcome)

    def _mark_closed(self, inc: Incident, outcome: str) -> bool:
        """State change only, no awaits, so callers can combine it atomically with other changes."""
        if not inc.open:
            return False
        inc.status, inc.outcome, inc.closed_at = "closed", outcome, time.time()
        for c in inc.channels:
            if self.by_channel.get(c) is inc:
                self.by_channel.pop(c, None)
        if outcome != "reset":  # an operator reset starts clean, with no cooldown
            until = time.time() + self.s.incident_cooldown
            for c in (self.engine.sim_ids() if inc.kind == "sim" else [inc.channel]):
                self.cooldown_until[c] = until
        sc = self.engine.sim.scenario if self.engine.sim else None
        if sc and sc.get("incident_id") == inc.id:
            sc["done"] = True  # later alarms are not attributed to this scenario
        if inc.kind == "sim":
            inc.report = self._make_report(inc)
        inc._wake.set()
        return True

    async def _announce_closed(self, inc: Incident, outcome: str) -> None:
        msg = {"self-recovered": "recovered on its own", "reset": "reset by the operator"}.get(outcome, "back to nominal")
        who = inc.spacecraft if inc.kind == "sim" else inc.channel
        await self.event("closed", f"{who} {msg}. Incident closed.", inc)
        if outcome != "reset":
            await self.say(f"{who} {msg}. Incident closed.", "high", inc.id)

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

    # ---------- SIM-1 ----------
    async def sim_reset(self) -> None:
        """Healthy satellite again. Closing the incidents and resetting the physics happen with no
        await in between, so the engine never sees one without the other."""
        closed = [i for i in self.open_incidents() if i.kind == "sim" and self._mark_closed(i, "reset")]
        self.engine.sim_reset()
        self.clear_sim_cooldowns()
        for inc in closed:
            await self._announce_closed(inc, "reset")

    def clear_sim_cooldowns(self) -> None:
        """A reset or a newly started scenario is deliberate: don't let old cooldowns hide it."""
        for c in self.engine.sim_ids():
            self.cooldown_until.pop(c, None)
        for k in [k for k in self.last_command if k.startswith("sim:")]:
            del self.last_command[k]

    def scoreboard(self) -> dict:
        """Every graded SIM-1 incident, and totals per diagnoser (rules vs Gemini)."""
        rows = []
        for inc in self.incidents.values():
            if inc.kind != "sim" or not inc.grade:
                continue
            resolved = (inc.outcome in ("recovered", "self-recovered")) if inc.status == "closed" and \
                inc.outcome != "reset" else False if inc.status == "escalated" else None
            first = inc.commands[0] if inc.commands else {"action": inc.action, "target": inc.target}
            rows.append({"incident": inc.id, "scenario": inc.truth.get("scenario"), "status": inc.status,
                         "first_action": first["action"], "first_target": first["target"],
                         "plan_b_used": inc.attempt > 1, **inc.grade, "resolved": resolved})
        totals: dict[str, dict] = {}
        for r in rows:
            t = totals.setdefault(r["diagnosed_by"], {"incidents": 0, "root_cause_ok": 0, "action_ok": 0, "both_ok": 0})
            t["incidents"] += 1
            t["root_cause_ok"] += r["root_cause_ok"]
            t["action_ok"] += r["action_ok"]
            t["both_ok"] += r["root_cause_ok"] and r["action_ok"]
        return {"incidents": rows, "totals": totals}

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
        total = sum(1 for st in self.engine.states.values() if st.ch.kind in self.engine.active)
        line = (f"Mission Control. All {total} channels nominal. "
                + (f"{n} {'incident' if n == 1 else 'incidents'} in the last {mins} minutes, all closed."
                   if n else f"No anomalies in the last {mins} minutes."))
        await self.event("heartbeat", "All-clear call: all channels nominal.")
        await self.say(line, "low", None)
