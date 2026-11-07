"""The always-on stream. Replays each channel's NASA test data in a loop,
runs the detector on every reading, and applies injected faults."""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import numpy as np

from .config import Settings
from .data import Channel
from .detector import BaselineDetector, Result

log = logging.getLogger(__name__)
TICK = 0.1  # seconds between engine batches
FAULT_TYPES = ("spike", "drift", "flatline")


class Busy(Exception):
    pass


@dataclass
class Fault:
    type: str
    src: str            # synthetic | sample-replay
    start_step: int
    level: float
    size: float
    ramp: int


@dataclass
class ChannelState:
    ch: Channel
    det: BaselineDetector
    ptr: int = 0
    step: int = 0
    fault: Fault | None = None
    replay_until: int | None = None  # pointer index where a sample replay ends
    last: Result | None = None
    recent: deque = field(default_factory=lambda: deque(maxlen=3000))  # (step, value, surprise)

    def in_label(self, idx: int) -> bool:
        return any(a <= idx <= b for a, b in self.ch.anomalies)


Batch = dict[str, dict[str, list]]
OnBatch = Callable[[Batch, dict[str, Result]], Awaitable[None]]


class Engine:
    def __init__(self, channels: list[Channel], s: Settings):
        self.s = s
        self.states = {c.id: ChannelState(c, BaselineDetector(
            c.train, s.z_threshold, s.consecutive, s.still_window, s.envelope_margin)) for c in channels}
        self.speed = 1.0
        self._acc = 0.0
        self.on_batch: OnBatch | None = None
        self.started = time.time()

    # ---------- one reading ----------
    def _step(self, st: ChannelState) -> tuple[float, Result, bool]:
        idx = st.ptr
        v = float(st.ch.test[idx])
        st.ptr = (idx + 1) % len(st.ch.test)
        if st.ptr == 0:
            st.det.reset()  # loop seam, not an anomaly
        if st.replay_until is not None and idx > st.replay_until:
            st.replay_until = None
        injected = False
        f = st.fault
        if f:
            age = st.step - f.start_step
            if f.type == "spike":
                if age < self.s.base_rate * 2:
                    v += f.size
                    injected = True
                else:
                    st.fault = None
            elif f.type == "drift":
                v += f.size * min(age / f.ramp, 1.0)
                injected = True
            elif f.type == "flatline":
                v = f.level
                injected = True
        r = st.det.update(v)
        st.last = r
        st.step += 1
        st.recent.append((idx, v, r.surprise))
        return v, r, injected

    # ---------- main loop ----------
    async def run(self) -> None:
        log.info("Engine running: %d channels, %.0f readings/s each at 1x", len(self.states), self.s.base_rate)
        while True:
            t0 = time.perf_counter()
            self._acc += self.s.base_rate * self.speed * TICK
            n = min(int(self._acc), 5000)
            self._acc -= n
            batch: Batch = {}
            flags: dict[str, Result] = {}
            for cid, st in self.states.items():
                b = batch[cid] = {"i": [], "v": [], "s": [], "x": []}
                for _ in range(n):
                    idx = st.ptr
                    v, r, inj = self._step(st)
                    b["i"].append(idx)
                    b["v"].append(round(v, 5))
                    b["s"].append(round(r.surprise, 3))
                    b["x"].append(1 if inj else 0)
                    if r.flagged:
                        flags[cid] = r
            if self.on_batch and n:
                try:
                    await self.on_batch(batch, flags)
                except Exception:  # never let a consumer stop the stream
                    log.exception("on_batch failed")
            await asyncio.sleep(max(0.0, TICK - (time.perf_counter() - t0)))

    # ---------- controls ----------
    def set_speed(self, speed: float) -> None:
        self.speed = float(max(0.25, min(speed, 100)))

    def inject(self, cid: str, ftype: str, src: str = "synthetic") -> Fault:
        st = self.states[cid]
        if ftype not in FAULT_TYPES:
            raise ValueError(f"fault type must be one of {FAULT_TYPES}")
        if st.fault:
            raise Busy(f"{cid} already has an active fault")
        d = st.det
        last = st.recent[-1][1] if st.recent else float(st.ch.test[st.ptr])
        size = {"spike": max(6 * d.train_std, 0.3), "drift": 0.8 * d.range, "flatline": 0.0}[ftype]
        if ftype != "flatline" and last > (d.lo + d.hi) / 2:
            size = -size  # push away from where the value currently sits
        st.fault = Fault(ftype, src, st.step, last, size, ramp=int(self.s.base_rate * 12))
        return st.fault

    def clear_fault(self, cid: str) -> bool:
        st = self.states[cid]
        had = st.fault is not None
        st.fault = None
        if had:
            st.det.reset()
        return had

    def replay_labeled(self, cid: str | None = None, which: int = 0) -> dict:
        """Jump a channel to ~20 s before one of NASA's labeled anomalies."""
        cands = [cid] if cid else [c for c, st in self.states.items() if st.ch.anomalies]
        for c in cands:
            st = self.states[c]
            if st.ch.anomalies and not st.fault:
                a, b = st.ch.anomalies[min(which, len(st.ch.anomalies) - 1)]
                st.ptr = max(0, a - int(self.s.base_rate * 20))
                st.replay_until = b
                st.det.reset()
                return {"channel": c, "anomaly": [a, b], "starts_at_index": st.ptr}
        raise Busy("no channel with a labeled anomaly is free right now")

    # ---------- reads ----------
    def source_of(self, cid: str) -> str:
        st = self.states[cid]
        if st.fault:
            return st.fault.src
        if st.replay_until is not None:
            return "sample-replay"
        if st.in_label(max(st.ptr - 1, 0)):
            return "labeled-anomaly"
        return "detected"

    def surprise(self, cid: str) -> float:
        last = self.states[cid].last
        return last.surprise if last else 0.0

    def stats(self, cid: str, seconds: float = 30) -> dict:
        st = self.states[cid]
        n = max(int(seconds * self.s.base_rate), 5)
        rows = list(st.recent)[-n:]
        if not rows:
            return {"channel": cid}
        vals = np.array([r[1] for r in rows])
        half = len(vals) // 2 or 1
        return {
            "channel": cid,
            "group": st.ch.group,
            "subsystem": st.ch.subsystem,
            "spacecraft": st.ch.spacecraft,
            "mean_recent": round(float(vals[half:].mean()), 4),
            "mean_before": round(float(vals[:half].mean()), 4),
            "std_recent": round(float(vals[half:].std()), 4),
            "training_std": round(st.det.train_std, 4),
            "surprise_now": round(self.surprise(cid), 2),
            "surprise_max": round(max(r[2] for r in rows), 2),
        }

    def related(self, cid: str, seconds: float = 30) -> list[dict]:
        me = self.states[cid].ch
        same = [c for c, st in self.states.items() if c != cid and
                (st.ch.group == me.group or st.ch.spacecraft == me.spacecraft)]
        same.sort(key=lambda c: (self.states[c].ch.group != me.group, -self.surprise(c)))
        return [self.stats(c, seconds) for c in same[:6]]

    def history(self, cid: str, n: int = 300) -> dict:
        rows = list(self.states[cid].recent)[-n:]
        return {"i": [r[0] for r in rows], "v": [round(r[1], 5) for r in rows], "s": [round(r[2], 3) for r in rows]}
