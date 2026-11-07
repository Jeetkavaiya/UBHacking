"""Baseline anomaly detector (Phase 1). The LSTM (Phase 4) plugs in with the same interface.

Three simple checks, each calibrated on the channel's own training data:
  jump     - readings change much faster than usual          -> looks like a spike
  envelope - the value leaves the range seen during training -> looks like a drift
  stuck    - the value freezes, which training says is rare  -> looks like a flatline

The surprise score is scaled so that z_threshold means "flag this".
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

ALPHA = 0.2  # smoothing for the jump check
SHAPES = {"jump": "spike", "envelope": "drift", "stuck": "flatline"}


@dataclass
class Result:
    surprise: float
    flagged: bool
    component: str | None  # which check fired: jump | envelope | stuck


def _ewma(x: np.ndarray, a: float) -> np.ndarray:
    out = np.empty_like(x)
    s = x[0] if len(x) else 0.0
    for i, v in enumerate(x):
        s = (1 - a) * s + a * v
        out[i] = s
    return out


class BaselineDetector:
    def __init__(self, train: np.ndarray, z: float = 4.0, consecutive: int = 3,
                 still_window: int = 30, margin: float = 0.10):
        x = np.asarray(train, dtype=float)
        self.z, self.k, self.w = z, consecutive, still_window

        # Jump check: typical smoothed step size and its spread.
        sm = _ewma(np.abs(np.diff(x)), ALPHA) if len(x) > 2 else np.array([0.0])
        self.mu = float(sm.mean())
        self.sd = max(float(sm.std()), 1e-4)

        # Envelope check: training range plus a margin.
        lo, hi = float(x.min()), float(x.max())
        self.range = max(hi - lo, 0.2)
        self.lo, self.hi = lo - margin * self.range, hi + margin * self.range

        # Stuck check: only enabled if training data is almost never still.
        self.train_std = max(float(x.std()), 1e-3)
        self.still_eps = 0.02 * self.train_std
        if len(x) > self.w:
            win = np.lib.stride_tricks.sliding_window_view(x, self.w)
            self.still_rate = float((win.std(axis=1) < self.still_eps).mean())
        else:
            self.still_rate = 1.0
        self.check_stuck = self.still_rate < 0.01
        self.reset()

    def reset(self) -> None:
        self.prev: float | None = None
        self.s = self.mu
        self.buf: deque[float] = deque(maxlen=self.w)
        self.run = 0

    def update(self, v: float) -> Result:
        if self.prev is not None:
            self.s = (1 - ALPHA) * self.s + ALPHA * abs(v - self.prev)
        self.prev = v
        self.buf.append(v)

        scores = {"jump": (self.s - self.mu) / self.sd}
        out = max(v - self.hi, self.lo - v, 0.0)
        scores["envelope"] = self.z + out / (0.05 * self.range) if out > 0 else 0.0
        if self.check_stuck and len(self.buf) == self.w and float(np.std(self.buf)) < self.still_eps:
            scores["stuck"] = self.z + 1.0

        surprise = max(0.0, max(scores.values()))
        hit = surprise >= self.z
        # Name the shape by priority: frozen > sudden jump > slow move out of range.
        comp = next((c for c in ("stuck", "jump", "envelope") if scores.get(c, 0) >= self.z), None)
        self.run = self.run + 1 if hit else 0
        flagged = self.run >= self.k
        return Result(surprise, flagged, comp)

    def info(self) -> dict:
        return {"type": "baseline", "stuck_check": self.check_stuck,
                "envelope": [round(self.lo, 4), round(self.hi, 4)]}
