"""Learned forecaster detector (Phase 4), Telemanom-style with a small model.

Per channel, a ridge-regression forecaster learns to predict the next reading from the last
LAGS readings (and, when available, the commands sent at that moment, which NASA includes in
each file). An anomaly is a period where reality keeps disagreeing with the forecast:

  forecast check - smoothed forecast error, z-scored against errors on held-out training data
                   (replaces the baseline's raw "jump" check, the main source of false alarms:
                   a channel that normally steps in a learned pattern no longer looks strange)
  envelope       - value outside the training range (kept from the baseline: a one-step
                   forecaster follows slow drifts instead of flagging them)
  stuck          - frozen reading where training says that is rare (kept from the baseline)

Same interface as BaselineDetector: update(value) -> Result, reset(), info().
Training is a closed-form least-squares fit: milliseconds per channel, numpy only.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .detector import BaselineDetector, Result

LAGS = 30
RIDGE = 1e-2
SMOOTH = 2 / (30 + 1)   # EWMA of absolute forecast errors (Telemanom smooths over ~30 readings)
VAL_FRACTION = 0.2      # calibrate error statistics on data the model was not fit on


def _design(x: np.ndarray, cmds: np.ndarray | None, lags: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(x) - lags
    cols = [np.lib.stride_tricks.sliding_window_view(x, lags)[:n]]
    if cmds is not None and cmds.shape[1]:
        cols.append(cmds[lags:])           # commands sent at the predicted step
    cols.append(np.ones((n, 1)))
    return np.hstack(cols), x[lags:]


class ForecastDetector:
    def __init__(self, train: np.ndarray, z: float = 4.0, consecutive: int = 12, still_window: int = 30,
                 margin: float = 0.10, train_cmds: np.ndarray | None = None, lags: int = LAGS,
                 min_peak: float = 6.0):
        x = np.asarray(train, dtype=float)
        self.z, self.k, self.lags = z, consecutive, lags
        # Live pruning (Telemanom-style): a run of high readings only becomes an alarm once its
        # peak surprise reaches min_peak, i.e. clearly above ordinary forecast noise.
        self.min_peak = max(min_peak, z)
        self.use_cmds = train_cmds is not None and np.asarray(train_cmds).ndim == 2 and train_cmds.shape[1] > 0
        cmds = np.asarray(train_cmds, dtype=float) if self.use_cmds else None
        # Envelope + stuck checks come from the baseline (its jump check is not used).
        self.base = BaselineDetector(x, z, consecutive, still_window, margin)
        self.ok = len(x) > lags * 4
        if self.ok:
            cut = max(int(len(x) * (1 - VAL_FRACTION)), lags * 3)
            X, y = _design(x[:cut], cmds[:cut] if self.use_cmds else None, lags)
            A = X.T @ X + RIDGE * np.eye(X.shape[1])
            self.w = np.linalg.solve(A, X.T @ y)
            # Error statistics on the held-out tail (fit errors would be optimistic).
            Xv, yv = _design(x[cut - lags:], cmds[cut - lags:] if self.use_cmds else None, lags)
            err = np.abs(Xv @ self.w - yv)
            s, sm = err[0] if len(err) else 0.0, []
            for e in err:
                s = (1 - SMOOTH) * s + SMOOTH * e
                sm.append(s)
            sm = np.asarray(sm) if sm else np.array([0.0])
            self.mu = float(sm.mean())
            self.sd = max(float(sm.std()), 1e-4)
        self.reset()

    def reset(self) -> None:
        self.base.reset()
        self.hist: deque[float] = deque(maxlen=self.lags)
        self.s = getattr(self, "mu", 0.0)
        self.rs = self.base.mu
        self.run = 0
        self.run_peak = 0.0
        self.run_comp: str | None = None

    def update(self, v: float, cmds: np.ndarray | None = None) -> Result:
        scores: dict[str, float] = {}
        # Envelope and stuck from the baseline's state.
        b = self.base
        if self.hist:  # smoothed size of the raw step between readings (the baseline's jump statistic)
            self.rs = (1 - 0.2) * self.rs + 0.2 * abs(v - self.hist[-1])
        b.buf.append(v)
        out = max(v - b.hi, b.lo - v, 0.0)
        scores["envelope"] = self.z + out / (0.05 * b.range) if out > 0 else 0.0
        if b.check_stuck and len(b.buf) == b.w and float(np.std(b.buf)) < b.still_eps:
            scores["stuck"] = self.z + 1.0
        # Forecast error. When the last few readings are flat, a broken forecast means the channel
        # froze, so report it as "stuck" (shape flatline), not "jump" (shape spike).
        recent = list(b.buf)[-5:]
        frozen = len(recent) == 5 and float(np.std(recent)) < b.still_eps
        if self.ok and len(self.hist) == self.lags:
            feats = list(self.hist)
            if self.use_cmds:
                c = np.zeros(self.w.shape[0] - self.lags - 1) if cmds is None else np.asarray(cmds, dtype=float)
                feats = np.concatenate([feats, c, [1.0]])
            else:
                feats = np.concatenate([feats, [1.0]])
            pred = float(np.dot(self.w, feats))
            self.s = (1 - SMOOTH) * self.s + SMOOTH * abs(v - pred)
            fz = (self.s - self.mu) / self.sd
            # Name the shape like the baseline checks do (spike / drift / flatline): flat readings ->
            # frozen; a sudden step between readings -> spike; a gradual departure -> drift.
            raw_z = (self.rs - b.mu) / b.sd
            key = "stuck" if frozen else "jump" if raw_z >= self.z else "envelope"
            scores[key] = max(scores.get(key, 0.0), fz)
        self.hist.append(v)

        surprise = max(0.0, max(scores.values()) if scores else 0.0)
        hit = surprise >= self.z
        comp = next((c for c in ("stuck", "jump", "envelope") if scores.get(c, 0) >= self.z), None)
        self.run = self.run + 1 if hit else 0
        self.run_peak = max(self.run_peak, surprise) if hit else 0.0
        if self.run == 1:
            self.run_comp = comp  # the shape is how the run STARTED (the step fades by the time it alarms)
        flagged = self.run >= self.k and self.run_peak >= self.min_peak
        return Result(surprise, flagged, self.run_comp if hit else comp)

    # The engine and inject() read these from the detector.
    @property
    def train_std(self) -> float:
        return self.base.train_std

    @property
    def lo(self) -> float:
        return self.base.lo

    @property
    def hi(self) -> float:
        return self.base.hi

    @property
    def range(self) -> float:
        return self.base.range

    @property
    def check_stuck(self) -> bool:
        return self.base.check_stuck

    def info(self) -> dict:
        return {"type": "forecast", "model": f"ridge AR({self.lags})" + (" + commands" if self.use_cmds else ""),
                "min_peak": self.min_peak,
                "stuck_check": self.base.check_stuck, "envelope": [round(self.base.lo, 4), round(self.base.hi, 4)]}


class EnsembleDetector(ForecastDetector):
    """Alarm when EITHER the learned forecaster or the baseline's jump/envelope/stuck checks
    (with the same strict persistence) fire. They catch different anomaly types: the forecaster
    sees changes in a channel's learned dynamics, the baseline sees raw bursts of change."""

    def __init__(self, train: np.ndarray, z: float = 4.0, consecutive: int = 12, still_window: int = 30,
                 margin: float = 0.10, min_peak: float = 6.0, **kw):
        super().__init__(train, z, consecutive, still_window, margin, min_peak=min_peak, **kw)
        self.raw = BaselineDetector(np.asarray(train, dtype=float), z, consecutive, still_window, margin)

    def reset(self) -> None:
        super().reset()
        if hasattr(self, "raw"):
            self.raw.reset()

    def update(self, v: float, cmds: np.ndarray | None = None) -> Result:
        f = super().update(v, cmds)
        b = self.raw.update(v)
        if b.flagged and not f.flagged:
            return b
        if f.flagged:
            return f
        return f if f.surprise >= b.surprise else b

    def info(self) -> dict:
        return {**super().info(), "type": "ensemble"}
