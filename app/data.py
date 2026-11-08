"""Loads NASA SMAP/MSL telemetry from DATA_DIR.

Expected layout (what scripts/download_data.sh produces):
    data/labeled_anomalies.csv
    data/train/<channel>.npy
    data/test/<channel>.npy

Column 0 of each .npy file is the telemetry value (pre-scaled to -1..1);
the other columns are command flags, which we ignore.

If the files are missing, we fall back to SIMULATED channels so the
backend still runs. Every API response says which mode is active.
"""
from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Only these letters are documented; everything else is shown as its letter.
GROUP_NAMES = {"P": "Power", "R": "Radiation"}


@dataclass
class Channel:
    id: str
    spacecraft: str
    train: np.ndarray
    test: np.ndarray
    anomalies: list[tuple[int, int]] = field(default_factory=list)
    synthetic: bool = False
    # Simulated satellite (kind="sim") channels are live, named and in physical units.
    kind: str = "nasa"
    name: str = ""
    units: str = ""
    part: str = ""
    sensor_part: str | None = None
    monitored: bool = True
    subsystem_name: str = ""
    # The detector sees (value - center) / half, i.e. -1..1 over the training range,
    # the same scaling NASA applied to its data. NASA channels are already scaled.
    center: float = 0.0
    half: float = 1.0

    @property
    def group(self) -> str:
        return self.subsystem_name or self.id.split("-")[0]

    @property
    def subsystem(self) -> str:
        return self.subsystem_name or GROUP_NAMES.get(self.group, f"Group {self.group}")

    def to_det(self, v: float) -> float:
        return (v - self.center) / self.half

    def from_det(self, x: float) -> float:
        return x * self.half + self.center

    def normal_range(self) -> list[float]:
        return [round(float(self.train.min()), 4), round(float(self.train.max()), 4)]

    def meta(self) -> dict:
        m = {
            "id": self.id,
            "spacecraft": self.spacecraft,
            "group": self.group,
            "subsystem": self.subsystem,
            "length": int(len(self.test)),
            "labeled_anomalies": [list(a) for a in self.anomalies],
            "synthetic": self.synthetic,
            "kind": self.kind,
        }
        if self.kind == "sim":
            m.update(name=self.name, units=self.units, part=self.part, monitored=self.monitored,
                     normal_range=self.normal_range())
        return m


def _values(path: Path) -> np.ndarray:
    arr = np.load(path)
    return (arr[:, 0] if arr.ndim == 2 else arr).astype(np.float64)


def _pick(df: pd.DataFrame, n: int) -> list[str]:
    """Pick up to n channels: half SMAP, half MSL, spread across groups."""
    picked: list[str] = []
    for craft, quota in (("SMAP", (n + 1) // 2), ("MSL", n // 2)):
        rows = df[df.spacecraft == craft]
        seen, first_pass, rest = set(), [], []
        for cid in rows.chan_id:
            g = cid.split("-")[0]
            (rest if g in seen else first_pass).append(cid)
            seen.add(g)
        picked += (first_pass + rest)[:quota]
    return picked[:n]


def load_real(data_dir: Path, wanted: list[str], max_n: int) -> list[Channel]:
    df = pd.read_csv(data_dir / "labeled_anomalies.csv").drop_duplicates("chan_id")
    df = df[[(data_dir / "test" / f"{c}.npy").exists() and (data_dir / "train" / f"{c}.npy").exists()
             for c in df.chan_id]]
    if df.empty:
        return []
    ids = [c for c in wanted if c in set(df.chan_id)] if wanted else _pick(df, max_n)
    missing = set(wanted) - set(ids)
    if missing:
        log.warning("Channels not found in data dir, skipped: %s", sorted(missing))
    rows = df.set_index("chan_id")
    out = []
    for cid in ids:
        r = rows.loc[cid]
        seqs = [tuple(map(int, s)) for s in ast.literal_eval(r.anomaly_sequences)]
        out.append(Channel(cid, str(r.spacecraft), _values(data_dir / "train" / f"{cid}.npy"),
                           _values(data_dir / "test" / f"{cid}.npy"), sorted(seqs)))
    return out


def synthetic_channels(n: int, seed: int = 7) -> list[Channel]:
    """Simulated stand-in data. Clearly marked synthetic everywhere."""
    rng = np.random.default_rng(seed)
    ids = ["P-1", "P-2", "T-1", "T-2", "A-1", "A-2", "C-1", "C-2", "D-1", "D-2", "R-1", "R-2"][:n]
    out = []
    for k, cid in enumerate(ids):
        period, noise, phase = rng.uniform(40, 200), rng.uniform(0.01, 0.03), rng.uniform(0, 6)

        def sig(length: int, offset: int = 0) -> np.ndarray:
            t = np.arange(offset, offset + length)
            x = 0.5 * np.sin(2 * np.pi * t / period + phase) + 0.15 * np.sin(2 * np.pi * t / (period * 0.37))
            return x + rng.normal(0, noise, length)

        train, test = sig(4000), sig(6000, 4000)
        start = 2500 + 300 * k
        test[start:start + 150] += 0.9  # one labeled anomaly per channel
        out.append(Channel(cid, "SIMULATED", train, test, [(start, start + 150)], synthetic=True))
    return out


def load_channels(data_dir: Path, wanted: list[str], max_n: int) -> tuple[list[Channel], str]:
    if (data_dir / "labeled_anomalies.csv").exists() and (data_dir / "test").is_dir():
        chans = load_real(data_dir, wanted, max_n)
        if chans:
            log.info("Loaded %d real NASA channels: %s", len(chans), [c.id for c in chans])
            return chans, "real"
    log.warning("NASA data not found in %s. Running on SIMULATED channels.", data_dir)
    return synthetic_channels(max_n), "synthetic"
