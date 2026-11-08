"""Build a small copy of the NASA data for deployment: only the demo channels, only the
telemetry value (column 0; the command columns are not used), stored as float32.

The full dataset (~425 MB) stays out of git. This copy (< 1 MB) is committed so a cloud
host can build from GitHub. Source: NASA SMAP/MSL anomaly dataset (Hundman et al., 2018).

Usage (from the repo root):
    python scripts/make_demo_data.py            # data/ -> demo_data/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEMO = "P-1,S-1,E-1,A-1,D-1,G-1,P-10,T-4,F-7,C-1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data")
    ap.add_argument("--dst", default="demo_data")
    ap.add_argument("--channels", default=DEMO)
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    ids = [c.strip() for c in a.channels.split(",") if c.strip()]
    for sub in ("train", "test"):
        (dst / sub).mkdir(parents=True, exist_ok=True)
        for cid in ids:
            arr = np.load(src / sub / f"{cid}.npy")
            np.save(dst / sub / f"{cid}.npy", (arr[:, 0] if arr.ndim == 2 else arr).astype(np.float32))
    df = pd.read_csv(src / "labeled_anomalies.csv")
    df[df.chan_id.isin(ids)].to_csv(dst / "labeled_anomalies.csv", index=False)
    size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
    print(f"Wrote {len(ids)} channels to {dst}/ ({size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
