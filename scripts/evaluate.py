"""Offline precision and recall of our BaselineDetector on ALL 82 NASA SMAP/MSL channels.

Event-level scoring, in the style of Telemanom (Hundman et al., KDD 2018):
  - a labeled anomaly sequence is caught (true positive) if any predicted sequence overlaps it,
  - a predicted sequence that overlaps no labeled sequence is a false positive,
  - a labeled sequence that nothing overlaps is a false negative.
Predicted sequences are runs of flagged readings, merged when the gap is <= --merge readings.

The detector is calibrated on each channel's train split and run once over its test split, with
the same settings as the live demo. NASA's pretrained Telemanom results are quoted only as a
reference point; they are not our detector.

Usage (from the repo root):
    python scripts/evaluate.py
    python scripts/evaluate.py --merge 0 --out results/eval_merge0.json
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import settings  # noqa: E402
from app.detector import BaselineDetector  # noqa: E402

TELEMANOM_REFERENCE = {"precision": 0.875, "recall": 0.800,
                       "source": "Hundman et al., KDD 2018 (LSTM + nonparametric thresholding, SMAP+MSL)"}


def predicted_sequences(flags: np.ndarray, merge: int) -> list[tuple[int, int]]:
    idx = np.flatnonzero(flags)
    if not len(idx):
        return []
    seqs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > merge + 1:
            seqs.append((int(start), int(prev)))
            start = i
        prev = i
    seqs.append((int(start), int(prev)))
    return seqs


def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def evaluate_channel(cid: str, data: Path, labels: list[tuple[int, int]], z: float, k: int,
                     still: int, margin: float, merge: int) -> dict:
    train = np.load(data / "train" / f"{cid}.npy")[:, 0].astype(float)
    test = np.load(data / "test" / f"{cid}.npy")[:, 0].astype(float)
    det = BaselineDetector(train, z, k, still, margin)
    flags = np.fromiter((det.update(float(v)).flagged for v in test), dtype=bool, count=len(test))
    pred = predicted_sequences(flags, merge)
    tp = sum(1 for lab in labels if any(overlaps(lab, p) for p in pred))
    fp = sum(1 for p in pred if not any(overlaps(lab, p) for lab in labels))
    return {"channel": cid, "length": len(test), "labeled": len(labels), "predicted": len(pred),
            "tp": tp, "fn": len(labels) - tp, "fp": fp, "stuck_check": det.check_stuck,
            "labels": labels, "predicted_sequences": pred}


def summarize(rows: list[dict]) -> dict:
    tp, fn, fp = (sum(r[k] for r in rows) for k in ("tp", "fn", "fp"))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"channels": len(rows), "labeled_sequences": tp + fn, "tp": tp, "fn": fn, "fp": fp,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(settings.data_dir))
    ap.add_argument("--z", type=float, default=settings.z_threshold)
    ap.add_argument("--consecutive", type=int, default=settings.consecutive)
    ap.add_argument("--merge", type=int, default=50, help="merge flagged runs closer than this many readings")
    ap.add_argument("--out", default="results/eval_baseline.json")
    a = ap.parse_args()

    data = Path(a.data)
    df = pd.read_csv(data / "labeled_anomalies.csv")
    dupes = df.chan_id[df.chan_id.duplicated()].tolist()
    if dupes:  # one channel is listed twice in NASA's file: keep every labeled sequence of both rows
        print(f"note: {dupes} listed more than once in labeled_anomalies.csv; their sequences are merged")
    seqs: dict[str, list] = {}
    for _, row in df.iterrows():
        seqs.setdefault(row.chan_id, []).extend(tuple(map(int, s)) for s in ast.literal_eval(row.anomaly_sequences))
    df = df.drop_duplicates("chan_id")
    t0 = time.time()
    rows, crafts = [], {}
    for _, row in df.iterrows():
        cid = row.chan_id
        if not (data / "test" / f"{cid}.npy").exists():
            print(f"skip {cid}: no data")
            continue
        labels = sorted(set(seqs[cid]))
        r = evaluate_channel(cid, data, labels, a.z, a.consecutive, settings.still_window,
                             settings.envelope_margin, a.merge)
        r["spacecraft"] = row.spacecraft
        rows.append(r)
        crafts.setdefault(row.spacecraft, []).append(r)
        print(f"{cid:6s} {row.spacecraft:4s} labeled {r['labeled']}  caught {r['tp']}  false alarms {r['fp']}")

    demo = [c.strip() for c in "P-1,S-1,E-1,A-1,D-1,G-1,P-10,T-4,F-7,C-1".split(",")]
    result = {
        "detector": "BaselineDetector (jump + envelope + stuck checks)",
        "settings": {"z_threshold": a.z, "consecutive": a.consecutive, "still_window": settings.still_window,
                     "envelope_margin": settings.envelope_margin, "merge_gap": a.merge},
        "scoring": "event-level: labeled sequence caught if any predicted sequence overlaps it",
        "all_channels": summarize(rows),
        "by_spacecraft": {k: summarize(v) for k, v in crafts.items()},
        "demo_channels": summarize([r for r in rows if r["channel"] in demo]),
        "reference_telemanom": TELEMANOM_REFERENCE,
        "seconds": round(time.time() - t0, 1),
        "channels": rows,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    s = result["all_channels"]
    print(f"\nALL {s['channels']} channels: precision {s['precision']:.1%}  recall {s['recall']:.1%}  "
          f"F1 {s['f1']:.1%}  (TP {s['tp']}, FN {s['fn']}, FP {s['fp']})")
    for k, v in result["by_spacecraft"].items():
        print(f"  {k}: precision {v['precision']:.1%}  recall {v['recall']:.1%}  ({v['channels']} channels)")
    d = result["demo_channels"]
    print(f"  demo set ({d['channels']} channels): precision {d['precision']:.1%}  recall {d['recall']:.1%}")
    print(f"Reference, NASA Telemanom: precision 87.5%, recall 80.0%. Saved {out} in {result['seconds']}s.")


if __name__ == "__main__":
    main()
