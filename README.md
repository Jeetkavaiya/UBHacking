# Mission Control AI — backend

An always-on AI flight controller for spacecraft telemetry (UB Hacking 2026, Radial Atoms track).
It replays real NASA SMAP and Curiosity (MSL) telemetry, checks every reading for anomalies,
has a Gemini "flight director" diagnose each one, and acts with bounded autonomy.
Every step is spoken with ElevenLabs and saved to Tiger Data.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp env.template .env          # PowerShell: Copy-Item env.template .env. Add keys when you have them; everything works without them
./scripts/download_data.sh    # NASA data into ./data (needs a Kaggle API key)
uvicorn app.main:app --reload --port 8000
```

- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/api/health
- Watch events live: `python scripts/watch.py`
- Tests: `python -m pytest -q tests`

Without the NASA files, the server runs on clearly labeled **simulated** channels
(`data_mode: "synthetic"` everywhere). Missing keys switch to fallbacks:

| Missing | Fallback |
|---|---|
| `GEMINI_API_KEY` | Rule-based diagnoses (`source: "rules"`) |
| `ELEVENLABS_API_KEY` / `ELEVENLABS_VOICE_ID` | Voice lines sent as captions, `audio_url: null` |
| `DATABASE_URL` | In-memory only |

## How it works

```
NASA test data ──► Engine (loops forever, BASE_RATE readings/s/channel × speed)
                     │  detector runs on every reading
                     ├──► WebSocket "readings" ──► dashboard
                     ├──► Tiger Data `telemetry` hypertable
                     └──► flags ──► Incident manager ──► Flight director (Gemini, 2 requests)
                                         │                    └─ tool: get_related_channels (live + Tiger Data stats)
                                         ├─ tier 1: act now (monitor, call engineer)
                                         ├─ tier 2: act after a veto window (power cycle, switch to backup)
                                         ├─ tier 3: wait for operator approval (safe mode)
                                         └─ verify recovery → close, or escalate
```

**Detector (baseline).** Three checks calibrated on each channel's training data: *jump*
(spike), *envelope* (drift outside the trained range), *stuck* (flatline, only on channels
that are rarely still in training). The LSTM forecaster replaces this in Phase 4 behind the
same `update(value) -> Result` interface.

**Bounded autonomy.** The AI can only pick from a fixed action menu. Low confidence moves
tier 2 up to tier 3. One command per channel per `COMMAND_COOLDOWN`. If nobody answers an
approval request, it reminds every `REMIND_SECONDS` and never acts alone. Commands are
**simulated**: they clear the injected fault in the replay stream.

**Gemini budget.** Gemini is only called for incidents (2 requests each), never per reading.
Requests are queued one at a time, kept under `GEMINI_RPM`, retried on 429 errors, and
fall back to rules on any failure. Diagnoses for sample replays are cached for the demo.

## REST API

| Method | Path | What it does |
|---|---|---|
| GET | `/api/health` | Status, data mode, Gemini / voice / storage stats |
| GET | `/api/channels` | Channel list with spacecraft, group, labeled anomaly ranges |
| GET | `/api/channels/{id}/history?n=300` | Recent values and surprise scores |
| POST | `/api/control/speed` | `{"speed": 5}` (0.25–100) |
| POST | `/api/faults` | `{"channel": "P-1", "type": "spike" \| "drift" \| "flatline"}` (synthetic) |
| POST | `/api/replay?channel=P-1&which=0` | Jump to ~20 s before a real NASA labeled anomaly |
| GET | `/api/incidents?open_only=true` | Incidents, newest first |
| POST | `/api/incidents/{id}/approve` · `/decline` · `/cancel` | Operator decisions |
| GET | `/api/log?limit=100` | Mission log |
| POST | `/api/heartbeat` | Trigger the all-clear call now (demo) |

## WebSocket `/ws`

| `type` | When | Key fields |
|---|---|---|
| `hello` | On connect | `channels`, `history`, `incidents`, `log`, `data_mode`, `veto_seconds` |
| `readings` | Every 100 ms | `data[channel] = {i: data index[], v: value[], s: surprise[], x: injected flag[]}` |
| `event` | Every incident step | `event {kind, message, tag, incident_id}`, `incident` (full view) |
| `voice` | Every spoken line, in order | `text`, `priority` (high/low), `audio_url` or null, `incident_id` |

Event kinds: `anomaly_detected`, `agent_analyzing`, `diagnosis`, `veto_window`, `approval_needed`,
`reminder`, `action_taken`, `cancelled`, `declined`, `escalated`, `closed`, `heartbeat`,
`heartbeat_skipped`, `fault_injected`, `replay_started`, `system`.

The dashboard should be served from `frontend/` by this same server: published claude.ai
pages can't call other sites.

## Tiger Data

Set `DATABASE_URL` to your Tiger Cloud service. On start the app creates `telemetry`,
`incidents` and `events`, and turns `telemetry` and `events` into hypertables when
TimescaleDB is available (plain tables otherwise).

## Disclosure (for Devpost)

- **Data:** real NASA SMAP/MSL telemetry from the Telemanom dataset, replayed faster than
  real time. Values are anonymized and scaled to -1..1, so diagnoses are at group level.
- **Synthetic:** injected faults (tagged "Synthetic fault"), and all spacecraft commands.
- **Existing work used:** open-source libraries, the NASA dataset, and the Telemanom
  method (LSTM forecasting + error thresholds) that inspired the Phase 4 detector.
- **Built during the hackathon:** all code in this repository.
