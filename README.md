# Mission Control AI — backend

An always-on AI flight controller for spacecraft telemetry (UB Hacking 2026, Radial Atoms track).
It watches two sources at once:

- **NASA replay:** real SMAP and Curiosity (MSL) telemetry. This proves the detector on real data
  (precision and recall). The channels are anonymized and recorded, so diagnoses stay at group
  level and commands can't change anything.
- **SIM-1:** a fictional, clearly simulated satellite with named parts. This proves the decision
  making: parts affect each other, commands really change what happens, and every fault scenario
  has a known true cause to grade the flight director against.

It checks every reading for anomalies, has a Gemini "flight director" diagnose each one, and
acts with bounded autonomy. Every step is spoken with ElevenLabs and saved to Tiger Data.

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

**Gemini quota guard.** With a key set, only incidents whose source is in `GEMINI_SOURCES`
(default `synthetic,sample-replay`, i.e. demo-triggered) go to Gemini. Natural incidents get
rule-based diagnoses (`source: "rules"`). `GEMINI_DAILY_LIMIT` (default 18, free tier is 20) caps
requests per day, counted across restarts in `cache/gemini_usage.json` and reset at midnight
Pacific. Past the cap, diagnoses come from rules (`source: "rules-quota"`). `/api/health` shows
`gemini.daily_used` / `daily_limit`.

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

**Detector.** NASA channels use the **ensemble** (`DETECTOR=ensemble`, see Results): a learned
per-channel forecaster OR three baseline checks calibrated on the channel's training data:
- *jump*: a spike
- *envelope*: a drift outside the trained range
- *stuck*: a flatline, checked only on channels that are rarely still in training

An alarm needs 12 bad readings in a row (`CONSECUTIVE`). SIM-1 uses the baseline checks with its
own persistence (`SIM_CONSECUTIVE=6`).

**Bounded autonomy.** The AI can only pick from a fixed action menu. Low confidence moves
tier 2 up to tier 3. One command per channel per `COMMAND_COOLDOWN`. If nobody answers an
approval request, it reminds every `REMIND_SECONDS` and never acts alone. Commands are
**simulated**: they clear the injected fault in the replay stream.

**Gemini budget.** Gemini is only called for incidents (2 requests each), never per reading.
Requests are queued one at a time, kept under `GEMINI_RPM`, retried on 429 errors, and
fall back to rules on any failure. Diagnoses for sample replays are cached for the demo.

## Results

**NASA detection** (`python scripts/evaluate.py [--detector baseline|forecast|ensemble]`, saved in `results/`).

- **Data:** every labeled channel. NASA's file has 82 rows; P-2 is listed twice, so this is 81 unique series and 105 labeled anomaly sequences.
- **Detector:** the same baseline detector and settings as the live demo.
- **Scoring:** event-level, Telemanom style. A labeled sequence counts as caught if any flagged run overlaps it. Flagged runs closer than 50 readings are merged.

| Detector (all 81 channels) | Precision | Recall | F1 | False alarms | File |
|---|---|---|---|---|---|
| Baseline, persistence 3 (first version) | 33.2% | 67.6% | 44.5% | 143 | `eval_baseline_k3.json` |
| Baseline, persistence 12 | 72.8% | 56.2% | 63.4% | 22 | `eval_baseline.json` |
| Learned forecaster alone | 74.7% | 59.1% | 66.0% | 21 | `eval_forecast.json` |
| **Ensemble: forecaster OR baseline (live default)** | **67.6%** | **69.5%** | **68.5%** | **35** | `eval_ensemble.json` |
| Ensemble on SMAP (54 channels) | 72.5% | 72.5% | | | |
| Ensemble on MSL (27 channels) | 59.0% | 63.9% | | | |
| Ensemble on the curated 10-channel demo set | 100% | 56.2% | | 0 | |
| *Reference: NASA Telemanom LSTM (Hundman et al., 2018)* | *87.5%* | *80.0%* | | | |

**The learned forecaster** (`app/forecast.py`):
- **Model:** for each channel, ridge regression predicts the next reading from the last 30 readings. It is fit in closed form on the train split.
- **Calibration:** forecast errors are smoothed and z-scored against errors on held-out training data.
- **What it replaces:** the baseline's raw jump check. The range and frozen checks stay, because a one-step forecaster follows slow drifts and frozen values instead of flagging them.
- **Pruning:** a run of high readings only becomes an alarm once its peak surprise reaches 8, which is Telemanom's pruning idea applied while the run is live.
- **Ensemble:** the live detector alarms when either the forecaster or the baseline fires, at the same persistence of 12 readings. The two catch different anomaly types.
- **Commands:** NASA's command columns were also tried as model inputs. They made results worse (precision 31%), because commands are rare in the training data, so they are not used.

**How the settings were chosen, honestly:** persistence (3/5/8/12/20) and the peak gate
(6/8/12) came from a sweep on these same labels. The whole sweep is in the files above, and each
setting moved SMAP and MSL in the same direction. Before tuning, recall was 67.6% and precision
33.2%. Precision is still below Telemanom, mostly on MSL. NASA's pretrained outputs are a
reference only, never our detector.

**SIM-1 decisions** (`GET /api/sim/scoreboard`). Naive rules found the true root cause in
**3 of 6** scenarios. They fail exactly where the cause sits in a different part than the sensor
that alarmed first: the imaging pass in eclipse, the solar array loss, and the stuck heater.

## SIM-1, the simulated satellite

A small Earth-orbiting satellite, fictional and labeled `SIMULATED` everywhere (`app/sim.py`).

**Orbit.** A 95-minute orbit with 35 minutes in Earth's shadow, compressed to 3 minutes at 1×.

**Sensors.** 11 sensors in physical units:

| Subsystem | Sensors |
|---|---|
| Power | `SOLAR_I` (A), `BATT_V` (V), `LOAD_P` (W, context only) |
| Thermal | `BATT_T`, `PANEL_T` (°C) |
| Attitude | `WHEEL_RPM`, `WHEEL_I`, `WHEEL_T`, `POINT_ERR` |
| Comms | `RADIO_SNR`, `TX_T` |

**How parts affect each other.** Sunlight drives solar current, which charges the battery. The
battery heater's thermostat reads the battery temperature sensor. A worn reaction wheel draws more
current, heats up, and loses pointing.

**Detection.** The same `BaselineDetector` is used as for NASA. It is calibrated on 8 healthy orbits,
with values scaled to −1..1 over the training range (NASA's preprocessing). A persistence filter
(`SIM_CONSECUTIVE=6` readings) gave 0 false alarms in 18 healthy test orbits.

**Incidents.** One fault gives one incident. Every sensor that alarms while it is open is linked
to it as a symptom, with timing.

**Commands are real here.** For example:
- `switch_to_backup` swaps unit A for B, for the wheel, the temperature sensor, the heater controller, or the radio.
- `power_cycle_sensor` restarts a unit.
- `shed_load` switches the payload off.
- `safe_mode` turns the payload off and points the array at the sun.

A command that can't help, such as "no backup battery", escalates.

| Scenario | What the operator sees first | True cause | Correct action |
|---|---|---|---|
| `cosmic_ray` | Transmitter temperature spike | none (radiation glitch) | monitor |
| `eclipse_imaging` | Battery voltage below normal, in eclipse | none (planned imaging pass) | monitor |
| `wheel_friction` | Wheel current, speed, temperature, pointing | reaction wheel | switch wheel to backup |
| `solar_degradation` | Battery voltage below normal, in sunlight | solar array | shed load / safe mode / call engineer |
| `heater_stuck` | Battery temperature too high | heater controller | power cycle or switch heater controller |
| `temp_sensor_frozen` | Battery temperature frozen | temperature sensor | switch to backup sensor (a reboot doesn't help) |

**Flight director on SIM-1.** Gemini gets a spacecraft handbook (how the parts work and
connect, but no fault signatures) and the full case file:
- every symptom, with timing
- all sensors against their normal ranges
- the orbit, eclipse and ops plan
- past incidents and which fixes worked

It may call `get_sensor_history` before answering. It must name the **root-cause part**, the
**part to act on**, the evidence, and a **plan B**. If the first fix fails (the command can't
help, or nothing recovers within `SIM_VERIFY_MINUTES`), the agent runs plan B under the same
autonomy tiers, with no extra Gemini request. Every SIM-1 incident ends with an after-action
`report`. Each diagnosis costs 2 requests, the same as for NASA.

**Answer key.** The true cause is never sent to the flight director. It appears in the incident
(`truth`) only after the incident closes. `GET /api/sim/scoreboard` grades root cause and action
per diagnoser (rules vs Gemini).

**Rules baseline.** The rule-based baseline blames the part the first alarming sensor describes,
with no knowledge of the orbit, the ops plan, or how parts affect each other.

## REST API

| Method | Path | What it does |
|---|---|---|
| GET | `/api/health` | Status, data mode, Gemini / voice / storage stats |
| GET | `/api/channels` | Channel list with spacecraft, group, labeled anomaly ranges |
| GET | `/api/channels/{id}/history?n=300` | Recent values and surprise scores |
| POST | `/api/control/speed` | `{"speed": 5}` (0.25–100) |
| POST | `/api/control/mode` | `{"mode": "nasa" \| "sim" \| "both"}`: which source streams |
| GET | `/api/sim` | SIM-1 status: orbit, eclipse, active units, ops plan, timeline (no answer key) |
| GET | `/api/sim/scenarios` | The 6 fault scenarios |
| POST | `/api/sim/scenarios/{key}` | Start one (409 while a SIM-1 incident is open) |
| POST | `/api/sim/reset` | Healthy satellite again; closes open SIM-1 incidents |
| GET | `/api/sim/scoreboard` | Root-cause and action accuracy, rules vs Gemini |
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

## Deploy

**Backend** (Docker, for example on Render).
- `demo_data/` is a 370 KB copy of the 10 demo channels: the value column only, as float32. It is built by `python scripts/make_demo_data.py`.
- The full 425 MB dataset stays local, so the image builds straight from GitHub.
- Set `GEMINI_API_KEY`, `ELEVENLABS_API_KEY` and `ELEVENLABS_VOICE_ID` as host environment variables, never in git.

**Dashboard** (static, for example on Vercel, root directory `frontend`).
- Put the backend URL in `frontend/config.js`.
- The page then calls that backend for REST, the WebSocket, and voice audio. CORS is open.
- Locally, uvicorn serves the dashboard itself and `config.js` leaves the address empty.

## Tiger Data

Set `DATABASE_URL` to your Tiger Cloud service. On start the app creates `telemetry`,
`incidents` and `events`, and turns `telemetry` and `events` into hypertables when
TimescaleDB is available (plain tables otherwise).

## Disclosure (for Devpost)

- **Data:** real NASA SMAP/MSL telemetry from the Telemanom dataset, replayed faster than
  real time. Values are anonymized and scaled to -1..1, so diagnoses are at group level.
- **Synthetic:** injected faults (tagged "Synthetic fault"), and all spacecraft commands.
- **Simulated satellite:** SIM-1 is a fictional satellite we built during the event. Its physics,
  sensors, fault scenarios and answer key are ours, and they are labeled "Simulated satellite"
  everywhere. Precision and recall are reported on the real NASA data only. Root-cause accuracy
  is reported on SIM-1 only.
- **Existing work used:** open-source libraries, the NASA dataset, and the Telemanom
  method (LSTM forecasting + error thresholds) that inspired the Phase 4 detector.
- **Built during the hackathon:** all code in this repository.
