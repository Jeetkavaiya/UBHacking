import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

# Fast settings for tests, set before the app is imported.
os.environ.update({
    "DATA_DIR": "/nonexistent", "BASE_RATE": "50", "VETO_SECONDS": "1", "RECOVER_SECONDS": "0.5",
    "SELF_RECOVER_SECONDS": "30", "VERIFY_TIMEOUT": "30", "HEARTBEAT_SECONDS": "3600",
    "GEMINI_API_KEY": "", "ELEVENLABS_API_KEY": "", "DATABASE_URL": "", "CACHE_DIR": "/tmp/mc-test-cache",
    "INCIDENT_COOLDOWN": "1", "COMMAND_COOLDOWN": "0",
})

from app.agent import rule_diagnosis  # noqa: E402
from app.data import load_channels, synthetic_channels  # noqa: E402
from app.detector import BaselineDetector  # noqa: E402

LABELS = Path(__file__).parent / "labeled_anomalies.csv"


def smooth(n, offset=0, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(offset, offset + n)
    return 0.5 * np.sin(2 * np.pi * t / 80) + rng.normal(0, 0.02, n)


def run(det, xs):
    return [det.update(float(v)) for v in xs]


def test_no_false_alarms_on_normal_data():
    det = BaselineDetector(smooth(4000))
    assert not any(r.flagged for r in run(det, smooth(3000, 4000, seed=2)))


@pytest.mark.parametrize("kind,comp", [("spike", "jump"), ("drift", "envelope"), ("flatline", "stuck")])
def test_detects_each_fault(kind, comp):
    det = BaselineDetector(smooth(4000))
    x = smooth(600, 4000, seed=3)
    if kind == "spike":
        x[300:320] += 1.2
    elif kind == "drift":
        x[300:] += np.linspace(0, 1.5, 300)
    else:
        x[300:] = x[300]
    res = run(det, x)
    hits = [i for i, r in enumerate(res) if r.flagged]
    assert hits and hits[0] >= 300, f"{kind} not detected after injection"
    assert res[hits[0]].component == comp


def test_loader_real_mode(tmp_path):
    shutil.copy(LABELS, tmp_path / "labeled_anomalies.csv")
    for sub in ("train", "test"):
        (tmp_path / sub).mkdir()
        for cid in ("P-1", "S-1", "M-1", "T-4"):
            np.save(tmp_path / sub / f"{cid}.npy", np.c_[smooth(500), np.zeros((500, 3))])
    chans, mode = load_channels(tmp_path, [], 4)
    assert mode == "real"
    assert {c.spacecraft for c in chans} == {"SMAP", "MSL"}
    p1 = next(c for c in chans if c.id == "P-1")
    assert p1.anomalies[0] == (2149, 2349) and p1.subsystem == "Power"


def test_loader_falls_back_to_synthetic():
    chans, mode = load_channels(Path("/nonexistent"), [], 12)
    assert mode == "synthetic" and all(c.synthetic for c in chans) and len(chans) == 12


def test_rules():
    base = {"channel": "P-1", "subsystem": "Power", "group": "P", "threshold": 4.0}
    assert rule_diagnosis({**base, "shape": "flatline", "peak_surprise": 5}, [])["recommended_action"] == "power_cycle_sensor"
    assert rule_diagnosis({**base, "shape": "spike", "peak_surprise": 5}, [])["recommended_action"] == "monitor"
    assert rule_diagnosis({**base, "shape": "drift", "peak_surprise": 40}, [])["recommended_action"] == "safe_mode"
    assert rule_diagnosis({**base, "group": "T", "subsystem": "Group T", "shape": "drift", "peak_surprise": 40},
                          [])["recommended_action"] == "switch_to_backup"


# ---------- end to end through the API ----------
@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c


def wait_for(client, pred, timeout=15):
    end = time.time() + timeout
    while time.time() < end:
        items = client.get("/api/incidents").json()
        hit = [i for i in items if pred(i)]
        if hit:
            return hit[0]
        time.sleep(0.2)
    raise AssertionError(f"timed out; incidents: {client.get('/api/incidents').json()}")


def test_health(client):
    h = client.get("/api/health").json()
    assert h["data_mode"] == "synthetic" and h["channels"] == 12 and h["storage"]["mode"] == "memory"
    assert h["sim_channels"] == 11 and h["mode"] == "both"


def test_flatline_full_cycle(client):
    r = client.post("/api/faults", json={"channel": "T-1", "type": "flatline"})
    assert r.status_code == 200
    inc = wait_for(client, lambda i: i["channel"] == "T-1" and i["status"] == "closed")
    assert inc["shape"] == "flatline" and inc["action"] == "power_cycle_sensor" and inc["tier"] == 2
    kinds = [e["kind"] for e in client.get("/api/log").json() if e["incident_id"] == inc["id"]]
    for k in ("anomaly_detected", "agent_analyzing", "diagnosis", "veto_window", "action_taken", "closed"):
        assert k in kinds


def test_cancel_in_veto_window_escalates(client):
    client.post("/api/faults", json={"channel": "A-1", "type": "flatline"})
    inc = wait_for(client, lambda i: i["channel"] == "A-1" and i["status"] == "veto_window")
    assert client.post(f"/api/incidents/{inc['id']}/cancel").status_code == 200
    wait_for(client, lambda i: i["id"] == inc["id"] and i["status"] == "escalated")
    assert client.post(f"/api/incidents/{inc['id']}/approve").status_code == 409


@pytest.mark.parametrize("cid,kind,action", [("D-1", "spike", "monitor"), ("D-2", "drift", "switch_to_backup")])
def test_incident_shape_matches_injected_fault(client, cid, kind, action):
    # The incident's shape must come from how the event started, not which check fired most often.
    assert client.post("/api/faults", json={"channel": cid, "type": kind}).status_code == 200
    inc = wait_for(client, lambda i: i["channel"] == cid and i["shape"] is not None)
    assert inc["shape"] == kind and inc["action"] == action


def test_busy_channel_rejected(client):
    client.post("/api/faults", json={"channel": "C-1", "type": "drift"})
    assert client.post("/api/faults", json={"channel": "C-1", "type": "spike"}).status_code == 409


def test_replay_and_speed(client):
    assert client.post("/api/control/speed", json={"speed": 5}).json()["speed"] == 5
    r = client.post("/api/replay", params={"channel": "R-1"}).json()
    assert r["channel"] == "R-1" and r["real_data"] is False


def test_websocket_hello_and_stream(client):
    with client.websocket_connect("/ws") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello" and len(hello["channels"]) == 12 + 11
        assert {c["kind"] for c in hello["channels"]} == {"nasa", "sim"} and len(hello["scenarios"]) == 6
        types = {ws.receive_json()["type"] for _ in range(5)}
        assert "readings" in types


def test_gemini_path_with_fake_client(tmp_path):
    """Exercises the two-request tool-calling flow without a real key."""
    import asyncio
    import json as _json
    from types import SimpleNamespace as NS
    from google.genai import types
    from app.agent import FlightDirector

    seen = {"lookups": 0, "calls": 0}

    async def lookup(cid, secs):
        seen["lookups"] += 1
        return [{"channel": "P-2", "surprise_now": 0.4}]

    class FakeModels:
        async def generate_content(self, model, contents, config):
            seen["calls"] += 1
            if config.tools:  # request 1: ask for the lookup tool
                call = NS(name="get_related_channels", args={"channel_id": "P-1", "window_seconds": 30})
                content = types.Content(role="model", parts=[types.Part(text="checking")])
                return NS(function_calls=[call], candidates=[NS(content=content)])
            return NS(text=_json.dumps({"subsystem": "Power", "severity": "medium", "likely_cause": "Frozen sensor.",
                                        "recommended_action": "power_cycle_sensor", "confidence": "high",
                                        "voice_summary": "Medium severity on P-1."}))

    fd = FlightDirector("", "fake", 60, 5, tmp_path, lookup)
    fd.client = NS(aio=NS(models=FakeModels()))
    ctx = {"channel": "P-1", "shape": "flatline", "source": "synthetic", "peak_surprise": 6,
           "threshold": 4, "subsystem": "Power", "group": "P"}
    d = asyncio.run(fd.diagnose(ctx))
    assert d["source"] == "gemini" and d["recommended_action"] == "power_cycle_sensor"
    assert seen["calls"] == 2 and seen["lookups"] >= 2


def test_mode_switch(client):
    assert client.post("/api/control/mode", json={"mode": "nasa"}).json()["mode"] == "nasa"
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        batch = next(m for m in (ws.receive_json() for _ in range(20)) if m["type"] == "readings")
        assert "BATT_V" not in batch["data"] and "P-1" in batch["data"]
    assert client.post("/api/control/mode", json={"mode": "both"}).json()["mode"] == "both"
    assert client.post("/api/control/mode", json={"mode": "mars"}).status_code == 422


def test_big_spike_still_called_spike():
    det = BaselineDetector(smooth(4000))
    x = smooth(600, 4000, seed=4)
    x[300:320] += 3.0  # far outside the training range too
    hits = [r for r in run(det, x) if r.flagged]
    assert hits and hits[0].component == "jump"

# ---------- Gemini quota guard (fake client, no real requests) ----------
def _fake_director(tmp_path, **kw):
    from types import SimpleNamespace as NS
    from app.agent import FlightDirector

    seen = {"calls": 0}

    async def lookup(cid, secs):
        return []

    class FakeModels:
        async def generate_content(self, model, contents, config):
            seen["calls"] += 1
            if config.tools:  # request 1: no tool call, the director adds related channels itself
                return NS(function_calls=[], candidates=[])
            import json as _json
            return NS(text=_json.dumps({"subsystem": "Group E", "severity": "low", "likely_cause": "Brief jump.",
                                        "recommended_action": "monitor", "confidence": "medium",
                                        "voice_summary": "Low severity on E-1."}))

    fd = FlightDirector("", "fake", 60, 5, tmp_path, lookup, **kw)
    fd.client = NS(aio=NS(models=FakeModels()))
    return fd, seen


def _ctx(source):
    return {"channel": "E-1", "shape": "spike", "source": source, "peak_surprise": 5,
            "threshold": 4, "subsystem": "Group E", "group": "E"}


@pytest.mark.parametrize("source,expect,calls", [
    ("synthetic", "gemini", 2),        # injected demo fault -> Gemini
    ("labeled-anomaly", "rules", 0),   # natural incident -> rules, no quota used
    ("detected", "rules", 0),
])
def test_gemini_only_for_demo_sources(tmp_path, source, expect, calls):
    import asyncio
    fd, seen = _fake_director(tmp_path)
    d = asyncio.run(fd.diagnose(_ctx(source)))
    assert d["source"] == expect and seen["calls"] == calls
    assert fd.quota()["daily_used"] == calls


def test_gemini_sources_all(tmp_path):
    import asyncio
    fd, seen = _fake_director(tmp_path, sources=["all"])
    assert asyncio.run(fd.diagnose(_ctx("detected")))["source"] == "gemini"


def test_gemini_daily_limit_falls_back_and_persists(tmp_path):
    import asyncio
    fd, seen = _fake_director(tmp_path, daily_limit=4)
    out = [asyncio.run(fd.diagnose(_ctx("synthetic")))["source"] for _ in range(3)]
    assert out == ["gemini", "gemini", "rules-quota"] and seen["calls"] == 4
    assert fd.stats["quota_blocked"] == 1
    # A restart must not reset today's count.
    fd2, seen2 = _fake_director(tmp_path, daily_limit=4)
    assert fd2.quota()["daily_used"] == 4
    assert asyncio.run(fd2.diagnose(_ctx("synthetic")))["source"] == "rules-quota" and seen2["calls"] == 0


# ---------- SIM-1: simulated satellite ----------
def _sim_engine(seed):
    from app.config import settings
    from app.replay import Engine
    from app.sim import Satellite
    return Engine([], settings, Satellite(seed))


def _first_flag(eng, max_steps):
    for i in range(max_steps):
        _, flags = eng.advance(1)
        if flags:
            cid, r = next(iter(flags.items()))
            return cid, r.component, i
    return None, None, None


def test_sim_healthy_satellite_has_no_false_alarms():
    from app.sim import ORBIT_STEPS
    eng = _sim_engine(5)
    assert _first_flag(eng, 3 * ORBIT_STEPS)[0] is None


@pytest.mark.parametrize("key,channels,check", [
    ("cosmic_ray", {"TX_T"}, None),
    ("eclipse_imaging", {"BATT_V"}, "envelope"),
    ("wheel_friction", {"WHEEL_I", "WHEEL_RPM", "WHEEL_T", "POINT_ERR"}, "envelope"),
    ("solar_degradation", {"BATT_V"}, "envelope"),
    ("heater_stuck", {"BATT_T"}, "envelope"),
    ("temp_sensor_frozen", {"BATT_T"}, "stuck"),
])
def test_sim_scenarios_are_detected(key, channels, check):
    from app.sim import ORBIT_STEPS
    eng = _sim_engine(7)
    eng.advance(700)
    eng.sim_start(key)
    cid, comp, _ = _first_flag(eng, 2 * ORBIT_STEPS)
    assert cid in channels and (check is None or comp == check)


def test_sim_commands_have_real_effects():
    eng = _sim_engine(7)
    eng.sim_start("wheel_friction")
    eng.advance(600)
    lo, hi = eng.states["WHEEL_I"].ch.normal_range()
    assert eng.sim.last_reading["WHEEL_I"] > hi            # worn wheel draws extra current
    assert eng.sim_command("switch_to_backup", "reaction_wheel")["ok"]
    eng.advance(400)
    assert lo <= eng.sim.last_reading["WHEEL_I"] <= hi and eng.sim.active["reaction_wheel"] == "B"
    assert not eng.sim_command("switch_to_backup", "reaction_wheel")["ok"]   # no spare left
    assert not eng.sim_command("switch_to_backup", "battery")["ok"]          # no backup battery

    eng = _sim_engine(7)
    eng.sim_start("temp_sensor_frozen")
    eng.advance(50)
    eng.sim_command("power_cycle_sensor", "battery_temp_sensor")              # a reboot does not help
    vals = {eng.advance(1)[0]["BATT_T"]["v"][0] for _ in range(20)}
    assert len(vals) == 1
    eng.sim_command("switch_to_backup", "battery_temp_sensor")                # backup sensor does
    vals = {eng.advance(1)[0]["BATT_T"]["v"][0] for _ in range(20)}
    assert len(vals) > 5


def _sim_wait(client, pred, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        hit = [i for i in client.get("/api/incidents?limit=500").json() if i["kind"] == "sim" and pred(i)]
        if hit:
            return hit[0]
        time.sleep(0.2)
    raise AssertionError(f"timed out; sim incidents: {[i for i in client.get('/api/incidents').json() if i['kind'] == 'sim']}")


def test_sim_wheel_friction_end_to_end(client):
    client.post("/api/sim/reset")
    client.post("/api/control/speed", json={"speed": 10})  # 20x can lag on a busy machine
    try:
        r = client.post("/api/sim/scenarios/wheel_friction")
        assert r.status_code == 200
        inc = _sim_wait(client, lambda i: i["status"] == "closed" and (i["truth"] or {}).get("scenario")
                        == "wheel_friction", timeout=40)
        assert len(inc["symptoms"]) >= 2                       # one fault, one incident, several symptoms
        assert inc["action"] == "switch_to_backup" and inc["target"] == "reaction_wheel"
        assert inc["commands"][0]["ok"] and inc["outcome"] == "recovered"
        assert inc["truth"]["root_cause"] == "reaction_wheel"  # answer key revealed after closing
        assert inc["grade"]["root_cause_ok"] and inc["grade"]["action_ok"]
        assert client.get("/api/sim").json()["active_units"]["reaction_wheel"] == "B"
    finally:
        client.post("/api/control/speed", json={"speed": 1})


def test_sim_naive_rules_blame_the_wrong_part(client):
    client.post("/api/sim/reset")
    # 5x keeps SIM-1 in sunlight for the whole test: in Earth's shadow a stuck-on heater only
    # holds the battery at a normal temperature, so the incident would really self-recover.
    client.post("/api/control/speed", json={"speed": 5})
    try:
        assert client.post("/api/sim/scenarios/heater_stuck").status_code == 200
        inc = _sim_wait(client, lambda i: i["source"] == "sim-scenario" and i["status"] == "escalated")
        assert inc["truth"] is None                            # hidden while the incident is open
        assert inc["grade"]["root_cause_said"] == "battery" and not inc["grade"]["root_cause_ok"]
        assert inc["commands"] and not inc["commands"][0]["ok"]   # "no backup battery"
        assert client.post("/api/sim/scenarios/cosmic_ray").status_code == 409
        board = client.get("/api/sim/scoreboard").json()
        assert board["totals"]["rules"]["incidents"] >= 2
        from app.main import A
        assert any(k.startswith("sim:") for k in A.manager.last_command)
        client.post("/api/sim/reset")
        closed = [i for i in client.get("/api/incidents?limit=500").json() if i["id"] == inc["id"]][0]
        assert closed["status"] == "closed" and closed["outcome"] == "reset"
        # A reset starts clean: no command cooldowns left over from earlier scenarios.
        assert not any(k.startswith("sim:") for k in A.manager.last_command)
    finally:
        client.post("/api/control/speed", json={"speed": 1})


# ---------- SIM-1 flight director (fake Gemini, no quota) ----------
def _sim_ctx_for(source="sim-scenario"):
    return {"kind": "sim", "channel": "BATT_V", "sensor_name": "Battery voltage", "units": "V", "part": "battery",
            "sensor_part": None, "normal_range": [7.75, 8.34], "spacecraft": "SIM-1", "subsystem": "Power",
            "group": "Power", "shape": "drift", "detector_checks": {"envelope": 3}, "peak_surprise": 5.0,
            "threshold": 4.0, "seconds_open": 1.0, "source": source, "symptoms": [],
            "channel_stats": {"value_now": 7.7}, "spacecraft_status": {"orbit": {"in_eclipse": False}}}


def _sim_director(tmp_path, decision):
    import json as _json
    from types import SimpleNamespace as NS
    from google.genai import types
    from app.agent import FlightDirector
    seen = {"calls": 0, "lookups": []}

    async def lookup(cid, secs):
        return []

    async def sim_lookup(ids, minutes):
        seen["lookups"].append((ids, minutes))
        return [{"sensor": i, "mean": 1.0} for i in ids]

    class FakeModels:
        async def generate_content(self, model, contents, config):
            seen["calls"] += 1
            if config.tools:
                call = NS(name="get_sensor_history", args={"sensor_ids": ["SOLAR_I", "BATT_V"], "minutes": 60})
                return NS(function_calls=[call], candidates=[NS(content=types.Content(role="model", parts=[types.Part(text="look")]))])
            return NS(text=_json.dumps(decision))

    fd = FlightDirector("", "fake", 60, 5, tmp_path, lookup, sim_lookup=sim_lookup)
    fd.client = NS(aio=NS(models=FakeModels()))
    return fd, seen


GOOD = {"subsystem": "Power", "severity": "high", "root_cause": "solar_array",
        "likely_cause": "The battery is draining in sunlight because the solar array gives a quarter of its normal current.",
        "evidence": ["Solar current 0.6 A vs 2.1 A expected", "In sunlight"], "recommended_action": "shed_load",
        "target": "payload", "fallback_action": "call_engineer", "fallback_target": "spacecraft",
        "confidence": "high", "voice_summary": "Solar array output loss. Switching off the payload."}


def test_gemini_sim_decision_uses_tool_and_names_root_cause(tmp_path):
    import asyncio
    fd, seen = _sim_director(tmp_path, GOOD)
    d = asyncio.run(fd.diagnose(_sim_ctx_for()))
    assert d["source"] == "gemini" and d["root_cause"] == "solar_array" and d["recommended_action"] == "shed_load"
    assert seen["calls"] == 2 and seen["lookups"] == [(["SOLAR_I", "BATT_V"], 60.0)]
    assert fd.quota()["daily_used"] == 2


def test_gemini_sim_invalid_answer_falls_back_to_rules(tmp_path):
    import asyncio
    fd, _ = _sim_director(tmp_path, {**GOOD, "root_cause": "flux_capacitor"})   # not a SIM-1 part
    d = asyncio.run(fd.diagnose(_sim_ctx_for()))
    assert d["source"] == "rules-fallback" and d["root_cause"] == "battery"


def test_gemini_sim_only_for_scenarios(tmp_path):
    import asyncio
    fd, seen = _sim_director(tmp_path, GOOD)
    assert asyncio.run(fd.diagnose(_sim_ctx_for("sim-nominal")))["source"] == "rules" and seen["calls"] == 0


def test_sim_plan_b_after_failed_fix(client):
    client.post("/api/sim/reset")
    client.post("/api/control/speed", json={"speed": 5})
    try:
        assert client.post("/api/sim/scenarios/temp_sensor_frozen").status_code == 200
        inc = _sim_wait(client, lambda i: i["status"] == "closed" and (i["truth"] or {}).get("scenario")
                        == "temp_sensor_frozen" and i["outcome"] != "reset", timeout=40)
        acts = [(c["action"], c["target"]) for c in inc["commands"]]
        assert acts == [("power_cycle_sensor", "battery_temp_sensor"), ("switch_to_backup", "battery_temp_sensor")]
        assert inc["attempt"] == 2 and inc["outcome"] == "recovered"
        assert inc["report"]["plan_b_used"] and len(inc["report"]["actions"]) == 2
        kinds = [e["kind"] for e in client.get("/api/log?limit=500").json() if e["incident_id"] == inc["id"]]
        assert "plan_b" in kinds
        row = [r for r in client.get("/api/sim/scoreboard").json()["incidents"] if r["incident"] == inc["id"]][0]
        assert row["plan_b_used"] and row["resolved"] and row["root_cause_ok"]
    finally:
        client.post("/api/control/speed", json={"speed": 1})
        client.post("/api/sim/reset")


# ---------- learned forecaster / ensemble detector ----------
@pytest.mark.parametrize("cls_name", ["ForecastDetector", "EnsembleDetector"])
def test_forecaster_quiet_on_normal_data(cls_name):
    import app.forecast as fc
    det = getattr(fc, cls_name)(smooth(4000))
    assert not any(det.update(float(v)).flagged for v in smooth(3000, 4000, seed=2))


@pytest.mark.parametrize("cls_name", ["ForecastDetector", "EnsembleDetector"])
@pytest.mark.parametrize("kind,comp", [("spike", "jump"), ("drift", "envelope"), ("flatline", "stuck")])
def test_forecaster_detects_each_fault_with_right_shape(cls_name, kind, comp):
    import app.forecast as fc
    det = getattr(fc, cls_name)(smooth(4000))
    x = smooth(700, 4000, seed=3)
    if kind == "spike":
        x[300:330] += 1.2
    elif kind == "drift":
        x[300:] += np.linspace(0, 1.5, 400)
    else:
        x[300:] = x[300]
    res = [det.update(float(v)) for v in x]
    hits = [i for i, r in enumerate(res) if r.flagged]
    assert hits and hits[0] >= 300, f"{kind} not detected after injection"
    assert res[hits[0]].component == comp
