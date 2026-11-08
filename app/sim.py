"""SIM-1: a fictional, clearly SIMULATED satellite with named parts.

Why it exists: the NASA data is real but anonymized and recorded, so nobody can say what a
channel measures, commands can't change it, and there is no true cause to grade answers
against. SIM-1 fills those gaps:
  - every sensor has a name and a physical unit,
  - parts affect each other (sunlight -> solar current -> battery -> heater ...),
  - commands really change what happens next,
  - each fault scenario has a known true cause (the answer key), never shown to the AI.

Time: one reading = one step = DT seconds of mission time. A 95-minute orbit is ORBIT_STEPS
readings (3 minutes at 1x with BASE_RATE=10). The satellite is in Earth's shadow (eclipse)
for about 35 minutes of each orbit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .data import Channel

SPACECRAFT = "SIM-1"
ORBIT_STEPS = 1800
ORBIT_MIN = 95.0
DT = ORBIT_MIN * 60 / ORBIT_STEPS          # mission seconds per reading (~3.2 s)
ECLIPSE = (0.60, 0.97)                     # orbit phase spent in Earth's shadow
PENUMBRA = 0.04                            # phase width of the sun <-> shadow ramp
BATTERY_WH = 50.0
SOLAR_MAX_A = 2.4                          # array current with the sun straight on
ARRAY_V = 28.0
HEATER_W, PAYLOAD_W, IMAGING_W, BUS_W, RADIO_W = 10.0, 8.0, 20.0, 11.0, 6.0
HEATER_ON_BELOW, HEATER_OFF_ABOVE = 8.0, 12.0   # battery thermostat, deg C


@dataclass(frozen=True)
class Sensor:
    id: str
    name: str
    subsystem: str
    units: str
    part: str                       # the part this reading describes
    noise: float
    sensor_part: str | None = None  # a separately replaceable sensor, if any
    monitored: bool = True          # False = shown, not anomaly-checked (switches with every load change)


SENSORS = [
    # Noise is sized so a healthy reading never looks "frozen" to the detector's stuck check.
    Sensor("SOLAR_I", "Solar array current", "Power", "A", "solar_array", 0.06),
    Sensor("BATT_V", "Battery voltage", "Power", "V", "battery", 0.012),
    Sensor("LOAD_P", "Total power draw", "Power", "W", "power_bus", 0.2, monitored=False),
    Sensor("BATT_T", "Battery temperature", "Thermal", "°C", "battery", 0.35, sensor_part="battery_temp_sensor"),
    Sensor("PANEL_T", "Solar panel temperature", "Thermal", "°C", "solar_array", 0.3),
    Sensor("WHEEL_RPM", "Reaction wheel speed", "Attitude", "rpm", "reaction_wheel", 18.0),
    Sensor("WHEEL_I", "Wheel motor current", "Attitude", "A", "reaction_wheel", 0.01),
    Sensor("WHEEL_T", "Wheel temperature", "Attitude", "°C", "reaction_wheel", 0.2),
    Sensor("POINT_ERR", "Pointing error", "Attitude", "deg", "star_tracker", 0.004),
    Sensor("RADIO_SNR", "Radio signal margin", "Comms", "dB", "radio", 0.3),
    Sensor("TX_T", "Transmitter temperature", "Comms", "°C", "radio", 0.2),
]
SENSOR = {s.id: s for s in SENSORS}

# Parts the flight director can name as a root cause or send a command to.
PARTS = {
    "none": "no part (expected behavior or a harmless glitch)",
    "solar_array": "solar array",
    "battery": "battery",
    "power_bus": "power distribution",
    "payload": "payload (science camera)",
    "heater": "battery heater controller",
    "battery_temp_sensor": "battery temperature sensor",
    "reaction_wheel": "reaction wheel",
    "star_tracker": "pointing sensor (star tracker)",
    "radio": "radio transmitter",
    "spacecraft": "whole spacecraft",
}
BACKUPS = {"reaction_wheel", "battery_temp_sensor", "heater", "radio"}  # parts with a spare (A -> B)

# Fault scenarios. "truth" is the answer key: used only for grading, never sent to the AI.
SCENARIOS = {
    "cosmic_ray": {
        "title": "Radiation glitch",
        "operator_hint": "A charged particle flips a reading on the radio for a moment.",
        "start_phase": None,
        "truth": {"root_cause": "none", "ok_actions": ["monitor"],
                  "summary": "Single-event upset: one corrupted reading, no hardware fault."}},
    "eclipse_imaging": {
        "title": "Imaging pass in eclipse",
        "operator_hint": "A planned camera pass runs during Earth's shadow, draining the battery deeper than usual.",
        "start_phase": 0.565,
        "truth": {"root_cause": "none", "ok_actions": ["monitor"],
                  "summary": "Planned imaging during eclipse. Deep discharge is expected and recovers in sunlight."}},
    "wheel_friction": {
        "title": "Reaction wheel friction",
        "operator_hint": "The active reaction wheel's bearing wears and drags.",
        "start_phase": None,
        "truth": {"root_cause": "reaction_wheel", "ok_actions": ["switch_to_backup:reaction_wheel"],
                  "summary": "Bearing friction in wheel A. The backup wheel fixes pointing."}},
    "solar_degradation": {
        "title": "Solar array power loss",
        "operator_hint": "The solar array loses most of its output, so the battery stops charging.",
        "start_phase": 0.99,
        "truth": {"root_cause": "solar_array", "ok_actions": ["shed_load", "safe_mode", "call_engineer"],
                  "summary": "Solar array output fell to about 25%. Not fixable by command: save power and call engineers."}},
    "heater_stuck": {
        "title": "Heater stuck on",
        "operator_hint": "The battery heater controller latches on and ignores its thermostat.",
        "start_phase": 0.02,   # just after sunrise: in shadow a stuck-on heater looks normal
        "truth": {"root_cause": "heater", "ok_actions": ["power_cycle_sensor:heater", "switch_to_backup:heater"],
                  "summary": "Heater controller A latched on. A power cycle or the backup controller fixes it."}},
    "temp_sensor_frozen": {
        "title": "Frozen temperature sensor",
        "operator_hint": "Battery temperature sensor A freezes; the heater thermostat trusts it.",
        "start_phase": 0.50,
        "truth": {"root_cause": "battery_temp_sensor",
                  "ok_actions": ["switch_to_backup:battery_temp_sensor", "power_cycle_sensor:battery_temp_sensor"],
                  "summary": "Sensor A died. A reboot does not help; backup sensor B does."}},
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


class Satellite:
    """Physics of SIM-1, one step per reading. Deterministic for a given seed."""

    def __init__(self, seed: int = 2):
        self.rng = np.random.default_rng(seed)
        self.step = 0
        self.timeline: list[dict] = []   # operator-visible log: commands, plans, mode changes
        self.scenario: dict | None = None
        self.reset_state()

    # ---------- state ----------
    def reset_state(self) -> None:
        self.soc = 0.95
        self.batt_t, self.panel_t, self.tx_t = 20.0, 30.0, 36.0
        self.wheel_t = {"A": 24.0, "B": 23.0}
        self.perr_extra = 0.0
        self.mode = "nominal"
        self.payload_on = True
        self.imaging: dict | None = None        # {"start": step, "end": step} or {"at_eclipse": minutes}
        self.heater_on = False
        self.active = {p: "A" for p in sorted(BACKUPS)}
        # Faults (all off when nominal)
        self.friction = {"A": 0.0, "B": 0.0}
        self.friction_rate = {"A": 0.0, "B": 0.0}
        self.solar_eff, self.solar_target, self.solar_rate = 1.0, 1.0, 0.0
        self.heater_latched = {"A": False, "B": False}
        self.sensor_frozen: dict[str, float | None] = {"A": None, "B": None}
        self.glitch_left, self.glitch_size = 0, 0.0
        self.last_reading: dict[str, float] = {}
        self.sunlit_solar: list[float] = []      # solar current samples of the current sunlit arc
        self.last_sunlit_avg: float | None = None

    def reset(self) -> None:
        """Back to a healthy satellite. The mission clock keeps running."""
        self.reset_state()
        self.scenario = None
        self.note("Simulator reset: all parts healthy, primary units active, normal mode.")

    def note(self, text: str) -> None:
        self.timeline = (self.timeline + [{"t": self.clock(), "step": self.step, "text": text}])[-30:]

    # ---------- orbit ----------
    def phase(self, step: int | None = None) -> float:
        return ((self.step if step is None else step) % ORBIT_STEPS) / ORBIT_STEPS

    @staticmethod
    def sun_fraction(ph: float) -> float:
        a, b = ECLIPSE
        h = PENUMBRA / 2
        if a + h <= ph <= b - h:
            return 0.0
        if a - h < ph < a + h:
            return _clamp((a + h - ph) / PENUMBRA, 0, 1)
        if b - h < ph < b + h:
            return _clamp((ph - (b - h)) / PENUMBRA, 0, 1)
        return 1.0

    @staticmethod
    def _hump(ph: float) -> float:
        """0 at sunrise/sunset, 1 mid-day: how squarely the Earth-pointing array faces the sun."""
        sunlit = 1 - (ECLIPSE[1] - ECLIPSE[0])
        p = ((ph - ECLIPSE[1]) % 1.0) / sunlit
        return math.sin(math.pi * _clamp(p, 0, 1))

    def in_eclipse(self) -> bool:
        return self.sun_fraction(self.phase()) < 0.5

    def clock(self, step: int | None = None) -> str:
        s = int((self.step if step is None else step) * DT)
        return f"T+{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"

    def minutes_to_change(self) -> tuple[float, str]:
        ph = self.phase()
        a, b = ECLIPSE
        if self.in_eclipse():
            return (b - ph) % 1.0 * ORBIT_MIN, "sunlight"
        return (a - ph) % 1.0 * ORBIT_MIN, "eclipse"

    def expected_solar(self, ph: float | None = None) -> float:
        """Array current a healthy array would give right now (known from orbit geometry)."""
        ph = self.phase() if ph is None else ph
        point = 1.0 if self.mode == "safe" else 0.75 + 0.2 * self._hump(ph)
        return SOLAR_MAX_A * self.sun_fraction(ph) * point

    # ---------- physics ----------
    def tick(self) -> dict[str, float]:
        self.step += 1
        ph = self.phase()
        sun = self.sun_fraction(ph)
        n = self.rng.normal(0.0, 1.0, len(SENSORS))
        noise = {s.id: s.noise * n[k] for k, s in enumerate(SENSORS)}

        # Solar array
        self.solar_eff += _clamp(self.solar_target - self.solar_eff, -self.solar_rate, self.solar_rate)
        solar_i = self.expected_solar(ph) * self.solar_eff

        # Planned imaging pass (payload camera)
        if self.imaging and "at_eclipse" in self.imaging and sun < 0.5:
            mins = self.imaging["at_eclipse"]
            self.imaging = {"start": self.step, "end": self.step + int(mins * 60 / DT)}
            self.note(f"Planned imaging pass started ({mins:.0f} min, camera on).")
        imaging = bool(self.imaging and "start" in self.imaging and self.step < self.imaging["end"]
                       and self.payload_on and self.mode == "nominal")
        if self.imaging and "end" in self.imaging and self.step >= self.imaging["end"]:
            self.imaging = None
            self.note("Imaging pass finished, camera off.")

        # Reaction wheels (the active one controls pointing)
        for w in "AB":
            self.friction[w] = min(1.0, self.friction[w] + self.friction_rate[w])
        w = self.active["reaction_wheel"]
        f = self.friction[w]
        rpm = 3000 + 400 * math.sin(2 * math.pi * ph) - 1500 * f
        wheel_i = 0.22 + 0.05 * math.cos(2 * math.pi * ph) + 0.55 * f
        for k in "AB":
            target = 24 + 45 * (wheel_i - 0.22) if k == w else 23.0
            self.wheel_t[k] += (target - self.wheel_t[k]) * DT / 400
        self.perr_extra += (0.25 * f * f - self.perr_extra) / 60

        # Battery temperature: the thermostat trusts whichever sensor is active
        sens = self.active["battery_temp_sensor"]
        frozen = self.sensor_frozen[sens]
        batt_t_read = frozen if frozen is not None else self.batt_t + noise["BATT_T"]
        if self.heater_latched[self.active["heater"]]:
            self.heater_on = True
        elif batt_t_read < HEATER_ON_BELOW:
            self.heater_on = True
        elif batt_t_read > HEATER_OFF_ABOVE:
            self.heater_on = False
        t_env = -8 + 30 * sun
        self.batt_t += (t_env + 28 * self.heater_on - self.batt_t) * DT / 300
        self.panel_t += (-25 + 65 * sun - self.panel_t) * DT / 200
        self.tx_t += (26 + 1.3 * RADIO_W + 6 * sun - self.tx_t) * DT / 300
        snr = 9 + 1.5 * math.sin(2 * math.pi * 3 * ph)

        # Power balance and battery
        payload = self.payload_on and self.mode == "nominal"
        load = (BUS_W + RADIO_W + PAYLOAD_W * payload + IMAGING_W * imaging
                + HEATER_W * self.heater_on + 8 * wheel_i)
        net = solar_i * ARRAY_V * 0.92 - load
        if net > 0 and self.soc > 0.95:
            net *= max(0.0, (1 - self.soc) / 0.05)  # charge taper near full
        self.soc = _clamp(self.soc + net * DT / 3600 / BATTERY_WH, 0.0, 1.0)

        if sun > 0.99:
            self.sunlit_solar.append(solar_i)
        elif self.sunlit_solar:
            self.last_sunlit_avg = float(np.mean(self.sunlit_solar))
            self.sunlit_solar = []

        r = {
            "SOLAR_I": solar_i + noise["SOLAR_I"],
            "BATT_V": 6.9 + 1.4 * self.soc + noise["BATT_V"],
            "LOAD_P": load + noise["LOAD_P"],
            "BATT_T": batt_t_read,
            "PANEL_T": self.panel_t + noise["PANEL_T"],
            "WHEEL_RPM": rpm + noise["WHEEL_RPM"],
            "WHEEL_I": wheel_i + noise["WHEEL_I"],
            "WHEEL_T": self.wheel_t[w] + noise["WHEEL_T"],
            "POINT_ERR": abs(0.015 + self.perr_extra + noise["POINT_ERR"]),
            "RADIO_SNR": snr + noise["RADIO_SNR"],
            "TX_T": self.tx_t + noise["TX_T"],
        }
        if self.glitch_left > 0:  # single-event upset: corrupted readings, no physical change
            r["TX_T"] += self.glitch_size
            self.glitch_left -= 1
        self.last_reading = r
        return r

    def fast_forward_to(self, phase: float) -> int:
        """Skip ahead (nominal physics, no output) to an orbit phase. Returns steps skipped."""
        n = int(round((phase - self.phase()) % 1.0 * ORBIT_STEPS))
        for _ in range(n):
            self.tick()
        return n

    # ---------- scenarios ----------
    def start(self, key: str) -> dict:
        sc = SCENARIOS[key]
        skipped = self.fast_forward_to(sc["start_phase"]) if sc["start_phase"] is not None else 0
        if key == "cosmic_ray":
            self.glitch_left, self.glitch_size = 3, 25.0
        elif key == "eclipse_imaging":
            self.imaging = {"at_eclipse": 28.0}
            self.note("Ops plan: 28-minute imaging pass scheduled for the next eclipse (camera on, +20 W).")
        elif key == "wheel_friction":
            self.friction_rate[self.active["reaction_wheel"]] = 1 / 500
        elif key == "solar_degradation":
            self.solar_target, self.solar_rate = 0.25, 0.75 / 150
        elif key == "heater_stuck":
            self.heater_latched[self.active["heater"]] = True
        elif key == "temp_sensor_frozen":
            self.sensor_frozen[self.active["battery_temp_sensor"]] = round(self.batt_t + 0.1, 2)
        self.scenario = {"key": key, "title": sc["title"], "started_step": self.step,
                         "started_at": self.clock(), "skipped_minutes": round(skipped * DT / 60, 1),
                         "incident_id": None}
        return {**self.scenario, "operator_hint": sc["operator_hint"]}

    def truth(self) -> dict | None:
        if not self.scenario:
            return None
        return {"scenario": self.scenario["key"], **SCENARIOS[self.scenario["key"]]["truth"]}

    # ---------- commands (they really change the satellite) ----------
    def command(self, action: str, target: str | None) -> dict:
        """Returns {"ok": bool, "message": str, "reset": [sensor ids whose detectors restart]}."""
        target = target or "spacecraft"
        name = PARTS.get(target, target)
        reset: list[str] = []
        if action == "safe_mode":
            self.mode = "safe"
            msg, ok = "Safe mode: payload off, solar array turned to face the sun.", True
        elif action == "shed_load":
            self.payload_on = False
            if self.imaging:
                self.imaging = None
            msg, ok = "Payload switched off to save power (8 W, imaging cancelled).", True
        elif action == "switch_to_backup":
            if target not in BACKUPS:
                msg, ok = f"No backup exists for the {name}. Nothing changed.", False
            elif self.active[target] == "B":
                msg, ok = f"The {name} is already on its backup. No spare left.", False
            else:
                self.active[target] = "B"
                msg, ok = f"Switched the {name} to backup unit B.", True
                reset = {"reaction_wheel": ["WHEEL_RPM", "WHEEL_I", "WHEEL_T"],
                         "battery_temp_sensor": ["BATT_T"], "heater": [], "radio": ["TX_T", "RADIO_SNR"]}[target]
        elif action == "power_cycle_sensor":
            unit = self.active.get(target, "A")
            if target == "heater" and self.heater_latched[unit]:
                self.heater_latched[unit] = False
                msg, ok = "Heater controller restarted; it follows the thermostat again.", True
            elif target == "battery_temp_sensor":
                msg, ok = f"Battery temperature sensor {unit} restarted.", True
                reset = ["BATT_T"]
            elif target in PARTS and target not in ("none", "spacecraft"):
                msg, ok = f"The {name} was restarted.", True
            else:
                msg, ok = f"Cannot power cycle '{target}'.", False
        else:
            return {"ok": True, "message": "No command sent.", "reset": []}
        self.note(f"Command: {action.replace('_', ' ')} ({name}). {msg}")
        return {"ok": ok, "message": msg, "reset": reset}

    # ---------- what operators (and the AI) can see ----------
    def context(self) -> dict:
        """Operator view of the spacecraft. Contains no answer key."""
        mins, nxt = self.minutes_to_change()
        exp = self.expected_solar()
        img = None
        if self.imaging:
            img = (f"scheduled for the next eclipse ({self.imaging['at_eclipse']:.0f} min)"
                   if "at_eclipse" in self.imaging else
                   f"in progress, {max(0, self.imaging['end'] - self.step) * DT / 60:.0f} min left")
        return {
            "spacecraft": SPACECRAFT,
            "simulated": True,
            "mission_time": self.clock(),
            "orbit": {"number": self.step // ORBIT_STEPS + 1, "phase": round(self.phase(), 3),
                      "in_eclipse": self.in_eclipse(), "sunlight_percent": round(100 * self.sun_fraction(self.phase())),
                      "next": nxt, "minutes_to_next": round(mins, 1), "orbit_minutes": ORBIT_MIN,
                      "eclipse_minutes": round((ECLIPSE[1] - ECLIPSE[0]) * ORBIT_MIN)},
            "mode": self.mode,
            "payload": "on" if self.payload_on and self.mode == "nominal" else "off",
            "imaging_pass": img,
            "heater": "on" if self.heater_on else "off",
            "heater_thermostat_c": [HEATER_ON_BELOW, HEATER_OFF_ABOVE],
            "active_units": dict(self.active),
            "spares_left": {p: (1 if self.active[p] == "A" else 0) for p in self.active},
            "battery_charge_estimate_percent": round(self.soc * 100),
            "solar_current_expected_now_a": round(exp, 2),
            "solar_current_last_sunlit_avg_a": None if self.last_sunlit_avg is None else round(self.last_sunlit_avg, 2),
            "solar_current_healthy_sunlit_avg_a": round(SOLAR_MAX_A * (0.75 + 0.2 * 2 / math.pi), 2),
            "timeline": self.timeline[-8:],
        }

    def brief(self) -> dict:
        """Small status sent with every readings batch."""
        mins, nxt = self.minutes_to_change()
        return {"t": self.clock(), "eclipse": self.in_eclipse(), "phase": round(self.phase(), 4),
                "next": nxt, "minutes_to_next": round(mins, 1), "mode": self.mode,
                "payload": self.payload_on and self.mode == "nominal", "heater": self.heater_on,
                "imaging": bool(self.imaging and "start" in self.imaging), "active": dict(self.active),
                "battery_pct": round(self.soc * 100),
                "scenario": self.scenario and {k: self.scenario[k] for k in ("key", "title", "started_at")}}


@lru_cache(maxsize=2)
def nominal_training(orbits: int = 8, seed: int = 1) -> dict[str, np.ndarray]:
    """Healthy-satellite history used to calibrate the detector (like NASA's training split).
    Cached; treat the arrays as read-only."""
    sat = Satellite(seed)
    rows = [sat.tick() for _ in range(orbits * ORBIT_STEPS)]
    return {s.id: np.array([r[s.id] for r in rows]) for s in SENSORS}


def sim_channels() -> list[Channel]:
    """SIM-1 sensors as engine channels, calibrated on nominal orbits."""
    train = nominal_training()
    out = []
    for s in SENSORS:
        x = train[s.id]
        lo, hi = float(x.min()), float(x.max())
        out.append(Channel(s.id, SPACECRAFT, x, np.empty(0), synthetic=True, kind="sim", name=s.name,
                           units=s.units, part=s.part, sensor_part=s.sensor_part, monitored=s.monitored,
                           subsystem_name=s.subsystem, center=(hi + lo) / 2, half=max((hi - lo) / 2, 1e-9)))
    return out
