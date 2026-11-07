"""All settings come from environment variables (or a .env file)."""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _i(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _list(name: str) -> list[str]:
    return [x.strip() for x in os.getenv(name, "").split(",") if x.strip()]


@dataclass(frozen=True)
class Settings:
    # Data
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "data")))
    channels: list[str] = field(default_factory=lambda: _list("CHANNELS"))
    max_channels: int = _i("MAX_CHANNELS", 12)

    # Stream: readings per channel per second at 1x speed
    base_rate: float = _f("BASE_RATE", 10)

    # Detector
    z_threshold: float = _f("Z_THRESHOLD", 4.0)
    consecutive: int = _i("CONSECUTIVE", 3)
    still_window: int = _i("STILL_WINDOW", 30)
    envelope_margin: float = _f("ENVELOPE_MARGIN", 0.10)

    # Incident handling (real seconds)
    recover_seconds: float = _f("RECOVER_SECONDS", 3)
    self_recover_seconds: float = _f("SELF_RECOVER_SECONDS", 10)
    verify_timeout: float = _f("VERIFY_TIMEOUT", 60)
    veto_seconds: float = _f("VETO_SECONDS", 30)
    remind_seconds: float = _f("REMIND_SECONDS", 180)
    command_cooldown: float = _f("COMMAND_COOLDOWN", 120)
    incident_cooldown: float = _f("INCIDENT_COOLDOWN", 30)
    heartbeat_seconds: float = _f("HEARTBEAT_SECONDS", 300)

    # Gemini
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    gemini_rpm: int = _i("GEMINI_RPM", 8)
    gemini_timeout: float = _f("GEMINI_TIMEOUT", 25)

    # ElevenLabs
    elevenlabs_api_key: str = os.getenv("ELEVENLABS_API_KEY", "")
    elevenlabs_voice_id: str = os.getenv("ELEVENLABS_VOICE_ID", "")
    elevenlabs_model: str = os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5")

    # Storage (Tiger Data / PostgreSQL). Empty = in-memory only.
    database_url: str = os.getenv("DATABASE_URL", "")
    store_every: int = _i("STORE_EVERY", 1)

    # Files
    cache_dir: Path = field(default_factory=lambda: Path(os.getenv("CACHE_DIR", "cache")))
    frontend_dir: Path = field(default_factory=lambda: Path(os.getenv("FRONTEND_DIR", "frontend")))


settings = Settings()
