"""Turns spoken lines into audio with ElevenLabs.

Every line is cached by its text, so repeated phrases ("Flight director analyzing...")
are generated once and then cost nothing. Without a key, lines are still sent
as captions and the dashboard can use the browser's own speech.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

STATIC_LINES = [
    "Flight director analyzing. Checking related channels.",
    "Voice on. Mission Control is monitoring all channels.",
]


class Voice:
    def __init__(self, api_key: str, voice_id: str, model: str, audio_dir: Path):
        self.key, self.voice_id, self.model = api_key, voice_id, model
        self.dir = audio_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.stats = {"generated": 0, "cache_hits": 0, "failures": 0}
        self.client = httpx.AsyncClient(timeout=8) if self.enabled else None
        if not self.enabled:
            log.warning("ElevenLabs key or voice id missing: captions only.")

    @property
    def enabled(self) -> bool:
        return bool(self.key and self.voice_id)

    def _name(self, text: str) -> str:
        return hashlib.sha1(f"{self.voice_id}|{self.model}|{text}".encode()).hexdigest()[:16] + ".mp3"

    async def audio_for(self, text: str) -> str | None:
        """Returns the /audio/... URL, or None if voice is off or failed."""
        if not self.enabled:
            return None
        name = self._name(text)
        path = self.dir / name
        if path.exists():
            self.stats["cache_hits"] += 1
            return f"/audio/{name}"
        try:
            r = await self.client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}",
                params={"output_format": "mp3_44100_128"},
                headers={"xi-api-key": self.key, "Accept": "audio/mpeg"},
                json={"text": text, "model_id": self.model},
            )
            r.raise_for_status()
            path.write_bytes(r.content)
            self.stats["generated"] += 1
            return f"/audio/{name}"
        except Exception as e:
            self.stats["failures"] += 1
            log.warning("ElevenLabs failed (%s); caption only.", e)
            return None

    async def warm(self) -> None:
        await asyncio.gather(*(self.audio_for(t) for t in STATIC_LINES), return_exceptions=True)

    async def close(self) -> None:
        if self.client:
            await self.client.aclose()
