"""Prints the live event stream. Usage: python scripts/watch.py [ws://localhost:8000/ws]"""
import asyncio
import json
import sys

import websockets


async def main(url):
    async with websockets.connect(url, max_size=None) as ws:
        async for raw in ws:
            m = json.loads(raw)
            if m["type"] == "hello":
                print(f"connected: {len(m['channels'])} channels, {m['data_mode']} data")
            elif m["type"] == "event":
                e = m["event"]
                print(f"[{e['kind']}] {e['message']}" + (f"  ({e['tag']})" if e.get("tag") else ""))
            elif m["type"] == "voice":
                print(f"   voice: \"{m['text']}\"" + ("  [audio]" if m.get("audio_url") else ""))

asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8000/ws"))
