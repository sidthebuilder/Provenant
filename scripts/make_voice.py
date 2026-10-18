"""Generate the demo narration as MP3 files with a Microsoft neural voice (edge-tts).

Reads every say('<section>', '<text>') line from ui/index.html and writes
ui/demo/<sha1(text)[:12]>.mp3, which the ?demo=1 walkthrough plays. Lines with
no MP3 fall back to the browser's voice.

Setup (optional, needs internet):  pip install edge-tts
Usage:
    python scripts/make_voice.py                                   # defaults below
    python scripts/make_voice.py en-GB-RyanNeural --force          # another voice, regenerate all
    python scripts/make_voice.py --rate=+15% --volume=-30% --force # faster, quieter
List voices:  edge-tts --list-voices
"""
import asyncio
import hashlib
import re
import sys
from pathlib import Path

try:
    import edge_tts
except ImportError:
    sys.exit("edge-tts is not installed. Run: .venv\\Scripts\\pip install edge-tts")

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "ui" / "index.html"
OUT = ROOT / "ui" / "demo"

# Defaults: warm US male voice, a little faster than normal, a little quieter.
VOICE = "en-US-AndrewMultilingualNeural"
RATE = "+10%"     # speaking speed, e.g. "+0%" normal, "+15%" faster, "-10%" slower
VOLUME = "-20%"   # loudness,      e.g. "+0%" normal, "-30%" quieter

FORCE = False
for a in sys.argv[1:]:
    if a == "--force":
        FORCE = True
    elif a.startswith("--rate="):
        RATE = a.split("=", 1)[1]
    elif a.startswith("--volume="):
        VOLUME = a.split("=", 1)[1]
    elif not a.startswith("--"):
        VOICE = a

SAY = re.compile(r"say\('((?:[^'\\]|\\.)*)',\s*'((?:[^'\\]|\\.)*)'\)")


def unescape(s: str) -> str:
    return re.sub(r"\\(.)", r"\1", s)


def clip_id(text: str) -> str:
    # Must match clipUrl() in ui/index.html.
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


async def main():
    texts = [unescape(t) for _, t in SAY.findall(UI.read_text(encoding="utf-8"))]
    if not texts:
        sys.exit("No say(...) lines found in ui/index.html")
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"voice {VOICE}, rate {RATE}, volume {VOLUME}, {len(texts)} lines -> {OUT}")
    for i, t in enumerate(texts, 1):
        path = OUT / f"{clip_id(t)}.mp3"
        if path.exists() and not FORCE:
            print(f"  [{i:2}] exists  {path.name}")
            continue
        try:
            await edge_tts.Communicate(t, VOICE, rate=RATE, volume=VOLUME).save(str(path))
        except Exception as e:  # noqa: BLE001
            sys.exit(f"edge-tts failed on line {i} with voice {VOICE}: {e}\n"
                     f"Check the voice name with: edge-tts --list-voices")
        print(f"  [{i:2}] wrote   {path.name}  {t[:60]}...")
    print("done. Open http://localhost:8000/?demo=1 and press Ctrl+F5")


asyncio.run(main())
