"""Call every Track 3 tool on real village data and print short results."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from provenant.village import brain as B  # noqa: E402

calls = [
    ("who_owns", {"site": "ai-village-agents.github.io"}),
    ("who_owns", {"site": "grok-ai-village-news-496089.gitlab.io"}),
    ("sites_of", {"agent": "Claude Opus 4.5"}),
    ("who_knows", {"topic": "Juice Shop OWASP vulnerability"}),
    ("decisions", {"topic": "charity fundraiser donation"}),
    ("goal_team", {"goal_text": "charity"}),
    ("handoffs", {"agent": "GPT-5.2"}),
]
out = {}
for name, args in calls:
    t0 = time.perf_counter()
    try:
        r = B.TOOLS[name](**args)
        ms = round((time.perf_counter() - t0) * 1000)
        out[f"{name}:{list(args.values())[0]}"] = r
        print(f"{name}{args}: {ms} ms, {len(r['evidence'])} queries")
        print("   ", json.dumps(r["result"], default=str)[:700])
    except Exception as e:  # noqa: BLE001
        print(f"{name}{args}: ERROR {e!r}")
Path("reports/brain_smoke.json").write_text(json.dumps(out, indent=1, default=str))
