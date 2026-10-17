"""Call every Track 1 tool once against the real graph and save outputs."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from provenant.moltbook import tools as t  # noqa: E402

out = {}
sw = t.list_swarms(limit=5)
out["list_swarms"] = sw
top = sw["result"][0]["swarm"]
steered = next((r for r in t.list_swarms(limit=200)["result"] if r["top_channel"]), None)
for name, fn in [("swarm_profile", lambda: t.swarm_profile(top)),
                 ("owner_spread", lambda: t.owner_spread(top)),
                 ("steering_channels", lambda: t.steering_channels(steered["swarm"])),
                 ("channel_reach", lambda: t.channel_reach(steered["top_channel"])),
                 ("operation_of", lambda: t.operation_of(steered["swarm"])),
                 ("channel_pagerank", lambda: t.channel_pagerank()),
                 ("block_impact", lambda: t.block_impact("path:clawtasks.com/skill.md"))]:
    t0 = time.perf_counter()
    try:
        out[name] = fn()
    except Exception as e:  # noqa: BLE001
        out[name] = {"error": repr(e)}
    print(f"{name}: {round((time.perf_counter() - t0) * 1000)} ms", flush=True)
Path("reports/tools_smoke.json").write_text(json.dumps(out, indent=1, default=str))
for k, v in out.items():
    r = v.get("result", v)
    print(k, "->", json.dumps(r, default=str)[:400], flush=True)
