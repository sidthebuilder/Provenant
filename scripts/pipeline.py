"""Run the whole Track 1 pipeline in order: ingest, detect, rank blocks, agent answers.

Usage: python scripts/pipeline.py [--skip-ingest]
Writes reports/pipeline.json with the headline numbers so every figure quoted
in the README comes from one run.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
ap = argparse.ArgumentParser()
ap.add_argument("--skip-ingest", action="store_true")
a = ap.parse_args()

steps = ([] if a.skip_ingest else [["scripts/ingest_moltbook.py"]]) + [
    ["scripts/detect_moltbook.py"],
    ["scripts/block_impact.py", "--hours", "6"],
    ["scripts/ask.py"],
]
timing = {}
for s in steps:
    t0 = time.perf_counter()
    print(">>", " ".join(s), flush=True)
    r = subprocess.run([PY, "-u", *s], cwd=ROOT)
    timing[s[0]] = round(time.perf_counter() - t0, 1)
    if r.returncode:
        sys.exit(f"step failed: {s}")

sys.path.insert(0, str(ROOT))
from provenant.db import Graph  # noqa: E402

g = Graph("moltbook_w1")
det = json.loads((ROOT / "reports/detect_moltbook_w1.json").read_text())
blocks = json.loads((ROOT / "reports/block_impact_moltbook_w1_6h.json").read_text())
answers = json.loads((ROOT / "reports/agent_answers.json").read_text())
head = {
    "graph": g.read_dicts("MATCH (m:Meta) RETURN m.posts AS posts, m.agents AS agents, m.owners AS owners, "
                          "m.keys AS keys, m.channels AS channels")[0],
    "swarms": det["swarms"], "steered_swarms": det["steered_swarms"], "operations": det["operations"],
    "detect_timing_s": det["timing_s"],
    "largest": g.read_dicts("MATCH (s:Swarm) RETURN s.id AS id, s.kind AS kind, s.n_agents AS agents, "
                            "s.n_posts AS posts, s.distinct_owners AS owners ORDER BY s.n_agents DESC LIMIT 1")[0],
    "blocks_6h": [{k: b[k] for k in ("channel", "blocked_posts", "blocked_swarm_posts", "collateral_posts",
                                     "swarm_posts_prevented", "swarm_agents_cut_off", "ms")} for b in blocks],
    "agent": [{"question": x["question"], "recommendation": x.get("recommendation"),
               "tool_calls": len(x["steps"]), "queries": sum(len(s["evidence"]) for s in x["steps"]),
               "ms": x.get("ms")} for x in answers],
    "step_timing_s": timing,
}
(ROOT / "reports/pipeline.json").write_text(json.dumps(head, indent=1, default=str))
print(json.dumps({k: head[k] for k in ("graph", "swarms", "steered_swarms", "largest")}, indent=1, default=str))
