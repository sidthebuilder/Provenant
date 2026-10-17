"""Run in-graph swarm detection on a loaded Moltbook graph and save a summary."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from provenant.db import Graph  # noqa: E402
from provenant.moltbook.detect import detect  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--graph", default="moltbook_w1")
a = ap.parse_args()
g = Graph(a.graph)
out = detect(g, log=lambda m: print(m, flush=True))
out["top"] = g.read_dicts(
    "MATCH (s:Swarm) OPTIONAL MATCH (s)-[r:STEERED_BY]->(c:Channel) "
    "WITH s, collect({channel: c.id, share: r.share, owners: r.owners})[0..3] AS ch "
    "RETURN s.id AS id, s.kind AS kind, s.n_agents AS agents, s.n_posts AS posts, "
    "s.distinct_owners AS owners, s.top_owner_share AS top_owner_share, ch AS channels "
    "ORDER BY s.n_agents DESC LIMIT 15")
print(json.dumps(out, indent=1, default=str))
Path("reports").mkdir(exist_ok=True)
Path(f"reports/detect_{a.graph}.json").write_text(json.dumps(out, indent=1, default=str))
