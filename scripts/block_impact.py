"""Rank every steering channel by what blocking it would have cut off (replayed on forks)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from provenant.db import Graph, tracing  # noqa: E402
from provenant.moltbook.impact import rank_blocks  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--graph", default="moltbook_w1")
ap.add_argument("--hours", type=float, nargs="+", default=[1.0, 6.0])
a = ap.parse_args()
Path("reports").mkdir(exist_ok=True)
for h in a.hours:
    print(f"== block at first seen + {h} h", flush=True)
    with tracing() as tr:
        ranked = rank_blocks(Graph(a.graph), hours_after_first=h, log=lambda m: print(m, flush=True))
    summary = [{k: r[k] for k in ("channel", "hours_after_first_seen", "blocked_posts", "blocked_swarm_posts",
                                  "collateral_posts", "precision", "swarm_posts_prevented", "swarm_agents_cut_off", "ms")}
               for r in ranked]
    print(json.dumps(summary[:8], indent=1))
    print(f"queries run: {len(tr.queries)}, total query ms: {round(sum(q.ms for q in tr.queries), 1)}")
    Path(f"reports/block_impact_{a.graph}_{h:g}h.json").write_text(json.dumps(ranked, indent=1, default=str))
