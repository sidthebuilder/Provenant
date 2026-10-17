"""Track 2 benchmark: can graph memory answer the history searches agents really made?

Ground truth: every SEARCH_HISTORY event whose answer cites messages that
ingest matched back to the chat log (speaker + timestamp). Each retriever runs
as of the search time, with only past data, and is scored by recall@k of the
cited messages.

Honest protocol: the hybrid retriever's weights are chosen on the earlier half
of searches (dev, by time) from a small fixed grid, then every method is
reported on the later half (test), which the grid never saw.
"""
import itertools
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from provenant.db import Graph  # noqa: E402
from provenant.village.memory import hybrid, recall  # noqa: E402

g = Graph("village")
K = [10, 20, 50]
cases = g.read_dicts(
    "MATCH (a:Agent)-[:SEARCHED]->(s:Search)-[:CITED]->(m:Message) "
    "OPTIONAL MATCH (author:Agent)-[:SENT]->(m) "
    "RETURN s.id AS id, a.id AS agent, s.ts AS ts, s.lo_ts AS lo, s.query AS query, "
    "collect(DISTINCT m.id) AS gold, collect(DISTINCT CASE WHEN author.id = a.id THEN m.id END) AS own "
    "ORDER BY s.ts", timeout_ms=600000)
half = len(cases) // 2
dev, test = cases[:half], cases[half:]
print(f"cases: {len(cases)} (dev {len(dev)}, test {len(test)})", flush=True)


def rec(gold, ids, k):
    return len(set(gold) & set(ids[:k])) / len(gold)


# Cache candidate pools per case so the grid search is cheap.
caches = {}
t0 = time.perf_counter()
for c in dev:
    caches[c["id"]] = {}
    hybrid(g, c["query"], c["ts"], c["lo"], c["agent"], k=50, _cache=caches[c["id"]])
print(f"dev pools built in {time.perf_counter() - t0:.0f}s", flush=True)

grid = {"tau_h": [1, 3, 6, 24], "text": [0.0, 0.25, 0.5, 1.0], "named": [0.0, 0.25, 0.5],
        "to_asker": [0.0, 0.25, 0.5], "room": [0.0, 0.3, 0.6]}
# Vectorised grid search: score every candidate with numpy, take top 20.
import numpy as np  # noqa: E402

from provenant.village.memory import score_matrix  # noqa: E402

prepared = []
for c in dev:
    ids, X = caches[c["id"]]["X"]
    gold = set(c["gold"])
    prepared.append((X, np.array([i in gold for i in ids], dtype=bool), len(gold)))

best, best_w = -1, None
for vals in itertools.product(*grid.values()):
    w = dict(zip(grid, vals))
    tot = 0.0
    for X, is_gold, n_gold in prepared:
        if len(X) == 0:
            continue
        s = score_matrix(X, w)
        top = np.argpartition(-s, min(20, len(s)) - 1)[:20] if len(s) > 20 else np.arange(len(s))
        tot += is_gold[top].sum() / n_gold
    s = tot / len(prepared)
    if s > best:
        best, best_w = s, w
print(f"best dev recall@20 {best:.4f} with {best_w}", flush=True)

modes = {
    "recency": lambda c: recall(g, c["query"], c["ts"], c["lo"], c["agent"], k=50, mode="recency"),
    "fulltext": lambda c: recall(g, c["query"], c["ts"], c["lo"], c["agent"], k=50, mode="fulltext"),
    "graph_only": lambda c: recall(g, c["query"], c["ts"], c["lo"], c["agent"], k=50, mode="graph"),
    "hybrid": lambda c: hybrid(g, c["query"], c["ts"], c["lo"], c["agent"], k=50, w=best_w),
}
res = {m: {k: [] for k in K} for m in modes}
lat = {m: [] for m in modes}
for c in test:
    for m, fn in modes.items():
        t1 = time.perf_counter()
        ids = [r["id"] for r in fn(c)]
        lat[m].append((time.perf_counter() - t1) * 1000)
        for k in K:
            res[m][k].append(rec(c["gold"], ids, k))

other = [len(set(c["gold"]) - {x for x in c["own"] if x}) / len(c["gold"]) for c in cases]
summary = {
    "cases": len(cases), "dev": len(dev), "test": len(test),
    "gold_messages": sum(len(c["gold"]) for c in cases),
    "share_of_needed_messages_written_by_other_agents": round(statistics.mean(other), 4),
    "hybrid_weights_chosen_on_dev": best_w, "dev_recall_at_20": round(best, 4),
    "test_recall": {m: {f"@{k}": round(statistics.mean(v[k]), 4) for k in K} for m, v in res.items()},
    "test_hit_rate_at_20": {m: round(sum(1 for x in v[20] if x > 0) / len(v[20]), 4) for m, v in res.items()},
    "latency_ms_median": {m: round(statistics.median(v), 1) for m, v in lat.items()},
    "latency_ms_p95": {m: round(sorted(v)[int(0.95 * len(v))], 1) for m, v in lat.items()},
}
print(json.dumps(summary, indent=1))
Path("reports/bench_memory.json").write_text(json.dumps(summary, indent=1))
