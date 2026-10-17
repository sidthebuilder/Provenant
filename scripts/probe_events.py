"""Profile AI Village events and memories for Tracks 2 and 3 (counts and field shapes only)."""
import gzip
import json
from collections import Counter
from pathlib import Path

D = Path(__file__).resolve().parents[2] / "data"
out = {}
types, search, consol = Counter(), [], 0
keys_by_type = {}
with gzip.open(D / "events.jsonl.gz", "rt", encoding="utf-8") as f:
    for line in f:
        e = json.loads(line)
        d = e.get("data") or {}
        t = d.get("actionType")
        types[t] += 1
        keys_by_type.setdefault(t, sorted(d.keys())[:20])
        if t == "SEARCH_HISTORY" and len(search) < 3:
            search.append({k: (str(v)[:300] if v is not None else None) for k, v in d.items() if k != "output"})
        if t == "CONSOLIDATE":
            consol += 1
out["event_types"] = types.most_common(25)
out["fields"] = {k: v for k, v in keys_by_type.items() if k in ("SEARCH_HISTORY", "CONSOLIDATE", "AGENT_TALK")}
out["search_samples"] = search

n, lens, per_agent, sample = 0, [], Counter(), None
with gzip.open(D / "agent_memories.jsonl.gz", "rt", encoding="utf-8") as f:
    for line in f:
        m = json.loads(line)
        n += 1
        lens.append(len(m.get("content") or ""))
        per_agent[m["agent_id"]] += 1
        if sample is None and 500 < len(m.get("content") or "") < 3000:
            sample = {"agent_id": m["agent_id"], "created_at": m["created_at"], "content": m["content"][:1500]}
lens.sort()
out["memories"] = {"rows": n, "agents": len(per_agent), "median_chars": lens[len(lens) // 2],
                   "p90_chars": lens[int(len(lens) * 0.9)], "sample": sample}
Path("reports").mkdir(exist_ok=True)
Path("reports/probe_events.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if k != "memories"}, indent=1, default=str)[:6000])
print(json.dumps({k: v for k, v in out["memories"].items() if k != "sample"}))
