"""How many SEARCH_HISTORY answers cite messages we can match back to chat_messages?

Citation format in answers: "[Day 478, 23:50:05]" then "> **Speaker:** text".
Match rule: same speaker name, same HH:MM:SS (UTC), created before the search,
latest such message. Reports parse and match rates on real data.
"""
import gzip
import json
import re
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path

D = Path(__file__).resolve().parents[2] / "data"
CITE = re.compile(r"\[Day (\d+),\s*(\d{1,2}:\d{2}:\d{2})\]\s*\n+\s*>\s*\*\*([^*:]+?):?\*\*:?\s*(.{0,80})")


def rows(name):
    with gzip.open(D / name, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


names = {a["id"]: a["name"] for a in rows("agents.jsonl.gz")}
idx = defaultdict(list)  # (speaker, hh:mm:ss) -> sorted [(created_at, id)]
for m in rows("chat_messages.jsonl.gz"):
    if m["speaker_type"] != "agent":
        continue
    sp = names.get(m["agent_speaker_id"])
    ts = m["created_at"]
    idx[(sp, ts[11:19])].append((ts, m["id"], (m["content"] or "")[:60]))
for v in idx.values():
    v.sort()

n_search = n_with = n_cites = n_match = 0
prefix_ok = 0
per_search = []
samples = []
for e in rows("events.jsonl.gz"):
    d = e.get("data") or {}
    if d.get("actionType") != "SEARCH_HISTORY":
        continue
    n_search += 1
    ans = d.get("answerToQuery") or ""
    cites = CITE.findall(ans)
    if cites:
        n_with += 1
    hits = 0
    for day, hms, sp, text in cites:
        n_cites += 1
        hms = hms.zfill(8)
        cand = idx.get((sp.strip(), hms), [])
        i = bisect_right(cand, (e["created_at"], "~"))
        if i:
            n_match += 1
            hits += 1
            ts, mid, c = cand[i - 1]
            if text.strip()[:15].lower() and text.strip()[:15].lower().replace("*", "") in c.lower().replace("*", ""):
                prefix_ok += 1
    per_search.append((len(cites), hits))
    if cites and len(samples) < 2:
        samples.append({"query": d.get("query"), "cites": cites[:3]})

out = {"searches": n_search, "with_citations": n_with, "citations": n_cites, "matched": n_match,
       "match_rate": round(n_match / max(n_cites, 1), 4), "matched_with_text_prefix": prefix_ok,
       "searches_with_ge1_match": sum(1 for c, h in per_search if h),
       "cites_per_search": Counter(min(c, 10) for c, _ in per_search).most_common(),
       "samples": samples}
print(json.dumps(out, indent=1, default=str))
Path("reports/probe_citations.json").write_text(json.dumps(out, indent=1, default=str))
