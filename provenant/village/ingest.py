"""Load the AI Village export into one FalkorDB graph, ``village``.

Schema::

    (:Agent {id, name, model, slug})
    (:Room {id, name})
    (:Message {id, ts, text, day, decision, handoff, human})-[:IN]->(:Room)
    (:Agent)-[:SENT]->(:Message)
    (:Message)-[:MENTIONS]->(:Agent)                 agent names in the text
    (:Message)-[:LINKS]->(:Site {id, host, kind})    domains, repos, Pages sites
    (:Message)-[:DURING]->(:Goal {id, text, start_ts, end_ts})   village-wide goal
    (:Agent)-[:HAS_GOAL]->(:AgentGoal {name, short_name, description, start_ts})
    (:Agent)-[:SEARCHED]->(:Search {id, ts, query, lo_ts, hi_ts, answer_head})
    (:Search)-[:CITED]->(:Message)                   ground truth for Track 2

Only messages written by agents are loaded. Messages from human viewers are
skipped entirely: they are personal data, and the hackathon rules forbid
loading personal information into the graph.
Raw data is read from ``VILLAGE_DATA`` (default ``../data``) and never copied
into the repository.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import time
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..db import Graph, batched

DATA = Path(os.environ.get("VILLAGE_DATA", Path(__file__).resolve().parents[3] / "data"))
URL = re.compile(r"https?://([^\s)>\]\"'`]+)")
CITE = re.compile(r"\[Day (\d+),\s*(\d{1,2}:\d{2}:\d{2})\]\s*\n+\s*>\s*\*\*([^*:]+?):?\*\*")
DECISION = re.compile(r"\b(we (decided|agreed|will go with|should go with)|let's go with|consensus|final decision|"
                      r"decided to|agreed to|going with)\b", re.I)
HANDOFF = re.compile(r"\b(can you|could you|please (take|handle|review|check)|i'll take|i will take|taking over|"
                     r"over to you|handing (this )?off|assign(ed)? to)\b", re.I)
MAX_TEXT = 2000


def rows(name: str):
    with gzip.open(DATA / name, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def ts(s: str | None) -> float | None:
    if not s:
        return None
    return datetime.fromisoformat(s[:26]).replace(tzinfo=timezone.utc).timestamp()


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def site_id(rest: str) -> tuple[str, str, str] | None:
    """URL (without scheme) -> (site id, host, kind). Repos and Pages sites keep their project path."""
    rest = rest.rstrip(".,;:!?")
    host, _, path = rest.partition("/")
    host = host.lower().split(":")[0]
    if not host or "." not in host:
        return None
    parts = [p for p in path.split("/") if p and not p.startswith(("#", "?"))]
    if host in ("gitlab.com", "github.com") and len(parts) >= 2:
        return f"{host}/{parts[0]}/{parts[1]}".lower(), host, "repo"
    if host.endswith((".gitlab.io", ".github.io")):
        if host.startswith("ai-village-agents.") and parts:
            return f"{host}/{parts[0]}".lower(), host, "pages"
        return host, host, "pages"
    return host, host, "domain"


def build(log=print) -> dict:
    t0 = time.perf_counter()
    agents = list(rows("agents.jsonl.gz"))
    names = {a["id"]: a["name"] for a in agents}
    by_name = {a["name"]: a["id"] for a in agents}
    # Longest names first so "Claude Opus 4.5" wins over "Claude Opus 4".
    name_re = re.compile(r"(?<![\w.])@?(" + "|".join(re.escape(n) for n in sorted(by_name, key=len, reverse=True)) + r")(?![\w])")
    rooms = [{"id": r["id"], "name": r["name"]} for r in rows("chat_rooms.jsonl.gz")]
    goals = sorted(({"id": g["id"], "text": (g["goal"] or "")[:500], "start_ts": ts(g["start_time"]),
                     "end_ts": ts(g["end_time"])} for g in rows("village_goals.jsonl.gz")),
                   key=lambda g: g["start_ts"] or 0)
    goal_starts = [g["start_ts"] or 0 for g in goals]
    agent_goals = [{"agent": g["agent_id"], "name": g["name"], "short_name": g["short_name"],
                    "description": (g["description"] or "")[:500], "start_ts": ts(g["start_time"]),
                    "end_ts": ts(g["end_time"])} for g in rows("agent_goals.jsonl.gz")]

    msgs, sent, mentions, links, during = [], [], [], [], []
    sites: dict[str, dict] = {}
    by_key = defaultdict(list)  # (speaker name, HH:MM:SS) -> [(created_at, id)] for citation matching
    skipped_human = 0
    for m in rows("chat_messages.jsonl.gz"):
        # Only agent messages enter the graph. Messages from human viewers are
        # personal data, which the hackathon rules and the dataset terms keep out.
        if m["speaker_type"] != "agent" or m["agent_speaker_id"] not in names:
            skipped_human += 1
            continue
        text = (m["content"] or "")
        t = ts(m["created_at"])
        msgs.append({"id": m["id"], "ts": t, "text": text[:MAX_TEXT], "room": m["room_id"], "human": False,
                     "decision": bool(DECISION.search(text)), "handoff": bool(HANDOFF.search(text))})
        sent.append({"a": m["agent_speaker_id"], "m": m["id"]})
        by_key[(names[m["agent_speaker_id"]], m["created_at"][11:19])].append((m["created_at"], m["id"]))
        speaker = names.get(m["agent_speaker_id"])
        for n in set(name_re.findall(text)):
            if n != speaker:
                mentions.append({"m": m["id"], "a": by_name[n]})
        for u in set(URL.findall(text)):
            s = site_id(u)
            if s:
                sites.setdefault(s[0], {"id": s[0], "host": s[1], "kind": s[2]})
                links.append({"m": m["id"], "s": s[0]})
        i = bisect_right(goal_starts, t or 0) - 1
        if i >= 0 and (goals[i]["end_ts"] is None or t <= goals[i]["end_ts"]):
            during.append({"m": m["id"], "g": goals[i]["id"]})
    for v in by_key.values():
        v.sort()
    log(f"agent messages {len(msgs)} (skipped {skipped_human} non-agent), mentions {len(mentions)}, "
        f"links {len(links)}, sites {len(sites)} "
        f"in {time.perf_counter() - t0:.1f}s")

    # Searches and their cited messages. Day numbers are mapped to UTC dates
    # from the matched citations themselves (the most common date per day).
    searches, cited, day_dates = [], [], defaultdict(Counter)
    raw = []
    for e in rows("events.jsonl.gz"):
        d = e.get("data") or {}
        if d.get("actionType") != "SEARCH_HISTORY" or d.get("agentId") not in names:
            continue
        ans = d.get("answerToQuery") or ""
        hits = []
        for day, hms, sp in CITE.findall(ans):
            cand = by_key.get((sp.strip(), hms.zfill(8)), [])
            i = bisect_right(cand, (e["created_at"], "~"))
            if i:
                created, mid = cand[i - 1]
                hits.append(mid)
                day_dates[int(day)][created[:10]] += 1
        raw.append((e, d, list(dict.fromkeys(hits))))
    day_date = {d: c.most_common(1)[0][0] for d, c in day_dates.items()}

    def day_bounds(sd, ed, sdate, edate, at):
        lo = hi = None
        if sdate and edate:
            lo, hi = ts(sdate + " 00:00:00"), ts(edate + " 00:00:00") + 86400
        elif sd and ed:
            try:
                a, b = day_date.get(int(sd)), day_date.get(int(ed))
            except (TypeError, ValueError):
                a = b = None
            if a and b:
                lo, hi = ts(a + " 00:00:00") - 86400, ts(b + " 00:00:00") + 2 * 86400
        return lo or at - 14 * 86400, min(hi or at, at)

    for e, d, hits in raw:
        at = ts(e["created_at"])
        lo, hi = day_bounds(d.get("startDay"), d.get("endDay"), d.get("startDate"), d.get("endDate"), at)
        searches.append({"id": e["id"], "agent": d["agentId"], "ts": at, "query": (d.get("query") or "")[:1000],
                         "lo_ts": lo, "hi_ts": hi, "answer_head": (d.get("answerToQuery") or "")[:600],
                         "n_cited": len(hits)})
        cited += [{"s": e["id"], "m": mid} for mid in hits]
    log(f"searches {len(searches)}, with matched citations {sum(1 for s in searches if s['n_cited'])}, "
        f"cited edges {len(cited)}")

    return {
        "agents": [{"id": a["id"], "name": a["name"], "model": a["model_string"], "slug": slug(a["name"]),
                    "participating": bool(a["is_participating"])} for a in agents],
        "rooms": rooms, "goals": goals, "agent_goals": agent_goals, "messages": msgs, "sent": sent,
        "mentions": mentions, "sites": list(sites.values()), "links": links, "during": during,
        "searches": searches, "cited": cited,
    }


def load(graph: Graph, r: dict, log=print) -> dict:
    graph.delete()
    for label, prop in [("Agent", "id"), ("Agent", "name"), ("Room", "id"), ("Message", "id"), ("Message", "ts"),
                        ("Site", "id"), ("Goal", "id"), ("Search", "id")]:
        graph.write(f"CREATE INDEX FOR (n:{label}) ON (n.{prop})")
    graph.write("CREATE (:Meta {dataset: 'ai-village', ready: false})")
    steps = [
        ("agents", "UNWIND $rows AS r CREATE (:Agent {id: r.id, name: r.name, model: r.model, slug: r.slug, "
                   "participating: r.participating})", r["agents"]),
        ("rooms", "UNWIND $rows AS r CREATE (:Room {id: r.id, name: r.name})", r["rooms"]),
        ("goals", "UNWIND $rows AS r CREATE (:Goal {id: r.id, text: r.text, start_ts: r.start_ts, end_ts: r.end_ts})",
         r["goals"]),
        ("agent_goals", "UNWIND $rows AS r MATCH (a:Agent {id: r.agent}) CREATE (a)-[:HAS_GOAL]->(:AgentGoal "
                        "{name: r.name, short_name: r.short_name, description: r.description, start_ts: r.start_ts, "
                        "end_ts: r.end_ts})", r["agent_goals"]),
        ("messages", "UNWIND $rows AS r MATCH (room:Room {id: r.room}) CREATE (:Message {id: r.id, ts: toFloat(r.ts), "
                     "text: r.text, human: r.human, decision: r.decision, handoff: r.handoff})-[:IN]->(room)",
         r["messages"]),
        ("sent", "UNWIND $rows AS r MATCH (a:Agent {id: r.a}), (m:Message {id: r.m}) CREATE (a)-[:SENT]->(m)", r["sent"]),
        ("mentions", "UNWIND $rows AS r MATCH (m:Message {id: r.m}), (a:Agent {id: r.a}) CREATE (m)-[:MENTIONS]->(a)",
         r["mentions"]),
        ("sites", "UNWIND $rows AS r CREATE (:Site {id: r.id, host: r.host, kind: r.kind})", r["sites"]),
        ("links", "UNWIND $rows AS r MATCH (m:Message {id: r.m}), (s:Site {id: r.s}) CREATE (m)-[:LINKS]->(s)",
         r["links"]),
        ("during", "UNWIND $rows AS r MATCH (m:Message {id: r.m}), (g:Goal {id: r.g}) CREATE (m)-[:DURING]->(g)",
         r["during"]),
        ("searches", "UNWIND $rows AS r MATCH (a:Agent {id: r.agent}) CREATE (a)-[:SEARCHED]->(:Search {id: r.id, "
                     "ts: toFloat(r.ts), query: r.query, lo_ts: toFloat(r.lo_ts), hi_ts: toFloat(r.hi_ts), "
                     "answer_head: r.answer_head, n_cited: r.n_cited})", r["searches"]),
        ("cited", "UNWIND $rows AS r MATCH (s:Search {id: r.s}), (m:Message {id: r.m}) CREATE (s)-[:CITED]->(m)",
         r["cited"]),
    ]
    timing = {}
    for name, cypher, data in steps:
        t0 = time.perf_counter()
        for chunk in batched(data, 2000):
            graph.write(cypher, {"rows": chunk})
        timing[name] = round(time.perf_counter() - t0, 2)
        log(f"  {name}: {len(data)} in {timing[name]}s")
    t0 = time.perf_counter()
    graph.write("CALL db.idx.fulltext.createNodeIndex('Message', 'text')")
    graph.write("CALL db.idx.fulltext.createNodeIndex('Search', 'query')")
    graph.wait_indexes(600)
    timing["fulltext"] = round(time.perf_counter() - t0, 2)
    counts = {k: len(r[k]) for k in ("agents", "rooms", "goals", "messages", "mentions", "sites", "links",
                                     "searches", "cited")}
    graph.write("MATCH (m:Meta) SET m += $c, m.ready = true", {"c": counts})
    return {"counts": counts, "timing_s": timing}
