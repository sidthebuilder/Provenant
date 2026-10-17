"""Track 2: shared, time-aware agent memory over the village graph.

``recall`` answers "what do I need to know about X?" for one agent at one
moment, using only what existed before that moment (no peeking at the
future). Three retrievers, compared in the benchmark:

- ``recency``  the last k messages before t: what a context window holds.
- ``fulltext`` FalkorDB full-text search over messages in the time window.
- ``graph``    full-text seeds plus graph context: messages sent by or
               mentioning the agents and sites named in the question,
               messages addressed to the asking agent, re-ranked together.

The benchmark ground truth is real: every SEARCH_HISTORY event is a moment an
agent needed something it did not have in memory, and the village's own
search tool answered it citing specific messages, which ingest matched back
to chat messages by speaker and timestamp.
"""

from __future__ import annotations

import re

import numpy as np

from ..db import Graph

STOP = set("""a an the and or but if then else of to in on at by for with from as is are was were be been being
this that these those it its i me my we our you your he she they them their what which who whom when where why how
any all some no not only own same so than too very can will just do does did done have has had having about above
after again against before below between both during each few more most other such into through under until up
down out off over here there once also since still yet get got via per day days today tomorrow yesterday please
message messages mention mentions mentioned events event recent latest new update updates check status since
directed anything something including include""".split())
TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{2,}")


def keywords(q: str, n: int = 12) -> list[str]:
    seen, out = set(), []
    for w in TOKEN.findall(q):
        lw = w.lower().strip("-_")
        if lw in STOP or lw in seen or len(lw) < 3:
            continue
        seen.add(lw)
        out.append(lw)
    return out[:n]


def _ft_query(words: list[str]) -> str:
    # RediSearch syntax: OR of terms; strip characters that have meaning there.
    return "|".join(re.sub(r"[^a-z0-9]", "", w) for w in words if re.sub(r"[^a-z0-9]", "", w))


def named_agents(g: Graph, q: str) -> list[str]:
    names = [r[0] for r in g.read("MATCH (a:Agent) RETURN a.name")]
    hits = [n for n in sorted(names, key=len, reverse=True) if re.search(r"(?<![\w.])" + re.escape(n) + r"(?![\w])", q)]
    # drop names contained in a longer hit ("Claude Opus 4" inside "Claude Opus 4.5")
    return [n for n in hits if not any(n != m and n in m for m in hits)]


def recall(g: Graph, query: str, at_ts: float, lo_ts: float | None = None, agent_id: str | None = None,
           k: int = 20, mode: str = "graph") -> list[dict]:
    lo = lo_ts if lo_ts is not None else at_ts - 14 * 86400
    p = {"lo": float(lo), "t": float(at_ts), "k": int(k)}
    if mode == "recency":
        return g.read_dicts(
            "MATCH (m:Message) WHERE m.ts >= $lo AND m.ts < $t "
            "RETURN m.id AS id, m.ts AS ts, 0.0 AS score ORDER BY m.ts DESC LIMIT $k", p)

    words = keywords(query)
    fq = _ft_query(words)
    ft = []
    if fq:
        ft = g.read_dicts(
            "CALL db.idx.fulltext.queryNodes('Message', $q) YIELD node, score "
            "WHERE node.ts >= $lo AND node.ts < $t "
            "RETURN node.id AS id, node.ts AS ts, score ORDER BY score DESC LIMIT 300", {**p, "q": fq})
    if mode == "fulltext":
        return ft[:k]

    # Graph context: people named in the question, and messages to the asker.
    names = named_agents(g, query)
    ctx = g.read_dicts(
        "MATCH (a:Agent) WHERE a.name IN $names "
        "MATCH (a)-[:SENT]->(m:Message) WHERE m.ts >= $lo AND m.ts < $t "
        "RETURN m.id AS id, m.ts AS ts, 'sent_by_named' AS why "
        "UNION "
        "MATCH (a:Agent) WHERE a.name IN $names "
        "MATCH (m:Message)-[:MENTIONS]->(a) WHERE m.ts >= $lo AND m.ts < $t "
        "RETURN m.id AS id, m.ts AS ts, 'mentions_named' AS why "
        "UNION "
        "MATCH (me:Agent {id: $me})<-[:MENTIONS]-(m:Message) WHERE m.ts >= $lo AND m.ts < $t "
        "RETURN m.id AS id, m.ts AS ts, 'to_asker' AS why",
        {**p, "names": names, "me": agent_id or ""}) if (names or agent_id) else []

    top = max((r["score"] for r in ft), default=1.0) or 1.0
    span = max(at_ts - lo, 1.0)
    score: dict[str, float] = {}
    ts_of: dict[str, float] = {}
    why: dict[str, list] = {}
    for r in ft:
        score[r["id"]] = r["score"] / top
        ts_of[r["id"]] = r["ts"]
        why.setdefault(r["id"], []).append("text")
    for r in ctx:
        bonus = {"sent_by_named": 0.6, "mentions_named": 0.5, "to_asker": 0.4}[r["why"]]
        recency = 0.3 * (r["ts"] - lo) / span
        score[r["id"]] = score.get(r["id"], 0.0) + bonus + recency
        ts_of[r["id"]] = r["ts"]
        why.setdefault(r["id"], []).append(r["why"])
    ranked = sorted(score, key=lambda i: (score[i], ts_of[i]), reverse=True)[:k]
    return [{"id": i, "ts": ts_of[i], "score": round(score[i], 4), "why": why[i]} for i in ranked]


DEFAULT_W = {"tau_h": 6.0, "text": 0.5, "named": 0.5, "to_asker": 0.5, "room": 0.3, "pool": 400}


def hybrid(g: Graph, query: str, at_ts: float, lo_ts: float | None = None, agent_id: str | None = None,
           k: int = 20, w: dict | None = None, _cache: dict | None = None) -> list[dict]:
    """Recency as the base signal, plus graph evidence.

    Candidates: the last ``pool`` messages in the window, full-text hits, and
    graph context (messages by or mentioning agents named in the question,
    messages mentioning the asker). Score = exp(-age/tau) + text match +
    graph bonuses + same-room bonus (the room the asker last spoke in).
    """
    w = {**DEFAULT_W, **(w or {})}
    lo = lo_ts if lo_ts is not None else at_ts - 14 * 86400
    p = {"lo": float(lo), "t": float(at_ts), "me": agent_id or ""}
    c = _cache if _cache is not None else {}
    if "pool" not in c:
        c["pool"] = g.read_dicts(
            "MATCH (m:Message)-[:IN]->(r:Room) WHERE m.ts >= $lo AND m.ts < $t "
            "RETURN m.id AS id, m.ts AS ts, r.id AS room ORDER BY m.ts DESC LIMIT $n", {**p, "n": int(w["pool"])})
        fq = _ft_query(keywords(query))
        c["ft"] = g.read_dicts(
            "CALL db.idx.fulltext.queryNodes('Message', $q) YIELD node, score "
            "WHERE node.ts >= $lo AND node.ts < $t MATCH (node)-[:IN]->(r:Room) "
            "RETURN node.id AS id, node.ts AS ts, r.id AS room, score ORDER BY score DESC LIMIT 300",
            {**p, "q": fq}) if fq else []
        names = named_agents(g, query)
        c["ctx"] = g.read_dicts(
            "MATCH (a:Agent) WHERE a.name IN $names MATCH (a)-[:SENT]->(m:Message)-[:IN]->(r:Room) "
            "WHERE m.ts >= $lo AND m.ts < $t RETURN m.id AS id, m.ts AS ts, r.id AS room, 'named' AS why "
            "UNION MATCH (a:Agent) WHERE a.name IN $names MATCH (r:Room)<-[:IN]-(m:Message)-[:MENTIONS]->(a) "
            "WHERE m.ts >= $lo AND m.ts < $t RETURN m.id AS id, m.ts AS ts, r.id AS room, 'named' AS why "
            "UNION MATCH (me:Agent {id: $me})<-[:MENTIONS]-(m:Message)-[:IN]->(r:Room) "
            "WHERE m.ts >= $lo AND m.ts < $t RETURN m.id AS id, m.ts AS ts, r.id AS room, 'to_asker' AS why",
            {**p, "names": names}) if (names or agent_id) else []
        room = g.read("MATCH (me:Agent {id: $me})-[:SENT]->(m:Message)-[:IN]->(r:Room) WHERE m.ts < $t "
                      "RETURN r.id ORDER BY m.ts DESC LIMIT 1", p) if agent_id else []
        c["room"] = room[0][0] if room else None
    if "X" not in c:
        c["X"] = features(c, at_ts)
    ids, X = c["X"]
    if not ids:
        return []
    s = score_matrix(X, w)
    order = np.lexsort((X[:, 0], -s))[:k]  # score desc, then smallest age (newest) first
    why_cols = ["recent", "text", "named", "to_asker", "same_room"]
    return [{"id": ids[i], "ts": float(at_ts - X[i, 0] * 3600), "score": round(float(s[i]), 4),
             "why": [why_cols[j] for j in range(1, 5) if X[i, j] > 0] or ["recent"]} for i in order]


def features(c: dict, at_ts: float):
    """Per-candidate feature matrix: age_h, text score (0..1), named, to_asker, same room."""
    top = max((r["score"] for r in c["ft"]), default=1.0) or 1.0
    idx: dict[str, int] = {}
    rows: list[list[float]] = []
    for r in c["pool"] + c["ft"] + c["ctx"]:
        if r["id"] not in idx:
            idx[r["id"]] = len(rows)
            rows.append([(at_ts - r["ts"]) / 3600.0, 0.0, 0.0, 0.0,
                         1.0 if c["room"] and r["room"] == c["room"] else 0.0])
    for r in c["ft"]:
        rows[idx[r["id"]]][1] = max(rows[idx[r["id"]]][1], r["score"] / top)
    for r in c["ctx"]:
        rows[idx[r["id"]]][2 if r["why"] == "named" else 3] = 1.0
    return list(idx), (np.array(rows, dtype=np.float64) if rows else np.zeros((0, 5)))


def score_matrix(X, w: dict):
    return (np.exp(-X[:, 0] / w["tau_h"]) + w["text"] * X[:, 1] + w["named"] * X[:, 2]
            + w["to_asker"] * X[:, 3] + w["room"] * X[:, 4])
