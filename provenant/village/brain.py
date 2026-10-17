"""Track 3: the AI Village as a company, and a brain that answers with sources.

The village is a real organisation: about 50 frontier-model agents with
roles (individual goals), shared goals that change over time, rooms, and the
sites and repos they built. Each tool answers one company question from the
graph and returns the message ids that support it, so every answer can be
opened at its source.

- ``who_owns(site)``      first agent to link it, and who links it most
- ``sites_of(agent)``     what an agent built or maintains
- ``who_knows(topic)``    agents who wrote most about a topic, with examples
- ``decisions(topic)``    decision messages about a topic, with goal context
- ``goal_team(goal)``     who was active under a village goal, and how they talked
- ``handoffs(agent)``     requests to and from an agent (who works with whom)
- ``as_of(ts)``           every tool takes ``until_ts`` so answers reflect what was known then
"""

from __future__ import annotations

import re

from ..db import Graph, tracing
from .memory import _ft_query, keywords

GRAPH = "village"
FAR = 1e12


def _run(fn) -> dict:
    with tracing() as tr:
        result = fn()
    return {"result": result, "evidence": tr.to_list()}


def _g() -> Graph:
    return Graph(GRAPH)


def _site_key(s: str) -> str:
    s = re.sub(r"^https?://", "", s.strip().lower()).rstrip("/")
    return s


def who_owns(site: str, until_ts: float | None = None) -> dict:
    """Who introduced a site or repo, and who links it most. Matches by prefix (e.g. 'gitlab.com/ai-village-agents')."""
    key = _site_key(site)

    def f():
        g = _g()
        sites = g.read_dicts("MATCH (s:Site) WHERE s.id STARTS WITH $k OR s.host = $k "
                             "RETURN s.id AS site, s.kind AS kind LIMIT 25", {"k": key})
        if not sites:
            return {"site": key, "error": "no such site in the chat record"}
        ids = [s["site"] for s in sites]
        t = until_ts or FAR
        first = g.read_dicts(
            "MATCH (s:Site)<-[:LINKS]-(m:Message)<-[:SENT]-(a:Agent) WHERE s.id IN $ids AND m.ts <= $t "
            "RETURN s.id AS site, a.name AS agent, m.id AS message, m.ts AS ts, left(m.text, 200) AS text "
            "ORDER BY m.ts LIMIT 1", {"ids": ids, "t": t})
        top = g.read_dicts(
            "MATCH (s:Site)<-[:LINKS]-(m:Message)<-[:SENT]-(a:Agent) WHERE s.id IN $ids AND m.ts <= $t "
            "WITH a, count(m) AS n, collect(m.id)[0..3] AS examples, max(m.ts) AS last "
            "RETURN a.name AS agent, n AS messages, last AS last_ts, examples ORDER BY n DESC LIMIT 5",
            {"ids": ids, "t": t})
        return {"site": key, "matched_sites": ids[:10], "introduced_by": first[0] if first else None,
                "most_active": top,
                "note": "Ownership is inferred from who linked the site in chat; repo permissions are not in the data."}
    return _run(f)


def sites_of(agent: str, until_ts: float | None = None, limit: int = 10) -> dict:
    """Sites and repos an agent introduced first (likely built or owns), with the introducing message."""
    def f():
        # Sites this agent linked, kept only where its first link is the
        # earliest link by anyone (it introduced the site).
        return _g().read_dicts(
            "MATCH (a:Agent {name: $a})-[:SENT]->(m:Message)-[:LINKS]->(s:Site) "
            "WHERE m.ts <= $t AND s.kind IN ['repo', 'pages'] "
            "WITH s, min(m.ts) AS mine "
            "MATCH (s)<-[:LINKS]-(o:Message) WHERE o.ts <= $t "
            "WITH s, mine, min(o.ts) AS earliest, count(o) AS links WHERE mine = earliest "
            "MATCH (s)<-[:LINKS]-(intro:Message {ts: mine}) "
            "RETURN s.id AS site, s.kind AS kind, intro.id AS message, mine AS ts, links "
            "ORDER BY links DESC LIMIT $n", {"a": agent, "t": until_ts or FAR, "n": int(limit)},
            timeout_ms=60000)
    return _run(f)


def who_knows(topic: str, until_ts: float | None = None, limit: int = 5) -> dict:
    """Agents who wrote most about a topic (full-text over their messages), with example sources."""
    def f():
        q = _ft_query(keywords(topic))
        if not q:
            return []
        return _g().read_dicts(
            "CALL db.idx.fulltext.queryNodes('Message', $q) YIELD node, score "
            "WHERE node.ts <= $t "
            "MATCH (a:Agent)-[:SENT]->(node) "
            "WITH a, count(node) AS n, sum(score) AS weight, collect(node.id)[0..3] AS examples, "
            "min(node.ts) AS first_ts, max(node.ts) AS last_ts "
            "RETURN a.name AS agent, n AS messages, round(weight * 100) / 100 AS weight, first_ts, last_ts, examples "
            "ORDER BY weight DESC LIMIT $n", {"q": q, "t": until_ts or FAR, "n": int(limit)}, timeout_ms=60000)
    return _run(f)


def decisions(topic: str, until_ts: float | None = None, limit: int = 8) -> dict:
    """Messages that record a decision about a topic, with who made it, when, and the village goal at the time."""
    def f():
        q = _ft_query(keywords(topic))
        if not q:
            return []
        return _g().read_dicts(
            "CALL db.idx.fulltext.queryNodes('Message', $q) YIELD node, score "
            "WHERE node.decision AND node.ts <= $t "
            "OPTIONAL MATCH (a:Agent)-[:SENT]->(node) "
            "OPTIONAL MATCH (node)-[:DURING]->(goal:Goal) "
            "OPTIONAL MATCH (node)-[:IN]->(r:Room) "
            "RETURN node.id AS message, node.ts AS ts, coalesce(a.name, '(human)') AS by, r.name AS room, "
            "left(goal.text, 120) AS goal, left(node.text, 300) AS text, score "
            "ORDER BY score DESC, ts DESC LIMIT $n", {"q": q, "t": until_ts or FAR, "n": int(limit)},
            timeout_ms=60000)
    return _run(f)


def goal_team(goal_text: str, limit: int = 10) -> dict:
    """For the village goal matching the text: who talked most, who addressed whom, and decisions made."""
    def f():
        g = _g()
        goals = g.read_dicts("MATCH (x:Goal) WHERE toLower(x.text) CONTAINS toLower($s) "
                             "RETURN x.id AS id, x.text AS text, x.start_ts AS start_ts, x.end_ts AS end_ts "
                             "ORDER BY x.start_ts LIMIT 1", {"s": goal_text})
        if not goals:
            return {"error": f"no village goal matching '{goal_text}'"}
        gid = goals[0]["id"]
        members = g.read_dicts(
            "MATCH (x:Goal {id: $g})<-[:DURING]-(m:Message)<-[:SENT]-(a:Agent) "
            "RETURN a.name AS agent, count(m) AS messages ORDER BY messages DESC LIMIT $n", {"g": gid, "n": int(limit)})
        pairs = g.read_dicts(
            "MATCH (x:Goal {id: $g})<-[:DURING]-(m:Message)<-[:SENT]-(a:Agent), (m)-[:MENTIONS]->(b:Agent) "
            "RETURN a.name AS from, b.name AS to, count(m) AS messages ORDER BY messages DESC LIMIT $n",
            {"g": gid, "n": int(limit)})
        dec = g.read_dicts(
            "MATCH (x:Goal {id: $g})<-[:DURING]-(m:Message) WHERE m.decision "
            "OPTIONAL MATCH (a:Agent)-[:SENT]->(m) "
            "RETURN m.id AS message, m.ts AS ts, coalesce(a.name, '(human)') AS by, left(m.text, 200) AS text "
            "ORDER BY m.ts LIMIT 5", {"g": gid})
        return {"goal": goals[0], "most_active": members, "who_addressed_whom": pairs, "early_decisions": dec}
    return _run(f)


def handoffs(agent: str, until_ts: float | None = None, limit: int = 10) -> dict:
    """Requests for help to and from an agent: the working relationships, with sources."""
    def f():
        g = _g()
        p = {"a": agent, "t": until_ts or FAR, "n": int(limit)}
        to_me = g.read_dicts(
            "MATCH (b:Agent)-[:SENT]->(m:Message)-[:MENTIONS]->(a:Agent {name: $a}) WHERE m.handoff AND m.ts <= $t "
            "RETURN b.name AS from, count(m) AS requests, collect(m.id)[0..2] AS examples "
            "ORDER BY requests DESC LIMIT $n", p)
        from_me = g.read_dicts(
            "MATCH (a:Agent {name: $a})-[:SENT]->(m:Message)-[:MENTIONS]->(b:Agent) WHERE m.handoff AND m.ts <= $t "
            "RETURN b.name AS to, count(m) AS requests, collect(m.id)[0..2] AS examples "
            "ORDER BY requests DESC LIMIT $n", p)
        return {"agent": agent, "asked_by": to_me, "asks": from_me}
    return _run(f)


def message(mid: str) -> dict:
    """Open a source message: text, sender, room, time and the village goal then."""
    def f():
        r = _g().read_dicts(
            "MATCH (m:Message {id: $id}) OPTIONAL MATCH (a:Agent)-[:SENT]->(m) "
            "OPTIONAL MATCH (m)-[:IN]->(r:Room) OPTIONAL MATCH (m)-[:DURING]->(goal:Goal) "
            "RETURN m.id AS id, m.ts AS ts, a.name AS by, "
            "r.name AS room, left(goal.text, 160) AS goal, m.text AS text", {"id": mid})
        return r[0] if r else {"error": "not found"}
    return _run(f)


TOOLS = {f.__name__: f for f in (who_owns, sites_of, who_knows, decisions, goal_team, handoffs, message)}
