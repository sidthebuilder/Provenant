"""HTTP API and static UI.

Run:  uvicorn provenant.api:app --port 8000
Endpoints are read-only against the main graphs; block_impact writes only to
a throwaway GRAPH.COPY fork. There is no authentication: for a public demo,
put this behind a reverse proxy with rate limiting.
"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .db import Graph
from .moltbook import tools as t1
from .moltbook.agent import investigate

UI = Path(__file__).resolve().parents[1] / "ui"
app = FastAPI(title="Provenant", version="0.1.0")


@app.exception_handler(RedisConnectionError)
@app.exception_handler(RedisTimeoutError)
async def _db_down(_: Request, exc: Exception):
    # Say plainly that the graph is unreachable instead of returning a stack trace.
    return JSONResponse(status_code=503, content={"error": "FalkorDB is not reachable", "detail": type(exc).__name__})


def _require_ready(name: str):
    g = Graph(name)
    if not g.exists():
        raise HTTPException(503, f"graph '{name}' is not loaded; run the ingest script")
    ready = g.read("MATCH (m:Meta) RETURN m.ready")
    if not ready or not ready[0][0]:
        raise HTTPException(503, f"graph '{name}' is still loading")
    return g

# Simple in-process rate limit for the expensive endpoints (fork + replay).
_last: dict[str, float] = {}


def _throttle(key: str, seconds: float = 1.0):
    now = time.time()
    if now - _last.get(key, 0) < seconds:
        raise HTTPException(429, "slow down")
    _last[key] = now


class Ask(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    mode: str | None = Field(None, pattern="^(planner|llm)$")


@app.get("/api/health")
def health():
    g = _require_ready(t1.DEFAULT_GRAPH)
    meta = g.read_dicts("MATCH (m:Meta) RETURN m.posts AS posts, m.agents AS agents, m.ready AS ready")
    swarms = g.read("MATCH (s:Swarm) RETURN count(s)")[0][0]
    return {"graph": g.name, **(meta[0] if meta else {}), "swarms": swarms}


@app.get("/api/swarms")
def swarms(limit: int = Query(25, ge=1, le=200)):
    return t1.list_swarms(limit=limit)


@app.get("/api/swarm/{sid}")
def swarm(sid: int):
    return {"profile": t1.swarm_profile(sid), "owners": t1.owner_spread(sid),
            "channels": t1.steering_channels(sid), "operation": t1.operation_of(sid)}


@app.get("/api/swarm/{sid}/subgraph")
def subgraph(sid: int, agents: int = Query(60, ge=5, le=300)):
    """Nodes and edges for the graph view: swarm, sample agents, owners, keys, steering channels."""
    g = Graph(t1.DEFAULT_GRAPH)
    rows = g.read_dicts(
        "MATCH (s:Swarm {id: $s})-[:INCLUDES]->(a:Agent) WITH s, a LIMIT $n "
        "OPTIONAL MATCH (a)-[:OWNED_BY]->(o:Owner) "
        "OPTIONAL MATCH (a)-[:USES_KEY]->(k:Key)<-[:DEFINED_BY]-(s) "
        "RETURN a.id AS agent, a.name AS name, o.handle AS owner, collect(DISTINCT k.id) AS keys",
        {"s": sid, "n": agents})
    chans = g.read_dicts(
        "MATCH (s:Swarm {id: $s})-[r:STEERED_BY]->(c:Channel) RETURN c.id AS channel, r.share AS share", {"s": sid})
    keys = g.read_dicts(
        "MATCH (s:Swarm {id: $s})-[:DEFINED_BY]->(k:Key) RETURN k.id AS key, k.family AS family, "
        "left(k.sample, 80) AS sample LIMIT 12", {"s": sid})
    return {"swarm": sid, "agents": rows, "channels": chans, "keys": keys}


@app.get("/api/pagerank")
def pagerank(limit: int = Query(10, ge=1, le=50)):
    return t1.channel_pagerank(limit=limit)


@app.get("/api/block")
def block(channel: str = Query(..., min_length=3, max_length=300), hours: float = Query(6.0, ge=0, le=240),
          swarm: int | None = None):
    _throttle("block")
    return t1.block_impact(channel, hours_after_first_seen=hours, swarm=swarm)


@app.post("/api/ask")
def ask(body: Ask):
    _throttle("ask", 2.0)
    return investigate(body.question, mode=body.mode)


# ------------------------------------------------ Tracks 2 and 3: AI Village
from .village import brain as t3  # noqa: E402
from .village.memory import hybrid as _hybrid  # noqa: E402
from .db import tracing as _tracing  # noqa: E402


@app.get("/api/village/searches")
def village_searches(limit: int = Query(30, ge=1, le=200)):
    """Real history searches agents made, with how many cited messages are known (benchmark cases)."""
    g = Graph("village")
    return g.read_dicts(
        "MATCH (a:Agent)-[:SEARCHED]->(s:Search) WHERE s.n_cited >= 2 "
        "RETURN s.id AS id, a.name AS agent, a.id AS agent_id, s.ts AS ts, left(s.query, 240) AS query, "
        "s.n_cited AS cited ORDER BY s.ts DESC LIMIT $n", {"n": limit})


@app.get("/api/village/recall/{search_id}")
def village_recall(search_id: str, k: int = Query(20, ge=5, le=50)):
    """Replay one real search: what shared graph memory returns as of that moment, scored against the citations."""
    g = Graph("village")
    s = g.read_dicts(
        "MATCH (a:Agent)-[:SEARCHED]->(s:Search {id: $id}) OPTIONAL MATCH (s)-[:CITED]->(m:Message) "
        "RETURN a.id AS agent_id, a.name AS agent, s.ts AS ts, s.lo_ts AS lo, s.query AS query, "
        "s.answer_head AS answer_head, collect(m.id) AS gold", {"id": search_id})
    if not s:
        raise HTTPException(404, "no such search")
    s = s[0]
    with _tracing() as tr:
        got = _hybrid(g, s["query"], s["ts"], s["lo"], s["agent_id"], k=k)
    ids = [r["id"] for r in got]
    rows = {r["id"]: r for r in g.read_dicts(
        "MATCH (m:Message) WHERE m.id IN $ids OPTIONAL MATCH (a:Agent)-[:SENT]->(m) "
        "RETURN m.id AS id, m.ts AS ts, a.name AS by, "
        "left(m.text, 260) AS text", {"ids": list(set(ids) | set(s["gold"]))})}
    gold = set(s["gold"])
    return {
        "search": {k2: s[k2] for k2 in ("agent", "ts", "query", "answer_head")},
        "retrieved": [{**rows.get(i, {"id": i}), "why": r["why"], "score": r["score"], "cited": i in gold}
                      for i, r in zip(ids, got)],
        "missed": [rows[i] for i in gold - set(ids) if i in rows],
        "recall": round(len(gold & set(ids)) / len(gold), 4) if gold else None,
        "other_agents_share": round(sum(1 for i in gold if rows.get(i, {}).get("by") != s["agent"]) / len(gold), 4)
        if gold else None,
        "evidence": tr.to_list(),
    }


@app.get("/api/village/brain/{tool}")
def village_brain(tool: str, q: str = Query(..., min_length=2, max_length=300), until_ts: float | None = None):
    """Company-brain tools: who_owns, sites_of, who_knows, decisions, goal_team, handoffs, message."""
    if tool not in t3.TOOLS:
        raise HTTPException(404, f"tool must be one of {sorted(t3.TOOLS)}")
    arg = {"who_owns": "site", "sites_of": "agent", "who_knows": "topic", "decisions": "topic",
           "goal_team": "goal_text", "handoffs": "agent", "message": "mid"}[tool]
    kw = {arg: q}
    if until_ts is not None and tool not in ("goal_team", "message"):
        kw["until_ts"] = until_ts
    return t3.TOOLS[tool](**kw)


@app.get("/api/village/stats")
def village_stats():
    g = _require_ready("village")
    m = g.read_dicts("MATCH (m:Meta) RETURN m.messages AS messages, m.agents AS agents, m.sites AS sites, "
                     "m.searches AS searches, m.cited AS cited, m.goals AS goals")
    bench = UI.parent / "reports" / "bench_memory.json"
    import json as _json
    return {**(m[0] if m else {}), "bench": _json.loads(bench.read_text()) if bench.exists() else None}


@app.get("/")
def index():
    return FileResponse(UI / "index.html")


if UI.exists():
    app.mount("/ui", StaticFiles(directory=UI), name="ui")
