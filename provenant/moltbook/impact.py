"""block_impact: what blocking a channel would have cut off, measured on a fork.

The question a trust-and-safety responder actually has is not "is this
channel suspicious" but "if I block it now, what stops, and what else breaks".
We answer it by replay, not estimate:

1. Fork the live graph with ``GRAPH.COPY`` so the real graph is never touched.
2. On the fork, simulate the platform rejecting every post that links to the
   channel from time ``T`` on (``DETACH DELETE`` of those posts).
3. Recount each affected swarm (members who still have at least one swarm
   post, swarm posts remaining) and compare with the unforked graph.
4. Count collateral damage: blocked posts that belong to no swarm. These are
   ordinary users the block would also hit, and the number a responder needs
   before acting.

Limits, returned with every result: this measures direct reach only. Agents
would likely adapt (switch link, rephrase), so "posts prevented" is an upper
bound on what the block achieves, and "agents cut off" assumes no adaptation.
"""

from __future__ import annotations

import hashlib
import time

from ..db import Graph

LIMITS = [
    "Direct reach only: assumes swarm agents do not switch to another link after the block.",
    "Posts are blocked from time T on; anything posted before T already happened.",
    "Collateral counts posts outside every detected swarm; some may be undetected coordination.",
]

_SWARM_COUNTS = (
    "MATCH (s:Swarm)-[:DEFINED_BY]->(:Key)<-[:HAS_KEY]-(p:Post)<-[:POSTED]-(a:Agent)<-[:INCLUDES]-(s) "
    "WHERE s.id IN $ids "
    "RETURN s.id AS swarm, count(DISTINCT a) AS agents, count(DISTINCT p) AS posts"
)


_UNIQUE = (
    "MATCH (s:Swarm)-[:DEFINED_BY]->(:Key)<-[:HAS_KEY]-(p:Post)<-[:POSTED]-(a:Agent)<-[:INCLUDES]-(s) "
    "WHERE s.id IN $ids "
    "RETURN count(DISTINCT a) AS agents, count(DISTINCT p) AS posts"
)


def _fork_name(base: str, channel: str, t: float) -> str:
    h = hashlib.sha1(f"{channel}|{t}".encode()).hexdigest()[:10]
    return f"{base}__whatif_{h}"


def channel_window(g: Graph, channel: str) -> dict:
    rows = g.read_dicts(
        "MATCH (c:Channel {id: $c})<-[:LINKS_TO]-(p:Post) "
        "RETURN min(p.ts) AS first_ts, max(p.ts) AS last_ts, count(p) AS posts",
        {"c": channel})
    return rows[0] if rows else {"first_ts": None, "last_ts": None, "posts": 0}


def block_impact(g: Graph, channel: str, at_ts: float | None = None,
                 hours_after_first: float = 6.0, keep_fork: bool = False) -> dict:
    """Simulate blocking ``channel`` from ``at_ts`` (default: first seen + 6 h)."""
    t_start = time.perf_counter()
    win = channel_window(g, channel)
    if not win["posts"]:
        return {"channel": channel, "error": "channel not found or never linked"}
    t = at_ts if at_ts is not None else win["first_ts"] + hours_after_first * 3600

    # Which swarms are touched: steered by the channel, or with posts linking it.
    affected = g.read_dicts(
        "MATCH (c:Channel {id: $c}) "
        "OPTIONAL MATCH (s1:Swarm)-[r:STEERED_BY]->(c) "
        "WITH c, collect(DISTINCT {id: s1.id, share: r.share, owners: r.owners}) AS steered "
        "OPTIONAL MATCH (c)<-[:LINKS_TO]-(:Post)-[:HAS_KEY]->(:Key)<-[:DEFINED_BY]-(s2:Swarm) "
        "RETURN steered, collect(DISTINCT s2.id) AS touched",
        {"c": channel})[0]
    steered = {d["id"]: d for d in affected["steered"] if d["id"] is not None}
    ids = sorted(set(steered) | set(affected["touched"]))

    # Posts the block would reject, split into swarm posts and collateral.
    blocked = g.read_dicts(
        "MATCH (c:Channel {id: $c})<-[:LINKS_TO]-(p:Post) WHERE p.ts >= $t "
        "OPTIONAL MATCH (p)-[:HAS_KEY]->(:Key)<-[:DEFINED_BY]-(s:Swarm) "
        "WITH p, count(s) AS in_swarm "
        "RETURN count(p) AS blocked, sum(CASE WHEN in_swarm > 0 THEN 1 ELSE 0 END) AS swarm_posts, "
        "sum(CASE WHEN in_swarm = 0 THEN 1 ELSE 0 END) AS collateral",
        {"c": channel, "t": t})[0]

    before = {r["swarm"]: r for r in g.read_dicts(_SWARM_COUNTS, {"ids": ids})} if ids else {}
    uniq_before = g.read_dicts(_UNIQUE, {"ids": ids})[0] if ids else {"agents": 0, "posts": 0}

    fork = g.copy_to(_fork_name(g.name, channel, t))
    try:
        # Collect first, then delete: deleting inside the same index scan
        # skipped rows in testing. Then verify nothing blocked survived.
        # `p.ts + 0.0` keeps the planner off the Post.ts index on the fork, so
        # the delete walks from the channel and cannot depend on index state.
        fork.write("MATCH (c:Channel {id: $c})<-[:LINKS_TO]-(p:Post) WHERE p.ts + 0.0 >= $t "
                   "WITH collect(DISTINCT p) AS ps UNWIND ps AS p DETACH DELETE p",
                   {"c": channel, "t": t})
        left = fork.read("MATCH (c:Channel {id: $c})<-[:LINKS_TO]-(p:Post) WHERE p.ts + 0.0 >= $t RETURN count(p)",
                         {"c": channel, "t": t})[0][0]
        if left:
            raise RuntimeError(f"block replay incomplete: {left} blocked posts remain on fork")
        after = {r["swarm"]: r for r in fork.read_dicts(_SWARM_COUNTS, {"ids": ids})} if ids else {}
        # Unique totals across all affected swarms: a post or agent that sits
        # in two swarms is counted once.
        uniq_after = fork.read_dicts(_UNIQUE, {"ids": ids})[0] if ids else {"agents": 0, "posts": 0}
        # Agents who never got to post in the swarm because their first swarm
        # post would have been blocked: the recruits the block prevents.
        meta = {r["id"]: r for r in g.read_dicts(
            "MATCH (s:Swarm) WHERE s.id IN $ids RETURN s.id AS id, s.kind AS kind, "
            "s.first_ts AS first_ts, s.last_ts AS last_ts", {"ids": ids})} if ids else {}
    finally:
        if not keep_fork:
            fork.delete()

    swarms = []
    for sid in ids:
        b = before.get(sid, {"agents": 0, "posts": 0})
        a = after.get(sid, {"agents": 0, "posts": 0})
        swarms.append({
            "swarm": sid, "kind": meta.get(sid, {}).get("kind"),
            "steered_by_channel": sid in steered,
            "channel_share": steered.get(sid, {}).get("share"),
            "agents_before": b["agents"], "agents_after": a["agents"],
            "agents_cut_off": b["agents"] - a["agents"],
            "posts_before": b["posts"], "posts_after": a["posts"],
            "posts_prevented": b["posts"] - a["posts"],
            "share_posts_prevented": round((b["posts"] - a["posts"]) / b["posts"], 4) if b["posts"] else 0.0,
        })
    swarms.sort(key=lambda d: d["posts_prevented"], reverse=True)

    # Evidence: a few concrete agent -> post -> channel paths after T.
    paths = g.read_dicts(
        "MATCH (a:Agent)-[:POSTED]->(p:Post)-[:LINKS_TO]->(c:Channel {id: $c}) WHERE p.ts >= $t "
        "MATCH (p)-[:HAS_KEY]->(:Key)<-[:DEFINED_BY]-(s:Swarm) "
        "RETURN DISTINCT a.name AS agent, p.id AS post, p.ts AS ts, s.id AS swarm, left(p.text, 140) AS text "
        "ORDER BY ts LIMIT 3", {"c": channel, "t": t})

    total_prev = uniq_before["posts"] - uniq_after["posts"]
    agents_cut = uniq_before["agents"] - uniq_after["agents"]
    return {
        "channel": channel,
        "block_at_ts": t,
        "hours_after_first_seen": round((t - win["first_ts"]) / 3600, 2),
        "channel_posts_total": win["posts"],
        "blocked_posts": int(blocked["blocked"] or 0),
        "blocked_swarm_posts": int(blocked["swarm_posts"] or 0),
        "collateral_posts": int(blocked["collateral"] or 0),
        "collateral_share": round(blocked["collateral"] / blocked["blocked"], 4) if blocked["blocked"] else 0.0,
        "swarm_posts_prevented": total_prev,
        "swarm_agents_cut_off": agents_cut,
        "swarm_agents_before": uniq_before["agents"],
        "swarm_posts_before": uniq_before["posts"],
        "swarms": swarms,
        "evidence_paths": paths,
        "limits": LIMITS,
        "ms": round((time.perf_counter() - t_start) * 1000, 1),
    }


def rank_blocks(g: Graph, hours_after_first: float = 6.0, log=print) -> list[dict]:
    """Run block_impact for every steering channel; rank by prevented swarm posts per collateral post."""
    chans = [r[0] for r in g.read("MATCH (:Swarm)-[:STEERED_BY]->(c:Channel) RETURN DISTINCT c.id")]
    out = []
    for c in chans:
        r = block_impact(g, c, hours_after_first=hours_after_first)
        if "error" in r:
            continue
        r["precision"] = round(r["blocked_swarm_posts"] / r["blocked_posts"], 4) if r["blocked_posts"] else 0.0
        out.append(r)
        log(f"  {c}: prevents {r['swarm_posts_prevented']} swarm posts, cuts {r['swarm_agents_cut_off']} agents, "
            f"collateral {r['collateral_posts']} ({r['ms']} ms)")
    out.sort(key=lambda r: (r["swarm_posts_prevented"], -r["collateral_posts"]), reverse=True)
    return out
