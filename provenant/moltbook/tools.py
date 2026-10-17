"""Agent tools for Track 1. Each returns {"result": ..., "evidence": [queries]}.

Every tool is a thin, read-only wrapper over Cypher or a FalkorDB algorithm,
except ``block_impact``, which writes only to a throwaway fork. The evidence
list is the exact Cypher, parameters, row count and time of each query run,
so any answer the agent gives can be traced back to the graph.
"""

from __future__ import annotations

from ..db import Graph, tracing
from .impact import block_impact as _block_impact

DEFAULT_GRAPH = "moltbook_w1"


def _run(fn, *args, **kwargs) -> dict:
    with tracing() as tr:
        result = fn(*args, **kwargs)
    return {"result": result, "evidence": tr.to_list()}


def _g(graph: str | None) -> Graph:
    return Graph(graph or DEFAULT_GRAPH)


KINDS = ["payload campaign", "copy-paste wave", "agent-recruitment", "single-owner fleet"]


def list_swarms(graph: str | None = None, limit: int = 15, kind: str | None = None,
                search: str | None = None) -> dict:
    """Largest detected swarms with their real numeric ids, size, owners, kind and steering channels.

    `search` matches a word against swarm kinds and steering channels (e.g. 'clawtasks', 'mint',
    'skill', 'chatr'). `kind` is one of: payload campaign, copy-paste wave, agent-recruitment,
    single-owner fleet. Use the returned `swarm` id in every other tool; never invent ids.
    """
    term = (search or "").strip().lower()
    if term in ("mint", "token", "token mint", "payload"):
        kind, term = "payload campaign", ""
    def f():
        return _g(graph).read_dicts(
            "MATCH (s:Swarm) WHERE ($kind IS NULL OR toLower(s.kind) CONTAINS toLower($kind)) "
            "OPTIONAL MATCH (s)-[r0:STEERED_BY]->(c0:Channel) "
            "WITH s, collect(c0.id) AS all_ch "
            "WHERE $term = '' OR toLower(s.kind) CONTAINS $term OR any(x IN all_ch WHERE toLower(x) CONTAINS $term) "
            "OPTIONAL MATCH (s)-[r:STEERED_BY]->(c:Channel) "
            "WITH s, r, c ORDER BY r.share DESC "
            "WITH s, collect(c.id) AS chans, collect(r.share)[0] AS top_share "
            "RETURN s.id AS swarm, s.kind AS kind, s.n_agents AS agents, s.n_posts AS posts, "
            "s.distinct_owners AS owners, round(1000 * s.top_owner_share) / 1000 AS top_owner_share, "
            "s.operation AS operation, chans[0] AS top_channel, top_share, chans AS channels "
            "ORDER BY agents DESC LIMIT $limit", {"limit": int(limit), "kind": kind, "term": term})
    return _run(f)


def _no_swarm(swarm) -> dict:
    return {"error": f"no swarm with id {swarm}. Swarm ids are large numbers returned by list_swarms; "
                     f"call list_swarms (optionally with search='clawtasks', 'mint', ...) to get a real id."}


def swarm_profile(swarm: int, graph: str | None = None) -> dict:
    """Who is in a swarm, what text defines it, when it ran, and who owns its agents."""
    def f():
        g = _g(graph)
        head = g.read_dicts(
            "MATCH (s:Swarm {id: $s}) RETURN s.kind AS kind, s.n_agents AS agents, s.n_posts AS posts, "
            "s.first_ts AS first_ts, s.last_ts AS last_ts, s.distinct_owners AS owners, "
            "s.owners_known AS owners_known, s.top_owner AS top_owner, s.top_owner_share AS top_owner_share, "
            "s.families AS key_families, s.text_similarity AS text_similarity, s.operation AS operation",
            {"s": int(swarm)})
        if not head:
            return _no_swarm(swarm)
        keys = g.read_dicts(
            "MATCH (s:Swarm {id: $s})-[:DEFINED_BY]->(k:Key) "
            "RETURN k.id AS key, k.family AS family, k.n_agents AS agents, left(k.sample, 160) AS sample "
            "ORDER BY agents DESC LIMIT 5", {"s": int(swarm)})
        sample = g.read_dicts(
            "MATCH (s:Swarm {id: $s})-[:INCLUDES]->(a:Agent) OPTIONAL MATCH (a)-[:OWNED_BY]->(o:Owner) "
            "RETURN a.name AS agent, o.handle AS owner LIMIT 8", {"s": int(swarm)})
        return {**head[0], "defining_keys": keys, "sample_members": sample}
    return _run(f)


def steering_channels(swarm: int, graph: str | None = None) -> dict:
    """Outside links/handles a swarm carries far more than the rest of the platform (Fisher, Bonferroni)."""
    def f():
        g = _g(graph)
        if not g.read("MATCH (s:Swarm {id: $s}) RETURN 1", {"s": int(swarm)}):
            return _no_swarm(swarm)
        rows = g.read_dicts(
            "MATCH (s:Swarm {id: $s})-[r:STEERED_BY]->(c:Channel) "
            "RETURN c.id AS channel, c.kind AS kind, r.share AS share_of_swarm_posts, r.enrichment AS enrichment, "
            "r.p_adj AS p_bonferroni, r.agents AS agents, r.owners AS owners "
            "ORDER BY share_of_swarm_posts DESC", {"s": int(swarm)})
        return rows  # empty list = no statistically significant steering channel (UI and planner expect a list)
    return _run(f)


def channel_reach(channel: str, graph: str | None = None) -> dict:
    """Every swarm that uses a channel, and how much of each swarm it reaches (multi-hop)."""
    def f():
        return _g(graph).read_dicts(
            "MATCH (c:Channel {id: $c})<-[:LINKS_TO]-(p:Post)-[:HAS_KEY]->(:Key)<-[:DEFINED_BY]-(s:Swarm) "
            "WITH s, count(DISTINCT p) AS posts "
            "OPTIONAL MATCH (s)-[r:STEERED_BY]->(:Channel {id: $c}) "
            "RETURN s.id AS swarm, s.kind AS kind, s.n_agents AS swarm_agents, posts AS posts_with_channel, "
            "r IS NOT NULL AS significant ORDER BY posts DESC LIMIT 20", {"c": channel})
    return _run(f)


def owner_spread(swarm: int, graph: str | None = None) -> dict:
    """How many different human owners run a swarm's agents (is there one operator to blame?)."""
    def f():
        return _g(graph).read_dicts(
            "MATCH (s:Swarm {id: $s})-[:INCLUDES]->(a:Agent) OPTIONAL MATCH (a)-[:OWNED_BY]->(o:Owner) "
            "WITH o, count(a) AS agents ORDER BY agents DESC "
            "RETURN coalesce(o.handle, '(unknown)') AS owner, agents LIMIT 10", {"s": int(swarm)})
    return _run(f)


def operation_of(swarm: int, graph: str | None = None) -> dict:
    """Other swarms connected to this one through shared steering channels (WCC operation)."""
    def f():
        # Two-hop path swarm -> shared channel -> other swarm; the operation id
        # comes from algo.WCC over STEERED_BY during detection.
        return _g(graph).read_dicts(
            "MATCH (s:Swarm {id: $s})-[:STEERED_BY]->(c:Channel)<-[:STEERED_BY]-(o:Swarm) WHERE o <> s "
            "RETURN o.id AS swarm, o.kind AS kind, o.n_agents AS agents, collect(c.id) AS shared_channels, "
            "s.operation AS operation ORDER BY agents DESC LIMIT 10", {"s": int(swarm)})
    return _run(f)


def channel_pagerank(graph: str | None = None, limit: int = 10) -> dict:
    """FalkorDB PageRank over Swarm-[:STEERED_BY]->Channel: the most central steering infrastructure."""
    def f():
        return _g(graph).read_dicts(
            # Unlabelled call runs PageRank over every node joined by STEERED_BY
            # edges (Swarm -> Channel), so channels steering many big swarms rank high.
            "CALL algo.pageRank(null, 'STEERED_BY') YIELD node, score "
            "WITH node, score WHERE node:Channel "
            "MATCH (node)<-[r:STEERED_BY]-(s:Swarm) "
            "RETURN node.id AS channel, score AS pagerank, count(s) AS swarms, "
            "sum(s.n_agents) AS agents ORDER BY pagerank DESC, agents DESC LIMIT $limit", {"limit": int(limit)})
    return _run(f)


def block_impact(channel: str, hours_after_first_seen: float = 6.0, swarm: int | None = None,
                 graph: str | None = None) -> dict:
    """Replay a block of `channel` on a GRAPH.COPY fork: swarm posts/agents cut off vs collateral.

    With `swarm`, the block starts `hours_after_first_seen` hours after that swarm's first post
    (the responder's clock); otherwise after the channel's first appearance anywhere.
    """
    def f():
        g = _g(graph)
        at = None
        if swarm is not None:
            rows = g.read("MATCH (s:Swarm {id: $s}) RETURN s.first_ts", {"s": int(swarm)})
            if not rows:
                return _no_swarm(swarm)
            if rows[0][0]:
                at = rows[0][0] + float(hours_after_first_seen) * 3600
        r = _block_impact(g, channel, at_ts=at, hours_after_first=float(hours_after_first_seen))
        if at is not None and "error" not in r:
            r["clock"] = f"swarm {swarm} start + {hours_after_first_seen} h"
            r["hours_after_swarm_start"] = float(hours_after_first_seen)
        return r
    return _run(f)


TOOLS = {f.__name__: f for f in (list_swarms, swarm_profile, steering_channels, channel_reach,
                                 owner_spread, operation_of, channel_pagerank, block_impact)}
