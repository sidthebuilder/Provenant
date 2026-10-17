"""MCP server exposing Provenant's graph tools to any MCP client (Kiro, Claude Desktop, etc.).

Run:  python -m provenant.mcp_server
Each tool returns JSON with the result and the exact Cypher evidence behind it.
"""

from __future__ import annotations

import json

from mcp.server.mcpserver import MCPServer  # mcp 2.x (FastMCP was renamed)

from .moltbook import tools as t1

mcp = MCPServer("provenant")


def _j(x) -> str:
    return json.dumps(x, default=str)


@mcp.tool()
def list_swarms(limit: int = 15, kind: str | None = None, search: str | None = None) -> str:
    """List detected swarms with their real ids. `search` matches kinds and steering channels (e.g. 'clawtasks')."""
    return _j(t1.list_swarms(limit=limit, kind=kind, search=search))


@mcp.tool()
def swarm_profile(swarm: int) -> str:
    """Members, defining text keys, time span and owners of one swarm."""
    return _j(t1.swarm_profile(swarm))


@mcp.tool()
def steering_channels(swarm: int) -> str:
    """Outside links, skill files and handles that statistically steer a swarm."""
    return _j(t1.steering_channels(swarm))


@mcp.tool()
def channel_reach(channel: str) -> str:
    """All swarms that carry a channel and how many of their posts do."""
    return _j(t1.channel_reach(channel))


@mcp.tool()
def owner_spread(swarm: int) -> str:
    """Distribution of human owners behind a swarm's agents."""
    return _j(t1.owner_spread(swarm))


@mcp.tool()
def operation_of(swarm: int) -> str:
    """Swarms linked to this one through shared steering channels, with graph paths."""
    return _j(t1.operation_of(swarm))


@mcp.tool()
def channel_pagerank(limit: int = 10) -> str:
    """Most central steering channels by FalkorDB PageRank."""
    return _j(t1.channel_pagerank(limit=limit))


@mcp.tool()
def block_impact(channel: str, hours_after_first_seen: float = 6.0) -> str:
    """Simulate blocking a channel on a forked graph and report what it would cut off and the collateral."""
    return _j(t1.block_impact(channel, hours_after_first_seen))


# ------------------------------------------------ Tracks 2 and 3: AI Village
from .db import Graph, tracing  # noqa: E402
from .village import brain as t3  # noqa: E402
from .village.memory import hybrid  # noqa: E402


@mcp.tool()
def village_recall(query: str, at_ts: float, agent: str | None = None, k: int = 20) -> str:
    """Shared village memory as of a moment: messages relevant to `query` written before `at_ts` (epoch s), by any agent."""
    g = Graph("village")
    aid = None
    if agent:
        r = g.read("MATCH (a:Agent {name: $n}) RETURN a.id", {"n": agent})
        aid = r[0][0] if r else None
    with tracing() as tr:
        got = hybrid(g, query, at_ts, None, aid, k=k)
        texts = {r["id"]: r for r in g.read_dicts(
            "MATCH (m:Message) WHERE m.id IN $ids OPTIONAL MATCH (a:Agent)-[:SENT]->(m) "
            "RETURN m.id AS id, m.ts AS ts, a.name AS by, "
            "left(m.text, 300) AS text", {"ids": [x["id"] for x in got]})}
    return _j({"result": [{**texts.get(x["id"], {}), "why": x["why"]} for x in got], "evidence": tr.to_list()})


@mcp.tool()
def who_owns(site: str) -> str:
    """Company brain: who introduced a site or repo in the village, and who links it most."""
    return _j(t3.who_owns(site))


@mcp.tool()
def sites_of(agent: str) -> str:
    """Company brain: repos and sites an agent introduced first."""
    return _j(t3.sites_of(agent))


@mcp.tool()
def who_knows(topic: str) -> str:
    """Company brain: agents who wrote most about a topic, with source messages."""
    return _j(t3.who_knows(topic))


@mcp.tool()
def decisions(topic: str) -> str:
    """Company brain: decision messages about a topic, with who, when, room and the goal at the time."""
    return _j(t3.decisions(topic))


@mcp.tool()
def goal_team(goal_text: str) -> str:
    """Company brain: who worked under a village goal, who addressed whom, early decisions."""
    return _j(t3.goal_team(goal_text))


@mcp.tool()
def handoffs(agent: str) -> str:
    """Company brain: help requests to and from an agent."""
    return _j(t3.handoffs(agent))


@mcp.tool()
def open_message(message_id: str) -> str:
    """Open a source message cited by another tool."""
    return _j(t3.message(message_id))


if __name__ == "__main__":
    mcp.run()
