"""Swarm detection done inside FalkorDB.

1. Candidate keys: shared by at least ``min_agents`` agents and by at most
   ``max_key_agent_share`` of all agents (wider keys are platform boilerplate).
2. Key-overlap edges: two candidate keys are linked when they share at least
   half of the smaller key's agents. This is the rule that stops one very
   busy agent from gluing unrelated campaigns together.
3. ``algo.WCC`` over (Key)-[:OVERLAP]-(Key) gives campaigns.
4. Each campaign with at least ``min_agents`` agents becomes a (:Swarm) node
   with [:DEFINED_BY] edges to its keys and [:INCLUDES] edges to its agents.
5. Steering channels: per swarm, Cypher counts how many swarm posts carry each
   channel versus the rest of the corpus; a one-sided Fisher exact test with
   Bonferroni correction decides, and a channel only counts when agents with
   at least two different owners carried it. Results become
   (:Swarm)-[:STEERED_BY {share, enrichment, p_adj, agents, owners}]->(:Channel).
6. ``algo.WCC`` over (Swarm)-[:STEERED_BY]-(Channel) groups swarms that share
   infrastructure into operations: (:Swarm {operation}).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

from scipy.stats import fisher_exact

from ..db import Graph, batched


@dataclass
class DetectConfig:
    min_agents: int = 3
    max_key_agent_share: float = 0.05
    overlap: float = 0.5
    alpha: float = 0.01
    min_owners: int = 2
    max_channels_per_swarm: int = 5
    min_near_similarity: float = 0.3


def reset(g: Graph):
    g.write("MATCH (s:Swarm) DETACH DELETE s")
    g.write("MATCH (:Key)-[r:OVERLAP]->() DELETE r")
    g.write("MATCH (k:Candidate) REMOVE k:Candidate")


def detect(g: Graph, cfg: DetectConfig | None = None, log=print) -> dict:
    cfg = cfg or DetectConfig()
    t = {}
    t0 = time.perf_counter()
    reset(g)
    n_agents = g.read("MATCH (a:Agent) RETURN count(a)")[0][0]
    max_agents = int(cfg.max_key_agent_share * n_agents)
    g.write("MATCH (k:Key) WHERE k.n_agents >= $lo AND k.n_agents <= $hi SET k:Candidate",
            {"lo": cfg.min_agents, "hi": max_agents})
    n_cand = g.read("MATCH (k:Candidate) RETURN count(k)")[0][0]
    t["candidates"] = round(time.perf_counter() - t0, 2)
    log(f"  candidate keys: {n_cand} (n_agents in [{cfg.min_agents}, {max_agents}])")

    # Overlap edges, built from shared agents. Agents with many keys are
    # processed in one pass; the WHERE on internal ids avoids duplicates.
    t0 = time.perf_counter()
    g.write(
        "MATCH (k1:Candidate)<-[:USES_KEY]-(a:Agent)-[:USES_KEY]->(k2:Candidate) "
        "WHERE id(k1) < id(k2) "
        "WITH k1, k2, count(a) AS shared "
        "WHERE shared >= $min AND shared >= $ov * CASE WHEN k1.n_agents < k2.n_agents THEN k1.n_agents ELSE k2.n_agents END "
        "CREATE (k1)-[:OVERLAP {shared: shared}]->(k2)",
        {"ov": cfg.overlap, "min": cfg.min_agents},
    )
    n_ov = g.read("MATCH ()-[r:OVERLAP]->() RETURN count(r)")[0][0]
    t["overlap"] = round(time.perf_counter() - t0, 2)
    log(f"  overlap edges: {n_ov}")

    # Campaigns = weakly connected components of the key-overlap graph.
    t0 = time.perf_counter()
    comp = g.read(
        "CALL algo.WCC({nodeLabels: ['Candidate'], relationshipTypes: ['OVERLAP']}) "
        "YIELD node, componentId RETURN node.id, componentId", timeout_ms=600000)
    by_comp: dict[int, list[str]] = {}
    for kid, c in comp:
        by_comp.setdefault(c, []).append(kid)
    t["wcc"] = round(time.perf_counter() - t0, 2)

    # Materialise swarms; members are agents using any key of the campaign.
    t0 = time.perf_counter()
    # Stable swarm id: hash of the smallest defining key, so the same campaign
    # keeps the same id across re-ingests (WCC component ids are internal ids).
    rows = [{"cid": int(hashlib.sha1(min(ks).encode()).hexdigest()[:7], 16), "keys": ks} for ks in by_comp.values()]
    for chunk in batched(rows, 500):
        g.write(
            "UNWIND $rows AS r "
            "MATCH (k:Candidate) WHERE k.id IN r.keys "
            "WITH r, collect(k) AS ks "
            "UNWIND ks AS k MATCH (k)<-[:USES_KEY]-(a:Agent) "
            "WITH r, ks, collect(DISTINCT a) AS members "
            "WHERE size(members) >= $min "
            "CREATE (s:Swarm {id: r.cid, n_agents: size(members), n_keys: size(ks)}) "
            "FOREACH (k IN ks | CREATE (s)-[:DEFINED_BY]->(k)) "
            "FOREACH (a IN members | CREATE (s)-[:INCLUDES]->(a))",
            {"rows": chunk, "min": cfg.min_agents},
        )
    # Swarm posts: posts by members that carry one of the defining keys.
    g.write(
        "MATCH (s:Swarm)-[:DEFINED_BY]->(:Key)<-[:HAS_KEY]-(p:Post)<-[:POSTED]-(a:Agent)<-[:INCLUDES]-(s) "
        "WITH s, collect(DISTINCT p) AS ps "
        "SET s.n_posts = size(ps), "
        "    s.first_ts = reduce(m = 1e18, p IN ps | CASE WHEN p.ts < m THEN p.ts ELSE m END), "
        "    s.last_ts = reduce(m = 0.0, p IN ps | CASE WHEN p.ts > m THEN p.ts ELSE m END)"
    )
    g.write(
        "MATCH (s:Swarm)-[:INCLUDES]->(a:Agent) OPTIONAL MATCH (a)-[:OWNED_BY]->(o:Owner) "
        "WITH s, count(DISTINCT o) AS owners, count(a) AS n, count(o) AS known "
        "SET s.distinct_owners = owners, s.owners_known = known"
    )
    g.write(
        "MATCH (s:Swarm)-[:INCLUDES]->(a:Agent)-[:OWNED_BY]->(o:Owner) "
        "WITH s, o, count(a) AS n ORDER BY n DESC "
        "WITH s, collect({h: o.handle, n: n})[0] AS top "
        "SET s.top_owner = top.h, s.top_owner_share = toFloat(top.n) / s.n_agents"
    )
    g.write(
        "MATCH (s:Swarm)-[:DEFINED_BY]->(k:Key) "
        "WITH s, collect(k.family) AS fams, avg(k.sim) AS sim "
        "SET s.families = fams, s.text_similarity = sim"
    )
    # A campaign held together only by paraphrase families must actually
    # read alike; otherwise it is LSH chaining noise.
    dropped = g.read(
        "MATCH (s:Swarm) WHERE all(f IN s.families WHERE f = 'near') AND s.text_similarity < $t "
        "RETURN count(s)", {"t": cfg.min_near_similarity})[0][0]
    g.write("MATCH (s:Swarm) WHERE all(f IN s.families WHERE f = 'near') AND s.text_similarity < $t "
            "DETACH DELETE s", {"t": cfg.min_near_similarity})
    log(f"  dropped low-similarity paraphrase-only swarms: {dropped}")
    n_sw = g.read("MATCH (s:Swarm) RETURN count(s)")[0][0]
    t["swarms"] = round(time.perf_counter() - t0, 2)
    log(f"  swarms: {n_sw}")

    t0 = time.perf_counter()
    ch = steering(g, cfg)
    t["steering"] = round(time.perf_counter() - t0, 2)
    t0 = time.perf_counter()
    ops = operations(g)
    t["operations"] = round(time.perf_counter() - t0, 2)
    _label(g)
    return {"candidate_keys": n_cand, "overlap_edges": n_ov, "swarms": n_sw,
            "steered_swarms": ch["steered_swarms"], "steering_edges": ch["edges"],
            "operations": ops, "timing_s": t}


def steering(g: Graph, cfg: DetectConfig) -> dict:
    """Fisher test per (swarm, channel) on counts pulled with Cypher."""
    g.write("MATCH (:Swarm)-[r:STEERED_BY]->() DELETE r")
    n_corpus = g.read("MATCH (p:Post) RETURN count(p)")[0][0]
    rows = g.read(
        "MATCH (s:Swarm)-[:DEFINED_BY]->(:Key)<-[:HAS_KEY]-(p:Post)<-[:POSTED]-(a:Agent)<-[:INCLUDES]-(s) "
        "WITH DISTINCT s, p, a "
        "MATCH (p)-[:LINKS_TO]->(c:Channel) "
        "OPTIONAL MATCH (a)-[:OWNED_BY]->(o:Owner) "
        "RETURN s.id, s.n_posts, c.id, c.n_posts, count(DISTINCT p), count(DISTINCT a), count(DISTINCT o)",
        timeout_ms=600000,
    )
    n_tests = max(1, len(rows))
    found: dict[int, list] = {}
    for sid, n_s, cid, c_total, k, n_ag, n_own in rows:
        if n_ag < cfg.min_agents or n_own < cfg.min_owners:
            continue
        k_rest, n_rest = c_total - k, n_corpus - n_s
        _, p = fisher_exact([[k, n_s - k], [k_rest, max(n_rest - k_rest, 0)]], alternative="greater")
        p_adj = min(1.0, p * n_tests)
        if p_adj > cfg.alpha:
            continue
        rest_rate = k_rest / max(n_rest, 1)
        found.setdefault(sid, []).append({
            "s": sid, "c": cid, "posts": int(k), "share": round(k / n_s, 4),
            "enrichment": round((k / n_s) / max(rest_rate, 1 / n_corpus), 1),
            "p_adj": float(p_adj), "agents": int(n_ag), "owners": int(n_own),
        })
    edges = []
    for sid, fs in found.items():
        fs.sort(key=lambda d: (d["share"], d["enrichment"]), reverse=True)
        edges += fs[:cfg.max_channels_per_swarm]
    for chunk in batched(edges, 2000):
        g.write("UNWIND $rows AS r MATCH (s:Swarm {id: r.s}), (c:Channel {id: r.c}) "
                "CREATE (s)-[:STEERED_BY {posts: r.posts, share: r.share, enrichment: r.enrichment, "
                "p_adj: r.p_adj, agents: r.agents, owners: r.owners, tests: $n}]->(c)",
                {"rows": chunk, "n": n_tests})
    return {"steered_swarms": len(found), "edges": len(edges), "tests": n_tests}


def operations(g: Graph) -> int:
    """Swarms that share steering infrastructure form one operation (WCC)."""
    rows = g.read("CALL algo.WCC({nodeLabels: ['Swarm', 'Channel'], relationshipTypes: ['STEERED_BY']}) "
                  "YIELD node, componentId WITH node, componentId WHERE node:Swarm "
                  "RETURN node.id, componentId")
    # Name each operation by its smallest swarm id so it is stable across runs.
    first: dict = {}
    for sid, comp in rows:
        first[comp] = min(first.get(comp, sid), sid)
    rows = [[sid, first[comp]] for sid, comp in rows]
    g.write("UNWIND $rows AS r MATCH (s:Swarm {id: r[0]}) SET s.operation = r[1]", {"rows": rows})
    multi = g.read("MATCH (s:Swarm) WITH s.operation AS op, count(s) AS n WHERE n > 1 RETURN count(op)")
    return multi[0][0] if multi else 0


def _label(g: Graph):
    """A short, rule-based kind label so the agent and UI can sort swarms."""
    g.write(
        "MATCH (s:Swarm) "
        "OPTIONAL MATCH (s)-[:DEFINED_BY]->(:Key)<-[:HAS_KEY]-(p:Post) "
        "WITH s, count(DISTINCT p) AS kp, sum(CASE WHEN p.injection THEN 1 ELSE 0 END) AS inj "
        "SET s.injection_share = CASE WHEN kp = 0 THEN 0.0 ELSE toFloat(inj) / kp END"
    )
    g.write(
        "MATCH (s:Swarm) SET s.kind = CASE "
        "WHEN s.injection_share >= 0.2 OR 'beneficiary' IN s.families THEN 'agent-recruitment' "
        "WHEN 'payload' IN s.families THEN 'payload campaign' "
        "WHEN s.owners_known >= 3 AND s.top_owner_share >= 0.5 THEN 'single-owner fleet' "
        "ELSE 'copy-paste wave' END"
    )
