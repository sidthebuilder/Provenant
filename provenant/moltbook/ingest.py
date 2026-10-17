"""Load a Moltbook slice into a FalkorDB graph.

Schema::

    (:Agent {id, name, created_ts})-[:OWNED_BY]->(:Owner {handle})
    (:Agent)-[:POSTED]->(:Post {id, ts, venue, text, injection})
    (:Post)-[:HAS_KEY]->(:Key {id, family, sample})
    (:Post)-[:LINKS_TO]->(:Channel {id, kind})
    (:Agent)-[:USES_KEY {posts}]->(:Key)            derived inside the graph
    (:Meta {dataset, posts, agents, ready})

Keys and channels are only stored when at least ``min_agents`` distinct
agents carry them: a key nobody else shares cannot be coordination evidence,
and dropping singletons keeps the graph about ten times smaller.
Timestamps are epoch seconds (float) so range filters use the Post.ts index.
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict

import pandas as pd

from ..db import Graph, batched
from ..reuse import MinHasher, extract_channels, load_moltbook, near_clusters, post_features


def _kid(key: str) -> str:
    fam, _, val = key.partition(":")
    return fam + ":" + hashlib.sha1(val.encode("utf-8")).hexdigest()[:16] if fam == "payload" else key


def _mean_jaccard(texts: list[str]) -> float:
    if len(texts) < 2:
        return 1.0
    vals = [MinHasher.jaccard(texts[i], texts[j]) for i in range(len(texts)) for j in range(i + 1, len(texts))]
    return round(sum(vals) / len(vals), 4)


def build_rows(posts: pd.DataFrame, agents: pd.DataFrame, min_agents: int = 3) -> dict:
    """Pure-Python feature pass. Returns row lists ready for UNWIND."""
    key_agents: dict[str, set] = defaultdict(set)
    chan_agents: dict[str, set] = defaultdict(set)
    key_sample: dict[str, str] = {}
    post_rows, post_keys, post_chans, post_tmpl = [], [], [], []
    pid = posts["post_id"].to_numpy()
    aid = posts["agent_id"].to_numpy()
    ts = (posts["created_at"].astype("int64") / 1e9).to_numpy()
    for i, (p, a, t, title, content, venue) in enumerate(zip(
            pid, aid, ts, posts["title"].to_numpy(), posts["content"].to_numpy(), posts["venue"].to_numpy())):
        f = post_features(title, content)
        keys = [_kid(k) for k in f.keys]
        chans = sorted(extract_channels(f"{title or ''}\n{content or ''}"))
        for k, raw in zip(keys, f.keys):
            key_agents[k].add(a)
            if k not in key_sample:
                key_sample[k] = (f.template or raw)[:200]
        for c in chans:
            chan_agents[c].add(a)
        body = (content or "").strip() or (title or "")
        post_rows.append({"id": p, "agent": a, "ts": float(t), "venue": venue or "",
                          "text": body[:280], "injection": bool(f.injection)})
        post_keys.append(keys)
        post_chans.append(chans)
        post_tmpl.append(f.template)

    # Paraphrase families (near-duplicate wording), one "near:<n>" key each.
    cluster = near_clusters({t for t in post_tmpl if t})
    near_tmpls: dict[str, set] = defaultdict(set)
    for keys, t, a in zip(post_keys, post_tmpl, aid):
        c = cluster.get(t) if t else None
        if c is not None:
            k = f"near:{c}"
            keys.append(k)
            key_agents[k].add(a)
            near_tmpls[k].add(t)
            key_sample.setdefault(k, t[:200])

    keep_k = {k for k, s in key_agents.items() if len(s) >= min_agents}
    # Mean pairwise similarity per paraphrase family (sampled), so detection can
    # drop families that are LSH chaining noise rather than shared wording.
    near_sim = {k: _mean_jaccard(sorted(near_tmpls[k])[:30]) for k in keep_k if k.startswith("near:")}
    keep_c = {c for c, s in chan_agents.items() if len(s) >= 2}
    has_key = [{"p": p["id"], "k": k} for p, ks in zip(post_rows, post_keys) for k in ks if k in keep_k]
    links = [{"p": p["id"], "c": c} for p, cs in zip(post_rows, post_chans) for c in cs if c in keep_c]

    names = posts.drop_duplicates("agent_id").set_index("agent_id")["agent_name"].to_dict()
    meta = {}
    if agents is not None and len(agents):
        a = agents.drop_duplicates("agent_id").set_index("agent_id")
        meta = {i: (r["operator"], r["agent_created_at"]) for i, r in a.iterrows()}
    agent_rows = []
    for a in posts["agent_id"].unique():
        op, created = meta.get(a, (None, None))
        agent_rows.append({
            "id": a, "name": str(names.get(a) or ""),
            "owner": op if isinstance(op, str) and op else None,
            "created_ts": float(pd.Timestamp(created).timestamp()) if created is not None and not pd.isna(created) else None,
        })
    return {
        "agents": agent_rows,
        "posts": post_rows,
        "keys": [{"id": k, "family": k.split(":", 1)[0], "sample": key_sample[k],
                  "sim": near_sim.get(k, 1.0)} for k in keep_k],
        "channels": [{"id": c, "kind": c.split(":", 1)[0]} for c in keep_c],
        "has_key": has_key,
        "links": links,
    }


def load_graph(graph: Graph, rows: dict, dataset: str, log=print) -> dict:
    graph.delete()
    for label, prop in [("Agent", "id"), ("Owner", "handle"), ("Post", "id"), ("Post", "ts"),
                        ("Key", "id"), ("Channel", "id"), ("Swarm", "id")]:
        graph.write(f"CREATE INDEX FOR (n:{label}) ON (n.{prop})")
    graph.write("CREATE (:Meta {dataset: $d, ready: false})", {"d": dataset})

    steps = [
        ("agents", "UNWIND $rows AS r CREATE (:Agent {id: r.id, name: r.name, created_ts: r.created_ts})", rows["agents"]),
        ("owners", "UNWIND $rows AS r MATCH (a:Agent {id: r.id}) MERGE (o:Owner {handle: r.owner}) CREATE (a)-[:OWNED_BY]->(o)",
         [r for r in rows["agents"] if r["owner"]]),
        ("posts", "UNWIND $rows AS r MATCH (a:Agent {id: r.agent}) "
                  # toFloat: a whole-second timestamp would otherwise be stored as an
                  # integer and be skipped by float range scans on the ts index.
                  "CREATE (a)-[:POSTED]->(:Post {id: r.id, ts: toFloat(r.ts), venue: r.venue, text: r.text, injection: r.injection})",
         rows["posts"]),
        ("keys", "UNWIND $rows AS r CREATE (:Key {id: r.id, family: r.family, sample: r.sample, sim: r.sim})", rows["keys"]),
        ("channels", "UNWIND $rows AS r CREATE (:Channel {id: r.id, kind: r.kind})", rows["channels"]),
        ("has_key", "UNWIND $rows AS r MATCH (p:Post {id: r.p}), (k:Key {id: r.k}) CREATE (p)-[:HAS_KEY]->(k)", rows["has_key"]),
        ("links", "UNWIND $rows AS r MATCH (p:Post {id: r.p}), (c:Channel {id: r.c}) CREATE (p)-[:LINKS_TO]->(c)", rows["links"]),
    ]
    timing = {}
    for name, cypher, data in steps:
        t0 = time.perf_counter()
        for chunk in batched(data, 5000):
            graph.write(cypher, {"rows": chunk})
        timing[name] = round(time.perf_counter() - t0, 2)
        log(f"  {name}: {len(data)} rows in {timing[name]}s")

    # Derived inside the graph: which agents use which key, and how often.
    t0 = time.perf_counter()
    graph.write("MATCH (a:Agent)-[:POSTED]->(p:Post)-[:HAS_KEY]->(k:Key) "
                "WITH a, k, count(p) AS n CREATE (a)-[:USES_KEY {posts: n}]->(k)")
    graph.write("MATCH (k:Key)<-[:USES_KEY]-(a:Agent) WITH k, count(a) AS n SET k.n_agents = n")
    graph.write("MATCH (c:Channel)<-[:LINKS_TO]-(p:Post) WITH c, count(p) AS n SET c.n_posts = n")
    timing["derive"] = round(time.perf_counter() - t0, 2)

    counts = {
        "agents": len(rows["agents"]), "posts": len(rows["posts"]), "keys": len(rows["keys"]),
        "channels": len(rows["channels"]), "has_key": len(rows["has_key"]), "links": len(rows["links"]),
        "owners": len({r["owner"] for r in rows["agents"] if r["owner"]}),
    }
    graph.write("MATCH (m:Meta) SET m += $c, m.ready = true", {"c": counts})
    return {"counts": counts, "timing_s": timing}


def ingest(data_dir: str, graph_name: str, pattern: str = "*.parquet", log=print) -> dict:
    t0 = time.perf_counter()
    posts, agents = load_moltbook(data_dir, pattern)
    log(f"loaded {len(posts)} posts, {posts['agent_id'].nunique()} agents in {time.perf_counter() - t0:.1f}s")
    t1 = time.perf_counter()
    rows = build_rows(posts, agents)
    log(f"features in {time.perf_counter() - t1:.1f}s")
    out = load_graph(Graph(graph_name), rows, dataset=f"moltbook:{data_dir}:{pattern}", log=log)
    out["timing_s"]["features"] = round(time.perf_counter() - t1, 2)
    out["timing_s"]["total"] = round(time.perf_counter() - t0, 2)
    return out
