"""Track 1 investigator agent.

Two modes, same tools, same evidence contract:

- ``planner`` (default, no API key needed): a fixed investigation plan that
  picks the swarm, profiles it, checks owners, finds steering channels, then
  replays a block of each candidate channel and recommends the one with the
  best trade-off. Deterministic, so the demo and tests are reproducible.
- ``llm``: an OpenAI-compatible chat model chooses tools itself (function
  calling). Used when ``LLM_API_KEY`` is set. The model only sees tool
  results; it cannot run Cypher or write to the graph.

Every step records the tool, its arguments, its result and the exact Cypher
behind it. The final answer is assembled from those numbers and cites the
step each claim came from as [s<n>].
"""

from __future__ import annotations

import json
import os
import re
import time

import httpx

from . import tools as T

MIN_PRECISION = 0.5  # recommend a block only if at least half of what it blocks is swarm traffic


def _step(steps: list, tool: str, args: dict) -> dict:
    t0 = time.perf_counter()
    out = T.TOOLS[tool](**args)
    s = {"n": len(steps) + 1, "tool": tool, "args": args, "result": out["result"],
         "evidence": out["evidence"], "ms": round((time.perf_counter() - t0) * 1000, 1)}
    steps.append(s)
    return s


def _pick_swarm(question: str, swarms: list[dict]) -> dict | None:
    q = question.lower()
    m = re.search(r"swarm\s*#?\s*(\d{3,})", q)
    if m:
        sid = int(m.group(1))
        return next((s for s in swarms if s["swarm"] == sid), {"swarm": sid})
    for word, kind in [("mint", "payload campaign"), ("token", "payload campaign"), ("payload", "payload campaign"),
                       ("recruit", "agent-recruitment"), ("injection", "agent-recruitment")]:
        if word in q:
            hit = next((s for s in swarms if s["kind"] == kind), None)
            if hit:
                return hit
    words = set(re.findall(r"[a-z0-9.\-]{4,}", q))
    if "skill" in q or "skills" in q:
        words.add("skill")
    for word in sorted(words, key=len, reverse=True):
        word = word.strip(".-")
        if word in {"what", "should", "block", "swarm", "steering", "which", "there", "link", "behind"}:
            continue
        hit = next((s for s in swarms if any(word in c for c in (s.get("channels") or []))), None)
        if hit:
            return hit
    if any(w in q for w in ("steer", "block", "skill", "link", "channel")):
        hit = next((s for s in swarms if s.get("top_channel")), None)
        if hit:
            return hit
    return swarms[0] if swarms else None


def _hours(ts0, ts1) -> float:
    return round((ts1 - ts0) / 3600, 1) if ts0 and ts1 else 0.0


def run_planner(question: str, graph: str | None = None, hours: float = 6.0) -> dict:
    steps: list[dict] = []
    t0 = time.perf_counter()
    sw = _step(steps, "list_swarms", {"limit": 200, "graph": graph})
    target = _pick_swarm(question, sw["result"])
    if not target:
        return {"question": question, "answer": "No swarms found in this graph.", "steps": steps}
    sid = target["swarm"]
    prof = _step(steps, "swarm_profile", {"swarm": sid, "graph": graph})
    if "error" in prof["result"]:
        return {"question": question, "answer": prof["result"]["error"], "steps": steps}
    own = _step(steps, "owner_spread", {"swarm": sid, "graph": graph})
    ch = _step(steps, "steering_channels", {"swarm": sid, "graph": graph})
    p = prof["result"]
    lines = [
        f"Swarm {sid} is a {p['kind']}: {p['agents']} agents posted {p['posts']} matching posts over "
        f"{_hours(p['first_ts'], p['last_ts'])} hours [s{prof['n']}].",
    ]
    known = [r for r in own["result"] if r["owner"] != "(unknown)"]
    top = known[0] if known else None
    if p["owners_known"]:
        lines.append(
            f"It is not one operator: {p['owners']} different owners among {p['owners_known']} agents with a known owner; "
            f"the largest owner runs {top['agents'] if top else 0} agent(s) [s{own['n']}]."
            if (p["top_owner_share"] or 0) < 0.5 else
            f"One owner, {p['top_owner']}, runs {round(100 * p['top_owner_share'])}% of its agents [s{own['n']}].")
    channels = ch["result"]
    rec, impacts = None, []
    if not channels:
        lines.append(f"No outside channel is significantly over-represented in its posts, so there is no single link to block [s{ch['n']}].")
    else:
        names = ", ".join(f"{c['channel']} ({round(100 * c['share_of_swarm_posts'])}% of posts, {c['owners']} owners)"
                          for c in channels[:3])
        lines.append(f"Steering channels: {names} [s{ch['n']}].")
        seen = set()
        for c in channels[:3]:
            if c["channel"] in seen:
                continue
            seen.add(c["channel"])
            st = _step(steps, "block_impact", {"channel": c["channel"], "hours_after_first_seen": hours,
                                               "swarm": sid, "graph": graph})
            r = st["result"]
            if "error" in r:
                continue
            prec = r["blocked_swarm_posts"] / r["blocked_posts"] if r["blocked_posts"] else 0.0
            impacts.append((st, r, prec))
        good = [x for x in impacts if x[1]["swarm_posts_prevented"] > 0 and x[2] >= MIN_PRECISION]
        good.sort(key=lambda x: (x[1]["swarm_posts_prevented"], -x[1]["collateral_posts"]), reverse=True)
        for st, r, prec in impacts:
            lines.append(
                f"Blocking {r['channel']} {hours:g} h after the swarm's first post would have stopped "
                f"{r['swarm_posts_prevented']} of {r['swarm_posts_before']} swarm posts and cut off {r['swarm_agents_cut_off']} "
                f"of {r['swarm_agents_before']} swarm agents, while also blocking {r['collateral_posts']} posts outside any swarm "
                f"({round(100 * prec)}% of blocked posts were swarm traffic) [s{st['n']}].")
        if good:
            st, r, prec = good[0]
            rec = {"action": "block", "channel": r["channel"], "at_hours_after_swarm_start": hours, "step": st["n"],
                   "swarm_posts_prevented": r["swarm_posts_prevented"], "collateral_posts": r["collateral_posts"]}
            lines.append(f"Recommendation: block {r['channel']}. It prevents the most swarm posts with at least "
                         f"{round(100 * MIN_PRECISION)}% precision [s{st['n']}].")
        elif impacts:
            rec = {"action": "monitor", "reason": "no block reaches the precision bar or prevents swarm posts"}
            lines.append("Recommendation: do not block yet. No candidate link stops swarm posts without mostly hitting "
                         "ordinary posts at this point in time; keep watching and re-run detection.")
    lines.append("Limit: block replays measure direct reach and assume agents do not switch links.")
    return {"question": question, "mode": "planner", "swarm": sid, "answer": " ".join(lines),
            "recommendation": rec, "steps": steps, "ms": round((time.perf_counter() - t0) * 1000, 1)}


# ---------------------------------------------------------------- LLM mode
_SCHEMAS = {
    "list_swarms": {"limit": "integer", "kind": "string"},
    "swarm_profile": {"swarm": "integer"},
    "steering_channels": {"swarm": "integer"},
    "channel_reach": {"channel": "string"},
    "owner_spread": {"swarm": "integer"},
    "operation_of": {"swarm": "integer"},
    "channel_pagerank": {"limit": "integer"},
    "block_impact": {"channel": "string", "hours_after_first_seen": "number", "swarm": "integer"},
}
_REQUIRED = {"swarm_profile": ["swarm"], "steering_channels": ["swarm"], "channel_reach": ["channel"],
             "owner_spread": ["swarm"], "operation_of": ["swarm"], "block_impact": ["channel"]}

SYSTEM = (
    "You investigate coordinated AI-agent swarms using graph tools over a FalkorDB graph of real Moltbook posts. "
    "Use tools to gather evidence; never invent numbers. Before recommending any block, call block_impact for it and "
    "weigh swarm posts prevented against collateral posts. In the final answer cite the step of every number as [s<n>], "
    "and state the limit that block replays assume agents do not adapt."
)


def _tool_specs():
    out = []
    for name, props in _SCHEMAS.items():
        out.append({"type": "function", "function": {
            "name": name, "description": (T.TOOLS[name].__doc__ or "").strip(),
            "parameters": {"type": "object", "properties": {k: {"type": v} for k, v in props.items()},
                           "required": _REQUIRED.get(name, [])}}})
    return out


def run_llm(question: str, graph: str | None = None, max_steps: int = 8) -> dict:
    key = os.environ.get("LLM_API_KEY")
    base = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    steps: list[dict] = []
    t0 = time.perf_counter()
    with httpx.Client(timeout=60) as http:
        for _ in range(max_steps + 1):
            r = http.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {key}"},
                          json={"model": model, "messages": msgs, "tools": _tool_specs(), "temperature": 0})
            r.raise_for_status()
            m = r.json()["choices"][0]["message"]
            msgs.append(m)
            calls = m.get("tool_calls") or []
            if not calls or len(steps) >= max_steps:
                return {"question": question, "mode": "llm", "model": model, "answer": m.get("content") or "",
                        "steps": steps, "ms": round((time.perf_counter() - t0) * 1000, 1)}
            for c in calls:
                name = c["function"]["name"]
                args = json.loads(c["function"].get("arguments") or "{}")
                args = {k: v for k, v in args.items() if k in _SCHEMAS.get(name, {})}
                if name not in T.TOOLS:
                    content = json.dumps({"error": f"unknown tool {name}"})
                else:
                    st = _step(steps, name, {**args, "graph": graph})
                    content = json.dumps({"step": st["n"], "result": st["result"]}, default=str)[:12000]
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": content})
    return {"question": question, "mode": "llm", "answer": "step limit reached", "steps": steps}


def investigate(question: str, graph: str | None = None, mode: str | None = None) -> dict:
    mode = mode or ("llm" if os.environ.get("LLM_API_KEY") else "planner")
    return run_llm(question, graph) if mode == "llm" else run_planner(question, graph)
