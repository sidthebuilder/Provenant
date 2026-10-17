# Provenant

**Agents that work from a FalkorDB graph of real agent data, and show the graph evidence behind every answer.**

Built for [Graph Hacks](https://www.wemakedevs.org/hackathons/falkordb) (WeMakeDevs × FalkorDB, 15 to 18 October 2026). One FalkorDB instance, two real datasets, one entry per track:

| Track | What it does | Data |
|---|---|---|
| **1. Agents That Act on Connected Data** | Finds coordinated swarms of AI agents, then replays a block of their steering link on a forked graph before recommending it | Moltbook Observatory Archive (public, MIT) |
| **2. Agent Memory and Coordination** | Shared, time-aware memory for a group of agents, scored on 1,020 real moments when agents had to search their own history | AI Village dataset (AI Digest) |
| **3. Company Brain** | Who owns what, who knows what, what was decided under which goal, with every answer linked to its source messages | AI Village dataset (AI Digest) |

Everything runs locally: FalkorDB in Docker, a FastAPI backend, a single-page UI, and an MCP server exposing the same tools.

---

## Track 1: replay a block before you make it

### The problem

A trust-and-safety responder looking at hundreds of agents posting the same message has two questions: *what is steering them?* and *what happens if I block that?* Detection tools answer the first. The second is usually guessed, and a link a swarm uses heavily can also be a link ordinary users post every day.

Provenant answers both from the graph. For each candidate block it copies the graph with `GRAPH.COPY`, deletes every post on the copy that links the channel from a chosen time, recounts the swarm, and counts the ordinary posts the block would also hit. The real graph is never written.

### Results on real data (Moltbook, 27 Jan to 2 Feb 2026)

Numbers come from `reports/pipeline.json`, `reports/detect_moltbook_w1.json` and `reports/block_impact_moltbook_w1_6h.json`, produced by `scripts/pipeline.py`.

| | |
|---|---|
| Posts / agents / owners in the graph | 125,885 / 26,303 / 18,818 |
| Swarms found in the graph | 117, identical across repeated runs (the earlier pandas pipeline in Swarmchaser found 119) |
| Largest swarm | token-mint campaign: 442 agents, 5,942 posts, 278 different owners |
| Detection time in FalkorDB | about 7 s |
| One block replay, including the fork | about 1 to 2 s (up to about 15 s for very widely used @handles) |

Block replays, starting 6 hours after the swarm's first post:

| Channel | Swarm posts stopped | Swarm agents cut off | Ordinary posts also hit |
|---|---|---|---|
| `clawtasks.com/skill.md` (skill file) | 172 of 207 | 136 of 171 | 112 |
| `clawtasks.com` (whole domain) | 172 of 207 | 136 of 171 | 125 |
| `bags.fm/<token page>` | 156 of 510 | 7 of 283 | 0 |
| `chatr.ai/skills.md` | 29 of 31 | 24 of 26 | 98 |
| `iili.io` (image host) | 22 | 17 | 624 |

What the replays show:

- **The narrowest target wins.** Blocking the skill file stops as much swarm traffic as blocking the whole domain and hits fewer ordinary posts. The agent recommends the skill file.
- **Centrality alone misleads.** FalkorDB PageRank over the steering edges ranks `iili.io` as the most central hub (it steers 4 swarms). Blocking it would hit 624 ordinary posts to stop 22 swarm posts, so the agent does not recommend it.
- **A strong match is not always a good block.** `chatr.ai/skills.md` appears in 97% of its swarm's posts, but blocking it would hit 98 ordinary posts to stop 29 swarm posts. The agent recommends monitoring instead.
- **No single operator.** The mint campaign's 442 agents belong to 278 different owners, and no outside link is significantly over-represented in it. The agent reports that there is no single link to block, rather than inventing one.

### Graph model (graph `moltbook_w1`)

```
(:Agent {id, name, created_ts})-[:OWNED_BY]->(:Owner {handle})
(:Agent)-[:POSTED]->(:Post {id, ts, venue, text, injection})
(:Post)-[:HAS_KEY]->(:Key {id, family, sample, sim, n_agents})   template, paraphrase family, payload, beneficiary
(:Post)-[:LINKS_TO]->(:Channel {id, kind, n_posts})              domain, skill-file path, @handle
(:Agent)-[:USES_KEY {posts}]->(:Key)                             derived in Cypher
(:Key)-[:OVERLAP {shared}]->(:Key)                               derived in Cypher
(:Swarm {id, kind, n_agents, n_posts, first_ts, last_ts, distinct_owners, operation, ...})
(:Swarm)-[:DEFINED_BY]->(:Key)
(:Swarm)-[:INCLUDES]->(:Agent)
(:Swarm)-[:STEERED_BY {share, enrichment, p_adj, agents, owners}]->(:Channel)
(:Meta {dataset, ready, counts...})
```

Indexes on `Agent.id`, `Owner.handle`, `Post.id`, `Post.ts`, `Key.id`, `Channel.id` and `Swarm.id`.

### How the graph does the work

1. **Detection** (`provenant/moltbook/detect.py`). Keys shared by at least 3 agents and at most 5% of all agents become candidates. Two keys get an `OVERLAP` edge when at least 3 agents use both and they share at least half of the smaller key's agents, so one busy agent cannot glue unrelated campaigns together. `algo.WCC` over `OVERLAP` gives campaigns, which become `Swarm` nodes. Campaigns held together only by paraphrase must also read alike (mean word-bigram Jaccard at least 0.3).
2. **Steering.** Cypher counts how many swarm posts carry each channel against the rest of the corpus. A one-sided Fisher exact test with Bonferroni correction decides (SciPy, on counts from the graph), and a channel only counts if agents with at least 2 different owners carried it.
3. **Operations.** `algo.WCC` over `STEERED_BY` groups swarms that share steering infrastructure. `algo.pageRank` ranks the channels.
4. **Block replay** (`provenant/moltbook/impact.py`). `GRAPH.COPY`, wait for the copy's indexes to be ready, delete the blocked posts on the copy, verify none survived, recount, delete the copy.

### Key Cypher queries

Overlap edges between keys (detection):

```cypher
MATCH (k1:Candidate)<-[:USES_KEY]-(a:Agent)-[:USES_KEY]->(k2:Candidate)
WHERE id(k1) < id(k2)
WITH k1, k2, count(a) AS shared
WHERE shared >= $min AND shared >= $ov * CASE WHEN k1.n_agents < k2.n_agents THEN k1.n_agents ELSE k2.n_agents END
CREATE (k1)-[:OVERLAP {shared: shared}]->(k2)
```

Campaigns, then operations:

```cypher
CALL algo.WCC({nodeLabels: ['Candidate'], relationshipTypes: ['OVERLAP']}) YIELD node, componentId
CALL algo.WCC({nodeLabels: ['Swarm', 'Channel'], relationshipTypes: ['STEERED_BY']}) YIELD node, componentId
```

Counts for the steering test:

```cypher
MATCH (s:Swarm)-[:DEFINED_BY]->(:Key)<-[:HAS_KEY]-(p:Post)<-[:POSTED]-(a:Agent)<-[:INCLUDES]-(s)
WITH DISTINCT s, p, a
MATCH (p)-[:LINKS_TO]->(c:Channel)
OPTIONAL MATCH (a)-[:OWNED_BY]->(o:Owner)
RETURN s.id, s.n_posts, c.id, c.n_posts, count(DISTINCT p), count(DISTINCT a), count(DISTINCT o)
```

Channel centrality:

```cypher
CALL algo.pageRank(null, 'STEERED_BY') YIELD node, score
WITH node, score WHERE node:Channel
MATCH (node)<-[r:STEERED_BY]-(s:Swarm)
RETURN node.id, score, count(s), sum(s.n_agents) ORDER BY score DESC
```

Block replay, on the fork only:

```
GRAPH.COPY moltbook_w1 moltbook_w1__whatif_<hash>
```
```cypher
MATCH (c:Channel {id: $c})<-[:LINKS_TO]-(p:Post) WHERE p.ts + 0.0 >= $t
WITH collect(DISTINCT p) AS ps UNWIND ps AS p DETACH DELETE p
```

### The agent and its tools

Eight tools in `provenant/moltbook/tools.py`: `list_swarms`, `swarm_profile`, `owner_spread`, `steering_channels`, `channel_reach`, `operation_of`, `channel_pagerank`, `block_impact`.

Every tool returns its result plus the exact Cypher, parameters, row count and time of each query it ran. Reads go through `GRAPH.RO_QUERY` with a timeout, so a read tool cannot write. `block_impact` writes only to a throwaway copy.

The agent (`provenant/moltbook/agent.py`) has two modes:

- **planner** (default, also when an API key is set): a fixed investigation plan. It picks the swarm, profiles it, checks owners, finds steering channels, replays a block of each, and recommends one only if it prevents swarm posts and at least half of what it blocks is swarm traffic. It is deterministic, so the demo and tests reproduce. This is the mode that was tested.
- **llm**: a model chooses the tools itself by function calling, through any OpenAI-compatible endpoint (`LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`). It sees only tool results and cannot run Cypher. Run it with `scripts/ask_llm.py`, `PROVENANT_AGENT_MODE=llm`, or `{"mode": "llm"}` on `/api/ask`.

**LLM mode, tested with a small local model.** On 18 October I ran `scripts/ask_llm.py` with `qwen2.5:7b` through Ollama on a laptop RTX 5050 (8 GB). Transcripts are in `reports/agent_answers_llm.json`.

| Question | Tools the model chose | Verdict | Problems |
|---|---|---|---|
| What steers the clawtasks skill-file swarm, should we block it? | `list_swarms(search)` → `steering_channels` → `block_impact` | Block `clawtasks.com` (58% of blocked posts are swarm posts) | Did not try the narrower `skill.md`, which hits fewer ordinary posts; restated one figure inconsistently (167 vs 172) |
| Who is behind the token-mint campaign, is there a link to block? | `list_swarms` → `swarm_profile` → `steering_channels` → `owner_spread` → `operation_of` → `channel_reach` | Correct: 442 agents, 278 owners, no single operator, no significant link, no block | One unneeded `channel_reach` call |
| Is iili.io a good link to block? | `channel_pagerank` → `block_impact` | Correct: do not block, most blocked posts are ordinary | Timed the replay from the start of an unrelated swarm |

All three verdicts were right, but the 7B model never wrote `[s<n>]` citations, and its reasoning was less careful than the planner's. On the first run, before `list_swarms` gained a `search` argument and errors that point back to it, the model invented swarm ids and two of three answers failed. That change is what made it work. A larger model would likely do better; none was tested.

Every number in an answer cites its step as `[s<n>]`, and the UI links each citation to the queries behind it.

---

## Track 2: shared memory, tested on real recall failures

**Data:** the [AI Village](https://theaidigest.org/village) export by AI Digest. About 50 frontier-model agents have worked together there since April 2025. `scripts/ingest_village.py` loads it into graph `village` in about 50 s: 182,159 agent messages, 159,898 agent-to-agent mentions, 1,152 sites and repos, 51 village goals and 11,246 history searches. **Only messages written by agents are loaded.** The 10,008 messages from human viewers are skipped entirely, and a test checks that every message in the graph has an agent sender.

**The test set is real.** When an agent's own memory did not hold what it needed, it called a `search_history` tool. Ingest matches the messages each answer cited back to the chat log by speaker and timestamp. That gives 1,020 searches and 4,354 messages the agent actually needed. A test checks that every citation points to a message written before its search.

**Finding:** 73% of the messages an agent needed were written by *another* agent. Memory kept separately per agent could have held at most about a quarter of what agents went on to search for, so memory has to be shared.

**Benchmark** (`scripts/bench_memory.py`, results in `reports/bench_memory.json`). Each retriever answers as of the search time, using only messages written before it. Hybrid weights were chosen on the earlier half of searches (510, dev) from a fixed grid; the table reports the later half (510, test), which the grid search never saw.

| Retriever | recall@10 | recall@20 | recall@50 | searches with ≥1 hit in top 20 | median ms |
|---|---|---|---|---|---|
| Latest messages (context-window baseline) | 31.9% | 41.7% | 53.9% | 51.0% | 3 |
| Full-text only | 5.2% | 8.4% | 14.6% | 17.1% | 19 |
| Graph context only | 14.8% | 21.3% | 32.8% | 32.8% | 30 |
| **Hybrid: recency + graph** | **44.2%** | **52.3%** | **63.4%** | **65.1%** | 46 |

Weights chosen on dev: recency half-life 1 hour, named-agent bonus 0.5, addressed-to-asker bonus 0.25, same-room bonus 0.6, full-text weight 0.

The hybrid ranks recent messages and adds graph evidence: messages by or mentioning the agents named in the question, messages addressed to the asker, and the room the asker last spoke in. At 20 results it recovers about a quarter more of the needed messages than the recency baseline (52.3% against 41.7%), at about 46 ms per lookup. Keyword matching was nearly useless here, and the grid search gave it zero weight. Agents' questions are about people, rooms and recent events, which are relations in the graph rather than shared words.

The UI replays any of these searches: what the graph retrieved, which of those the real search cited, and what was missed.

### Graph model (graph `village`)

```
(:Agent {id, name, model, slug, participating})
(:Agent)-[:HAS_GOAL]->(:AgentGoal {name, short_name, description, start_ts, end_ts})
(:Goal {id, text, start_ts, end_ts})                         village-wide goals over time
(:Room {id, name})
(:Agent)-[:SENT]->(:Message {id, ts, text, decision, handoff})-[:IN]->(:Room)
(:Message)-[:MENTIONS]->(:Agent)
(:Message)-[:LINKS]->(:Site {id, host, kind})                 repo, Pages site or domain
(:Message)-[:DURING]->(:Goal)
(:Agent)-[:SEARCHED]->(:Search {id, ts, query, lo_ts, hi_ts, answer_head, n_cited})
(:Search)-[:CITED]->(:Message)                               benchmark ground truth
```

Full-text indexes on `Message.text` and `Search.query`.

### Key Cypher queries

Graph context for a recall, as of time `$t` only:

```cypher
MATCH (a:Agent) WHERE a.name IN $names
MATCH (a)-[:SENT]->(m:Message)-[:IN]->(r:Room) WHERE m.ts >= $lo AND m.ts < $t
RETURN m.id, m.ts, r.id, 'named'
UNION
MATCH (a:Agent) WHERE a.name IN $names
MATCH (r:Room)<-[:IN]-(m:Message)-[:MENTIONS]->(a) WHERE m.ts >= $lo AND m.ts < $t
RETURN m.id, m.ts, r.id, 'named'
UNION
MATCH (me:Agent {id: $me})<-[:MENTIONS]-(m:Message)-[:IN]->(r:Room) WHERE m.ts >= $lo AND m.ts < $t
RETURN m.id, m.ts, r.id, 'to_asker'
```

Full-text candidates:

```cypher
CALL db.idx.fulltext.queryNodes('Message', $q) YIELD node, score
WHERE node.ts >= $lo AND node.ts < $t
RETURN node.id, node.ts, score ORDER BY score DESC LIMIT 300
```

---

## Track 3: a company brain for the AI Village

The village works like a company: agents with roles (individual goals), shared goals that change every week or so, rooms, and real sites and repos they built. `provenant/village/brain.py` answers company questions with graph queries, and every answer links to the messages behind it. Most tools also accept `until_ts`, to answer as of an earlier date.

| Question | Tool | Example on real data |
|---|---|---|
| Who owns this site? | `who_owns` | `grok-ai-village-news-496089.gitlab.io` was introduced by Grok 4.5, which went on to link it in 1,964 messages |
| What did this agent build? | `sites_of` | Claude Opus 4.5 introduced `ai-village-agents.github.io/edge-garden` (185 links) and `opus-breaking-news` |
| Who knows about this? | `who_knows` | Juice Shop / OWASP: GPT-5.2, Claude Haiku 4.5, Claude 3.7 Sonnet |
| What was decided, under which goal? | `decisions` | choosing a charity in April 2025, with the goal in force at the time |
| Who worked on this goal, and who talked to whom? | `goal_team` | charity goal: GPT-4.1 was most active (1,608 messages), most often addressing o3 |
| Who asks this agent for help? | `handoffs` | GPT-5.2 is asked most by DeepSeek-V3.2 (251 requests) |
| Show me the source | `message` | opens any cited message with sender, room and goal |

All figures in this table were re-measured on the final graph (agent messages only) with `scripts/smoke_brain.py`; `who_owns` and `sites_of` are also checked by the tests.

"Ownership" means who first linked a site in chat and who links it most. Repository permissions are not in the data, and `who_owns` says so in its answer. "Decision" and "handoff" messages are found with a keyword rule (for example "we decided", "can you"), so they are leads to read, not a complete list.

Example query (`who_owns`, earliest introducer):

```cypher
MATCH (s:Site)<-[:LINKS]-(m:Message)<-[:SENT]-(a:Agent)
WHERE s.id IN $ids AND m.ts <= $t
RETURN s.id, a.name, m.id, m.ts, left(m.text, 200) ORDER BY m.ts LIMIT 1
```

---

## Run it

Needs Python 3.11 and Docker. Commands are for Windows PowerShell; on macOS or Linux use `source .venv/bin/activate`.

```powershell
docker run -d --name falkordb --restart unless-stopped -p 6379:6379 -p 3000:3000 falkordb/falkordb:latest
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # adjust paths if needed

# Track 1 data: Moltbook Observatory Archive (MIT), downloaded by the Swarmchaser script
git clone https://github.com/sidthebuilder/swarmchaser ..\swarmchaser
python ..\swarmchaser\scripts\fetch_moltbook.py --out ..\swarmchaser\data\moltbook

python scripts\pipeline.py              # ingest, detect, rank every block, agent answers (about 4 min)
python scripts\ask.py "What is steering the skill-file swarm, and what should we block?"

# Tracks 2 and 3 data: request access to the AI Village dataset on Hugging Face
# (aidigestorg/ai-village), download the *.jsonl.gz tables (not the images) into ..\data, then:
python scripts\ingest_village.py        # about 1 min
python scripts\bench_memory.py          # about 3 min

uvicorn provenant.api:app --port 8000   # UI at http://localhost:8000
python -m provenant.mcp_server          # the same tools over MCP (stdio)
python -m pytest tests -q
```

FalkorDB Browser at http://localhost:3000 shows the same graphs.

There is no live deployment. The AI Village data cannot be redistributed, and the API has no authentication, so the project is run locally and shown in the demo video.

## Checks

All 16 tests pass (`python -m pytest tests -q`, about 13 s), and `scripts/check_mcp.py` lists 16 MCP tools and calls two of them successfully.

- `tests/test_track1.py` builds a small graph with hand-computed answers and checks detection, owner attribution, steering and block-replay counts exactly. It also checks that the real graph is untouched by a replay and that the fork is deleted, that reads cannot write, that the real graph reproduces the mint campaign (442 agents, 5,942 posts, 278 owners), and runs the API end to end.
- `tests/test_village.py` checks the site parser and keyword extraction, that recall never returns a message from after the search, that citations point to earlier messages, that the brain tools return sources, that no human message is in the graph, and the village API.
- `scripts/check_ui.py` calls every endpoint the UI uses through the real app. `scripts/check_mcp.py` lists the MCP tools and calls two of them.
- Two full runs from a fresh ingest gave identical Track 1 counts, after fixing a set-ordering source of non-determinism in paraphrase clustering.
- In-graph detection was compared against the earlier pandas pipeline on the same data: 117 swarms against 119, with the largest swarm identical.

Bugs found and fixed during the event, because they changed results:

- Querying a `GRAPH.COPY` fork before its indexes finished rebuilding silently missed rows, so early block replays undercounted. Replays now wait for the indexes and verify the deletion.
- Whole-second timestamps were stored as integers and skipped by float range scans. They are now stored as floats.
- Posts belonging to two swarms were counted twice in block totals. Totals now count unique posts and agents.

## Limits

- Replays measure direct reach. They assume swarm agents do not switch to another link after a block, so "swarm posts stopped" is an upper bound.
- "Ordinary posts hit" are posts in no detected swarm. Some may be coordination the detector missed.
- Detection misses swarms whose text is fully rewritten and swarms coordinated only by timing.
- The Track 2 benchmark measures retrieval of messages a real search cited. It does not measure whether an agent's later actions improved.
- The API has no authentication, only a simple per-endpoint throttle. Run it behind a proxy with rate limiting before exposing it.
- Reports name public Moltbook agent accounts and owner handles. Treat them as leads, not proof. Some swarm posts contain offensive text, quoted as written.

## Disclosures

**Reused code.** The Moltbook loader, per-post key extractors and MinHash paraphrase clustering are reused as a library from the author's earlier project [Swarmchaser](https://github.com/sidthebuilder/swarmchaser) (see `provenant/reuse.py`). Everything that touches FalkorDB is new work started during the event: graph ingestion, in-graph detection, the steering test on graph counts, block replay on forks, the tools, the MCP server, the agent, the AI Village graph, memory, benchmark and company-brain tools, the API, the UI and the tests.

**AI coding assistant.** AI coding assistant. I used Kiro as a coding assistant during development, while the dataset selection, graph design, and architecture were mine. All code and benchmarks are tested and verified by me. 

**Data.** Moltbook is public under the MIT licence. The AI Village dataset is released by AI Digest for research and analysis behind a free access request on Hugging Face. It is used here under those terms: analysis only, no model training, no re-identification, only agent messages loaded, nothing redistributed.

## Licence and citation

Code: Apache-2.0 (see `LICENSE`).

- Gautam, S. and Riegler, M. A. (2026). *Moltbook Observatory Archive*. Hugging Face Datasets. MIT licence.
- AI Digest (2026). *AI Village dataset*. https://theaidigest.org/village
