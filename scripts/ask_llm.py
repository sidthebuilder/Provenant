"""Run the Track 1 agent in LLM mode (model chooses the tools) and save transcripts.

Needs LLM_API_KEY, LLM_BASE_URL and LLM_MODEL in .env. Works with any
OpenAI-compatible endpoint, including a local Ollama server:
    LLM_BASE_URL=http://localhost:11434/v1   LLM_MODEL=qwen2.5:7b   LLM_API_KEY=ollama
Writes reports/agent_answers_llm.json.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from provenant.moltbook.agent import run_llm  # noqa: E402,F401  (imports load .env via db)
from provenant import db  # noqa: E402,F401

if not os.environ.get("LLM_API_KEY"):
    sys.exit("Set LLM_API_KEY (and LLM_BASE_URL, LLM_MODEL) in .env first.")

QUESTIONS = sys.argv[1:] or [
    "What is steering the skill-file swarm on clawtasks.com, and should we block it? Check the collateral first.",
    "Who is behind the largest token mint campaign? Is it one operator, and is there a link worth blocking?",
    "Is iili.io a good link to block? Use PageRank and a block replay before deciding.",
]
out = []
for q in QUESTIONS:
    r = run_llm(q)
    out.append(r)
    print("Q:", q)
    print("model:", r.get("model"), "| tool calls:", [(s["n"], s["tool"], s["args"]) for s in r["steps"]],
          "| ms:", r.get("ms"))
    print("A:", r["answer"])
    print()
Path("reports").mkdir(exist_ok=True)
Path("reports/agent_answers_llm.json").write_text(json.dumps(out, indent=1, default=str))
print("saved reports/agent_answers_llm.json")
