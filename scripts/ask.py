"""Ask the Track 1 investigator questions and save full answers with evidence."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from provenant.moltbook.agent import investigate  # noqa: E402

QUESTIONS = sys.argv[1:] or [
    "What is steering the skill-file swarm, and what should we block?",
    "Who is behind the token mint campaign, and is there a link to block?",
    "Should we block the bags.fm identity verification wave?",
]
out = []
for q in QUESTIONS:
    r = investigate(q)
    out.append(r)
    print("Q:", q)
    print("A:", r["answer"])
    print("steps:", [(s["n"], s["tool"], s["ms"]) for s in r["steps"]], "total ms", r.get("ms"))
    print("queries:", sum(len(s["evidence"]) for s in r["steps"]))
    print()
Path("reports/agent_answers.json").write_text(json.dumps(out, indent=1, default=str))
