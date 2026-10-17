"""Reused code from the author's earlier Swarmchaser project, used as a library.

Disclosed reuse (allowed by the Graph Hacks rules for general libraries):
only the Moltbook Parquet loader and the per-post key extractors
(text template, payload, beneficiary) and the URL/handle channel extractor.
Everything that touches FalkorDB in this repository is new work.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SWARMCHASER = Path(os.environ.get("SWARMCHASER_PATH", Path(__file__).resolve().parents[2] / "swarmchaser"))
if str(SWARMCHASER) not in sys.path:
    sys.path.insert(0, str(SWARMCHASER))

from src.swarm.channels import extract_channels  # noqa: E402
from src.swarm.detector import SwarmDetector  # noqa: E402
from src.swarm.features import post_features  # noqa: E402
from src.swarm.loader import load_comments, load_moltbook  # noqa: E402
from src.swarm.minhash import MinHasher  # noqa: E402


def near_clusters(templates: set[str]) -> dict[str, int]:
    """Paraphrase families: MinHash LSH candidates verified by exact bigram Jaccard >= 0.3."""
    # Sorted list, not a set: set order depends on per-process string hashing,
    # which made the sampled LSH buckets (and so the swarm count) vary by run.
    return SwarmDetector()._near_clusters(sorted(templates))


__all__ = ["SWARMCHASER", "extract_channels", "post_features", "load_comments", "load_moltbook",
           "near_clusters", "MinHasher"]
