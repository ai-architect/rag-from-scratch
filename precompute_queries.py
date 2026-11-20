"""
precompute_queries.py - embed a fixed set of questions once, for offline dry runs.

rag_search_no_deps.py --dry-run uses these stored query vectors instead of
calling the embedding API, so people without an API key (or without pip) can
still run the full search. This script itself needs the SDK + API key once:

    python precompute_queries.py --index index_gemini.pkl
    python precompute_queries.py --index index_gemini.pkl --questions eval_questions.txt \
        --out index_gemini.queries.json

The questions are embedded with the index's provider/model and task="query",
exactly as a live search would do. The output is plain JSON (no pickle), so it
is safe to load and easy to inspect.
"""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

from embedders import embed_texts


def default_cache_path(index_path: Path) -> Path:
    """index_gemini.pkl -> index_gemini.queries.json (next to the index)."""
    return index_path.with_suffix(".queries.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute query embeddings for offline dry runs.")
    parser.add_argument("--index", type=Path, default=Path("index_gemini.pkl"))
    parser.add_argument("--questions", type=Path, default=Path("eval_questions.txt"),
                        help="one question per line")
    parser.add_argument("--out", type=Path, help="default: <index stem>.queries.json")
    args = parser.parse_args()

    with args.index.open("rb") as f:
        meta = pickle.load(f)["meta"]  # only load pickles you trust!

    questions = [line.strip() for line in args.questions.read_text(encoding="utf-8").splitlines()]
    questions = list(dict.fromkeys(q for q in questions if q))  # drop blanks/duplicates, keep order

    print(f"Embedding {len(questions)} questions with {meta['provider']} / {meta['model']} ...")
    vectors = embed_texts(questions, meta["provider"], meta["model"], task="query")
    for q, v in zip(questions, vectors):
        if len(v) != meta["dim"]:
            raise ValueError(f"{q!r}: got dim {len(v)}, index expects {meta['dim']}")

    cache = {
        "meta": {
            "provider": meta["provider"],
            "model": meta["model"],
            "dim": meta["dim"],
            "task": "query",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "queries": [{"text": q, "vector": v} for q, v in zip(questions, vectors)],
    }
    out = args.out or default_cache_path(args.index)
    out.write_text(json.dumps(cache), encoding="utf-8")
    print(f"Wrote {len(questions)} query vectors to {out} ({out.stat().st_size / 1e6:.1f} MB).")


if __name__ == "__main__":
    main()
