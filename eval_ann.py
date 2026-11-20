"""
eval_ann.py - measure the speed/recall trade-off of IVF and LSH against exhaustive search.

For a set of test queries it runs the exhaustive search (the ground truth) and
the approximate search with several settings, then reports:

    recall@k     share of the true top-k chunks the approximate search found
    candidates   share of all chunks that had to be scored exactly
    ms/query     pure-math search time (embedding not included)

Test queries (no API calls by default):
    Each query is a randomly chosen chunk vector plus random noise, normalized.
    The noise moves it away from its chunk, like a real question that is
    phrased differently from the passage that answers it. The program prints
    the average cosine between query and best chunk, next to the cosine of
    two random chunks, so you can see how hard the queries are. For Gemini,
    --noise 0.5 gives ~0.89 (random pairs: ~0.70), while --noise 1.0 already
    makes queries as unrelated as random chunks. Use --questions
    eval_questions.txt to embed real questions instead (one batched API call).

Usage:
    python eval_ann.py --index index_gemini.pkl --ann index_gemini.ivf.pkl index_gemini.lsh.pkl
    python eval_ann.py --index index_gemini.pkl --ann index_gemini.ivf.pkl --questions eval_questions.txt
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

from ann_index import ann_search, describe, load_ann
from embedders import embed_texts
from rag_search import dot, exhaustive_search, load_index, normalize


def synthetic_queries(unit_vectors, count: int, noise: float, seed: int):
    """Chunk vectors pushed in a random direction: normalize(v + noise * g), with ||g|| ~ 1."""
    rng = random.Random(seed)
    d = len(unit_vectors[0])
    sigma = noise / math.sqrt(d)  # each of d components ~ N(0, sigma^2) -> ||g|| ~ noise
    queries = []
    for _ in range(count):
        v = unit_vectors[rng.randrange(len(unit_vectors))]
        queries.append(normalize([x + rng.gauss(0.0, sigma) for x in v]))
    return queries


def evaluate(queries, truth, unit_vectors, ann, top_k, **params):
    recall_sum, cand_sum, t_sum = 0.0, 0, 0.0
    for q, exact in zip(queries, truth):
        t = time.perf_counter()
        hits, stats = ann_search(ann, q, unit_vectors, top_k, **params)
        t_sum += time.perf_counter() - t
        found = {i for _, i in hits}
        recall_sum += sum(1 for _, i in exact if i in found) / len(exact)
        cand_sum += stats["candidates"]
    n = len(queries)
    return recall_sum / n, cand_sum / n / len(unit_vectors), 1000 * t_sum / n


def main() -> None:
    parser = argparse.ArgumentParser(description="Recall vs. speed of IVF/LSH compared with exhaustive search.")
    parser.add_argument("--index", type=Path, default=Path("index.pkl"))
    parser.add_argument("--ann", type=Path, nargs="+", required=True, help="one or more files from build_ann.py")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--queries", type=int, default=50, help="number of synthetic queries")
    parser.add_argument("--noise", type=float, default=0.5, help="how far synthetic queries are from their chunk")
    parser.add_argument("--questions", type=Path, help="text file, one real question per line (uses the API)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    meta, chunks, unit_vectors = load_index(args.index)
    if args.questions:
        lines = [l.strip() for l in args.questions.read_text(encoding="utf-8").splitlines() if l.strip()]
        vectors = embed_texts(lines, meta["provider"], meta["model"], task="query")
        queries, source = [normalize(v) for v in vectors], f"{len(lines)} questions from {args.questions}"
    else:
        queries = synthetic_queries(unit_vectors, args.queries, args.noise, args.seed)
        source = f"{len(queries)} synthetic queries (noise={args.noise})"
    print(f"{len(chunks)} chunks, dim={meta['dim']}, {source}, top-{args.top_k}")
    rng = random.Random(args.seed + 1)
    pair_cos = [dot(unit_vectors[rng.randrange(len(chunks))], unit_vectors[rng.randrange(len(chunks))])
                for _ in range(500)]
    print(f"avg cosine of two random chunks: {sum(pair_cos) / len(pair_cos):.3f}")

    t = time.perf_counter()
    truth = [exhaustive_search(q, unit_vectors, args.top_k) for q in queries]
    exact_ms = 1000 * (time.perf_counter() - t) / len(queries)
    top1 = sum(exact[0][0] for exact in truth) / len(truth)
    print(f"avg cosine of query and its best chunk: {top1:.3f}\n")
    print(f"{'setting':<28} {'recall@' + str(args.top_k):>9} {'candidates':>11} {'ms/query':>9} {'speedup':>8}")
    print(f"{'exhaustive':<28} {1.0:>9.3f} {1.0:>11.1%} {exact_ms:>9.1f} {1.0:>7.1f}x")

    for path in args.ann:
        ann = load_ann(path, meta, len(chunks))
        print(f"\n{describe(ann)}")
        if ann["meta"]["method"] == "ivf":
            nlist = ann["meta"]["params"]["nlist"]
            settings = [("nprobe", p) for p in sorted({1, 2, 4, 8, 16, nlist}) if p <= nlist]
        else:
            settings = [("probe_radius", r) for r in (0, 1, 2)]
        for name, value in settings:
            recall, cand, ms = evaluate(queries, truth, unit_vectors, ann, args.top_k, **{name: value})
            print(f"{f'  {name}={value}':<28} {recall:>9.3f} {cand:>11.1%} {ms:>9.1f} {exact_ms / ms:>7.1f}x")


if __name__ == "__main__":
    main()
