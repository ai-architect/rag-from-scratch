"""
rag_search.py - the "R" in RAG, using nothing but Python lists and the math module.

Loads the index written by build_index.py, asks for a query, embeds it with the
same provider/model as the corpus, and runs an *exhaustive* search: the cosine
similarity between the query and every single chunk is computed, and the top-k
chunks are printed together with where they came from.

Usage:
    python rag_search.py --index index_gemini.pkl
    python rag_search.py --index index_openai.pkl --top-k 3
    python rag_search.py --index index_openai.pkl --query "Who founded Goldman Sachs?"

WARNING: pickle files can execute code when loaded. Only load index files
you created yourself.
"""

from __future__ import annotations

import argparse
import heapq
import math
import pickle
import textwrap
import time
from pathlib import Path

from embedders import embed_texts

Vector = list[float]


# --------------------------------------------------------------------------- #
# Vector math with the standard library only
# --------------------------------------------------------------------------- #


def dot(a: Vector, b: Vector) -> float:
    """Dot product:  a . b = sum_i a_i * b_i

    math.fsum adds floats exactly (no rounding build-up), which matters
    a little when you add up thousands of small products.
    """
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    return math.fsum(x * y for x, y in zip(a, b))


def norm(a: Vector) -> float:
    """Euclidean (L2) length:  ||a|| = sqrt(a . a) = sqrt(sum_i a_i^2)"""
    return math.sqrt(dot(a, a))


def normalize(a: Vector) -> Vector:
    """Scale a vector to length 1:  a_hat = a / ||a||

    The direction is kept and only the length changes. For text embeddings
    the direction carries the meaning, and the length is mostly noise.
    """
    n = norm(a)
    if n == 0.0:
        raise ValueError("cannot normalize the zero vector")
    return [x / n for x in a]


def cosine_similarity(a: Vector, b: Vector) -> float:
    """cos(theta) = (a . b) / (||a|| * ||b||), a value in [-1, 1].

     1 -> same direction (very similar meaning)
     0 -> orthogonal (unrelated)
    -1 -> opposite direction
    """
    return dot(a, b) / (norm(a) * norm(b))


def cosine_similarity_unit(a_hat: Vector, b_hat: Vector) -> float:
    """Fast path for vectors that are already normalized.

    If ||a|| = ||b|| = 1, the denominator above is 1, so the cosine is just
    the dot product. That is why we normalize every chunk vector ONCE at load
    time: each comparison then costs d multiplications instead of 3d plus two
    square roots.
    """
    return dot(a_hat, b_hat)


# --------------------------------------------------------------------------- #
# Index loading and exhaustive search
# --------------------------------------------------------------------------- #


def load_index(path: Path) -> tuple[dict, list[dict], list[Vector]]:
    """Return (meta, chunks, unit_vectors). unit_vectors[i] belongs to chunks[i]."""
    with path.open("rb") as f:
        index = pickle.load(f)  # only load pickles you trust!
    meta, chunks = index["meta"], index["chunks"]
    dim = meta["dim"]
    for chunk in chunks:
        if len(chunk["vector"]) != dim:
            raise ValueError(f"chunk {chunk['id']} has dim {len(chunk['vector'])}, expected {dim}")
    unit_vectors = [normalize(chunk["vector"]) for chunk in chunks]
    return meta, chunks, unit_vectors


def exhaustive_search(query_hat: Vector, unit_vectors: list[Vector], top_k: int) -> list[tuple[float, int]]:
    """Score EVERY chunk against the query; return [(score, chunk_position), ...] best first.

    This is O(N * d) for N chunks of dimension d. That is fine for thousands
    of chunks; for millions you would want an approximate index (IVF, LSH,
    HNSW; see the README).

    heapq.nlargest keeps only the k best in a small heap (O(N log k)).
    sorted(scores, reverse=True)[:top_k] gives the same result in O(N log N).
    """
    scores = ((cosine_similarity_unit(query_hat, v), i) for i, v in enumerate(unit_vectors))
    return heapq.nlargest(top_k, scores)


def print_results(results, chunks, width: int = 100) -> None:
    for rank, (score, i) in enumerate(results, start=1):
        chunk = chunks[i]
        print(f"\n#{rank}  score={score:.4f}  {chunk['source']}#chunk{chunk['chunk_index']}"
              f"  chars {chunk['start_char']}-{chunk['end_char']}")
        print("-" * width)
        for paragraph in chunk["text"].split("\n"):
            if paragraph.strip():
                print(textwrap.fill(paragraph, width=width, initial_indent="  ", subsequent_indent="  "))


def answer(query: str, meta: dict, chunks, unit_vectors, top_k: int) -> None:
    # 1) Embed the query with the SAME provider/model as the corpus.
    t0 = time.perf_counter()
    [query_vector] = embed_texts([query], meta["provider"], meta["model"], task="query")
    t1 = time.perf_counter()
    # 2) Normalize it, so cosine similarity == dot product.
    query_hat = normalize(query_vector)
    # 3) Compare against every chunk.
    results = exhaustive_search(query_hat, unit_vectors, top_k)
    t2 = time.perf_counter()

    print_results(results, chunks)
    n, d = len(unit_vectors), meta["dim"]
    print(f"\n[embedding {1000 * (t1 - t0):.0f} ms | exhaustive search over {n} chunks: "
          f"{n * d:,} multiplications in {1000 * (t2 - t1):.0f} ms]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure-math exhaustive vector search.")
    parser.add_argument("--index", type=Path, default=Path("index.pkl"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query", help="run a single query and exit (otherwise interactive)")
    args = parser.parse_args()

    meta, chunks, unit_vectors = load_index(args.index)
    print(f"Loaded {len(chunks)} chunks from {len(meta['corpus_files'])} files "
          f"({meta['provider']} / {meta['model']}, dim={meta['dim']}).")

    if args.query:
        answer(args.query, meta, chunks, unit_vectors, args.top_k)
        return

    print("Type a question (empty line or 'quit' to exit).")
    while True:
        try:
            query = input("\nQuery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query or query.lower() in {"quit", "exit", "q"}:
            break
        answer(query, meta, chunks, unit_vectors, args.top_k)


if __name__ == "__main__":
    main()
