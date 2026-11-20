"""
rag_search_numpy.py - the same exhaustive search as rag_search.py, vectorized with NumPy.

What changes compared with the pure-math version:
    list of N lists of d floats  ->  one (N, d) matrix M
    N separate dot products      ->  one matrix-vector product  M_hat @ q_hat
    heapq.nlargest               ->  np.argpartition + argsort on the k best

The index file, the embedding call, the CLI and the output are identical, so you
can run both programs side by side and compare results and timings:

    python rag_search.py       --index index_gemini.pkl --query "Why did Lehman collapse?"
    python rag_search_numpy.py --index index_gemini.pkl --query "Why did Lehman collapse?"

WARNING: pickle files can execute code when loaded. Only load index files
you created yourself.
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np

from embedders import embed_texts
from rag_search import print_results  # output formatting is shared, not re-implemented


# --------------------------------------------------------------------------- #
# Vector math with NumPy (compare with dot / norm / normalize in rag_search.py)
# --------------------------------------------------------------------------- #


def normalize_rows(M: np.ndarray) -> np.ndarray:
    """Scale every row of M to length 1:  M_hat[i] = M[i] / ||M[i]||

    np.linalg.norm(axis=1) computes all N row norms at once. keepdims=True gives
    them shape (N, 1), so the division broadcasts across each row.
    """
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("cannot normalize the zero vector")
    return M / norms


def normalize(v: np.ndarray) -> np.ndarray:
    """Scale a single vector to length 1:  v_hat = v / ||v||"""
    n = np.linalg.norm(v)
    if n == 0.0:
        raise ValueError("cannot normalize the zero vector")
    return v / n


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """cos(theta) = (a . b) / (||a|| * ||b||). Kept for completeness and tests."""
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# --------------------------------------------------------------------------- #
# Index loading and exhaustive search
# --------------------------------------------------------------------------- #


def load_index(path: Path) -> tuple[dict, list[dict], np.ndarray]:
    """Return (meta, chunks, M_hat). Row i of M_hat belongs to chunks[i]."""
    with path.open("rb") as f:
        index = pickle.load(f)  # only load pickles you trust!
    meta, chunks = index["meta"], index["chunks"]
    dim = meta["dim"]
    for chunk in chunks:
        if len(chunk["vector"]) != dim:
            raise ValueError(f"chunk {chunk['id']} has dim {len(chunk['vector'])}, expected {dim}")
    # float64 gives the same precision as the pure-Python floats in rag_search.py.
    M = np.asarray([chunk["vector"] for chunk in chunks], dtype=np.float64)  # shape (N, d)
    return meta, chunks, normalize_rows(M)


def exhaustive_search(query_hat: np.ndarray, M_hat: np.ndarray, top_k: int) -> list[tuple[float, int]]:
    """Score EVERY chunk against the query; return [(score, chunk_position), ...] best first.

    scores = M_hat @ q_hat is the same N * d multiplications as the Python
    loop, but it runs as one call into optimized BLAS code.

    np.argpartition moves the k largest scores to the end in O(N) without
    sorting the rest. Only those k are then sorted, which is the same idea
    as heapq.nlargest.
    """
    scores = M_hat @ query_hat  # shape (N,)
    k = min(top_k, scores.shape[0])
    best = np.argpartition(scores, -k)[-k:]  # the k best, in no particular order
    best = best[np.argsort(-scores[best])]  # sort just those k, best first
    return [(float(scores[i]), int(i)) for i in best]


def answer(query: str, meta: dict, chunks, M_hat, top_k: int) -> None:
    # 1) Embed the query with the SAME provider/model as the corpus.
    t0 = time.perf_counter()
    [query_vector] = embed_texts([query], meta["provider"], meta["model"], task="query")
    t1 = time.perf_counter()
    # 2) Normalize it, so cosine similarity == dot product.
    query_hat = normalize(np.asarray(query_vector, dtype=np.float64))
    # 3) Compare against every chunk in one matrix-vector product.
    results = exhaustive_search(query_hat, M_hat, top_k)
    t2 = time.perf_counter()

    print_results(results, chunks)
    n, d = M_hat.shape
    print(f"\n[embedding {1000 * (t1 - t0):.0f} ms | exhaustive search over {n} chunks: "
          f"{n * d:,} multiplications in {1000 * (t2 - t1):.1f} ms (NumPy)]")


def main() -> None:
    parser = argparse.ArgumentParser(description="NumPy exhaustive vector search.")
    parser.add_argument("--index", type=Path, default=Path("index.pkl"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query", help="run a single query and exit (otherwise interactive)")
    args = parser.parse_args()

    meta, chunks, M_hat = load_index(args.index)
    print(f"Loaded {len(chunks)} chunks from {len(meta['corpus_files'])} files "
          f"({meta['provider']} / {meta['model']}, dim={meta['dim']}) into a "
          f"{M_hat.shape[0]}x{M_hat.shape[1]} {M_hat.dtype} matrix.")

    if args.query:
        answer(args.query, meta, chunks, M_hat, args.top_k)
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
        answer(query, meta, chunks, M_hat, args.top_k)


if __name__ == "__main__":
    main()
