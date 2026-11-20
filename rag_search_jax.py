"""
rag_search_jax.py - the same exhaustive search as rag_search.py, written in JAX.

What changes compared with the pure-math version:
    list of N lists of d floats  ->  one (N, d) jax.Array, placed on the GPU/TPU if one is available
    N separate dot products      ->  one matrix-vector product  M_hat @ q_hat
    heapq.nlargest               ->  jax.lax.top_k
    normalize + score + top-k    ->  one function compiled by XLA with @jax.jit

The index file, the embedding call, the CLI and the output are identical to
rag_search.py and rag_search_numpy.py.

Notes:
  * JAX uses float32 by default; scores match the float64 versions to about
    6 decimal places. Pass --x64 for float64 (slow on most GPUs).
  * The first call of a jitted function compiles it, which takes noticeably
    longer than later calls. We run one warm-up search at load time so the
    per-query timing shows only the search itself.
  * For GPU support install a CUDA build, e.g.  pip install "jax[cuda12]".

WARNING: pickle files can execute code when loaded. Only load index files
you created yourself.
"""

from __future__ import annotations

import argparse
import pickle
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from embedders import embed_texts
from rag_search import print_results  # output formatting is shared, not re-implemented


# --------------------------------------------------------------------------- #
# Vector math with jax.numpy (compare with dot / norm / normalize in rag_search.py)
# --------------------------------------------------------------------------- #


def normalize_rows(M: jax.Array) -> jax.Array:
    """Scale every row of M to length 1:  M_hat[i] = M[i] / ||M[i]||"""
    norms = jnp.linalg.norm(M, axis=1, keepdims=True)  # shape (N, 1), broadcasts over rows
    return M / norms


def normalize(v: jax.Array) -> jax.Array:
    """Scale a single vector to length 1:  v_hat = v / ||v||"""
    return v / jnp.linalg.norm(v)


def cosine_similarity(a: jax.Array, b: jax.Array) -> float:
    """cos(theta) = (a . b) / (||a|| * ||b||). Kept for completeness and tests."""
    return float(a @ b / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))


@partial(jax.jit, static_argnames="top_k")
def _search_kernel(query: jax.Array, M_hat: jax.Array, top_k: int) -> tuple[jax.Array, jax.Array]:
    """Normalize the query, score all chunks, return (top scores, top indices).

    jax.jit traces this function once and compiles it with XLA into a single
    fused program. `top_k` must be static (fixed at compile time) because it
    determines the shape of the output; a new value triggers a recompile.
    """
    scores = M_hat @ normalize(query)  # N * d multiplications, shape (N,)
    return jax.lax.top_k(scores, top_k)  # already sorted, best first


# --------------------------------------------------------------------------- #
# Index loading and exhaustive search
# --------------------------------------------------------------------------- #


def load_index(path: Path, dtype) -> tuple[dict, list[dict], jax.Array]:
    """Return (meta, chunks, M_hat). Row i of M_hat belongs to chunks[i]."""
    with path.open("rb") as f:
        index = pickle.load(f)  # only load pickles you trust!
    meta, chunks = index["meta"], index["chunks"]
    dim = meta["dim"]
    for chunk in chunks:
        if len(chunk["vector"]) != dim:
            raise ValueError(f"chunk {chunk['id']} has dim {len(chunk['vector'])}, expected {dim}")
    M = jnp.asarray([chunk["vector"] for chunk in chunks], dtype=dtype)  # shape (N, d), on device
    if bool(jnp.any(jnp.linalg.norm(M, axis=1) == 0.0)):
        raise ValueError("cannot normalize the zero vector")
    return meta, chunks, normalize_rows(M)


def exhaustive_search(query: jax.Array, M_hat: jax.Array, top_k: int) -> list[tuple[float, int]]:
    """Score EVERY chunk against the query; return [(score, chunk_position), ...] best first."""
    k = min(top_k, M_hat.shape[0])
    # JAX runs asynchronously. jax.device_get waits for the result and copies
    # both outputs back to the host in a single transfer.
    scores, indices = jax.device_get(_search_kernel(query, M_hat, k))
    return [(float(s), int(i)) for s, i in zip(scores.tolist(), indices.tolist())]


def to_device(vector: list[float], like: jax.Array) -> jax.Array:
    """Copy a Python list to the device that holds `like`, in its dtype.

    Going through NumPy first matters: jnp.asarray(list) converts element
    by element (~0.9 ms for 1,536 floats), np.asarray + device_put copies
    one contiguous buffer (~0.3 ms).
    """
    return jax.device_put(np.asarray(vector, dtype=like.dtype), like.device)


def answer(query: str, meta: dict, chunks, M_hat, top_k: int) -> None:
    # 1) Embed the query with the SAME provider/model as the corpus.
    t0 = time.perf_counter()
    [query_vector] = embed_texts([query], meta["provider"], meta["model"], task="query")
    t1 = time.perf_counter()
    # 2) + 3) Normalize and compare against every chunk inside the jitted kernel.
    if not any(query_vector):
        raise ValueError("cannot normalize the zero vector")
    results = exhaustive_search(to_device(query_vector, M_hat), M_hat, top_k)
    t2 = time.perf_counter()

    print_results(results, chunks)
    n, d = M_hat.shape
    print(f"\n[embedding {1000 * (t1 - t0):.0f} ms | exhaustive search over {n} chunks: "
          f"{n * d:,} multiplications in {1000 * (t2 - t1):.1f} ms (JAX on {M_hat.device})]")


def main() -> None:
    parser = argparse.ArgumentParser(description="JAX exhaustive vector search.")
    parser.add_argument("--index", type=Path, default=Path("index.pkl"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query", help="run a single query and exit (otherwise interactive)")
    parser.add_argument("--x64", action="store_true", help="use float64 instead of float32")
    args = parser.parse_args()

    if args.x64:
        jax.config.update("jax_enable_x64", True)
    dtype = jnp.float64 if args.x64 else jnp.float32

    meta, chunks, M_hat = load_index(args.index, dtype)
    print(f"Loaded {len(chunks)} chunks from {len(meta['corpus_files'])} files "
          f"({meta['provider']} / {meta['model']}, dim={meta['dim']}) into a "
          f"{M_hat.shape[0]}x{M_hat.shape[1]} {M_hat.dtype} array on {M_hat.device}.")

    # Warm-up: compile the kernel now so the first real query is not slowed down.
    t0 = time.perf_counter()
    exhaustive_search(to_device([1.0] * M_hat.shape[1], M_hat), M_hat, args.top_k)
    print(f"JIT-compiled search kernel in {1000 * (time.perf_counter() - t0):.0f} ms.")

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
