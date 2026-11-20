"""
ann_index.py - save, load and query approximate-nearest-neighbour (ANN) index files.

The embeddings file (index.pkl from build_index.py) is never modified. An ANN
index is a small *extra* file next to it that only stores the search
structure:

    index_gemini.pkl        chunks + vectors           (from build_index.py)
    index_gemini.ivf.pkl    centroids + inverted lists (from build_ann.py --method ivf)
    index_gemini.lsh.pkl    hyperplanes + buckets      (from build_ann.py --method lsh)

The extra files refer to chunks by their position in index.pkl's chunk list.
To make sure that still holds, each extra file stores a fingerprint of the
embeddings file it was built from, and loading checks it.

WARNING: pickle files can execute code when loaded. Only load files you
created yourself.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from ivf import search_ivf
from lsh import search_lsh
from rag_search import Vector

METHODS = ("ivf", "lsh")


def fingerprint(meta: dict, num_chunks: int) -> dict:
    """What identifies an embeddings file: same model, same chunks, same build."""
    return {
        "provider": meta.get("provider"),
        "model": meta.get("model"),
        "dim": meta.get("dim"),
        "num_chunks": num_chunks,
        "created_at": meta.get("created_at"),
    }


def default_ann_path(index_path: Path, method: str) -> Path:
    """index_gemini.pkl -> index_gemini.ivf.pkl"""
    return index_path.with_suffix(f".{method}.pkl")


def save_ann(path: Path, method: str, params: dict, structure: dict, base_fingerprint: dict,
             build_seconds: float) -> None:
    ann = {
        "meta": {"method": method, "params": params, "base": base_fingerprint,
                 "build_seconds": round(build_seconds, 2)},
        "structure": structure,
    }
    with path.open("wb") as f:
        pickle.dump(ann, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_ann(path: Path, meta: dict, num_chunks: int) -> dict:
    """Load an ANN file and check that it belongs to the loaded embeddings file."""
    with path.open("rb") as f:
        ann = pickle.load(f)  # only load pickles you trust!
    expected, found = fingerprint(meta, num_chunks), ann["meta"]["base"]
    if expected != found:
        raise SystemExit(
            f"{path} was built from a different embeddings file.\n"
            f"  expected: {expected}\n  found:    {found}\n"
            f"Rebuild it with:  python build_ann.py --index <your index.pkl> --method {ann['meta']['method']}"
        )
    return ann


def ann_search(ann: dict, query_hat: Vector, unit_vectors: list[Vector], top_k: int,
               nprobe: int = 4, probe_radius: int = 0) -> tuple[list[tuple[float, int]], dict]:
    """Dispatch to the right search function. Returns (hits, stats)."""
    method = ann["meta"]["method"]
    if method == "ivf":
        return search_ivf(query_hat, unit_vectors, ann["structure"], top_k, nprobe=nprobe)
    if method == "lsh":
        return search_lsh(query_hat, unit_vectors, ann["structure"], top_k, probe_radius=probe_radius)
    raise ValueError(f"unknown ANN method {method!r}")


def describe(ann: dict) -> str:
    """One-line summary, e.g. 'IVF (nlist=38, seed=0), built in 85.3 s'."""
    m = ann["meta"]
    params = ", ".join(f"{k}={v}" for k, v in m["params"].items())
    return f"{m['method'].upper()} ({params}), built in {m['build_seconds']} s"
