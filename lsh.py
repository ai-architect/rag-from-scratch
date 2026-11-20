"""
lsh.py - Locality-Sensitive Hashing (random hyperplanes) in pure Python + math.

Idea:
    A random vector r defines a hyperplane through the origin. Every vector v
    lies on one side of it:  bit = 1 if r . v >= 0 else 0.
    With `bits` random hyperplanes, each vector gets a bit pattern
    (its "signature"), e.g. 0b1011001, which we use as a bucket key.

    Why similar vectors share buckets: the chance that a random hyperplane
    separates two vectors is theta / pi, where theta is the angle between
    them (Charikar, 2002). Small angle (high cosine) means rarely separated,
    so the vectors usually get the same signature.

    BUILD   For each of `tables` independent sets of hyperplanes, put every
            chunk into the bucket given by its signature.
    SEARCH  Compute the query's signature in each table and collect all chunks
            in the matching buckets (the candidates). Then rank the candidates
            exactly by cosine, as in the exhaustive search.

Mean-centering:
    Real embeddings are anisotropic: they all point roughly the same way.
    With gemini-embedding-001, even unrelated chunks have cosine ~0.7, and
    the mean vector has length ~0.84. A random hyperplane through the origin
    then puts most chunks on the same side, so a few giant buckets swallow
    the corpus. We therefore hash (v - mean) instead of v. That moves the
    origin to the centre of the data cloud. The final ranking still uses the
    original vectors, so the scores do not change; only the bucketing does.

Knobs:
    bits          more bits mean smaller buckets: faster, but lower recall.
                  2^bits buckets per table, about N / 2^bits chunks each.
    tables        more tables mean more chances to meet the true neighbour:
                  better recall, but more candidates.
    probe_radius  "multi-probe": also look at buckets whose key differs in
                  up to this many bits (neighbouring buckets). Raises recall
                  without building more tables.
"""

from __future__ import annotations

import heapq
import random
from itertools import combinations

from rag_search import Vector, dot


def make_hyperplanes(dim: int, bits: int, tables: int, seed: int = 0) -> list[list[Vector]]:
    """tables x bits random normal vectors of length dim.

    Each component is drawn from N(0, 1), so the direction of the vector is
    uniformly random. Only the sign of r . v matters, so no normalization
    is needed.
    """
    rng = random.Random(seed)
    return [[[rng.gauss(0.0, 1.0) for _ in range(dim)] for _ in range(bits)] for _ in range(tables)]


def mean_vector(vectors: list[Vector]) -> Vector:
    """Component-wise average:  mu_t = (1/N) * sum_i v_i[t]"""
    n = len(vectors)
    return [sum(column) / n for column in zip(*vectors)]


def signature(v: Vector, planes: list[Vector], center: Vector) -> int:
    """Pack the sign bits of r . (v - center) for every hyperplane r into one integer."""
    shifted = [x - c for x, c in zip(v, center)]
    key = 0
    for b, r in enumerate(planes):
        if dot(r, shifted) >= 0.0:
            key |= 1 << b
    return key


def build_lsh(unit_vectors: list[Vector], bits: int, tables: int, seed: int = 0,
              center: bool = True) -> dict:
    """Return {"hyperplanes", "center", "tables": [{key: [chunk positions]}, ...]}.

    center=False hashes the raw vectors (center = zero vector), so you can
    see the effect of anisotropy yourself.
    """
    d = len(unit_vectors[0])
    hyperplanes = make_hyperplanes(d, bits, tables, seed)
    mu = mean_vector(unit_vectors) if center else [0.0] * d
    buckets: list[dict[int, list[int]]] = []
    for planes in hyperplanes:
        table: dict[int, list[int]] = {}
        for i, v in enumerate(unit_vectors):
            table.setdefault(signature(v, planes, mu), []).append(i)
        buckets.append(table)
    return {"hyperplanes": hyperplanes, "center": mu, "tables": buckets}


def probe_keys(key: int, bits: int, radius: int) -> list[int]:
    """The key itself plus every key within Hamming distance `radius` (flip 1..radius bits)."""
    keys = [key]
    for r in range(1, radius + 1):
        for flip in combinations(range(bits), r):
            k = key
            for b in flip:
                k ^= 1 << b
            keys.append(k)
    return keys


def search_lsh(
    query_hat: Vector, unit_vectors: list[Vector], lsh: dict, top_k: int, probe_radius: int = 0
) -> tuple[list[tuple[float, int]], dict]:
    """Approximate top-k search. Returns (hits, stats), hits best first as in rag_search.py."""
    hyperplanes, tables, center = lsh["hyperplanes"], lsh["tables"], lsh["center"]
    bits = len(hyperplanes[0])

    # 1) hash the query in every table and collect the chunks from its buckets
    candidates: set[int] = set()
    buckets_probed = 0
    for planes, table in zip(hyperplanes, tables):
        for key in probe_keys(signature(query_hat, planes, center), bits, probe_radius):
            buckets_probed += 1
            candidates.update(table.get(key, ()))
    # 2) exact cosine ranking of the candidates only
    hits = heapq.nlargest(top_k, ((dot(query_hat, unit_vectors[i]), i) for i in candidates))

    stats = {
        "candidates": len(candidates),
        "dot_products": len(tables) * bits + len(candidates),  # hashing + exact scoring
        "detail": f"probed {buckets_probed} buckets in {len(tables)} tables x {bits} bits, radius {probe_radius}",
    }
    return hits, stats
