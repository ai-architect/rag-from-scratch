"""
ivf.py - Inverted File Index (IVF) in pure Python + math.

Idea:
    BUILD   Cluster the N chunk vectors into `nlist` groups with k-means.
            Every chunk goes into the list of its closest centroid
            (an "inverted list", like the posting lists of a search engine).
    SEARCH  Compare the query with the nlist centroids only, pick the
            `nprobe` closest ones, and run the exact cosine search on the
            chunks in those lists alone.

Cost per query:  nlist + (chunks in the probed lists) dot products,
instead of N. With nlist ~ sqrt(N) and a small nprobe, that's a fraction
of the exhaustive search.

The price is recall: if the true best chunk sits in a list we did not
probe, we miss it. A larger nprobe gives better recall but slower search;
nprobe = nlist is exactly the exhaustive search again.

All vectors here are unit vectors, so "closest" means "largest dot
product" (= cosine similarity). That variant is called spherical k-means.
"""

from __future__ import annotations

import heapq
import random
import sys

from rag_search import Vector, dot, normalize


# --------------------------------------------------------------------------- #
# k-means (spherical: unit vectors, similarity = dot product)
# --------------------------------------------------------------------------- #


def _nearest(v: Vector, centroids: list[Vector]) -> int:
    """Index of the centroid with the largest dot product with v."""
    best_j, best_score = 0, -2.0
    for j, c in enumerate(centroids):
        score = dot(v, c)
        if score > best_score:
            best_j, best_score = j, score
    return best_j


def kmeans_plus_plus_init(vectors: list[Vector], k: int, rng: random.Random) -> list[Vector]:
    """Pick k starting centroids that are spread out (k-means++).

    The first centroid is a random vector. Each further one is drawn at random
    with probability proportional to distance^2 to the nearest centroid chosen
    so far, so new centroids tend to land in regions not yet covered.
    Distance here is the cosine distance 1 - cos(theta).
    """
    centroids = [vectors[rng.randrange(len(vectors))]]
    dist = [max(0.0, 1.0 - dot(v, centroids[0])) for v in vectors]
    while len(centroids) < k:
        weights = [d * d for d in dist]
        total = sum(weights)
        if total == 0.0:  # all remaining points coincide with a centroid
            pick = rng.randrange(len(vectors))
        else:
            r, acc, pick = rng.random() * total, 0.0, len(vectors) - 1
            for i, w in enumerate(weights):
                acc += w
                if acc >= r:
                    pick = i
                    break
        c = vectors[pick]
        centroids.append(c)
        dist = [min(d, max(0.0, 1.0 - dot(v, c))) for d, v in zip(dist, vectors)]
    return centroids


def kmeans(
    vectors: list[Vector], k: int, max_iter: int = 10, seed: int = 0, verbose: bool = False
) -> tuple[list[Vector], list[int]]:
    """Spherical k-means. Returns (centroids, assignment) with assignment[i] = cluster of vectors[i].

    Each iteration has two steps:
      1. assign: every vector goes to its nearest centroid    (N * k dot products)
      2. update: every centroid becomes the normalized mean of its members
    The loop stops early once no vector changes its cluster.
    """
    if not 1 <= k <= len(vectors):
        raise ValueError(f"k must be between 1 and N={len(vectors)}, got {k}")
    rng = random.Random(seed)
    d = len(vectors[0])
    centroids = kmeans_plus_plus_init(vectors, k, rng)
    assignment = [-1] * len(vectors)

    for iteration in range(1, max_iter + 1):
        # 1) assignment step
        changed = 0
        for i, v in enumerate(vectors):
            j = _nearest(v, centroids)
            if j != assignment[i]:
                assignment[i], changed = j, changed + 1

        # 2) update step: centroid = normalize(sum of member vectors)
        sums = [[0.0] * d for _ in range(k)]
        counts = [0] * k
        for v, j in zip(vectors, assignment):
            s = sums[j]
            for t in range(d):
                s[t] += v[t]
            counts[j] += 1
        for j in range(k):
            if counts[j] == 0:  # empty cluster: re-seed it with a random vector
                centroids[j] = vectors[rng.randrange(len(vectors))]
            else:
                centroids[j] = normalize(sums[j])

        if verbose:
            print(f"  k-means iteration {iteration}: {changed} vectors changed cluster", file=sys.stderr)
        if changed == 0:
            break
    return centroids, assignment


# --------------------------------------------------------------------------- #
# IVF build + search
# --------------------------------------------------------------------------- #


def build_ivf(unit_vectors: list[Vector], nlist: int, max_iter: int = 10, seed: int = 0,
              verbose: bool = False) -> dict:
    """Return {"centroids": [...], "lists": [[chunk positions], ...]}."""
    centroids, assignment = kmeans(unit_vectors, nlist, max_iter, seed, verbose)
    lists: list[list[int]] = [[] for _ in range(nlist)]
    for i, j in enumerate(assignment):
        lists[j].append(i)
    return {"centroids": centroids, "lists": lists}


def search_ivf(
    query_hat: Vector, unit_vectors: list[Vector], ivf: dict, top_k: int, nprobe: int = 4
) -> tuple[list[tuple[float, int]], dict]:
    """Approximate top-k search. Returns (hits, stats), hits best first as in rag_search.py."""
    centroids, lists = ivf["centroids"], ivf["lists"]
    nprobe = max(1, min(nprobe, len(centroids)))

    # 1) coarse step: which clusters are closest to the query?
    probed = heapq.nlargest(nprobe, ((dot(query_hat, c), j) for j, c in enumerate(centroids)))
    # 2) fine step: exact cosine scores for the chunks in those clusters only
    candidates = [i for _, j in probed for i in lists[j]]
    hits = heapq.nlargest(top_k, ((dot(query_hat, unit_vectors[i]), i) for i in candidates))

    stats = {
        "candidates": len(candidates),
        "dot_products": len(centroids) + len(candidates),
        "detail": f"probed {nprobe}/{len(centroids)} lists",
    }
    return hits, stats
