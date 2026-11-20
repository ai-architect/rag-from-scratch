"""
build_ann.py - build an IVF or LSH index on top of an existing embeddings file.

No API calls: it only reads the vectors in index.pkl and writes an extra file
next to it (index.pkl itself is left untouched).

Usage:
    python build_ann.py --index index_gemini.pkl --method ivf               # nlist = sqrt(N)
    python build_ann.py --index index_gemini.pkl --method ivf --nlist 50
    python build_ann.py --index index_gemini.pkl --method lsh               # bits ~ log2(N/50), 16 tables
    python build_ann.py --index index_gemini.pkl --method lsh --bits 6 --tables 12

Output defaults to index_gemini.ivf.pkl / index_gemini.lsh.pkl.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from ann_index import METHODS, default_ann_path, fingerprint, save_ann
from ivf import build_ivf
from lsh import build_lsh
from rag_search import load_index


def _sizes(groups) -> str:
    sizes = [len(g) for g in groups]
    return f"min {min(sizes)}, avg {sum(sizes) / len(sizes):.1f}, max {max(sizes)}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an IVF or LSH index for rag_search_ann.py.")
    parser.add_argument("--index", type=Path, default=Path("index.pkl"), help="embeddings file from build_index.py")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--out", type=Path, help="default: <index>.<method>.pkl")
    parser.add_argument("--seed", type=int, default=0, help="random seed (k-means init / hyperplanes)")
    ivf_args = parser.add_argument_group("IVF")
    ivf_args.add_argument("--nlist", type=int, help="number of clusters (default: sqrt(N))")
    ivf_args.add_argument("--max-iter", type=int, default=10, help="max k-means iterations")
    lsh_args = parser.add_argument_group("LSH")
    lsh_args.add_argument("--bits", type=int, help="hyperplanes per table (default: log2(N/50), ~50 chunks per bucket)")
    lsh_args.add_argument("--tables", type=int, default=16, help="number of hash tables")
    lsh_args.add_argument("--no-center", action="store_true",
                          help="hash raw vectors instead of mean-centered ones (to see why centering matters)")
    args = parser.parse_args()

    meta, chunks, unit_vectors = load_index(args.index)
    n, d = len(unit_vectors), meta["dim"]
    print(f"Loaded {n} chunks, dim={d} ({meta['provider']} / {meta['model']}).")
    out = args.out or default_ann_path(args.index, args.method)

    t0 = time.perf_counter()
    if args.method == "ivf":
        nlist = args.nlist or max(1, round(math.sqrt(n)))
        params = {"nlist": nlist, "max_iter": args.max_iter, "seed": args.seed}
        print(f"Running k-means with k={nlist} (about {n * nlist * d:,} multiplications per iteration) ...")
        structure = build_ivf(unit_vectors, nlist, args.max_iter, args.seed, verbose=True)
        summary = f"{nlist} inverted lists, chunks per list: {_sizes(structure['lists'])}"
    else:
        bits = args.bits or max(1, round(math.log2(max(2, n / 50))))
        params = {"bits": bits, "tables": args.tables, "seed": args.seed, "center": not args.no_center}
        print(f"Hashing with {args.tables} tables x {bits} random hyperplanes ...")
        structure = build_lsh(unit_vectors, bits, args.tables, args.seed, center=not args.no_center)
        used = [len(table) for table in structure["tables"]]
        all_buckets = [b for table in structure["tables"] for b in table.values()]
        summary = (f"{2 ** bits} possible buckets per table, {min(used)}-{max(used)} non-empty; "
                   f"chunks per non-empty bucket: {_sizes(all_buckets)}")
    seconds = time.perf_counter() - t0

    save_ann(out, args.method, params, structure, fingerprint(meta, n), seconds)
    print(summary)
    print(f"Built in {seconds:.1f} s, saved to {out}")
    print(f"Search with:  python rag_search_ann.py --index {args.index} --ann {out}")


if __name__ == "__main__":
    main()
