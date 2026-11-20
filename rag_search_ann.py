"""
rag_search_ann.py - like rag_search.py, but searches with an IVF or LSH index.

The embeddings file, the query embedding and the output are the same as in
rag_search.py. The difference is that only a subset of the chunks (the
"candidates") is scored. The program reports how many, and with --compare
it also runs the exhaustive search and shows what the approximate search
missed (recall@k).

Usage:
    python rag_search_ann.py --index index_gemini.pkl --ann index_gemini.ivf.pkl [--nprobe 4]
    python rag_search_ann.py --index index_gemini.pkl --ann index_gemini.lsh.pkl [--probe-radius 1]
    python rag_search_ann.py ... --compare --query "Why did Lehman Brothers collapse?"
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from ann_index import ann_search, describe, load_ann
from embedders import embed_texts
from rag_search import exhaustive_search, load_index, normalize, print_results


def answer(query: str, meta, chunks, unit_vectors, ann, args) -> None:
    # 1) Embed the query with the SAME provider/model as the corpus, then normalize it.
    t0 = time.perf_counter()
    [query_vector] = embed_texts([query], meta["provider"], meta["model"], task="query")
    t1 = time.perf_counter()
    query_hat = normalize(query_vector)
    # 2) Approximate search: score only the candidates IVF/LSH hands us.
    hits, stats = ann_search(ann, query_hat, unit_vectors, args.top_k, args.nprobe, args.probe_radius)
    t2 = time.perf_counter()

    print_results(hits, chunks)
    n, d = len(unit_vectors), meta["dim"]
    print(f"\n[embedding {1000 * (t1 - t0):.0f} ms | {ann['meta']['method'].upper()}: {stats['detail']}; "
          f"scored {stats['candidates']} of {n} chunks ({100 * stats['candidates'] / n:.1f}%), "
          f"{stats['dot_products'] * d:,} multiplications in {1000 * (t2 - t1):.0f} ms]")
    if len(hits) < args.top_k:
        print(f"[only {len(hits)} candidates found; raise --nprobe / --probe-radius for more]")

    if args.compare:
        t3 = time.perf_counter()
        exact = exhaustive_search(query_hat, unit_vectors, args.top_k)
        t4 = time.perf_counter()
        found = {i for _, i in hits}
        missed = [(s, i) for s, i in exact if i not in found]
        print(f"[exhaustive: {n * d:,} multiplications in {1000 * (t4 - t3):.0f} ms | "
              f"recall@{args.top_k} = {len(exact) - len(missed)}/{len(exact)}]")
        for score, i in missed:
            c = chunks[i]
            print(f"  missed: score={score:.4f}  {c['source']}#chunk{c['chunk_index']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Approximate vector search with IVF or LSH.")
    parser.add_argument("--index", type=Path, default=Path("index.pkl"), help="embeddings file")
    parser.add_argument("--ann", type=Path, required=True, help="IVF/LSH file from build_ann.py")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--nprobe", type=int, default=4, help="IVF: number of lists to search")
    parser.add_argument("--probe-radius", type=int, default=0, help="LSH: also probe buckets up to this many bits away")
    parser.add_argument("--compare", action="store_true", help="also run the exhaustive search and report recall")
    parser.add_argument("--query", help="run a single query and exit (otherwise interactive)")
    args = parser.parse_args()

    meta, chunks, unit_vectors = load_index(args.index)
    ann = load_ann(args.ann, meta, len(chunks))
    print(f"Loaded {len(chunks)} chunks ({meta['provider']} / {meta['model']}, dim={meta['dim']}) "
          f"and {describe(ann)}.")

    if args.query:
        answer(args.query, meta, chunks, unit_vectors, ann, args)
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
        answer(query, meta, chunks, unit_vectors, ann, args)


if __name__ == "__main__":
    main()
