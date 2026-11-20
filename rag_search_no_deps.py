"""
rag_search_no_deps.py - rag_search.py without any pip-installed packages.

rag_search.py already does the vector math with nothing but lists and the math
module, but it embeds the query through the official SDKs (google-genai /
openai). This script calls the same embedding REST endpoints directly with
urllib from the standard library instead, so a bare Python 3.9+ is enough:

    python rag_search_no_deps.py --index index_gemini.pkl
    python rag_search_no_deps.py --index index_gemini.pkl --top-k 3 --query "Who founded Goldman Sachs?"

The corpus vectors are already stored in the index pickle; only the query
is embedded at search time (one HTTPS request). The search itself (normalize,
dot products, top-k) is reused unchanged from rag_search.py.

No API key? Use --dry-run. The query is then not embedded live; its vector is
read from a precomputed cache (index_gemini.queries.json, written by
precompute_queries.py from eval_questions.txt). No network, no key, no pip:

    python rag_search_no_deps.py --dry-run                                   # Deutsche Bank question
    python rag_search_no_deps.py --dry-run --query "Who founded Goldman Sachs?"
    python rag_search_no_deps.py --dry-run --interactive                     # pick from a menu

API keys are read from the environment, as in embedders.py:
    Gemini : GEMINI_API_KEY (or GOOGLE_API_KEY)
    OpenAI : OPENAI_API_KEY

WARNING: pickle files can execute code when loaded. Only load index files
you created yourself.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Pure-stdlib helpers shared with rag_search.py (no third-party imports there).
from rag_search import Vector, exhaustive_search, load_index, normalize, print_results

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"
OPENAI_URL = "https://api.openai.com/v1/embeddings"
DRY_RUN_DEFAULT_QUERY = "When was Deutsche Bank founded?"


# --------------------------------------------------------------------------- #
# Query embedding over plain HTTPS (urllib + json)
# --------------------------------------------------------------------------- #


def _post_json(url: str, payload: dict, headers: dict[str, str], timeout: float = 60.0) -> dict:
    """POST a JSON body and return the decoded JSON response."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


def _embed_gemini(text: str, model: str) -> Vector:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Please set the GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable.")
    payload = {
        "model": f"models/{model}",
        "content": {"parts": [{"text": text}]},
        # Queries and documents are embedded asymmetrically; build_index.py used
        # RETRIEVAL_DOCUMENT for the chunks.
        "taskType": "RETRIEVAL_QUERY",
    }
    data = _post_json(GEMINI_URL.format(model=model), payload, {"x-goog-api-key": api_key})
    return [float(x) for x in data["embedding"]["values"]]


def _embed_openai(text: str, model: str) -> Vector:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Please set the OPENAI_API_KEY environment variable.")
    payload = {"model": model, "input": [text]}
    data = _post_json(OPENAI_URL, payload, {"Authorization": f"Bearer {api_key}"})
    return [float(x) for x in data["data"][0]["embedding"]]


def embed_query(text: str, provider: str, model: str, attempts: int = 5) -> Vector:
    """Embed one query with the SAME provider/model as the corpus, retrying transient errors."""
    if provider == "gemini":
        call = _embed_gemini
    elif provider == "openai":
        call = _embed_openai
    else:
        raise ValueError(f"Unknown provider {provider!r}")

    delay = 2.0
    for attempt in range(1, attempts + 1):
        try:
            return call(text, model)
        except (RuntimeError, urllib.error.URLError, TimeoutError) as exc:
            if attempt == attempts:
                raise
            print(f"  API error ({exc}); retrying in {delay:.0f}s [{attempt}/{attempts - 1}]",
                  file=sys.stderr)
            time.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable")


# --------------------------------------------------------------------------- #
# Dry run: precomputed query vectors instead of the API
# --------------------------------------------------------------------------- #


def _key(text: str) -> str:
    """Forgiving match: ignore case and extra whitespace."""
    return " ".join(text.split()).casefold()


class QueryCache:
    """Query vectors precomputed by precompute_queries.py (plain JSON, safe to load)."""

    def __init__(self, path: Path, index_meta: dict):
        if not path.exists():
            raise SystemExit(f"No query cache at {path}. Create it (needs the SDK + API key once) with:\n"
                             f"  python precompute_queries.py --out {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        cache_meta = data["meta"]
        for field in ("provider", "model", "dim"):
            if cache_meta[field] != index_meta[field]:
                raise SystemExit(f"{path} was built with {field}={cache_meta[field]!r}, "
                                 f"but the index has {field}={index_meta[field]!r}.")
        self.path = path
        self.texts = [q["text"] for q in data["queries"]]
        self._vectors = {_key(q["text"]): q["vector"] for q in data["queries"]}

    def lookup(self, query: str) -> Vector:
        """Accept a cached question (any case/spacing) or its 1-based menu number."""
        if query.isdigit() and 1 <= int(query) <= len(self.texts):
            query = self.texts[int(query) - 1]
        try:
            return self._vectors[_key(query)]
        except KeyError:
            raise KeyError(f"{query!r} is not in {self.path}. --dry-run only knows these questions:\n"
                           + self.menu()) from None

    def menu(self) -> str:
        return "\n".join(f"  {i:2d}. {text}" for i, text in enumerate(self.texts, start=1))


# --------------------------------------------------------------------------- #
# Same flow as rag_search.py
# --------------------------------------------------------------------------- #


def answer(query: str, meta: dict, chunks, unit_vectors, top_k: int, cache: QueryCache | None = None) -> None:
    # 1) Embed the query with the SAME provider/model as the corpus
    #    (or, in a dry run, look up its precomputed embedding).
    t0 = time.perf_counter()
    if cache is not None:
        query_vector = cache.lookup(query)
    else:
        query_vector = embed_query(query, meta["provider"], meta["model"])
    t1 = time.perf_counter()
    if len(query_vector) != meta["dim"]:
        raise ValueError(f"query has dim {len(query_vector)}, index expects {meta['dim']}")
    # 2) Normalize it, so cosine similarity == dot product.
    query_hat = normalize(query_vector)
    # 3) Compare against every chunk.
    results = exhaustive_search(query_hat, unit_vectors, top_k)
    t2 = time.perf_counter()

    print_results(results, chunks)
    n, d = len(unit_vectors), meta["dim"]
    source = "cached embedding" if cache is not None else "embedding"
    print(f"\n[{source} {1000 * (t1 - t0):.0f} ms | exhaustive search over {n} chunks: "
          f"{n * d:,} multiplications in {1000 * (t2 - t1):.0f} ms]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure-math exhaustive vector search, no dependencies.")
    parser.add_argument("--index", type=Path, default=Path("index_gemini.pkl"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query", help="run a single query and exit (otherwise interactive)")
    parser.add_argument("--dry-run", action="store_true",
                        help="no API call: use precomputed query vectors (no key, no network needed). "
                             f"Without --query/--interactive, runs {DRY_RUN_DEFAULT_QUERY!r}.")
    parser.add_argument("--interactive", action="store_true",
                        help="with --dry-run: choose questions from the cached list")
    parser.add_argument("--query-cache", type=Path,
                        help="with --dry-run: cache file (default: <index stem>.queries.json)")
    args = parser.parse_args()

    meta, chunks, unit_vectors = load_index(args.index)
    print(f"Loaded {len(chunks)} chunks from {len(meta['corpus_files'])} files "
          f"({meta['provider']} / {meta['model']}, dim={meta['dim']}).")

    cache = None
    if args.dry_run:
        cache = QueryCache(args.query_cache or args.index.with_suffix(".queries.json"), meta)
        print(f"Dry run: using {len(cache.texts)} precomputed query vectors from {cache.path} (no API calls).")
        if not args.query and not args.interactive:
            args.query = DRY_RUN_DEFAULT_QUERY

    if args.query:
        print(f"\nQuery: {args.query}")
        try:
            answer(args.query, meta, chunks, unit_vectors, args.top_k, cache)
        except KeyError as exc:
            raise SystemExit(exc.args[0]) from None
        return

    if cache is not None:
        print("Pick a question by number, or type one of them (empty line or 'quit' to exit):")
        print(cache.menu())
    else:
        print("Type a question (empty line or 'quit' to exit).")
    while True:
        try:
            query = input("\nQuery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query or query.lower() in {"quit", "exit", "q"}:
            break
        try:
            answer(query, meta, chunks, unit_vectors, args.top_k, cache)
        except KeyError as exc:
            print(exc.args[0])


if __name__ == "__main__":
    main()
