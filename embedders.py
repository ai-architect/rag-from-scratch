"""
embedders.py - thin wrapper around the Gemini and OpenAI embedding APIs.

This is the only module that talks to the network. It turns a list of strings
into a list of plain Python lists of floats (one vector per string). No vector
math happens here; that lives in rag_search.py.

Both build_index.py (documents) and rag_search.py (queries) use this module,
because a query must be embedded with the *same* model as the documents.
Otherwise the vectors live in different spaces and cosine similarity is
meaningless.

API keys are read from the environment:
    OpenAI : OPENAI_API_KEY
    Gemini : GEMINI_API_KEY (or GOOGLE_API_KEY)
"""

from __future__ import annotations

import os
import sys
import time
from typing import Callable

DEFAULT_MODELS = {
    "openai": "text-embedding-3-small",  # 1536 dimensions
    "gemini": "gemini-embedding-001",  # 3072 dimensions by default
}

# Max number of texts sent in one API request.
BATCH_SIZES = {
    "openai": 512,  # the API allows up to 2048 inputs; 512 keeps requests small
    "gemini": 100,  # the API allows up to 100 inputs per batch request
}

PROVIDERS = tuple(DEFAULT_MODELS)


def embed_texts(
    texts: list[str],
    provider: str,
    model: str | None = None,
    task: str = "document",
    verbose: bool = False,
) -> list[list[float]]:
    """Embed `texts` and return one list[float] per input, in the same order.

    task: "document" for corpus chunks, "query" for user questions.
          Gemini uses this to embed the two asymmetrically
          (RETRIEVAL_DOCUMENT vs. RETRIEVAL_QUERY); OpenAI ignores it.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r}; choose from {PROVIDERS}")
    if task not in ("document", "query"):
        raise ValueError("task must be 'document' or 'query'")
    model = model or DEFAULT_MODELS[provider]

    embed_batch = _openai_batch_fn(model) if provider == "openai" else _gemini_batch_fn(model, task)

    batch_size = BATCH_SIZES[provider]
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors.extend(_with_retry(lambda: embed_batch(batch)))
        if verbose:
            done = min(start + batch_size, len(texts))
            print(f"  embedded {done}/{len(texts)} chunks", file=sys.stderr)

    if len(vectors) != len(texts):
        raise RuntimeError(f"Expected {len(texts)} vectors, got {len(vectors)}")
    return vectors


# --------------------------------------------------------------------------- #
# Provider-specific code. SDKs are imported lazily, so only the one you
# actually use needs to be installed.
# --------------------------------------------------------------------------- #


def _openai_batch_fn(model: str) -> Callable[[list[str]], list[list[float]]]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install the OpenAI SDK first:  pip install openai") from exc
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Please set the OPENAI_API_KEY environment variable.")

    client = OpenAI()

    def embed_batch(batch: list[str]) -> list[list[float]]:
        response = client.embeddings.create(model=model, input=batch)
        # response.data is a list of objects with .index and .embedding
        ordered = sorted(response.data, key=lambda item: item.index)
        return [list(item.embedding) for item in ordered]

    return embed_batch


def _gemini_batch_fn(model: str, task: str) -> Callable[[list[str]], list[list[float]]]:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise SystemExit("Install the Gemini SDK first:  pip install google-genai") from exc
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Please set the GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable.")

    client = genai.Client(api_key=api_key)
    task_type = "RETRIEVAL_DOCUMENT" if task == "document" else "RETRIEVAL_QUERY"
    config = types.EmbedContentConfig(task_type=task_type)

    def embed_batch(batch: list[str]) -> list[list[float]]:
        response = client.models.embed_content(model=model, contents=batch, config=config)
        return [list(embedding.values) for embedding in response.embeddings]

    return embed_batch


def _with_retry(call: Callable[[], list[list[float]]], attempts: int = 5) -> list[list[float]]:
    """Retry transient failures (rate limits, timeouts) with exponential backoff."""
    delay = 2.0
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:  # SDK error classes differ per provider
            if attempt == attempts:
                raise
            print(
                f"  API error ({type(exc).__name__}: {exc}); retrying in {delay:.0f}s "
                f"[{attempt}/{attempts - 1}]",
                file=sys.stderr,
            )
            time.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable")
