"""
rag_search_no_deps_minimal.py - the "R" in RAG in one file, standard library only.

    python rag_search_no_deps_minimal.py

1. Load the corpus: text chunks plus one embedding vector per chunk (index_gemini.pkl).
2. Load a few questions whose embeddings were computed in advance (index_gemini.queries.json),
   so no API key or network access is needed.
3. Pick a question, compare its vector with EVERY chunk vector (cosine similarity),
   and print the best-matching chunks.

WARNING: pickle files can execute code when loaded. Only load index files you trust.
"""

import json
import math
import pickle

INDEX_FILE = "index_gemini.pkl"
QUERIES_FILE = "index_gemini.queries.json"
TOP_K = 5


# --- 1. Vector math -----------------------------------------------------------

def dot(a, b):
    """a . b = sum_i a_i * b_i"""
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    """||a|| = sqrt(a . a)"""
    return math.sqrt(dot(a, a))


def cosine_similarity(a, b):
    """cos(theta) = (a . b) / (||a|| * ||b||): 1 = same direction, 0 = unrelated."""
    return dot(a, b) / (norm(a) * norm(b))


# --- 2. Load the data ---------------------------------------------------------

# The index is a dict: {"meta": {...}, "chunks": [chunk, chunk, ...]}.
# Each chunk is a dict that holds the text AND its embedding together:
#   {"text": "...", "vector": [0.012, -0.034, ...], "source": "Deutsche_Bank.txt", ...}
with open(INDEX_FILE, "rb") as f:
    chunks = pickle.load(f)["chunks"]

# The query file holds questions embedded with the same model:
#   {"meta": {...}, "queries": [{"text": "...", "vector": [...]}, ...]}
with open(QUERIES_FILE, encoding="utf-8") as f:
    queries = json.load(f)["queries"]

print(f"Loaded {len(chunks)} chunks with {len(chunks[0]['vector'])}-dimensional embeddings.\n")


# --- 3. Pick a question -------------------------------------------------------

print("Available pre-computed queries:")
for number, query in enumerate(queries, start=1):
    print(f"  {number:2d}. {query['text']}")

choice = int(input("\nPick a number: "))
query = queries[choice - 1]
print(f"\nQuery: {query['text']}")


# --- 4. Exhaustive search: score every chunk, keep the best -------------------

scored = []
for chunk in chunks:
    score = cosine_similarity(query["vector"], chunk["vector"])
    scored.append((score, chunk))  # the score stays paired with its chunk (and so its text)

scored.sort(key=lambda pair: pair[0], reverse=True)


# --- 5. Show the top matches --------------------------------------------------

for rank, (score, chunk) in enumerate(scored[:TOP_K], start=1):
    print(f"\n#{rank}  score={score:.4f}  {chunk['source']} (chunk {chunk['chunk_index']})")
    print("-" * 80)
    print(chunk["text"])
