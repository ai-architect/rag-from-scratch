# RAG from Scratch — retrieval in raw math

An educational Retrieval-Augmented Generation (RAG) example. The **retrieval** part
is written with nothing but Python lists and the standard `math` module:
no NumPy, no vector database, no framework. Normalization, dot products and cosine
similarity are spelled out by hand, so you can see exactly what a vector search does.

The only external pieces are the embedding APIs (Gemini or OpenAI) that turn text into
vectors.

```
 ┌──────────── build_index.py (run once) ─────────────┐
 │ corpus/*.txt → clean → chunk (+overlap) → embed API │──► index.pkl
 └─────────────────────────────────────────────────────┘    (text + source + offsets + vector per chunk)
                                                                 │
 ┌──────────── rag_search.py (interactive) ───────────────────────▼─────────┐
 │ query → embed API (same model!) → normalize → cosine vs. EVERY chunk     │
 │       → top-k (heapq) → print passages with source file + char range      │
 └───────────────────────────────────────────────────────────────────────────┘

 Optional approximate search (no extra API calls):
   index.pkl ──► build_ann.py ──► index.ivf.pkl / index.lsh.pkl ──► rag_search_ann.py
                                  (search structure only; index.pkl is unchanged)
```

## Files

| File | Purpose |
|------|---------|
| `fetch_corpus.py` | Downloads plain-text Wikipedia banking articles into `corpus/` (standard library only). |
| `build_index.py`  | Reads `corpus/`, chunks the text by hand (or with LangChain), embeds the chunks, and writes a pickle. |
| `embedders.py`    | Small wrapper around the Gemini and OpenAI embedding APIs (batching, retries). |
| `rag_search.py`   | Loads the pickle and runs an exhaustive cosine-similarity search in pure `math`. |
| `rag_search_numpy.py` | The same search, vectorized: an `(N, d)` matrix, `M_hat @ q_hat`, `np.argpartition`. |
| `rag_search_jax.py`   | The same search in JAX: a `jax.jit`-compiled kernel with `jax.lax.top_k`; runs on a GPU or TPU if available. |
| `ivf.py`          | IVF in pure `math`: spherical k-means (k-means++ start) plus inverted lists; search probes the `nprobe` closest lists. |
| `lsh.py`          | LSH in pure `math`: random-hyperplane signatures as bucket keys, several tables, optional multi-probe. |
| `ann_index.py`    | Saves and loads IVF/LSH files and checks that each one belongs to its embeddings file. |
| `build_ann.py`    | Builds `index.ivf.pkl` or `index.lsh.pkl` from an existing `index.pkl`, without API calls. |
| `rag_search_ann.py` | Like `rag_search.py`, but searches with IVF or LSH. `--compare` shows the recall against exhaustive search. |
| `eval_ann.py`     | Measures recall@k, candidate share and speed over many queries and settings. |
| `eval_questions.txt` | 20 banking questions for `eval_ann.py --questions`. |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # openai + google-genai
pip install langchain-text-splitters       # optional, only for --chunker langchain

export OPENAI_API_KEY=...                  # if you use OpenAI
export GEMINI_API_KEY=...                  # if you use Gemini (GOOGLE_API_KEY also works)
```

## Usage

```bash
# 1. Get a corpus (see "Corpus suggestions" below)
python fetch_corpus.py

# 2. Inspect the chunking without spending API calls
python build_index.py --provider gemini --dry-run

# 3. Build an index (one per provider; the vectors are NOT interchangeable)
python build_index.py --provider gemini --out index_gemini.pkl
python build_index.py --provider openai --out index_openai.pkl
#   options: --model, --chunk-size 800, --overlap 150, --chunker manual|langchain

# 4. Search (see the next section for the three search methods)
python rag_search.py --index index_gemini.pkl
```

## Three ways to search: exhaustive, IVF, LSH

All three methods use the same `index.pkl` and print results in the same format. IVF and
LSH need one extra build step, which reads the stored vectors and makes no API calls.

| | Exhaustive | IVF | LSH |
|---|---|---|---|
| Build step | none | `build_ann.py --method ivf` | `build_ann.py --method lsh` |
| Chunks scored per query | all N | chunks in the `nprobe` closest clusters | chunks in the query's buckets |
| Exact? | yes | approximate | approximate |
| Main knob | — | `--nprobe` at search time | `--bits`/`--tables` at build time, `--probe-radius` at search time |

### A) Exhaustive search: exact, scores every chunk

```bash
python rag_search.py       --index index_gemini.pkl                    # pure math, interactive
python rag_search.py       --index index_gemini.pkl --top-k 3 --query "When was Deutsche Bank founded?"
python rag_search_numpy.py --index index_gemini.pkl                    # same, NumPy (pip install numpy)
python rag_search_jax.py   --index index_gemini.pkl                    # same, JAX (add --x64 for float64)
```

### B) IVF: cluster first, then search only the nearest clusters

```bash
# build once (k-means in pure math: ~110 s for 1,475 Gemini chunks at d=3072, ~30 s at d=1536)
python build_ann.py --index index_gemini.pkl --method ivf                 # nlist = √N ≈ 38
# -> writes index_gemini.ivf.pkl

# search
python rag_search_ann.py --index index_gemini.pkl --ann index_gemini.ivf.pkl              # nprobe=4
python rag_search_ann.py --index index_gemini.pkl --ann index_gemini.ivf.pkl --nprobe 8 \
       --compare --query "Why did Lehman Brothers collapse?"
```

### C) LSH: hash into buckets, search only the query's buckets

```bash
# build once (~20 s at d=3072); vectors are mean-centered before hashing (see below)
python build_ann.py --index index_gemini.pkl --method lsh                 # bits = log2(N/50) ≈ 5, 16 tables
python build_ann.py --index index_gemini.pkl --method lsh --bits 4 --tables 32 --out index_gemini.lsh4.pkl
# -> writes index_gemini.lsh.pkl

# search
python rag_search_ann.py --index index_gemini.pkl --ann index_gemini.lsh.pkl
python rag_search_ann.py --index index_gemini.pkl --ann index_gemini.lsh.pkl --probe-radius 1 --compare
```

`--compare` also runs the exhaustive search and prints the recall for that query, plus
any chunks the approximate search missed (example output):

```
[embedding 210 ms | IVF: probed 2/38 lists; scored 157 of 1475 chunks (10.6%), 299,520 multiplications in 17 ms]
[exhaustive: 2,265,600 multiplications in 129 ms | recall@5 = 5/5]
```

### Comparing all methods at once

```bash
python eval_ann.py --index index_gemini.pkl --ann index_gemini.ivf.pkl index_gemini.lsh.pkl
python eval_ann.py --index index_gemini.pkl --ann index_gemini.ivf.pkl index_gemini.lsh.pkl \
       --questions eval_questions.txt         # 20 real banking questions, one batched API call
```

By default `eval_ann.py` makes no API calls: it creates 50 test queries by adding noise
to random chunk vectors (`--noise 0.5`). It also prints how similar the queries are to
their best chunk, compared with two random chunks, so you can judge how hard the queries
are. `--questions` embeds real questions instead.

Measured on the real Gemini index (1,475 chunks, d = 3,072) with the synthetic queries.
The average query-to-best-chunk cosine was 0.89, against 0.70 for two random chunks:

| Setting | recall@5 | chunks scored | ms/query | speedup |
|---|---:|---:|---:|---:|
| exhaustive | 1.000 | 100% | 255 | 1.0× |
| IVF nlist=38, nprobe=1 | 0.852 | 3.6% | 15.8 | 16.1× |
| IVF nlist=38, nprobe=2 | 0.972 | 6.9% | 23.8 | 10.7× |
| IVF nlist=38, nprobe=4 | 0.996 | 12.6% | 38.5 | 6.6× |
| IVF nlist=38, nprobe=8 | 1.000 | 25.5% | 72.1 | 3.5× |
| LSH 5 bits × 16 tables, radius 0 | 0.804 | 39.8% | 118 | 2.2× |
| LSH 5 bits × 16 tables, radius 1 | 1.000 | 95.9% | 262 | 1.0× |
| LSH as above, **without** centering (`--no-center`) | 1.000 | 98.2% | 271 | 0.9× |

IVF clearly wins: with 2 of 38 lists it finds 97% of the true top-5 while scoring
7% of the chunks.

### Why LSH struggles, and why it needs mean-centering

**Problem 1: anisotropy.** Real embeddings do not spread evenly over the sphere; they
all point roughly the same way. In the Gemini index, two *unrelated* chunks still have
an average cosine of 0.70, and the mean vector has length 0.84 (a length of 1 would
mean all vectors are identical). A random hyperplane through the origin therefore puts
almost every chunk on the same side. Without centering, one 5-bit bucket held 1,132 of
the 1,475 chunks. `lsh.py` therefore hashes `v − mean` instead of `v`, which moves the
origin into the middle of the data cloud. The largest bucket then shrinks to 108. The
final ranking still uses the original vectors, so the scores do not change. Build with
`--no-center` to see the effect yourself.

**Problem 2: hyperplanes ignore the data.** A random hyperplane separates two vectors
with probability θ/π, where θ is the angle between them. Even after centering, a query
and its answer are often 50–70° apart, so each bit matches only with probability
≈ 0.65, and all 5 bits match only 0.65⁵ ≈ 12% of the time per table. To compensate you
need few bits and many tables, which makes the candidate sets large. IVF learns where
the data actually lies (k-means), whereas LSH uses directions that know nothing about
the data. That is why IVF-style indexes, and graph indexes like HNSW, dominate in
practice. LSH works well for near-duplicate detection, where cosines are above 0.9.

### Tuning rules of thumb
- **IVF**: `nlist ≈ √N`. Raise `--nprobe` until recall is good enough;
  `nprobe = nlist` is the exhaustive search again. Rebuild with a new `--seed` to see
  how much the k-means start matters.
- **LSH**: each extra bit halves the bucket size, which is faster but lowers recall.
  Each extra table adds another chance to find a neighbor, which helps recall but adds
  candidates. `--probe-radius 1` also checks all buckets one bit away, without
  rebuilding.
- **The IVF/LSH file is tied to its `index.pkl`.** It stores a fingerprint (model, chunk
  count, build time), and the search programs refuse a mismatched pair. After rebuilding
  `index.pkl`, rebuild the IVF/LSH file too.

### Exhaustive search engines: pure math vs. NumPy vs. JAX

All three programs read the same index and return the same top-k chunks. Scores differ
by less than 1e-7, because JAX uses float32 by default.

| Engine | How it searches |
|---|---|
| `rag_search.py` | Python loop: N dot products of length d |
| `rag_search_numpy.py` | one BLAS matrix-vector product `M_hat @ q_hat`, then `np.argpartition` |
| `rag_search_jax.py` | one XLA-compiled kernel (`jax.jit`, `jax.lax.top_k`), compiled once at startup, running on CPU or GPU |

**Benchmark.** Median search time per query in milliseconds, for d = 1,536 and top-5,
excluding the embedding API call (which takes hundreds of ms anyway). The first row is
the real Wikipedia corpus size; the larger rows use random unit vectors. For every
size, JAX on the GPU returned the same top-5 as NumPy. Measured on a 12-core CPU and an
NVIDIA RTX 5090 with `jax[cuda12]`:

| Chunks (N) | pure math | NumPy (f64) | JAX CPU (f32) | JAX GPU (f32) | GPU kernel only |
|---:|---:|---:|---:|---:|---:|
| 1,475 | 237 | **0.52** | 0.54 | 0.59 | 0.09 |
| 10,000 | 1,772 | 5.42 | 2.93 | **0.64** | 0.11 |
| 100,000 | — | 49.7 | 27.3 | **1.07** | 0.64 |
| 1,000,000 | — | 498 | 268 | **5.19** | 4.52 |

"GPU kernel only" counts just the computation on the GPU. The rest of the GPU time is
the fixed cost of copying the query to the GPU, launching the kernel and copying the
results back.

What the numbers show:
- **Pure math → NumPy is about 450× faster** at the same algorithm: the speedup comes
  entirely from leaving the Python interpreter.
- **NumPy and JAX CPU scale linearly with N.** JAX CPU is about 2× faster at large N,
  thanks to float32 (half the memory traffic) and ahead-of-time compilation.
- **The GPU has a fixed cost of about 0.5 ms**, so it only wins once the work is large
  enough. At 1M chunks it is about 95× faster than NumPy.
- **Moving data to the GPU efficiently matters.** `jnp.asarray(python_list)` took
  0.9 ms per query on its own; `jax.device_put(np.asarray(...))` (see `to_device` in
  `rag_search_jax.py`) takes 0.3 ms. Returning scores and indices with one
  `jax.device_get` saves another synchronization. Together these cut the GPU time at
  N = 1,475 from 1.37 ms to 0.59 ms.

Exact timings vary from run to run (the pure-math column varied between 120 and 240 ms
across runs), but the ratios stay stable.

Defaults: OpenAI `text-embedding-3-small` (1536 dims), Gemini `gemini-embedding-001`
(3072 dims). The index records which provider and model built it, and `rag_search.py`
always embeds the query with that same model.

Each result shows the score, source file, chunk number and character range, then the
passage itself (the score below is illustrative):

```
#1  score=0.7412  Deutsche_Bank.txt#chunk3  chars 1830-2611
----------------------------------------------------------------
  Deutsche Bank was founded in Berlin in 1870 as a specialist bank for ...
```

## The math in 30 seconds

For vectors `a, b ∈ ℝᵈ`:

| Quantity | Formula | Function in `rag_search.py` |
|---|---|---|
| Dot product | `a·b = Σᵢ aᵢ bᵢ` | `dot` (uses `math.fsum`) |
| L2 norm | `‖a‖ = √(a·a)` | `norm` |
| Normalization | `â = a / ‖a‖` (so `‖â‖ = 1`) | `normalize` |
| Cosine similarity | `cos θ = a·b / (‖a‖ ‖b‖)` ∈ [-1, 1] | `cosine_similarity` |

**Why normalize once?** For unit vectors the denominator is 1, so `cos θ = â·b̂`.
`rag_search.py` normalizes every chunk vector once at load time and the query once per
search. After that, each comparison is a plain dot product: `d` multiplications.
Searching `N` chunks therefore costs `N·d` multiplications, which the program prints after
every query. That number is what IVF, LSH and similar indexes try to reduce.

**Chunking.** The manual chunker splits text at the most natural break that keeps a
piece under `--chunk-size`: blank line, then line break, then sentence end, then a
fixed-size cut. It then packs consecutive pieces greedily into chunks and repeats about
`--overlap` characters from the end of one chunk at the start of the next. Every chunk
keeps its exact character offsets into the source file. Known limitation: the
sentence splitter is a simple regex and treats abbreviations like "U.S." as sentence ends.

> ⚠️ Pickle files can execute code when loaded. Only load index files you built yourself.

## Corpus suggestions

1. **Wikipedia banking articles (recommended).** The text is public (CC BY-SA 4.0),
   clean, and the articles overlap in topic: many banks, the same crisis, the same
   regulators. That overlap makes "near miss" results interesting to study.
   `python fetch_corpus.py` downloads these 16 articles (about 800k characters, roughly
   1,500 chunks with the default settings):
   JPMorgan Chase, Deutsche Bank, Goldman Sachs, Morgan Stanley, HSBC, Barclays,
   BNP Paribas, UBS, Credit Suisse, Lehman Brothers, Federal Reserve,
   European Central Bank, Basel III, Fractional-reserve banking, Investment banking,
   2008 financial crisis.
   Add your own titles: `python fetch_corpus.py Commerzbank Citigroup Bank_run`.
2. **Project Gutenberg** (public domain), for longer, older prose:
   - Walter Bagehot, *Lombard Street: A Description of the Money Market* (1873), a
     classic on central banking and the "lender of last resort".
   - Adam Smith, *The Wealth of Nations*, especially Book II on money and banking.
   Download the "Plain Text UTF-8" version into `corpus/`. You may want to remove the
   Gutenberg license header and footer first.
3. **Other public sources**: Federal Reserve publications and speeches (US government
   works), Basel Committee / BIS documents, ECB publications, and SEC EDGAR 10-K annual
   reports of the banks above. These are usually PDF or HTML, so convert them to `.txt`
   first.

Some good test queries: *"When was Deutsche Bank founded?"*, *"Why did Lehman Brothers
collapse?"*, *"What does Basel III require for capital ratios?"*, *"Which bank took over
Credit Suisse?"*, *"How does fractional-reserve banking create money?"*

## Suggested extensions

Done so far:
- ~~NumPy version~~ and ~~JAX version~~: see `rag_search_numpy.py` and `rag_search_jax.py`.
- ~~IVF~~ and ~~LSH~~: see `ivf.py`, `lsh.py`, `build_ann.py`, `rag_search_ann.py` and
  `eval_ann.py`.

Not implemented yet:
- **Batched queries**: many queries in one matrix-matrix product (`Q @ M_hat.T`, or
  `jax.vmap`), benchmarked on a GPU with large synthetic N.
- **IVF/LSH in NumPy or JAX**: the `.ivf.pkl`/`.lsh.pkl` files convert directly to
  arrays: centroids become an `(nlist, d)` matrix, and hyperplanes become a
  `(tables, bits, d)` tensor, so a key is `(planes @ q >= 0)` packed into bits. k-means
  in NumPy would also cut the IVF build time from ~30 s to under a second.
- **Product quantization (PQ)** for compressing vectors, and **HNSW** graphs as the
  approach most production vector databases use.
- **The "G" in RAG**: send the top-k passages plus the question to an LLM (Gemini, GPT,
  Claude) with a prompt like "answer only from these sources and cite them".
- **Hybrid search**: combine BM25 keyword scores (also implementable from scratch) with
  cosine scores, then optionally re-rank the results with a cross-encoder.
- **Evaluation**: a small hand-labelled set of question → relevant chunk pairs, scored
  with recall@k and MRR. Compare chunk sizes, overlap, manual vs. LangChain chunking,
  and Gemini vs. OpenAI.
- **Better storage**: JSON or SQLite instead of pickle (safer and inspectable), plus
  caching embeddings by a hash of the chunk content so rebuilds only embed new chunks.
- **Reduced dimensions**: OpenAI `dimensions=` and Gemini `output_dimensionality` let you
  study how much quality you lose at 256 or 768 dimensions in exchange for speed.

## Performance

