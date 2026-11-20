"""
build_index.py - ingest a text corpus, chunk it, embed the chunks, save to pickle.

Pipeline:
    corpus/*.txt  ->  clean text  ->  chunks (with overlap)  ->  embedding API
                  ->  index.pkl  (chunk text + metadata + raw vector, side by side)

Usage:
    python build_index.py --corpus corpus --provider gemini --out index_gemini.pkl
    python build_index.py --corpus corpus --provider openai --out index_openai.pkl
    python build_index.py --corpus corpus --provider openai --chunker langchain

The default "manual" chunker is written by hand so you can see exactly how
chunking works. The optional "langchain" chunker (RecursiveCharacterTextSplitter)
is there for comparison only.
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from embedders import DEFAULT_MODELS, PROVIDERS, embed_texts

FILE_PATTERNS = ("*.txt", "*.md")

# A "span" is a (start, end) pair of character offsets into a document.
Span = tuple[int, int]


# --------------------------------------------------------------------------- #
# 1. Reading and cleaning the corpus
# --------------------------------------------------------------------------- #


def read_corpus(corpus_dir: Path) -> list[tuple[str, str]]:
    """Return [(relative_path, cleaned_text), ...] for every text file in the folder."""
    files = sorted({p for pattern in FILE_PATTERNS for p in corpus_dir.rglob(pattern)})
    documents = []
    for path in files:
        text = clean_text(path.read_text(encoding="utf-8", errors="replace"))
        if text:
            documents.append((str(path.relative_to(corpus_dir)), text))
    return documents


def clean_text(text: str) -> str:
    """Normalize line endings and whitespace while keeping paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)  # collapse runs of spaces/tabs
    text = re.sub(r" *\n *", "\n", text)  # trim spaces around newlines
    text = re.sub(r"\n{3,}", "\n\n", text)  # at most one blank line in a row
    return text.strip()


# --------------------------------------------------------------------------- #
# 2. Manual chunking
#
# Idea: cut the document into small "units" that should not be split further
# (paragraphs; if a paragraph is too long, its lines, then its sentences;
# if a sentence is too long, fixed-size pieces). Then pack consecutive units greedily into
# chunks of at most `chunk_size` characters. Each new chunk starts with the
# last ~`overlap` characters of the previous chunk, so a fact that sits on a
# chunk border still appears whole in at least one chunk.
#
# Chunks are kept as offsets into the document, so we always know exactly
# where a passage came from.
# --------------------------------------------------------------------------- #

SEPARATORS = [
    re.compile(r"\n\s*\n"),  # blank line -> paragraph break
    re.compile(r"\n"),  # single line break (Wikipedia uses these between paragraphs)
    re.compile(r"(?<=[.!?])\s+"),  # sentence end
]


def _strip_span(text: str, start: int, end: int) -> Span:
    """Shrink a span so it has no leading/trailing whitespace."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _split_spans(text: str, start: int, end: int, separator: re.Pattern) -> list[Span]:
    """Split text[start:end] at every separator match; return non-empty spans."""
    pieces, pos = [], start
    for match in separator.finditer(text, start, end):
        pieces.append((pos, match.start()))
        pos = match.end()
    pieces.append((pos, end))
    spans = [_strip_span(text, s, e) for s, e in pieces]
    return [(s, e) for s, e in spans if e > s]


def _hard_split(text: str, start: int, end: int, size: int) -> list[Span]:
    """Last resort: fixed-size pieces, preferably cut at a space."""
    spans = []
    while end - start > size:
        cut = text.rfind(" ", start, start + size)
        if cut <= start:  # no space found -> cut mid-word
            cut = start + size
        spans.append(_strip_span(text, start, cut))
        start = cut
    spans.append(_strip_span(text, start, end))
    return [(s, e) for s, e in spans if e > s]


def _units(text: str, size: int) -> list[Span]:
    """Split into pieces of at most `size` chars, preferring the "biggest" natural break.

    Separators are tried in order: blank line (paragraph), single line break,
    sentence end. A piece that is still too long after all of them is cut at
    a fixed size.
    """

    def split(start: int, end: int, level: int) -> list[Span]:
        if end - start <= size:
            return [(start, end)]
        if level == len(SEPARATORS):
            return _hard_split(text, start, end, size)
        pieces = []
        for s, e in _split_spans(text, start, end, SEPARATORS[level]):
            pieces.extend(split(s, e, level + 1))
        return pieces

    start, end = _strip_span(text, 0, len(text))
    return split(start, end, 0) if end > start else []


def _overlap_start(text: str, chunk_start: int, chunk_end: int, overlap: int) -> int:
    """Where the next chunk should begin so it repeats ~`overlap` chars of the last one.

    The position is moved forward to the next word boundary so the overlap
    does not start in the middle of a word.
    """
    if overlap <= 0:
        return chunk_end
    pos = max(chunk_start + 1, chunk_end - overlap)
    while pos < chunk_end and not text[pos - 1].isspace():
        pos += 1
    return pos


def chunk_manual(text: str, chunk_size: int, overlap: int) -> list[Span]:
    """Greedily pack units into chunks of at most `chunk_size` characters."""
    chunks: list[Span] = []
    current: Span | None = None
    for unit_start, unit_end in _units(text, chunk_size):
        if current is None:
            current = (unit_start, unit_end)
        elif unit_end - current[0] <= chunk_size:
            current = (current[0], unit_end)  # unit still fits -> extend chunk
        else:
            chunks.append(current)  # chunk is full -> emit it
            new_start = _overlap_start(text, current[0], current[1], overlap)
            if unit_end - new_start > chunk_size:
                new_start = unit_start  # overlap + unit too long -> skip overlap
            current = (new_start, unit_end)
    if current is not None:
        chunks.append(current)
    return [_strip_span(text, s, e) for s, e in chunks]


def chunk_langchain(text: str, chunk_size: int, overlap: int) -> list[Span]:
    """Same job, done by LangChain's RecursiveCharacterTextSplitter (for comparison)."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:
        raise SystemExit("Install it first:  pip install langchain-text-splitters") from exc
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, add_start_index=True
    )
    spans = []
    for doc in splitter.create_documents([text]):
        start = doc.metadata["start_index"]
        spans.append((start, start + len(doc.page_content)))
    return spans


CHUNKERS = {"manual": chunk_manual, "langchain": chunk_langchain}


# --------------------------------------------------------------------------- #
# 3. Putting it together
# --------------------------------------------------------------------------- #


def build_chunks(documents, chunker: str, chunk_size: int, overlap: int) -> list[dict]:
    chunk_fn = CHUNKERS[chunker]
    records = []
    for source, text in documents:
        for chunk_index, (start, end) in enumerate(chunk_fn(text, chunk_size, overlap)):
            records.append(
                {
                    "id": len(records),
                    "source": source,
                    "chunk_index": chunk_index,
                    "start_char": start,
                    "end_char": end,
                    "text": text[start:end],
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk a corpus and embed it.")
    parser.add_argument("--corpus", type=Path, default=Path("corpus"), help="folder with .txt/.md files")
    parser.add_argument("--provider", choices=PROVIDERS, required=True)
    parser.add_argument("--model", help="embedding model (default: %s)" % DEFAULT_MODELS)
    parser.add_argument("--chunk-size", type=int, default=800, help="max characters per chunk")
    parser.add_argument("--overlap", type=int, default=150, help="characters shared by neighbouring chunks")
    parser.add_argument("--chunker", choices=tuple(CHUNKERS), default="manual")
    parser.add_argument("--out", type=Path, default=Path("index.pkl"))
    parser.add_argument("--dry-run", action="store_true", help="chunk only, print stats, no API calls")
    args = parser.parse_args()

    if not 0 <= args.overlap < args.chunk_size:
        parser.error("--overlap must be >= 0 and smaller than --chunk-size")
    if not args.corpus.is_dir():
        parser.error(f"corpus folder {args.corpus} does not exist")

    documents = read_corpus(args.corpus)
    if not documents:
        sys.exit(f"No {'/'.join(FILE_PATTERNS)} files found in {args.corpus}")

    chunks = build_chunks(documents, args.chunker, args.chunk_size, args.overlap)
    lengths = [len(c["text"]) for c in chunks]
    print(f"Files:  {len(documents)}")
    print(f"Chunks: {len(chunks)}  (chunker={args.chunker}, size<={args.chunk_size}, overlap={args.overlap})")
    print(f"Chunk length: avg {sum(lengths) / len(lengths):.0f}, min {min(lengths)}, max {max(lengths)} chars")
    if args.dry_run:
        print("\nFirst chunk:\n" + chunks[0]["text"])
        return

    model = args.model or DEFAULT_MODELS[args.provider]
    print(f"Embedding with {args.provider} / {model} ...")
    vectors = embed_texts(
        [c["text"] for c in chunks], args.provider, model, task="document", verbose=True
    )
    for chunk, vector in zip(chunks, vectors):
        chunk["vector"] = vector  # stored raw (NOT normalized); rag_search.py normalizes

    index = {
        "meta": {
            "provider": args.provider,
            "model": model,
            "dim": len(vectors[0]),
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
            "chunker": args.chunker,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "num_chunks": len(chunks),
            "corpus_files": [source for source, _ in documents],
        },
        "chunks": chunks,
    }
    with args.out.open("wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Vector dimension: {index['meta']['dim']}")
    print(f"Saved {len(chunks)} chunks to {args.out}")


if __name__ == "__main__":
    main()
