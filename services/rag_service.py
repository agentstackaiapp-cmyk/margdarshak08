"""
services/rag_service.py
───────────────────────
Vectorless Reasoning-Based RAG (VR-RAG)

Unlike vector RAG (embeddings + cosine similarity), VR-RAG works in three steps:
  1. RETRIEVE   — BM25 keyword scoring pulls the most lexically relevant passages
                  from the selected scripture PDF(s). No GPU, no vector DB.
  2. GROUND     — The top-k passages are injected verbatim into the LLM system
                  prompt as "RETRIEVED SCRIPTURE PASSAGES".
  3. REASON     — The LLM reads those passages and reasons over them to compose
                  a contextually-grounded answer.

Scripture PDFs live in:
  backend/database/
    ├── Bhagvad_Gita/
    ├── Vedas/
    └── Puran/

DATABASE_MAP maps user-facing scripture keys → list of PDF paths.
"""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
_BASE_DIR = Path(__file__).parent.parent / "database"

# ─────────────────────────────────────────────────────────────────────────────
# DATABASE MAP  (scripture_key → list[Path])
# Populated lazily so import never crashes even if PDFs are missing.
# ─────────────────────────────────────────────────────────────────────────────
def _pdf_list(*sub_dirs: str) -> list[Path]:
    """Collect all *.pdf files under one or more sub-directories."""
    result: list[Path] = []
    for sub in sub_dirs:
        folder = _BASE_DIR / sub
        if folder.exists():
            result.extend(sorted(folder.glob("*.pdf")))
    return result


DATABASE_MAP: dict[str, list[Path]] = {
    # User-preference keys (matches models/preferences.py scriptures choices)
    "gita":        _pdf_list("Bhagvad_Gita"),
    "vedas":       _pdf_list("Vedas"),
    "upanishads":  _pdf_list("Vedas"),          # Upanishads embedded in Vedic texts
    "mahabharata": _pdf_list("Bhagvad_Gita"),   # Gita is part of Mahabharata
    "ramayana":    _pdf_list(),                 # Not yet in database
    "puran":       _pdf_list("Puran"),
    # Convenience aliases
    "bhagavad_gita": _pdf_list("Bhagvad_Gita"),
    "puranas":       _pdf_list("Puran"),
    "all": _pdf_list("Bhagvad_Gita", "Vedas", "Puran"),
}

# ─────────────────────────────────────────────────────────────────────────────
# CHUNK CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CHUNK_WORDS    = 300   # words per chunk
CHUNK_OVERLAP  = 60    # overlap between consecutive chunks (context continuity)
TOP_K          = 4     # max chunks returned by retrieve_relevant_chunks
MAX_CTX_WORDS  = 900   # hard cap on total words injected into the LLM prompt

# ─────────────────────────────────────────────────────────────────────────────
# IN-MEMORY CHUNK CACHE   pdf_path_str → list[str]
# Lives for the process lifetime — PDFs are large; we chunk once and reuse.
# ─────────────────────────────────────────────────────────────────────────────
_chunk_cache: dict[str, list[str]] = {}


# ─────────────────────────────────────────────────────────────────────────────
# PDF EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def _extract_pdf_text(pdf_path: Path) -> str:
    """
    Extract plain text from a PDF using PyMuPDF (fitz).
    Falls back gracefully if PyMuPDF is not installed.
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        pages: list[str] = []
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                pages.append(text)
        doc.close()
        full_text = "\n".join(pages)
        logger.info(
            "[VR-RAG] Extracted %d chars from %s",
            len(full_text), pdf_path.name
        )
        return full_text
    except ImportError:
        logger.error(
            "[VR-RAG] PyMuPDF not found. Install with: pip install pymupdf"
        )
        return ""
    except Exception as exc:
        logger.error("[VR-RAG] PDF extraction failed for %s: %s", pdf_path, exc)
        return ""


def _chunk_text(text: str) -> list[str]:
    """
    Split *text* into overlapping word-boundary chunks.
    Chunks preserve sentence-level context through the overlap window.
    """
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    step = max(1, CHUNK_WORDS - CHUNK_OVERLAP)
    i = 0
    while i < len(words):
        chunk_words = words[i : i + CHUNK_WORDS]
        chunk = " ".join(chunk_words).strip()
        if chunk:
            chunks.append(chunk)
        i += step

    return chunks


def _get_chunks(database_key: str) -> list[str]:
    """
    Return all text chunks for the given database key.
    Results are cached in _chunk_cache so each PDF is parsed only once
    per process lifetime.
    """
    pdf_paths = DATABASE_MAP.get(database_key.lower(), [])
    if not pdf_paths:
        logger.warning("[VR-RAG] No PDFs mapped for database key: %r", database_key)
        return []

    all_chunks: list[str] = []
    for pdf_path in pdf_paths:
        cache_key = str(pdf_path)
        if cache_key not in _chunk_cache:
            logger.info("[VR-RAG] Chunking PDF: %s", pdf_path.name)
            raw_text = _extract_pdf_text(pdf_path)
            _chunk_cache[cache_key] = _chunk_text(raw_text)
            logger.info(
                "[VR-RAG] %s → %d chunks",
                pdf_path.name, len(_chunk_cache[cache_key])
            )
        all_chunks.extend(_chunk_cache[cache_key])

    return all_chunks


# ─────────────────────────────────────────────────────────────────────────────
# BM25 SCORING  (pure Python — zero extra dependencies)
# ─────────────────────────────────────────────────────────────────────────────
def _tokenize(text: str) -> list[str]:
    """Lowercase word tokeniser that strips punctuation."""
    return re.findall(r'\b\w+\b', text.lower())


def _bm25_score(
    query_tokens: list[str],
    chunk: str,
    avg_dl: float,
    k1: float = 1.5,
    b:  float = 0.75,
) -> float:
    """
    Compute the BM25 score for *chunk* against *query_tokens*.

    BM25 formula (Okapi BM25):
        score = Σ_t  IDF(t) * (tf(t,d) * (k1+1)) / (tf(t,d) + k1*(1-b+b*|d|/avgdl))

    IDF is approximated as uniform (1.0) because we don't maintain a corpus-level
    document-frequency index — this is the "vectorless" trade-off.  BM25's TF
    saturation curve still outperforms plain TF for long documents.
    """
    chunk_tokens = _tokenize(chunk)
    dl = len(chunk_tokens)
    if dl == 0:
        return 0.0

    # Build term-frequency map for this chunk
    tf_map: dict[str, int] = {}
    for t in chunk_tokens:
        tf_map[t] = tf_map.get(t, 0) + 1

    score = 0.0
    for t in set(query_tokens):
        tf = tf_map.get(t, 0)
        if tf == 0:
            continue
        numerator   = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * dl / avg_dl)
        score += numerator / denominator   # IDF = 1.0 (uniform)

    return score


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────
def retrieve_relevant_chunks(
    query: str,
    database_key: str,
    top_k: int = TOP_K,
) -> list[str]:
    """
    VR-RAG retrieval: return the *top_k* most relevant text chunks from the
    selected scripture database using BM25 keyword scoring.

    Args:
        query:        The user's question / input text.
        database_key: Scripture key — "gita", "vedas", "puran", "all", etc.
        top_k:        Maximum number of chunks to return.

    Returns:
        List of text chunks (strings), most relevant first.
        Total word count is capped at MAX_CTX_WORDS to protect the LLM context.
    """
    chunks = _get_chunks(database_key)
    if not chunks:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    # BM25 normalisation constant
    avg_dl = sum(len(_tokenize(c)) for c in chunks) / len(chunks)

    # Score all chunks against the query
    scored: list[tuple[str, float]] = [
        (chunk, _bm25_score(query_tokens, chunk, avg_dl))
        for chunk in chunks
    ]

    # Keep only non-zero scores and sort descending
    scored = [(c, s) for c, s in scored if s > 0.0]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Assemble result, respecting the word-count budget
    result: list[str] = []
    total_words = 0
    for chunk, _score in scored[:top_k]:
        words = chunk.split()
        remaining_budget = MAX_CTX_WORDS - total_words
        if remaining_budget <= 50:
            break
        if len(words) > remaining_budget:
            result.append(" ".join(words[:remaining_budget]))
            break
        result.append(chunk)
        total_words += len(words)

    logger.info(
        "[VR-RAG] query=%r db=%r → %d chunks / %d words returned",
        query[:60], database_key, len(result), total_words
    )
    return result


def get_database_info() -> dict[str, dict]:
    """
    Return metadata about all available scripture databases.
    Used by the /api/databases endpoint so the frontend can populate
    a database-selector UI.
    """
    info: dict[str, dict] = {}
    for key, paths in DATABASE_MAP.items():
        info[key] = {
            "pdf_count": len(paths),
            "files": [p.name for p in paths],
            "available": len(paths) > 0,
        }
    return info


def warm_cache(database_key: str = "all") -> None:
    """
    Pre-load and chunk all PDFs for *database_key* at startup so the first
    user request isn't delayed by PDF parsing.  Call this from a FastAPI
    startup event.
    """
    logger.info("[VR-RAG] Warming chunk cache for database_key=%r …", database_key)
    _get_chunks(database_key)
    logger.info("[VR-RAG] Cache warm-up complete.")
