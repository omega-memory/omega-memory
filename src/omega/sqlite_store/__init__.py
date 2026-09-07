"""
OMEGA SQLite Store -- SQLite-backed storage with sqlite-vec for vector search.

Replaces the in-memory graph system (OmegaMemory) with a single SQLite database.
All nodes, embeddings, and edges live on disk. Queries use SQL + vector similarity.

RAM impact: ~5-10 MB (SQLite overhead) vs 372 MB (in-memory graphs at 3,716 nodes).

Usage:
    store = SQLiteStore()
    node_id = store.store(content="Hello world", session_id="s1")
    results = store.query("hello", limit=5)
"""

from .manager import SQLiteStore
from ._types import (
    SurfacingContext,
    QueryIntent,
    MemoryResult,
    EMBEDDING_DIM,
    SCHEMA_VERSION,
    _cosine_similarity,
    coerce_priority,
    _serialize_f32,
    _deserialize_f32,
    _canonicalize,
    _trigram_fingerprint,
    _trigram_jaccard,
    _SURFACING_THRESHOLDS,
    _INTENT_WEIGHTS,
    _HOT_CACHE_SIZE,
    _HOT_CACHE_REFRESH_S,
    _FAST_PATH_MIN_OVERLAP,
)

__all__ = [
    "SQLiteStore",
    "MemoryResult",
    "SurfacingContext",
    "QueryIntent",
    "EMBEDDING_DIM",
    "SCHEMA_VERSION",
    "_cosine_similarity",
    "coerce_priority",
    "_serialize_f32",
    "_deserialize_f32",
    "_canonicalize",
    "_trigram_fingerprint",
    "_trigram_jaccard",
    "_SURFACING_THRESHOLDS",
    "_INTENT_WEIGHTS",
    "_HOT_CACHE_SIZE",
    "_HOT_CACHE_REFRESH_S",
    "_FAST_PATH_MIN_OVERLAP",
]
