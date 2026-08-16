"""Data types, enums, and utility functions for SQLiteStore."""

import re
import struct
import unicodedata
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from omega.embedding_config import get_embedding_config
from omega.schema import SCHEMA_VERSION  # noqa: F401 -- re-exported

# Resolved from OMEGA_EMBEDDING_DIM at import. Must match the dimension the
# vec0 tables were created with; SQLiteStore validates that on open.
EMBEDDING_DIM = get_embedding_config().dim

# Pre-compiled regex for query deduplication (strip trailing git hashes)
_TRAILING_HASH_RE = re.compile(r"\s*-\s*[0-9a-f]{6,40}\s*$")

# Periodic TTL cleanup state
_last_cleanup: Optional[float] = None
_CLEANUP_INTERVAL = 3600  # seconds


# ---------------------------------------------------------------------------
# Query result cache — avoids re-running vector + FTS5 pipeline for repeated
# queries within a short window (e.g., surface_memories on sequential edits).
# Invalidated on any write (store/delete/update).
# ---------------------------------------------------------------------------
_QUERY_CACHE_MAX = 128
_QUERY_CACHE_TTL_S = 60  # seconds
_QUERY_CACHE_WARM_TTL_S = 300  # seconds — extended TTL for high-confidence results (#2)
_HOT_CACHE_SIZE = 50  # Top N memories by access_count to keep in-memory (#2)
_HOT_CACHE_REFRESH_S = 300  # Refresh hot cache every 5 minutes (#2)
_SESSION_CACHE_MAX = 32  # Max session_id entries in session affinity cache
_PREFETCH_CACHE_MAX = 32  # Max stem entries in prefetch cache
_TRIGRAM_FINGERPRINT_CHARS = 200  # Max chars for trigram fingerprint (#1)
_FAST_PATH_MIN_OVERLAP = 0.60  # Minimum trigram Jaccard for fast-path match (#1)
_RRF_K = 60  # Reciprocal Rank Fusion constant (Cormack et al., 2009)

# Regex for content canonicalization (#6)
_MARKDOWN_STRIP_RE = re.compile(r'[*#`~\[\]()>|_]')
_WHITESPACE_COLLAPSE_RE = re.compile(r'\s+')

# Pre-compiled regex for query decomposition (moved from _decompose_query)
_CONJUNCTION_PATTERN = re.compile(
    r",?\s+(?:and\s+(?:also\s+)?|as\s+well\s+as\s+|also\s+)"
    r"(?!(?:the|a|an|in|on|at|to|of|by|or|if|is|was|were|are|it|its)\s)",
    re.IGNORECASE,
)
_CLAUSE_STARTS = re.compile(
    r"^(?:what|which|when|where|who|how|why|did|do|does|"
    r"was|were|is|are|has|have|will|can|should|tell|show|find|list)\b",
    re.IGNORECASE,
)

# FTS5 rebuild rate-limit guard
_last_fts_rebuild: Optional[float] = None
_FTS_REBUILD_INTERVAL = 3600  # seconds


# ---------------------------------------------------------------------------
# Surfacing context (#4) — dynamic threshold profiles
# ---------------------------------------------------------------------------

class SurfacingContext(Enum):
    """Context in which memories are being surfaced."""
    GENERAL = "general"
    ERROR_DEBUG = "error_debug"
    FILE_EDIT = "file_edit"
    SESSION_START = "session_start"
    PLANNING = "planning"
    REVIEW = "review"

# Thresholds per context: (min_vec_similarity, min_text_relevance, min_composite_score, context_weight_boost)
_SURFACING_THRESHOLDS = {
    SurfacingContext.GENERAL:       (0.50, 0.35, 0.10, 1.0),
    SurfacingContext.ERROR_DEBUG:   (0.40, 0.45, 0.08, 1.0),
    SurfacingContext.FILE_EDIT:     (0.50, 0.35, 0.10, 2.0),
    SurfacingContext.SESSION_START: (0.45, 0.40, 0.10, 1.0),
    SurfacingContext.PLANNING:     (0.45, 0.40, 0.10, 1.5),
    SurfacingContext.REVIEW:       (0.45, 0.40, 0.10, 1.5),
}


# ---------------------------------------------------------------------------
# Query intent (#5) — adaptive retrieval budget
# ---------------------------------------------------------------------------

class QueryIntent(Enum):
    """Classified intent for adaptive phase weighting."""
    FACTUAL = "factual"
    CONCEPTUAL = "conceptual"
    NAVIGATIONAL = "navigational"

# Intent weights: (vec, text, word_overlap, context, graph)
_INTENT_WEIGHTS = {
    QueryIntent.FACTUAL:      (0.3, 1.5, 1.8, 1.0, 1.0),
    QueryIntent.CONCEPTUAL:   (1.8, 0.5, 0.3, 1.0, 1.0),
    QueryIntent.NAVIGATIONAL: (0.1, 2.0, 2.0, 0.5, 0.3),
}


# ---------------------------------------------------------------------------
# Content canonicalization (#6)
# ---------------------------------------------------------------------------

def _canonicalize(text: str) -> str:
    """Canonicalize text for better matching: NFKC normalize, strip markdown, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = _MARKDOWN_STRIP_RE.sub(" ", text)
    text = _WHITESPACE_COLLAPSE_RE.sub(" ", text).strip()
    return text.lower()


def _trigram_fingerprint(text: str) -> frozenset:
    """Compute character-level trigram fingerprint for fast-path lookup (#1)."""
    canonical = _canonicalize(text[:_TRIGRAM_FINGERPRINT_CHARS])
    if len(canonical) < 3:
        return frozenset()
    return frozenset(canonical[i:i+3] for i in range(len(canonical) - 2))


def _trigram_jaccard(fp_a: frozenset, fp_b: frozenset) -> float:
    """Jaccard similarity between two trigram fingerprints."""
    if not fp_a or not fp_b:
        return 0.0
    intersection = len(fp_a & fp_b)
    union = len(fp_a | fp_b)
    return intersection / union if union > 0 else 0.0



def _serialize_f32(vector: List[float]) -> bytes:
    """Serialize a float32 vector to bytes for sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)


def _deserialize_f32(data: bytes, dim: int = EMBEDDING_DIM) -> List[float]:
    """Deserialize bytes to a float32 vector."""
    return list(struct.unpack(f"{dim}f", data))


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors. Pure Python, no numpy."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# MemoryResult -- lightweight result object matching MemoryNode interface
# ---------------------------------------------------------------------------


class MemoryResult:
    """Lightweight result object that matches the MemoryNode interface used by bridge.py."""

    __slots__ = (
        "id",
        "content",
        "metadata",
        "created_at",
        "access_count",
        "last_accessed",
        "ttl_seconds",
        "relevance",
        "strength",
        "embedding",
        "_content_lower",
        "valid_from",
        "valid_until",
        "derived_from",
        "source_uri",
        "status",
    )

    def __init__(
        self,
        id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None,
        access_count: int = 0,
        last_accessed: Optional[datetime] = None,
        ttl_seconds: Optional[int] = None,
        relevance: float = 0.0,
        strength: float = 0.0,
        embedding: Optional[List[float]] = None,
        valid_from: Optional[datetime] = None,
        valid_until: Optional[datetime] = None,
        derived_from: Optional[str] = None,
        source_uri: Optional[str] = None,
        status: Optional[str] = None,
    ):
        self.id = id
        self.content = content
        self.metadata = metadata or {}
        self.created_at = created_at or datetime.now(timezone.utc)
        self.access_count = access_count
        self.last_accessed = last_accessed
        self.ttl_seconds = ttl_seconds
        self.relevance = relevance
        self.strength = strength
        self.embedding = embedding
        self._content_lower = None
        self.valid_from = valid_from
        self.valid_until = valid_until
        self.derived_from = derived_from
        self.source_uri = source_uri
        self.status = status or "active"

    @property
    def content_lower(self) -> str:
        if self._content_lower is None:
            self._content_lower = self.content.lower()
        return self._content_lower

    @property
    def expires_at(self) -> Optional[datetime]:
        if self.ttl_seconds is None:
            return None
        return self.created_at + timedelta(seconds=self.ttl_seconds)

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        if self.ttl_seconds is None:
            return False
        now = now or datetime.now(timezone.utc)
        # Normalize both sides to be TZ-aware for safe comparison
        ca = self.created_at
        if ca.tzinfo is None:
            ca = ca.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return now > ca + timedelta(seconds=self.ttl_seconds)

    def touch(self) -> None:
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc)

    def time_until_expiry(self) -> Optional[timedelta]:
        if self.ttl_seconds is None:
            return None
        now = datetime.now(timezone.utc)
        exp = self.expires_at
        # Normalize to TZ-aware
        if exp is not None and exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        remaining = exp - now
        return remaining if remaining.total_seconds() > 0 else timedelta(0)
