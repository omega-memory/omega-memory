"""
OMEGA Bridge -- High-level API for OMEGA memory system.

Provides the public interface used by the MCP server handlers.
All functions are thin wrappers that delegate to the SQLiteStore singleton.

Public API (36 functions, see __all__ for full list):
    Core:       auto_capture, store, remember, delete_memory, edit_memory
    Query:      query, query_structured, phrase_search, find_similar_memories
    Session:    welcome, clear_session, batch_store
    Health:     check_health, status, get_dedup_stats
    Profile:    get_profile, save_profile, extract_preferences, list_preferences
    Lessons:    get_cross_session_lessons, get_cross_project_lessons
    Maintenance: consolidate, compact, deduplicate, timeline, traverse
    Export:     export_memories, import_memories, reingest
    Stats:      type_stats, session_stats
    Constraints: check_constraints, list_constraints, save_constraints
    Feedback:   record_feedback
    Testing:    reset_memory
"""

import atexit
import logging
import os
import re
import threading
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from omega import json_compat as json
from omega.dedup_config import load_dedup_thresholds
from omega.exceptions import ValidationError
from omega.llm import llm_complete  # noqa: F401 — used in distill_trajectory, module-level for test patchability
from omega.types import TTLCategory, AutoCaptureEventType

logger = logging.getLogger("omega.bridge")


# ---------------------------------------------------------------------------
# Storage configuration
# ---------------------------------------------------------------------------

OMEGA_HOME = Path(os.environ.get("OMEGA_HOME", str(Path.home() / ".omega")))

# Per-event-type dedup thresholds for Jaccard similarity. Environment
# overrides are validated once at process import; restart after changing them.
DEDUP_THRESHOLDS: Dict[str, float] = load_dedup_thresholds()

# Event types that participate in memory evolution (Zettelkasten-style).
EVOLUTION_TYPES = {
    AutoCaptureEventType.LESSON_LEARNED,
    AutoCaptureEventType.DECISION,
    AutoCaptureEventType.ERROR_PATTERN,
    AutoCaptureEventType.CONSTRAINT,
    AutoCaptureEventType.SKILL_TEMPLATE,
    AutoCaptureEventType.PROJECT_STATUS,
    AutoCaptureEventType.ADVISOR_INSIGHT,
    AutoCaptureEventType.SESSION_SUMMARY,
}
EVOLUTION_THRESHOLD = 0.65

# Content blocklist — reject system noise at ingestion time.
# Startswith patterns (checked against content[:50])
_BLOCKLIST_STARTSWITH = [
    "[BROADCAST",
    "[WORK BREADCRUMB",
    "[WORK DISPATCH",
    "<task-notification>",
    "Decision: <task-notification>",
]
# Substring patterns (checked anywhere in content)
_BLOCKLIST_CONTAINS = [
    '"error":',
    '"stderr":',
    '"stdout":',
    "[BROADCAST",
]

# Minimum content length for auto-capture (reject very short noise).
# Raised from 40 to 80 to filter infrastructure noise that inflates never-accessed count.
_MIN_CONTENT_LENGTH = 80

# Event types from hooks that generate infrastructure noise (never accessed, inflate DB).
_INFRASTRUCTURE_EVENT_TYPES = frozenset({
    "consolidate", "compact", "checkpoint", "coordination_snapshot",
    "session_respawn", "file_summary", "code_chunk",
})


def _check_milestone(name: str) -> bool:
    """Return True if milestone not yet achieved (first time). Creates marker.

    DEPRECATED: Use omega.milestones._check_milestone instead.
    Kept as thin redirect for any callers that import from bridge.
    """
    from omega.milestones import _check_milestone as _cm
    return _cm(name)


# ---------------------------------------------------------------------------
# Lazy singleton -- SQLiteStore replaces OmegaMemory
# ---------------------------------------------------------------------------

_store_instance = None
_store_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Bridge initialization options
# ---------------------------------------------------------------------------

_bridge_enable_vector_search: bool = False
_bridge_onnx_model_path: Optional[str] = None


def initialize_bridge(
    enable_vector_search: bool = False,
    onnx_model_path: Optional[str] = None,
) -> None:
    """Configure bridge options before first store access.

    Must be called before any store operation if non-default options are needed.

    Args:
        enable_vector_search: If True, pass the ONNX embed model to SQLiteStore
            to enable offline semantic (vector) search via sqlite-vec.
        onnx_model_path: Path to the ONNX embedding model file.  When *None*
            the store will use its own default path (if any).
    """
    global _bridge_enable_vector_search, _bridge_onnx_model_path
    _bridge_enable_vector_search = enable_vector_search
    _bridge_onnx_model_path = onnx_model_path
    logger.debug(
        "Bridge configured: enable_vector_search=%s, onnx_model_path=%s",
        enable_vector_search,
        onnx_model_path,
    )


def _get_store():
    """Get or create the SQLiteStore singleton (thread-safe).

    Uses local variable for init to ensure _store_instance is only set
    after full initialization (migration + cleanup + atexit registration).
    """
    global _store_instance
    if _store_instance is not None:
        return _store_instance
    with _store_lock:
        if _store_instance is not None:
            return _store_instance
        # Auto-migrate from JSON graphs if needed (first run after upgrade)
        from omega.migrate_to_sqlite import auto_migrate_if_needed

        auto_migrate_if_needed()

        from omega.sqlite_store import SQLiteStore

        # Build keyword arguments based on bridge configuration.
        store_kwargs: Dict[str, Any] = {}
        if _bridge_enable_vector_search:
            store_kwargs["onnx_model_path"] = _bridge_onnx_model_path
            logger.info(
                "Vector search enabled; ONNX model path: %s",
                _bridge_onnx_model_path or "<store default>",
            )

        # Init into local var first; only publish to global after full setup
        store = SQLiteStore(**store_kwargs)
        # Purge expired nodes on startup
        expired = store.cleanup_expired()
        if expired > 0:
            logger.info(f"Startup: purged {expired} expired nodes")
        atexit.register(_close_store)
        _store_instance = store  # Publish only after full init
    return _store_instance


def _close_store():
    """Close SQLiteStore on process exit."""
    global _store_instance
    if _store_instance is not None:
        try:
            _store_instance.close()
        except Exception as e:
            logger.debug("Store close failed during refresh: %s", e)


def reset_memory():
    """Reset the singleton (useful for testing)."""
    global _store_instance
    if _store_instance is not None:
        try:
            _store_instance.close()
        except Exception as e:
            logger.debug("Store close failed during reset: %s", e)
    _store_instance = None
    _welcome_cache.clear()


def semantic_search(
    query: str,
    top_k: int = 10,
    project: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Perform offline semantic (vector) search using the ONNX embed model.

    Requires the bridge to have been initialised with
    ``enable_vector_search=True`` and the underlying SQLiteStore to support
    the ``semantic_search`` method (sqlite-vec extension + ONNX runtime).

    Args:
        query:   Natural-language query string.
        top_k:   Maximum number of results to return.
        project: Optional project filter applied before vector ranking.

    Returns:
        List of memory dicts ordered by descending similarity score, each
        containing at least ``{id, content, score}``.

    Raises:
        RuntimeError: If vector search is not enabled or not supported by
            the current store backend.
    """
    if not _bridge_enable_vector_search:
        raise RuntimeError(
            "semantic_search requires enable_vector_search=True passed to "
            "initialize_bridge() before the store is created."
        )
    store = _get_store()
    if not hasattr(store, "semantic_search"):
        raise RuntimeError(
            "The current SQLiteStore backend does not expose semantic_search. "
            "Ensure sqlite-vec and onnxruntime are installed."
        )
    kwargs: Dict[str, Any] = {"query": query, "top_k": top_k}
    if project is not None:
        kwargs["project"] = project
    return store.semantic_search(**kwargs)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _human_ttl(ttl: Optional[int]) -> str:
    """Format TTL seconds as human-readable string."""
    if not ttl:
        return "permanent"
    if ttl < 3600:
        return f"{ttl // 60}m"
    if ttl < 86400:
        return f"{ttl // 3600}h"
    return f"{ttl // 86400}d"


def _normalize_for_dedup(text: str) -> str:
    """Normalize text for dedup comparison by stripping variable parts."""
    t = text.lower()
    t = re.sub(r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}", "<ID>", t)
    t = re.sub(r"/[\w/.\-]+\.\w{1,5}", "<PATH>", t)
    t = re.sub(r"'[^']{1,80}'", "<NAME>", t)
    t = re.sub(r'"[^"]{1,80}"', "<NAME>", t)
    t = re.sub(r"\b\d+\b", "<N>", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


_TAG_LANGUAGES = {
    "python",
    "javascript",
    "typescript",
    "rust",
    "go",
    "java",
    "ruby",
    "swift",
    "kotlin",
    "c++",
    "c#",
    "php",
    "sql",
    "bash",
    "zsh",
    "html",
    "css",
}
_TAG_TOOLS = {
    "react",
    "next.js",
    "nextjs",
    "django",
    "flask",
    "fastapi",
    "docker",
    "kubernetes",
    "git",
    "npm",
    "pip",
    "pytest",
    "webpack",
    "vite",
    "redis",
    "postgres",
    "sqlite",
    "mongodb",
    "aws",
    "gcp",
    "azure",
    "vercel",
    "nginx",
    "mysql",
    "jest",
    "vitest",
    "yarn",
    "pnpm",
    "bun",
    "deno",
    "supabase",
    "onnx",
    "mcp",
    "asyncio",
    "threading",
    "sqlalchemy",
    "celery",
    "graphql",
    "prisma",
    "terraform",
    "ansible",
    "helm",
    "zustand",
    "tailwind",
    "shadcn",
    "storybook",
    "playwright",
    "cypress",
    "openai",
    "anthropic",
    "langchain",
    "chromadb",
    "pinecone",
    "homebrew",
    "launchd",
    "systemd",
    "cron",
}
_TAG_ALIASES = {
    "postgresql": "postgres",
    "k8s": "kubernetes",
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "rb": "ruby",
    "tf": "terraform",
    "cdk": "aws-cdk",
    "nextjs": "next.js",
    "reactjs": "react",
    "sqlite3": "sqlite",
    "onnxruntime": "onnx",
    "fts5": "sqlite",
    "sqlitevec": "sqlite",
}
_GO_CONTEXT_WORDS = {"goroutine", "goroutines", "chan", "defer", "func", "gomod", "gofmt"}


_TAG_CONCEPTS = {
    "hook",
    "hooks",
    "daemon",
    "migration",
    "api",
    "config",
    "configuration",
    "testing",
    "test",
    "tests",
    "debug",
    "debugging",
    "performance",
    "cache",
    "caching",
    "auth",
    "authentication",
    "authorization",
    "deploy",
    "deployment",
    "refactor",
    "refactoring",
    "schema",
    "embedding",
    "embeddings",
    "vector",
    "coordination",
    "concurrency",
    "async",
    "sync",
}

# Map concept words to canonical tags
_CONCEPT_CANONICAL = {
    "hooks": "hook",
    "tests": "testing",
    "test": "testing",
    "debugging": "debug",
    "caching": "cache",
    "authentication": "auth",
    "authorization": "auth",
    "deployment": "deploy",
    "refactoring": "refactor",
    "configuration": "config",
    "embeddings": "embedding",
}


def _extract_tags(content: str, project: Optional[str] = None) -> List[str]:
    """Extract auto-tags from content (languages, tools, file paths, concepts, project)."""
    tags: set = set()
    words = set(re.findall(r"\b[\w.+#]+\b", content.lower()))
    # Apply aliases first (e.g. "postgresql" -> "postgres")
    for alias, canonical in _TAG_ALIASES.items():
        if alias in words:
            tags.add(canonical)
    # Languages (with Go disambiguation)
    for w in words:
        if w in _TAG_LANGUAGES:
            if w == "go":
                # Only tag "go" if Go-specific context words are present
                if words & _GO_CONTEXT_WORDS:
                    tags.add(w)
            else:
                tags.add(w)
    tags.update(w for w in words if w in _TAG_TOOLS)
    # Concepts (hook, testing, auth, etc.)
    for w in words:
        if w in _TAG_CONCEPTS:
            tags.add(_CONCEPT_CANONICAL.get(w, w))
    # File paths
    for match in re.findall(r"(?:/[\w.\-]+){2,}", content):
        tags.add(match)
    # File extensions mentioned inline (e.g. ".py", ".ts")
    for ext in re.findall(r"\b\w+\.(py|js|ts|tsx|rs|go|rb|java|swift|sql|sh|yaml|json|toml)\b", content.lower()):
        ext_map = {
            "py": "python",
            "js": "javascript",
            "ts": "typescript",
            "tsx": "typescript",
            "rs": "rust",
            "rb": "ruby",
            "sh": "bash",
        }
        if ext in ext_map:
            tags.add(ext_map[ext])
    # Project name
    if project:
        tags.add(Path(project).name.lower())
    return sorted(tags)[:10]


def _infer_temporal_range(query_text: str) -> Optional[tuple]:
    """Infer a (start_iso, end_iso) temporal range from natural-language time references.

    Supports: "last week", "yesterday", "N days/hours ago", "today",
    "this week/month/year", month names, day-of-week references,
    "the week/month of <date>", ISO dates.
    Returns None if no temporal signal is detected.
    """
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    text = query_text.lower()

    # "last N days/hours/weeks/months/years"
    m = re.search(r"last\s+(\d+)\s+(day|hour|week|month|year)s?", text)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        delta = {
            "day": timedelta(days=n),
            "hour": timedelta(hours=n),
            "week": timedelta(weeks=n),
            "month": timedelta(days=n * 30),
            "year": timedelta(days=n * 365),
        }[unit]
        return ((now - delta).isoformat(), now.isoformat())

    # "N days/hours ago"
    m = re.search(r"(\d+)\s+(day|hour|week|month|year)s?\s+ago", text)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        delta = {
            "day": timedelta(days=n),
            "hour": timedelta(hours=n),
            "week": timedelta(weeks=n),
            "month": timedelta(days=n * 30),
            "year": timedelta(days=n * 365),
        }[unit]
        return ((now - delta).isoformat(), now.isoformat())

    if "yesterday" in text:
        start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0)
        end = start + timedelta(days=1)
        return (start.isoformat(), end.isoformat())

    if "today" in text:
        start = now.replace(hour=0, minute=0, second=0)
        return (start.isoformat(), now.isoformat())

    if "last week" in text:
        # Previous Mon-Sun week
        days_since_monday = now.weekday()
        last_monday = now - timedelta(days=days_since_monday + 7)
        start = last_monday.replace(hour=0, minute=0, second=0)
        end = start + timedelta(days=7)
        return (start.isoformat(), end.isoformat())

    if "this week" in text:
        days_since_monday = now.weekday()
        start = (now - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0)
        return (start.isoformat(), now.isoformat())

    if "last month" in text:
        # Previous calendar month
        first_this_month = now.replace(day=1, hour=0, minute=0, second=0)
        end = first_this_month
        if now.month == 1:
            start = datetime(now.year - 1, 12, 1, tzinfo=timezone.utc)
        else:
            start = datetime(now.year, now.month - 1, 1, tzinfo=timezone.utc)
        return (start.isoformat(), end.isoformat())

    if "this month" in text:
        start = now.replace(day=1, hour=0, minute=0, second=0)
        return (start.isoformat(), now.isoformat())

    if "this year" in text:
        start = datetime(now.year, 1, 1, tzinfo=timezone.utc)
        return (start.isoformat(), now.isoformat())

    if "last year" in text:
        start = datetime(now.year - 1, 1, 1, tzinfo=timezone.utc)
        end = datetime(now.year, 1, 1, tzinfo=timezone.utc)
        return (start.isoformat(), end.isoformat())

    # Day-of-week references: "last Monday", "on Friday", etc.
    _DAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
             "friday": 4, "saturday": 5, "sunday": 6}
    for day_name, day_num in _DAYS.items():
        if day_name in text:
            # Find the most recent occurrence of this day
            days_ago = (now.weekday() - day_num) % 7
            if days_ago == 0:
                days_ago = 7  # "last Monday" means previous, not today
            target = now - timedelta(days=days_ago)
            start = target.replace(hour=0, minute=0, second=0)
            end = start + timedelta(days=1)
            return (start.isoformat(), end.isoformat())

    # "Month YYYY" or "in Month YYYY" (e.g., "January 2025")
    months = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }
    for name, num in months.items():
        # Check for "Month YYYY" first
        m = re.search(rf"\b{name}\s+(\d{{4}})\b", text)
        if m:
            year = int(m.group(1))
            start = datetime(year, num, 1, tzinfo=timezone.utc)
            if num == 12:
                end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            else:
                end = datetime(year, num + 1, 1, tzinfo=timezone.utc)
            return (start.isoformat(), end.isoformat())
        # Bare month name (assume most recent occurrence)
        if name in text:
            year = now.year if num <= now.month else now.year - 1
            start = datetime(year, num, 1, tzinfo=timezone.utc)
            if num == 12:
                end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            else:
                end = datetime(year, num + 1, 1, tzinfo=timezone.utc)
            return (start.isoformat(), end.isoformat())

    # ISO date (YYYY-MM-DD)
    m = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    if m:
        date_str = m.group(1)
        start = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        return (start.isoformat(), end.isoformat())

    # Bare year reference (e.g., "in 2024")
    m = re.search(r"\bin\s+(20\d{2})\b", text)
    if m:
        year = int(m.group(1))
        start = datetime(year, 1, 1, tzinfo=timezone.utc)
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        return (start.isoformat(), end.isoformat())

    return None


def _relative_time(created_at) -> str:
    """Format a datetime as a human-readable relative time string."""
    if not created_at:
        return ""
    now = datetime.now(timezone.utc)
    if isinstance(created_at, str):
        try:
            if created_at.endswith("Z"):
                created_at = created_at[:-1] + "+00:00"
            created_at = datetime.fromisoformat(created_at)
        except (ValueError, TypeError):
            return ""
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    delta = now - created_at
    seconds = delta.total_seconds()
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        m = int(seconds // 60)
        return f"{m}m ago"
    if seconds < 86400:
        h = int(seconds // 3600)
        return f"{h}h ago"
    days = int(seconds // 86400)
    if days == 1:
        return "yesterday"
    if days < 30:
        return f"{days}d ago"
    months = days // 30
    if months == 1:
        return "1 month ago"
    return f"{months} months ago"


def _extract_facts(content: str) -> List[str]:
    """Extract atomic facts from content for multi-key retrieval (no LLM).

    Extracts:
    - Technical terms (CamelCase, UPPER_CASE, dotted.paths)
    - Quoted strings and backtick-delimited tokens
    - Decision verbs with their objects ("chose X", "switched to Y")
    - Key noun phrases from short sentences

    Returns deduplicated list of fact strings, capped at 20.
    """
    facts: set = set()

    # 1. CamelCase identifiers (e.g., SQLiteStore, MemoryResult)
    # Match words starting with uppercase that have at least one lower-to-upper transition
    for m in re.findall(r"\b([A-Z][a-zA-Z]*[a-z][A-Z][a-zA-Z]*)\b", content):
        facts.add(m.lower())

    # 2. UPPER_CASE constants (e.g., MAX_NODES, API_KEY)
    for m in re.findall(r"\b([A-Z][A-Z0-9_]{2,})\b", content):
        facts.add(m.lower())

    # 3. Backtick-delimited tokens (e.g., `jwt`, `sqlite_store.py`)
    for m in re.findall(r"`([^`]{2,40})`", content):
        facts.add(m.lower().strip())

    # 4. Quoted strings (e.g., "refresh token", 'auth method')
    for m in re.findall(r"""["']([^"']{2,40})["']""", content):
        facts.add(m.lower().strip())

    # 5. Decision/action verb phrases — extract the object of key verbs
    _DECISION_VERBS = (
        r"(?:chose|choose|decided|switched?\s+to|migrated?\s+to|"
        r"replaced?\s+with|use[ds]?|adopted?|selected?|"
        r"implemented?|configured?|set\s+up|enabled?|disabled?)"
    )
    for m in re.finditer(
        rf"\b{_DECISION_VERBS}\s+([A-Za-z0-9][\w\s./-]{{1,30}}?)(?:[.,;!?\n]|$)",
        content, re.IGNORECASE,
    ):
        phrase = m.group(1).strip().rstrip(".")
        if len(phrase) > 2:
            facts.add(phrase.lower())

    # 6. Dotted paths / module references (e.g., omega.sqlite_store, src/omega)
    for m in re.findall(r"\b([\w]+(?:\.[\w]+){1,4})\b", content):
        if not re.match(r"^\d+\.\d+", m):  # Skip version numbers like 1.0.0
            facts.add(m.lower())

    # 7. Technical compound terms (hyphenated, e.g., "multi-session", "cross-agent")
    for m in re.findall(r"\b([a-z]+-[a-z]+(?:-[a-z]+)?)\b", content.lower()):
        if len(m) > 4:
            facts.add(m)

    # Filter out very short or stopword-only facts
    _STOP = {"the", "and", "for", "with", "that", "this", "from", "have", "been", "will", "not"}
    filtered = []
    for f in facts:
        words = f.split()
        meaningful = [w for w in words if w not in _STOP and len(w) > 1]
        if meaningful:
            filtered.append(f)

    return sorted(set(filtered))[:20]


def _compress_to_observation(content: str, event_type: str = "") -> Optional[str]:
    """Compress content to a concise observation (extractive, no LLM).

    Selects the 1-2 most information-dense sentences from the content.
    Returns None if content is already concise (< 150 chars) or compression fails.
    """
    if len(content) < 150:
        return None  # Already concise

    # Split into sentences (preserve abbreviations, version numbers)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\[])", content)
    if len(sentences) <= 1:
        # Try simpler split
        sentences = re.split(r"(?<=[.!?])\s+", content)
    sentences = [s.strip() for s in sentences if len(s.strip()) >= 15]

    if not sentences:
        return None

    # Score each sentence by information density
    scored = []
    for s in sentences:
        words = s.split()
        unique_words = len(set(w.lower() for w in words if len(w) > 3))
        # Bonus for code tokens (backticks, paths, CamelCase)
        code_tokens = len(re.findall(r"`[^`]+`|/[\w/.]+|\b[A-Z][a-z]+[A-Z]\w*\b", s))
        # Diminishing returns on length
        length_score = min(len(s), 200) / 200.0
        density = unique_words * 1.0 + code_tokens * 2.0 + length_score * 3.0
        scored.append((density, s))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Select top 1-2 diverse sentences
    selected = [scored[0][1]]
    if len(scored) > 1:
        # Add second only if sufficiently different
        s2 = scored[1][1]
        if _jaccard(selected[0].lower(), s2.lower(), min_word_len=3) < 0.7:
            selected.append(s2)

    observation = " ".join(selected)
    if len(observation) > 200:
        observation = observation[:197] + "..."

    return observation


def _jaccard(text_a: str, text_b: str, min_word_len: int = 4) -> float:
    """Jaccard similarity on word sets (fast, no embeddings)."""
    words_a = {w for w in text_a.split() if len(w) >= min_word_len}
    words_b = {w for w in text_b.split() if len(w) >= min_word_len}
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def _auto_relate(store, node_id: str, max_related: int = 3, min_similarity: float = 0.65) -> int:
    """Create typed edges from node_id to its most similar existing memories.

    Edge types are inferred from metadata signals:
      - same_entity: both memories share the same entity_id
      - evolution: same event_type with high similarity (concept update)
      - temporal_cluster: created within 1 hour of each other
      - related: fallback for very high similarity (>= 0.80) with no stronger type

    Returns the number of edges created. Silently returns 0 on any error.
    """
    try:
        embedding = store.get_embedding(node_id)
        if not embedding:
            return 0
        similar = store.find_similar(embedding, limit=max_related + 1)
        candidates = [r for r in similar if r.id != node_id and r.relevance >= min_similarity][:max_related]
        if not candidates:
            return 0

        source_node = store.get_node(node_id)
        if not source_node:
            return 0
        src_meta = source_node.metadata or {}
        src_entity = src_meta.get("entity_id", "")
        src_event = src_meta.get("event_type", "")
        src_created = source_node.created_at

        count = 0
        for r in candidates:
            r_meta = r.metadata or {}
            r_entity = r_meta.get("entity_id", "")
            r_event = r_meta.get("event_type", "")

            # Classify edge type from strongest to weakest signal
            if src_entity and r_entity and src_entity == r_entity:
                edge_type = "same_entity"
            elif src_event and r_event and src_event == r_event and r.relevance >= 0.75:
                edge_type = "evolution"
            elif src_created and r.created_at:
                delta = abs((src_created - r.created_at).total_seconds())
                if delta <= 3600:
                    edge_type = "temporal_cluster"
                elif r.relevance >= 0.80:
                    edge_type = "related"
                else:
                    continue  # Below 0.80 with no typed signal: skip
            elif r.relevance >= 0.80:
                edge_type = "related"
            else:
                continue  # No strong signal: skip the generic edge

            if store.add_edge(node_id, r.id, edge_type, r.relevance):
                count += 1
        if count:
            logger.debug(f"Auto-related {node_id[:12]} to {count} memories (typed)")
        return count
    except Exception as e:
        logger.debug(f"_auto_relate failed for {node_id[:12]}: {e}")
        return 0


def _schedule_auto_relate(store, node_id: str) -> None:
    """Fire _auto_relate in a background daemon thread (non-blocking).

    Set ``OMEGA_AUTO_RELATE=0`` (also ``false``/``off``) to skip enrichment
    entirely — parity with ``OMEGA_ENTITY_EXTRACTION``. Requested by issue #58
    reporter as a safety valve for environments where auto-relate's thread
    behavior is undesirable.

    Registers the thread on the store so close() can join it before tearing
    down the sqlite connection (prevents use-after-close segfaults in
    sqlite-vec native code during test teardown).
    """
    if os.environ.get("OMEGA_AUTO_RELATE", "").lower() in ("0", "false", "off"):
        return

    t_ref: list[threading.Thread] = []

    def _run():
        try:
            _auto_relate(store, node_id)
        except Exception as e:
            logger.debug(f"Background _auto_relate failed for {node_id[:12]}: {e}")
        finally:
            if t_ref and hasattr(store, "unregister_background_thread"):
                try:
                    store.unregister_background_thread(t_ref[0])
                except Exception:
                    pass

    t = threading.Thread(target=_run, daemon=True, name="auto-relate")
    t_ref.append(t)
    if hasattr(store, "register_background_thread"):
        store.register_background_thread(t)
    t.start()


_CROSS_TYPE_SUPERSEDE = {
    "user_preference": {"decision"},
}


def _detect_and_supersede(
    store, node_id: str, content: str, event_type: str,
    entity_id: Optional[str] = None,
) -> int:
    """Detect contradicting memories and mark old ones as superseded.

    Only runs for decision, user_preference, user_fact types.
    Uses embedding similarity to find candidates, then checks for topic
    overlap with different content — indicating a contradiction/update.

    Cross-type supersession: user_preference can supersede decision memories
    (e.g. "stop suggesting HN" supersedes "post Show HN on Tuesday").

    Returns count of superseded memories.
    """
    _SUPERSEDE_TYPES = {"decision", "user_preference", "user_fact"}
    if event_type not in _SUPERSEDE_TYPES:
        return 0
    try:
        embedding = store.get_embedding(node_id)
        if not embedding:
            return 0
        similar = store.find_similar(embedding, limit=5)
        superseded = 0
        content_norm = content[:100].strip().lower()
        cross_targets = _CROSS_TYPE_SUPERSEDE.get(event_type)
        for r in similar:
            if r.id == node_id:
                continue
            if (r.metadata or {}).get("superseded"):
                continue
            r_type = (r.metadata or {}).get("event_type", "")
            if r_type != event_type:
                if not cross_targets or r_type not in cross_targets:
                    continue
            if r.relevance < 0.80:
                continue
            if entity_id:
                r_entity = (r.metadata or {}).get("entity_id", "")
                if r_entity and r_entity != entity_id:
                    continue
            existing_norm = r.content[:100].strip().lower()
            if content_norm == existing_norm:
                continue
            store.mark_superseded(r.id, superseded_by=node_id)
            store.add_edge(node_id, r.id, "supersedes", r.relevance)
            superseded += 1
            logger.info(
                "Ingest superseded %s (sim=%.2f) by %s",
                r.id[:12], r.relevance, node_id[:12],
            )
        if superseded:
            store.stats.setdefault("ingest_superseded", 0)
            store.stats["ingest_superseded"] += superseded
        return superseded
    except Exception as e:
        logger.debug("_detect_and_supersede failed for %s: %s", node_id[:12], e)
        return 0


def _split_atomic_facts(content: str, event_type: str) -> List[str]:
    """Extract sentence-level atomic facts from content.

    Identifies standalone factual statements for storage as separate
    user_fact nodes to improve single-mention recall.

    Returns list of fact strings (max 5).

    Gated behind OMEGA_ATOMIC_FACTS=1 (off by default).
    """
    if os.environ.get("OMEGA_ATOMIC_FACTS", "0") != "1":
        return []
    if len(content) < 50:
        return []
    # Only split facts from user-authored types, not agent-generated content
    _FACT_SPLIT_TYPES = {
        "decision", "user_fact",
    }
    if event_type not in _FACT_SPLIT_TYPES:
        return []
    facts = []
    sentences = re.split(r"(?<=[.!?])\s+", content)
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 15 or len(sentence) > 200:
            continue
        s_lower = sentence.lower()
        # Require first-person or infrastructure context to confirm it's a user fact,
        # not an agent observation like "The component was a God object"
        _has_user_signal = bool(re.search(
            r"\b(?:we|our|my|i)\s+|"
            r"\b(?:the|our)\s+(?:db|database|server|api|config|project|repo|app|service|endpoint|url|path|port)\b",
            s_lower,
        ))
        if not _has_user_signal:
            continue
        if re.search(r"\b(?:is|are|was|were)\s+\w", s_lower):
            facts.append(sentence)
            continue
        if re.search(
            r"\b(?:we\s+use|using|uses?|adopted?|switched?\s+to)\s+\w",
            s_lower,
        ):
            facts.append(sentence)
            continue
        if re.search(
            r"\b(?:moved?\s+to|lives?\s+in|located?\s+(?:in|at)"
            r"|based\s+in)\s+\w",
            s_lower,
        ):
            facts.append(sentence)
            continue
        if re.search(
            r"\b(?:password|key|token|secret|api.?key)"
            r"\s+(?:is|=|:)\s*\S",
            s_lower,
        ):
            facts.append(sentence)
            continue
    seen = set()
    unique = []
    for f in facts:
        f_norm = f.strip().lower()
        if f_norm not in seen:
            seen.add(f_norm)
            unique.append(f)
    return unique[:5]


def _schedule_entity_extraction(
    store: Any,
    node_id: str,
    content: str,
    event_type: str,
) -> None:
    """Fire entity extraction in a background daemon thread.

    Non-blocking. Silently skipped if API key missing or extraction disabled.
    """
    import os as _os
    if _os.environ.get("OMEGA_ENTITY_EXTRACTION", "").lower() in ("0", "false", "off"):
        return
    if not _os.environ.get("ANTHROPIC_API_KEY"):
        return

    t_ref: list[threading.Thread] = []

    def _run():
        try:
            from omega_platform.entity.extraction import extract_entities, resolve_and_link
            from omega_platform.entity.engine import get_entity_manager
            from pathlib import Path as _Path

            extraction = extract_entities(content, event_type)
            if extraction["entities"]:
                em = get_entity_manager(_Path(store.db_path))
                resolve_and_link(store, em, node_id, extraction)
        except Exception as e:
            logger.debug("Async entity extraction failed: %s", e)
        finally:
            if t_ref and hasattr(store, "unregister_background_thread"):
                try:
                    store.unregister_background_thread(t_ref[0])
                except Exception:
                    pass

    t = threading.Thread(target=_run, daemon=True, name="entity-extraction")
    t_ref.append(t)
    if hasattr(store, "register_background_thread"):
        store.register_background_thread(t)
    t.start()


# ---------------------------------------------------------------------------
# Public API -- Core CRUD
# ---------------------------------------------------------------------------


def auto_capture(
    content: str,
    event_type: str,
    metadata: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    project: Optional[str] = None,
    ttl_override: Optional[int] = None,
    entity_id: Optional[str] = None,
    agent_type: Optional[str] = None,
) -> str:
    """Store a memory with auto-classification, dedup, and evolution.

    This is the primary ingestion function. It:
    1. Checks for near-duplicate content (Jaccard) and reuses if found.
    2. Tries to *evolve* an existing memory with new insights (Zettelkasten).
    3. Falls back to creating a new memory node.

    Returns:
        Markdown confirmation string.
    """
    content = unicodedata.normalize("NFC", content)

    # Auto-resolve entity_id from project if not explicitly provided
    if not entity_id and project:
        try:
            from omega_platform.entity.engine import resolve_project_entity
            entity_id = resolve_project_entity(project)
        except Exception as e:
            logger.debug("Entity resolution failed: %s", e)

    # Determine source early — hooks vs direct API calls have different filtering rules.
    _source = (metadata or {}).get("source", "")
    _is_hook = _source.startswith("auto_") or _source.endswith("_hook")

    # Block system noise early — startswith patterns are position-specific, safe for all sources.
    for pattern in _BLOCKLIST_STARTSWITH:
        if content.startswith(pattern):
            return "**Memory Blocked** (system noise)"
    # Contains patterns only apply to hook-sourced content to avoid false positives
    # on direct API calls (e.g. storing a lesson that mentions "error":).
    # Also skip for preference/fact types — users legitimately store prefs containing "error".
    _BLOCKLIST_EXEMPT_TYPES = {AutoCaptureEventType.USER_PREFERENCE, AutoCaptureEventType.USER_FACT}
    if _is_hook and event_type not in _BLOCKLIST_EXEMPT_TYPES:
        for pattern in _BLOCKLIST_CONTAINS:
            if pattern in content:
                return "**Memory Blocked** (system noise)"

    # Min-length gate — only for auto-captured content from hooks, not direct API calls.
    if _is_hook and len(content) < _MIN_CONTENT_LENGTH and event_type != AutoCaptureEventType.USER_PREFERENCE:
        return "**Memory Blocked** (too short)"

    # Block infrastructure event types that generate noise and inflate never-accessed count
    if _is_hook and event_type in _INFRASTRUCTURE_EVENT_TYPES:
        return "**Memory Blocked** (infrastructure noise)"

    # Block zero-value outcome records (tokens=0 partial sessions)
    if _is_hook and event_type == "task_completion" and "tokens=0" in content:
        return "**Memory Blocked** (zero-token outcome)"

    # Block JSON-blob decisions — raw tool output stored as "decisions"
    # Exempt coord_dual_write: its [domain] prefix is not JSON
    if event_type == "decision" and _source != "coord_dual_write":
        # Strip known prefixes to check the actual body
        _body = content
        for _pfx in ("Decision: ", "Plan/decision captured: ", "Fact: "):
            if _body.startswith(_pfx):
                _body = _body[len(_pfx):]
        _body_stripped = _body.lstrip()
        if _body_stripped.startswith(("{", "[", '"filePath', '"type"')):
            return "**Memory Blocked** (JSON blob, not a decision)"

    store = _get_store()
    meta = dict(metadata or {})
    meta["event_type"] = event_type
    if project:
        meta["project"] = project
    meta["captured_at"] = datetime.now(timezone.utc).isoformat()

    # Set capture confidence (if not already set by caller)
    if not meta.get("capture_confidence"):
        source = meta.get("source", "")
        if source == "user_remember":
            meta["capture_confidence"] = "high"
        elif event_type == "user_preference":
            meta["capture_confidence"] = "high"
        elif event_type in ("lesson_learned", "error_pattern") and not source.startswith("auto_"):
            # Direct API calls for lessons/errors = validated by agent
            meta["capture_confidence"] = "high"
        elif source in ("auto_plan_capture",):
            # Auto-captured plans are speculative
            meta["capture_confidence"] = "low"
        elif source.startswith("auto_") or source.endswith("_hook"):
            meta["capture_confidence"] = "medium"
        else:
            # Direct API calls (agent-initiated store) = higher trust
            meta["capture_confidence"] = "high"

    # Auto-tag extraction
    auto_tags = _extract_tags(content, project)
    if auto_tags:
        existing_tags = meta.get("tags", [])
        meta["tags"] = sorted(set(existing_tags + auto_tags))[:15]

    # Fact extraction for high-value types — merge fact terms into tags
    # (boosted in Phase 2.5 word/tag overlap for retrieval).
    _FACT_EXTRACTION_TYPES = {"decision", "lesson_learned", "session_summary", "error_pattern", "advisor_insight"}
    if event_type in _FACT_EXTRACTION_TYPES:
        try:
            facts = _extract_facts(content)
            if facts:
                # Merge fact terms into tags (boosted in Phase 2.5 word/tag overlap)
                existing_tags = meta.get("tags", [])
                # Take the shortest/most specific facts as tags (avoid long phrases)
                fact_tags = [f for f in facts if len(f) <= 25 and " " not in f]
                meta["tags"] = sorted(set(existing_tags + fact_tags))[:20]
        except Exception as e:
            logger.debug(f"Fact extraction failed: {e}")

    ttl = ttl_override if ttl_override is not None else TTLCategory.for_event_type(event_type)

    # System insights are architectural knowledge — make them permanent
    if ttl is not None and meta.get("category") == "system_insight":
        ttl = None  # None = permanent (no expiry)

    # ------------------------------------------------------------------
    # Phase 1 + 2: Content dedup, error burst, and evolution
    # ------------------------------------------------------------------
    # Single query for both dedup and evolution (same search text).
    # This avoids duplicate embedding generation + DB round-trips.
    dedup_threshold = DEDUP_THRESHOLDS.get(event_type)
    _similar_results = None  # Lazy-loaded, shared between dedup and evolution

    # Pre-compute embedding once — reused for dedup query and final store
    _precomputed_embedding = None
    if dedup_threshold is not None or event_type in EVOLUTION_TYPES:
        try:
            from omega.embedding import generate_embedding
            _precomputed_embedding = generate_embedding(content)
        except Exception as e:
            logger.debug(f"Pre-computed embedding generation failed: {e}")

    if dedup_threshold is not None or event_type in EVOLUTION_TYPES:
        try:
            _similar_results = store.query(
                content[:200], limit=8,
                query_embedding=_precomputed_embedding,
            )
        except Exception as e:
            logger.debug(f"Similar-content query failed: {e}")

    # Phase 1: Content-level dedup
    if dedup_threshold is not None and _similar_results:
        try:
            for existing in _similar_results:
                if (existing.metadata or {}).get("event_type", "") != event_type:
                    continue
                # Session filter for dedup: only dedup within same session
                # Exception: decisions, lessons, and task completions dedup cross-session
                # (same architectural choice, lesson, or completion restated across sessions)
                _CROSS_SESSION_DEDUP_TYPES = {
                    AutoCaptureEventType.DECISION,
                    AutoCaptureEventType.LESSON_LEARNED,
                    AutoCaptureEventType.TASK_COMPLETION,
                    AutoCaptureEventType.ADVISOR_INSIGHT,
                }
                if session_id and event_type not in _CROSS_SESSION_DEDUP_TYPES:
                    existing_session = (existing.metadata or {}).get("session_id", "")
                    if existing_session and existing_session != session_id:
                        continue
                if event_type == AutoCaptureEventType.ERROR_PATTERN:
                    sim = _jaccard(_normalize_for_dedup(content), _normalize_for_dedup(existing.content))
                else:
                    sim = _jaccard(content.lower(), existing.content.lower())
                if sim > dedup_threshold:
                    store.update_node(existing.id, access_count=(existing.access_count or 0) + 1)
                    store.stats.setdefault("content_dedup_skips", 0)
                    store.stats["content_dedup_skips"] += 1
                    _schedule_auto_relate(store, existing.id)
                    logger.debug(f"Content dedup: skipped {event_type} (jaccard={sim:.2f}), reusing {existing.id[:12]}")
                    return f"Deduped → {existing.id}"
        except Exception as e:
            logger.debug(f"Content dedup check skipped: {e}")

    # Phase 1.5: Error burst detection
    if event_type == AutoCaptureEventType.ERROR_PATTERN and session_id:
        try:
            # Use similar results if available, otherwise minimal query
            burst_candidates = _similar_results or []
            session_errors = [
                r
                for r in burst_candidates
                if (r.metadata or {}).get("event_type") == AutoCaptureEventType.ERROR_PATTERN
                and (r.metadata or {}).get("session_id") == session_id
            ]
            if len(session_errors) >= 3:
                # Only capture if truly novel (Jaccard < 0.40 with all recent errors)
                is_novel = all(_jaccard(content.lower(), e.content.lower()) < 0.40 for e in session_errors)
                if not is_novel:
                    store.stats.setdefault("error_burst_skips", 0)
                    store.stats["error_burst_skips"] += 1
                    return "Blocked (error burst — duplicate)"
        except Exception as e:
            logger.debug(f"Error burst check skipped: {e}")

    # Phase 2: Memory evolution (Zettelkasten-inspired)
    if event_type in EVOLUTION_TYPES and _similar_results:
        try:
            for existing in _similar_results[:3]:
                if (existing.metadata or {}).get("event_type", "") != event_type:
                    continue
                sim = _jaccard(content.lower(), existing.content.lower())
                if EVOLUTION_THRESHOLD <= sim < (dedup_threshold or 0.95):
                    old_words = {w.lower() for w in existing.content.split() if len(w) > 3}
                    new_info = {w.lower() for w in content.split() if len(w) > 3} - old_words
                    if len(new_info) == 0:
                        # Near-exact reconfirmation — bump access count to strengthen memory
                        store.update_node(
                            existing.id,
                            access_count=(existing.access_count or 0) + 1,
                        )
                        store.stats.setdefault("reconfirmation_bumps", 0)
                        store.stats["reconfirmation_bumps"] += 1
                        return f"Reconfirmed {existing.id} (access bumped)"
                    # Allow evolution with even 1 new word (was 3 — caused dead zone)
                    # The sentence-level filter below still requires >= 2 new words per sentence

                    evolved = existing.content.rstrip()
                    if not evolved.endswith("."):
                        evolved += "."

                    new_sentences = []
                    # Split on sentence boundaries, preserving abbreviations
                    # like "Dr.", "e.g.", "i.e.", version numbers "v2.0"
                    for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", content):
                        sentence = sentence.strip()
                        if not sentence or len(sentence) < 10:
                            continue
                        s_words = {w.lower() for w in sentence.split() if len(w) > 3}
                        if s_words and len(s_words - old_words) >= 2:
                            new_sentences.append(sentence)

                    if new_sentences:
                        addition = " ".join(new_sentences[:2])
                        new_content = f"{evolved} [Updated] {addition}"
                        emeta = dict(existing.metadata or {})
                        evo_count = emeta.get("evolution_count", 0) + 1
                        emeta["evolution_count"] = evo_count
                        emeta["last_evolved"] = datetime.now(timezone.utc).isoformat()
                        emeta["evolved_from_sessions"] = list(
                            set(emeta.get("evolved_from_sessions", []) + ([session_id] if session_id else []))
                        )[:10]

                        store.update_node(
                            existing.id,
                            content=new_content,
                            metadata=emeta,
                            access_count=(existing.access_count or 0) + 1,
                        )

                        store.stats.setdefault("memory_evolutions", 0)
                        store.stats["memory_evolutions"] += 1
                        _schedule_auto_relate(store, existing.id)
                        logger.info(f"Memory evolved: {existing.id[:12]} (evolution #{evo_count}, jaccard={sim:.2f})")
                        return f"Evolved {existing.id} (#{evo_count})"
                    break  # Only try the top match
        except Exception as e:
            logger.debug(f"Memory evolution check skipped: {e}")

    # ------------------------------------------------------------------
    # Phase 2.5: Conflict detection — find contradictions with existing
    # ------------------------------------------------------------------
    _conflict_results = []
    if _similar_results and event_type in (
        "user_preference", "decision", "lesson_learned", "error_pattern"
    ):
        try:
            from omega.conflicts import detect_conflicts

            _conflict_results = detect_conflicts(
                content, event_type, _similar_results[:5],
            )

            # Auto-resolve: mark old memory as outdated
            for conflict in _conflict_results:
                if conflict["auto_resolve"] and conflict.get("existing_id"):
                    try:
                        store.record_feedback(
                            conflict["existing_id"], "outdated",
                            reason=f"Conflict auto-resolved: {conflict['reason']}",
                        )
                    except Exception as e:
                        logger.debug(f"Auto-resolve feedback failed: {e}")

                # Flag-only: add conflict_flags to metadata for visibility
                if not conflict["auto_resolve"] and conflict.get("existing_id"):
                    try:
                        existing_node = store.get_node(conflict["existing_id"])
                        if existing_node:
                            emeta = dict(existing_node.metadata or {})
                            flags = emeta.get("conflict_flags", [])
                            flags.append({
                                "reason": conflict["reason"],
                                "confidence": conflict["confidence"],
                            })
                            emeta["conflict_flags"] = flags[-5:]  # Keep last 5
                            store.update_node(conflict["existing_id"], metadata=emeta)
                    except Exception as e:
                        logger.debug(f"Conflict flagging failed: {e}")
        except Exception as e:
            logger.debug(f"Conflict detection failed: {e}")

    # ------------------------------------------------------------------
    # Phase 3: Store new node
    # ------------------------------------------------------------------
    # Wire entity_id into metadata for tag-based discovery
    if entity_id:
        meta["entity_id"] = entity_id

    # Wire agent_type into metadata for discovery
    if agent_type:
        meta["agent_type"] = agent_type

    node_id = store.store(
        content=content,
        session_id=session_id,
        metadata=meta,
        embedding=_precomputed_embedding,
        ttl_seconds=ttl,
        entity_id=entity_id,
        agent_type=agent_type,
    )

    ttl_str = _human_ttl(ttl)
    # store() returns a node ID whether it inserted or collapsed into an
    # existing memory, so report the outcome the store actually took —
    # "Stored" on a dedup would claim a write that never happened.
    try:
        _deduped = store.get_last_store_deduped()
    except AttributeError:
        _deduped = False
    if _deduped:
        output = f"Deduped → {node_id}"
    else:
        output = f"Stored {node_id} ({event_type}, {ttl_str})"

    # Surface deep contradiction detection results
    try:
        contradiction_results = store.get_last_contradiction_results()
        if contradiction_results:
            cr_lines = []
            for cr in contradiction_results:
                cr_lines.append(
                    f"  - `{cr['node_id'][:16]}` ({cr['confidence']:.0%}): {cr['reason']}"
                )
            output += "\n\n[CONTRADICTION] New memory may contradict:\n" + "\n".join(cr_lines)
    except Exception as e:
        logger.debug("Contradiction surfacing failed: %s", e)

    # Surface capacity warning if near limit
    if hasattr(store, '_capacity_warning') and store._capacity_warning:
        output += f"\n\n**Warning:** {store._capacity_warning}"

    # Append conflict information compactly
    if _conflict_results:
        resolved = sum(1 for c in _conflict_results if c["auto_resolve"])
        flagged = len(_conflict_results) - resolved
        parts = []
        if resolved:
            parts.append(f"{resolved} auto-resolved")
        if flagged:
            parts.append(f"{flagged} flagged")
        output += f" | conflicts: {', '.join(parts)}"

    # Milestone check (cheap: one node_count query + file existence check).
    # A full-retrieval extension capability suppresses Free-tier upgrade CTA
    # suffixes. Core never trusts a local license function for entitlement.
    try:
        from omega.milestones import check_capture_milestones
        full_retrieval = False
        try:
            from omega.plugins import has_capability
            full_retrieval = has_capability("full_retrieval")
        except Exception as e:
            logger.debug("Capability check failed in milestone: %s", e)
        count = store.node_count()
        milestone_msg = check_capture_milestones(count, is_pro_user=full_retrieval)
        if milestone_msg:
            output += f" | {milestone_msg}"
    except Exception as e:
        logger.debug("Milestone check failed: %s", e)

    # ------------------------------------------------------------------
    # Phase 3.1: Async entity extraction (non-blocking)
    # ------------------------------------------------------------------
    _schedule_entity_extraction(store, node_id, content, event_type)

    # ------------------------------------------------------------------
    # Phase 3.5: Observation compression for high-value types
    # ------------------------------------------------------------------
    _HIGH_VALUE_OBSERVATION_TYPES = {"decision", "lesson_learned", "error_pattern", "user_preference", "constraint", "advisor_insight"}
    if event_type in _HIGH_VALUE_OBSERVATION_TYPES:
        try:
            observation = _compress_to_observation(content, event_type)
            if observation:
                meta["observation"] = observation
                store.update_node(node_id, metadata=meta)
        except Exception as e:
            logger.debug(f"Observation compression failed for {node_id[:12]}: {e}")

    # ------------------------------------------------------------------
    # Phase 4: Auto-relate — link to similar existing memories (background)
    # ------------------------------------------------------------------
    _schedule_auto_relate(store, node_id)

    # ------------------------------------------------------------------
    # Phase 4.1: Contradiction detection — supersede old conflicting memories
    # ------------------------------------------------------------------
    try:
        supersede_count = _detect_and_supersede(
            store, node_id, content, event_type, entity_id,
        )
        if supersede_count:
            output += f" | {supersede_count} superseded"
    except Exception as e:
        logger.debug(f"Contradiction detection failed for {node_id[:12]}: {e}")

    # ------------------------------------------------------------------
    # Phase 4.2: Atomic fact splitting — create sub-nodes for recall
    # ------------------------------------------------------------------
    try:
        atomic_facts = _split_atomic_facts(content, event_type)
        fact_count = 0
        for fact_text in atomic_facts:
            fact_meta = {
                "event_type": "user_fact",
                "source_node": node_id,
                "auto_extracted": True,
            }
            if session_id:
                fact_meta["session_id"] = session_id
            if project:
                fact_meta["project"] = project
            if entity_id:
                fact_meta["entity_id"] = entity_id
            fact_id = store.store(
                content=fact_text,
                session_id=session_id,
                metadata=fact_meta,
                entity_id=entity_id,
            )
            store.add_edge(node_id, fact_id, "contains_fact", 1.0)
            fact_count += 1
        if fact_count:
            output += f" | {fact_count} facts extracted"
    except Exception as e:
        logger.debug(f"Atomic fact splitting failed for {node_id[:12]}: {e}")

    # ------------------------------------------------------------------
    # Phase 4.5: Auto-supersede stale reminders
    # ------------------------------------------------------------------
    _COMPLETION_TYPES = {"decision", "task_completion"}
    if event_type in _COMPLETION_TYPES:
        try:
            superseded_count = 0
            superseded_ids: set = set()
            content_words = {w.lower() for w in content.split() if len(w) > 3}

            # --- Pass 1: Embedding similarity (threshold lowered to 0.40) ---
            embedding = store.get_embedding(node_id)
            if embedding:
                similar = store.find_similar(embedding, limit=10)
                for r in similar:
                    if r.id == node_id:
                        continue
                    r_type = (r.metadata or {}).get("event_type")
                    if r_type not in ("reminder", "checkpoint"):
                        continue
                    if (r.metadata or {}).get("superseded"):
                        continue
                    if r.relevance < 0.40:
                        continue
                    superseded_ids.add(r.id)

            # --- Pass 2: Keyword matching (3+ word overlap, like task auto-resolve) ---
            with store._lock:
                pending_rows = store._conn.execute(
                    "SELECT node_id, content FROM memories "
                    "WHERE event_type = 'reminder' "
                    "AND json_extract(metadata, '$.reminder_status') = 'pending'"
                ).fetchall()
            for r_id, r_content in pending_rows:
                if r_id in superseded_ids:
                    continue
                r_words = {w.lower() for w in (r_content or "").split() if len(w) > 3}
                matches = sum(1 for w in r_words if w in content_words)
                if matches >= 3:
                    superseded_ids.add(r_id)

            # --- Apply: mark superseded AND set reminder_status = dismissed ---
            for s_id in superseded_ids:
                r_row = store.get(s_id)
                if not r_row:
                    continue
                r_meta = dict(r_row.metadata or {})
                r_meta["superseded"] = True
                r_meta["superseded_by"] = node_id
                r_meta["reminder_status"] = "dismissed"
                r_meta["dismissed_at"] = datetime.now(timezone.utc).isoformat()
                r_meta["dismissed_reason"] = "auto_superseded"
                store.update_node(s_id, metadata=r_meta)
                r_type = r_meta.get("event_type", "reminder")
                store._log_forgetting_external(
                    s_id, r_row.content, r_type,
                    "auto_superseded", {"superseded_by": node_id},
                )
                superseded_count += 1
            if superseded_count:
                output += f" | superseded {superseded_count} reminder(s)"
                logger.info(f"Auto-superseded {superseded_count} reminders for {node_id}")
        except Exception as e:
            logger.debug(f"Auto-supersede failed for {node_id}: {e}")

    # ------------------------------------------------------------------
    # Phase 5: Implicit positive feedback — retrieval-then-store signal
    # ------------------------------------------------------------------
    _IMPLICIT_FB_TYPES = {"decision", "lesson_learned", "error_pattern"}
    if event_type in _IMPLICIT_FB_TYPES:
        try:
            ctx_entries = store.get_retrieval_context()
            if ctx_entries:
                content_words = {w.lower() for w in content.split() if len(w) > 3}
                implicit_count = 0
                for entry in ctx_entries:
                    query_text = entry.get("query_text", "")
                    if not query_text:
                        continue
                    query_words = {w.lower() for w in query_text.split() if len(w) > 3}
                    if not query_words:
                        continue
                    overlap = len(content_words & query_words) / len(query_words)
                    if overlap >= 0.30:
                        entry_nid = entry.get("node_id", "")
                        if entry_nid and entry_nid != node_id:
                            store.record_feedback(
                                entry_nid, "helpful",
                                reason="implicit: retrieval-then-store",
                            )
                            implicit_count += 1
                if implicit_count:
                    store.stats.setdefault("implicit_feedback_boosts", 0)
                    store.stats["implicit_feedback_boosts"] += implicit_count
                    logger.debug(f"Implicit feedback: boosted {implicit_count} memories for {node_id[:12]}")
        except Exception as e:
            logger.debug(f"Implicit feedback failed for {node_id[:12]}: {e}")

    logger.info(f"Auto-captured {event_type}: {node_id}")
    return output


def store(
    content: str,
    event_type: str = "memory",
    metadata: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    project: Optional[str] = None,
    entity_id: Optional[str] = None,
    agent_type: Optional[str] = None,
) -> str:
    """Direct store -- wraps auto_capture with a default event type."""
    return auto_capture(
        content=content,
        event_type=event_type,
        metadata=metadata,
        session_id=session_id,
        project=project,
        entity_id=entity_id,
        agent_type=agent_type,
    )


def remember(text: str, session_id: Optional[str] = None, entity_id: Optional[str] = None) -> str:
    """User-facing 'remember this' -- stores with user_preference type."""
    return auto_capture(
        content=text,
        event_type=AutoCaptureEventType.USER_PREFERENCE,
        session_id=session_id,
        entity_id=entity_id or "omega",
        metadata={"source": "user_remember"},
    )


def delete_memory(memory_id: str) -> Dict[str, Any]:
    """Delete a memory by its node ID."""
    db = _get_store()
    try:
        success = db.delete_node(memory_id)
        if success:
            logger.info(f"Deleted memory {memory_id[:12]}")
            return {"success": True, "deleted_id": memory_id}
        return {"success": False, "error": f"Memory {memory_id} not found"}
    except Exception as e:
        logger.error(f"Failed to delete memory {memory_id[:12]}: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def edit_memory(
    memory_id: str,
    new_content: Optional[str] = None,
    priority: Optional[int] = None,
) -> Dict[str, Any]:
    """Edit a memory's content, priority, or both without replacing its ID.

    After editing, extracts style observations from the diff and stores
    them as user_preference memories (memory-from-edits pattern).
    """
    if new_content is None and priority is None:
        return {"success": False, "error": "new_content or priority is required"}
    if new_content is not None and not new_content.strip():
        return {"success": False, "error": "new_content must not be empty"}
    if priority is not None and (
        isinstance(priority, bool)
        or not isinstance(priority, int)
        or not 1 <= priority <= 5
    ):
        return {"success": False, "error": "priority must be an integer from 1 to 5"}

    db = _get_store()
    try:
        node = db.get_node(memory_id, track_access=False)
        if node is None:
            return {"success": False, "error": f"Memory {memory_id} not found"}

        old_content = node.content
        old_preview = old_content[:80]
        emeta = dict(node.metadata or {})
        emeta["edited_at"] = datetime.now(timezone.utc).isoformat()
        emeta["edit_count"] = emeta.get("edit_count", 0) + 1

        success = db.update_node(
            memory_id,
            content=new_content,
            metadata=emeta,
            priority=priority,
        )
        if not success:
            return {"success": False, "error": f"Memory {memory_id} not found"}

        logger.info(f"Edited memory {memory_id[:12]}")

        # Memory-from-edits: extract style observations from the diff
        edit_observation = None
        if new_content is not None:
            edit_observation = _extract_edit_observation(
                old_content,
                new_content,
                event_type=(node.metadata or {}).get("event_type", "memory"),
                memory_id=memory_id,
            )

        result = {
            "success": True,
            "id": memory_id,
            "old_content_preview": old_preview,
        }
        if new_content is not None:
            result["new_content_preview"] = new_content[:80]
        if priority is not None:
            result["priority"] = priority

        if edit_observation:
            result["style_observation"] = edit_observation

        return result
    except Exception as e:
        logger.error(f"Failed to edit memory {memory_id[:12]}: {e}", exc_info=True)
        return {"success": False, "error": "Failed to edit memory"}


def _extract_edit_observation(
    old_content: str,
    new_content: str,
    event_type: str = "memory",
    memory_id: str = "",
) -> Optional[str]:
    """Extract a style observation from a human edit and store it.

    Analyzes the diff between old and new content to detect patterns:
    - Length changes (conciseness preference)
    - Word additions/removals (terminology preferences)
    - Structural changes (formatting preferences)

    Returns the observation text if one was stored, None otherwise.
    """
    # Skip trivial edits
    if not old_content or not new_content:
        return None
    if old_content.strip() == new_content.strip():
        return None

    old_words = set(old_content.lower().split())
    new_words = set(new_content.lower().split())
    added_words = new_words - old_words
    removed_words = old_words - new_words

    # Skip if change is too small to learn from
    if len(added_words) + len(removed_words) < 3:
        return None

    observations = []

    # Length change
    old_len = len(old_content)
    new_len = len(new_content)
    if new_len < old_len * 0.7:
        observations.append("prefers more concise/shorter content")
    elif new_len > old_len * 1.5:
        observations.append("prefers more detailed/longer content")

    # Significant word additions (filter noise words)
    _noise = {"the", "a", "an", "is", "are", "was", "were", "and", "or", "but",
              "in", "on", "at", "to", "for", "of", "with", "by", "from", "that",
              "this", "it", "not", "be", "have", "has", "had", "do", "does", "did"}
    meaningful_added = {w for w in added_words if w not in _noise and len(w) > 2}
    meaningful_removed = {w for w in removed_words if w not in _noise and len(w) > 2}

    if meaningful_removed and meaningful_added and len(meaningful_removed) <= 5:
        # Word replacement pattern — most valuable signal
        removed_sample = ", ".join(sorted(meaningful_removed)[:3])
        added_sample = ", ".join(sorted(meaningful_added)[:3])
        observations.append(f"replaced terms ({removed_sample}) with ({added_sample})")

    # Formatting changes
    old_has_bullets = "- " in old_content or "* " in old_content
    new_has_bullets = "- " in new_content or "* " in new_content
    if not old_has_bullets and new_has_bullets:
        observations.append("prefers bullet-point formatting")
    elif old_has_bullets and not new_has_bullets:
        observations.append("prefers prose over bullet points")

    old_has_headers = "## " in old_content or "# " in old_content
    new_has_headers = "## " in new_content or "# " in new_content
    if not old_has_headers and new_has_headers:
        observations.append("prefers headers/structure in content")

    if not observations:
        return None

    # Build and store the observation
    observation_text = (
        f"Edit pattern on {event_type} memory: " + "; ".join(observations) + "."
    )

    try:
        auto_capture(
            content=observation_text,
            event_type="user_preference",
            metadata={
                "source": "edit_observation",
                "derived_from": memory_id,
                "edited_event_type": event_type,
                "observation_type": "style",
            },
        )
        logger.info("Stored edit observation for %s: %s", memory_id[:12], observation_text[:80])
        return observation_text
    except Exception as e:
        logger.warning("Failed to store edit observation: %s", e)
        return None


# ---------------------------------------------------------------------------
# Public API -- Query
# ---------------------------------------------------------------------------


def query(
    query_text: str,
    limit: int = 10,
    session_id: Optional[str] = None,
    project: Optional[str] = None,
    event_type: Optional[str] = None,
    context_file: Optional[str] = None,
    context_tags: Optional[List[str]] = None,
    filter_tags: Optional[List[str]] = None,
    temporal_range: Optional[tuple] = None,
    entity_id: Optional[str] = None,
    agent_type: Optional[str] = None,
    scope: Optional[str] = None,
    surfacing_context: Optional[Any] = None,
    perspective: Optional[str] = None,
    strength_min: Optional[float] = None,
    memory_type: Optional[str] = None,
    include_contradicted: bool = False,
    valid_at: Optional[str] = None,
    status: Optional[str] = None,
) -> str:
    """Search memories with optional intent-aware routing.

    Args:
        context_file: Current file being edited (for contextual re-ranking).
        context_tags: Current context tags like language, tools (for re-ranking boost).
        filter_tags: Hard filter -- only return memories containing ALL specified tags.
        temporal_range: Optional (start_iso, end_iso) tuple. Auto-inferred from query if not given.
        surfacing_context: SurfacingContext enum for context-aware scoring (error_debug, planning, etc.).
        strength_min: Minimum strength score (0.0-1.0). Filters out weak/decayed memories.

    Returns:
        Formatted markdown string with results.
    """
    db = _get_store()
    query_text = unicodedata.normalize("NFC", query_text)

    try:
        # Auto-infer temporal range from query text if not explicitly provided
        effective_temporal = temporal_range or _infer_temporal_range(query_text)
        # When range was auto-inferred, use soft scoring (boost only, no harsh penalty)
        _temporal_boost_only = temporal_range is None and effective_temporal is not None

        enhanced = query_text
        if event_type:
            enhanced = f"{event_type} {enhanced}"
        if project:
            enhanced = f"{Path(project).name} {enhanced}"

        # Pass scope through to store; "session" restricts to caller's session
        _scope = scope if scope in ("project", "session") else "project"
        query_kwargs: Dict[str, Any] = {
            "limit": limit * 3 if (filter_tags or entity_id or agent_type) else limit,
            "session_id": session_id,
            "context_file": context_file or "",
            "context_tags": context_tags,
            "temporal_range": effective_temporal,
            "entity_id": entity_id,
            "agent_type": agent_type,
            "query_hint": event_type,
            "temporal_boost_only": _temporal_boost_only,
            "scope": _scope,
        }
        if surfacing_context is not None:
            query_kwargs["surfacing_context"] = surfacing_context
        if perspective:
            query_kwargs["perspective"] = perspective
        if valid_at:
            query_kwargs["valid_at"] = valid_at
        results = db.query(enhanced, **query_kwargs)

        # Filter by event_type if specified
        if event_type and results:
            results = [r for r in results if (r.metadata or {}).get("event_type") == event_type]

        # Hard filter by tags (AND logic — all specified tags must be present)
        if filter_tags and results:
            filter_set = {t.lower() for t in filter_tags}
            results = [
                r for r in results if filter_set.issubset({str(t).lower() for t in (r.metadata or {}).get("tags", [])})
            ]

        # Filter by memory_type
        if memory_type and results:
            results = [r for r in results if (r.metadata or {}).get("memory_type") == memory_type]

        # Filter by lifecycle status (active, superseded, speculative, archived)
        if status and results:
            results = [r for r in results if (r.metadata or {}).get("status", "active") == status]

        # Filter to only contradicted memories
        if include_contradicted and results:
            results = [r for r in results if (r.metadata or {}).get("contradicted_by")]

        results = results[:limit]

        # Filter by minimum strength score
        if strength_min is not None and strength_min > 0:
            results = [r for r in results if getattr(r, "strength", 0.0) >= strength_min]

        # Extract query confidence from results
        _qconf = None
        if results:
            _qconf = (results[0].metadata or {}).get("_query_confidence")

        # Format
        _conf_label = ""
        if _qconf is not None and _qconf < 0.3:
            _conf_label = " (confidence: low -- results may not be relevant)"
        elif _qconf is not None and _qconf <= 0.7:
            _conf_label = " (confidence: medium)"
        output = f"Results: {len(results)}{_conf_label}\n"

        if results:
            for i, node in enumerate(results[:limit], 1):
                ntype = (node.metadata or {}).get("event_type", "memory")
                preview = node.content[:200] + "..." if len(node.content) > 200 else node.content
                _str = getattr(node, "strength", 0.0)
                _meta = node.metadata or {}
                _status = _meta.get("status", "active")
                _status_tag = f" [{_status}]" if _status != "active" else ""
                output += f"## {i}. [{ntype}] `{node.id}` (str: {_str:.2f}){_status_tag}\n"
                output += f"{preview}\n"
                created = node.created_at.isoformat()[:16] if node.created_at else ""
                _extras = []
                if _meta.get("source_uri"):
                    _extras.append(f"source: {_meta['source_uri']}")
                if _meta.get("derived_from"):
                    _extras.append(f"derived from: {_meta['derived_from']}")
                _extras_str = f" | {' | '.join(_extras)}" if _extras else ""
                output += f"*{created}{_extras_str}*\n\n"
        else:
            output += "*No matching memories found.*\n"

        # Auto-inject relevant constraints (always, regardless of event_type filter)
        if event_type != "constraint":
            try:
                result_ids = {n.id for n in results}
                constraint_nodes = db.get_by_type("constraint", limit=10)
                matching_constraints = []
                if constraint_nodes:
                    query_words = {w.lower() for w in query_text.split() if len(w) > 2}
                    for cn in constraint_nodes:
                        if cn.id in result_ids:
                            continue
                        if (cn.metadata or {}).get("superseded"):
                            continue
                        content_words = {w.lower() for w in cn.content.split() if len(w) > 2}
                        if query_words & content_words:
                            matching_constraints.append(cn)
                if matching_constraints:
                    output += "\n---\n**Active Constraints:**\n"
                    for cr in matching_constraints[:3]:
                        preview = cr.content[:150]
                        output += f"- [`{cr.id}`] {preview}\n"
            except Exception as e:
                logger.debug("Constraint injection failed: %s", e)

        # Auto-inject relevant user preferences for preference-intent queries
        _PREF_SIGNAL_WORDS = {
            "rule", "rules", "preference", "setting", "configured",
            "should", "allowed", "policy", "default", "location",
            "timezone", "where", "how",
        }
        if event_type != "user_preference":
            try:
                query_words_lower = {w.lower().rstrip("?.,!") for w in query_text.split() if len(w) > 1}
                if query_words_lower & _PREF_SIGNAL_WORDS:
                    result_ids = {n.id for n in results}
                    pref_nodes = db.get_by_type("user_preference", limit=20)
                    matching_prefs = []
                    if pref_nodes:
                        query_words = {w.lower() for w in query_text.split() if len(w) > 2}
                        for pn in pref_nodes:
                            if pn.id in result_ids:
                                continue
                            if (pn.metadata or {}).get("superseded"):
                                continue
                            content_words = {w.lower() for w in pn.content.split() if len(w) > 2}
                            if query_words & content_words:
                                matching_prefs.append(pn)
                    if matching_prefs:
                        output += "\n---\n**User Preferences:**\n"
                        for pr in matching_prefs[:3]:
                            preview = pr.content[:150]
                            output += f"- [`{pr.id}`] {preview}\n"
            except Exception as e:
                logger.debug("Preference injection failed: %s", e)

        # Warn if embedding model is degraded (hash fallback active)
        try:
            from omega.embedding import is_embedding_degraded
            if is_embedding_degraded() and results:
                output += "\n**Note:** Semantic search is degraded (embedding model unavailable). Results are text-match only.\n"
        except Exception as e:
            logger.warning("Embedding degradation check failed: %s", e)

        logger.info(f"Query '{query_text[:30]}...' returned {len(results)} results")
        return output

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        return f"# Query Error\n\n**Error:** {str(e)}\n"


def query_structured(
    query_text: str,
    limit: int = 10,
    session_id: Optional[str] = None,
    project: Optional[str] = None,
    event_type: Optional[str] = None,
    context_file: Optional[str] = None,
    context_tags: Optional[List[str]] = None,
    filter_tags: Optional[List[str]] = None,
    temporal_range: Optional[tuple] = None,
    entity_id: Optional[str] = None,
    agent_type: Optional[str] = None,
    surfacing_context: Optional["SurfacingContext"] = None,
    strength_min: Optional[float] = None,
    memory_type: Optional[str] = None,
    include_contradicted: bool = False,
    valid_at: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Query memories and return structured dicts (machine-readable)."""
    db = _get_store()

    try:
        effective_temporal = temporal_range or _infer_temporal_range(query_text)
        _temporal_boost_only = temporal_range is None and effective_temporal is not None

        enhanced = query_text
        if event_type:
            enhanced = f"{event_type} {enhanced}"
        if project:
            enhanced = f"{Path(project).name} {enhanced}"

        query_kwargs_s: Dict[str, Any] = {
            "limit": limit * 2 if (filter_tags or entity_id or agent_type) else limit,
            "session_id": session_id,
            "context_file": context_file or "",
            "context_tags": context_tags,
            "temporal_range": effective_temporal,
            "entity_id": entity_id,
            "agent_type": agent_type,
            "query_hint": event_type,
            "surfacing_context": surfacing_context,
            "temporal_boost_only": _temporal_boost_only,
        }
        if valid_at:
            query_kwargs_s["valid_at"] = valid_at
        results = db.query(enhanced, **query_kwargs_s)

        if event_type and results:
            results = [r for r in results if (r.metadata or {}).get("event_type") == event_type]

        # Hard filter by tags (AND logic — all specified tags must be present)
        if filter_tags and results:
            filter_set = {t.lower() for t in filter_tags}
            results = [
                r for r in results if filter_set.issubset({str(t).lower() for t in (r.metadata or {}).get("tags", [])})
            ]

        # Filter by memory_type
        if memory_type and results:
            results = [r for r in results if (r.metadata or {}).get("memory_type") == memory_type]

        # Filter to only contradicted memories
        if include_contradicted and results:
            results = [r for r in results if (r.metadata or {}).get("contradicted_by")]

        results = results[:limit]

        # Filter by minimum strength score
        if strength_min is not None and strength_min > 0:
            results = [r for r in results if getattr(r, "strength", 0.0) >= strength_min]

        structured = []
        for node in results:
            structured.append(
                {
                    "id": node.id,
                    "content": node.content,
                    "event_type": (node.metadata or {}).get("event_type", "memory"),
                    "session_id": (node.metadata or {}).get("session_id", ""),
                    "created_at": node.created_at.isoformat() if node.created_at else "",
                    "tags": (node.metadata or {}).get("tags", []),
                    "metadata": node.metadata,
                    "relevance": getattr(node, "relevance", 0.0),
                    "_query_confidence": (node.metadata or {}).get("_query_confidence", 0.0),
                    "strength": round(getattr(node, "strength", 0.0), 3),
                    "valid_from": node.valid_from.isoformat() if hasattr(node, "valid_from") and node.valid_from else None,
                    "valid_until": node.valid_until.isoformat() if hasattr(node, "valid_until") and node.valid_until else None,
                }
            )

        # Auto-inject relevant constraints
        if event_type != "constraint":
            try:
                result_ids = {node.id for node in results}
                constraint_nodes = db.get_by_type("constraint", limit=10)
                if constraint_nodes:
                    query_words = {w.lower() for w in query_text.split() if len(w) > 2}
                    injected = 0
                    for cn in constraint_nodes:
                        if cn.id in result_ids:
                            continue
                        if (cn.metadata or {}).get("superseded"):
                            continue
                        content_words = {w.lower() for w in cn.content.split() if len(w) > 2}
                        if query_words & content_words:
                            structured.insert(0, {
                                "id": cn.id,
                                "content": cn.content,
                                "event_type": "constraint",
                                "session_id": (cn.metadata or {}).get("session_id", ""),
                                "created_at": cn.created_at.isoformat() if cn.created_at else "",
                                "tags": (cn.metadata or {}).get("tags", []),
                                "metadata": cn.metadata,
                                "relevance": getattr(cn, "relevance", 0.0),
                                "is_constraint": True,
                            })
                            injected += 1
                            if injected >= 3:
                                break
            except Exception as e:
                logger.debug("Constraint injection failed (structured): %s", e)

        # Auto-inject relevant user preferences for preference-intent queries
        _PREF_SIGNAL_WORDS_S = {
            "rule", "rules", "preference", "setting", "configured",
            "should", "allowed", "policy", "default", "location",
            "timezone", "where", "how",
        }
        if event_type != "user_preference":
            try:
                query_words_lower = {w.lower().rstrip("?.,!") for w in query_text.split() if len(w) > 1}
                if query_words_lower & _PREF_SIGNAL_WORDS_S:
                    result_ids = {node.id for node in results}
                    pref_nodes = db.get_by_type("user_preference", limit=20)
                    if pref_nodes:
                        query_words = {w.lower() for w in query_text.split() if len(w) > 2}
                        injected = 0
                        for pn in pref_nodes:
                            if pn.id in result_ids:
                                continue
                            if (pn.metadata or {}).get("superseded"):
                                continue
                            content_words = {w.lower() for w in pn.content.split() if len(w) > 2}
                            if query_words & content_words:
                                structured.insert(0, {
                                    "id": pn.id,
                                    "content": pn.content,
                                    "event_type": "user_preference",
                                    "session_id": (pn.metadata or {}).get("session_id", ""),
                                    "created_at": pn.created_at.isoformat() if pn.created_at else "",
                                    "tags": (pn.metadata or {}).get("tags", []),
                                    "metadata": pn.metadata,
                                    "relevance": getattr(pn, "relevance", 0.0),
                                    "is_preference": True,
                                })
                                injected += 1
                                if injected >= 3:
                                    break
            except Exception as e:
                logger.debug("Preference injection failed (structured): %s", e)

        return structured

    except Exception as e:
        logger.error(f"Structured query failed: {e}", exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Public API -- Welcome / Session Bootstrap
# ---------------------------------------------------------------------------

# Welcome cache — keyed by (project or ""), stores (timestamp, result).
# Avoids 10+ DB round-trips on every session start (daemon serves many sessions).
_welcome_cache: Dict[str, tuple] = {}  # key -> (monotonic_ts, result_dict)
_WELCOME_CACHE_TTL = 30.0  # seconds


def welcome(session_id: Optional[str] = None, project: Optional[str] = None) -> Dict[str, Any]:
    """Generate a session welcome briefing with relevant memories.

    Returns observation_prefix (grouped by type) and project_context
    for Claude to reference throughout the session.

    Uses a 30s cache to avoid 10+ DB round-trips when multiple sessions
    start in quick succession (common with HTTP daemon transport).
    """
    import time as _time_mod

    cache_key = project or ""
    now_mono = _time_mod.monotonic()

    # Fast path: return cached result if fresh
    if cache_key in _welcome_cache:
        cached_ts, cached_result = _welcome_cache[cache_key]
        if now_mono - cached_ts < _WELCOME_CACHE_TTL:
            # Update memory_count (cheap) so it's current
            try:
                db = _get_store()
                cached_result["memory_count"] = db.node_count()
            except Exception:
                pass
            return dict(cached_result)  # shallow copy

    db = _get_store()

    # Get recent high-value memories (decisions, lessons, preferences, errors)
    _HIGH_VALUE_TYPES = {"decision", "lesson_learned", "user_preference", "user_fact", "error_pattern", "constraint"}
    recent = []
    recent_activity = []  # Last few memories of any type for freshness
    try:
        candidates = db.get_recent(limit=200)
        for node in candidates:
            meta = node.metadata or {}
            # Skip superseded memories
            if meta.get("superseded"):
                continue
            event_type = meta.get("event_type", "")
            # Track recent activity (useful types only, up to 5)
            _NOISE_TYPES = {"session_respawn"}
            if len(recent_activity) < 5 and event_type not in _NOISE_TYPES:
                recent_activity.append(node)
            if event_type in _HIGH_VALUE_TYPES:
                recent.append(node)
                if len(recent) >= 30:
                    break
        # If no high-value memories found, fall back to most recent of any type
        if not recent:
            recent = candidates[:5]
    except Exception as e:
        logger.debug("Welcome recent memory filtering failed: %s", e)

    # Ensure user_preference and user_fact are always represented
    # These types have 95-98% never-accessed rates because recency-based
    # selection misses older entries. Direct type queries fix this.
    try:
        _entity_id = None
        if project:
            try:
                from omega_platform.entity.engine import resolve_project_entity
                _entity_id = resolve_project_entity(project)
            except Exception as e:
                logger.debug("Welcome entity resolution failed: %s", e)
        recent_ids = {n.id for n in recent}
        _welcome_types = ("user_preference", "user_fact", "decision", "task_completion", "checkpoint", "session_summary", "behavioral_pattern")
        _limit_per_type = 8
        # Batch query: fetch all 7 types in one SQL call instead of 7 separate queries
        _placeholders = ",".join("?" for _ in _welcome_types)
        if _entity_id:
            _batch_rows = db._conn.execute(
                f"""SELECT node_id, content, metadata, created_at,
                          access_count, last_accessed, ttl_seconds, event_type
                   FROM memories WHERE event_type IN ({_placeholders})
                   AND (entity_id = ? OR entity_id IS NULL)
                   ORDER BY event_type, created_at DESC""",
                (*_welcome_types, _entity_id),
            ).fetchall()
        else:
            _batch_rows = db._conn.execute(
                f"""SELECT node_id, content, metadata, created_at,
                          access_count, last_accessed, ttl_seconds, event_type
                   FROM memories WHERE event_type IN ({_placeholders})
                   ORDER BY event_type, created_at DESC""",
                _welcome_types,
            ).fetchall()
        # Group by event_type and apply per-type limit
        _type_counts: Dict[str, int] = {}
        for row in _batch_rows:
            _et = row[7]  # event_type column
            _type_counts.setdefault(_et, 0)
            if _type_counts[_et] >= _limit_per_type:
                continue
            _type_counts[_et] += 1
            node = db._row_to_result(row[:7])
            if (node.metadata or {}).get("superseded"):
                continue
            if node.id not in recent_ids:
                recent.append(node)
                recent_ids.add(node.id)
    except Exception as e:
        logger.debug("Welcome type-based enrichment failed: %s", e)

    # Sort by blended score: priority * 0.45 + recency * 0.35 + access_boost * 0.20
    # Recency uses 3-day half-life so decisions from days ago aren't pushed out by noise
    _now_ts = _time_mod.time()
    def _recency_score(n):
        if not n.created_at:
            return 0.0
        age_hours = (_now_ts - n.created_at.timestamp()) / 3600.0
        return max(0.0, 1.0 / (1.0 + age_hours / 72.0))

    import math as _math_mod
    def _access_boost(n):
        ac = n.access_count or 0
        return min(_math_mod.log2(1 + ac), 5.0)

    recent.sort(
        key=lambda n: (
            (n.metadata or {}).get("priority", 3) * 0.45
            + _recency_score(n) * 5.0 * 0.35
            + _access_boost(n) * 0.20
        ),
        reverse=True,
    )

    # Build observation_prefix — grouped by type
    observation_prefix = ""
    try:
        grouped: Dict[str, List[str]] = {}
        type_labels = {
            "constraint": "Active Constraints",
            "user_preference": "User Preferences",
            "user_fact": "User Context",
            "decision": "Active Decisions",
            "lesson_learned": "Key Lessons",
            "error_pattern": "Known Pitfalls",
            "task_completion": "Recent Completions",
            "checkpoint": "Saved Checkpoints",
            "session_summary": "Session History",
            "behavioral_pattern": "Behavioral Patterns",
            "project_status": "Project Status",
        }
        for n in recent[:25]:
            etype = (n.metadata or {}).get("event_type", "")
            label = type_labels.get(etype)
            if not label:
                continue
            text = (n.metadata or {}).get("observation") or n.content[:300]
            if label not in grouped:
                grouped[label] = []
            if len(grouped[label]) < 7:
                grouped[label].append(text)

        if grouped:
            _STABLE_LABELS = ["Project Status", "Active Constraints", "User Preferences", "User Context", "Active Decisions", "Key Lessons", "Known Pitfalls", "Behavioral Patterns"]
            _VOLATILE_LABELS = ["Recent Completions", "Saved Checkpoints", "Session History"]

            stable_sections = []
            for label in _STABLE_LABELS:
                items = grouped.get(label, [])
                if items:
                    items_sorted = sorted(items)
                    section = f"### {label}\n"
                    section += "\n".join(f"- {item}" for item in items_sorted)
                    stable_sections.append(section)

            volatile_sections = []
            for label in _VOLATILE_LABELS:
                items = grouped.get(label, [])
                if items:
                    section = f"### {label}\n"
                    section += "\n".join(f"- {item}" for item in items)
                    volatile_sections.append(section)

            if recent_activity:
                activity_items = []
                for n in recent_activity:
                    meta = n.metadata or {}
                    etype = meta.get("event_type", "unknown")
                    text = meta.get("observation") or n.content[:120]
                    ts_str = n.created_at.strftime("%b %d %H:%M") if n.created_at else ""
                    activity_items.append(f"- [{etype}] {ts_str}: {text}")
                volatile_sections.append("### Recent Activity\n" + "\n".join(activity_items))

            all_sections = stable_sections
            if volatile_sections:
                if stable_sections:
                    all_sections = stable_sections + ["<!-- omega:cache_breakpoint -->"] + volatile_sections
                else:
                    all_sections = volatile_sections
            observation_prefix = "\n".join(all_sections)
    except Exception as e:
        logger.debug("Welcome observation_prefix failed: %s", e)

    # Build project_context — skip expensive embedding search (db.query) on fast path.
    # Use only direct type lookups and coordination queries.
    project_context = ""
    _entity_id = None
    if project:
        try:
            from omega_platform.entity.engine import resolve_project_entity
            _entity_id = resolve_project_entity(project)
        except Exception:
            pass

        # Surface latest project_status snapshot
        try:
            status_nodes = db.get_by_type("project_status", limit=3, entity_id=_entity_id)
            project_statuses = [
                n for n in status_nodes
                if (n.metadata or {}).get("project", "") == project
            ]
            if project_statuses:
                latest = project_statuses[0]
                text = latest.content[:400]
                project_context = f"### Project Status\n- {text}\n\n"
        except Exception as e:
            logger.debug("Welcome project_status query failed: %s", e)

        # Surface active coordination decisions for this project
        try:
            from omega_platform.orchestrator.coordination import get_manager
            cm = get_manager()
            coord_decs = cm.query_decisions(project=project, status="active", limit=5)
            if coord_decs:
                dec_items = []
                for d in coord_decs:
                    dec_text = f"[{d['domain']}] {d['decision'][:200]}"
                    dec_items.append(f"- {dec_text}")
                if dec_items:
                    project_context += "\n### Coordination Decisions\n" + "\n".join(dec_items)
        except Exception as e:
            logger.debug("Welcome coord_decisions query failed: %s", e)

    node_count = db.node_count()

    # Dedup stats summary (in-memory, cheap)
    dedup_prevented = 0
    try:
        dedup_prevented = db.stats.get("content_dedup_skips", 0) + db.stats.get("embedding_dedup_skips", 0)
    except Exception:
        pass

    result = {
        "memory_count": node_count,
        "recent_memories": [
            {
                "id": n.id[:12],
                "content": n.content[:400],
                "type": (n.metadata or {}).get("event_type", "unknown"),
            }
            for n in recent[:5]
        ],
        "observation_prefix": observation_prefix,
        "project_context": project_context,
    }

    if dedup_prevented > 0:
        result["duplicates_prevented"] = dedup_prevented

    # NOTE: access_count bumps, behavioral reinforcement, and predictive
    # prefetch removed from welcome() — they caused 60s of DB write
    # contention when the deferred startup thread holds the write lock.
    # Access counts are still updated on actual queries (omega_query).

    # NOTE: Supplementary enrichments (weekly digest, advisor suggestions)
    # removed from welcome() hot path — they trigger query_structured which
    # needs embedding computation + SQLite vector search, blocking 100s+ when
    # the deferred startup thread holds the DB lock during integrity_check.
    # Advisor data is surfaced via the hook system's [HANDOFF] blocks instead.

    _welcome_cache[cache_key] = (now_mono, result)

    return result


def get_session_context(
    project: Optional[str] = None,
    exclude_session: Optional[str] = None,
    limit: int = 5,
) -> Dict[str, Any]:
    """Gather all data needed for session start briefing.

    Returns a dict with context_items (typed high-value memories),
    memory_count, health status, and last_capture_ago.
    """
    db = _get_store()
    node_count = db.node_count()

    # Health status
    health_status = "ok"
    try:
        health = db.check_memory_health()
        health_status = health.get("status", "ok")
    except Exception as e:
        logger.warning("Health check failed: %s", e)
        health_status = "unknown"

    # Last capture time
    last_capture_ago = ""
    try:
        recent_all = db.get_recent(limit=1)
        if recent_all:
            last_capture_ago = _relative_time(recent_all[0].created_at)
    except Exception as e:
        logger.debug("Last capture time query failed: %s", e)

    # Gather typed high-value items for [CONTEXT] section
    from omega.types import STABLE_EVENT_TYPES
    _TYPE_TAG = {
        "constraint": "RULE",
        "user_preference": "PREF",
        "decision": "DECISION",
        "lesson_learned": "LESSON",
        "error_pattern": "PITFALL",
    }
    context_items: list[Dict[str, str]] = []

    # Always-surface constraints (separate budget, not recency-dependent)
    try:
        constraint_nodes = db.get_by_type("constraint", limit=10)
        for node in constraint_nodes:
            if (node.metadata or {}).get("superseded"):
                continue
            text = (node.metadata or {}).get("observation") or node.content[:300]
            text = text.replace("\n", " ").strip()
            context_items.append({"tag": "RULE", "text": text, "stability": "stable"})
            if len(context_items) >= 3:
                break
    except Exception as e:
        logger.debug("Constraint surfacing failed: %s", e)

    # Regular high-value items (separate budget from constraints)
    # Velocity-adaptive freshness: measure memories/day, adjust window accordingly
    try:
        from datetime import datetime as _dt_ctx, timedelta as _td_ctx, timezone as _tz_ctx
        candidates = db.get_recent(limit=100)
        _now_ctx = _dt_ctx.now(_tz_ctx.utc)

        # Compute velocity: count high-value memories in last 24h
        _24h_ago_check = _now_ctx - _td_ctx(hours=24)
        _velocity = 0
        for node in candidates:
            try:
                ca = node.created_at
                if isinstance(ca, str):
                    dt = _dt_ctx.fromisoformat(ca.replace("Z", "+00:00"))
                else:
                    dt = _dt_ctx.fromtimestamp(ca, tz=_tz_ctx.utc)
                if dt >= _24h_ago_check:
                    _velocity += 1
            except (ValueError, TypeError, OSError):
                pass

        # Adaptive windows based on velocity
        if _velocity >= 15:
            # High velocity: tight 12h fresh window
            _fresh_cutoff = _now_ctx - _td_ctx(hours=12)
            _recent_cutoff = _now_ctx - _td_ctx(hours=36)
        elif _velocity >= 5:
            # Moderate velocity: 24h fresh window
            _fresh_cutoff = _now_ctx - _td_ctx(hours=24)
            _recent_cutoff = _now_ctx - _td_ctx(hours=72)
        else:
            # Low velocity: wider 72h fresh window
            _fresh_cutoff = _now_ctx - _td_ctx(hours=72)
            _recent_cutoff = _now_ctx - _td_ctx(hours=168)

        def _node_age_bucket(node):
            """Return 0 (fresh), 1 (recent), 2 (stale) based on velocity-adaptive windows."""
            try:
                ca = node.created_at
                if isinstance(ca, str):
                    dt = _dt_ctx.fromisoformat(ca.replace("Z", "+00:00"))
                else:
                    dt = _dt_ctx.fromtimestamp(ca, tz=_tz_ctx.utc)
                if dt >= _fresh_cutoff:
                    return 0
                elif dt >= _recent_cutoff:
                    return 1
                return 2
            except (ValueError, TypeError, OSError):
                return 2

        # Sort candidates: fresh first, then recent, then stale
        candidates_sorted = sorted(candidates, key=_node_age_bucket)

        seen_tags: Dict[str, int] = {}
        regular_count = 0
        for node in candidates_sorted:
            etype = (node.metadata or {}).get("event_type", "")
            if etype == "constraint":
                continue  # Already handled above
            tag = _TYPE_TAG.get(etype)
            if not tag:
                continue
            if seen_tags.get(tag, 0) >= 2:
                continue
            text = (node.metadata or {}).get("observation") or node.content[:300]
            text = text.replace("\n", " ").strip()
            _stability = "stable" if etype in STABLE_EVENT_TYPES else "volatile"
            context_items.append({"tag": tag, "text": text, "stability": _stability})
            seen_tags[tag] = seen_tags.get(tag, 0) + 1
            regular_count += 1
            if regular_count >= limit:
                break
    except Exception as e:
        logger.debug("get_session_context context_items failed: %s", e)

    # Type breakdown for stats line
    type_stats: Dict[str, int] = {}
    try:
        type_stats = db.get_type_stats()
    except Exception as e:
        logger.debug("get_session_context type_stats failed: %s", e)

    # Count memories added in last 7 days
    period_new_7d = 0
    try:
        cutoff_7d = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        row = db._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE created_at >= ?", (cutoff_7d,)
        ).fetchone()
        period_new_7d = row[0] if row else 0
    except Exception as e:
        logger.debug("get_session_context period_new_7d failed: %s", e)

    return {
        "memory_count": node_count,
        "health_status": health_status,
        "last_capture_ago": last_capture_ago or "unknown",
        "context_items": context_items,
        "type_stats": type_stats,
        "period_new_7d": period_new_7d,
    }


# ---------------------------------------------------------------------------
# Public API -- Health & Status
# ---------------------------------------------------------------------------


def check_health(
    warn_mb: float = 350,
    critical_mb: float = 800,
    max_nodes: int = 10000,
) -> str:
    """Check OMEGA memory health. Returns formatted markdown."""
    db = _get_store()
    health = db.check_memory_health(warn_mb=warn_mb, critical_mb=critical_mb, max_nodes=max_nodes)

    status_label = health.get("status", "unknown").upper()
    parts = [
        f"Status: {status_label} | Mem: {health.get('memory_mb', 0):.1f}MB"
        f" | DB: {health.get('db_size_mb', 0):.2f}MB"
        f" | Nodes: {health.get('node_count', 0)}",
    ]

    warnings = health.get("warnings", [])
    if warnings:
        parts.append("Warnings: " + "; ".join(warnings))

    recommendations = health.get("recommendations", [])
    if recommendations:
        parts.append("Recs: " + "; ".join(recommendations))

    return "\n".join(parts) + "\n"


def status() -> Dict[str, Any]:
    """Return a machine-readable health/status dict."""
    db = _get_store()
    try:
        health = db.check_memory_health()
        return {
            "ok": health.get("status") == "healthy",
            "status": health.get("status", "unknown"),
            "node_count": health.get("node_count", 0),
            "memory_mb": health.get("memory_mb", 0),
            "db_size_mb": health.get("db_size_mb", 0),
            "warnings": health.get("warnings", []),
            "store_path": str(OMEGA_HOME),
            "backend": "sqlite",
            "vec_enabled": health.get("usage", {}).get("vec_enabled", False),
        }
    except Exception as e:
        logger.error(f"Status check failed: {e}", exc_info=True)
        return {"ok": False, "error": str(e)}


def get_dedup_stats() -> Dict[str, Any]:
    """Return deduplication statistics."""
    db = _get_store()
    return {
        "content_dedup_skips": db.stats.get("dedup_canonical", 0) + db.stats.get("dedup_exact", 0),
        "memory_evolutions": db.stats.get("memory_evolutions", 0),
        "embedding_dedup_skips": db.stats.get("embedding_dedup_skips", 0),
        "node_count": db.node_count(),
    }


# ---------------------------------------------------------------------------
# Public API -- Export / Import
# ---------------------------------------------------------------------------


def export_memories(filepath: str) -> str:
    """Export all OMEGA memories to a file."""
    db = _get_store()
    result = db.export_to_file(Path(filepath))

    output = "# OMEGA Export Complete\n\n"
    output += f"**File:** {result.get('filepath', filepath)}\n"
    output += f"**Nodes:** {result.get('node_count', 0)}\n"
    output += f"**Sessions:** {result.get('session_count', 0)}\n"
    output += f"**Size:** {result.get('file_size_kb', 0):.1f} KB\n"
    output += f"**Exported:** {result.get('exported_at', 'now')}\n"

    logger.info(f"Exported OMEGA memories to {filepath}")
    return output


def import_memories(filepath: str, clear_existing: bool = True) -> str:
    """Import OMEGA memories from a file."""
    db = _get_store()
    result = db.import_from_file(Path(filepath), clear_existing=clear_existing)

    output = "# OMEGA Import Complete\n\n"
    output += f"**File:** {result.get('filepath', filepath)}\n"
    output += f"**Nodes Imported:** {result.get('node_count', 0)}\n"
    output += f"**Sessions:** {result.get('session_count', 0)}\n"
    output += f"**Cleared Existing:** {'Yes' if clear_existing else 'No'}\n"

    logger.info(f"Imported OMEGA memories from {filepath}")
    return output


# ---------------------------------------------------------------------------
# Public API -- Deduplication
# ---------------------------------------------------------------------------


def deduplicate(
    event_type: Optional[str] = "lesson_learned",
    similarity_threshold: float = 0.80,
    dry_run: bool = False,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Find and merge duplicate memories using Jaccard similarity."""
    db = _get_store()
    result: Dict[str, Any] = {
        "event_type": event_type or "all",
        "similarity_threshold": similarity_threshold,
        "dry_run": dry_run,
        "groups_found": 0,
        "duplicates_removed": 0,
        "memories_kept": 0,
        "details": [],
    }

    # Gather candidates
    if event_type:
        candidates = db.get_by_type(event_type, limit=500)
    else:
        candidates = db.get_recent(limit=500)

    if session_id:
        candidates = [n for n in candidates if (n.metadata or {}).get("session_id") == session_id]

    if len(candidates) < 2:
        result["message"] = f"Only {len(candidates)} memories found, nothing to deduplicate."
        return result

    # Build word sets
    def _norm(text: str) -> set:
        return {re.sub(r"[^\w]", "", w) for w in text.lower().split() if len(w) > 3}

    node_words = [(node, _norm(node.content)) for node in candidates]

    # Union-find style grouping
    merged_into: Dict[str, str] = {}
    groups: Dict[str, list] = {}

    for i, (node_i, words_i) in enumerate(node_words):
        if node_i.id in merged_into or not words_i:
            continue

        group = [node_i]
        for node_j, words_j in node_words[i + 1:]:
            if node_j.id in merged_into or not words_j:
                continue
            intersection = len(words_i & words_j)
            union = len(words_i | words_j)
            if union and (intersection / union) >= similarity_threshold:
                group.append(node_j)
                merged_into[node_j.id] = node_i.id

        if len(group) > 1:
            groups[node_i.id] = group

    result["groups_found"] = len(groups)

    for _rep_id, group in groups.items():
        group.sort(key=lambda n: len(n.content), reverse=True)
        keeper = group[0]
        duplicates = group[1:]
        total_access = sum(getattr(n, "access_count", 0) or 0 for n in group)

        detail = {
            "kept": {
                "id": keeper.id[:12],
                "content_preview": keeper.content[:100],
                "access_count": total_access,
            },
            "removed": [{"id": n.id[:12], "content_preview": n.content[:80]} for n in duplicates],
            "group_size": len(group),
        }
        result["details"].append(detail)

        if not dry_run:
            db.update_node(keeper.id, access_count=total_access)
            for dup in duplicates:
                try:
                    db.delete_node(dup.id)
                    result["duplicates_removed"] += 1
                except Exception as e:
                    logger.warning(f"Failed to remove duplicate {dup.id[:12]}: {e}")
            result["memories_kept"] += 1

    if not dry_run and result["duplicates_removed"] > 0:
        logger.info(
            f"Deduplication complete: {result['groups_found']} groups, "
            f"{result['duplicates_removed']} removed, "
            f"{result['memories_kept']} kept"
        )

    return result


# ---------------------------------------------------------------------------
# Public API -- Preferences
# ---------------------------------------------------------------------------


def extract_preferences(text: str) -> Dict[str, Any]:
    """Extract user preferences from free text and store them."""
    try:
        from omega.preferences import PreferenceExtractor

        extractor = PreferenceExtractor()
        prefs = extractor.extract(text)
        stored = []
        for pref in prefs:
            auto_capture(
                content=f"[Preference] {pref.get('key', 'unknown')}: {pref.get('value', text[:100])}",
                event_type=AutoCaptureEventType.USER_PREFERENCE,
                metadata={"preference_key": pref.get("key"), "preference_value": pref.get("value")},
            )
            stored.append({"key": pref.get("key"), "stored": True})
        return {"success": True, "preferences": stored, "count": len(stored)}
    except ImportError:
        auto_capture(
            content=f"[Preference] {text[:500]}",
            event_type=AutoCaptureEventType.USER_PREFERENCE,
            metadata={"source": "raw_text"},
        )
        return {"success": True, "preferences": [{"key": "raw", "stored": True}], "count": 1}
    except Exception as e:
        logger.error(f"Preference extraction failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def list_preferences() -> List[Dict[str, Any]]:
    """List stored user preferences."""
    db = _get_store()
    try:
        nodes = db.get_by_type(AutoCaptureEventType.USER_PREFERENCE, limit=100)
        return [
            {
                "id": n.id,
                "content": n.content,
                "created_at": n.created_at.isoformat() if n.created_at else "",
                "metadata": n.metadata or {},
            }
            for n in nodes
        ]
    except Exception as e:
        logger.error(f"list_preferences failed: {e}", exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Public API -- Profile
# ---------------------------------------------------------------------------


def get_profile() -> Dict[str, Any]:
    """Get the user profile from the OMEGA home directory, augmented with preference memories."""
    profile_path = OMEGA_HOME / "profile.json"
    profile: Dict[str, Any] = {}
    try:
        if profile_path.exists():
            with open(profile_path, "r") as f:
                profile = json.loads(f.read())
    except Exception as e:
        logger.debug(f"Failed to load profile: {e}")
    # Augment with preference memories
    try:
        store = _get_store()
        prefs = store.get_by_type("user_preference", limit=20)
        if prefs:
            profile["preferences_from_memory"] = [
                {
                    "content": m.content,
                    "created": m.created_at.isoformat() if hasattr(m.created_at, "isoformat") else str(m.created_at),
                }
                for m in prefs
            ]
    except Exception as e:
        logger.debug(f"Failed to load preference memories: {e}")
    return profile


def save_profile(profile: Dict[str, Any]) -> bool:
    """Persist the user profile to disk (atomic write via temp+rename)."""
    profile_path = OMEGA_HOME / "profile.json"
    try:
        profile_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        import tempfile

        fd, tmp_path = tempfile.mkstemp(dir=profile_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps(profile, indent=2))
            os.replace(tmp_path, profile_path)
        except BaseException:
            os.unlink(tmp_path)
            raise
        return True
    except Exception as e:
        logger.error(f"Failed to save profile: {e}", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Public API -- Cross-session lessons
# ---------------------------------------------------------------------------


def get_cross_session_lessons(
    task: Optional[str] = None,
    project_path: Optional[str] = None,
    exclude_session: Optional[str] = None,
    limit: int = 5,
    agent_type: Optional[str] = None,
    context_file: Optional[str] = None,
    context_tags: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Retrieve top lessons from ALL past sessions for injection."""
    db = _get_store()
    lessons: List[Dict[str, Any]] = []
    seen_content: set = set()
    lesson_sessions: Dict[str, set] = {}

    try:
        if task and (context_file or context_tags):
            # Use full query() for contextual re-ranking when context is available
            enhanced = f"lesson_learned {task}"
            if project_path:
                enhanced = f"{Path(project_path).name} {enhanced}"
            raw = db.query(
                enhanced,
                limit=limit * 3,
                context_file=context_file or "",
                context_tags=context_tags,
                project_path=project_path or "",
            )
            nodes = [r for r in raw if (r.metadata or {}).get("event_type") == "lesson_learned"]
        elif task:
            nodes = db.query_by_type(query=task, event_type="lesson_learned", limit=limit * 3)
        else:
            nodes = db.get_by_type("lesson_learned", limit=limit * 3)

        for node in nodes:
            meta = node.metadata or {}
            if exclude_session and meta.get("session_id") == exclude_session:
                continue
            if agent_type and meta.get("agent_type") != agent_type:
                continue

            key = node.content[:80].lower()
            node_session = meta.get("session_id", "")

            if key in seen_content:
                if node_session and key in lesson_sessions:
                    lesson_sessions[key].add(node_session)
                continue

            seen_content.add(key)
            lesson_sessions[key] = {node_session} if node_session else set()

            lessons.append(
                {
                    "content": node.content,
                    "source": "omega",
                    "lesson_id": meta.get("lesson_id") or node.id,
                    "session_id": node_session,
                    "access_count": getattr(node, "access_count", 0) or 0,
                    "created_at": node.created_at.isoformat() if node.created_at else "",
                    "verified_count": 0,
                    "_key": key,
                }
            )
    except Exception as e:
        logger.debug(f"Lesson query failed: {e}")

    for lesson in lessons:
        key = lesson.get("_key", "")
        session_count = len(lesson_sessions.get(key, set()))
        if session_count > 1:
            lesson["verified_count"] = max(lesson.get("verified_count", 0), session_count)
        lesson["verified"] = lesson.get("verified_count", 0) > 0
        lesson.pop("_key", None)

    lessons.sort(
        key=lambda lesson: (lesson.get("verified_count", 0), lesson.get("access_count", 0)),
        reverse=True,
    )

    return lessons[:limit]


# ---------------------------------------------------------------------------
# Public API -- Trajectory Distillation
# ---------------------------------------------------------------------------


def _get_event_type(m) -> str:
    """Extract event_type from a memory (dict or MemoryResult)."""
    if isinstance(m, dict):
        return m.get("event_type", "unknown")
    return getattr(m, "event_type", "unknown")


def _get_content(m) -> str:
    """Extract content from a memory (dict or MemoryResult)."""
    if isinstance(m, dict):
        return m.get("content", "")
    return getattr(m, "content", "") or ""


def _safe_meta(m) -> dict:
    """Extract metadata dict from a memory (dict or MemoryResult)."""
    if isinstance(m, dict):
        meta = m.get("metadata", {})
    else:
        meta = getattr(m, "metadata", {})
    if isinstance(meta, str):
        try:
            return json.loads(meta)
        except Exception:
            return {}
    return meta or {}


def distill_trajectory(session_id: str) -> Optional[str]:
    """Distill a session's memory trajectory into a reusable skill template.

    Called at session stop. Returns the stored node_id, or None if the session
    didn't pass the quality gate or distillation failed.

    Fail-open: any error results in None (no skill stored), never blocks session stop.
    """
    import json as _json

    try:
        db = _get_store()
        memories = db.get_by_session(session_id, limit=50)

        # Quality gate: minimum 3 memories
        if len(memories) < 3:
            logger.debug("distill_trajectory: skipped session %s (only %d memories)", session_id, len(memories))
            return None

        # Quality gate: must have task_completion event type OR a commit in metadata
        has_completion = any(
            _get_event_type(m) == "task_completion"
            for m in memories
        )
        has_commit = any(
            _safe_meta(m).get("commit")
            for m in memories
        )
        if not has_completion and not has_commit:
            logger.debug("distill_trajectory: skipped session %s (no completion/commit)", session_id)
            return None

        # Gather trajectory context (chronological — oldest first)
        memories = list(reversed(memories))  # get_by_session returns DESC
        mem_lines = []
        for m in memories:
            et = _get_event_type(m)
            content = _get_content(m)[:200]
            mem_lines.append(f"- [{et}] {content}")

        trajectory_text = "\n".join(mem_lines[:20])  # Cap at 20 entries

        # Gather tool sequence from coord_audit if available
        tool_sequence = ""
        try:
            from omega_platform.orchestrator.coordination import get_manager
            mgr = get_manager()
            if mgr:
                audit_rows = mgr._conn.execute(
                    "SELECT tool_name, result_status FROM coord_audit "
                    "WHERE session_id = ? ORDER BY call_index ASC LIMIT 30",
                    (session_id,),
                ).fetchall()
                if audit_rows:
                    tools = [f"{r[0]}({'ok' if r[1] == 'ok' else 'err'})" for r in audit_rows]
                    tool_sequence = f"\nTool sequence: {' → '.join(tools)}"
        except Exception:
            pass  # Coordination unavailable — continue without tool sequence

        # LLM distillation call
        system_prompt = (
            "You extract reusable skill templates from agent work sessions. "
            "Output valid JSON only, no markdown fencing."
        )
        user_prompt = f"""Analyze this agent session and extract a reusable skill template.

Memory sequence (chronological):
{trajectory_text}
{tool_sequence}

Extract a JSON skill template:
{{
  "skill_type": "debugging|feature|refactor|config|deploy",
  "summary": "One sentence describing the workflow in imperative form",
  "steps": ["verb_phrase_1", "verb_phrase_2", ...],
  "key_insight": "The most important actionable lesson from this session",
  "tools_used": ["Tool1", "Tool2"],
  "files_involved": ["path1", "path2"],
  "outcome": "success|partial|failed_then_recovered"
}}

Rules:
- Steps should be abstract enough to transfer (not "edit auth.py line 42" but "apply null-safe fix")
- key_insight should be actionable advice, not a description
- 3-7 steps maximum
- If the session is too routine or trivial to extract a skill, return {{"skip": true}}"""

        raw = llm_complete(
            prompt=user_prompt,
            system=system_prompt,
            max_tokens=512,
            temperature=0.0,
            timeout=10.0,
            model_tier="fast",
        )

        if not raw:
            logger.debug("distill_trajectory: LLM returned empty for session %s", session_id)
            return None

        # Parse JSON (strip markdown fencing if present)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        try:
            skill = _json.loads(cleaned)
        except _json.JSONDecodeError:
            logger.debug("distill_trajectory: malformed JSON from LLM for session %s", session_id)
            return None

        # Handle skip response
        if skill.get("skip"):
            logger.debug("distill_trajectory: LLM said skip for session %s", session_id)
            return None

        # Validate required fields
        required = ("skill_type", "summary", "steps", "key_insight")
        if not all(skill.get(k) for k in required):
            logger.debug("distill_trajectory: missing required fields for session %s", session_id)
            return None

        # Build content string (human-readable)
        steps_str = " → ".join(skill["steps"])
        files_str = ", ".join(skill.get("files_involved", [])[:5])
        content = (
            f"{skill['summary']}. "
            f"Steps: {steps_str}. "
            f"Insight: {skill['key_insight']}"
        )
        if files_str:
            content += f". Files: {files_str}"

        # Build metadata
        meta = {
            "source": "trajectory_distillation",
            "session_id": session_id,
            "skill_type": skill["skill_type"],
            "steps": skill["steps"],
            "tools_used": skill.get("tools_used", []),
            "files_involved": skill.get("files_involved", []),
            "key_insight": skill["key_insight"],
            "outcome": skill.get("outcome", "success"),
            "memory_count": len(memories),
            "distillation_model": "haiku",
        }

        node_id = auto_capture(
            content=content,
            event_type="skill_template",
            metadata=meta,
            session_id=session_id,
        )

        logger.info("distill_trajectory: distilled %s skill from session %s → %s",
                     skill["skill_type"], session_id, node_id)
        return node_id

    except Exception as e:
        logger.debug("distill_trajectory: failed for session %s: %s", session_id, e)
        return None


# ---------------------------------------------------------------------------
# Public API -- Constraint Enforcement
# ---------------------------------------------------------------------------

CONSTRAINTS_DIR = OMEGA_HOME / "constraints"


def _load_constraints(project: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load constraint rules for a project from ~/.omega/constraints/.

    Loads global.json first, then <project-name>.json if project is given.
    Returns merged list of rule dicts.
    """
    rules: List[Dict[str, Any]] = []
    if not CONSTRAINTS_DIR.exists():
        return rules

    # Global constraints
    global_file = CONSTRAINTS_DIR / "global.json"
    if global_file.exists():
        try:
            data = json.loads(global_file.read_text())
            for r in data.get("rules", []):
                r["source"] = "global"
                rules.append(r)
        except Exception as e:
            logger.debug(f"Failed to load global constraints: {e}")

    # Project-specific constraints
    if project:
        proj_name = Path(project).name
        proj_file = CONSTRAINTS_DIR / f"{proj_name}.json"
        if proj_file.exists():
            try:
                data = json.loads(proj_file.read_text())
                for r in data.get("rules", []):
                    r["source"] = proj_name
                    rules.append(r)
            except Exception as e:
                logger.debug(f"Failed to load {proj_name} constraints: {e}")

    return rules


def check_constraints(file_path: str, project: Optional[str] = None) -> List[Dict[str, Any]]:
    """Check a file path against loaded constraint rules.

    Returns list of matching constraints with severity and message.
    """
    import fnmatch

    rules = _load_constraints(project)
    if not rules:
        return []

    matches = []
    filename = os.path.basename(file_path)

    for rule in rules:
        pattern = rule.get("pattern", "")
        if not pattern:
            continue
        # Match against filename or full path
        if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(file_path, pattern):
            matches.append(
                {
                    "pattern": pattern,
                    "constraint": rule.get("constraint", ""),
                    "severity": rule.get("severity", "warn"),
                    "source": rule.get("source", "unknown"),
                }
            )

    return matches


def list_constraints(project: Optional[str] = None) -> Dict[str, Any]:
    """List all loaded constraint rules for a project."""
    rules = _load_constraints(project)
    return {
        "count": len(rules),
        "rules": rules,
        "constraints_dir": str(CONSTRAINTS_DIR),
    }


def save_constraints(
    rules: List[Dict[str, Any]],
    project: Optional[str] = None,
) -> Dict[str, Any]:
    """Save constraint rules to the appropriate file.

    If project is given, saves to <project-name>.json, else global.json.
    """
    CONSTRAINTS_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)

    if project:
        target = CONSTRAINTS_DIR / f"{Path(project).name}.json"
    else:
        target = CONSTRAINTS_DIR / "global.json"

    # Clean source field from rules before saving
    clean_rules = []
    for r in rules:
        clean = {k: v for k, v in r.items() if k != "source"}
        clean_rules.append(clean)

    data = {"rules": clean_rules}
    target.write_text(json.dumps(data, indent=2))

    return {"saved": str(target), "count": len(clean_rules)}


# ---------------------------------------------------------------------------
# Public API -- Cross-project Learning
# ---------------------------------------------------------------------------


def get_cross_project_lessons(
    task: Optional[str] = None,
    exclude_project: Optional[str] = None,
    exclude_session: Optional[str] = None,
    limit: int = 5,
    agent_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve lessons from ALL projects (cross-project knowledge transfer).

    Unlike get_cross_session_lessons which may filter by project,
    this explicitly queries without project scope to find patterns
    that recur across different codebases.
    """
    db = _get_store()
    lessons: List[Dict[str, Any]] = []
    seen_content: set = set()
    project_sets: Dict[str, set] = {}

    try:
        if task:
            nodes = db.query_by_type(query=task, event_type="lesson_learned", limit=limit * 4)
        else:
            nodes = db.get_by_type("lesson_learned", limit=limit * 4)

        for node in nodes:
            meta = node.metadata or {}
            node_project = meta.get("project", "")

            if exclude_session and meta.get("session_id") == exclude_session:
                continue
            if exclude_project and node_project == exclude_project:
                continue
            if agent_type and meta.get("agent_type") != agent_type:
                continue

            key = node.content[:80].lower()

            if key in seen_content:
                if node_project and key in project_sets:
                    project_sets[key].add(node_project)
                continue

            seen_content.add(key)
            project_sets[key] = {node_project} if node_project else set()

            lessons.append(
                {
                    "content": node.content,
                    "source_project": node_project,
                    "lesson_id": meta.get("lesson_id") or node.id,
                    "session_id": meta.get("session_id", ""),
                    "access_count": getattr(node, "access_count", 0) or 0,
                    "created_at": node.created_at.isoformat() if node.created_at else "",
                    "projects_seen": 1,
                    "_key": key,
                }
            )
    except Exception as e:
        logger.debug(f"Cross-project lesson query failed: {e}")

    # Enrich with cross-project counts
    for lesson in lessons:
        key = lesson.get("_key", "")
        proj_count = len(project_sets.get(key, set()))
        lesson["projects_seen"] = max(1, proj_count)
        lesson["cross_project"] = proj_count > 1
        lesson.pop("_key", None)

    # Sort by cross-project occurrence, then access count
    lessons.sort(
        key=lambda lesson: (lesson.get("projects_seen", 0), lesson.get("access_count", 0)),
        reverse=True,
    )

    return lessons[:limit]


# ---------------------------------------------------------------------------
# Public API -- Reingest (legacy JSONL → SQLite)
# ---------------------------------------------------------------------------


def reingest(
    store_path: Optional[Path] = None,
    batch_size: int = 50,
    skip_types: Optional[set] = None,
) -> Dict[str, Any]:
    """Bulk-load JSONL store entries into SQLite.

    Reads every line from store.jsonl and inserts into the SQLite database.
    Content-hash dedup prevents duplicates automatically.
    """
    db = _get_store()
    src = store_path or (OMEGA_HOME / "store.jsonl")

    if not src.exists():
        return {"error": f"Store file not found: {src}", "ingested": 0}

    skip_types = skip_types or set()
    stats = {"ingested": 0, "skipped": 0, "duplicates": 0, "errors": 0, "total": 0}

    logger.info(f"Reingesting from {src}")

    from omega.crypto import decrypt_line

    with open(src, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            stats["total"] += 1

            try:
                entry = json.loads(decrypt_line(line))
            except Exception as e:
                logger.debug("Import line parse failed at line %d: %s", line_num, e)
                stats["errors"] += 1
                continue

            content = entry.get("content", "").strip()
            if not content:
                stats["skipped"] += 1
                continue

            meta = entry.get("metadata", {})
            event_type = meta.get("event_type", "memory")

            if event_type in skip_types:
                stats["skipped"] += 1
                continue

            session_id = meta.get("session_id")
            ttl = TTLCategory.for_event_type(event_type)

            try:
                db.store(
                    content=content[:2000],
                    session_id=session_id,
                    metadata=meta,
                    ttl_seconds=ttl,
                    skip_inference=True,
                )
                stats["ingested"] += 1
            except Exception as e:
                stats["errors"] += 1
                if stats["errors"] <= 5:
                    logger.warning(f"Reingest error line {line_num}: {e}")

            if stats["ingested"] > 0 and stats["ingested"] % batch_size == 0:
                logger.info(f"  Progress: {stats['ingested']} ingested, {stats['total']} processed")

    logger.info(
        f"Reingest complete: {stats['ingested']} ingested, "
        f"{stats['duplicates']} duplicates, {stats['errors']} errors "
        f"out of {stats['total']} entries"
    )
    return stats


# ---------------------------------------------------------------------------
# Public API -- Feedback
# ---------------------------------------------------------------------------


def record_feedback(
    memory_id: str,
    rating: str,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Record feedback on a surfaced memory."""
    db = _get_store()
    return db.record_feedback(node_id=memory_id, rating=rating, reason=reason)


def batch_record_feedback(items: List[tuple]) -> int:
    """Record feedback for multiple memories in a single transaction.

    Each item is (node_id, rating, reason). Returns count of updated memories.
    """
    db = _get_store()
    return db.batch_record_feedback(items)


def _check_graduation(memory_id: str) -> Optional[str]:
    """Check if a memory should graduate or decay based on diff-correlation history.

    Graduation: memory was diff-correlated (positive) in 2+ feedback signals -> promote priority.
    Decay: memory was surfaced 3+ times with zero correlation -> demote priority.

    Reads from the feedback_signals list stored in memory metadata by record_feedback().

    Returns "graduated", "decayed", or None.
    """
    db = _get_store()
    try:
        row = db._conn.execute(
            "SELECT metadata FROM memories WHERE node_id = ?",
            (memory_id,),
        ).fetchone()

        if not row or not row[0]:
            return None

        meta = json.loads(row[0]) if isinstance(row[0], str) else (row[0] or {})
        signals = meta.get("feedback_signals", [])

        if not signals:
            return None

        diff_positive = sum(
            1 for s in signals
            if s.get("rating") == "helpful" and s.get("reason") and "diff-correlated" in s["reason"]
        )
        surfaced_not_committed = sum(
            1 for s in signals
            if s.get("rating") == "unhelpful" and s.get("reason") and "not committed" in s["reason"]
        )

        if diff_positive >= 2:
            # Graduate: boost priority
            db._conn.execute(
                "UPDATE memories SET priority = MIN(COALESCE(priority, 3) + 1, 5) WHERE node_id = ?",
                (memory_id,),
            )
            db._conn.commit()
            return "graduated"
        elif surfaced_not_committed >= 3 and diff_positive == 0:
            # Decay: reduce priority
            db._conn.execute(
                "UPDATE memories SET priority = MAX(COALESCE(priority, 3) - 1, 1) WHERE node_id = ?",
                (memory_id,),
            )
            db._conn.commit()
            return "decayed"

        return None
    except Exception as e:
        logger.debug("_check_graduation failed for %s: %s", memory_id[:12], e)
        return None


def backfill_embeddings(batch_size: int = 50) -> dict:
    """Backfill missing embeddings for memories not in memories_vec."""
    db = _get_store()
    return db.backfill_embeddings(batch_size=batch_size)


# ---------------------------------------------------------------------------
# Public API -- Session management
# ---------------------------------------------------------------------------


def clear_session(session_id: str) -> Dict[str, Any]:
    """Clear all memories for a session."""
    db = _get_store()
    count = db.clear_session(session_id)
    logger.info(f"Cleared session {session_id[:16]}: {count} memories removed")
    return {"session_id": session_id, "removed": count}


# ---------------------------------------------------------------------------
# Public API -- Batch operations
# ---------------------------------------------------------------------------


def batch_store(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Store multiple memories efficiently."""
    db = _get_store()
    ids = db.batch_store(items)
    return {"ids": ids, "count": len(ids)}


# ---------------------------------------------------------------------------
# Public API -- Similar memories
# ---------------------------------------------------------------------------


def find_similar_memories(memory_id: str, limit: int = 5) -> str:
    """Find memories similar to a given memory via vector search."""
    db = _get_store()
    node = db.get_node(memory_id)
    if node is None:
        return f"Memory `{memory_id}` not found."
    embedding = db.get_embedding(memory_id)
    if embedding is None:
        return f"No embedding found for `{memory_id[:12]}`. Vector search unavailable."
    # limit+1 because the source memory will be in results
    results = db.find_similar(embedding, limit=limit + 1)
    # Filter out the source memory itself
    results = [r for r in results if r.id != memory_id][:limit]
    # Format output
    output = f"# Similar Memories ({len(results)})\n\n"
    output += f"**Source:** `{memory_id[:12]}` — {node.content[:100]}\n\n"
    for i, r in enumerate(results, 1):
        ntype = (r.metadata or {}).get("event_type", "memory")
        preview = r.content[:200]
        output += f"## {i}. [{ntype}] `{r.id[:12]}` (similarity: {r.relevance:.2f})\n"
        output += f"{preview}\n\n"
    if not results:
        output += "*No similar memories found.*\n"
    return output


# ---------------------------------------------------------------------------
# Public API -- Timeline
# ---------------------------------------------------------------------------


def timeline(days: int = 7, limit_per_day: int = 10) -> str:
    """Show memory timeline grouped by day."""
    db = _get_store()
    data = db.get_timeline(days=days, limit_per_day=limit_per_day)
    if not data:
        return f"No memories in the last {days} days."
    total = sum(len(v) for v in data.values())
    output = f"Timeline ({total} memories, last {days}d)\n\n"
    for day in sorted(data.keys(), reverse=True):
        memories = data[day]
        output += f"{day} ({len(memories)})\n"
        for m in memories:
            etype = (m.metadata or {}).get("event_type", "memory")
            preview = m.content[:120].replace("\n", " ")
            output += f"- [{etype}] {preview} ({m.id[:8]} {m.created_at.strftime('%H:%M')})\n"
        output += "\n"
    return output


# ---------------------------------------------------------------------------
# Public API -- Consolidation
# ---------------------------------------------------------------------------


def _auto_backup_before_consolidate():
    """Create a backup before consolidation (rotate to keep last 3)."""
    db_path = OMEGA_HOME / "omega.db"
    if not db_path.exists():
        return
    try:
        import sqlite3
        from omega.crypto import secure_connect

        backups_dir = OMEGA_HOME / "backups"
        backups_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        backup_path = backups_dir / f"pre-consolidate-{timestamp}.db"
        src = sqlite3.connect(str(db_path))
        dst = secure_connect(backup_path)
        src.backup(dst)
        dst.close()
        src.close()
        logger.info(f"Pre-consolidation backup: {backup_path}")
        # Rotate — keep only last 3
        backups = sorted(backups_dir.glob("pre-consolidate-*.db"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in backups[3:]:
            old.unlink()
    except Exception as e:
        logger.warning(f"Auto-backup before consolidation failed: {e}")


def consolidate(prune_days: int = 14, max_summaries: int = 50) -> str:
    """Run memory consolidation: prune stale entries, cap summaries, clean edges.

    Returns formatted markdown report.
    """
    _auto_backup_before_consolidate()
    db = _get_store()
    before = db.node_count()
    stats = db.consolidate(prune_days=prune_days, max_summaries=max_summaries)
    after = stats.get("node_count_after", before)
    removed = before - after

    output = "# Memory Consolidation Report\n\n"
    output += f"**Before:** {before} memories\n"
    output += f"**After:** {after} memories\n"
    output += f"**Removed:** {removed} total\n\n"
    output += "## Breakdown\n\n"
    output += f"- **Stale (0 access, >{prune_days}d old):** {stats.get('pruned_stale', 0)}\n"
    output += f"- **Session summaries (beyond cap of {max_summaries}):** {stats.get('pruned_summaries', 0)}\n"
    output += f"- **Orphaned edges:** {stats.get('pruned_edges', 0)}\n"
    output += f"- **Strength-decayed:** {stats.get('decayed_memories', 0)}\n"
    output += f"- **Merged entities:** {stats.get('merged_entities', 0)}\n"

    if removed == 0:
        output += "\n*Nothing to consolidate — memory store is clean.*\n"
    else:
        logger.info(f"Consolidation: removed {removed} memories ({stats})")

    return output


# ---------------------------------------------------------------------------
# Public API -- Memory Compaction
# ---------------------------------------------------------------------------


def _smart_extract(cluster) -> str:
    """Extract diverse, information-dense sentences from a cluster of memories.

    Scores sentences by: unique-word count (IDF-like), sentence length
    (diminishing returns), presence of proper nouns / code tokens, and
    cross-memory term frequency (words appearing in 2+ cluster members
    are more generalizable — ALMA-inspired strategy extraction).

    For large clusters (5+), extracts a strategy header from the most
    common bigram theme across cluster members.

    Skips near-duplicate sentences (Jaccard > 0.7).
    Orders selected sentences chronologically by source memory.
    Returns consolidated text capped at 1000 chars.
    """
    # Build cross-memory word frequency map (words appearing in 2+ members)
    from collections import Counter
    word_to_members: dict = {}  # word -> set of node indices
    for idx, node in enumerate(cluster):
        for w in set(node.content.lower().split()):
            if len(w) > 3:
                word_to_members.setdefault(w, set()).add(idx)
    cross_freq_words = {w for w, members in word_to_members.items() if len(members) >= 2}

    # Collect all sentences with source metadata
    all_sentences = []  # [(sentence, density_score, created_at)]
    seen_keys: set = set()

    for node in cluster:
        created = node.created_at.isoformat() if node.created_at else ""
        for sentence in re.split(r"(?<=[.!?])\s+", node.content):
            sentence = sentence.strip()
            if len(sentence) < 15:
                continue
            key = " ".join(sentence.lower().split())[:100]
            if key in seen_keys:
                continue
            seen_keys.add(key)

            words = sentence.split()
            unique_words = len(set(w.lower() for w in words if len(w) > 3))

            # Proper nouns / capitalized words (not sentence-start)
            proper_nouns = len([w for w in words[1:] if w[0].isupper()]) if len(words) > 1 else 0

            # Code tokens: backtick spans, paths, CamelCase
            code_tokens = len(re.findall(r"`[^`]+`|/[\w/.]+|\b[A-Z][a-z]+[A-Z]\w*\b", sentence))

            # Diminishing returns on length
            length_score = min(len(sentence), 200) / 200.0

            # Cross-memory term frequency boost (ALMA-inspired)
            cross_freq = sum(1 for w in words if w.lower() in cross_freq_words)

            density = (unique_words * 1.0 + proper_nouns * 1.5 + code_tokens * 2.0
                       + length_score * 3.0 + cross_freq * 0.8)
            all_sentences.append((sentence, density, created))

    if not all_sentences:
        return ""

    # Sort by density (highest first)
    all_sentences.sort(key=lambda x: x[1], reverse=True)

    # Select top-K diverse sentences
    selected = []
    for sentence, _score, created in all_sentences:
        # Check diversity against already selected
        is_diverse = all(_jaccard(sentence.lower(), sel[0].lower(), min_word_len=3) < 0.7 for sel in selected)
        if is_diverse:
            selected.append((sentence, created))
            if len(selected) >= 8:  # Max sentences to consider
                break

    # Order chronologically by source memory created_at
    selected.sort(key=lambda x: x[1])

    # Build consolidated text (cap at 1000 chars)
    consolidated = " ".join(s for s, _ in selected)

    # Strategy header for large clusters (5+ members): extract common bigram theme
    if len(cluster) >= 5:
        bigram_counter: Counter = Counter()
        for node in cluster:
            words = [w.lower() for w in node.content.split() if len(w) > 3]
            for w1, w2 in zip(words, words[1:]):
                bigram_counter[(w1, w2)] += 1
        if bigram_counter:
            top_bigram, top_count = bigram_counter.most_common(1)[0]
            if top_count >= 3:  # Only if bigram appears in 3+ members
                theme = f"{top_bigram[0]} {top_bigram[1]}"
                consolidated = f"Strategy: {theme}. {consolidated}"

    if len(consolidated) > 1000:
        consolidated = consolidated[:997] + "..."

    return consolidated


def compact(
    event_type: str = "lesson_learned",
    similarity_threshold: float = 0.60,
    min_cluster_size: int = 3,
    dry_run: bool = False,
) -> str:
    """Compact clusters of related memories into consolidated knowledge nodes.

    Unlike deduplicate() which removes exact/near duplicates, compact() finds
    clusters of semantically related memories and creates new summary nodes
    that consolidate the knowledge, marking originals as superseded.

    Returns formatted markdown report.
    """
    db = _get_store()
    all_candidates = db.get_by_type(event_type, limit=500)
    # Filter out superseded memories — these were already compacted into a
    # consolidated node.  Re-including them causes nested "[Consolidated from]"
    # prefixes and duplicate consolidated nodes.
    candidates = [
        n for n in all_candidates
        if not (n.metadata or {}).get("superseded")
    ]

    if len(candidates) < min_cluster_size:
        return (
            f"# Memory Compaction\n\n"
            f"Only {len(candidates)} `{event_type}` memories found "
            f"(minimum cluster size: {min_cluster_size}). Nothing to compact.\n"
        )

    # Build word sets for Jaccard clustering
    def _norm(text: str) -> set:
        return {re.sub(r"[^\w]", "", w) for w in text.lower().split() if len(w) > 3}

    node_words = [(node, _norm(node.content)) for node in candidates]

    # Union-find style clustering
    assigned: set = set()
    clusters: List[List] = []

    for i in range(len(node_words)):
        if len(assigned) >= len(node_words):
            break  # All items assigned, no more clusters possible
        node_i, words_i = node_words[i]
        if node_i.id in assigned or not words_i:
            continue

        cluster = [node_i]
        assigned.add(node_i.id)

        for j in range(i + 1, len(node_words)):
            node_j, words_j = node_words[j]
            if node_j.id in assigned or not words_j:
                continue
            intersection = len(words_i & words_j)
            union = len(words_i | words_j)
            if union and (intersection / union) >= similarity_threshold:
                cluster.append(node_j)
                assigned.add(node_j.id)

        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)

    if not clusters:
        return (
            f"# Memory Compaction\n\n"
            f"No clusters found with >= {min_cluster_size} similar `{event_type}` memories "
            f"at {similarity_threshold:.0%} similarity. Store is already compact.\n"
        )

    # Build report and optionally perform compaction
    output = f"# Memory Compaction {'(DRY RUN)' if dry_run else 'Report'}\n\n"
    output += f"**Event type:** {event_type}\n"
    output += f"**Similarity threshold:** {similarity_threshold:.0%}\n"
    output += f"**Clusters found:** {len(clusters)}\n\n"

    total_compacted = 0
    total_created = 0

    for ci, cluster in enumerate(clusters, 1):
        # Sort by content length (longest first — most information)
        cluster.sort(key=lambda n: len(n.content), reverse=True)

        consolidated = _smart_extract(cluster)

        # Merge tags from all cluster members
        merged_tags: set = set()
        total_access = 0
        for node in cluster:
            merged_tags.update(str(t) for t in (node.metadata or {}).get("tags", []))
            total_access += getattr(node, "access_count", 0) or 0

        output += f"## Cluster {ci} ({len(cluster)} memories)\n\n"
        output += f"**Summary:** {consolidated[:200]}...\n"
        for node in cluster[:5]:
            preview = node.content[:80]
            output += f"- `{node.id[:12]}`: {preview}\n"
        if len(cluster) > 5:
            output += f"- ... and {len(cluster) - 5} more\n"
        output += "\n"

        if not dry_run:
            # Strip any existing "[Consolidated from ...]" prefix from the
            # extracted content to prevent nested consolidation headers.
            consolidated = re.sub(
                r"^(\[Consolidated from \d+ memories\]\s*)+",
                "",
                consolidated,
            ).lstrip()
            # Prefix consolidated content to distinguish from originals (avoids dedup)
            compact_header = f"[Consolidated from {len(cluster)} memories] "
            compact_content = compact_header + consolidated

            # Create the consolidated node with quality metadata
            # Quality scale: 1.0 (min cluster) to 3.0 (10+ members)
            consolidation_quality = min(3.0, 1.0 + (len(cluster) - min_cluster_size) * 0.3)
            meta = {
                "event_type": event_type,
                "source": "compaction",
                "compacted_from": [n.id for n in cluster],
                "compacted_count": len(cluster),
                "tags": sorted(merged_tags)[:15],
                "consolidation_quality": round(consolidation_quality, 2),
            }
            new_id = db.store(
                content=compact_content,
                metadata=meta,
                ttl_seconds=TTLCategory.for_event_type(event_type),
                skip_inference=True,  # Bypass embedding dedup
            )
            db.update_node(new_id, access_count=total_access)

            # Mark originals as superseded + log to forgetting audit trail
            for node in cluster:
                nmeta = dict(node.metadata or {})
                nmeta["superseded"] = True
                nmeta["superseded_by"] = new_id
                nmeta["compacted_at"] = datetime.now(timezone.utc).isoformat()
                db.update_node(node.id, metadata=nmeta)
                db._log_forgetting_external(
                    node.id, node.content, event_type,
                    "compaction_superseded", {"superseded_by": new_id},
                )
                db.queue_cloud_delete_by_node_id(node.id)

            total_compacted += len(cluster)
            total_created += 1
            output += f"**Created:** `{new_id[:12]}` | **Superseded:** {len(cluster)} memories\n\n"

    output += "---\n"
    if dry_run:
        output += f"**Would compact:** {sum(len(c) for c in clusters)} memories into {len(clusters)} nodes\n"
    else:
        output += f"**Compacted:** {total_compacted} memories into {total_created} consolidated nodes\n"

    return output


# ---------------------------------------------------------------------------
# Public API -- Active Connection Discovery (Consolidation Daemon)
# ---------------------------------------------------------------------------


def discover_connections(
    lookback_hours: int = 24,
    similarity_threshold: float = 0.70,
    max_memories: int = 100,
    max_connections_per_memory: int = 3,
    dry_run: bool = False,
) -> str:
    """Actively discover and link related memories that aren't yet connected.

    Scans recent memories, finds semantically similar ones that lack edges,
    and creates 'related' edges between them. When cross-cutting patterns
    are found (clusters spanning multiple event types), generates
    advisor_insight entries.

    This is the core of the active consolidation daemon — it generates
    new knowledge from existing memories rather than just pruning.

    Args:
        lookback_hours: How far back to scan for unlinked memories.
        similarity_threshold: Minimum cosine similarity to create an edge (0.0-1.0).
        max_memories: Maximum memories to process per run.
        max_connections_per_memory: Maximum new edges per memory.
        dry_run: If True, report what would be linked without modifying.

    Returns:
        Formatted markdown report.
    """
    db = _get_store()

    if not db._vec_available:
        return "# Connection Discovery\n\nVector search unavailable — cannot discover connections.\n"

    # Phase 1: Find recent memories without many edges
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=lookback_hours)).isoformat()
    candidates = db._conn.execute(
        """SELECT m.node_id, m.content, m.event_type, m.id, m.created_at,
                  m.entity_id, m.status
           FROM memories m
           WHERE m.created_at > ?
             AND (m.status IS NULL OR m.status = 'active')
             AND m.event_type NOT IN ('session_summary', 'coordination_snapshot',
                                       'session_respawn', 'code_chunk', 'file_summary')
           ORDER BY m.created_at DESC
           LIMIT ?""",
        (cutoff, max_memories),
    ).fetchall()

    if not candidates:
        return (
            f"# Connection Discovery\n\n"
            f"No recent active memories in the last {lookback_hours}h to analyze.\n"
        )

    # Phase 2: For each candidate, find similar memories and create edges
    edges_created = 0
    edges_skipped = 0
    cross_type_clusters = []  # Track cross-type connections for insight generation

    # Get existing edges for candidates to avoid redundant checks
    candidate_ids = {c[0] for c in candidates}
    existing_edges = set()
    if candidate_ids:
        placeholders = ",".join("?" * len(candidate_ids))
        rows = db._conn.execute(
            f"SELECT source_id, target_id FROM edges WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})",
            list(candidate_ids) + list(candidate_ids),
        ).fetchall()
        for r in rows:
            existing_edges.add((r[0], r[1]))
            existing_edges.add((r[1], r[0]))  # Bidirectional check

    report_lines = []

    for node_id, content, event_type, rowid, created_at, entity_id, status in candidates:
        # Get embedding for this memory
        try:
            emb_row = db._conn.execute(
                "SELECT embedding FROM memories_vec WHERE rowid = ?", (rowid,)
            ).fetchone()
            if not emb_row:
                continue
        except Exception:
            continue

        # Find similar memories
        import struct

        from omega.sqlite_store import EMBEDDING_DIM as _EMBED_DIM

        expected_size = _EMBED_DIM * 4  # 4 bytes per float
        if len(emb_row[0]) != expected_size:
            continue
        embedding = list(struct.unpack(f"{_EMBED_DIM}f", emb_row[0]))
        similar = db._vec_query(embedding, limit=max_connections_per_memory + 5)

        connections_made = 0
        for sim_rowid, distance in similar:
            if connections_made >= max_connections_per_memory:
                break

            similarity = 1.0 - distance
            if similarity < similarity_threshold:
                continue

            # Look up the similar memory
            sim_row = db._conn.execute(
                "SELECT node_id, event_type, content FROM memories WHERE id = ?",
                (sim_rowid,),
            ).fetchone()
            if not sim_row or sim_row[0] == node_id:
                continue

            sim_node_id, sim_event_type, sim_content = sim_row

            # Skip if edge already exists
            if (node_id, sim_node_id) in existing_edges:
                edges_skipped += 1
                continue

            # Create the edge
            if not dry_run:
                db.add_edge(
                    source_id=node_id,
                    target_id=sim_node_id,
                    edge_type="related",
                    weight=round(similarity, 3),
                    metadata={"source": "discover_connections", "auto": True},
                )

            existing_edges.add((node_id, sim_node_id))
            existing_edges.add((sim_node_id, node_id))
            edges_created += 1
            connections_made += 1

            # Track cross-type connections for insight generation
            if event_type != sim_event_type:
                cross_type_clusters.append({
                    "source_id": node_id,
                    "source_type": event_type,
                    "source_preview": content[:80],
                    "target_id": sim_node_id,
                    "target_type": sim_event_type,
                    "target_preview": sim_content[:80],
                    "similarity": round(similarity, 3),
                })

            report_lines.append(
                f"  - `{node_id[:16]}` ({event_type}) ↔ `{sim_node_id[:16]}` "
                f"({sim_event_type}) [{similarity:.0%}]"
            )

    # Phase 3: Generate insights from cross-type patterns
    insights_generated = 0
    insight_lines = []

    if cross_type_clusters and not dry_run:
        # Group cross-type connections by type pairs
        type_pairs: Dict[tuple, list] = {}
        for conn in cross_type_clusters:
            pair = tuple(sorted([conn["source_type"], conn["target_type"]]))
            type_pairs.setdefault(pair, []).append(conn)

        # Generate insight for type pairs with 3+ connections
        for pair, connections in type_pairs.items():
            if len(connections) >= 3:
                previews = [
                    f"- {c['source_preview']}... ↔ {c['target_preview']}..."
                    for c in connections[:5]
                ]
                insight_content = (
                    f"Cross-cutting pattern: {len(connections)} connections discovered "
                    f"between {pair[0]} and {pair[1]} memories.\n"
                    f"Examples:\n" + "\n".join(previews)
                )
                try:
                    auto_capture(
                        content=insight_content,
                        event_type="advisor_insight",
                        metadata={
                            "category": "system_insight",
                            "source": "discover_connections",
                            "type_pair": list(pair),
                            "connection_count": len(connections),
                        },
                        entity_id="omega",
                    )
                    insights_generated += 1
                    insight_lines.append(
                        f"  - {pair[0]} ↔ {pair[1]}: {len(connections)} connections"
                    )
                except Exception as e:
                    logger.debug("Failed to store cross-type insight: %s", e)

    # Format report
    mode = "(DRY RUN) " if dry_run else ""
    output = f"# Connection Discovery {mode}Report\n\n"
    output += f"**Scanned:** {len(candidates)} memories (last {lookback_hours}h)\n"
    output += f"**New edges:** {edges_created}\n"
    output += f"**Skipped (existing):** {edges_skipped}\n"
    output += f"**Cross-type insights:** {insights_generated}\n\n"

    if report_lines:
        output += "## Connections\n"
        output += "\n".join(report_lines[:30])
        if len(report_lines) > 30:
            output += f"\n  ... and {len(report_lines) - 30} more\n"
        output += "\n\n"

    if insight_lines:
        output += "## Cross-Type Patterns\n"
        output += "\n".join(insight_lines)
        output += "\n\n"

    if not report_lines and not insight_lines:
        output += "*No new connections found. Memories are already well-linked or too diverse.*\n"

    return output


# ---------------------------------------------------------------------------
# Public API -- System Insight Synthesis
# ---------------------------------------------------------------------------


def synthesize_system_insights(
    similarity_threshold: float = 0.50,
    min_cluster_size: int = 3,
    dry_run: bool = True,
) -> str:
    """Synthesize clusters of system insights into consolidated subsystem briefs.

    Like compact() but scoped to advisor_insight memories with category=system_insight.
    Consolidated nodes inherit the system_insight category and permanent TTL.

    Args:
        similarity_threshold: Jaccard similarity threshold for clustering (lower = broader clusters).
        min_cluster_size: Minimum insights in a cluster to trigger synthesis.
        dry_run: If True, report what would be synthesized without modifying anything.

    Returns:
        Formatted markdown report.
    """
    db = _get_store()
    all_insights = db.get_by_type("advisor_insight", limit=500)

    # Filter to system_insight category
    candidates = []
    for node in all_insights:
        meta = node.metadata or {}
        if meta.get("category") == "system_insight":
            candidates.append(node)

    if len(candidates) < min_cluster_size:
        return (
            f"# System Insight Synthesis\n\n"
            f"Only {len(candidates)} system insights found "
            f"(minimum cluster size: {min_cluster_size}). Nothing to synthesize.\n"
        )

    # Jaccard clustering (same algorithm as compact())
    def _norm(text: str) -> set:
        return {re.sub(r"[^\w]", "", w) for w in text.lower().split() if len(w) > 3}

    node_words = [(node, _norm(node.content)) for node in candidates]
    assigned: set = set()
    clusters: List[List] = []

    for i in range(len(node_words)):
        if len(assigned) >= len(node_words):
            break
        node_i, words_i = node_words[i]
        if node_i.id in assigned or not words_i:
            continue

        cluster = [node_i]
        assigned.add(node_i.id)

        for j in range(i + 1, len(node_words)):
            node_j, words_j = node_words[j]
            if node_j.id in assigned or not words_j:
                continue
            intersection = len(words_i & words_j)
            union = len(words_i | words_j)
            if union and (intersection / union) >= similarity_threshold:
                cluster.append(node_j)
                assigned.add(node_j.id)

        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)

    if not clusters:
        return (
            f"# System Insight Synthesis\n\n"
            f"No clusters found with >= {min_cluster_size} similar system insights "
            f"at {similarity_threshold:.0%} similarity. Insights are already diverse.\n"
        )

    output = f"# System Insight Synthesis {'(DRY RUN)' if dry_run else 'Report'}\n\n"
    output += f"**System insights:** {len(candidates)}\n"
    output += f"**Clusters found:** {len(clusters)}\n\n"

    total_compacted = 0
    total_created = 0

    for ci, cluster in enumerate(clusters, 1):
        cluster.sort(key=lambda n: len(n.content), reverse=True)
        consolidated = _smart_extract(cluster)

        # Merge tags from all cluster members
        merged_tags: set = set()
        total_access = 0
        for node in cluster:
            merged_tags.update(str(t) for t in (node.metadata or {}).get("tags", []))
            total_access += getattr(node, "access_count", 0) or 0

        # Identify primary subsystem from most common tag
        primary_subsystem = max(merged_tags, key=lambda t: sum(
            1 for n in cluster if t in (n.metadata or {}).get("tags", [])
        )) if merged_tags else "general"

        output += f"## Cluster {ci}: {primary_subsystem} ({len(cluster)} insights)\n\n"
        output += f"**Summary:** {consolidated[:300]}...\n"
        for node in cluster[:5]:
            preview = node.content[:80]
            output += f"- `{node.id[:12]}`: {preview}\n"
        if len(cluster) > 5:
            output += f"- ... and {len(cluster) - 5} more\n"
        output += "\n"

        if not dry_run:
            compact_header = f"[Subsystem brief: {primary_subsystem}] "
            compact_content = compact_header + consolidated

            meta = {
                "event_type": "advisor_insight",
                "category": "system_insight",
                "source": "insight_synthesis",
                "subsystem": primary_subsystem,
                "compacted_from": [n.id for n in cluster],
                "compacted_count": len(cluster),
                "tags": sorted(merged_tags)[:15],
            }
            new_id = db.store(
                content=compact_content,
                metadata=meta,
                ttl_seconds=None,  # Permanent
                skip_inference=True,
            )
            db.update_node(new_id, access_count=total_access)

            # Mark originals as superseded
            for node in cluster:
                nmeta = dict(node.metadata or {})
                nmeta["superseded"] = True
                nmeta["superseded_by"] = new_id
                nmeta["synthesized_at"] = datetime.now(timezone.utc).isoformat()
                db.update_node(node.id, metadata=nmeta)
                db._log_forgetting_external(
                    node.id, node.content, "advisor_insight",
                    "insight_synthesis_superseded", {"superseded_by": new_id},
                )
                db.queue_cloud_delete_by_node_id(node.id)

            total_compacted += len(cluster)
            total_created += 1
            output += f"**Created:** `{new_id[:12]}` | **Superseded:** {len(cluster)} insights\n\n"

    output += "---\n"
    if dry_run:
        output += f"**Would synthesize:** {sum(len(c) for c in clusters)} insights into {len(clusters)} subsystem briefs\n"
    else:
        output += f"**Synthesized:** {total_compacted} insights into {total_created} subsystem briefs\n"

    return output


# ---------------------------------------------------------------------------
# Public API -- Forgetting Audit Trail
# ---------------------------------------------------------------------------


def forgetting_log(limit: int = 50, reason: Optional[str] = None) -> str:
    """Retrieve the forgetting audit log as formatted markdown."""
    db = _get_store()
    entries = db.get_forgetting_log(limit=limit, reason=reason)

    if not entries:
        return "# Forgetting Log\n\nNo forgetting events recorded yet.\n"

    output = "# Forgetting Log\n\n"
    if reason:
        output += f"**Filter:** reason = `{reason}`\n\n"
    output += f"**Entries:** {len(entries)}\n\n"
    output += "| Time | Reason | Type | Node | Preview |\n"
    output += "|------|--------|------|------|---------|\n"

    for entry in entries:
        deleted = entry["deleted_at"][:19] if entry.get("deleted_at") else "?"
        reason_str = entry.get("reason", "?")
        et = entry.get("event_type", "") or ""
        nid = entry.get("node_id", "")[:12]
        preview = (entry.get("content_preview") or "")[:60].replace("|", "/").replace("\n", " ")
        output += f"| {deleted} | `{reason_str}` | {et} | `{nid}` | {preview} |\n"

    return output


# ---------------------------------------------------------------------------
# Public API -- Graph Traversal
# ---------------------------------------------------------------------------


def traverse(
    memory_id: str,
    max_hops: int = 2,
    min_weight: float = 0.0,
    edge_types: Optional[List[str]] = None,
) -> str:
    """Traverse the relationship graph from a starting memory.

    Walks the `related` edges table up to max_hops, returning all
    connected memories with their hop distance and edge weight.

    Returns formatted markdown string.
    """
    db = _get_store()
    node = db.get_node(memory_id)
    if node is None:
        return f"Memory `{memory_id}` not found."

    results = db.get_related_chain(
        start_id=memory_id,
        max_hops=max_hops,
        min_weight=min_weight,
        edge_types=edge_types,
    )

    output = f"# Graph Traversal ({len(results)} connected memories)\n\n"
    output += f"**Start:** `{memory_id[:12]}` — {node.content[:100]}\n"
    output += f"**Max hops:** {max_hops}\n\n"

    if not results:
        output += "*No connected memories found.*\n"
        return output

    current_hop = 0
    for r in results:
        if r["hop"] != current_hop:
            current_hop = r["hop"]
            output += f"## Hop {current_hop}\n\n"

        etype = (r.get("metadata") or {}).get("event_type", "memory")
        preview = r["content"][:200]
        output += f"- **[{etype}]** `{r['node_id'][:12]}` (weight: {r['weight']:.2f}, edge: {r['edge_type']})\n"
        output += f"  {preview}\n\n"

    return output


# ---------------------------------------------------------------------------
# Public API -- Phrase Search
# ---------------------------------------------------------------------------


def phrase_search(
    phrase: str,
    limit: int = 10,
    event_type: Optional[str] = None,
    project: Optional[str] = None,
    case_sensitive: bool = False,
) -> str:
    """Search memories for exact phrase matches using FTS5.

    Returns formatted markdown string.
    """
    db = _get_store()
    try:
        results = db.phrase_search(
            phrase=phrase,
            limit=limit,
            event_type=event_type,
            case_sensitive=case_sensitive,
            project_path=project or "",
        )

        output = f"# Phrase Search Results ({len(results)})\n\n"
        output += f'**Phrase:** "{phrase}"\n'
        if event_type:
            output += f"**Event Type:** {event_type}\n"
        output += "\n"

        if results:
            for i, node in enumerate(results[:limit], 1):
                ntype = (node.metadata or {}).get("event_type", "memory")
                preview = node.content[:200] + "..." if len(node.content) > 200 else node.content
                output += f"## {i}. [{ntype}] `{node.id}`\n"
                output += f"{preview}\n"
                tags = (node.metadata or {}).get("tags", [])
                if tags:
                    output += f"*Tags: {', '.join(str(t) for t in tags[:5])}*\n"
                output += f"*Created: {node.created_at.isoformat()[:16]}*\n\n"
        else:
            output += "*No matching memories found.*\n"

        return output

    except Exception as e:
        logger.error(f"Phrase search failed: {e}", exc_info=True)
        return f"# Phrase Search Error\n\n**Error:** {str(e)}\n"


# ---------------------------------------------------------------------------
# Public API -- Stats
# ---------------------------------------------------------------------------


def type_stats() -> Dict[str, int]:
    """Get memory counts grouped by event type."""
    db = _get_store()
    return db.get_type_stats()


def stats_card_data() -> Dict[str, Any]:
    """Get data for the shareable stats card display."""
    db = _get_store()
    return db.get_stats_card_data()


def session_stats() -> Dict[str, int]:
    """Get memory counts grouped by session ID."""
    db = _get_store()
    return db.get_session_stats()


def retrieval_context() -> List[Dict[str, Any]]:
    """Return recent retrieval context entries for diagnostics."""
    return _get_store().get_retrieval_context()


def access_rate_stats() -> Dict[str, Any]:
    """Get access rate breakdown: never-accessed count, by-type, top accessed."""
    db = _get_store()
    total = db.node_count()

    zero_access = db._conn.execute(
        "SELECT COUNT(*) FROM memories WHERE access_count = 0"
    ).fetchone()[0]
    never_accessed_pct = (zero_access / total * 100) if total > 0 else 0

    # Retrieval count (semantic search hits) — separate from access_count
    zero_retrieval = db._conn.execute(
        "SELECT COUNT(*) FROM memories WHERE COALESCE(retrieval_count, 0) = 0"
    ).fetchone()[0]
    never_retrieved_pct = (zero_retrieval / total * 100) if total > 0 else 0

    # Breakdown by event_type: avg access_count + retrieval_count per type
    type_rows = db._conn.execute(
        """SELECT event_type, COUNT(*) as cnt,
                  AVG(access_count) as avg_access,
                  SUM(CASE WHEN access_count = 0 THEN 1 ELSE 0 END) as zero_cnt,
                  AVG(COALESCE(retrieval_count, 0)) as avg_retrieval,
                  SUM(CASE WHEN COALESCE(retrieval_count, 0) = 0 THEN 1 ELSE 0 END) as zero_retr_cnt
           FROM memories
           GROUP BY event_type
           ORDER BY avg_access DESC"""
    ).fetchall()
    by_type = []
    for row in type_rows:
        by_type.append({
            "event_type": row[0] or "unknown",
            "count": row[1],
            "avg_access_count": round(row[2], 2),
            "zero_access_count": row[3],
            "zero_access_pct": round(row[3] / row[1] * 100, 1) if row[1] > 0 else 0,
            "avg_retrieval_count": round(row[4], 2),
            "zero_retrieval_count": row[5],
            "zero_retrieval_pct": round(row[5] / row[1] * 100, 1) if row[1] > 0 else 0,
        })

    # Top 10 most-accessed memories
    top_rows = db._conn.execute(
        """SELECT node_id, content, access_count, event_type
           FROM memories
           WHERE access_count > 0
           ORDER BY access_count DESC LIMIT 10"""
    ).fetchall()
    top_accessed = []
    for row in top_rows:
        top_accessed.append({
            "id": row[0],
            "content": row[1][:100],
            "access_count": row[2],
            "event_type": row[3] or "unknown",
        })

    # Overall average — computed from per-type aggregates (no extra query)
    _total_count = sum(row[1] for row in type_rows)
    avg_access = round(
        sum(row[2] * row[1] for row in type_rows) / _total_count, 2
    ) if _total_count else 0

    return {
        "total_memories": total,
        "zero_access_count": zero_access,
        "never_accessed_pct": round(never_accessed_pct, 1),
        "zero_retrieval_count": zero_retrieval,
        "never_retrieved_pct": round(never_retrieved_pct, 1),
        "avg_access_count": avg_access,
        "by_type": by_type,
        "top_accessed": top_accessed,
    }


# ---------------------------------------------------------------------------
# Public API -- Unified Diagnostic Report
# ---------------------------------------------------------------------------


def diagnostic_report(days: int = 30) -> Dict[str, Any]:
    """Unified OMEGA health and value diagnostic.

    Aggregates data from memory store, coordination audit, session tracking,
    and LLM usage into a single report with a computed verdict.
    """
    report: Dict[str, Any] = {}

    # --- 1. Memory Health ---------------------------------------------------
    db = _get_store()
    rate_stats = access_rate_stats()

    # Velocity: memories created in last 7 days by event type
    velocity_rows = db._conn.execute(
        """SELECT event_type, COUNT(*) FROM memories
           WHERE created_at > datetime('now', '-7 days')
           GROUP BY event_type ORDER BY COUNT(*) DESC"""
    ).fetchall()
    velocity = [{"event_type": r[0], "count": r[1]} for r in velocity_rows]
    week_total = sum(r[1] for r in velocity_rows)

    # Dead memories: never accessed, older than 14 days
    dead_row = db._conn.execute(
        """SELECT COUNT(*) FROM memories
           WHERE access_count = 0 AND created_at < datetime('now', '-14 days')"""
    ).fetchone()
    dead_count = dead_row[0] if dead_row else 0
    total = rate_stats["total_memories"]
    dead_pct = (dead_count / max(total, 1)) * 100

    # Access buckets
    bucket_row = db._conn.execute(
        """SELECT
             SUM(CASE WHEN access_count = 0 THEN 1 ELSE 0 END),
             SUM(CASE WHEN access_count BETWEEN 1 AND 2 THEN 1 ELSE 0 END),
             SUM(CASE WHEN access_count BETWEEN 3 AND 9 THEN 1 ELSE 0 END),
             SUM(CASE WHEN access_count >= 10 THEN 1 ELSE 0 END)
           FROM memories"""
    ).fetchone()
    access_buckets = {
        "never": bucket_row[0] or 0,
        "low_1_2": bucket_row[1] or 0,
        "medium_3_9": bucket_row[2] or 0,
        "high_10_plus": bucket_row[3] or 0,
    }

    report["memory_health"] = {
        "total": total,
        "hit_rate_pct": round(100 - rate_stats["never_accessed_pct"], 1),
        "velocity_7d": velocity,
        "velocity_total_7d": week_total,
        "dead_memories": dead_count,
        "dead_pct": round(dead_pct, 1),
        "access_buckets": access_buckets,
        "avg_access_count": rate_stats["avg_access_count"],
    }

    # --- 2. Tool Usage (from coord_audit) -----------------------------------
    tool_usage: Dict[str, Any] = {"top_tools": [], "omega_tools": [], "total_calls": 0, "omega_calls": 0}
    try:
        from omega_platform.orchestrator.coordination import get_manager
        mgr = get_manager()
        if mgr:
            # Top 20 tools by call count
            top_rows = mgr._conn.execute(
                """SELECT tool_name, COUNT(*) as calls,
                          AVG(latency_ms) as avg_latency
                   FROM coord_audit
                   WHERE created_at > datetime('now', '-' || ? || ' days')
                   GROUP BY tool_name ORDER BY calls DESC LIMIT 20""",
                (days,),
            ).fetchall()
            tool_usage["top_tools"] = [
                {"tool": r[0], "calls": r[1], "avg_latency_ms": round(r[2]) if r[2] else None}
                for r in top_rows
            ]

            # Total call count
            total_row = mgr._conn.execute(
                """SELECT COUNT(*) FROM coord_audit
                   WHERE created_at > datetime('now', '-' || ? || ' days')""",
                (days,),
            ).fetchone()
            tool_usage["total_calls"] = total_row[0] if total_row else 0

            # OMEGA-specific tools
            omega_rows = mgr._conn.execute(
                """SELECT tool_name, COUNT(*) FROM coord_audit
                   WHERE tool_name LIKE 'mcp__omega%'
                     AND created_at > datetime('now', '-' || ? || ' days')
                   GROUP BY tool_name ORDER BY COUNT(*) DESC""",
                (days,),
            ).fetchall()
            tool_usage["omega_tools"] = [{"tool": r[0], "calls": r[1]} for r in omega_rows]
            tool_usage["omega_calls"] = sum(r[1] for r in omega_rows)
    except Exception as e:
        logger.debug("diagnostic: coord_audit unavailable: %s", e)
    report["tool_usage"] = tool_usage

    # --- 3. Session Activity ------------------------------------------------
    sessions: Dict[str, Any] = {"total": 0, "week": 0, "month": 0}
    try:
        from omega_platform.orchestrator.coordination import get_manager
        mgr = get_manager()
        if mgr:
            sess_row = mgr._conn.execute(
                """SELECT
                     COUNT(*),
                     SUM(CASE WHEN started_at > datetime('now', '-7 days') THEN 1 ELSE 0 END),
                     SUM(CASE WHEN started_at > datetime('now', '-30 days') THEN 1 ELSE 0 END)
                   FROM coord_sessions"""
            ).fetchone()
            if sess_row:
                sessions = {
                    "total": sess_row[0] or 0,
                    "week": sess_row[1] or 0,
                    "month": sess_row[2] or 0,
                }
    except Exception as e:
        logger.debug("diagnostic: coord_sessions unavailable: %s", e)
    report["sessions"] = sessions

    # --- 4. LLM Costs -------------------------------------------------------
    llm_costs: Dict[str, Any] = {}
    try:
        from omega.usage_tracker import UsageTracker
        tracker = UsageTracker()
        llm_costs = tracker.get_cost_estimate(days=days)
        llm_costs["by_model"] = tracker.get_usage(days=days, group_by="model")
        tracker.close()
    except Exception as e:
        logger.debug("diagnostic: usage_tracker unavailable: %s", e)
    report["llm_costs"] = llm_costs

    # --- 5. Value Assessment ------------------------------------------------
    hit_rate = report["memory_health"]["hit_rate_pct"]
    omega_calls = tool_usage["omega_calls"]
    total_calls = tool_usage["total_calls"]

    verdict = "idle"
    if hit_rate > 60 and omega_calls > 50:
        verdict = "healthy"
    elif hit_rate > 40 and omega_calls >= 5:
        verdict = "underused"

    report["value_assessment"] = {
        "memory_hit_rate": f"{hit_rate:.0f}%",
        "memory_velocity": f"{week_total} new in 7 days",
        "dead_memory_pct": f"{dead_pct:.0f}%",
        "omega_tool_calls": omega_calls,
        "total_tool_calls": total_calls,
        "omega_usage_pct": f"{omega_calls / max(total_calls, 1) * 100:.1f}%",
        "verdict": verdict,
    }

    report["period_days"] = days
    return report


# ---------------------------------------------------------------------------
# Public API -- Weekly Knowledge Digest
# ---------------------------------------------------------------------------


def get_weekly_digest(days: int = 7) -> Dict[str, Any]:
    """Generate a weekly knowledge digest with stats, trends, and highlights.

    Returns dict with: summary, type_breakdown, top_topics, growth, highlights.
    """
    db = _get_store()
    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(days=days)).isoformat()
    prev_cutoff = (now - timedelta(days=days * 2)).isoformat()

    total = db.node_count()

    # Delegate all period queries to the store's single-lock method
    stats = db.get_period_stats(cutoff=cutoff, prev_cutoff=prev_cutoff)
    period_count = stats["period_count"]
    type_breakdown = stats["type_breakdown"]
    session_count = stats["session_count"]
    prev_count = stats["prev_period_count"]

    # Top topics: extract most common words from recent content (simple TF)
    _STOP_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "out", "off", "over",
        "under", "again", "further", "then", "once", "and", "but", "or", "nor",
        "not", "so", "yet", "both", "each", "few", "more", "most", "other",
        "some", "such", "no", "only", "own", "same", "than", "too", "very",
        "just", "because", "if", "when", "while", "how", "what", "which",
        "who", "whom", "this", "that", "these", "those", "it", "its", "my",
        "your", "his", "her", "our", "their", "all", "any", "up", "about",
        "error", "memory", "session", "plan", "decision", "captured",
    }
    top_topics: list[str] = []
    word_counts: Dict[str, int] = {}
    for content in stats["content_samples"]:
        words = re.findall(r'[a-zA-Z_]{4,}', content.lower())
        for w in words:
            if w not in _STOP_WORDS:
                word_counts[w] = word_counts.get(w, 0) + 1
    top_topics = [w for w, _ in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:8]]

    growth_pct = ((period_count - prev_count) / max(prev_count, 1)) * 100 if prev_count > 0 else 0

    # Oldest memory recalled this week
    oldest_recalled_days = None
    try:
        oldest_recalled_days = db.get_oldest_accessed_since(cutoff)
    except Exception as e:
        logger.debug("get_weekly_digest oldest_recalled failed: %s", e)

    return {
        "period_days": days,
        "total_memories": total,
        "period_new": period_count,
        "session_count": session_count,
        "type_breakdown": type_breakdown,
        "top_topics": top_topics,
        "growth_pct": round(growth_pct, 1),
        "prev_period_count": prev_count,
        "oldest_recalled_days": oldest_recalled_days,
    }


# ---------------------------------------------------------------------------
# Public API -- Activity Summary (CLI)
# ---------------------------------------------------------------------------


def get_activity_summary(days: int = 7) -> Dict[str, Any]:
    """Gather activity data for the CLI activity command.

    Returns: {sessions: [...], tasks: [...], insights: [...], claims: [...]}
    """
    result: Dict[str, Any] = {"sessions": [], "tasks": [], "insights": [], "claims": []}

    # Recent insights from timeline
    try:
        db = _get_store()
        data = db.get_timeline(days=days, limit_per_day=10)
        if data:
            for day in sorted(data.keys(), reverse=True):
                for m in data[day]:
                    etype = (m.metadata or {}).get("event_type", "memory")
                    preview = m.content[:120].replace("\n", " ")
                    result["insights"].append(
                        {
                            "type": etype,
                            "preview": preview,
                            "created_at": m.created_at.isoformat() if m.created_at else "",
                            "id": m.id[:12] if m.id else "",
                        }
                    )
            # Limit to 15 most recent across all days
            result["insights"] = result["insights"][:15]
    except Exception as e:
        logger.warning(f"Activity summary: insights failed: {e}")

    # Coordination data (sessions, tasks, claims)
    try:
        from omega_platform.orchestrator.coordination import get_manager

        mgr = get_manager()

        # Sessions (active)
        try:
            sessions = mgr.list_sessions(auto_clean=False)
            for s in sessions:
                result["sessions"].append(
                    {
                        "session_id": s.get("session_id", "")[:16],
                        "project": s.get("project", ""),
                        "task": s.get("task", ""),
                        "started_at": s.get("started_at", ""),
                        "last_heartbeat": s.get("last_heartbeat", ""),
                        "status": s.get("status", ""),
                    }
                )
        except Exception as e:
            logger.warning(f"Activity summary: sessions failed: {e}")

        # Tasks (pending + in_progress)
        try:
            for st in ("pending", "in_progress"):
                tasks = mgr.list_tasks(status=st)
                for t in tasks:
                    result["tasks"].append(
                        {
                            "id": t.get("id", ""),
                            "title": t.get("title", ""),
                            "status": t.get("status", ""),
                            "progress": t.get("progress", 0),
                            "created_at": t.get("created_at", ""),
                        }
                    )
        except Exception as e:
            logger.warning(f"Activity summary: tasks failed: {e}")

        # File + branch claims across active sessions
        try:
            for s in sessions:
                sid = s.get("session_id", "")
                claims = mgr.get_session_claims(sid)
                for fp in claims.get("file_claims", []):
                    result["claims"].append({"type": "file", "path": fp, "session": sid[:16]})
                for br in claims.get("branch_claims", []):
                    result["claims"].append({"type": "branch", "path": br, "session": sid[:16]})
        except Exception as e:
            logger.warning(f"Activity summary: claims failed: {e}")

    except ImportError:
        logger.info("Coordination module not available for activity summary")
    except Exception as e:
        logger.warning(f"Activity summary: coordination failed: {e}")

    return result


# ---------------------------------------------------------------------------
# Reminders (experimental)
# ---------------------------------------------------------------------------

# Regex for parsing human-friendly durations: "1h", "30m", "2d", "1w", "1d12h", "2 hours"
_DURATION_RE = re.compile(
    r"(?:(\d+)\s*w(?:eeks?)?)?\s*"
    r"(?:(\d+)\s*d(?:ays?)?)?\s*"
    r"(?:(\d+)\s*h(?:ours?|rs?)?)?\s*"
    r"(?:(\d+)\s*m(?:in(?:utes?|s?)?)?)?",
)


def parse_duration(text: str) -> timedelta:
    """Parse a human-friendly duration string into a timedelta.

    Supported formats: "1h", "30m", "2d", "1w", "1d12h", "2 hours", "30 minutes".
    Raises ValueError on invalid or zero duration.
    """
    text = text.strip().lower()
    m = _DURATION_RE.fullmatch(text)
    if not m or not any(m.groups()):
        raise ValidationError(f"Invalid duration: {text!r}. Use e.g. '1h', '30m', '2d', '1w', '1d12h'.")
    weeks = int(m.group(1) or 0)
    days = int(m.group(2) or 0)
    hours = int(m.group(3) or 0)
    minutes = int(m.group(4) or 0)
    td = timedelta(weeks=weeks, days=days, hours=hours, minutes=minutes)
    if td.total_seconds() <= 0:
        raise ValidationError("Duration must be positive.")
    return td


def create_reminder(
    text: str,
    duration: str,
    context: Optional[str] = None,
    session_id: Optional[str] = None,
    project: Optional[str] = None,
) -> dict:
    """Create a time-based reminder.

    Stores directly via SQLiteStore.store() to bypass dedup/evolution —
    identical reminder text with different times should create separate entries.
    """
    td = parse_duration(duration)
    now = datetime.now(timezone.utc)
    remind_at = now + td

    meta = {
        "event_type": "reminder",
        "reminder_status": "pending",
        "remind_at": remind_at.isoformat(),
        "created_at_utc": now.isoformat(),
        "notified_out_of_session": False,
    }
    if context:
        meta["context"] = context
    if session_id:
        meta["session_id"] = session_id
    if project:
        meta["project"] = project

    # Include remind_at in content to avoid content-hash dedup
    # (same text at different times = different reminders)
    store_content = f"{text}\n[due: {remind_at.isoformat()}]"

    db = _get_store()
    node_id = db.store(
        content=store_content,
        session_id=session_id,
        metadata=meta,
        ttl_seconds=None,  # Permanent until dismissed
        skip_inference=True,  # Skip embedding dedup — same text, different times = different reminders
    )

    # Human-readable local time
    try:
        local_str = remind_at.astimezone().strftime("%Y-%m-%d %H:%M %Z")
    except Exception as e:
        logger.debug("Timezone conversion failed: %s", e)
        local_str = remind_at.isoformat()

    return {
        "reminder_id": node_id,
        "text": text,
        "remind_at": remind_at.isoformat(),
        "remind_at_local": local_str,
        "duration": duration,
    }


def list_reminders(
    status: Optional[str] = None,
    include_dismissed: bool = False,
    entity_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List reminders, sorted by overdue first then by remind_at ascending.

    Args:
        entity_id: If provided, only return reminders scoped to this entity.
    """
    db = _get_store()
    sql = "SELECT node_id, content, metadata, created_at FROM memories WHERE event_type = 'reminder'"
    params: list = []
    if entity_id:
        sql += " AND COALESCE(entity_id, '') = ?"
        params.append(entity_id)
    with db._lock:
        rows = db._conn.execute(sql, params).fetchall()

    now = datetime.now(timezone.utc)
    # Regex to strip the internal [due: ...] suffix from stored content
    _due_suffix_re = re.compile(r"\n\[due: [^\]]+\]$")

    results = []
    for node_id, content, meta_json, created_at in rows:
        try:
            meta = json.loads(meta_json) if isinstance(meta_json, str) else (meta_json or {})
        except (json.JSONDecodeError, TypeError):
            meta = {}

        r_status = meta.get("reminder_status", "pending")

        # Filter out superseded reminders (safety net for Phase 4.5)
        # But keep superseded reminders that are overdue — if the superseding
        # reminder also hasn't fired, the user still needs to be notified.
        if meta.get("superseded") and not include_dismissed and status != "all":
            remind_at_str_check = meta.get("remind_at", "")
            try:
                remind_at_check = datetime.fromisoformat(remind_at_str_check)
                if remind_at_check.tzinfo is None:
                    remind_at_check = remind_at_check.replace(tzinfo=timezone.utc)
                is_overdue_check = now >= remind_at_check and r_status == "pending"
            except (ValueError, TypeError):
                is_overdue_check = False
            if not is_overdue_check:
                continue

        # Filter by status
        if status and status != "all" and r_status != status:
            continue
        if not include_dismissed and not status and r_status == "dismissed":
            continue

        remind_at_str = meta.get("remind_at", "")
        try:
            remind_at = datetime.fromisoformat(remind_at_str)
            if remind_at.tzinfo is None:
                remind_at = remind_at.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            remind_at = now

        is_due = now >= remind_at
        is_overdue = is_due and r_status == "pending"
        time_until = remind_at - now

        try:
            remind_at_local = remind_at.astimezone().strftime("%Y-%m-%d %H:%M %Z")
        except Exception as e:
            logger.debug("Timezone conversion failed: %s", e)
            remind_at_local = remind_at.isoformat()

        # Strip internal [due: ...] suffix for clean display
        clean_text = _due_suffix_re.sub("", content)

        # Compute age since creation for staleness detection
        try:
            created_dt = datetime.fromisoformat(created_at) if created_at else now
            if created_dt.tzinfo is None:
                created_dt = created_dt.replace(tzinfo=timezone.utc)
            pending_days = (now - created_dt).days
        except (ValueError, TypeError):
            pending_days = 0

        results.append({
            "id": node_id,
            "text": clean_text,
            "status": r_status,
            "remind_at": remind_at.isoformat(),
            "remind_at_local": remind_at_local,
            "is_due": is_due,
            "is_overdue": is_overdue,
            "pending_days": pending_days,
            "time_until": str(time_until).split(".")[0] if not is_due else "overdue",
            "context": meta.get("context"),
            "created_at": created_at,
        })

    # Sort: overdue first, then by remind_at ascending
    results.sort(key=lambda r: (not r["is_overdue"], r["remind_at"]))
    return results


def dismiss_reminder(reminder_id: str) -> Dict[str, Any]:
    """Dismiss a reminder by updating its status."""
    db = _get_store()
    node = db.get_node(reminder_id)
    if node is None:
        return {"success": False, "error": f"Reminder {reminder_id} not found"}

    meta = dict(node.metadata or {})
    if meta.get("event_type") != "reminder":
        return {"success": False, "error": f"{reminder_id} is not a reminder"}

    meta["reminder_status"] = "dismissed"
    meta["dismissed_at"] = datetime.now(timezone.utc).isoformat()
    db.update_node(reminder_id, metadata=meta)
    clean_text = re.sub(r"\n\[due: [^\]]+\]$", "", node.content)
    return {"success": True, "dismissed_id": reminder_id, "text": clean_text}


def get_due_reminders(
    mark_fired: bool = False,
    entity_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Get all pending reminders that are due now.

    If mark_fired=True, transitions their status from 'pending' to 'fired'.
    If entity_id is provided, only returns reminders scoped to that entity.
    """
    all_reminders = list_reminders(status="pending", entity_id=entity_id)
    due = [r for r in all_reminders if r["is_due"]]

    if mark_fired and due:
        db = _get_store()
        now_iso = datetime.now(timezone.utc).isoformat()
        for r in due:
            node = db.get_node(r["id"])
            if node:
                meta = dict(node.metadata or {})
                meta["reminder_status"] = "fired"
                meta["fired_at"] = now_iso
                db.update_node(r["id"], metadata=meta)
                r["status"] = "fired"

    return due


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "auto_capture",
    "query",
    "query_structured",
    "check_health",
    "get_dedup_stats",
    "export_memories",
    "import_memories",
    "welcome",
    "get_profile",
    "save_profile",
    "remember",
    "store",
    "delete_memory",
    "edit_memory",
    "extract_preferences",
    "list_preferences",
    "deduplicate",
    "reingest",
    "status",
    "get_cross_session_lessons",
    "get_cross_project_lessons",
    "distill_trajectory",
    "reset_memory",
    "record_feedback",
    "clear_session",
    "batch_store",
    "find_similar_memories",
    "timeline",
    "consolidate",
    "traverse",
    "compact",
    "phrase_search",
    "type_stats",
    "session_stats",
    "check_constraints",
    "list_constraints",
    "save_constraints",
    "get_activity_summary",
    "get_weekly_digest",
    "parse_duration",
    "create_reminder",
    "list_reminders",
    "dismiss_reminder",
    "get_due_reminders",
]


def _install_bridge_submodule_aliases() -> None:
    """Expose 1.5.1-style bridge submodules for Pro compatibility.

    Core 1.5.0 shipped `omega.bridge` as a flat module. Some Pro 1.5.x builds
    import helpers from `omega.bridge._core`, `omega.bridge._ingest`, and
    `omega.bridge._query`. Until Core fully moves to a package layout, alias
    those submodules to this module so the published flat layout remains
    import-compatible.
    """
    import sys

    module = sys.modules[__name__]
    if not hasattr(module, "__path__"):
        module.__path__ = []  # type: ignore[attr-defined]
    for name in ("_core", "_ingest", "_query"):
        fullname = f"{__name__}.{name}"
        sys.modules.setdefault(fullname, module)
        setattr(module, name, module)


_install_bridge_submodule_aliases()
