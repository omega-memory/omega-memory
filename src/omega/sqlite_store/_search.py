"""Search, retrieval, and caching mixin for SQLiteStore."""

import logging
import time as _time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path


from ._types import (
    MemoryResult,
    _serialize_f32,
    _deserialize_f32,
    _trigram_fingerprint,
    _trigram_jaccard,
    _FAST_PATH_MIN_OVERLAP,
    _FTS_REBUILD_INTERVAL,
    _HOT_CACHE_SIZE,
    _PREFETCH_CACHE_MAX,
)

from . import _types as _types_mod

logger = logging.getLogger("omega.sqlite_store")


class SearchMixin:
    """Search, retrieval, and caching methods extracted from SQLiteStore."""

    def _vec_query(self, embedding: List[float], limit: int = 10) -> List[tuple]:
        """Query the sqlite-vec virtual table. Returns [(rowid, distance), ...]."""
        if not self._vec_available:
            return []
        try:
            rows = self._conn.execute(
                "SELECT rowid, distance FROM memories_vec WHERE embedding MATCH ? AND k = ?",
                (_serialize_f32(embedding), limit),
            ).fetchall()
            return rows
        except Exception as e:
            logger.debug(f"Vec query failed: {e}")
            return []

    @staticmethod
    def _sanitize_fts5_word(word: str) -> Optional[str]:
        """Sanitize a word for FTS5 MATCH query.

        Strips characters that cause FTS5 syntax errors and filters
        out words that FTS5 interprets as operators or column names.
        Returns None if the word is not usable in FTS5.
        """
        # Strip all non-alphanumeric characters (keeps ascii letters/digits/underscores)
        import re as _re
        cleaned = _re.sub(r'[^a-z0-9_]', '', word)
        if len(cleaned) < 3:
            return None
        # FTS5 reserved words and common false-positive column names.
        # These get interpreted as FTS5 operators ("syntax error near X")
        # or column qualifiers ("no such column: X").
        _FTS5_RESERVED = frozenset({
            # FTS5 operators
            'and', 'or', 'not', 'near',
            # Interpreted as column names by FTS5
            'for', 'from', 'the', 'with', 'that', 'this', 'have', 'has',
            'was', 'are', 'been', 'were', 'will', 'can', 'may',
            # SQL keywords that FTS5 might confuse
            'select', 'where', 'insert', 'delete', 'update', 'into',
            'values', 'set', 'join', 'like', 'between', 'null',
            # Shell/code tokens that appear in queries
            'grep', 'find', 'cat', 'echo', 'eval', 'exec',
            # Content column name (our FTS5 table's only column)
            'content',
        })
        if cleaned in _FTS5_RESERVED:
            return None
        return cleaned

    def _text_search(self, query_text: str, limit: int = 20, entity_id: Optional[str] = None) -> List[MemoryResult]:
        """Text-based search using FTS5 (fast) or LIKE fallback."""
        query_lower = query_text.lower()
        words = [w for w in query_lower.split() if len(w) > 2]
        if not words:
            return []

        # Sanitize words for FTS5 (strip special chars, filter reserved words)
        fts_words = []
        for w in words:
            cleaned = self._sanitize_fts5_word(w)
            if cleaned:
                fts_words.append(cleaned)

        # Try FTS5 first (O(log n) vs O(n) for LIKE)
        if getattr(self, "_fts_available", False) and fts_words:
            try:
                # FTS5 query: OR-match sanitized words, quote each for safety
                fts_terms = " OR ".join(f'"{w}"' for w in fts_words)
                # Add bigram phrases for queries with 3+ words (improves precision)
                if len(fts_words) >= 3:
                    bigrams = [f'"{fts_words[i]} {fts_words[i+1]}"' for i in range(len(fts_words) - 1)]
                    fts_terms = fts_terms + " OR " + " OR ".join(bigrams)
                if entity_id:
                    rows = self._conn.execute(
                        """SELECT m.node_id, m.content, m.metadata, m.created_at,
                                   m.access_count, m.last_accessed, m.ttl_seconds,
                                   f.rank
                            FROM memories_fts f
                            JOIN memories m ON f.rowid = m.id
                            WHERE memories_fts MATCH ?
                            AND (m.entity_id = ? OR m.entity_id IS NULL)
                            ORDER BY f.rank LIMIT ?""",
                        (fts_terms, entity_id, limit * 3),
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        """SELECT m.node_id, m.content, m.metadata, m.created_at,
                                   m.access_count, m.last_accessed, m.ttl_seconds,
                                   f.rank
                            FROM memories_fts f
                            JOIN memories m ON f.rowid = m.id
                            WHERE memories_fts MATCH ?
                            ORDER BY f.rank LIMIT ?""",
                        (fts_terms, limit * 3),
                    ).fetchall()

                if not rows:
                    return []

                results = []
                # BM25 rank values are negative (more negative = better match)
                # Filter out None ranks (FTS5 can return NULL for corrupt index entries)
                ranks = [row[7] for row in rows if row[7] is not None]
                if not ranks:
                    return []
                best_rank = min(ranks)  # Most negative = best
                worst_rank = max(ranks)  # Closest to 0 = worst
                rank_spread = worst_rank != best_rank

                for row in rows:
                    result = self._row_to_result(row[:7])
                    bm25_rank = row[7]
                    if bm25_rank is None:
                        continue
                    # Normalize BM25: best -> 1.0, worst -> 0.1
                    if rank_spread:
                        bm25_norm = 0.1 + 0.9 * (worst_rank - bm25_rank) / (worst_rank - best_rank)
                    else:
                        bm25_norm = 1.0  # Single result or all identical ranks

                    # Word-match ratio (existing logic)
                    content_lower = result.content.lower()
                    matched = sum(1 for w in words if w in content_lower)
                    word_ratio = matched / len(words)

                    # Blend: 70% BM25 (IDF-weighted) + 30% word-match
                    result.relevance = 0.7 * bm25_norm + 0.3 * word_ratio
                    results.append(result)

                results.sort(key=lambda r: r.relevance, reverse=True)
                return results[:limit]
            except Exception as e:
                logger.warning(f"FTS5 search failed: {e} — attempting auto-repair")
                _fts_now = _time.monotonic()
                if _types_mod._last_fts_rebuild is not None and (_fts_now - _types_mod._last_fts_rebuild) < _FTS_REBUILD_INTERVAL:
                    logger.warning(
                        "FTS5 rebuild skipped (last rebuild %.0fs ago) — falling back to LIKE",
                        _fts_now - _types_mod._last_fts_rebuild,
                    )
                else:
                    try:
                        self._conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
                        self._commit()
                        _types_mod._last_fts_rebuild = _fts_now
                        logger.info("FTS5 index rebuilt successfully")
                        # Retry the query once after repair
                        if entity_id:
                            rows = self._conn.execute(
                                """SELECT m.node_id, m.content, m.metadata, m.created_at,
                                           m.access_count, m.last_accessed, m.ttl_seconds,
                                           f.rank
                                    FROM memories_fts f
                                    JOIN memories m ON f.rowid = m.id
                                    WHERE memories_fts MATCH ?
                                    AND (m.entity_id = ? OR m.entity_id IS NULL)
                                    ORDER BY f.rank LIMIT ?""",
                                (fts_terms, entity_id, limit * 3),
                            ).fetchall()
                        else:
                            rows = self._conn.execute(
                                """SELECT m.node_id, m.content, m.metadata, m.created_at,
                                           m.access_count, m.last_accessed, m.ttl_seconds,
                                           f.rank
                                    FROM memories_fts f
                                    JOIN memories m ON f.rowid = m.id
                                    WHERE memories_fts MATCH ?
                                    ORDER BY f.rank LIMIT ?""",
                                (fts_terms, limit * 3),
                            ).fetchall()
                        if not rows:
                            return []
                        results = []
                        ranks = [row[7] for row in rows if row[7] is not None]
                        if not ranks:
                            return []
                        best_rank = min(ranks)
                        worst_rank = max(ranks)
                        rank_spread = worst_rank != best_rank
                        for row in rows:
                            result = self._row_to_result(row[:7])
                            bm25_rank = row[7]
                            if bm25_rank is None:
                                continue
                            if rank_spread:
                                bm25_norm = 0.1 + 0.9 * (worst_rank - bm25_rank) / (worst_rank - best_rank)
                            else:
                                bm25_norm = 1.0
                            content_lower = result.content.lower()
                            matched = sum(1 for w in words if w in content_lower)
                            word_ratio = matched / len(words)
                            result.relevance = 0.7 * bm25_norm + 0.3 * word_ratio
                            results.append(result)
                        results.sort(key=lambda r: r.relevance, reverse=True)
                        return results[:limit]
                    except Exception as rebuild_err:
                        logger.warning(f"FTS5 rebuild also failed: {rebuild_err} — falling back to LIKE")

        # Fallback: LIKE-based search (O(n))
        conditions = " OR ".join(["LOWER(content) LIKE ?" for _ in words])
        params = [f"%{w}%" for w in words]
        params.append(limit * 3)

        rows = self._conn.execute(
            f"""SELECT node_id, content, metadata, created_at,
                       access_count, last_accessed, ttl_seconds
                FROM memories WHERE ({conditions})
                ORDER BY created_at DESC LIMIT ?""",
            params,
        ).fetchall()

        results = []
        for row in rows:
            result = self._row_to_result(row)
            content_lower = result.content.lower()
            matched = sum(1 for w in words if w in content_lower)
            result.relevance = matched / len(words)
            results.append(result)

        results.sort(key=lambda r: r.relevance, reverse=True)
        return results[:limit]

    def _temporal_search(
        self,
        start_date: str,
        end_date: str,
        limit: int = 50,
        entity_id: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """Temporal retrieval channel (P4): score memories by date proximity.

        Returns [(node_id, proximity_score), ...] sorted by score descending.
        Memories within the date range score highest; nearby memories get
        distance-decayed scores.
        """
        try:
            from datetime import datetime as dt
            # Parse reference midpoint for proximity scoring
            t_start = dt.fromisoformat(start_date.replace("Z", "+00:00"))
            t_end = dt.fromisoformat(end_date.replace("Z", "+00:00"))
            t_mid = t_start + (t_end - t_start) / 2
            range_days = max((t_end - t_start).days, 1)

            # Widen the SQL window to catch near-misses (3x the range)
            wide_start = (t_start - timedelta(days=range_days)).isoformat()
            wide_end = (t_end + timedelta(days=range_days)).isoformat()

            if entity_id:
                rows = self._conn.execute(
                    """SELECT node_id, referenced_date, created_at
                       FROM memories
                       WHERE (
                           (referenced_date BETWEEN ? AND ?)
                           OR (referenced_date IS NULL AND created_at BETWEEN ? AND ?)
                       )
                       AND (entity_id = ? OR entity_id IS NULL)
                       LIMIT ?""",
                    (wide_start, wide_end, wide_start, wide_end, entity_id, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """SELECT node_id, referenced_date, created_at
                       FROM memories
                       WHERE (
                           (referenced_date BETWEEN ? AND ?)
                           OR (referenced_date IS NULL AND created_at BETWEEN ? AND ?)
                       )
                       LIMIT ?""",
                    (wide_start, wide_end, wide_start, wide_end, limit),
                ).fetchall()

            scored: List[Tuple[str, float]] = []
            for nid, ref_date, created_at in rows:
                date_str = ref_date or created_at
                if not date_str:
                    continue
                try:
                    mem_date = dt.fromisoformat(date_str.replace("Z", "+00:00"))
                    # Proximity score: 1.0 for in-range, decays with distance
                    if t_start <= mem_date <= t_end:
                        proximity = 1.0
                    else:
                        days_away = min(
                            abs((mem_date - t_start).days),
                            abs((mem_date - t_end).days),
                        )
                        proximity = 1.0 / (1.0 + days_away / max(range_days, 1))
                    scored.append((nid, proximity))
                except (ValueError, TypeError):
                    continue

            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:limit]
        except Exception as e:
            logger.debug("Temporal search failed: %s", e)
            return []

    def retrieve_by_session(
        self,
        query_text: str,
        top_k_sessions: int = 5,
        memories_per_session: int = 20,
        **kwargs,
    ) -> List[MemoryResult]:
        """Session-level retrieval aggregation (P3).

        Scores individual memories via query(), then aggregates by session.
        Returns all memories from the top-K sessions, preserving session
        boundaries. This improves multi-session synthesis by giving the
        reading model full session context instead of isolated fragments.

        Args:
            query_text: The search query.
            top_k_sessions: Number of top sessions to return.
            memories_per_session: Max memories to fetch per selected session.
            **kwargs: Passed through to query().
        """
        # Step 1: Score individual memories with a generous limit
        kwargs.setdefault("limit", 50)
        scored_results = self.query(query_text, **kwargs)
        if not scored_results:
            return []

        # Step 2: Group by session and compute session-level scores
        session_memories: Dict[str, List[MemoryResult]] = {}
        session_scores: Dict[str, float] = {}
        for r in scored_results:
            sid = r.metadata.get("session_id", "unknown")
            session_memories.setdefault(sid, []).append(r)

        for sid, members in session_memories.items():
            member_scores = [m.relevance for m in members]
            # Session score: peak relevance + breadth bonus
            session_scores[sid] = max(member_scores) + 0.1 * (
                sum(member_scores) / len(member_scores)
            )

        # Step 3: Select top-K sessions
        top_sessions = sorted(
            session_scores, key=session_scores.get, reverse=True
        )[:top_k_sessions]
        top_session_set = set(top_sessions)

        # Step 4: Fetch full session context for selected sessions
        final_results: List[MemoryResult] = []
        for sid in top_sessions:
            # Start with already-scored members
            existing = {m.id for m in session_memories.get(sid, [])}
            final_results.extend(session_memories.get(sid, []))

            # Fetch remaining session members for full context
            session_rows = self._conn.execute(
                """SELECT node_id, content, metadata, created_at,
                          access_count, last_accessed, ttl_seconds
                   FROM memories WHERE session_id = ?
                   ORDER BY created_at ASC LIMIT ?""",
                (sid, memories_per_session),
            ).fetchall()
            for row in session_rows:
                result = self._row_to_result(row)
                if result.id not in existing:
                    result.relevance = 0.1  # Low relevance for context-only members
                    final_results.append(result)

        # Sort: group by session, then by creation time within session
        def _sort_key(r):
            sid = r.metadata.get("session_id", "unknown")
            session_rank = top_sessions.index(sid) if sid in top_sessions else 999
            ca = r.created_at.isoformat() if r.created_at else ""
            return (session_rank, ca)

        final_results.sort(key=_sort_key)
        return final_results

    # ------------------------------------------------------------------
    # Index-style lookups (replacing TypeIndex, SessionIndex)
    # ------------------------------------------------------------------

    def get_by_type(
        self, event_type: str, limit: int = 100, entity_id: Optional[str] = None
    ) -> List[MemoryResult]:
        """Get memories by event type, sorted by recency."""
        if entity_id:
            rows = self._conn.execute(
                """SELECT node_id, content, metadata, created_at,
                          access_count, last_accessed, ttl_seconds
                   FROM memories WHERE event_type = ?
                   AND (entity_id = ? OR entity_id IS NULL)
                   ORDER BY created_at DESC LIMIT ?""",
                (event_type, entity_id, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT node_id, content, metadata, created_at,
                          access_count, last_accessed, ttl_seconds
                   FROM memories WHERE event_type = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (event_type, limit),
            ).fetchall()
        return [self._row_to_result(row) for row in rows]

    def get_by_session(self, session_id: str, limit: int = 100) -> List[MemoryResult]:
        """Get memories by session ID, sorted by recency."""
        rows = self._conn.execute(
            """SELECT node_id, content, metadata, created_at,
                      access_count, last_accessed, ttl_seconds
               FROM memories WHERE session_id = ?
               ORDER BY created_at DESC LIMIT ?""",
            (session_id, limit),
        ).fetchall()
        return [self._row_to_result(row) for row in rows]

    def query_by_type(
        self,
        query: str,
        event_type: str,
        limit: int = 10,
        min_similarity: float = 0.3,
        project_path: str = "",
        scope: str = "project",
    ) -> List[MemoryResult]:
        """Search within a specific event type using embeddings or text."""
        # Try vector search first
        if self._vec_available:
            try:
                from omega.embedding import generate_embedding

                query_emb = generate_embedding(query)
                if query_emb:
                    vec_results = self._vec_query(query_emb, limit=limit * 5)
                    results = []
                    for rowid, distance in vec_results:
                        similarity = 1.0 - distance
                        if similarity < min_similarity:
                            continue
                        row = self._conn.execute(
                            """SELECT node_id, content, metadata, created_at,
                                      access_count, last_accessed, ttl_seconds
                               FROM memories WHERE id = ? AND event_type = ?""",
                            (rowid, event_type),
                        ).fetchone()
                        if row:
                            result = self._row_to_result(row)
                            # Project filter
                            if project_path and scope == "project":
                                node_project = result.metadata.get("project", "")
                                if node_project and node_project != project_path:
                                    continue
                            result.relevance = similarity
                            results.append(result)
                            if len(results) >= limit:
                                break
                    return results
            except Exception as e:
                logger.debug("Type-filtered vec search failed, falling back to text: %s", e)

        # Fallback: text search within type
        query_lower = query.lower()
        words = [w for w in query_lower.split() if len(w) > 2]
        if not words:
            return self.get_by_type(event_type, limit)

        conditions = " AND ".join(["LOWER(content) LIKE ?" for _ in words[:3]])
        params = [event_type] + [f"%{w}%" for w in words[:3]]
        params.append(limit)

        rows = self._conn.execute(
            f"""SELECT node_id, content, metadata, created_at,
                       access_count, last_accessed, ttl_seconds
                FROM memories WHERE event_type = ? AND ({conditions})
                ORDER BY created_at DESC LIMIT ?""",
            params,
        ).fetchall()

        results = []
        for row in rows:
            result = self._row_to_result(row)
            content_lower = result.content.lower()
            matched = sum(1 for w in words if w in content_lower)
            result.relevance = matched / len(words)
            results.append(result)
        return results

    def get_type_stats(self) -> Dict[str, int]:
        """Get counts for all event types."""
        rows = self._conn.execute(
            "SELECT event_type, COUNT(*) FROM memories WHERE event_type IS NOT NULL GROUP BY event_type"
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    def get_session_stats(self) -> Dict[str, int]:
        """Get counts for all sessions."""
        rows = self._conn.execute(
            "SELECT session_id, COUNT(*) FROM memories WHERE session_id IS NOT NULL GROUP BY session_id"
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    def get_stats_card_data(self) -> Dict[str, Any]:
        """Get data for the shareable stats card display."""
        row = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(access_count), 0), MIN(created_at) FROM memories"
        ).fetchone()
        memory_count = row[0] if row else 0
        total_queries = row[1] if row else 0
        oldest_date = row[2] if row else None

        session_row = self._conn.execute(
            "SELECT COUNT(DISTINCT session_id) FROM memories WHERE session_id IS NOT NULL"
        ).fetchone()
        session_count = session_row[0] if session_row else 0

        edge_row = self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()
        edge_count = edge_row[0] if edge_row else 0

        db_size_mb = self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0

        return {
            "memory_count": memory_count,
            "total_queries": total_queries,
            "session_count": session_count,
            "edge_count": edge_count,
            "oldest_date": oldest_date,
            "db_size_mb": round(db_size_mb, 1),
        }

    def get_period_stats(
        self,
        cutoff: str,
        prev_cutoff: Optional[str] = None,
        content_limit: int = 200,
    ) -> Dict[str, Any]:
        """Aggregate stats for a time period (used by weekly digest).

        Args:
            cutoff: ISO timestamp — count memories created >= this time.
            prev_cutoff: ISO timestamp — count previous period for growth comparison.
            content_limit: Max rows to fetch for topic extraction.

        Returns dict with: period_count, type_breakdown, session_count,
            content_samples, prev_period_count.
        """
        with self._lock:
            # Memories in current period
            row = self._conn.execute(
                "SELECT COUNT(*) FROM memories WHERE created_at >= ?", (cutoff,)
            ).fetchone()
            period_count = row[0] if row else 0

            # Type breakdown
            rows = self._conn.execute(
                "SELECT event_type, COUNT(*) FROM memories "
                "WHERE created_at >= ? AND event_type IS NOT NULL "
                "GROUP BY event_type ORDER BY COUNT(*) DESC",
                (cutoff,),
            ).fetchall()
            type_breakdown = {r[0]: r[1] for r in rows if r[0]}

            # Session count
            row = self._conn.execute(
                "SELECT COUNT(DISTINCT session_id) FROM memories "
                "WHERE created_at >= ? AND session_id IS NOT NULL",
                (cutoff,),
            ).fetchone()
            session_count = row[0] if row else 0

            # Content samples for topic extraction
            rows = self._conn.execute(
                "SELECT content FROM memories WHERE created_at >= ? LIMIT ?",
                (cutoff, content_limit),
            ).fetchall()
            content_samples = [r[0] for r in rows]

            # Previous period count for growth comparison
            prev_period_count = 0
            if prev_cutoff:
                row = self._conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE created_at >= ? AND created_at < ?",
                    (prev_cutoff, cutoff),
                ).fetchone()
                prev_period_count = row[0] if row else 0

        return {
            "period_count": period_count,
            "type_breakdown": type_breakdown,
            "session_count": session_count,
            "content_samples": content_samples,
            "prev_period_count": prev_period_count,
        }

    def get_oldest_accessed_since(self, cutoff: str) -> Optional[int]:
        """Get age in days of the oldest memory that was accessed since cutoff.

        Returns days as int, or None if no memories were accessed in the period.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT MIN(julianday('now') - julianday(created_at)) "
                "FROM memories WHERE access_count > 0 AND last_accessed >= ?",
                (cutoff,),
            ).fetchone()
            if row and row[0] is not None:
                return int(row[0])
            return None

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def clear_session(self, session_id: str) -> int:
        """Clear all memories for a session. Returns count removed."""
        with self._lock:
            # Capture IDs BEFORE deleting memories
            rows = self._conn.execute("SELECT id, node_id FROM memories WHERE session_id = ?", (session_id,)).fetchall()

            if not rows:
                return 0

            rowids = [r[0] for r in rows]
            node_ids = [r[1] for r in rows]

            # Delete memories
            self._conn.execute("DELETE FROM memories WHERE session_id = ?", (session_id,))

            # Delete vec embeddings
            if self._vec_available:
                for rid in rowids:
                    try:
                        self._conn.execute("DELETE FROM memories_vec WHERE rowid = ?", (rid,))
                    except Exception as e:
                        logger.debug("Failed to delete vec embedding rowid=%s: %s", rid, e)

            # Clean up edges referencing deleted nodes
            if node_ids:
                placeholders = ",".join("?" * len(node_ids))
                self._conn.execute(
                    f"DELETE FROM edges WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})",
                    node_ids + node_ids,
                )

            self._commit()
            return len(rows)

    def find_similar(self, embedding: List[float], limit: int = 10) -> List[MemoryResult]:
        """Find semantically similar memories by embedding.

        Filters out expired and superseded memories automatically.
        """
        if not self._vec_available or not embedding:
            return []

        # Over-fetch to account for filtered results
        vec_results = self._vec_query(embedding, limit=limit * 2)
        results = []
        for rowid, distance in vec_results:
            row = self._conn.execute(
                """SELECT node_id, content, metadata, created_at,
                          access_count, last_accessed, ttl_seconds
                   FROM memories WHERE id = ?""",
                (rowid,),
            ).fetchone()
            if row:
                result = self._row_to_result(row)
                if result.is_expired() or result.metadata.get("superseded"):
                    continue
                result.relevance = 1.0 - distance
                results.append(result)
                if len(results) >= limit:
                    break
        return results

    def get_timeline(self, days: int = 7, limit_per_day: int = 10) -> Dict[str, List[MemoryResult]]:
        """Get memories grouped by date for the last N days."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = self._conn.execute(
            """SELECT node_id, content, metadata, created_at,
                      access_count, last_accessed, ttl_seconds
               FROM memories
               WHERE created_at >= ?
               ORDER BY created_at DESC""",
            (cutoff,),
        ).fetchall()
        timeline: Dict[str, List[MemoryResult]] = {}
        for row in rows:
            result = self._row_to_result(row)
            day = result.created_at.strftime("%Y-%m-%d")
            if day not in timeline:
                timeline[day] = []
            if len(timeline[day]) < limit_per_day:
                timeline[day].append(result)
        return timeline

    def phrase_search(
        self,
        phrase: str,
        case_sensitive: bool = False,
        event_type: Optional[str] = None,
        limit: int = 10,
        project_path: str = "",
        scope: str = "project",
        entity_id: Optional[str] = None,
    ) -> List[MemoryResult]:
        """Exact substring search across memories."""
        conditions = []
        params = []

        if case_sensitive:
            conditions.append("content LIKE ?")
            params.append(f"%{phrase}%")
        else:
            conditions.append("LOWER(content) LIKE ?")
            params.append(f"%{phrase.lower()}%")

        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)

        if project_path and scope == "project":
            conditions.append("(project IS NULL OR project = '' OR project = ?)")
            params.append(project_path)

        if entity_id:
            conditions.append("(entity_id = ? OR entity_id IS NULL)")
            params.append(entity_id)

        params.append(limit)

        rows = self._conn.execute(
            f"""SELECT node_id, content, metadata, created_at,
                       access_count, last_accessed, ttl_seconds
                FROM memories WHERE {" AND ".join(conditions)}
                ORDER BY created_at DESC LIMIT ?""",
            params,
        ).fetchall()
        return [self._row_to_result(row) for row in rows]

    def get_session_context(self, session_id: str, limit: int = 50, include_recent: bool = True) -> List[MemoryResult]:
        """Get context for a session."""
        results = {}
        for node in self.get_by_session(session_id, limit=limit):
            results[node.id] = node

        if include_recent and len(results) < limit:
            remaining = limit - len(results)
            recent = self.get_recent(limit=remaining)
            for node in recent:
                if node.id not in results:
                    results[node.id] = node

        return list(results.values())[:limit]

    def get_recent(self, limit: int = 10) -> List[MemoryResult]:
        """Get most recent memories."""
        rows = self._conn.execute(
            """SELECT node_id, content, metadata, created_at,
                      access_count, last_accessed, ttl_seconds
               FROM memories ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [self._row_to_result(row) for row in rows]

    def get_embedding(self, node_id: str) -> Optional[List[float]]:
        """Retrieve the stored embedding for a node."""
        if not self._vec_available:
            return None
        row = self._conn.execute("SELECT id FROM memories WHERE node_id = ?", (node_id,)).fetchone()
        if not row:
            return None
        rowid = row[0]
        vec_row = self._conn.execute("SELECT embedding FROM memories_vec WHERE rowid = ?", (rowid,)).fetchone()
        if not vec_row:
            return None
        return _deserialize_f32(vec_row[0])

    def get_all_memory_embeddings(
        self,
        event_types: Optional[List[str]] = None,
        limit: int = 2000,
    ) -> List[tuple]:
        """Retrieve all memory embeddings for pattern learning.

        Returns list of (node_id, content, metadata_json, session_id,
        event_type, extracted_keywords, embedding_bytes) tuples.
        """
        if not self._vec_available:
            return []

        query = """
            SELECT m.node_id, m.content, m.metadata, m.session_id,
                   m.event_type, m.extracted_keywords, v.embedding
            FROM memories m
            JOIN memories_vec v ON v.rowid = m.id
        """
        params: list = []
        if event_types:
            placeholders = ",".join("?" for _ in event_types)
            query += f" WHERE m.event_type IN ({placeholders})"
            params.extend(event_types)

        query += " ORDER BY m.created_at DESC LIMIT ?"
        params.append(limit)

        return self._conn.execute(query, params).fetchall()

    def get_retrieval_context(self) -> List[Dict[str, Any]]:
        """Return recent retrieval context entries (A/B feedback tracking data)."""
        with self._cache_lock:
            return [{"node_id": nid, **ctx} for nid, ctx in self._recent_query_context.items()]

    # ------------------------------------------------------------------
    # Engram-inspired improvements
    # ------------------------------------------------------------------

    def _fast_path_lookup(self, query_text: str, limit: int = 10) -> List[MemoryResult]:
        """Hash-based fast-path lookup (#1): O(1) trigram fingerprint match."""
        query_fp = _trigram_fingerprint(query_text)
        if not query_fp or len(query_fp) < 5:
            return []
        if not self._is_keyword_sufficient(query_text):
            return []

        matches: List[Tuple[float, MemoryResult]] = []
        for nid, mem in self._hot_memories.items():
            mem_fp = _trigram_fingerprint(mem.content or "")
            sim = _trigram_jaccard(query_fp, mem_fp)
            if sim >= _FAST_PATH_MIN_OVERLAP:
                matches.append((sim, mem))

        if len(matches) < limit:
            try:
                query_lower = query_text.lower()
                words = [w for w in query_lower.split() if len(w) > 2]
                if words:
                    conditions = " AND ".join(["LOWER(content) LIKE ?" for _ in words[:3]])
                    params = [f"%{w}%" for w in words[:3]]
                    params.append(limit * 3)
                    rows = self._conn.execute(
                        f"""SELECT node_id, content, metadata, created_at,
                                   access_count, last_accessed, ttl_seconds
                            FROM memories WHERE ({conditions})
                            ORDER BY access_count DESC LIMIT ?""",
                        params,
                    ).fetchall()
                    seen_ids = {m[1].id for m in matches}
                    for row in rows:
                        result = self._row_to_result(row)
                        if result.id in seen_ids:
                            continue
                        if result.is_expired() or result.metadata.get("superseded"):
                            continue
                        mem_fp = _trigram_fingerprint(result.content)
                        sim = _trigram_jaccard(query_fp, mem_fp)
                        if sim >= _FAST_PATH_MIN_OVERLAP:
                            matches.append((sim, result))
                            seen_ids.add(result.id)
            except Exception as e:
                logger.debug("Fast-path SQL lookup failed: %s", e)

        if not matches:
            return []
        matches.sort(key=lambda x: x[0], reverse=True)
        results = []
        for sim, mem in matches[:limit]:
            mem.relevance = round(sim, 3)
            results.append(mem)
        return results

    def _check_hot_tier(self, query_text: str, limit: int = 10) -> List[MemoryResult]:
        """Check hot memory tier (#2) for quick matches."""
        if not self._hot_memories:
            return []
        query_words = [w for w in query_text.lower().split() if len(w) > 2]
        if not query_words:
            return []
        matches: List[Tuple[float, MemoryResult]] = []
        for nid, mem in self._hot_memories.items():
            if mem.is_expired() or mem.metadata.get("superseded"):
                continue
            overlap = self._word_overlap(query_words, (mem.content or "").lower())
            if overlap >= 0.4:
                matches.append((overlap, mem))
        if not matches:
            return []
        matches.sort(key=lambda x: x[0], reverse=True)
        results = []
        for overlap, mem in matches[:limit]:
            result = MemoryResult(
                id=mem.id, content=mem.content, metadata=mem.metadata,
                created_at=mem.created_at, access_count=mem.access_count,
                last_accessed=mem.last_accessed, ttl_seconds=mem.ttl_seconds,
                relevance=round(overlap, 3),
            )
            results.append(result)
        return results

    def _refresh_hot_cache(self) -> None:
        """Refresh the hot memory cache (#2) with top memories by access_count."""
        try:
            rows = self._conn.execute(
                """SELECT node_id, content, metadata, created_at,
                          access_count, last_accessed, ttl_seconds
                   FROM memories WHERE access_count > 0
                     AND json_extract(metadata, '$.superseded') IS NULL
                   ORDER BY access_count DESC LIMIT ?""",
                (_HOT_CACHE_SIZE,),
            ).fetchall()
            new_hot: Dict[str, MemoryResult] = {}
            for row in rows:
                result = self._row_to_result(row)
                if not result.is_expired():
                    new_hot[result.id] = result
            self._hot_memories = new_hot
            self._hot_cache_ts = _time.monotonic()
        except Exception as e:
            logger.debug("Hot cache refresh failed: %s", e)

    def clear_session_cache(self, session_id: str) -> None:
        """Clear session affinity cache (#2) when session ends."""
        self._session_cache.pop(session_id, None)

    def prefetch_for_project(self, project_path: str, file_stems: Optional[List[str]] = None) -> int:
        """Prefetch memories for a project's key files (#5)."""
        if not file_stems:
            try:
                rows = self._conn.execute(
                    """SELECT content FROM memories
                       WHERE project = ? AND access_count > 0
                       ORDER BY access_count DESC LIMIT 100""",
                    (project_path,),
                ).fetchall()
                import re as _re
                file_pattern = _re.compile(r'\b[\w/.-]+\.\w{1,5}\b')
                file_counts: Dict[str, int] = {}
                for row in rows:
                    for match in file_pattern.findall(row[0]):
                        stem = Path(match).stem
                        if len(stem) > 2:
                            file_counts[stem] = file_counts.get(stem, 0) + 1
                file_stems = sorted(file_counts, key=file_counts.get, reverse=True)[:10]
            except Exception as e:
                logger.debug("Prefetch file extraction failed: %s", e)
                return 0
        if not file_stems:
            return 0
        total_prefetched = 0
        for stem in file_stems:
            try:
                rows = self._conn.execute(
                    """SELECT node_id, content, metadata, created_at,
                              access_count, last_accessed, ttl_seconds
                       FROM memories WHERE LOWER(content) LIKE ?
                       AND (project = ? OR project IS NULL)
                       ORDER BY access_count DESC LIMIT 10""",
                    (f"%{stem.lower()}%", project_path),
                ).fetchall()
                results = []
                for row in rows:
                    result = self._row_to_result(row)
                    if not result.is_expired() and not result.metadata.get("superseded"):
                        results.append(result)
                if results:
                    self._prefetch_cache[stem] = results
                    total_prefetched += len(results)
                    if len(self._prefetch_cache) > _PREFETCH_CACHE_MAX:
                        try:
                            oldest = next(iter(self._prefetch_cache))
                            del self._prefetch_cache[oldest]
                        except StopIteration:
                            pass
            except Exception as e:
                logger.debug("Prefetch for stem '%s' failed: %s", stem, e)
        return total_prefetched
