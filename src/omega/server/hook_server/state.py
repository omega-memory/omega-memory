"""Lifecycle management for the hook daemon's in-memory debounce state.

The daemon lives as long as the MCP server, so every per-session dict it keeps
must be released when that session stops and pruned when entries go stale.
Modules register their dicts here once; the daemon calls :meth:`cleanup` from
the session-stop handler and :meth:`prune_stale` from a periodic tick.
"""

from __future__ import annotations

import time
from collections.abc import MutableMapping


class DebouncedState:
    """Registry of debounce dicts with session cleanup and age-based pruning."""

    def __init__(self) -> None:
        self._session_keyed: dict[str, MutableMapping] = {}
        self._session_tuple_keyed: dict[str, MutableMapping] = {}
        self._time_keyed: dict[str, MutableMapping] = {}

    def register_session_keyed(self, name: str, mapping: MutableMapping) -> None:
        """Track a dict whose keys are session ids."""
        self._session_keyed[name] = mapping

    def register_session_tuple_keyed(self, name: str, mapping: MutableMapping) -> None:
        """Track a dict whose keys are tuples starting with a session id."""
        self._session_tuple_keyed[name] = mapping

    def register_time_keyed(self, name: str, mapping: MutableMapping) -> None:
        """Track a dict whose values are ``time.monotonic()`` timestamps."""
        self._time_keyed[name] = mapping

    def cleanup(self, session_id: str) -> None:
        """Drop every entry that belongs to ``session_id``."""
        for mapping in self._session_keyed.values():
            mapping.pop(session_id, None)
        for mapping in self._session_tuple_keyed.values():
            for key in [k for k in mapping if k[0] == session_id]:
                del mapping[key]

    def prune_stale(self, max_age: float) -> int:
        """Evict timestamped entries older than ``max_age`` seconds; return the count."""
        cutoff = time.monotonic() - max_age
        evicted = 0
        for mapping in self._time_keyed.values():
            stale = [k for k, v in mapping.items() if isinstance(v, (int, float)) and v < cutoff]
            for key in stale:
                del mapping[key]
            evicted += len(stale)
        return evicted

    def stats(self) -> dict[str, int]:
        """Entry counts per registered dict, for diagnostics."""
        return {name: len(mapping) for name, mapping in sorted(self._all().items())}

    def reset(self) -> None:
        """Clear every registered dict. Used for test isolation."""
        for mapping in self._all().values():
            mapping.clear()

    def _all(self) -> dict[str, MutableMapping]:
        return {**self._session_keyed, **self._session_tuple_keyed, **self._time_keyed}
