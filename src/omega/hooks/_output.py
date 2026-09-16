"""Output sink shared by the hook scripts and the in-process hook daemon.

Every hook script produces its user-visible text through :func:`emit`.
When a script runs as its own process (the ``fast_hook.py`` fallback path)
``emit`` prints to stdout, which Claude Code reads as the hook's output.
When the same module runs inside the MCP server's hook daemon, the handler
wraps the call in :func:`capture` and ``emit`` collects the lines instead,
so nothing reaches the server's stdout (which carries the MCP protocol).

The buffer is thread-local because the daemon runs handlers on a thread
pool: two hooks executing concurrently must never see each other's lines.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

_local = threading.local()


def emit(text: str = "") -> None:
    """Write one line of hook output to the active sink."""
    buffer = getattr(_local, "buffer", None)
    if buffer is None:
        print(text)
    else:
        buffer.append(text)


@contextmanager
def capture() -> Iterator[list[str]]:
    """Collect every :func:`emit` call on this thread into the yielded list."""
    previous = getattr(_local, "buffer", None)
    lines: list[str] = []
    _local.buffer = lines
    try:
        yield lines
    finally:
        _local.buffer = previous


def captured_text(lines: list[str]) -> str:
    """Join captured lines the way stdout would have shown them."""
    return "\n".join(lines).strip("\n")
