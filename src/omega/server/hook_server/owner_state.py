"""Hook server owner-state sidecar for fast delegation health checks."""

from __future__ import annotations

import json
import os
from pathlib import Path


OWNER_STATE_PATH = Path.home() / ".omega" / "hook.sock.owner.json"


def write_owner_state(pid: int, transport: str, status: str) -> None:
    """Persist current hook-server owner metadata."""
    OWNER_STATE_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = {
        "pid": int(pid),
        "transport": transport,
        "status": status,
    }
    OWNER_STATE_PATH.write_text(json.dumps(payload), encoding="utf-8")
    try:
        os.chmod(OWNER_STATE_PATH, 0o600)
    except OSError:
        pass


def read_owner_state() -> dict | None:
    """Return parsed owner metadata, or None when unavailable/invalid."""
    try:
        data = json.loads(OWNER_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def clear_owner_state(pid: int | None = None) -> None:
    """Delete owner-state sidecar if it belongs to pid or pid is omitted."""
    if pid is not None:
        current = read_owner_state()
        if not current or current.get("pid") != int(pid):
            return
    try:
        OWNER_STATE_PATH.unlink()
    except FileNotFoundError:
        return
    except OSError:
        return
