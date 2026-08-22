#!/usr/bin/env python3.11
"""Fail closed when a public Core wheel contains private or personal material.

The verifier examines ZIP members and bounded text payloads only.  It never
installs, imports, or otherwise executes the wheel under inspection.
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path, PurePosixPath


_MAX_MEMBER_BYTES = 10 * 1024 * 1024
_TEXT_SUFFIXES = frozenset({".py", ".pyi", ".json", ".toml", ".txt", ".md", ".cfg", ".ini", ".rst", ""})
_SECRET_VALUE = re.compile(
    r"(?i)\b(?:api[_-]?key|token|secret|password|private[_-]?key|credential)\b\s*[:=]\s*(?:['\"][^'\"]{8,}['\"]|[A-Za-z0-9_-]{16,})"
)
_PRIVATE_KEY = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
_AWS_KEY = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
_PERSONAL_PATH = re.compile(r"(?i)(?:/users/singularityjason|[A-Z]:\\\\Users\\\\singularityjason)")
_SECRET_NAME = re.compile(r"(?i)(?:^|[_\-.])(api[_-]?key|token|secret|password|private[_-]?key|credential)(?:[_\-.]|$)")


def _member_violation(name: str) -> str | None:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    parts = [part.lower() for part in path.parts]
    if path.is_absolute() or ".." in parts:
        return "unsafe archive path"
    if not parts:
        return "empty archive path"
    if parts[0] == "omega":
        pass
    elif re.fullmatch(r"omega[-_]memory[-_].+\.dist-info", parts[0]):
        pass
    else:
        return "member is outside the Core package or standard distribution metadata"
    if any(part in {".omega", "omega_platform", "synaptic", "logs", "results"} for part in parts):
        return "private, personal, log, or results path"
    filename = parts[-1]
    if filename == ".env" or filename.startswith(".env."):
        return "environment file"
    if filename.endswith((".db", ".sqlite", ".sqlite3", ".db-wal", ".db-shm", ".log")):
        return "database or log file"
    if _SECRET_NAME.search(filename):
        return "secret-like filename"
    return None


def _content_violation(payload: bytes) -> str | None:
    text = payload.decode("utf-8", errors="replace")
    if _PERSONAL_PATH.search(text):
        return "personal absolute path in payload"
    if _SECRET_VALUE.search(text) or _PRIVATE_KEY.search(text) or _AWS_KEY.search(text):
        return "secret-like value in payload"
    return None


def verify_core_wheel(wheel: str | Path) -> list[str]:
    """Return privacy-boundary violations after inspecting *wheel* without execution."""
    path = Path(wheel)
    violations: list[str] = []
    if path.suffix != ".whl" or not path.is_file():
        return ["expected an existing .whl file"]
    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                name = info.filename
                violation = _member_violation(name)
                if violation:
                    violations.append(f"member {name!r}: {violation}")
                    continue
                if info.is_dir():
                    continue
                if info.file_size > _MAX_MEMBER_BYTES:
                    violations.append(f"member {name!r}: exceeds {_MAX_MEMBER_BYTES} byte inspection limit")
                    continue
                suffix = PurePosixPath(name).suffix.lower()
                if suffix in _TEXT_SUFFIXES:
                    payload_violation = _content_violation(archive.read(info))
                    if payload_violation:
                        violations.append(f"member {name!r}: {payload_violation}")
    except (OSError, zipfile.BadZipFile) as error:
        return [f"unreadable wheel: {error}"]
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a Core wheel without installing or executing it")
    parser.add_argument("wheel", type=Path, help="path to the Core wheel")
    args = parser.parse_args(argv)
    violations = verify_core_wheel(args.wheel)
    if violations:
        print("Core artifact privacy violation:", file=sys.stderr)
        for violation in violations:
            print(f"  {violation}", file=sys.stderr)
        return 1
    print(f"OK: inspected {args.wheel.name}; Core-only members with no personal paths or secret values")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
