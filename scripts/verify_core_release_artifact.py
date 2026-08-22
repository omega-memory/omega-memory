#!/usr/bin/env python3.11
"""Fail closed when a public Core wheel contains private or personal material.

The verifier examines ZIP members and bounded text payloads only.  It never
installs, imports, or otherwise executes the wheel under inspection.
"""

from __future__ import annotations

import argparse
import configparser
import re
import stat
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath


EXPECTED_NAME = "omega-memory"
EXPECTED_VERSION = "1.5.13"
_EXPECTED_FILENAME = "omega_memory-1.5.13-py3-none-any.whl"
_EXPECTED_DIST_INFO = "omega_memory-1.5.13.dist-info"
_EXPECTED_CLASSIFIERS = {
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Software Development :: Libraries",
}
_MAX_MEMBER_BYTES = 10 * 1024 * 1024
_SECRET_VALUE = re.compile(
    r"(?i)\b(?:api[_-]?key|token|secret|password|private[_-]?key|credential)\b"
    r"\s*[:=]\s*(?:['\"][^'\"]{8,}['\"]|"
    r"(?=[A-Za-z0-9_-]{16,}(?![A-Za-z0-9_-]))(?=[A-Za-z0-9_-]*[0-9])"
    r"[A-Za-z0-9_-]+(?!\s*\())"
)
_PRIVATE_KEY = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
_AWS_KEY = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
_PERSONAL_PATH = re.compile(
    r"(?i)(?:/users/[^/\s]+/|/home/[^/\s]+/|/root/\.omega(?:/|\b)|"
    r"[A-Z]:\\+Users\\+[^\\\s]+\\+)"
)
_CUSTOMER_VALUE = re.compile(
    r"(?i)\b(?:customer|client)[_-](?:name|email|id)\b\s*[:=]\s*['\"][^'\"]+['\"]"
)
_CUSTOMER_EMAIL_UNQUOTED = re.compile(
    r"(?i)\b(?:customer|client)[_-]email\b\s*[:=]\s*[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}"
)
_CUSTOMER_NAME_UNQUOTED = re.compile(
    r"\b(?:customer|client)[_-]name\b\s*[:=]\s*[A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+)+"
)
_INTERNAL_PRODUCT_MARKER = re.compile(r"(?i)\bsynaptic\b")
_SECRET_NAME = re.compile(r"(?i)(?:^|[_\-.])(api[_-]?key|token|secret|password|private[_-]?key|credential)(?:[_\-.]|$)")


def _member_violation(info: zipfile.ZipInfo) -> str | None:
    name = info.filename
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    parts = [part.lower() for part in path.parts]
    if path.is_absolute() or ".." in parts:
        return "unsafe archive path"
    if not parts:
        return "empty archive path"
    if parts[0] == "omega":
        pass
    elif parts[0] == _EXPECTED_DIST_INFO:
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
    unix_mode = info.external_attr >> 16
    if unix_mode and stat.S_ISLNK(unix_mode):
        return "symbolic link"
    if info.flag_bits & 0x1:
        return "encrypted archive member"
    return None


def _content_violation(payload: bytes) -> str | None:
    """Inspect all bounded textual package data, failing closed on binary data."""
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return "unrecognized non-text payload"
    if _PERSONAL_PATH.search(text):
        return "personal absolute path in payload"
    if _INTERNAL_PRODUCT_MARKER.search(text):
        return "internal-only product marker in payload"
    if (
        _CUSTOMER_VALUE.search(text)
        or _CUSTOMER_EMAIL_UNQUOTED.search(text)
        or _CUSTOMER_NAME_UNQUOTED.search(text)
    ):
        return "customer-like value in payload"
    if _SECRET_VALUE.search(text) or _PRIVATE_KEY.search(text) or _AWS_KEY.search(text):
        return "secret-like value in payload"
    return None


def _metadata_violations(archive: zipfile.ZipFile, names: list[str]) -> list[str]:
    violations: list[str] = []
    metadata_name = f"{_EXPECTED_DIST_INFO}/METADATA"
    if names.count(metadata_name) != 1:
        return ["wheel must contain exactly one expected Core METADATA file"]
    metadata = BytesParser().parsebytes(archive.read(metadata_name))
    if metadata.get("Name") != EXPECTED_NAME:
        violations.append(f"metadata Name must be {EXPECTED_NAME}")
    if metadata.get("Version") != EXPECTED_VERSION:
        violations.append(f"metadata Version must be {EXPECTED_VERSION}")
    if metadata.get("License-Expression") != "Apache-2.0":
        violations.append("metadata License-Expression must be Apache-2.0")
    if metadata.get("Requires-Python") != ">=3.11":
        violations.append("metadata Requires-Python must be >=3.11")
    classifiers = metadata.get_all("Classifier", [])
    if len(classifiers) != len(_EXPECTED_CLASSIFIERS) or set(classifiers) != _EXPECTED_CLASSIFIERS:
        violations.append("metadata classifier set does not match the exact public Core allowlist")
    for raw_requirement in metadata.get_all("Requires-Dist", []):
        match = re.match(r"\s*([A-Za-z0-9_.-]+)", raw_requirement)
        package = re.sub(r"[-_.]+", "-", match.group(1)).lower() if match else ""
        if package in {"omega-memory-pro", "omega-platform"}:
            violations.append(f"metadata contains forbidden private dependency {package}")
    entry_name = f"{_EXPECTED_DIST_INFO}/entry_points.txt"
    if names.count(entry_name) != 1:
        violations.append("wheel must contain exactly one expected entry_points.txt")
    else:
        entry_text = archive.read(entry_name).decode("utf-8", errors="replace")
        parser = configparser.ConfigParser(interpolation=None, strict=True)
        parser.optionxform = str
        try:
            parser.read_string(entry_text)
            exact_entry_points = parser.sections() == ["console_scripts"] and dict(
                parser.items("console_scripts")
            ) == {"omega": "omega.cli:main"}
        except configparser.Error:
            exact_entry_points = False
        if not exact_entry_points:
            violations.append("wheel entry-point groups must contain only the expected Core console script")
    return violations


def verify_core_wheel(wheel: str | Path) -> list[str]:
    """Return privacy-boundary violations after inspecting *wheel* without execution."""
    path = Path(wheel)
    violations: list[str] = []
    if path.name != _EXPECTED_FILENAME or not path.is_file():
        return [f"expected the {_EXPECTED_FILENAME} release candidate"]
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                violations.append("wheel contains duplicate member names")
            for info in archive.infolist():
                name = info.filename
                violation = _member_violation(info)
                if violation:
                    violations.append(f"member {name!r}: {violation}")
                    continue
                if info.is_dir():
                    continue
                if info.file_size > _MAX_MEMBER_BYTES:
                    violations.append(f"member {name!r}: exceeds {_MAX_MEMBER_BYTES} byte inspection limit")
                    continue
                payload_violation = _content_violation(archive.read(info))
                if payload_violation:
                    violations.append(f"member {name!r}: {payload_violation}")
            violations.extend(_metadata_violations(archive, names))
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
