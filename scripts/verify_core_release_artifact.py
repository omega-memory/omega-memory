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
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath


EXPECTED_NAME = "omega-memory"

# The expected version is derived from the artifact filename rather than pinned
# here. A pinned constant silently rots: it sat at 1.5.13 while 1.5.14 and
# 1.5.15 shipped, so this verifier rejected every real release candidate and
# nobody noticed -- it was not wired into the release path either.
_VERSION_RE = r"(?P<version>[0-9]+(?:\.[0-9]+)*(?:[abrc.dev+-][0-9A-Za-z.+-]*)?)"
_WHEEL_RE = re.compile(rf"^omega_memory-{_VERSION_RE}-py3-none-any\.whl$")
_SDIST_RE = re.compile(rf"^omega_memory-{_VERSION_RE}\.tar\.gz$")

# Home-directory names permitted in documentation and test fixtures. Anything
# else under /Users/<name>/ or /home/<name>/ is treated as a real person's
# path. Two fixtures once embedded a maintainer's actual home directory and
# shipped it in the sdist; use one of these names instead.
_PLACEHOLDER_HOME_NAMES = frozenset(
    {"test", "me", "you", "dev", "user", "username", "someone", "example",
     "maintainer", "another-user", "private-user", "different", "runner",
     "ci", "home", "root"}
)
# The captured segment must look like a real account name. Requiring a leading
# alphanumeric keeps the scanner off documentation ellipses ("/Users/.../x"),
# f-string placeholders ("/Users/{username}") and this file's own regex source.
_HOME_PATH_RE = re.compile(
    r"(?i)(?:/users/|/home/|[A-Z]:\\+Users\\+)([A-Za-z0-9][A-Za-z0-9._-]*)(?=[/\\]|$)"
)
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
# This scanner and its test necessarily contain the marker they search for.
# Exempt those two files from the marker rule only; every other rule, including
# the home-path rule, still applies to them.
_MARKER_RULE_EXEMPT = frozenset(
    {"scripts/verify_core_release_artifact.py", "tests/test_release_artifact_boundary.py"}
)
_SECRET_NAME = re.compile(r"(?i)(?:^|[_\-.])(api[_-]?key|token|secret|password|private[_-]?key|credential)(?:[_\-.]|$)")


def _member_violation(info: zipfile.ZipInfo, dist_info: str) -> str | None:
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
    elif parts[0] == dist_info:
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


def _content_violation_text(text: str) -> str | None:
    """Content rules shared by both artifacts, excluding the home-path rule.

    The two artifacts need different home-path rules: a wheel ships only the
    package and must contain no absolute user path at all, while an sdist ships
    docs and tests whose examples legitimately use placeholder homes.
    """
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


def _content_violation(payload: bytes) -> str | None:
    """Inspect bounded textual wheel data, failing closed on binary data."""
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return "unrecognized non-text payload"
    if _PERSONAL_PATH.search(text):
        return "personal absolute path in payload"
    return _content_violation_text(text)


def _metadata_violations(archive: zipfile.ZipFile, names: list[str], version: str) -> list[str]:
    violations: list[str] = []
    dist_info = f"omega_memory-{version}.dist-info"
    metadata_name = f"{dist_info}/METADATA"
    if names.count(metadata_name) != 1:
        return ["wheel must contain exactly one expected Core METADATA file"]
    metadata = BytesParser().parsebytes(archive.read(metadata_name))
    if metadata.get("Name") != EXPECTED_NAME:
        violations.append(f"metadata Name must be {EXPECTED_NAME}")
    if metadata.get("Version") != version:
        violations.append(f"metadata Version must be {version}")
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
    entry_name = f"{dist_info}/entry_points.txt"
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


def _real_home_path_violation(text: str) -> str | None:
    """Flag a home directory that looks like a real person's, not a placeholder."""
    for match in _HOME_PATH_RE.finditer(text):
        name = match.group(1)
        if name.lower() not in _PLACEHOLDER_HOME_NAMES:
            return f"real home directory in payload: {match.group(0)!r}"
    return None


def _sdist_member_violation(name: str, is_dir: bool, is_link: bool) -> str | None:
    """Member rules for a source distribution.

    An sdist legitimately carries docs, tests and scripts, so it does not get
    the wheel's package-only allowlist -- but it must still never carry private
    namespaces, databases, logs or credential-shaped filenames.
    """
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    parts = [part.lower() for part in path.parts]
    if path.is_absolute() or ".." in parts:
        return "unsafe archive path"
    if not parts:
        return "empty archive path"
    if is_link:
        return "symbolic link"
    if any(part in {".omega", "omega_platform", "synaptic", "logs", "results"} for part in parts):
        return "private, personal, log, or results path"
    filename = parts[-1]
    if is_dir:
        return None
    if filename == ".env" or filename.startswith(".env."):
        return "environment file"
    if filename.endswith((".db", ".sqlite", ".sqlite3", ".db-wal", ".db-shm", ".log", ".whl")):
        return "database, log, or bundled-wheel file"
    if _SECRET_NAME.search(filename):
        return "secret-like filename"
    return None


def verify_core_sdist(sdist: str | Path) -> list[str]:
    """Return privacy-boundary violations for a source distribution.

    The sdist is the wider artifact -- it ships tests and docs that the wheel
    omits -- and is exactly where a maintainer's home path shipped undetected,
    because the only content scanner in this file inspected wheels alone.
    """
    path = Path(sdist)
    match = _SDIST_RE.match(path.name)
    if not match or not path.is_file():
        return [f"not a recognizable Core sdist: {path.name}"]
    version = match.group("version")
    root = f"omega_memory-{version}"
    violations: list[str] = []
    try:
        with tarfile.open(path, "r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)):
                violations.append("sdist contains duplicate member names")
            for member in members:
                name = member.name
                relative = name[len(root) + 1:] if name.startswith(root + "/") else name
                if name != root and not name.startswith(root + "/"):
                    violations.append(f"member {name!r}: outside the {root} root")
                    continue
                violation = _sdist_member_violation(
                    relative, member.isdir(), member.issym() or member.islnk()
                )
                if violation:
                    violations.append(f"member {name!r}: {violation}")
                    continue
                if not member.isfile() or member.size > _MAX_MEMBER_BYTES:
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                payload = handle.read()
                try:
                    text = payload.decode("utf-8")
                except UnicodeDecodeError:
                    continue  # binary assets are expected in an sdist
                # A real home directory is never legitimate, including under
                # tests/ -- that is exactly where one shipped. Credential rules
                # are scoped out of tests/, whose fixtures deliberately contain
                # fake keys and private-key blocks; flagging those would train
                # everyone to ignore this check.
                checks = [_real_home_path_violation(text)]
                if relative in _MARKER_RULE_EXEMPT:
                    pass
                elif not relative.startswith("tests/"):
                    checks.append(_content_violation_text(text))
                else:
                    checks.append(
                        "internal-only product marker in payload"
                        if _INTERNAL_PRODUCT_MARKER.search(text)
                        else None
                    )
                for check in checks:
                    if check:
                        violations.append(f"member {name!r}: {check}")
                        break
    except (OSError, tarfile.TarError) as error:
        return [f"unreadable sdist: {error}"]
    return violations


def verify_core_wheel(wheel: str | Path) -> list[str]:
    """Return privacy-boundary violations after inspecting *wheel* without execution."""
    path = Path(wheel)
    violations: list[str] = []
    match = _WHEEL_RE.match(path.name)
    if not match or not path.is_file():
        return [f"not a recognizable Core wheel: {path.name}"]
    version = match.group("version")
    dist_info = f"omega_memory-{version}.dist-info"
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                violations.append("wheel contains duplicate member names")
            for info in archive.infolist():
                name = info.filename
                violation = _member_violation(info, dist_info)
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
            violations.extend(_metadata_violations(archive, names, version))
    except (OSError, zipfile.BadZipFile) as error:
        return [f"unreadable wheel: {error}"]
    return violations


def verify_core_artifact(artifact: str | Path) -> list[str]:
    """Dispatch to the wheel or sdist verifier based on the filename."""
    path = Path(artifact)
    if path.name.endswith(".whl"):
        return verify_core_wheel(path)
    if path.name.endswith(".tar.gz"):
        return verify_core_sdist(path)
    return [f"unrecognized artifact type: {path.name}"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect Core release artifacts without installing or executing them"
    )
    parser.add_argument("artifacts", type=Path, nargs="+", help="paths to the Core wheel and/or sdist")
    args = parser.parse_args(argv)

    failed = False
    for artifact in args.artifacts:
        violations = verify_core_artifact(artifact)
        if violations:
            failed = True
            print(f"Core artifact privacy violation in {artifact.name}:", file=sys.stderr)
            for violation in violations:
                print(f"  {violation}", file=sys.stderr)
        else:
            print(f"OK: {artifact.name} -- Core-only members, no personal paths or secret values")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
