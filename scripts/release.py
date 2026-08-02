#!/usr/bin/env python3.11
"""Release omega-memory to PyPI.

Alternative path to the GH Actions publish.yml workflow on omega-public.
The script pushes a git tag but does not create a GitHub release, so the
auto-publish workflow does not fire — no double-publish risk.

Use when you don't want to wait for or trust GitHub Actions runners.

Usage:
    python3.11 scripts/release.py <version>            # publish for real
    python3.11 scripts/release.py <version> --dry-run  # build + verify only
    python3.11 scripts/release.py <version> --skip-confirm  # CI-like, no prompts

Pre-flight:
    - PYPI_TOKEN_OMEGA in ~/.omega/secrets.json
    - Working tree clean on main, up to date with origin
    - Version not already tagged
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile
from pathlib import Path
from pathlib import PurePosixPath

REPO = Path(__file__).resolve().parent.parent
PYPROJECT = REPO / "pyproject.toml"
INIT_PY = REPO / "src" / "omega" / "__init__.py"
SECRETS = Path.home() / ".omega" / "secrets.json"


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=REPO, check=True, **kw)


def step(name: str) -> None:
    print(f"\n=== {name} ===")


def confirm(prompt: str, skip: bool) -> None:
    if skip:
        return
    answer = input(f"\n{prompt} [y/N] ").strip().lower()
    if answer != "y":
        sys.exit("Aborted.")


def preflight(version: str) -> None:
    step("Pre-flight")
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        sys.exit(f"Version must be X.Y.Z, got {version!r}")

    if not SECRETS.exists():
        sys.exit(f"Missing {SECRETS}")
    secrets = json.loads(SECRETS.read_text())
    if not secrets.get("PYPI_TOKEN_OMEGA"):
        sys.exit("PYPI_TOKEN_OMEGA not in ~/.omega/secrets.json")

    # Only block on uncommitted changes to files this script will modify.
    tracked_targets = ["pyproject.toml", "src/omega/__init__.py"]
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--"] + tracked_targets,
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.strip()
    if dirty:
        sys.exit(f"Uncommitted changes to release-target files:\n{dirty}")

    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.strip()
    if branch != "main":
        sys.exit(f"Not on main, on {branch!r}")

    run(["git", "fetch", "origin", "main"])
    behind = subprocess.run(
        ["git", "rev-list", "--count", "HEAD..origin/main"], cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.strip()
    if behind != "0":
        sys.exit(f"Local main is {behind} commits behind origin/main. Pull first.")

    tag = f"v{version}"
    existing = subprocess.run(
        ["git", "tag", "-l", tag], cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.strip()
    if existing:
        sys.exit(f"Tag {tag} already exists locally.")

    print(f"  OK: version={version}, branch=main, clean, no tag {tag}")


def bump_version(version: str) -> None:
    step(f"Bumping version to {version}")
    for path, pattern, replacement in [
        (PYPROJECT, r'^version = "[^"]+"', f'version = "{version}"'),
        (INIT_PY, r'^__version__ = "[^"]+"', f'__version__ = "{version}"'),
    ]:
        text = path.read_text()
        if not re.search(pattern, text, flags=re.MULTILINE):
            sys.exit(f"Pattern not found in {path}: {pattern!r}")
        new = re.sub(pattern, replacement, text, count=1, flags=re.MULTILINE)
        if new == text:
            print(f"  unchanged {path.relative_to(REPO)}: already at {version}")
        else:
            path.write_text(new)
            print(f"  updated {path.relative_to(REPO)}")


def build() -> tuple[Path, Path]:
    step("Building wheel + sdist")
    dist = REPO / "dist"
    if dist.exists():
        for f in dist.iterdir():
            f.unlink()
    run([sys.executable, "-m", "build", "--wheel", "--sdist"])
    wheels = list(dist.glob("omega_memory-*.whl"))
    sdists = list(dist.glob("omega_memory-*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        sys.exit(f"Expected 1 wheel + 1 sdist, got {wheels=} {sdists=}")
    return wheels[0], sdists[0]


def _private_archive_members(names: list[str]) -> list[str]:
    """Return archive members that would expose a private Pro distribution."""
    offenders = []
    for name in names:
        parts = [part.lower() for part in PurePosixPath(name.replace("\\", "/")).parts]
        filename = parts[-1] if parts else ""
        if "omega_platform" in parts or "omega_platform.py" in parts:
            offenders.append(name)
        elif filename.endswith(".whl"):
            # A public source archive must never carry a bundled wheel, even if
            # a future private distribution uses a different filename.
            offenders.append(name)
        elif any(filename.startswith(prefix) for prefix in ("omega_memory_pro-", "omega-memory-pro-")):
            offenders.append(name)
    return offenders


def _private_dependencies(metadata: str) -> list[str]:
    """Return private distribution requirements from wheel/sdist metadata."""
    offenders = []
    for line in metadata.splitlines():
        if not line.lower().startswith("requires-dist:"):
            continue
        requirement = line.split(":", 1)[1].strip()
        normalized = requirement.lower().replace("_", "-")
        if normalized.startswith(("omega-platform", "omega-memory-pro")):
            offenders.append(line)
    return offenders


def verify_public_artifact_boundary(wheel: Path, sdist: Path) -> None:
    """Fail closed if public artifacts contain or depend on private Pro code."""
    step("Verifying public/Core artifact boundary")
    violations = []

    if (REPO / "src" / "omega_platform").exists():
        violations.append("repository contains src/omega_platform")
    if not wheel.name.startswith("omega_memory-") or wheel.name.startswith("omega_memory_pro-"):
        violations.append(f"unexpected public wheel name: {wheel.name}")
    if not sdist.name.startswith("omega_memory-") or sdist.name.startswith("omega_memory_pro-"):
        violations.append(f"unexpected public sdist name: {sdist.name}")

    with zipfile.ZipFile(wheel) as archive:
        wheel_names = archive.namelist()
        violations.extend(f"wheel member: {name}" for name in _private_archive_members(wheel_names))
        metadata_names = [name for name in wheel_names if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            violations.append(f"wheel contains {len(metadata_names)} METADATA files")
        else:
            metadata = archive.read(metadata_names[0]).decode("utf-8", errors="replace")
            violations.extend(f"wheel dependency: {line}" for line in _private_dependencies(metadata))

    with tarfile.open(sdist, "r:gz") as archive:
        sdist_names = archive.getnames()
        violations.extend(f"sdist member: {name}" for name in _private_archive_members(sdist_names))
        metadata_members = [member for member in archive.getmembers() if member.name.endswith("/PKG-INFO")]
        if len(metadata_members) != 1:
            violations.append(f"sdist contains {len(metadata_members)} PKG-INFO files")
        else:
            metadata_file = archive.extractfile(metadata_members[0])
            if metadata_file is None:
                violations.append("sdist PKG-INFO is unreadable")
            else:
                metadata = metadata_file.read().decode("utf-8", errors="replace")
                violations.extend(f"sdist dependency: {line}" for line in _private_dependencies(metadata))

    if violations:
        detail = "\n  ".join(violations)
        sys.exit(f"Public artifact boundary violation:\n  {detail}")
    print("  OK: Core-only archives; no private namespace, bundled wheel, or Pro dependency")


def verify(wheel: Path, expected_version: str) -> None:
    step("Verifying wheel in clean venv")
    with tempfile.TemporaryDirectory(prefix="omega-mem-verify-") as tmp:
        env_dir = Path(tmp) / "venv"
        venv.create(str(env_dir), with_pip=True)
        py = env_dir / "bin" / "python3.11"
        if not py.exists():
            py = env_dir / "bin" / "python"
        run([str(py), "-m", "pip", "install", "--quiet", str(wheel)])
        proc = subprocess.run(
            [str(py), "-c", "import omega; print(omega.__version__)"],
            capture_output=True, text=True, check=True,
        )
        installed = proc.stdout.strip()
        if installed != expected_version:
            sys.exit(f"Wheel reports {installed!r}, expected {expected_version!r}")
        print(f"  OK: installed and reported version={installed}")


def publish_pypi(wheel: Path, sdist: Path) -> None:
    step("Publishing to PyPI")
    secrets = json.loads(SECRETS.read_text())
    env = {**os.environ, "TWINE_USERNAME": "__token__", "TWINE_PASSWORD": secrets["PYPI_TOKEN_OMEGA"]}
    subprocess.run(
        [sys.executable, "-m", "twine", "upload", "--non-interactive", str(wheel), str(sdist)],
        env=env, check=True,
    )
    print(f"  Published: https://pypi.org/project/omega-memory/{wheel.stem.split('-')[1]}/")


def git_commit_tag_push(version: str) -> None:
    step("Committing + tagging + pushing")
    run(["git", "add", "pyproject.toml", "src/omega/__init__.py"])
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only"], cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.strip()
    if staged:
        run(["git", "commit", "-m", f"chore: release v{version}"])
    else:
        print("  no version-file changes to commit (idempotent re-run)")
    existing = subprocess.run(
        ["git", "tag", "-l", f"v{version}"], cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.strip()
    if not existing:
        run(["git", "tag", f"v{version}"])
    run(["git", "push", "origin", "main"])
    run(["git", "push", "origin", f"v{version}"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("version", help="Version X.Y.Z")
    ap.add_argument("--dry-run", action="store_true", help="Build + verify only; do not publish or push")
    ap.add_argument("--skip-confirm", action="store_true", help="Skip interactive confirmation")
    args = ap.parse_args()

    preflight(args.version)
    bump_version(args.version)
    wheel, sdist = build()
    verify_public_artifact_boundary(wheel, sdist)
    verify(wheel, args.version)

    if args.dry_run:
        print(f"\nDRY RUN: would publish {wheel.name} + {sdist.name} and push v{args.version}")
        print("Reverting version bump...")
        run(["git", "checkout", "--", "pyproject.toml", "src/omega/__init__.py"])
        return 0

    confirm(f"Publish omega-memory {args.version} to PyPI?", args.skip_confirm)
    publish_pypi(wheel, sdist)

    confirm(f"Commit + tag v{args.version} + push to origin/main?", args.skip_confirm)
    git_commit_tag_push(args.version)

    step("Done")
    print(f"omega-memory {args.version} released.")
    print(f"  PyPI:   https://pypi.org/project/omega-memory/{args.version}/")
    print(f"  GitHub: https://github.com/omega-memory/omega-memory/releases/tag/v{args.version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
