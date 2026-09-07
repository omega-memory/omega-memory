#!/usr/bin/env python3.11
"""Release preflight for OMEGA Core (omega-memory).

Every gate here exists because something actually went wrong, not because it
seemed prudent. Run it before publishing:

    python3.11 scripts/preflight.py 1.5.16          # full, builds artifacts
    python3.11 scripts/preflight.py 1.5.16 --fast   # skips the clean-venv gates

Exit status is 0 only when every gate passes. `release.py` invokes this, so a
failing gate blocks a publish rather than merely printing a warning.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import venv
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYPROJECT = REPO / "pyproject.toml"
INIT_PY = REPO / "src" / "omega" / "__init__.py"
CHANGELOG = REPO / "CHANGELOG.md"
VERIFIER = REPO / "scripts" / "verify_core_release_artifact.py"

# Core must stay below 1.6: the Pro wheel pins `omega-memory>=1.5.13,<1.6`, so a
# Core 1.6.0 breaks dependency resolution for every Pro install. Separately,
# the 1.6 and 1.7 lines are reserved as internal milestone labels and must not
# be used as public Core versions.
CORE_VERSION_CEILING = (1, 6)

_results: list[tuple[str, bool, str]] = []


def reset_results() -> None:
    """Clear collected gate results so a caller can run groups independently."""
    _results.clear()


def failures() -> list[str]:
    """Names of gates that failed since the last reset."""
    return [name for name, ok, _ in _results if not ok]


def gate(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" -- {detail}" if detail else ""))
    return ok


def run(*args: str, cwd: Path | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=cwd or REPO, env=env, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# 1. Version and identity
# ---------------------------------------------------------------------------
def check_version(version: str) -> None:
    print("\n[1] Version and identity")

    pyproject_version = re.search(r'^version = "([^"]+)"', PYPROJECT.read_text(), re.M)
    init_version = re.search(r'^__version__ = "([^"]+)"', INIT_PY.read_text(), re.M)
    declared = {pyproject_version.group(1) if pyproject_version else None,
                init_version.group(1) if init_version else None}
    gate("pyproject and __init__ agree on a version", len(declared) == 1, str(sorted(map(str, declared))))

    parts = tuple(int(p) for p in re.findall(r"^\d+|(?<=\.)\d+", version)[:2] or [0, 0])
    gate(f"version stays below {'.'.join(map(str, CORE_VERSION_CEILING))} (Pro pins <1.6; 1.6/1.7 are internal labels)",
         parts < CORE_VERSION_CEILING, version)

    gate("CHANGELOG documents this version", f"## [{version}]" in CHANGELOG.read_text(), f"## [{version}]")

    local = run("git", "tag", "-l", f"v{version}").stdout.strip()
    remote = run("git", "ls-remote", "--tags", "origin", f"v{version}").stdout.strip()
    gate("tag is unused locally and on origin", not local and not remote, local or remote or "free")


# ---------------------------------------------------------------------------
# 2. Git hygiene
# ---------------------------------------------------------------------------
def check_git() -> None:
    print("\n[2] Git hygiene")

    branch = run("git", "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    gate("on main", branch == "main", branch)

    dirty = run("git", "status", "--porcelain").stdout.strip()
    tracked_dirty = [ln for ln in dirty.splitlines() if not ln.startswith("??")]
    gate("no uncommitted tracked changes", not tracked_dirty, "; ".join(tracked_dirty[:3]) or "clean")

    run("git", "fetch", "--quiet", "origin", "main")
    behind = run("git", "rev-list", "--count", "HEAD..origin/main").stdout.strip()
    gate("not behind origin/main", behind in ("", "0"), f"{behind} behind")

    ahead = run("git", "rev-list", "--count", "origin/main..HEAD").stdout.strip()
    print(f"  note  {ahead} commit(s) to be published")

    # Releases are cut from worktrees; a second checkout of main means someone
    # else may be committing to the ref this build is about to tag.
    worktrees = run("git", "worktree", "list").stdout.strip().splitlines()
    on_main = [w for w in worktrees if w.rstrip().endswith("[main]")]
    gate("main is checked out in exactly one worktree", len(on_main) <= 1, f"{len(on_main)} checkouts")


# ---------------------------------------------------------------------------
# 3. Artifacts: privacy, secrets, Core/Pro boundary
# ---------------------------------------------------------------------------
def check_artifacts(version: str) -> Path | None:
    print("\n[3] Artifact privacy and Core/Pro boundary")

    out = Path(tempfile.mkdtemp(prefix="omega-preflight-"))
    build = run(sys.executable, "-m", "build", "--wheel", "--sdist", "--outdir", str(out))
    if build.returncode != 0:
        gate("artifacts build", False, build.stderr.strip().splitlines()[-1] if build.stderr else "build failed")
        return None
    gate("artifacts build", True, out.name)

    wheels = list(out.glob("*.whl"))
    sdists = list(out.glob("*.tar.gz"))
    if not (len(wheels) == 1 and len(sdists) == 1):
        gate("exactly one wheel and one sdist", False, f"{len(wheels)} wheels, {len(sdists)} sdists")
        return None
    gate("exactly one wheel and one sdist", True)

    # The scanner covers personal home paths, secret values, private keys,
    # internal product markers and the omega_platform namespace, across BOTH
    # artifacts. Scanning the wheel alone is how a home path shipped in the
    # sdist from 1.5.11 to 1.5.15.
    spec = importlib.util.spec_from_file_location("_verifier", VERIFIER)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    for artifact in (wheels[0], sdists[0]):
        violations = verifier.verify_core_artifact(artifact)
        gate(f"{artifact.name} passes the privacy and boundary scan",
             not violations, "; ".join(violations[:2]) if violations else "clean")

    with tarfile.open(sdists[0], "r:gz") as archive:
        vcs = [n for n in archive.getnames() if Path(n).name in {".git", ".gitignore.bak"} or "/.git/" in n]
    gate("sdist carries no VCS metadata", not vcs, "; ".join(vcs[:2]) or "none")

    return wheels[0]


# ---------------------------------------------------------------------------
# 4. Paywall and free-tier cap, checked against the built wheel
# ---------------------------------------------------------------------------
_BOUNDARY_PROBE = r'''
import json, os, sys
result = {}
from omega.sqlite_store import _base
result["core_hard_limit"] = _base._CORE_HARD_LIMIT
os.environ["OMEGA_MAX_NODES"] = "99999"
result["cap_with_env_override"] = _base._get_effective_max_nodes()
from omega import plugins
result["capabilities"] = sorted(plugins.get_capabilities())
result["pro_tools"] = plugins.has_capability("pro_tools")
result["unlimited_memory"] = plugins.has_capability("unlimited_memory")
try:
    import omega_platform  # noqa: F401
    result["omega_platform_importable"] = True
except ImportError:
    result["omega_platform_importable"] = False
print(json.dumps(result))
'''


def check_paywall(wheel: Path) -> None:
    print("\n[4] Free-tier cap and Pro paywall (clean venv, wheel only)")

    env_dir = Path(tempfile.mkdtemp(prefix="omega-preflight-venv-"))
    try:
        venv.create(env_dir, with_pip=True)
        python = env_dir / "bin" / "python3"
        if not python.exists():
            python = env_dir / "Scripts" / "python.exe"
        install = run(str(python), "-m", "pip", "install", "--quiet", str(wheel))
        if install.returncode != 0:
            gate("wheel installs standalone", False, "pip install failed")
            return
        gate("wheel installs standalone", True)

        probe = subprocess.run([str(python), "-c", _BOUNDARY_PROBE],
                               capture_output=True, text=True, cwd=str(env_dir))
        if probe.returncode != 0:
            gate("Core imports without omega_platform", False,
                 probe.stderr.strip().splitlines()[-1] if probe.stderr else "probe failed")
            return
        import json as _json
        data = _json.loads(probe.stdout)

        gate("Core imports without omega_platform", not data["omega_platform_importable"])
        gate("free-tier hard cap is 5000", data["core_hard_limit"] == 5000, str(data["core_hard_limit"]))
        gate("OMEGA_MAX_NODES cannot lift the cap without a capability",
             data["cap_with_env_override"] == 5000, str(data["cap_with_env_override"]))
        gate("Pro capabilities are absent in a Core-only install",
             not data["pro_tools"] and not data["unlimited_memory"],
             f"capabilities={data['capabilities']}")
    finally:
        shutil.rmtree(env_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 5. Lint and tests
# ---------------------------------------------------------------------------
def check_quality() -> None:
    print("\n[5] Lint and tests")

    lint = run("ruff", "check", "src/")
    gate("ruff clean on src/", lint.returncode == 0, lint.stdout.strip().splitlines()[-1] if lint.returncode else "")

    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    tests = run(sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no", env=env)
    summary = [ln for ln in tests.stdout.splitlines() if "passed" in ln or "failed" in ln]
    gate("test suite green", tests.returncode == 0, summary[-1] if summary else "no summary")


def main() -> int:
    parser = argparse.ArgumentParser(description="Release preflight for OMEGA Core")
    parser.add_argument("version", help="version about to be released, e.g. 1.5.16")
    parser.add_argument("--fast", action="store_true", help="skip the clean-venv paywall gates")
    parser.add_argument("--skip-tests", action="store_true", help="skip lint and the test suite")
    args = parser.parse_args()

    print(f"=== OMEGA Core preflight for {args.version} ===")
    check_version(args.version)
    check_git()
    wheel = check_artifacts(args.version)
    if wheel and not args.fast:
        check_paywall(wheel)
    if not args.skip_tests:
        check_quality()

    failed = [name for name, ok, _ in _results if not ok]
    print(f"\n=== {len(_results) - len(failed)}/{len(_results)} gates passed ===")
    for name in failed:
        print(f"  BLOCKED BY: {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
