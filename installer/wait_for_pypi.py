"""Wait until pip can download a given omega-memory release from PyPI.

Usage: python wait_for_pypi.py VERSION [--timeout SECONDS] [--interval SECONDS]

The installer workflows start when the release tag is pushed, seconds after
scripts/release.py uploads to PyPI and before PyPI's index lists the new
version, so the build's pip fails with "No matching distribution found"
(v1.5.15, v1.5.17 and v1.5.19 on macOS). This asks pip itself, the resolver
the builds use, until it can fetch the release's wheel or the timeout passes.

Every attempt bypasses pip's cache: pip keeps index pages for up to ten
minutes, so a cached attempt would keep reporting the version as missing.
The target Python is fixed at 3.12 (what both installers bundle), so the
answer does not depend on the runner's own Python.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time

PACKAGE = "omega-memory"
TARGET_PYTHON = "3.12"


def pip_can_fetch(version: str) -> tuple[bool, str]:
    """Ask pip for the release's wheel. Returns (found, pip's last error line)."""
    with tempfile.TemporaryDirectory() as dest:
        result = subprocess.run(
            [
                sys.executable, "-m", "pip", "download",
                "--no-deps", "--no-cache-dir", "--disable-pip-version-check", "--quiet",
                "--only-binary=:all:", "--python-version", TARGET_PYTHON,
                "--dest", dest,
                f"{PACKAGE}=={version}",
            ],
            capture_output=True,
            text=True,
        )
    lines = result.stderr.strip().splitlines()
    return result.returncode == 0, lines[-1] if lines else f"pip exited {result.returncode}"


def main() -> int:
    """Poll until the release is installable; exit 1 if it never appears."""
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("version", help="omega-memory version, e.g. 1.5.20")
    parser.add_argument("--timeout", type=float, default=900, help="give up after this many seconds (default 900)")
    parser.add_argument("--interval", type=float, default=30, help="seconds between attempts (default 30)")
    args = parser.parse_args()

    deadline = time.monotonic() + args.timeout
    attempt = 0
    while True:
        attempt += 1
        found, detail = pip_can_fetch(args.version)
        if found:
            print(f"{PACKAGE} {args.version} is available from PyPI (attempt {attempt}).")
            return 0
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(
                f"ERROR: pip still cannot find {PACKAGE}=={args.version} after {args.timeout:.0f}s "
                f"({attempt} attempts). Last error: {detail}",
                file=sys.stderr,
            )
            return 1
        wait = min(args.interval, remaining)
        print(f"Attempt {attempt}: {PACKAGE} {args.version} is not on PyPI's index yet ({detail}). "
              f"Retrying in {wait:.0f}s.", flush=True)
        time.sleep(wait)


if __name__ == "__main__":
    sys.exit(main())
