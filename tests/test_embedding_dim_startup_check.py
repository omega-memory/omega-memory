"""Startup validation of the configured dimension against the existing store.

``CREATE VIRTUAL TABLE IF NOT EXISTS`` keeps whatever dimension the table was
first built with. So an upgrade that reverts a patched default (or a changed
OMEGA_EMBEDDING_DIM) leaves the process configured for one dimension and the
store built for another, and every later write fails its dimension check one at
a time. The customer who reported this lost seven hours to that trickle.

Opening the store must surface the mismatch immediately.

These run in SUBPROCESSES on purpose. ``OMEGA_EMBEDDING_DIM`` is read once at
import and copied into module-level names, so exercising a second dimension
means importing the package again. Doing that in-process by editing
``sys.modules`` corrupts the rest of the suite: other modules captured the
original module objects and patch attributes on them, so their patches start
landing on a module the code under test no longer uses. A subprocess gets a
genuinely fresh interpreter and cannot leak into anyone else.
"""

import subprocess
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"


def _run(code: str, dim: str) -> subprocess.CompletedProcess:
    """Execute ``code`` in a fresh interpreter configured for ``dim``."""
    import os

    env = dict(os.environ)
    env["OMEGA_EMBEDDING_DIM"] = dim
    env["PYTHONPATH"] = str(_SRC) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )


def _skip_if_no_vec(result: subprocess.CompletedProcess) -> None:
    if "NO_VEC" in result.stdout:
        pytest.skip("sqlite-vec unavailable; dimension enforcement is vec-specific")


_MAKE_STORE = """
import sys
from omega.sqlite_store import SQLiteStore
store = SQLiteStore(sys.argv[1] if len(sys.argv) > 1 else {db!r})
if not store._vec_available:
    print("NO_VEC")
    raise SystemExit(0)
{body}
"""


def test_store_is_created_at_the_configured_dimension(tmp_path):
    db = str(tmp_path / "omega1024.db")
    result = _run(
        _MAKE_STORE.format(
            db=db,
            body="""
sql = store._conn.execute(
    "SELECT sql FROM sqlite_master WHERE name = 'memories_vec'").fetchone()[0]
assert "float[1024]" in sql, sql
store.store("multilingual memory", embedding=[0.1] * 1024)
n = store._conn.execute("SELECT COUNT(*) FROM memories_vec").fetchone()[0]
assert n == 1, n
print("OK")
""",
        ),
        dim="1024",
    )
    _skip_if_no_vec(result)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_opening_a_store_at_the_wrong_dimension_raises(tmp_path):
    db = str(tmp_path / "omega.db")

    built = _run(
        _MAKE_STORE.format(
            db=db,
            body="""
store.store("built at 384", embedding=[0.1] * 384)
store.close()
print("OK")
""",
        ),
        dim="384",
    )
    _skip_if_no_vec(built)
    assert built.returncode == 0, built.stderr

    # Simulate the upgrade: same database, process now configured for 1024.
    reopened = _run(
        f"""
from omega.sqlite_store import SQLiteStore
try:
    SQLiteStore({db!r})
except Exception as exc:
    print("RAISED:" + str(exc))
else:
    print("NO_ERROR")
""",
        dim="1024",
    )
    assert "RAISED:" in reopened.stdout, reopened.stdout + reopened.stderr
    message = reopened.stdout
    assert "384" in message and "1024" in message
    assert "OMEGA_EMBEDDING_DIM" in message


def test_matching_dimension_opens_cleanly(tmp_path):
    db = str(tmp_path / "omega.db")
    first = _run(_MAKE_STORE.format(db=db, body='store.close()\nprint("OK")'), dim="384")
    _skip_if_no_vec(first)
    assert first.returncode == 0, first.stderr

    again = _run(
        f"""
from omega.sqlite_store import SQLiteStore
store = SQLiteStore({db!r})
store.close()
print("OK")
""",
        dim="384",
    )
    assert again.returncode == 0, again.stderr
    assert "OK" in again.stdout


def test_legacy_store_without_vec_table_is_not_blocked(tmp_path):
    """A database with no vec table predates vectors — opening must not fail."""
    db = str(tmp_path / "legacy.db")
    result = _run(
        f"""
import sqlite3
conn = sqlite3.connect({db!r})
conn.execute("CREATE TABLE placeholder (id INTEGER PRIMARY KEY)")
conn.commit()
conn.close()

from omega.sqlite_store import SQLiteStore
store = SQLiteStore({db!r})
store.close()
print("OK")
""",
        dim="1024",
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
