"""Regression test for issue #58: store() SIGSEGVs under concurrency.

The reporter showed 8 workers × 40 stores = 320 concurrent store() calls
reliably crashed _sqlite3 _pysqlite_query_execute on macOS Python 3.12/3.14.
Root cause: bridge's auto-relate daemon thread called find_similar()/
_vec_query() on the shared sqlite connection without serialization against
the calling-thread writes; sqlite-vec's vec0 C state corrupted under
concurrent access.

Test strategy: SIGSEGV is uncatchable in-process — the test worker dies and
pytest gets an opaque worker-crash report. Run the stress in a subprocess
and assert clean exit instead.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


STRESS_SCRIPT = r"""
import os
import sys
import threading
from pathlib import Path

# Force a non-default OMEGA_HOME so the test never touches a real DB.
os.environ["OMEGA_HOME"] = sys.argv[1]
os.environ.setdefault("OMEGA_SKIP_EMBEDDINGS", "1")  # hash fallback — no ONNX dependency

from omega import bridge
from omega.sqlite_store import SQLiteStore

# Reuse one store across workers — that's the real-world MCP/CLI shape and
# the exact configuration the reporter crashed on.
store = SQLiteStore()
bridge._store = store  # bridge module-level singleton

N_WORKERS = 8
PER_WORKER = 40

def _worker(worker_id: int) -> None:
    for i in range(PER_WORKER):
        bridge.auto_capture(
            content=f"stress worker {worker_id} write {i} unique-{worker_id * 1000 + i}",
            event_type="observation",
            session_id=f"stress-{worker_id}",
        )

threads = [threading.Thread(target=_worker, args=(w,)) for w in range(N_WORKERS)]
for t in threads:
    t.start()
for t in threads:
    t.join()

# Wait briefly for any straggling auto-relate daemon threads, then close.
# close() joins registered background threads — this is where stale cursors
# would surface as a use-after-close segfault if locking is wrong.
store.close()
print(f"OK: {N_WORKERS * PER_WORKER} concurrent stores completed cleanly")
"""


@pytest.mark.timeout(120)
def test_concurrent_store_no_sigsegv():
    """8 workers × 40 stores = 320 concurrent calls, subprocess-isolated.

    Regression coverage for issue #58. Lock-only is necessary (single-conn
    architecture); without the find_similar/_vec_query lock, this reliably
    SIGSEGVs on macOS Py3.12+ within seconds.
    """
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            [sys.executable, "-c", STRESS_SCRIPT, tmp],
            capture_output=True,
            text=True,
            timeout=90,
        )

    # SIGSEGV on POSIX = negative returncode -11; any nonzero is a fail.
    assert result.returncode == 0, (
        f"Concurrent store subprocess crashed (returncode={result.returncode}).\n"
        f"stdout: {result.stdout!r}\n"
        f"stderr: {result.stderr!r}"
    )
    assert "OK:" in result.stdout, f"Expected OK marker; got: {result.stdout!r}"
