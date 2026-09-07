"""OMEGA integration tests — verify the full stack works end-to-end."""

import os
import sys
import subprocess
from contextlib import contextmanager
import pytest
from pathlib import Path

from omega.exceptions import StorageError

# Ensure omega package is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestPackageImport:
    """P0: The package must import without error."""

    def test_import_omega(self):
        from omega import __version__
        assert __version__

    def test_import_bridge(self):
        from omega.bridge import auto_capture, query, status
        assert callable(auto_capture)
        assert callable(query)
        assert callable(status)

    def test_import_handlers(self):
        from omega.server.handlers import HANDLERS
        assert len(HANDLERS) >= 20

    def test_import_tool_schemas(self):
        from omega.server.tool_schemas import TOOL_SCHEMAS
        assert len(TOOL_SCHEMAS) >= 10  # 12 consolidated tools

    def test_import_types(self):
        from omega.types import AutoCaptureEventType, TTLCategory
        assert AutoCaptureEventType.GIT_CONFLICT == "git_conflict"
        assert TTLCategory.PERMANENT is None


@contextmanager
def _skip_embeddings():
    """Context manager that skips embeddings and resets the circuit breaker after."""
    from omega.embedding import reset_embedding_state
    os.environ["OMEGA_SKIP_EMBEDDINGS"] = "1"
    try:
        yield
    finally:
        os.environ.pop("OMEGA_SKIP_EMBEDDINGS", None)
        reset_embedding_state()


class TestDatabaseRoundtrip:
    """Verify store → query → delete works with a temp database."""

    def test_store_and_query(self, tmp_omega_dir):
        with _skip_embeddings():
            from omega.sqlite_store import SQLiteStore
            store = SQLiteStore(db_path=tmp_omega_dir / "test.db")

            # Store
            node_id = store.store(
                content="Test memory: Python prefers spaces over tabs",
                metadata={"event_type": "lesson_learned"},
            )
            assert node_id.startswith("mem-")

            # Query
            results = store.query("Python spaces tabs", limit=5)
            assert len(results) >= 1
            assert any("spaces" in r.content for r in results)

            # Delete
            deleted = store.delete_node(node_id)
            assert deleted is True

            # Verify gone
            assert store.get_node(node_id) is None

            store.close()

    def test_content_dedup(self, tmp_omega_dir):
        """Storing identical content twice should return the same node_id."""
        with _skip_embeddings():
            from omega.sqlite_store import SQLiteStore
            store = SQLiteStore(db_path=tmp_omega_dir / "test.db")

            id1 = store.store(content="Exact duplicate test content here")
            id2 = store.store(content="Exact duplicate test content here")
            assert id1 == id2
            assert store.node_count() == 1

            store.close()

    def test_null_content_rejected(self, tmp_omega_dir):
        """Storing empty content should raise ValueError."""
        with _skip_embeddings():
            from omega.sqlite_store import SQLiteStore
            store = SQLiteStore(db_path=tmp_omega_dir / "test.db")

            with pytest.raises(StorageError):
                store.store(content="")

            store.close()


class TestBridge:
    """Verify bridge-level API works."""

    def test_auto_capture_and_query(self, tmp_omega_dir):
        with _skip_embeddings():
            from omega.bridge import auto_capture, query, reset_memory
            reset_memory()

            result = auto_capture(
                content="Integration test: always use type hints in Python",
                event_type="lesson_learned",
                session_id="test-session",
            )
            assert "Stored" in result or "Deduped" in result or "Evolved" in result

            query_result = query(query_text="type hints Python", limit=5)
            assert "type hints" in query_result

            reset_memory()


class TestHandlerValidation:
    """Verify handler input validation."""

    @pytest.mark.asyncio
    async def test_backup_export_rejects_path_traversal(self):
        from omega.server.handlers import handle_omega_backup
        result = await handle_omega_backup({"filepath": "/etc/passwd", "mode": "export"})
        assert result.get("isError")

    @pytest.mark.asyncio
    async def test_backup_import_rejects_path_traversal(self):
        from omega.server.handlers import handle_omega_backup
        result = await handle_omega_backup({"filepath": "/etc/passwd", "mode": "import"})
        assert result.get("isError")


@pytest.mark.skipif(
    subprocess.run(
        [sys.executable, "-c", "import omega"],
        capture_output=True,
    ).returncode != 0,
    reason="omega not installed as package",
)
class TestCLIDoctor:
    """Verify omega doctor runs without crashing."""

    def test_doctor_runs(self):
        result = subprocess.run(
            [sys.executable, "-m", "omega.cli", "doctor"],
            capture_output=True, text=True, timeout=45,
        )
        # Doctor may exit 0 or 1 depending on environment, but should not crash
        assert result.returncode in (0, 1)
        assert "OMEGA Doctor" in result.stdout


class TestFrameworkAdapters:
    """The CrewAI adapter must reference APIs that actually exist on the store.

    Regression: the adapter imported ``OmegaSQLiteStore``, a name the store has
    never exported, so constructing the backend raised ImportError. Its search
    path then called ``search_by_embedding`` behind a ``hasattr`` guard, which
    turned a second missing-API bug into a silent empty result set.
    """

    def test_store_exposes_the_names_the_adapter_uses(self):
        from omega.sqlite_store import SQLiteStore

        assert hasattr(SQLiteStore, "find_similar")
        assert hasattr(SQLiteStore, "delete_node")

    def test_adapter_does_not_reference_absent_store_apis(self):
        from pathlib import Path as _Path

        import omega.integrations.crewai as crewai_adapter
        from omega.sqlite_store import SQLiteStore

        source = _Path(crewai_adapter.__file__).read_text()
        # Call/import sites only -- the names may still appear in comments that
        # explain the historical bug.
        for absent, call_site in (
            ("OmegaSQLiteStore", "import OmegaSQLiteStore"),
            ("search_by_embedding", "self._db.search_by_embedding"),
        ):
            assert not hasattr(SQLiteStore, absent)
            assert call_site not in source
