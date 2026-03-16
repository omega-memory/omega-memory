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


# ===================================================================
# Phase 4: Multi-tenant integration tests
# ===================================================================

import json
import uuid
from configparser import ConfigParser
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine, inspect as sa_inspect
from sqlalchemy.orm import Session

from omega.db.models import (
    Base,
    Memory,
    Edge,
    SyncState,
    Document,
    DocumentChunk,
    User,
)
from omega.schemas import (
    MemoryCreate,
    MemoryRead,
    MemorySync,
    EdgeCreate,
    DocumentCreate,
    SyncStateRecord,
    UserCreate,
    UserRead,
    WorkspaceCreate,
    CertificateInfo,
    TokenCreate,
)
from omega.cloud.backend import CloudBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sa_engine():
    """Create an in-memory SQLite engine with all OMEGA tables."""
    eng = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture()
def sa_session(sa_engine):
    """Provide a SQLAlchemy session that rolls back after each test."""
    with Session(sa_engine) as sess:
        yield sess
        sess.rollback()


@pytest.fixture()
def two_users(sa_session):
    """Insert two distinct users and return their IDs."""
    user_a = User(
        id=str(uuid.uuid4()),
        email="alice@example.com",
        display_name="Alice",
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    user_b = User(
        id=str(uuid.uuid4()),
        email="bob@example.com",
        display_name="Bob",
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    sa_session.add_all([user_a, user_b])
    sa_session.flush()
    return user_a.id, user_b.id


# ===================================================================
# 1. Multi-tenant isolation tests
# ===================================================================

class TestMultiTenantIsolation:
    """Ensure two users sharing the same database cannot see each other's data."""

    def test_document_isolation(self, sa_session, two_users):
        """User A's documents are invisible to a query filtered by User B."""
        uid_a, uid_b = two_users
        now = datetime.now(timezone.utc).isoformat()

        doc_a = Document(
            id=str(uuid.uuid4()),
            local_id=1,
            source_path="/a/file.txt",
            source_type="text",
            user_id=uid_a,
            created_at=now,
        )
        doc_b = Document(
            id=str(uuid.uuid4()),
            local_id=1,
            source_path="/b/file.txt",
            source_type="text",
            user_id=uid_b,
            created_at=now,
        )
        sa_session.add_all([doc_a, doc_b])
        sa_session.flush()

        docs_a = (
            sa_session.query(Document).filter(Document.user_id == uid_a).all()
        )
        assert len(docs_a) == 1
        assert docs_a[0].source_path == "/a/file.txt"

        docs_b = (
            sa_session.query(Document).filter(Document.user_id == uid_b).all()
        )
        assert len(docs_b) == 1
        assert docs_b[0].source_path == "/b/file.txt"

    def test_edge_isolation_via_source_scope(self, sa_session, two_users):
        """Edges authored by different users are separate when scoped by source_id prefix."""
        uid_a, uid_b = two_users
        now = datetime.now(timezone.utc).isoformat()

        edge_a = Edge(
            source_id=f"{uid_a}:node1",
            target_id=f"{uid_a}:node2",
            edge_type="related",
            created_at=now,
        )
        edge_b = Edge(
            source_id=f"{uid_b}:node1",
            target_id=f"{uid_b}:node2",
            edge_type="related",
            created_at=now,
        )
        sa_session.add_all([edge_a, edge_b])
        sa_session.flush()

        edges_a = (
            sa_session.query(Edge)
            .filter(Edge.source_id.startswith(uid_a))
            .all()
        )
        assert len(edges_a) == 1
        assert edges_a[0].source_id.startswith(uid_a)

    def test_sync_state_isolation(self, sa_session, two_users):
        """SyncState rows are unique per (table_name, user_id)."""
        uid_a, uid_b = two_users
        now = datetime.now(timezone.utc).isoformat()

        ss_a = SyncState(
            id=str(uuid.uuid4()),
            table_name="memories",
            user_id=uid_a,
            last_local_id=42,
            last_sync_at=now,
        )
        ss_b = SyncState(
            id=str(uuid.uuid4()),
            table_name="memories",
            user_id=uid_b,
            last_local_id=7,
            last_sync_at=now,
        )
        sa_session.add_all([ss_a, ss_b])
        sa_session.flush()

        result = (
            sa_session.query(SyncState)
            .filter(SyncState.table_name == "memories", SyncState.user_id == uid_a)
            .one()
        )
        assert result.last_local_id == 42

        result_b = (
            sa_session.query(SyncState)
            .filter(SyncState.table_name == "memories", SyncState.user_id == uid_b)
            .one()
        )
        assert result_b.last_local_id == 7

    def test_deletion_scoped_by_user(self, sa_session, two_users):
        """Deleting documents for user A does not affect user B."""
        uid_a, uid_b = two_users
        now = datetime.now(timezone.utc).isoformat()

        for i in range(3):
            sa_session.add(
                Document(
                    id=str(uuid.uuid4()),
                    local_id=i,
                    source_path=f"/a/{i}.txt",
                    source_type="text",
                    user_id=uid_a,
                    created_at=now,
                )
            )
        sa_session.add(
            Document(
                id=str(uuid.uuid4()),
                local_id=0,
                source_path="/b/0.txt",
                source_type="text",
                user_id=uid_b,
                created_at=now,
            )
        )
        sa_session.flush()

        sa_session.query(Document).filter(Document.user_id == uid_a).delete()
        sa_session.flush()

        remaining = sa_session.query(Document).all()
        assert len(remaining) == 1
        assert remaining[0].user_id == uid_b

    def test_document_chunk_isolation(self, sa_session, two_users):
        """DocumentChunks scoped to different users stay separate."""
        uid_a, uid_b = two_users
        now = datetime.now(timezone.utc).isoformat()

        chunk_a = DocumentChunk(
            id=str(uuid.uuid4()),
            local_id=1,
            chunk_index=0,
            content="Alice's chunk",
            user_id=uid_a,
            created_at=now,
        )
        chunk_b = DocumentChunk(
            id=str(uuid.uuid4()),
            local_id=1,
            chunk_index=0,
            content="Bob's chunk",
            user_id=uid_b,
            created_at=now,
        )
        sa_session.add_all([chunk_a, chunk_b])
        sa_session.flush()

        chunks_a = (
            sa_session.query(DocumentChunk)
            .filter(DocumentChunk.user_id == uid_a)
            .all()
        )
        assert len(chunks_a) == 1
        assert chunks_a[0].content == "Alice's chunk"

    def test_cross_user_modification_blocked(self, sa_session, two_users):
        """An update scoped by user_id won't touch another user's rows."""
        uid_a, uid_b = two_users
        now = datetime.now(timezone.utc).isoformat()

        sa_session.add(
            Document(
                id=str(uuid.uuid4()),
                local_id=1,
                source_path="/b/original.txt",
                source_type="text",
                user_id=uid_b,
                created_at=now,
            )
        )
        sa_session.flush()

        updated = (
            sa_session.query(Document)
            .filter(Document.user_id == uid_a, Document.local_id == 1)
            .update({"source_path": "/hacked.txt"})
        )
        sa_session.flush()
        assert updated == 0

        doc_b = sa_session.query(Document).filter(Document.user_id == uid_b).one()
        assert doc_b.source_path == "/b/original.txt"


# ===================================================================
# 2. Sync engine tests (using SQLite in-memory)
# ===================================================================

class TestSyncEngine:
    """Test sync-related logic using real SQLite tables."""

    def test_push_creates_records_with_correct_user_id(self, sa_session, two_users):
        """Simulating a push: records inserted carry the correct user_id."""
        uid_a, _ = two_users
        now = datetime.now(timezone.utc).isoformat()

        doc = Document(
            id=str(uuid.uuid4()),
            local_id=100,
            source_path="/pushed.txt",
            source_type="text",
            user_id=uid_a,
            created_at=now,
        )
        sa_session.add(doc)
        sa_session.flush()

        fetched = sa_session.query(Document).filter(Document.local_id == 100).one()
        assert fetched.user_id == uid_a

    def test_pull_only_returns_current_users_records(self, sa_session, two_users):
        """A pull query filtered by user_id excludes other tenants."""
        uid_a, uid_b = two_users
        now = datetime.now(timezone.utc).isoformat()

        for uid, path in [(uid_a, "/a.txt"), (uid_b, "/b.txt")]:
            sa_session.add(
                Document(
                    id=str(uuid.uuid4()),
                    local_id=1,
                    source_path=path,
                    source_type="text",
                    user_id=uid,
                    created_at=now,
                )
            )
        sa_session.flush()

        pulled = (
            sa_session.query(Document).filter(Document.user_id == uid_a).all()
        )
        assert len(pulled) == 1
        assert pulled[0].source_path == "/a.txt"

    def test_conflict_resolution_composite_key(self, sa_engine, two_users):
        """SyncState uses composite unique (table_name, user_id).

        Inserting the same table_name for two users must not conflict.
        """
        uid_a, uid_b = two_users
        now = datetime.now(timezone.utc).isoformat()

        with Session(sa_engine) as sess:
            sess.add(
                SyncState(
                    id=str(uuid.uuid4()),
                    table_name="memories",
                    user_id=uid_a,
                    last_local_id=10,
                    last_sync_at=now,
                )
            )
            sess.add(
                SyncState(
                    id=str(uuid.uuid4()),
                    table_name="memories",
                    user_id=uid_b,
                    last_local_id=20,
                    last_sync_at=now,
                )
            )
            sess.commit()

            rows = sess.query(SyncState).filter(SyncState.table_name == "memories").all()
            assert len(rows) == 2
            ids = {r.user_id: r.last_local_id for r in rows}
            assert ids[uid_a] == 10
            assert ids[uid_b] == 20

    def test_device_tracking_user_scoped(self, sa_session, two_users):
        """Sync state tracks per-user sync_count independently."""
        uid_a, uid_b = two_users
        now = datetime.now(timezone.utc).isoformat()

        ss_a = SyncState(
            id=str(uuid.uuid4()),
            table_name="documents",
            user_id=uid_a,
            last_local_id=5,
            sync_count=3,
            last_sync_at=now,
        )
        ss_b = SyncState(
            id=str(uuid.uuid4()),
            table_name="documents",
            user_id=uid_b,
            last_local_id=5,
            sync_count=7,
            last_sync_at=now,
        )
        sa_session.add_all([ss_a, ss_b])
        sa_session.flush()

        count_a = (
            sa_session.query(SyncState)
            .filter(SyncState.table_name == "documents", SyncState.user_id == uid_a)
            .one()
            .sync_count
        )
        count_b = (
            sa_session.query(SyncState)
            .filter(SyncState.table_name == "documents", SyncState.user_id == uid_b)
            .one()
            .sync_count
        )
        assert count_a == 3
        assert count_b == 7

    def test_sync_state_update_increments(self, sa_session, two_users):
        """Updating sync state for one user does not disturb the other."""
        uid_a, uid_b = two_users
        now = datetime.now(timezone.utc).isoformat()

        for uid in (uid_a, uid_b):
            sa_session.add(
                SyncState(
                    id=str(uuid.uuid4()),
                    table_name="memories",
                    user_id=uid,
                    last_local_id=0,
                    sync_count=0,
                    last_sync_at=now,
                )
            )
        sa_session.flush()

        sa_session.query(SyncState).filter(
            SyncState.table_name == "memories",
            SyncState.user_id == uid_a,
        ).update({"last_local_id": 50, "sync_count": 1})
        sa_session.flush()

        b = (
            sa_session.query(SyncState)
            .filter(SyncState.table_name == "memories", SyncState.user_id == uid_b)
            .one()
        )
        assert b.last_local_id == 0
        assert b.sync_count == 0


# ===================================================================
# 3. Schema validation tests
# ===================================================================

class TestSchemaValidation:
    """Pydantic schema validation and round-trip tests."""

    def test_memory_create_valid(self):
        m = MemoryCreate(content="hello world")
        assert m.content == "hello world"
        assert m.priority == 3
        assert m.memory_type == "semantic"

    def test_memory_create_all_fields(self):
        m = MemoryCreate(
            content="test",
            event_type="decision",
            priority=1,
            session_id="sess-1",
            project="omega",
            entity_id="ent-1",
            agent_type="claude",
            memory_type="episodic",
            referenced_date="2026-01-01",
            end_date="2026-12-31",
            ttl_seconds=3600,
            metadata={"key": "value"},
            derived_from="node-0",
            source_uri="https://example.com",
        )
        assert m.event_type == "decision"
        assert m.metadata == {"key": "value"}

    def test_memory_create_missing_required_field(self):
        with pytest.raises(Exception):  # ValidationError
            MemoryCreate()  # type: ignore[call-arg]

    def test_memory_create_wrong_type(self):
        with pytest.raises(Exception):
            MemoryCreate(content="ok", priority="not-an-int")  # type: ignore[arg-type]

    def test_memory_read_round_trip(self):
        """model -> dict -> model preserves data."""
        data = {
            "id": 1,
            "node_id": "n-1",
            "content": "hello",
            "created_at": "2026-01-01T00:00:00Z",
            "access_count": 5,
            "priority": 2,
            "retrieval_count": 10,
            "memory_type": "episodic",
            "status": "active",
        }
        schema = MemoryRead(**data)
        d = schema.model_dump()
        roundtripped = MemoryRead(**d)
        assert roundtripped.id == 1
        assert roundtripped.node_id == "n-1"
        assert roundtripped.content == "hello"
        assert roundtripped.access_count == 5
        assert roundtripped.memory_type == "episodic"

    def test_memory_sync_to_cloud_dict(self):
        ms = MemorySync(
            local_id=42,
            node_id="n-42",
            content="sync me",
            created_at="2026-03-15T00:00:00Z",
        )
        d = ms.to_cloud_dict("user-abc")
        assert d["user_id"] == "user-abc"
        assert d["local_id"] == 42

    def test_edge_create_valid(self):
        e = EdgeCreate(source_id="a", target_id="b", edge_type="related")
        assert e.weight == 1.0

    def test_edge_create_missing_required(self):
        with pytest.raises(Exception):
            EdgeCreate(source_id="a")  # type: ignore[call-arg]

    def test_document_create_to_cloud_dict(self):
        dc = DocumentCreate(
            source_path="/file.pdf",
            source_type="pdf",
            title="My Doc",
        )
        d = dc.to_cloud_dict("user-xyz")
        assert d["user_id"] == "user-xyz"
        assert d["source_path"] == "/file.pdf"

    def test_sync_state_record_round_trip(self):
        rec = SyncStateRecord(
            id="ss-1",
            table_name="memories",
            user_id="u-1",
            last_local_id=100,
            sync_count=5,
        )
        d = rec.model_dump()
        restored = SyncStateRecord(**d)
        assert restored.last_local_id == 100
        assert restored.sync_count == 5

    def test_user_create_valid(self):
        u = UserCreate(email="a@b.com", password="secret")
        assert u.email == "a@b.com"

    def test_user_create_missing_password(self):
        with pytest.raises(Exception):
            UserCreate(email="a@b.com")  # type: ignore[call-arg]

    def test_user_read_from_attributes(self):
        """Verify from_attributes config works with ORM-like objects."""

        class FakeUser:
            id = "u-1"
            email = "a@b.com"
            display_name = "Alice"
            created_at = "2026-01-01"
            updated_at = "2026-01-02"
            is_active = 1

        schema = UserRead.model_validate(FakeUser())
        assert schema.id == "u-1"
        assert schema.email == "a@b.com"

    def test_workspace_create(self):
        ws = WorkspaceCreate(name="My Workspace", slug="my-ws")
        assert ws.slug == "my-ws"

    def test_certificate_info_round_trip(self):
        ci = CertificateInfo(
            id="c-1",
            agent_name="agent-a",
            fingerprint="fp-123",
            serial_number="sn-456",
            expires_at="2027-01-01",
        )
        d = ci.model_dump()
        restored = CertificateInfo(**d)
        assert restored.fingerprint == "fp-123"
        assert restored.is_revoked == 0

    def test_token_create(self):
        tc = TokenCreate(name="dev-token", scopes=["read", "write"])
        assert tc.scopes == ["read", "write"]


# ===================================================================
# 4. Backend protocol conformance
# ===================================================================

class TestBackendProtocolConformance:
    """Verify both backends implement the CloudBackend protocol."""

    def test_postgres_backend_implements_protocol(self):
        """PostgresBackend structurally satisfies CloudBackend."""
        with patch("sqlalchemy.create_engine", return_value=MagicMock()):
            from omega.cloud.pg_backend import PostgresBackend

            backend = PostgresBackend("postgresql://localhost/test")
            for method in (
                "upsert_memories", "delete_memories", "get_sync_state",
                "update_sync_state", "upsert_documents", "upsert_document_chunks",
                "upsert_embeddings", "delete_by_local_ids",
                "search_memories_by_embedding", "rpc", "health_check",
            ):
                assert hasattr(backend, method), f"Missing method: {method}"

    def test_supabase_backend_implements_protocol(self):
        """SupabaseBackend structurally satisfies CloudBackend."""
        mock_supabase = MagicMock()
        mock_supabase.create_client.return_value = MagicMock()
        mock_opts = MagicMock()
        with patch.dict("sys.modules", {
            "supabase": mock_supabase,
            "supabase.lib": MagicMock(),
            "supabase.lib.client_options": mock_opts,
        }):
            import importlib
            import omega.cloud.supabase_backend as sb_mod
            importlib.reload(sb_mod)
            with patch.object(sb_mod.SupabaseBackend, "_load_user_id", return_value=None):
                backend = sb_mod.SupabaseBackend("https://example.supabase.co", "key")
                for method in (
                    "upsert_memories", "delete_memories", "get_sync_state",
                    "update_sync_state", "upsert_documents", "upsert_document_chunks",
                    "upsert_embeddings", "delete_by_local_ids",
                    "search_memories_by_embedding", "rpc", "health_check",
                ):
                    assert hasattr(backend, method), f"Missing method: {method}"

    def test_protocol_methods_match_spec(self):
        """CloudBackend protocol defines exactly the expected method set."""
        import inspect as _inspect

        protocol_methods = {
            name
            for name, _ in _inspect.getmembers(CloudBackend, predicate=_inspect.isfunction)
            if not name.startswith("_")
        }
        expected = {
            "upsert_memories",
            "delete_memories",
            "get_sync_state",
            "update_sync_state",
            "upsert_documents",
            "upsert_document_chunks",
            "upsert_embeddings",
            "delete_by_local_ids",
            "search_memories_by_embedding",
            "rpc",
            "health_check",
        }
        assert protocol_methods == expected

    def test_factory_returns_postgres_backend(self, tmp_path):
        """Factory creates PostgresBackend when config has postgres_dsn."""
        config = {"postgres_dsn": "postgresql://localhost/test"}
        config_file = tmp_path / "secrets.json"
        config_file.write_text(json.dumps(config))

        with patch("sqlalchemy.create_engine", return_value=MagicMock()):
            from omega.cloud.factory import create_backend
            from omega.cloud.pg_backend import PostgresBackend

            backend = create_backend(config_file)
            assert isinstance(backend, PostgresBackend)

    def test_factory_returns_supabase_backend(self, tmp_path):
        """Factory creates SupabaseBackend when config has supabase keys."""
        config = {
            "supabase_url": "https://example.supabase.co",
            "supabase_key": "test-key",
        }
        config_file = tmp_path / "secrets.json"
        config_file.write_text(json.dumps(config))

        mock_supabase = MagicMock()
        mock_supabase.create_client.return_value = MagicMock()
        mock_opts = MagicMock()
        with patch.dict("sys.modules", {
            "supabase": mock_supabase,
            "supabase.lib": MagicMock(),
            "supabase.lib.client_options": mock_opts,
        }):
            import importlib
            import omega.cloud.supabase_backend as sb_mod
            importlib.reload(sb_mod)
            with patch.object(sb_mod.SupabaseBackend, "_load_user_id", return_value=None):
                import omega.cloud.factory as factory_mod
                importlib.reload(factory_mod)
                backend = factory_mod.create_backend(config_file)
                assert isinstance(backend, sb_mod.SupabaseBackend)

    def test_factory_invalid_config_raises(self, tmp_path):
        """Factory raises ValueError on unrecognised config."""
        config_file = tmp_path / "secrets.json"
        config_file.write_text(json.dumps({"unknown_key": "val"}))

        from omega.cloud.factory import create_backend

        with pytest.raises(ValueError, match="Invalid configuration"):
            create_backend(config_file)

    def test_factory_missing_file_raises(self, tmp_path):
        """Factory raises FileNotFoundError when config is missing."""
        from omega.cloud.factory import create_backend

        with pytest.raises(FileNotFoundError):
            create_backend(tmp_path / "nonexistent.json")


# ===================================================================
# 5. Migration safety tests
# ===================================================================

class TestMigrationSafety:
    """Verify Alembic configuration and baseline migration integrity."""

    @pytest.fixture()
    def alembic_root(self):
        return Path(__file__).resolve().parent.parent / "src" / "omega" / "db" / "alembic"

    def test_alembic_ini_loads(self, alembic_root):
        """alembic.ini is valid and parseable."""
        ini_path = alembic_root / "alembic.ini"
        assert ini_path.exists(), f"alembic.ini not found at {ini_path}"

        cfg = ConfigParser()
        # ConfigParser needs interpolation disabled for %(here)s style markers
        cfg = ConfigParser(interpolation=None)
        cfg.read(str(ini_path))

        assert cfg.has_section("alembic")
        assert cfg.get("alembic", "script_location") is not None

    def test_alembic_ini_has_sqlalchemy_url(self, alembic_root):
        """alembic.ini contains a default sqlalchemy.url pointing to SQLite."""
        ini_path = alembic_root / "alembic.ini"
        cfg = ConfigParser(interpolation=None)
        cfg.read(str(ini_path))

        url = cfg.get("alembic", "sqlalchemy.url")
        assert url, "sqlalchemy.url must be set"
        assert "sqlite" in url

    def test_baseline_migration_exists(self, alembic_root):
        """The baseline migration file 0001 exists."""
        versions_dir = alembic_root / "versions"
        assert versions_dir.is_dir()

        baseline = versions_dir / "0001_baseline_v14.py"
        assert baseline.exists(), "Baseline migration 0001_baseline_v14.py not found"

    def test_baseline_migration_revision_id(self, alembic_root):
        """Baseline migration has the expected revision and down_revision."""
        # Load the migration file with alembic mocked out (not installed)
        import importlib.util

        mock_alembic = MagicMock()
        with patch.dict("sys.modules", {"alembic": mock_alembic, "alembic.op": mock_alembic.op}):
            spec = importlib.util.spec_from_file_location(
                "baseline_0001",
                alembic_root / "versions" / "0001_baseline_v14.py",
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            assert mod.revision == "0001"
            assert mod.down_revision is None  # baseline has no parent

    def test_baseline_migration_has_upgrade_downgrade(self, alembic_root):
        """Baseline migration has callable upgrade and downgrade functions."""
        import importlib.util

        mock_alembic = MagicMock()
        with patch.dict("sys.modules", {"alembic": mock_alembic, "alembic.op": mock_alembic.op}):
            spec = importlib.util.spec_from_file_location(
                "baseline_0001_v2",
                alembic_root / "versions" / "0001_baseline_v14.py",
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            assert callable(mod.upgrade)
            assert callable(mod.downgrade)

    def test_env_py_references_base_metadata(self, alembic_root):
        """env.py imports Base from models and sets target_metadata."""
        env_path = alembic_root / "env.py"
        source = env_path.read_text()
        assert "from omega.db.models import Base" in source
        assert "target_metadata = Base.metadata" in source

    def test_all_model_tables_exist_after_create_all(self, sa_engine):
        """Base.metadata.create_all produces all expected tables."""
        inspector = sa_inspect(sa_engine)
        table_names = set(inspector.get_table_names())

        expected_tables = {
            "memories",
            "edges",
            "forgetting_log",
            "cloud_delete_queue",
            "entity_index",
            "memory_clusters",
            "thompson_arms",
            "maintenance_dlq",
            "users",
            "api_tokens",
            "agent_certificates",
            "user_sessions",
            "oidc_providers",
            "workspaces",
            "workspace_members",
            "sync_state",
            "documents",
            "document_chunks",
            "secure_profiles",
        }
        missing = expected_tables - table_names
        assert not missing, f"Missing tables: {missing}"
