import sys
import os
from unittest.mock import MagicMock
from pathlib import Path

# Add src to path so we can import 'omega'
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Mock dependencies before importing our code
mock_sqlalchemy = MagicMock()
sys.modules["sqlalchemy"] = mock_sqlalchemy
sys.modules["supabase"] = MagicMock()
sys.modules["supabase.lib.client_options"] = MagicMock()

import sqlite3
from unittest.mock import patch, MagicMock

# Now import from omega
from omega.cloud.backend import CloudBackend
from omega.cloud.supabase_backend import SupabaseBackend
from omega.cloud.pg_backend import PostgresBackend


def test_supabase_protocol_compliance():
    """Verify SupabaseBackend conforms to the CloudBackend Protocol."""
    with patch("supabase.create_client") as mock_create:
        backend = SupabaseBackend("http://example.com", "key")
        assert isinstance(backend, CloudBackend)


def test_postgres_protocol_compliance():
    """Verify PostgresBackend conforms to the CloudBackend Protocol."""
    with patch("sqlalchemy.create_engine") as mock_engine:
        backend = PostgresBackend("postgresql://user:pass@localhost/db")
        assert isinstance(backend, CloudBackend)


def test_postgres_basic_operations():
    """Verify basic operations on PostgresBackend using actual SQLAlchemy if possible, 
    but here we'll mock the engine since sqlalchemy is missing on host.
    """
    # Configure sqlalchemy.text to return the input string
    mock_sqlalchemy.text.side_effect = lambda x: x
    
    with patch("sqlalchemy.create_engine") as mock_engine_func:
        mock_engine = MagicMock()
        mock_engine_func.return_value = mock_engine
        
        backend = PostgresBackend("postgresql://user:pass@localhost/db")
        
        # Mock connection and result
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        
        mock_result = MagicMock()
        mock_conn.execute.return_value = mock_result
        mock_result.rowcount = 1
        
        # Test delete_memories
        deleted = backend.delete_memories([1], "user-123")
        assert deleted == 1
        
        # Verify execute was called with expected query pattern
        args, kwargs = mock_conn.execute.call_args
        assert "DELETE FROM memories" in str(args[0])
        assert "user_id = :uid" in str(args[0])


if __name__ == "__main__":
    try:
        test_supabase_protocol_compliance()
        test_postgres_protocol_compliance()
        test_postgres_basic_operations()
        print("Manual tests passed!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
