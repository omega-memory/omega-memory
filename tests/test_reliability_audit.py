"""Tests for the read-only Core reliability audit."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from omega.reliability_audit import audit_database


def _fixture_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE memories (
            node_id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            metadata TEXT,
            access_count INTEGER DEFAULT 0,
            priority INTEGER DEFAULT 3,
            status TEXT DEFAULT 'active'
        );
        CREATE TABLE edges (
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            edge_type TEXT NOT NULL
        );
        """
    )
    rows = [
        ("steady", "ordinary memory", "{}", 1, 1, "active"),
        ("hot", "frequently used", "{}", 64, 5, "active"),
        ("old", "outdated", json.dumps({"superseded": True, "superseded_by": "new"}), 2, 5, "superseded"),
        ("new", "replacement", "{}", 1, 5, "active"),
        ("mismatched", "mismatch", json.dumps({"superseded": True}), 0, 9, "active"),
        ("empty", "", "{}", 0, 3, "active"),
    ]
    connection.executemany(
        "INSERT INTO memories(node_id, content, metadata, access_count, priority, status) VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    connection.execute("INSERT INTO edges(source_id, target_id, edge_type) VALUES ('new', 'old', 'supersedes')")
    connection.execute("INSERT INTO edges(source_id, target_id, edge_type) VALUES ('missing-source', 'old', 'supersedes')")
    connection.commit()
    connection.close()


def test_audit_reports_reliability_findings_without_mutating_database(tmp_path: Path):
    database = tmp_path / "isolated.db"
    _fixture_database(database)
    before = hashlib.sha256(database.read_bytes()).hexdigest()

    report = audit_database(database)

    assert report["database"] == str(database)
    assert report["read_only"] is True
    assert report["summary"]["memory_count"] == 6
    assert [item["node_id"] for item in report["access"]["outliers"]] == ["hot"]
    assert report["priority"]["dominance"]["priority"] == 5
    assert {item["node_id"] for item in report["invalid_active_records"]} == {"empty", "mismatched"}
    assert {item["node_id"] for item in report["supersede_inconsistencies"]} == {"mismatched", "old"}
    assert {
        item["reason"] for item in report["supersede_inconsistencies"] if item["node_id"] == "old"
    } == {"supersedes_edge_source_missing:missing-source"}
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before


def test_audit_requires_an_explicit_existing_database(tmp_path: Path):
    with pytest.raises(ValueError, match="explicit existing SQLite database"):
        audit_database(tmp_path / "missing.db")


def test_cli_audit_requires_db_path(monkeypatch, capsys):
    from omega import cli

    monkeypatch.setattr(sys, "argv", ["omega", "audit-reliability", "--json"])
    with pytest.raises(SystemExit) as error:
        cli.main()

    assert error.value.code == 2
    assert "--db" in capsys.readouterr().err


def test_cli_audit_emits_json_for_explicit_database(tmp_path: Path, monkeypatch, capsys):
    database = tmp_path / "isolated.db"
    _fixture_database(database)

    from omega import cli

    monkeypatch.setattr(sys, "argv", ["omega", "audit-reliability", "--json", "--db", str(database)])
    cli.main()

    result = json.loads(capsys.readouterr().out)
    assert result["database"] == str(database)
    assert result["read_only"] is True
