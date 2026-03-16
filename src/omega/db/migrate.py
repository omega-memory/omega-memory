"""Programmatic Alembic migration helpers for OMEGA."""

from __future__ import annotations

import os
from pathlib import Path

from alembic import command
from alembic.config import Config


def _alembic_cfg(db_url: str) -> Config:
    """Build an Alembic Config pointing at our bundled migration scripts."""
    alembic_dir = Path(__file__).resolve().parent / "alembic"
    ini_path = alembic_dir / "alembic.ini"

    cfg = Config(str(ini_path))
    cfg.set_main_option("script_location", str(alembic_dir))
    cfg.set_main_option("sqlalchemy.url", db_url)
    return cfg


def migrate(db_url: str, target: str = "head") -> None:
    """Run Alembic migrations up to *target* (default ``head``)."""
    cfg = _alembic_cfg(db_url)
    command.upgrade(cfg, target)


def stamp_baseline(db_url: str) -> None:
    """Stamp an existing database at the baseline revision without running migrations.

    Use this when adopting Alembic on a database that was created by the
    legacy ``schema.py`` path and already has all v14 tables.
    """
    cfg = _alembic_cfg(db_url)
    command.stamp(cfg, "0001")
