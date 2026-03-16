import json
import os
from pathlib import Path
from typing import Optional

from .backend import CloudBackend
from .supabase_backend import SupabaseBackend
from .pg_backend import PostgresBackend


def _omega_home() -> Path:
    return Path(os.environ.get("OMEGA_HOME", str(Path.home() / ".omega")))


def create_backend(config_path: Optional[Path] = None) -> CloudBackend:
    """Create a cloud backend instance based on configuration."""
    if config_path is None:
        config_path = _omega_home() / "secrets.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    if "postgres_dsn" in config:
        return PostgresBackend(config["postgres_dsn"])
    elif "supabase_url" in config and "supabase_key" in config:
        return SupabaseBackend(config["supabase_url"], config["supabase_key"])
    else:
        raise ValueError(
            "Invalid configuration. Must contain 'postgres_dsn' or "
            "'supabase_url' and 'supabase_key'."
        )
