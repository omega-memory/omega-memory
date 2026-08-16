"""Validated embedding model configuration.

The ``OMEGA_EMBEDDING_*`` variables are read once when the process imports
this module. Restart the MCP process after changing them.

These exist so that non-default embedding models -- notably multilingual ones
such as ``bge-m3`` (1024-dim, CLS pooling) -- can be selected through
configuration instead of source patches that every upgrade silently reverts.

The dimension recorded here must match the ``vec0`` tables in an existing
store. ``omega.sqlite_store`` validates that on startup and refuses to run on
a mismatch rather than degrading to a silent no-op.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_EMBEDDING_DIM",
    "DEFAULT_EMBEDDING_POOLING",
    "VALID_POOLING",
    "EmbeddingConfig",
    "load_embedding_config",
    "get_embedding_config",
]

DEFAULT_EMBEDDING_MODEL = "bge-small-en-v1.5"
DEFAULT_EMBEDDING_DIM = 384
DEFAULT_EMBEDDING_POOLING = "mean"
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_PAD_ID = 0

VALID_POOLING = ("mean", "cls")

# Upper bound is a sanity guard, not a model limit: it catches a mistyped
# dimension before it reaches the vec0 table, where the cost of being wrong is
# a full re-embed of the store.
_MAX_EMBEDDING_DIM = 16384


@dataclass(frozen=True)
class EmbeddingConfig:
    """Resolved embedding configuration for this process."""

    model_name: str
    dim: int
    pooling: str
    model_dir: str
    pad_token: str
    pad_id: int


def _require_str(source: Mapping[str, str], key: str, default: str) -> str:
    """Return the trimmed value, treating unset and blank alike as 'use default'.

    Blank is deliberately not an error: ``export OMEGA_EMBEDDING_MODEL=`` is a
    common way to clear a variable, and it means the same thing as never
    setting it. This matches ``dedup_config.load_dedup_thresholds``.
    """
    raw = source.get(key)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


def _require_int(source: Mapping[str, str], key: str, default: int) -> int:
    raw = source.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer, got {raw!r}") from exc


def load_embedding_config(env: Optional[Mapping[str, str]] = None) -> EmbeddingConfig:
    """Return the embedding configuration, validated.

    Args:
        env: Environment mapping to read from. Defaults to ``os.environ``.

    Returns:
        The resolved :class:`EmbeddingConfig`.

    Raises:
        ValueError: If any variable is malformed or out of range. Failing here
            is deliberate -- a bad dimension that reaches the store writes
            memories with no retrievable vector.
    """
    source = os.environ if env is None else env

    model_name = _require_str(source, "OMEGA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

    dim = _require_int(source, "OMEGA_EMBEDDING_DIM", DEFAULT_EMBEDDING_DIM)
    if not 1 <= dim <= _MAX_EMBEDDING_DIM:
        raise ValueError(
            f"OMEGA_EMBEDDING_DIM must be between 1 and {_MAX_EMBEDDING_DIM}, got {dim}"
        )

    pooling = _require_str(
        source, "OMEGA_EMBEDDING_POOLING", DEFAULT_EMBEDDING_POOLING
    ).lower()
    if pooling not in VALID_POOLING:
        raise ValueError(
            f"OMEGA_EMBEDDING_POOLING must be one of {', '.join(VALID_POOLING)}, got {pooling!r}"
        )

    model_dir = _require_str(
        source,
        "OMEGA_EMBEDDING_MODEL_DIR",
        f"~/.cache/omega/models/{model_name}-onnx",
    )

    pad_token = _require_str(source, "OMEGA_EMBEDDING_PAD_TOKEN", DEFAULT_PAD_TOKEN)

    pad_id = _require_int(source, "OMEGA_EMBEDDING_PAD_ID", DEFAULT_PAD_ID)
    if pad_id < 0:
        raise ValueError(f"OMEGA_EMBEDDING_PAD_ID must be non-negative, got {pad_id}")

    return EmbeddingConfig(
        model_name=model_name,
        dim=dim,
        pooling=pooling,
        model_dir=model_dir,
        pad_token=pad_token,
        pad_id=pad_id,
    )


# Resolved once at import. Mirrors the dedup-threshold configuration contract:
# a malformed value fails the process at startup rather than at the first write.
_CONFIG = load_embedding_config()


def get_embedding_config() -> EmbeddingConfig:
    """Return the process-wide embedding configuration resolved at import."""
    return _CONFIG
