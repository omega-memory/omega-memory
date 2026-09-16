"""
OMEGA Cross-Encoder Reranker — ONNX-based passage reranking for search quality.

Provides:
- cross_encoder_score(query, passages) → list of relevance scores (or None)
- Lazy model loading with circuit breaker (3 attempts, 5-min cooldown)
- Env-var disable: OMEGA_CROSS_ENCODER=0
- Precision selection: OMEGA_RERANKER_PRECISION=fp32|int8 (default: fp32)
  fp32: full precision, ~2.3 GB on disk, ~2.5 GB RSS
  int8: quantized, ~571 MB on disk, ~650 MB RSS
- Auto-download on first use: OMEGA_RERANKER_AUTODOWNLOAD=1 (default).
  Set to 0 to require manual download via `python -m omega --download-model`.

Uses cross-encoder/ms-marco-MiniLM-L-6-v2 via ONNX Runtime.
Mirrors the loading patterns from omega.embedding.
"""

import contextlib
import io
import logging
import os
import time as _time_module
from pathlib import Path
from typing import Any

__all__ = [
    "cross_encoder_score",
    "get_reranker_model_info",
    "preload_reranker_model",
    "reset_reranker_state",
    "download_model",
]

logger = logging.getLogger("omega.reranker")

# NumPy for vectorized operations (required by ONNX backend)
try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

# --------------------------------------------------------------------------
# Module state
# --------------------------------------------------------------------------

_RERANKER_MODEL = None  # Tuple of (tokenizer, session) when loaded

# Model selection: bge-reranker-v2-m3 is general-purpose (not web-search-specific),
# which performs better on conversational memory than MS-MARCO (P2).
# Override model with OMEGA_RERANKER_MODEL, precision with OMEGA_RERANKER_PRECISION.
_AVAILABLE_MODELS = {
    "bge-reranker-v2-m3": {
        "repo_id": "onnx-community/bge-reranker-v2-m3-ONNX",
        "default_precision": "fp32",
        "precisions": {
            "fp32": {
                "dir": "~/.cache/omega/models/bge-reranker-v2-m3-onnx",
                "files": [
                    ("onnx/model.onnx", "model.onnx"),
                    ("onnx/model.onnx_data", "model.onnx_data"),
                    ("tokenizer.json", "tokenizer.json"),
                    ("config.json", "config.json"),
                ],
            },
            "int8": {
                "dir": "~/.cache/omega/models/bge-reranker-v2-m3-onnx-int8",
                "files": [
                    ("onnx/model_quantized.onnx", "model.onnx"),
                    ("tokenizer.json", "tokenizer.json"),
                    ("config.json", "config.json"),
                ],
            },
        },
    },
    "ms-marco-MiniLM-L-6-v2": {
        "repo_id": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "dir": "~/.cache/omega/models/ms-marco-MiniLM-L-6-v2-onnx",
        "files": [
            ("onnx/model.onnx", "model.onnx"),
            ("tokenizer.json", "tokenizer.json"),
            ("config.json", "config.json"),
        ],
    },
}


def _resolve_model_config(model_name: str, precision: str | None = None) -> tuple[str, str, list[tuple[str, str]]]:
    """Resolve model config to (repo_id, dir, files) including precision selection."""
    model = _AVAILABLE_MODELS.get(model_name, _AVAILABLE_MODELS["ms-marco-MiniLM-L-6-v2"])

    if "precisions" in model:
        if precision is None:
            precision = os.environ.get("OMEGA_RERANKER_PRECISION", model["default_precision"])
        if precision not in model["precisions"]:
            logger.warning("Unknown precision '%s', falling back to %s", precision, model["default_precision"])
            precision = model["default_precision"]
        variant = model["precisions"][precision]
        return model["repo_id"], variant["dir"], variant["files"]

    return model["repo_id"], model["dir"], model["files"]


# Auto-detect best available model: prefer bge-reranker-v2-m3 (general-purpose,
# better on conversational memory) over ms-marco (web-search-specific).
# Override with OMEGA_RERANKER_MODEL env var.
def _resolve_reranker_model() -> tuple:
    env = os.environ.get("OMEGA_RERANKER_MODEL")
    if env:
        _, model_dir, _ = _resolve_model_config(env)
        return env, model_dir
    # Auto-detect: prefer bge-reranker-v2-m3 if ONNX model exists on disk.
    # Check the precision-specific dir first (env var or default), then any variant.
    _, preferred_dir, _ = _resolve_model_config("bge-reranker-v2-m3")
    if (Path(os.path.expanduser(preferred_dir)) / "model.onnx").exists():
        return "bge-reranker-v2-m3", preferred_dir
    for variant in _AVAILABLE_MODELS["bge-reranker-v2-m3"]["precisions"].values():
        d = variant["dir"]
        if (Path(os.path.expanduser(d)) / "model.onnx").exists():
            return "bge-reranker-v2-m3", d
    return "ms-marco-MiniLM-L-6-v2", _AVAILABLE_MODELS["ms-marco-MiniLM-L-6-v2"]["dir"]

_RERANKER_MODEL_NAME, _RERANKER_DEFAULT_DIR = _resolve_reranker_model()

# Circuit breaker
_FIRST_FAILURE_TIME: float = 0.0
_CIRCUIT_BREAKER_COOLDOWN_S = 300  # 5 minutes


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def cross_encoder_score(
    query: str,
    passages: list[str],
    temporal_metadata: list[str] | None = None,
) -> list[float] | None:
    """Score (query, passage) pairs using a cross-encoder model.

    Args:
        query: The search query string.
        passages: List of passage strings to score against the query.
        temporal_metadata: Optional list of date strings (one per passage).
            When provided, prepended to each passage to help the reranker
            consider temporal relevance (P2 improvement).

    Returns:
        List of float scores (one per passage), or None if the model is
        unavailable (disabled via env var or failed to load).
        Returns [] for empty passages list.
    """
    if not passages:
        return []

    if os.environ.get("OMEGA_CROSS_ENCODER") == "0":
        logger.debug("Cross-encoder disabled (OMEGA_CROSS_ENCODER=0)")
        return None

    model = _get_reranker_model()
    if model is None:
        return None

    tokenizer, session = model

    try:
        # Enrich passages with temporal metadata when available (P2)
        enriched_passages = passages
        if temporal_metadata and len(temporal_metadata) == len(passages):
            enriched_passages = [
                f"[Date: {date}] {passage}" if date else passage
                for date, passage in zip(temporal_metadata, passages)
            ]

        # Cross-encoder: encode (query, passage) pairs
        pairs = [(query, p) for p in enriched_passages]
        encoded_batch = tokenizer.encode_batch(pairs)

        ids = np.array([e.ids for e in encoded_batch], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded_batch], dtype=np.int64)

        feed = {"input_ids": ids, "attention_mask": attention_mask}

        # Check if the model expects token_type_ids
        input_names = {inp.name for inp in session.get_inputs()}
        if "token_type_ids" in input_names:
            token_type_ids = np.array(
                [e.type_ids for e in encoded_batch], dtype=np.int64
            )
            feed["token_type_ids"] = token_type_ids

        outputs = session.run(None, feed)

        # Cross-encoder output is logits — shape (batch_size, 1) or (batch_size,)
        logits = outputs[0]
        if logits.ndim == 2:
            logits = logits.squeeze(-1)

        return logits.tolist()

    except Exception as e:
        logger.warning(f"Cross-encoder scoring failed: {e}")
        return None


def get_reranker_model_info() -> dict[str, Any]:
    """Return model metadata dict with 'model_name' and 'available' keys."""
    return {
        "model_name": _RERANKER_MODEL_NAME,
        "available": _RERANKER_MODEL is not None,
    }


def preload_reranker_model() -> bool:
    """Pre-load the cross-encoder model. Returns True if loaded successfully."""
    model = _get_reranker_model()
    return model is not None


def reset_reranker_state():
    """Reset all module state for testing."""
    global _RERANKER_MODEL, _FIRST_FAILURE_TIME
    _RERANKER_MODEL = None
    _FIRST_FAILURE_TIME = 0.0
    if hasattr(_get_reranker_model, "_attempt_count"):
        _get_reranker_model._attempt_count = 0


def download_model(
    target_dir: str | None = None,
    model_name: str | None = None,
    precision: str | None = None,
) -> str | None:
    """Download cross-encoder model from HuggingFace Hub.

    Args:
        target_dir: Directory to download into. Defaults to model-specific dir.
        model_name: Model name from _AVAILABLE_MODELS. Defaults to current selection.
        precision: "fp32" (~2.3 GB) or "int8" (~571 MB). Defaults to
            OMEGA_RERANKER_PRECISION env var, then "fp32".

    Returns:
        Path to the model directory, or None on failure.
    """
    model_name = model_name or _RERANKER_MODEL_NAME
    repo_id, default_dir, files_to_download = _resolve_model_config(model_name, precision)

    if target_dir is None:
        target_dir = os.path.expanduser(default_dir)

    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded (all expected files present)
    all_present = all(
        (target_path / local_name).exists()
        for _, local_name in files_to_download
    )
    if all_present:
        logger.info(f"Cross-encoder model already exists at {target_path}")
        return str(target_path)

    try:
        from huggingface_hub import hf_hub_download

        for repo_path, local_name in files_to_download:
            dest = target_path / local_name
            if dest.exists():
                continue
            logger.info(f"Downloading {repo_path} from {repo_id}...")
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=repo_path,
                local_dir=str(target_path),
            )
            # hf_hub_download may place the file in a subdirectory; move it to
            # the expected location if needed. Compare resolved paths: the hub
            # returns a canonical path, and a target under a symlink (macOS
            # /tmp, a linked ~/.cache) would otherwise be copied onto itself.
            downloaded_path = Path(downloaded)
            if downloaded_path.resolve() != dest.resolve() and downloaded_path.exists():
                import shutil
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(downloaded_path), str(dest))

        if (target_path / "model.onnx").exists():
            logger.info(f"Cross-encoder model downloaded to {target_path}")
            return str(target_path)
        else:
            # hf_hub_download may have put files in onnx/ subdir
            onnx_subdir = target_path / "onnx"
            if (onnx_subdir / "model.onnx").exists():
                import shutil
                shutil.copy2(str(onnx_subdir / "model.onnx"), str(target_path / "model.onnx"))
                # Also copy sidecar data file if present (required by bge-reranker-v2-m3)
                if (onnx_subdir / "model.onnx_data").exists():
                    shutil.copy2(str(onnx_subdir / "model.onnx_data"), str(target_path / "model.onnx_data"))
                logger.info(f"Cross-encoder model downloaded to {target_path}")
                return str(target_path)
            logger.error("model.onnx not found after download")
            return None

    except ImportError:
        logger.error("huggingface_hub not installed — cannot download model", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Failed to download cross-encoder model: {e}", exc_info=True)
        return None


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------


def _get_model_dir() -> str | None:
    """Get the cross-encoder model directory, checking if model files exist."""
    model_dir = Path(os.path.expanduser(_RERANKER_DEFAULT_DIR))
    model_path = model_dir / "model.onnx"
    if model_path.exists():
        return str(model_dir)

    # Environment override
    env_dir = os.environ.get("OMEGA_CROSS_ENCODER_DIR")
    if env_dir:
        env_path = Path(env_dir) / "model.onnx"
        if env_path.exists():
            return env_dir

    return None


def _get_reranker_model():
    """Lazy-load the cross-encoder model with circuit breaker.

    Returns (tokenizer, session) tuple or None if unavailable.
    """
    global _RERANKER_MODEL, _FIRST_FAILURE_TIME

    if _RERANKER_MODEL is not None:
        return _RERANKER_MODEL

    # Circuit breaker: allow up to 3 load attempts before giving up
    if not hasattr(_get_reranker_model, "_attempt_count"):
        _get_reranker_model._attempt_count = 0
    if _get_reranker_model._attempt_count >= 3:
        # Time-based recovery: reset after cooldown period
        if _FIRST_FAILURE_TIME > 0 and (
            _time_module.monotonic() - _FIRST_FAILURE_TIME
        ) >= _CIRCUIT_BREAKER_COOLDOWN_S:
            _get_reranker_model._attempt_count = 0
            _FIRST_FAILURE_TIME = 0.0
            logger.info("Reranker circuit breaker cooldown expired, retrying model load")
        else:
            return None
    _get_reranker_model._attempt_count += 1
    if _get_reranker_model._attempt_count == 1:
        _FIRST_FAILURE_TIME = _time_module.monotonic()

    # Check env-var disable
    if os.environ.get("OMEGA_CROSS_ENCODER") == "0":
        logger.info("Cross-encoder disabled (OMEGA_CROSS_ENCODER=0)")
        return None

    # Check ONNX runtime availability
    try:
        import importlib.util

        if importlib.util.find_spec("onnxruntime") is None:
            logger.warning("onnxruntime not installed — cross-encoder unavailable")
            return None
    except Exception:
        return None

    # Find model directory, auto-download if missing
    model_dir = _get_model_dir()
    if model_dir is None:
        if os.environ.get("OMEGA_RERANKER_AUTODOWNLOAD", "1") == "0":
            logger.warning(
                "Cross-encoder model not found and OMEGA_RERANKER_AUTODOWNLOAD=0. "
                "Run `python -m omega --download-model` or call omega.reranker.download_model()."
            )
            return None
        precision = os.environ.get("OMEGA_RERANKER_PRECISION", "fp32")
        size_hint = "~2.3 GB" if precision == "fp32" else "~571 MB"
        logger.info(
            "Cross-encoder model not found, downloading %s (%s, precision=%s). "
            "Set OMEGA_RERANKER_AUTODOWNLOAD=0 to disable auto-download.",
            _RERANKER_MODEL_NAME, size_hint, precision,
        )
        downloaded = download_model()
        if downloaded is None:
            logger.warning(
                "Cross-encoder auto-download failed. "
                "Run `python -m omega --download-model` manually to retry."
            )
            return None
        model_dir = _get_model_dir()
        if model_dir is None:
            logger.error("Cross-encoder model still not found after download to %s", downloaded)
            return None

    try:
        import onnxruntime as ort
        from tokenizers import Tokenizer as FastTokenizer

        tokenizer = FastTokenizer.from_file(f"{model_dir}/tokenizer.json")
        tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        tokenizer.enable_truncation(max_length=512)

        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 4
        sess_opts.log_verbosity_level = 0
        sess_opts.enable_cpu_mem_arena = False  # Save RAM

        # CPU-only: CoreML leaks memory on long-running processes
        providers = ["CPUExecutionProvider"]
        with contextlib.redirect_stderr(io.StringIO()):
            session = ort.InferenceSession(
                f"{model_dir}/model.onnx",
                sess_options=sess_opts,
                providers=providers,
            )

        _RERANKER_MODEL = (tokenizer, session)
        _get_reranker_model._attempt_count = 0
        _FIRST_FAILURE_TIME = 0.0
        if _RERANKER_MODEL_NAME != "bge-reranker-v2-m3":
            logger.info(
                "Loaded cross-encoder ONNX model (%s). "
                "For better quality on conversational queries (at +2GB RAM cost), run: "
                "python3 scripts/convert_bge_reranker.py",
                _RERANKER_MODEL_NAME,
            )
        else:
            logger.info("Loaded cross-encoder ONNX model (%s)", _RERANKER_MODEL_NAME)
        return _RERANKER_MODEL

    except Exception as e:
        logger.warning(
            f"Failed to load cross-encoder model (attempt {_get_reranker_model._attempt_count}): {e}"
        )
        import traceback

        logger.debug(f"Cross-encoder load traceback: {traceback.format_exc()}")
        return None
