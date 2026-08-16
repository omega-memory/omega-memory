"""
OMEGA Embeddings — ONNX-based embedding generation for semantic search.

Provides:
- generate_embedding(text) → normalized vector at the configured dimension
- generate_embeddings_batch(texts) → list of vectors
- Async variants for non-blocking operation
- LRU cache for repeated queries
- Hash-based fallback when no ML model available

Uses bge-small-en-v1.5 by default via ONNX Runtime (~90MB RAM, ~170MB with arena
disabled). The model, dimension, pooling and pad token are configurable through
the OMEGA_EMBEDDING_* variables (see omega.embedding_config) so multilingual
models such as bge-m3 can be selected without patching this module.
Falls back to all-MiniLM-L6-v2 if bge model not downloaded, or hash-based pseudo-embeddings if ONNX unavailable.
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Any
import hashlib
import logging
import math
import os

from omega.embedding_config import DEFAULT_EMBEDDING_MODEL, get_embedding_config

__all__ = [
    "generate_embedding",
    "generate_embeddings_batch",
    "generate_embedding_async",
    "generate_embeddings_batch_async",
    "preload_embedding_model",
    "preload_embedding_model_async",
    "get_embedding_model_info",
    "get_embedding_info",
    "get_active_backend",
    "has_onnx_runtime",
    "has_sentence_transformers",
    "reset_embedding_state",
]

logger = logging.getLogger("omega.embedding")

# NumPy for vectorized operations (required by ONNX backend)
try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]  # ONNX path won't be entered

# Embedding model state
_EMBEDDING_MODEL = None
_EMBEDDING_BACKEND = None  # "onnx" only (sentence-transformers removed to prevent 7.5GB PyTorch blowup)
_LOAD_ATTEMPTED = False  # Circuit breaker: don't retry after first failure
_EMBEDDING_CONFIG = get_embedding_config()
_EMBEDDING_MODEL_NAME = _EMBEDDING_CONFIG.model_name
# Version tracks the shipped default only. A configured model has its own
# versioning we cannot infer, and stamping "v1.5" on it would put a false
# version in the metadata of every memory it embeds.
_EMBEDDING_MODEL_VERSION = (
    "v1.5" if _EMBEDDING_MODEL_NAME == DEFAULT_EMBEDDING_MODEL else "configured"
)
_EMBEDDING_CACHE: OrderedDict = OrderedDict()
_CACHE_MAX = int(os.environ.get("OMEGA_EMBEDDING_CACHE_MAX", "512"))
_EMBEDDING_CACHE_MAX = _CACHE_MAX

# Idle-timeout model unloading — save RAM when not actively embedding
import time as _time_module

_LAST_EMBED_TIME: float = 0.0  # monotonic timestamp of last embedding call
_LAST_UNLOAD_CHECK: float = 0.0  # monotonic timestamp of last idle check
_IDLE_TIMEOUT_S = 600  # 10 minutes
_UNLOAD_CHECK_INTERVAL_S = 60  # check at most once per 60s

# Circuit breaker cooldown — allow retry after transient failures
_FIRST_FAILURE_TIME: float = 0.0  # monotonic timestamp of first failure in current window
_CIRCUIT_BREAKER_COOLDOWN_S = 300  # 5 minutes

# ONNX model paths (configured model primary, all-MiniLM-L6-v2 fallback)
_ONNX_MODEL_DIR = None
_ONNX_DEFAULT_DIR = _EMBEDDING_CONFIG.model_dir
_ONNX_FALLBACK_DIR = "~/.cache/omega/models/all-MiniLM-L6-v2-onnx"

# Availability checks (cached)
_ONNX_CHECKED = False
_ONNX_AVAILABLE = False


def get_embedding_model_info() -> Dict[str, Any]:
    """Return current embedding model name and version for metadata tracking."""
    return {
        "model_name": _EMBEDDING_MODEL_NAME,
        "model_version": _EMBEDDING_MODEL_VERSION,
        "model_loaded": _EMBEDDING_MODEL is not None,
        "backend": _EMBEDDING_BACKEND,
    }


def _resolve_model_name(model_name: Optional[str]) -> str:
    """Return the effective model name, falling back to the current global model.

    Args:
        model_name: Explicit model identifier (e.g. ``"nvidia/NV-Embed-v2"``),
            or ``None`` to use the currently-loaded model (``_EMBEDDING_MODEL_NAME``).

    Returns:
        The resolved model name string.
    """
    return model_name if model_name is not None else _EMBEDDING_MODEL_NAME


def reset_embedding_state():
    """Reset all embedding state including the circuit breaker.

    Call this after tests that set OMEGA_SKIP_EMBEDDINGS=1 to allow
    subsequent tests to load the real model.
    """
    global _EMBEDDING_MODEL, _EMBEDDING_BACKEND, _LOAD_ATTEMPTED
    global _ONNX_MODEL_DIR, _EMBEDDING_MODEL_NAME, _EMBEDDING_MODEL_VERSION
    global _FIRST_FAILURE_TIME
    _EMBEDDING_MODEL = None
    _EMBEDDING_BACKEND = None
    _LOAD_ATTEMPTED = False
    _ONNX_MODEL_DIR = None
    _EMBEDDING_MODEL_NAME = _EMBEDDING_CONFIG.model_name
    _EMBEDDING_MODEL_VERSION = (
        "v1.5" if _EMBEDDING_MODEL_NAME == DEFAULT_EMBEDDING_MODEL else "configured"
    )
    _FIRST_FAILURE_TIME = 0.0
    _EMBEDDING_CACHE.clear()
    if hasattr(_get_embedding_model, "_attempt_count"):
        _get_embedding_model._attempt_count = 0


def _maybe_unload_model():
    """Unload the embedding model if idle > 10 min to save RAM.

    Rate-limited to check at most once per 60s. On unload, resets the
    attempt count so the model can be lazy-reloaded on next call.
    Always unload when idle — don't gate on RSS threshold.
    """
    global _EMBEDDING_MODEL, _EMBEDDING_BACKEND, _LAST_UNLOAD_CHECK, _LAST_EMBED_TIME, _FIRST_FAILURE_TIME
    now = _time_module.monotonic()
    if now - _LAST_UNLOAD_CHECK < _UNLOAD_CHECK_INTERVAL_S:
        return
    _LAST_UNLOAD_CHECK = now

    if _EMBEDDING_MODEL is None:
        return
    if _LAST_EMBED_TIME == 0.0:
        return
    if now - _LAST_EMBED_TIME < _IDLE_TIMEOUT_S:
        return

    logger.info("Idle-unloading embedding model (idle > %ds)", _IDLE_TIMEOUT_S)
    _EMBEDDING_MODEL = None
    _EMBEDDING_BACKEND = None
    _LAST_EMBED_TIME = 0.0  # Reset so idle detection works correctly on reload
    _EMBEDDING_CACHE.clear()  # Free cached embeddings too
    # Reset circuit breaker so model can reload
    _FIRST_FAILURE_TIME = 0.0
    if hasattr(_get_embedding_model, "_attempt_count"):
        _get_embedding_model._attempt_count = 0


def _check_onnx_runtime() -> bool:
    """Check if onnxruntime is available (lazy check, cached result)."""
    global _ONNX_CHECKED, _ONNX_AVAILABLE
    if not _ONNX_CHECKED:
        try:
            import importlib.util

            _ONNX_AVAILABLE = importlib.util.find_spec("onnxruntime") is not None
        except Exception as e:
            logger.debug("Backend detection failed: %s", e)
            _ONNX_AVAILABLE = False
        _ONNX_CHECKED = True
    return _ONNX_AVAILABLE


def has_sentence_transformers() -> bool:
    """Stub for backward compatibility. Always returns False.

    PyTorch/sentence-transformers fallback was removed to prevent 7.5GB memory
    blowup when ONNX fails. Set OMEGA_ALLOW_PYTORCH_FALLBACK=1 to re-enable.
    """
    return False


def has_onnx_runtime() -> bool:
    """Check if onnxruntime is available."""
    return _check_onnx_runtime()


def _get_onnx_model_dir() -> Optional[str]:
    """Get the ONNX model directory path, checking if model exists.

    Checks in order: env override, configured model dir, all-MiniLM-L6-v2 (fallback).

    The env override is consulted FIRST. It used to come second, after the
    default cache path — and since ``omega setup --download-model`` installs
    exactly that path, the override was unreachable on any normal install.
    An operator who points OMEGA_ONNX_MODEL_DIR at a different model means it.
    """
    global _ONNX_MODEL_DIR, _EMBEDDING_MODEL_NAME, _EMBEDDING_MODEL_VERSION
    if _ONNX_MODEL_DIR is not None:
        return _ONNX_MODEL_DIR

    import os
    from pathlib import Path

    # Environment override — wins over the packaged default.
    env_dir = os.environ.get("OMEGA_ONNX_MODEL_DIR")
    if env_dir:
        env_path = Path(os.path.expanduser(env_dir)) / "model.onnx"
        if env_path.exists():
            _ONNX_MODEL_DIR = os.path.expanduser(env_dir)
            return _ONNX_MODEL_DIR
        logger.warning(
            "OMEGA_ONNX_MODEL_DIR=%s has no model.onnx; falling back to the configured model",
            env_dir,
        )

    # Configured model (OMEGA_EMBEDDING_MODEL / OMEGA_EMBEDDING_MODEL_DIR).
    model_dir = Path(os.path.expanduser(_ONNX_DEFAULT_DIR))
    model_path = model_dir / "model.onnx"
    if model_path.exists():
        _ONNX_MODEL_DIR = str(model_dir)
        return _ONNX_MODEL_DIR

    # Fallback: all-MiniLM-L6-v2 (existing installs)
    fallback_dir = Path(os.path.expanduser(_ONNX_FALLBACK_DIR))
    fallback_path = fallback_dir / "model.onnx"
    if fallback_path.exists():
        _ONNX_MODEL_DIR = str(fallback_dir)
        _EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
        _EMBEDDING_MODEL_VERSION = "v2"
        logger.info("Using fallback model all-MiniLM-L6-v2. Run 'omega setup --download-model' to upgrade.")
        return _ONNX_MODEL_DIR

    return None


def _get_embedding_model():
    """Lazy-load the embedding model.

    Priority: ONNX Runtime (~90MB) > hash fallback (0 RAM).
    PyTorch/sentence-transformers gated behind OMEGA_ALLOW_PYTORCH_FALLBACK=1.
    """
    global _EMBEDDING_MODEL, _EMBEDDING_BACKEND, _LOAD_ATTEMPTED
    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL

    # Allow up to 3 load attempts before giving up; reset after cooldown
    global _FIRST_FAILURE_TIME
    if not hasattr(_get_embedding_model, "_attempt_count"):
        _get_embedding_model._attempt_count = 0
    if _get_embedding_model._attempt_count >= 3:
        # Time-based recovery: reset after cooldown period
        if _FIRST_FAILURE_TIME > 0 and (_time_module.monotonic() - _FIRST_FAILURE_TIME) >= _CIRCUIT_BREAKER_COOLDOWN_S:
            _get_embedding_model._attempt_count = 0
            _FIRST_FAILURE_TIME = 0.0
            logger.info("Circuit breaker cooldown expired, retrying model load")
        else:
            return None
    _get_embedding_model._attempt_count += 1
    if _get_embedding_model._attempt_count == 1:
        _FIRST_FAILURE_TIME = _time_module.monotonic()

    _LOAD_ATTEMPTED = True

    os.environ.setdefault("TQDM_DISABLE", "1")

    if os.environ.get("OMEGA_SKIP_EMBEDDINGS") == "1":
        logger.info("Skipping embedding model load (OMEGA_SKIP_EMBEDDINGS=1)")
        return None

    # Fail-fast: if no model directory exists, skip retries entirely
    if _check_onnx_runtime():
        from pathlib import Path as _Path
        _primary = _Path(os.path.expanduser(_ONNX_DEFAULT_DIR))
        _fallback = _Path(os.path.expanduser(_ONNX_FALLBACK_DIR))
        _env_dir = os.environ.get("OMEGA_ONNX_MODEL_DIR")
        _no_model_dir = (
            not _primary.exists()
            and not _fallback.exists()
            and (_env_dir is None or not _Path(_env_dir).exists())
        )
        if _no_model_dir and os.environ.get("OMEGA_ALLOW_PYTORCH_FALLBACK") != "1":
            logger.info("No ONNX model directory found; skipping retries")
            _get_embedding_model._attempt_count = 3  # prevent further retries
            return None

    # Try ONNX Runtime first
    if _check_onnx_runtime():
        onnx_dir = _get_onnx_model_dir()
        if onnx_dir:
            try:
                import onnxruntime as ort
                from tokenizers import Tokenizer as FastTokenizer

                tokenizer = FastTokenizer.from_file(f"{onnx_dir}/tokenizer.json")
                tokenizer.enable_padding(
                    pad_id=_EMBEDDING_CONFIG.pad_id, pad_token=_EMBEDDING_CONFIG.pad_token
                )
                tokenizer.enable_truncation(max_length=512)
                sess_opts = ort.SessionOptions()
                sess_opts.log_severity_level = 4
                sess_opts.log_verbosity_level = 0
                sess_opts.enable_cpu_mem_arena = False  # Save ~50MB RAM
                # CPU-only: CoreML leaks ~700KB/op in native memory on long-running
                # processes (profiled Feb 2026). Speed difference is negligible for
                # single 384-dim embeddings (<5ms either way).
                import contextlib
                import io

                providers = ["CPUExecutionProvider"]
                with contextlib.redirect_stderr(io.StringIO()):
                    session = ort.InferenceSession(
                        f"{onnx_dir}/model.onnx",
                        sess_options=sess_opts,
                        providers=providers,
                    )
                _EMBEDDING_MODEL = (tokenizer, session)
                _EMBEDDING_BACKEND = "onnx"
                _get_embedding_model._attempt_count = 0
                _FIRST_FAILURE_TIME = 0.0
                logger.info("Loaded ONNX embedding model")
                return _EMBEDDING_MODEL
            except Exception as e:
                logger.warning(f"Failed to load ONNX model (attempt {_get_embedding_model._attempt_count}): {e}")
                import traceback

                logger.debug(f"ONNX load traceback: {traceback.format_exc()}")

    # PyTorch/sentence-transformers fallback: OFF by default.
    # Loading sentence-transformers pulls in torch, cv2, scipy, sklearn, pandas, PIL,
    # inflating a single MCP server process from ~1.6GB to 7.5GB. This caused database
    # locking cascades and ConnectionResetErrors for all concurrent sessions.
    # Gate behind explicit env var for users who need it and accept the RAM cost.
    if os.environ.get("OMEGA_ALLOW_PYTORCH_FALLBACK") == "1":
        try:
            import importlib.util

            if importlib.util.find_spec("sentence_transformers") is not None:
                from sentence_transformers import SentenceTransformer

                _EMBEDDING_MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5")
                _EMBEDDING_BACKEND = "sentence-transformers"
                _get_embedding_model._attempt_count = 0
                _FIRST_FAILURE_TIME = 0.0
                logger.warning(
                    "Loaded sentence-transformers (PyTorch) fallback. "
                    "This uses ~6GB extra RAM. Consider fixing ONNX instead."
                )
                return _EMBEDDING_MODEL
        except Exception as e:
            logger.warning(f"PyTorch fallback failed: {e}")

    if _EMBEDDING_MODEL is None:
        logger.critical(
            "No embedding model loaded (attempt %d/3). "
            "ONNX available: %s, ONNX dir: %s. "
            "Falling back to hash-based pseudo-embeddings (no semantic search). "
            "Fix: run 'omega setup --download-model' to install ONNX model.",
            _get_embedding_model._attempt_count,
            _check_onnx_runtime(),
            _get_onnx_model_dir(),
        )

    return _EMBEDDING_MODEL


def preload_embedding_model() -> bool:
    """Preload the embedding model at startup (optional warmup).

    Tries the shared daemon first. If daemon is healthy, skips local model load.
    """
    # Try daemon first -- if it's running, no need to load in-process
    try:
        from omega.embedding_client import get_client

        client = get_client()
        if client is not None:
            health = client.health()
            if health and health.get("status") == "ok":
                logger.info("Embedding daemon is healthy, skipping local model load")
                return True
    except Exception as e:
        logger.debug("Embedding daemon health check failed: %s", e)

    # Fall back to in-process ONNX
    if not _check_onnx_runtime():
        return False
    try:
        model = _get_embedding_model()
        if model is not None:
            generate_embedding("test")
            return True
    except Exception as e:
        logger.warning("Model preload failed: %s", e)
    return False


def get_active_backend() -> Optional[str]:
    """Return the currently active embedding backend, or None if using hash fallback."""
    return _EMBEDDING_BACKEND


def _hash_embedding(text: str, dimension: int = _EMBEDDING_CONFIG.dim) -> List[float]:
    """Fallback: deterministic pseudo-embedding from text hash."""
    hash_digest = hashlib.md5(text.encode()).digest()
    seed = int.from_bytes(hash_digest[:4], byteorder="big")

    import random

    rng = random.Random(seed)
    vector = [rng.gauss(0, 1) for _ in range(dimension)]

    magnitude = math.sqrt(sum(x * x for x in vector))
    if magnitude == 0:
        return [1.0 / math.sqrt(dimension)] * dimension
    return [x / magnitude for x in vector]


def _has_embedding_backend() -> bool:
    """Check if any embedding backend is available."""
    return _EMBEDDING_MODEL is not None or _check_onnx_runtime()


def _onnx_encode(tokenizer, session, texts: List[str]) -> "np.ndarray":
    """Encode texts using ONNX Runtime. Returns normalized embeddings."""
    batch = tokenizer.encode_batch(texts)
    ids = np.array([b.ids for b in batch], dtype=np.int64)
    mask = np.array([b.attention_mask for b in batch], dtype=np.int64)
    feed = {"input_ids": ids, "attention_mask": mask}
    input_names = {i.name for i in session.get_inputs()}
    if "token_type_ids" in input_names:
        feed["token_type_ids"] = np.zeros_like(ids)
    outputs = session.run(None, feed)
    embeddings = outputs[1] if len(outputs) > 1 else outputs[0]
    if embeddings.ndim == 3:
        # Pooling is model-specific: bge-small uses mean over the unmasked
        # tokens, bge-m3 and other CLS-trained models take the first token.
        # Pooling the wrong way produces vectors that are numerically valid
        # and semantically wrong, so it is configuration, not a heuristic.
        if _EMBEDDING_CONFIG.pooling == "cls":
            embeddings = embeddings[:, 0, :]
        else:
            mask_expanded = mask[:, :, np.newaxis].astype(np.float32)
            sum_emb = np.sum(embeddings * mask_expanded, axis=1)
            sum_mask = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)
            embeddings = sum_emb / sum_mask
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, a_min=1e-9, a_max=None)


_embedding_degraded = False


def is_embedding_degraded() -> bool:
    """Return True if the last embedding call fell back to hash embedding."""
    return _embedding_degraded


def generate_embedding(text: str, dimension: int = _EMBEDDING_CONFIG.dim) -> List[float]:
    """Generate semantic embedding from text. Returns a normalized vector.

    Priority: daemon > in-process ONNX > hash fallback.
    """
    global _LAST_EMBED_TIME, _embedding_degraded, _EMBEDDING_BACKEND
    # Check for idle-timeout unloading opportunity
    _maybe_unload_model()

    # Local cache check first (works regardless of backend)
    cache_key = hashlib.md5(text.encode()).hexdigest()
    if cache_key in _EMBEDDING_CACHE:
        _EMBEDDING_CACHE.move_to_end(cache_key)
        _LAST_EMBED_TIME = _time_module.monotonic()
        _embedding_degraded = False
        return _EMBEDDING_CACHE[cache_key]

    # Try shared embedding daemon first (avoids per-process model loading)
    try:
        from omega.embedding_client import get_client

        client = get_client()
        if client is not None:
            result = client.embed_single(text)
            if result is not None:
                _EMBEDDING_CACHE[cache_key] = result
                while len(_EMBEDDING_CACHE) > _EMBEDDING_CACHE_MAX:
                    _EMBEDDING_CACHE.popitem(last=False)
                _LAST_EMBED_TIME = _time_module.monotonic()
                _embedding_degraded = False
                _EMBEDDING_BACKEND = "daemon"
                return result
    except Exception as e:
        logger.debug("Embedding daemon unavailable: %s", e)

    # In-process ONNX fallback
    if _has_embedding_backend():
        try:
            model = _get_embedding_model()
            if model is not None:
                # Dispatch by model type, not _EMBEDDING_BACKEND flag.
                # The flag can be stale ("daemon") after daemon→ONNX fallback.
                if isinstance(model, tuple):
                    tokenizer, session = model
                    emb = _onnx_encode(tokenizer, session, [text])
                    result = emb[0].tolist()
                    _EMBEDDING_BACKEND = "onnx"
                else:
                    embedding = model.encode(text, normalize_embeddings=True)
                    result = embedding.tolist()
                _EMBEDDING_CACHE[cache_key] = result
                while len(_EMBEDDING_CACHE) > _EMBEDDING_CACHE_MAX:
                    _EMBEDDING_CACHE.popitem(last=False)
                _LAST_EMBED_TIME = _time_module.monotonic()
                _embedding_degraded = False
                return result
            else:
                logger.warning("Embedding model is None, circuit-breaker tripped. Using hash fallback.")
        except Exception as e:
            logger.warning(f"Embedding generation failed, falling back to hash: {e}")

    _embedding_degraded = True
    return _hash_embedding(text, dimension)


def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for multiple texts in a single batch.

    Priority: daemon > in-process ONNX > hash fallback.
    """
    global _EMBEDDING_BACKEND
    if not texts:
        return []

    # Try shared embedding daemon first
    try:
        from omega.embedding_client import get_client

        client = get_client()
        if client is not None:
            result = client.embed_batch(texts)
            if result is not None and len(result) == len(texts):
                return result
    except Exception as e:
        logger.debug("Batch embedding daemon unavailable: %s", e)

    if _has_embedding_backend():
        try:
            model = _get_embedding_model()
            if model is not None:
                if isinstance(model, tuple):
                    tokenizer, session = model
                    _EMBEDDING_BACKEND = "onnx"
                    # CoreML can't handle multi-item batches — use batch_size=1
                    providers = session.get_providers()
                    batch_size = 1 if any("CoreML" in p for p in providers) else 32
                    all_results = []
                    for i in range(0, len(texts), batch_size):
                        batch = texts[i : i + batch_size]
                        embs = _onnx_encode(tokenizer, session, batch)
                        all_results.extend(embs.tolist())
                    return all_results
                else:
                    embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32)
                    return [e.tolist() for e in embeddings]
            else:
                logger.warning(
                    "Batch embedding: model is None (load failed or circuit-breaker tripped). "
                    "Falling back to hash embeddings — these will NOT be findable by vector search."
                )
        except Exception as e:
            logger.warning(f"Batch embedding failed, falling back to single-item: {e}")
            # Fall back to per-item encoding before resorting to hash
            try:
                results = []
                for text in texts:
                    emb = generate_embedding(text)
                    results.append(emb)
                if get_active_backend() is not None:
                    return results
                logger.warning("Single-item fallback produced hash embeddings")
            except Exception as e2:
                logger.warning(f"Single-item fallback also failed: {e2}")

    logger.warning(f"Using hash-fallback embeddings for {len(texts)} texts")
    return [_hash_embedding(t) for t in texts]


def get_embedding_info() -> Dict[str, Any]:
    """Get info about the current embedding backend."""
    has_onnx = _check_onnx_runtime()
    if _EMBEDDING_BACKEND:
        backend = _EMBEDDING_BACKEND
    elif has_onnx:
        backend = "onnx (not loaded)"
    else:
        backend = "hash-fallback"
    return {
        "backend": backend,
        "model": _EMBEDDING_MODEL_NAME,
        "model_loaded": _EMBEDDING_MODEL is not None,
        "onnx_available": has_onnx,
        "onnx_model_dir": _get_onnx_model_dir() if has_onnx else None,
        "dimension": _EMBEDDING_CONFIG.dim,
        "cache_size": len(_EMBEDDING_CACHE),
        "lazy_loading": True,
    }


# Async support
import asyncio
from concurrent.futures import ThreadPoolExecutor

_EMBEDDING_EXECUTOR: Optional[ThreadPoolExecutor] = None


def _get_embedding_executor() -> ThreadPoolExecutor:
    """Get or create the thread pool executor for embedding operations."""
    global _EMBEDDING_EXECUTOR
    if _EMBEDDING_EXECUTOR is None:
        _EMBEDDING_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embedding")
    return _EMBEDDING_EXECUTOR


async def generate_embedding_async(text: str, dimension: int = _EMBEDDING_CONFIG.dim) -> List[float]:
    """Generate embedding asynchronously (non-blocking)."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_get_embedding_executor(), generate_embedding, text, dimension)


async def generate_embeddings_batch_async(texts: List[str]) -> List[List[float]]:
    """Generate batch embeddings asynchronously."""
    if not texts:
        return []
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_get_embedding_executor(), generate_embeddings_batch, texts)


async def preload_embedding_model_async() -> bool:
    """Preload embedding model asynchronously."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_get_embedding_executor(), preload_embedding_model)
