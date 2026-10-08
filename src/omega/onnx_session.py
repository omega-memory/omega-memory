"""Shared ONNX Runtime session options for the embedding and reranker models.

ONNX Runtime's defaults suit a dedicated inference server, not a memory tool
that runs one small inference at a time beside the user's own work. It starts
one worker thread per core and keeps those threads spinning after each run
instead of letting them sleep. Measured on a 14-core machine with the default
models (bge-small-en-v1.5 and ms-marco-MiniLM-L-6-v2), that cost 6 to 7 times
each call's wall time in CPU, and about 45 ms of CPU burned in the 200 ms
after every inference while nothing was running.

Four threads with spinning off used half the CPU or less for the same wall
time on reranking and batch embedding, and nothing while idle. A single
embedding takes about 1 ms longer, because a sleeping thread has to wake.
"""

import logging
import os

logger = logging.getLogger("omega.onnx_session")

__all__ = ["onnx_thread_count", "make_session_options"]

_DEFAULT_MAX_THREADS = 4


def onnx_thread_count() -> int:
    """Threads per ONNX session: ``OMEGA_ONNX_THREADS``, else min(4, CPU count)."""
    raw = os.environ.get("OMEGA_ONNX_THREADS", "").strip()
    if raw:
        try:
            threads = int(raw)
        except ValueError:
            threads = 0
        if threads >= 1:
            return threads
        logger.warning("Ignoring OMEGA_ONNX_THREADS=%r: expected a positive integer", raw)
    return max(1, min(_DEFAULT_MAX_THREADS, os.cpu_count() or 1))


def make_session_options(ort):
    """SessionOptions for a CPU inference session: quiet, arena off, few threads, no spinning."""
    options = ort.SessionOptions()
    options.log_severity_level = 4
    options.log_verbosity_level = 0
    options.enable_cpu_mem_arena = False  # Saves ~50 MB RSS per model
    options.intra_op_num_threads = onnx_thread_count()
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return options
