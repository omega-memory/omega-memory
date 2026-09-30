"""ONNX session options: bounded threads, no idle spinning."""

import pytest

from omega.onnx_session import make_session_options, onnx_thread_count

ort = pytest.importorskip("onnxruntime")


def test_default_threads_are_bounded(monkeypatch):
    monkeypatch.delenv("OMEGA_ONNX_THREADS", raising=False)
    monkeypatch.setattr("os.cpu_count", lambda: 14)
    assert onnx_thread_count() == 4
    monkeypatch.setattr("os.cpu_count", lambda: 2)
    assert onnx_thread_count() == 2
    monkeypatch.setattr("os.cpu_count", lambda: None)
    assert onnx_thread_count() == 1


def test_env_override(monkeypatch):
    monkeypatch.setenv("OMEGA_ONNX_THREADS", "8")
    assert onnx_thread_count() == 8


@pytest.mark.parametrize("raw", ["0", "-2", "many"])
def test_invalid_env_falls_back(monkeypatch, raw):
    monkeypatch.setenv("OMEGA_ONNX_THREADS", raw)
    monkeypatch.setattr("os.cpu_count", lambda: 14)
    assert onnx_thread_count() == 4


def test_options_disable_spinning_and_arena(monkeypatch):
    monkeypatch.setenv("OMEGA_ONNX_THREADS", "3")
    options = make_session_options(ort)
    assert options.intra_op_num_threads == 3
    assert options.inter_op_num_threads == 1
    assert options.enable_cpu_mem_arena is False
    assert options.get_session_config_entry("session.intra_op.allow_spinning") == "0"


def test_both_models_load_with_shared_options(monkeypatch):
    """The embedding and reranker loaders build their sessions from make_session_options."""
    import inspect

    import omega.embedding as embedding
    import omega.reranker as reranker

    for module in (embedding, reranker):
        source = inspect.getsource(module)
        assert "make_session_options(ort)" in source
        assert "ort.SessionOptions()" not in source
