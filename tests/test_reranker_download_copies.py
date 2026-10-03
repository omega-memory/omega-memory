"""The reranker download keeps one copy of each model file.

The hub writes ``onnx/model.onnx`` under the model directory and the loader
reads ``model.onnx`` beside it. The download used to copy one to the other
and keep both: a second 87 MB for the default model.
"""

import sys
import types
from pathlib import Path

import pytest

import omega.reranker as reranker

MODEL = "ms-marco-MiniLM-L-6-v2"
PAYLOAD = b"onnx-bytes" * 4096


def _tree_bytes(root: Path) -> int:
    return sum(f.stat().st_size for f in root.rglob("*") if f.is_file())


@pytest.fixture
def fake_hub(monkeypatch):
    """A stand-in hub that writes each file at its repository path, as the real one does."""
    calls = []

    def hf_hub_download(repo_id, filename, local_dir):
        calls.append(filename)
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(PAYLOAD if filename.endswith(".onnx") else b"{}")
        return str(path)

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=hf_hub_download)
    )
    return calls


def test_download_keeps_a_single_copy(tmp_path, fake_hub):
    target = tmp_path / "model"

    assert reranker.download_model(str(target), model_name=MODEL) == str(target)

    assert (target / "model.onnx").read_bytes() == PAYLOAD
    assert not (target / "onnx").exists()
    assert _tree_bytes(target) == len(PAYLOAD) + 2 * len(b"{}")


def test_a_file_from_the_hubs_own_cache_is_copied_not_moved(tmp_path, monkeypatch):
    hub_cache = tmp_path / "hub-cache"
    hub_cache.mkdir()

    def hf_hub_download(repo_id, filename, local_dir):
        path = hub_cache / Path(filename).name
        path.write_bytes(PAYLOAD)
        return str(path)

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=hf_hub_download)
    )
    target = tmp_path / "model"

    assert reranker.download_model(str(target), model_name=MODEL) == str(target)

    assert (target / "model.onnx").read_bytes() == PAYLOAD
    assert (hub_cache / "model.onnx").exists(), "the hub's cache is not ours to empty"


def test_an_existing_install_loses_its_duplicate(tmp_path, fake_hub):
    target = tmp_path / "model"
    reranker.download_model(str(target), model_name=MODEL)
    (target / "onnx").mkdir()
    (target / "onnx" / "model.onnx").write_bytes(PAYLOAD)  # what 1.5.19 left behind
    fake_hub.clear()

    reranker.download_model(str(target), model_name=MODEL)

    assert fake_hub == [], "a complete model is not downloaded again"
    assert not (target / "onnx").exists()
    assert (target / "model.onnx").read_bytes() == PAYLOAD


def test_a_different_file_at_the_hub_path_is_kept(tmp_path, fake_hub):
    target = tmp_path / "model"
    reranker.download_model(str(target), model_name=MODEL)
    (target / "onnx").mkdir()
    other = target / "onnx" / "model.onnx"
    other.write_bytes(PAYLOAD[::-1])

    reranker.download_model(str(target), model_name=MODEL)

    assert other.read_bytes() == PAYLOAD[::-1]


def test_only_our_own_model_directories_are_tidied(tmp_path, monkeypatch):
    ours = tmp_path / "ours"
    monkeypatch.setitem(reranker._AVAILABLE_MODELS[MODEL], "dir", str(ours))

    assert ("onnx/model.onnx", "model.onnx") in reranker._downloaded_files(ours)
    assert reranker._downloaded_files(tmp_path / "set-by-the-user") == []
