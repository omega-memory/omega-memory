"""Fresh-install behaviour: model downloads, doctor's verdict on them, and first-session noise.

Every fresh Core install from 1.0 to 1.5.16 ended `omega setup` with an HTTP
404, because the tokenizer was fetched from the wrong Hugging Face path, and
then ran on hash pseudo-embeddings while `omega doctor` said embeddings worked.
These tests pin the download file map, the post-download check, doctor's
verdict, and the reranker pre-fetch.
"""
import argparse
import json
import sqlite3

import pytest

from omega import cli


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------


def test_minilm_file_map_fetches_weights_from_onnx_and_tokenizer_from_root():
    files = cli._MINILM_MODEL_FILES
    assert files["model.onnx"].endswith("/resolve/main/onnx/model.onnx")
    for name in ("tokenizer.json", "config.json", "tokenizer_config.json", "vocab.txt"):
        assert "/onnx/" not in files[name], name
        assert files[name].endswith(f"/resolve/main/{name}")


def test_download_minilm_reports_a_model_that_cannot_load(tmp_path, monkeypatch):
    def only_the_weights(url, target):
        if target.name == "model.onnx":
            target.write_bytes(b"onnx")

    monkeypatch.setattr(cli, "_download_file", only_the_weights)
    errors: list = []

    assert cli._download_minilm_model(tmp_path, errors) is False
    assert errors == ["tokenizer.json not present after download"]


def test_download_minilm_succeeds_when_weights_and_tokenizer_arrive(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_download_file", lambda url, target: target.write_bytes(b"x"))
    errors: list = []

    assert cli._download_minilm_model(tmp_path, errors) is True
    assert errors == []
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(cli._MINILM_MODEL_FILES)


def test_download_minilm_reports_http_failures_as_setup_errors(tmp_path, monkeypatch):
    def refuse(url, target):
        raise OSError("HTTP Error 404: Not Found")

    monkeypatch.setattr(cli, "_download_file", refuse)
    errors: list = []

    assert cli._download_minilm_model(tmp_path, errors) is False
    assert "404" in str(errors[0])


def test_setup_repairs_a_model_dir_that_has_weights_but_no_tokenizer(tmp_path, monkeypatch):
    """Installs made before 1.5.17 have model.onnx alone; setup must complete them, not call them fine."""
    minilm = tmp_path / "minilm"
    minilm.mkdir()
    (minilm / "model.onnx").write_bytes(b"onnx")
    monkeypatch.setattr(cli, "BGE_MODEL_DIR", tmp_path / "bge")
    monkeypatch.setattr(cli, "MINILM_MODEL_DIR", minilm)
    monkeypatch.setattr(cli, "_download_file", lambda url, target: target.write_bytes(b"x"))
    errors: list = []
    done: list = []

    cli._install_embedding_model(False, errors, done)

    assert errors == []
    assert done == ["Embedding model (repaired)"]
    assert (minilm / "tokenizer.json").exists()


def test_setup_leaves_a_complete_model_alone(tmp_path, monkeypatch):
    minilm = tmp_path / "minilm"
    minilm.mkdir()
    for name in cli._MODEL_LOAD_FILES:
        (minilm / name).write_bytes(b"x")
    monkeypatch.setattr(cli, "BGE_MODEL_DIR", tmp_path / "bge")
    monkeypatch.setattr(cli, "MINILM_MODEL_DIR", minilm)
    monkeypatch.setattr(cli, "_download_file", lambda url, target: pytest.fail("must not download"))
    done: list = []

    cli._install_embedding_model(False, [], done)

    assert done == ["Embedding model (already present)"]


# ---------------------------------------------------------------------------
# Reranker pre-fetch during setup
# ---------------------------------------------------------------------------


def test_setup_prefetches_the_reranker_when_absent(monkeypatch):
    from omega import reranker

    monkeypatch.setattr(reranker, "_get_model_dir", lambda: None)
    monkeypatch.setattr(reranker, "download_model", lambda: "/models/ms-marco")
    done: list = []
    skipped: list = []

    cli._download_reranker_model(done, skipped)

    assert done == ["Reranker model (downloaded)"]
    assert skipped == []


def test_setup_leaves_a_present_reranker_alone(monkeypatch):
    from omega import reranker

    monkeypatch.setattr(reranker, "_get_model_dir", lambda: "/models/present")
    monkeypatch.setattr(reranker, "download_model", lambda: pytest.fail("must not download"))
    done: list = []

    cli._download_reranker_model(done, [])

    assert done == ["Reranker model (already present)"]


def test_setup_treats_a_failed_reranker_download_as_skipped_not_error(monkeypatch, capsys):
    from omega import reranker

    monkeypatch.setattr(reranker, "_get_model_dir", lambda: None)
    monkeypatch.setattr(reranker, "download_model", lambda: None)
    done: list = []
    skipped: list = []

    cli._download_reranker_model(done, skipped)

    assert done == []
    assert skipped == ["Reranker model (download failed)"]
    assert "search works without it" in capsys.readouterr().out


def test_reranker_download_survives_a_symlinked_target_dir(tmp_path, monkeypatch):
    """The hub returns canonical paths; a target reached through a symlink must not be copied onto itself."""
    import huggingface_hub

    from omega import reranker

    real_dir = tmp_path / "real"
    real_dir.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real_dir)

    def fake_hub_download(repo_id, filename, local_dir):
        dest = (real_dir / filename)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x")
        return str(dest)  # canonical path, like the real hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hub_download)

    result = reranker.download_model(target_dir=str(alias), model_name="ms-marco-MiniLM-L-6-v2")

    assert result == str(alias)
    assert (alias / "model.onnx").exists()
    assert (alias / "tokenizer.json").exists()


# ---------------------------------------------------------------------------
# Doctor's embedding verdict
# ---------------------------------------------------------------------------


def _run_doctor_json(tmp_path, monkeypatch, capsys) -> dict:
    conn = sqlite3.connect(str(tmp_path / "omega.db"))
    conn.execute("CREATE TABLE memories (id TEXT, content TEXT, metadata TEXT)")
    conn.execute("CREATE VIRTUAL TABLE memories_fts USING fts5(content)")
    conn.execute("CREATE TABLE memories_vec (rowid INTEGER PRIMARY KEY, embedding BLOB)")
    conn.commit()
    conn.close()
    monkeypatch.setattr(cli, "OMEGA_DIR", tmp_path)
    monkeypatch.setattr(cli, "BGE_MODEL_DIR", tmp_path / "no-model")
    monkeypatch.setattr(cli, "MINILM_MODEL_DIR", tmp_path / "no-model")
    monkeypatch.setattr(cli, "SETTINGS_JSON_PATH", tmp_path / "no-settings.json")
    monkeypatch.setattr(cli, "_probe_hook_daemon", lambda timeout=1.0: ("absent", "/x/hook.sock"))
    monkeypatch.setattr(cli, "_mcp_servers_running", lambda: False)
    with pytest.raises(SystemExit):
        cli.cmd_doctor(argparse.Namespace(json=True, client=None))
    return json.loads(capsys.readouterr().out)


def _messages(report: dict, status: str) -> list[str]:
    return [c["message"] for c in report["checks"] if c["status"] == status]


def _fake_embedding(monkeypatch, *, model_loaded: bool):
    from omega import embedding
    from omega.embedding_config import get_embedding_config

    dim = get_embedding_config().dim
    monkeypatch.setattr(embedding, "generate_embedding", lambda text: [0.0] * dim)
    monkeypatch.setattr(
        embedding,
        "get_embedding_info",
        lambda: {"backend": "onnx" if model_loaded else "onnx (not loaded)", "model_loaded": model_loaded, "onnx_available": True},
    )


def test_doctor_fails_when_the_embedding_model_did_not_load(tmp_path, monkeypatch, capsys):
    _fake_embedding(monkeypatch, model_loaded=False)

    report = _run_doctor_json(tmp_path, monkeypatch, capsys)

    assert any(m.startswith("Embedding model did not load") for m in _messages(report, "fail"))
    assert not any(m.startswith("Embedding generation works") for m in _messages(report, "ok"))


def test_doctor_passes_when_the_embedding_model_loaded(tmp_path, monkeypatch, capsys):
    _fake_embedding(monkeypatch, model_loaded=True)

    report = _run_doctor_json(tmp_path, monkeypatch, capsys)

    assert any(m.startswith("Embedding generation works") for m in _messages(report, "ok"))
