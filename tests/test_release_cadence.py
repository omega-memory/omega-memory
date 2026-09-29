"""The release scripts allow at most one OMEGA release in any 7 days."""
import importlib.util
import json
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(name):
    """Import a release script by path; scripts/ is not an importable package."""
    spec = importlib.util.spec_from_file_location(f"_omega_{name}", _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preflight = _load_script("preflight")
release = _load_script("release")

LAST_UPLOAD = datetime(2026, 9, 29, 3, 50, tzinfo=timezone.utc)


@pytest.fixture
def last_release_at(monkeypatch):
    """Pretend the newest omega-memory upload on PyPI happened at the given time."""
    preflight.reset_results()

    def set_last_upload(uploaded):
        monkeypatch.setattr(preflight, "last_pypi_upload", lambda: ("1.5.18", uploaded))

    return set_last_upload


def test_release_inside_the_window_is_blocked(last_release_at, capsys):
    last_release_at(LAST_UPLOAD)
    preflight.check_release_cadence(now=LAST_UPLOAD + timedelta(days=6, hours=23))
    assert preflight.failures() == ["7 days since the last release"]
    assert "next window opens 2026-10-06 03:50 UTC" in capsys.readouterr().out


def test_release_window_opens_seven_days_after_the_last_upload(last_release_at):
    last_release_at(LAST_UPLOAD)
    preflight.check_release_cadence(now=LAST_UPLOAD + timedelta(days=7))
    assert preflight.failures() == []


def test_owner_approved_early_release_passes_and_is_recorded(last_release_at, capsys):
    last_release_at(LAST_UPLOAD)
    preflight.check_release_cadence(
        early_release="owner approved in chat: data-loss fix", now=LAST_UPLOAD + timedelta(hours=1),
    )
    assert preflight.failures() == []
    assert "early release approved by the owner: owner approved in chat: data-loss fix" in capsys.readouterr().out


def test_unknown_last_release_blocks(monkeypatch):
    preflight.reset_results()

    def offline():
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(preflight, "last_pypi_upload", offline)
    preflight.check_release_cadence()
    assert preflight.failures() == ["last PyPI release is known"]


def test_last_pypi_upload_is_the_newest_upload_of_any_version(tmp_path):
    # A backport uploaded after a newer version is still the last release.
    pypi_json = tmp_path / "omega-memory.json"
    pypi_json.write_text(json.dumps({"releases": {
        "1.5.18": [{"upload_time_iso_8601": "2026-09-29T03:50:00.307390Z"}],
        "1.5.17": [
            {"upload_time_iso_8601": "2026-09-16T08:30:00.000000Z"},
            {"upload_time_iso_8601": "2026-09-30T12:00:00.000000Z"},
        ],
        "1.5.0": [],
    }}))

    version, uploaded = preflight.last_pypi_upload(pypi_json.as_uri())

    assert (version, uploaded) == ("1.5.17", datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc))


def test_release_script_stops_before_bumping_inside_the_window(last_release_at, monkeypatch):
    last_release_at(datetime.now(timezone.utc) - timedelta(hours=1))
    monkeypatch.setattr(preflight, "check_version", lambda version: None)
    monkeypatch.setattr(preflight, "check_git", lambda: None)

    with pytest.raises(SystemExit) as exc:
        release.gate_before_bump(preflight, "1.5.19")
    assert "7 days since the last release" in str(exc.value)

    release.gate_before_bump(preflight, "1.5.19", early_release="owner approved in chat")
