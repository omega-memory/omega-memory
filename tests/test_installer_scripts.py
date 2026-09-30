"""The installer build helpers: the macOS payload check and the PyPI wait.

Up to v1.5.19 the macOS pkg said it ran on Intel and on macOS 12 while
carrying only an Apple Silicon Python whose packages needed macOS 14 and 15,
and the installer workflows failed whenever they started before PyPI listed
the new release. check_payload.py and wait_for_pypi.py guard those two; these
tests pin how they decide.
"""
import importlib.util
import struct
import subprocess
import sys
from pathlib import Path

import pytest

INSTALLER = Path(__file__).resolve().parent.parent / "installer"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses look their module up in sys.modules while it executes.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


check_payload = _load("check_payload", INSTALLER / "macos" / "check_payload.py")
wait_for_pypi = _load("wait_for_pypi", INSTALLER / "wait_for_pypi.py")

ARM64 = 0x0100000C
X86_64 = 0x01000007


def _thin(cputype: int, minos: tuple[int, int] | None, *, legacy: bool = False) -> bytes:
    """A minimal 64-bit Mach-O: header plus one version load command."""
    commands = b""
    if minos is not None:
        encoded = (minos[0] << 16) | (minos[1] << 8)
        if legacy:
            commands = struct.pack("<IIII", check_payload.LC_VERSION_MIN_MACOSX, 16, encoded, encoded)
        else:
            commands = struct.pack("<IIIIII", check_payload.LC_BUILD_VERSION, 24,
                                   check_payload.PLATFORM_MACOS, encoded, encoded, 0)
    ncmds = 1 if commands else 0
    header = struct.pack("<IiiIIIII", check_payload.MH_MAGIC_64, cputype, 0, 6, ncmds, len(commands), 0, 0)
    return header + commands


def _fat(*slices: bytes) -> bytes:
    """A fat (universal) file holding the given thin slices."""
    header = struct.pack(">II", check_payload.FAT_MAGIC, len(slices))
    offset = 8 + 20 * len(slices)
    entries, bodies = b"", b""
    for body in slices:
        cputype = struct.unpack_from("<i", body, 4)[0]
        entries += struct.pack(">iiIII", cputype, 0, offset + len(bodies), len(body), 0)
        bodies += body
    return header + entries + bodies


def test_reads_arch_and_macos_floor_from_thin_and_fat_files(tmp_path):
    thin = tmp_path / "thin.so"
    thin.write_bytes(_thin(ARM64, (14, 0)))
    legacy = tmp_path / "legacy.dylib"
    legacy.write_bytes(_thin(X86_64, (10, 9), legacy=True))
    fat = tmp_path / "universal.so"
    fat.write_bytes(_fat(_thin(X86_64, (10, 15)), _thin(ARM64, (11, 0))))
    text = tmp_path / "module.py"
    text.write_text("print('not a binary')\n")

    assert check_payload.read_slices(thin) == [check_payload.Slice("arm64", (14, 0))]
    assert check_payload.read_slices(legacy) == [check_payload.Slice("x86_64", (10, 9))]
    assert check_payload.read_slices(fat) == [
        check_payload.Slice("x86_64", (10, 15)),
        check_payload.Slice("arm64", (11, 0)),
    ]
    assert check_payload.read_slices(text) == []


def test_flags_a_binary_without_the_arch_and_one_needing_newer_macos(tmp_path):
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "ok.so").write_bytes(_fat(_thin(X86_64, (10, 15)), _thin(ARM64, (11, 0))))
    (tmp_path / "lib" / "arm_only.so").write_bytes(_thin(ARM64, (11, 0)))
    (tmp_path / "lib" / "too_new.dylib").write_bytes(_thin(X86_64, (15, 0)))

    checked, problems = check_payload.check_payload(tmp_path, "x86_64", (13, 4))

    assert checked == 3
    assert problems == [
        "lib/arm_only.so: no x86_64 code (has arm64)",
        "lib/too_new.dylib: needs macOS 15.0",
    ]


def test_command_line_fails_on_problems_and_on_an_empty_payload(tmp_path, capsys):
    (tmp_path / "vec0.dylib").write_bytes(_thin(X86_64, (15, 0)))
    assert check_payload.main(["check_payload.py", str(tmp_path), "x86_64", "15.0"]) == 0
    assert check_payload.main(["check_payload.py", str(tmp_path), "x86_64", "13.0"]) == 1
    assert "vec0.dylib: needs macOS 15.0" in capsys.readouterr().err

    empty = tmp_path / "empty"
    empty.mkdir()
    assert check_payload.main(["check_payload.py", str(empty), "arm64", "14.0"]) == 1


def test_pip_query_skips_the_cache_and_targets_the_bundled_python(monkeypatch):
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        return subprocess.CompletedProcess(command, 1, "", "ERROR: No matching distribution found\n")

    monkeypatch.setattr(wait_for_pypi.subprocess, "run", fake_run)

    assert wait_for_pypi.pip_can_fetch("1.5.20") == (False, "ERROR: No matching distribution found")
    command = seen["command"]
    assert "--no-cache-dir" in command
    assert command[command.index("--python-version") + 1] == "3.12"
    assert command[-1] == "omega-memory==1.5.20"


class _Clock:
    """A fake monotonic clock that sleep() advances."""

    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = _Clock()
    monkeypatch.setattr(wait_for_pypi.time, "monotonic", fake.monotonic)
    monkeypatch.setattr(wait_for_pypi.time, "sleep", fake.sleep)
    return fake


def test_wait_retries_until_the_release_appears(monkeypatch, clock, capsys):
    answers = iter([(False, "not yet"), (False, "not yet"), (True, "")])
    monkeypatch.setattr(wait_for_pypi, "pip_can_fetch", lambda version: next(answers))
    monkeypatch.setattr("sys.argv", ["wait_for_pypi.py", "1.5.20", "--interval", "30"])

    assert wait_for_pypi.main() == 0
    assert clock.sleeps == [30, 30]
    assert "available from PyPI (attempt 3)" in capsys.readouterr().out


def test_wait_gives_up_at_the_timeout(monkeypatch, clock, capsys):
    monkeypatch.setattr(wait_for_pypi, "pip_can_fetch", lambda version: (False, "No matching distribution"))
    monkeypatch.setattr("sys.argv", ["wait_for_pypi.py", "1.5.20", "--timeout", "70", "--interval", "30"])

    assert wait_for_pypi.main() == 1
    assert clock.sleeps == [30, 30, 10]
    assert "after 70s (4 attempts)" in capsys.readouterr().err
