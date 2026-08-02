"""Release-time guards for the public Core/Pro package boundary."""

from __future__ import annotations

import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path

import pytest


_RELEASE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "release.py"
_SPEC = importlib.util.spec_from_file_location("omega_release", _RELEASE_PATH)
assert _SPEC and _SPEC.loader
release = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(release)


def _write_wheel(path: Path, *, extra_members: dict[str, str] | None = None, dependency: str | None = None) -> None:
    metadata = "Metadata-Version: 2.4\nName: omega-memory\nVersion: 9.9.9\n"
    if dependency:
        metadata += f"Requires-Dist: {dependency}\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("omega/__init__.py", "")
        archive.writestr("omega/server/mcp_server.py", "import omega_platform  # optional integration\n")
        archive.writestr("omega_memory-9.9.9.dist-info/METADATA", metadata)
        for name, content in (extra_members or {}).items():
            archive.writestr(name, content)


def _write_sdist(path: Path, *, extra_members: dict[str, str] | None = None, dependency: str | None = None) -> None:
    metadata = "Metadata-Version: 2.4\nName: omega-memory\nVersion: 9.9.9\n"
    if dependency:
        metadata += f"Requires-Dist: {dependency}\n"
    members = {
        "omega_memory-9.9.9/PKG-INFO": metadata,
        "omega_memory-9.9.9/src/omega/__init__.py": "",
        **(extra_members or {}),
    }
    with tarfile.open(path, "w:gz") as archive:
        for name, content in members.items():
            data = content.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def _artifacts(tmp_path: Path, **kwargs) -> tuple[Path, Path]:
    wheel = tmp_path / "omega_memory-9.9.9-py3-none-any.whl"
    sdist = tmp_path / "omega_memory-9.9.9.tar.gz"
    _write_wheel(wheel, extra_members=kwargs.get("wheel_members"), dependency=kwargs.get("wheel_dependency"))
    _write_sdist(sdist, extra_members=kwargs.get("sdist_members"), dependency=kwargs.get("sdist_dependency"))
    return wheel, sdist


def test_boundary_accepts_core_with_optional_pro_import_references(tmp_path):
    wheel, sdist = _artifacts(tmp_path)

    release.verify_public_artifact_boundary(wheel, sdist)


def test_boundary_rejects_private_namespace_in_wheel(tmp_path):
    wheel, sdist = _artifacts(tmp_path, wheel_members={"omega_platform/license.py": ""})

    with pytest.raises(SystemExit, match=r"wheel member: omega_platform/license\.py"):
        release.verify_public_artifact_boundary(wheel, sdist)


def test_boundary_rejects_bundled_wheel_in_sdist(tmp_path):
    wheel, sdist = _artifacts(
        tmp_path,
        sdist_members={"omega_memory-9.9.9/vendor/omega_memory_pro-9.9.9-py3-none-any.whl": "private"},
    )

    with pytest.raises(SystemExit, match=r"sdist member: .*omega_memory_pro.*\.whl"):
        release.verify_public_artifact_boundary(wheel, sdist)


@pytest.mark.parametrize("artifact", ["wheel", "sdist"])
def test_boundary_rejects_private_distribution_dependency(tmp_path, artifact):
    wheel, sdist = _artifacts(tmp_path, **{f"{artifact}_dependency": "omega-platform>=1.6"})

    with pytest.raises(SystemExit, match=rf"{artifact} dependency: Requires-Dist: omega-platform"):
        release.verify_public_artifact_boundary(wheel, sdist)
