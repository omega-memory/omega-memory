"""Release-time guards for the public Core/Pro package boundary."""

from __future__ import annotations

import importlib.util
import io
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest


_RELEASE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "release.py"
_SPEC = importlib.util.spec_from_file_location("omega_release", _RELEASE_PATH)
assert _SPEC and _SPEC.loader
release = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(release)

_VERIFIER = Path(__file__).resolve().parents[1] / "scripts" / "verify_core_release_artifact.py"
_CORE_CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Software Development :: Libraries",
]


def _write_exact_core_wheel(
    path: Path,
    *,
    extra_members: dict[str, str | bytes] | None = None,
    name: str = "omega-memory",
    version: str = "1.5.13",
    license_expression: str = "Apache-2.0",
    requires_python: str = ">=3.11",
    classifiers: list[str] | None = None,
    entry_points: str = "[console_scripts]\nomega = omega.cli:main\n",
) -> None:
    classifier_lines = "".join(
        f"Classifier: {classifier}\n" for classifier in (classifiers or _CORE_CLASSIFIERS)
    )
    metadata = (
        "Metadata-Version: 2.4\n"
        f"Name: {name}\n"
        f"Version: {version}\n"
        f"License-Expression: {license_expression}\n"
        f"{classifier_lines}"
        f"Requires-Python: {requires_python}\n"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("omega/__init__.py", '__version__ = "1.5.13"\n')
        archive.writestr("omega/pure.py", "raise RuntimeError('must not execute')\n")
        archive.writestr("omega_memory-1.5.13.dist-info/METADATA", metadata)
        archive.writestr(
            "omega_memory-1.5.13.dist-info/entry_points.txt",
            entry_points,
        )
        archive.writestr("omega_memory-1.5.13.dist-info/WHEEL", "Wheel-Version: 1.0\n")
        for member, content in (extra_members or {}).items():
            archive.writestr(member, content)


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


@pytest.mark.parametrize(
    ("member", "content"),
    [
        (".omega/omega.db", ""),
        ("omega_platform/license.py", ""),
        ("synaptic/private.py", ""),
        ("omega/.env", ""),
        ("omega/cache.sqlite", ""),
        ("logs/hooks.log", ""),
        ("results/audit.json", ""),
        ("omega/private.py", "/Users/singularityjason/.omega/omega.db"),
        ("omega/private.py", "/Users/another-user/.omega/omega.db"),
        ("omega/private.py", "/home/private-user/.omega/omega.db"),
        ("omega/private.py", r"C:\Users\private-user\.omega\omega.db"),
        ("omega/private.py", "/root/.omega/omega.db"),
        ("omega/private.py", "synaptic release configuration"),
        ("omega/private.py", 'customer_email = "private.person@paid.invalid"'),
        ("omega/private.py", "customer_email = private.person@paid.invalid"),
        ("omega/private.py", "customer_name: Private Person"),
        ("omega/private.py", "api_key = super-secret-value-123"),
        ("omega/token.txt", ""),
        ("omega/settings.yaml", "data_path: /Users/singularityjason/.omega/omega.db"),
        ("omega/settings.yaml", "api_key: super-secret-value-123"),
        ("omega/credentials.pem", "-----BEGIN PRIVATE KEY-----\nprivate material\n-----END PRIVATE KEY-----"),
    ],
)
def test_core_artifact_verifier_rejects_private_members_and_content(tmp_path, member, content):
    wheel = tmp_path / "omega_memory-1.5.13-py3-none-any.whl"
    _write_exact_core_wheel(wheel, extra_members={member: content})

    result = subprocess.run(
        [sys.executable, str(_VERIFIER), str(wheel)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Core artifact privacy violation" in result.stderr


def test_core_artifact_verifier_accepts_core_members_without_executing_them(tmp_path):
    wheel = tmp_path / "omega_memory-1.5.13-py3-none-any.whl"
    _write_exact_core_wheel(wheel)

    result = subprocess.run(
        [sys.executable, str(_VERIFIER), str(wheel)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "OK: inspected" in result.stdout


@pytest.mark.parametrize(
    "member",
    [
        "omega_memory_pro-1.5.10.dist-info/METADATA",
        "omega_memory-1.5.12.dist-info/RECORD",
        "unrelated-1.0.dist-info/METADATA",
    ],
)
def test_core_artifact_verifier_rejects_every_other_distribution_root(tmp_path, member):
    wheel = tmp_path / "omega_memory-1.5.13-py3-none-any.whl"
    _write_exact_core_wheel(wheel, extra_members={member: "Metadata-Version: 2.4\n"})

    result = subprocess.run([sys.executable, str(_VERIFIER), str(wheel)], capture_output=True, text=True)

    assert result.returncode == 1
    assert member in result.stderr


@pytest.mark.parametrize(
    ("name", "version", "license_expression", "requires_python"),
    [
        ("omega-memory-pro", "1.5.13", "Apache-2.0", ">=3.11"),
        ("omega-memory", "1.5.12", "Apache-2.0", ">=3.11"),
        ("omega-memory", "1.5.13", "LicenseRef-Proprietary", ">=3.11"),
        ("omega-memory", "1.5.13", "Apache-2.0", ">=3.12"),
    ],
)
def test_core_artifact_verifier_requires_exact_metadata(
    tmp_path, name, version, license_expression, requires_python
):
    wheel = tmp_path / "omega_memory-1.5.13-py3-none-any.whl"
    _write_exact_core_wheel(
        wheel,
        name=name,
        version=version,
        license_expression=license_expression,
        requires_python=requires_python,
    )

    result = subprocess.run([sys.executable, str(_VERIFIER), str(wheel)], capture_output=True, text=True)

    assert result.returncode == 1
    assert "Core artifact privacy violation" in result.stderr


def test_core_artifact_verifier_requires_exact_candidate_filename(tmp_path):
    wheel = tmp_path / "omega_memory-1.5.13-py3-none-macosx_11_0_arm64.whl"
    _write_exact_core_wheel(wheel)

    result = subprocess.run([sys.executable, str(_VERIFIER), str(wheel)], capture_output=True, text=True)

    assert result.returncode == 1
    assert "release candidate" in result.stderr


@pytest.mark.parametrize(
    "classifiers",
    [
        _CORE_CLASSIFIERS + ["License :: Other/Proprietary License"],
        [item for item in _CORE_CLASSIFIERS if not item.startswith("License ::")],
        _CORE_CLASSIFIERS + ["Topic :: Database :: Database Engines/Servers"],
    ],
)
def test_core_artifact_verifier_requires_the_exact_classifier_set(tmp_path, classifiers):
    wheel = tmp_path / "omega_memory-1.5.13-py3-none-any.whl"
    _write_exact_core_wheel(wheel, classifiers=classifiers)

    result = subprocess.run([sys.executable, str(_VERIFIER), str(wheel)], capture_output=True, text=True)

    assert result.returncode == 1
    assert "classifier" in result.stderr.lower()


@pytest.mark.parametrize(
    "entry_points",
    [
        (
            "[console_scripts]\nomega = omega.cli:main\n"
            "[omega.plugins]\nomega_pro = omega_platform.plugin:OmegaPlatformPlugin\n"
        ),
        "[omega.plugins]\nomega_pro = omega_platform.plugin:OmegaPlatformPlugin\n",
        "[console_scripts]\nomega = omega_platform.cli:main\n",
    ],
)
def test_core_artifact_verifier_rejects_plugin_and_unexpected_entry_points(tmp_path, entry_points):
    wheel = tmp_path / "omega_memory-1.5.13-py3-none-any.whl"
    _write_exact_core_wheel(wheel, entry_points=entry_points)

    result = subprocess.run([sys.executable, str(_VERIFIER), str(wheel)], capture_output=True, text=True)

    assert result.returncode == 1
    assert "entry" in result.stderr.lower()
