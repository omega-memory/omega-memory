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


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_verifier():
    """Import the verifier script by path; scripts/ is not an importable package."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_omega_core_verifier", _VERIFIER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
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
    """A complete, valid Core wheel.

    release.py now runs the full artifact scan, which checks distribution
    metadata as well as the namespace and dependency rules these tests target.
    A stub METADATA block would fail on the metadata rules and obscure what is
    actually under test, so the fixture emits the real shape.
    """
    classifier_lines = "".join(f"Classifier: {classifier}\n" for classifier in _CORE_CLASSIFIERS)
    metadata = (
        "Metadata-Version: 2.4\n"
        "Name: omega-memory\n"
        "Version: 9.9.9\n"
        "License-Expression: Apache-2.0\n"
        f"{classifier_lines}"
        "Requires-Python: >=3.11\n"
    )
    if dependency:
        metadata += f"Requires-Dist: {dependency}\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("omega/__init__.py", "")
        archive.writestr("omega/server/mcp_server.py", "import omega_platform  # optional integration\n")
        archive.writestr("omega_memory-9.9.9.dist-info/METADATA", metadata)
        archive.writestr(
            "omega_memory-9.9.9.dist-info/entry_points.txt",
            "[console_scripts]\nomega = omega.cli:main\n",
        )
        archive.writestr("omega_memory-9.9.9.dist-info/WHEEL", "Wheel-Version: 1.0\n")
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
        ("omega/private.py", "/Users/maintainer/.omega/omega.db"),
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
        ("omega/settings.yaml", "data_path: /Users/maintainer/.omega/omega.db"),
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
    assert result.stdout.startswith("OK: ")


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
    assert "not a recognizable Core wheel" in result.stderr


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


def test_no_source_file_embeds_the_current_users_home_path():
    """No shipped file may contain the path of whoever is running the tests.

    Real home paths reach the sdist when a developer pastes one into a fixture
    or docstring. Two of them shipped this way, disclosing a macOS username.
    Deriving the name at runtime keeps this check honest without hardcoding
    anyone's identity here.
    """
    home = Path.home()
    username = home.name
    if not username or len(username) < 3:
        pytest.skip("home directory name too short to match reliably")

    needles = (str(home), f"/Users/{username}", f"/home/{username}")
    offenders = []
    for path in list(_REPO_ROOT.glob("src/**/*.py")) + list(_REPO_ROOT.glob("tests/**/*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for needle in needles:
            if needle in text:
                offenders.append(f"{path.relative_to(_REPO_ROOT)}: {needle}")

    assert not offenders, "real home path embedded in shipped files:\n  " + "\n  ".join(offenders)


# ---------------------------------------------------------------------------
# sdist verification
#
# The wheel was the only artifact ever scanned for content, so the sdist -- the
# wider one, carrying tests and docs -- went unchecked. Every release from
# 1.5.11 to 1.5.15 shipped a `.git` file holding an absolute home path, because
# releases are cut from git worktrees where `.git` is a file, not a directory.
# ---------------------------------------------------------------------------


# Assembled at runtime. A literal look-alike home path here would itself ship
# in the sdist -- exactly what these tests exist to detect.
_FAKE_ACCOUNT = "real" + "person"


def _write_core_sdist(path: Path, members: dict[str, str], version: str = "1.5.13") -> None:
    import io
    import tarfile

    root = f"omega_memory-{version}"
    with tarfile.open(path, "w:gz") as archive:
        for name, content in members.items():
            payload = content.encode("utf-8")
            info = tarfile.TarInfo(f"{root}/{name}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def test_sdist_verifier_accepts_a_clean_source_distribution(tmp_path):
    verify_core_sdist = _load_verifier().verify_core_sdist

    sdist = tmp_path / "omega_memory-1.5.13.tar.gz"
    _write_core_sdist(sdist, {
        "PKG-INFO": "Name: omega-memory\n",
        "src/omega/__init__.py": '__version__ = "1.5.13"\n',
        "docs/guide.md": "Run it from /Users/me/Projects/app\n",
        "tests/test_thing.py": 'api_key = "test-secret-value-not-real"\n',
    })

    assert verify_core_sdist(sdist) == []


def test_sdist_verifier_rejects_vcs_metadata_holding_a_home_path(tmp_path):
    verify_core_sdist = _load_verifier().verify_core_sdist

    sdist = tmp_path / "omega_memory-1.5.13.tar.gz"
    _write_core_sdist(sdist, {
        ".git": f"gitdir: /Users/{_FAKE_ACCOUNT}/Projects/omega-public/.git/worktrees/wt\n",
    })

    violations = verify_core_sdist(sdist)

    assert any(_FAKE_ACCOUNT in v for v in violations), violations


def test_sdist_verifier_rejects_a_real_home_path_inside_tests(tmp_path):
    verify_core_sdist = _load_verifier().verify_core_sdist

    sdist = tmp_path / "omega_memory-1.5.13.tar.gz"
    _write_core_sdist(sdist, {
        "tests/test_fixture.py": f'DB = "/Users/{_FAKE_ACCOUNT}/.omega/omega.db"\n',
    })

    violations = verify_core_sdist(sdist)

    assert any(_FAKE_ACCOUNT in v for v in violations), violations


def test_sdist_verifier_allows_documented_placeholder_homes(tmp_path):
    verify_core_sdist = _load_verifier().verify_core_sdist

    sdist = tmp_path / "omega_memory-1.5.13.tar.gz"
    _write_core_sdist(sdist, {
        "docs/a.md": "/Users/me/x /Users/test/y /home/dev/z /Users/maintainer/w\n",
    })

    assert verify_core_sdist(sdist) == []


def test_verifier_is_not_pinned_to_one_version(tmp_path):
    """A pinned version made this verifier reject every release after 1.5.13."""
    verify_core_sdist = _load_verifier().verify_core_sdist

    sdist = tmp_path / "omega_memory-9.9.9.tar.gz"
    _write_core_sdist(sdist, {"PKG-INFO": "Name: omega-memory\n"}, version="9.9.9")

    assert verify_core_sdist(sdist) == []
