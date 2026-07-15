from __future__ import annotations

import os
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
pytestmark = pytest.mark.packaging


def _artifact_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    configured = os.environ.get("ENVELOPE_DIST_DIR")
    if configured is not None:
        output = Path(configured)
        return output if output.is_absolute() else ROOT / output

    output = tmp_path / "dist"
    monkeypatch.setenv("UV_PYTHON", "3.12")
    subprocess.run(
        ["uv", "build", "--no-sources", "--out-dir", str(output)],
        cwd=ROOT,
        check=True,
    )
    return output


def test_built_artifacts_respect_the_release_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = _artifact_directory(tmp_path, monkeypatch)

    sdists = list(output.glob("*.tar.gz"))
    wheels = list(output.glob("*.whl"))
    assert len(sdists) == 1, f"expected one sdist in {output}, found {sdists}"
    assert len(wheels) == 1, f"expected one wheel in {output}, found {wheels}"
    sdist = sdists[0]
    wheel = wheels[0]

    with tarfile.open(sdist, "r:gz") as archive:
        relative_members = {
            member.name.split("/", 1)[1]
            for member in archive.getmembers()
            if "/" in member.name
        }

    allowed_directories = {"src", "tests", "docs", ".github"}
    allowed_files = {
        ".gitignore",
        ".readthedocs.yaml",
        "LICENSE",
        "README.md",
        "PKG-INFO",
        "mkdocs.yml",
        "pyproject.toml",
    }
    unexpected = {
        name
        for name in relative_members
        if name.split("/", 1)[0] not in allowed_directories
        and name not in allowed_files
    }
    assert not unexpected
    assert not any(name.startswith("examples/ppo") for name in relative_members)
    assert not any(name.startswith("tests/ppo") for name in relative_members)
    assert not any(
        ".claude" in name or "benchmarks" in name for name in relative_members
    )

    with zipfile.ZipFile(wheel) as archive:
        wheel_members = set(archive.namelist())

    assert "envelope/py.typed" in wheel_members
    assert not any(name.startswith("examples/") for name in wheel_members)
    assert sdist.stat().st_size < 2_000_000
    assert wheel.stat().st_size < 1_000_000
