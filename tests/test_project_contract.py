from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).parents[1]


def _project() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as file:
        return tomllib.load(file)


def test_release_metadata_describes_envelope_0_5() -> None:
    project = _project()

    assert project["project"]["version"] == "0.5.0"
    assert (ROOT / "src/envelope/py.typed").is_file()


def test_publishable_adapter_extras_are_explicit_and_bounded() -> None:
    extras = _project()["project"]["optional-dependencies"]

    publishable = {
        "brax",
        "craftax",
        "jumanji",
        "mujoco-playground",
        "navix",
    }
    assert publishable <= extras.keys()
    assert "adapters" in extras
    assert "gymnax" not in extras
    assert "kinetix" not in extras
    for name in publishable:
        assert extras[name]
        assert all(">=" in requirement and "<" in requirement for requirement in extras[name])


def test_source_backed_adapter_overrides_are_immutable() -> None:
    sources = _project()["tool"]["uv"]["sources"]

    assert sources["gymnax"]["rev"] == "18f2e7f3cffafc7042c76fdc538c83957418a9a9"
    assert sources["kinetix-env"]["rev"] == "df4de60cabd42dbd1c35fb5214fdc6728710e33d"


def test_ppo_is_a_repository_only_dependency_group() -> None:
    groups = _project()["dependency-groups"]

    assert {"flax", "optax", "distrax", "tyro"} <= {
        requirement.split("<", 1)[0].split(">", 1)[0].split("=", 1)[0]
        for requirement in groups["ppo"]
    }
    assert any(requirement.startswith("wandb") for requirement in groups["ppo-wandb"])


def test_distribution_boundaries_exclude_experimental_ppo() -> None:
    project = _project()
    hatch = project["tool"]["hatch"]["build"]
    sdist = hatch["targets"]["sdist"]

    assert "/examples/ppo" in sdist["exclude"]
    assert "/tests/ppo" in sdist["exclude"]
    assert "/examples/ppo" not in sdist["include"]
    assert not project["tool"].get("hatch", {}).get("metadata", {}).get(
        "allow-direct-references", False
    )


def test_docs_and_readme_links_have_release_safe_bounds() -> None:
    requirements = (ROOT / "docs/requirements.txt").read_text().splitlines()
    mkdocs = next(line for line in requirements if line.startswith("mkdocs>="))
    assert ",<2" in mkdocs

    readme = (ROOT / "README.md").read_text()
    assert "https://jax-envelope.readthedocs.io/" in readme
    assert "](docs/" not in readme


def test_ci_and_tag_release_workflows_exist() -> None:
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    adapters = (ROOT / ".github/workflows/adapters.yml").read_text()
    publish = (ROOT / ".github/workflows/publish.yml").read_text()

    assert "pyright" in ci.lower()
    assert "ruff format --check" in ci
    assert 'python-version: ["3.12", "3.13"]' in ci
    assert "fail-fast: false" in adapters
    assert "importorskip" not in adapters
    assert "tags:" in publish and "v*" in publish
    assert "types: [published, edited]" not in publish
    assert "schedule:" in adapters and "workflow_dispatch:" in adapters

    for workflow in (ci, publish):
        assert workflow.count("uv build --no-sources") == 1
        assert workflow.index("uv build --no-sources") < workflow.index(
            "ENVELOPE_DIST_DIR=dist"
        )

    assert "create_test:" in adapters
    assert "pytest ${{ matrix.create_test }}" in adapters
    create_integration = (ROOT / "tests/adapters/test_create_integration.py").read_text()
    assert "importorskip" not in create_integration
