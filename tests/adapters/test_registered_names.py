"""Adapter-local environment name enumeration tests."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.adapters


def test_brax_registered_names_match_upstream_registry():
    brax_envs = pytest.importorskip("brax.envs")
    from envelope.adapters.brax_envelope import BraxEnvelope

    assert BraxEnvelope.registered_names() == tuple(brax_envs._envs)


def test_gymnax_registered_names_match_upstream_registry():
    gymnax = pytest.importorskip("gymnax")
    from envelope.adapters.gymnax_envelope import GymnaxEnvelope

    assert GymnaxEnvelope.registered_names() == tuple(gymnax.registered_envs)


def test_jumanji_registered_names_match_upstream_registry():
    jumanji = pytest.importorskip("jumanji")
    from envelope.adapters.jumanji_envelope import JumanjiEnvelope

    assert set(JumanjiEnvelope.registered_names()) == jumanji.registered_environments()


def test_navix_registered_names_match_upstream_registry():
    navix = pytest.importorskip("navix")
    from envelope.adapters.navix_envelope import NavixEnvelope

    assert NavixEnvelope.registered_names() == tuple(navix.registry())


def test_mujoco_playground_registered_names_match_upstream_registry():
    mujoco_playground = pytest.importorskip("mujoco_playground")
    from envelope.adapters.mujoco_playground_envelope import MujocoPlaygroundEnvelope

    assert MujocoPlaygroundEnvelope.registered_names() == tuple(
        mujoco_playground.registry.ALL_ENVS
    )


def test_craftax_registered_names_are_canonical_ids():
    pytest.importorskip("craftax")
    from envelope.adapters.craftax_envelope import CraftaxEnvelope

    assert CraftaxEnvelope.registered_names() == (
        "Craftax-Symbolic-v1",
        "Craftax-Pixels-v1",
        "Craftax-Classic-Symbolic-v1",
        "Craftax-Classic-Pixels-v1",
    )


def test_kinetix_registered_names_include_modes_and_packaged_levels():
    saving = pytest.importorskip("kinetix.util.saving")
    from envelope.adapters.kinetix_envelope import KinetixEnvelope

    base_dir = Path(saving.BASE_DIR)
    packaged_levels = {
        path.relative_to(base_dir).with_suffix("").as_posix()
        for path in base_dir.glob("*/*.json")
    }
    names = set(KinetixEnvelope.registered_names())

    assert names == {"s", "m", "l", "random", *packaged_levels}
    assert all(not name.endswith(".json") for name in names)
