"""Integration tests for jenv.compat.create().

These require optional compatibility dependencies (brax/gymnax/navix). They are
kept separate from the unit tests so a minimal install can still run the suite.
"""

import pytest

from jenv.compat import create
from jenv.environment import Environment
from jenv.wrappers.truncation_wrapper import TruncationWrapper

pytestmark = pytest.mark.compat


def test_create_brax_smoke(prng_key):
    pytest.importorskip("brax")

    from jenv.compat.brax_jenv import BraxJenv

    env = create("brax::fast")
    assert isinstance(env, TruncationWrapper)
    assert isinstance(env.env, BraxJenv)
    assert isinstance(env, Environment)
    assert env.max_steps == 1000  # Brax default

    _state, info = env.reset(prng_key)
    assert hasattr(info, "obs")


def test_create_gymnax_smoke(prng_key):
    pytest.importorskip("gymnax")

    from jenv.compat.gymnax_jenv import GymnaxJenv

    env = create("gymnax::CartPole-v1")
    assert isinstance(env, TruncationWrapper)
    assert isinstance(env.env, GymnaxJenv)
    assert isinstance(env, Environment)
    assert env.max_steps == 500  # CartPole-v1 default

    _state, info = env.reset(prng_key)
    assert hasattr(info, "obs")


def test_create_navix_smoke(prng_key):
    pytest.importorskip("navix")

    from jenv.compat.navix_jenv import NavixJenv

    env = create("navix::Navix-Empty-5x5-v0")
    assert isinstance(env, TruncationWrapper)
    assert isinstance(env.env, NavixJenv)
    assert isinstance(env, Environment)
    assert env.max_steps == 100  # Navix default

    _state, info = env.reset(prng_key)
    assert hasattr(info, "obs")


def test_create_jumanji_smoke(prng_key):
    pytest.importorskip("jumanji")

    from jenv.compat.jumanji_jenv import JumanjiJenv

    env = create("jumanji::Snake-v1")
    assert isinstance(env, TruncationWrapper)
    assert isinstance(env.env, JumanjiJenv)
    assert isinstance(env, Environment)
    assert env.max_steps == 4000  # Snake-v1 default

    _state, info = env.reset(prng_key)
    assert hasattr(info, "obs")


def test_create_craftax_smoke(prng_key):
    pytest.importorskip("craftax")

    from jenv.compat.craftax_jenv import CraftaxJenv

    env = create("craftax::Craftax-Symbolic-v1")
    assert isinstance(env, TruncationWrapper)
    assert isinstance(env.env, CraftaxJenv)
    assert isinstance(env, Environment)
    assert env.max_steps == 100000  # Craftax default

    _state, info = env.reset(prng_key)
    assert hasattr(info, "obs")


def test_create_mujoco_playground_smoke(prng_key):
    pytest.importorskip("mujoco_playground")

    from jenv.compat.mujoco_playground_jenv import MujocoPlaygroundJenv

    env = create("mujoco_playground::CartpoleBalance")
    assert isinstance(env, TruncationWrapper)
    assert isinstance(env.env, MujocoPlaygroundJenv)
    assert isinstance(env, Environment)
    assert env.max_steps == 1000  # CartpoleBalance default

    _state, info = env.reset(prng_key)
    assert hasattr(info, "obs")


def test_create_kinetix_smoke(prng_key):
    pytest.importorskip("kinetix")

    from jenv.compat.kinetix_jenv import KinetixJenv

    env = create("kinetix::random")
    assert isinstance(env, TruncationWrapper)
    assert isinstance(env.env, KinetixJenv)
    assert isinstance(env, Environment)
    assert env.max_steps == 256  # Kinetix default

    _state, info = env.reset(prng_key)
    assert hasattr(info, "obs")


def test_create_rejects_max_steps_override():
    """Test that create() raises ValueError when trying to override max_steps."""
    pytest.importorskip("gymnax")

    with pytest.raises(ValueError, match="Cannot override"):
        create("gymnax::CartPole-v1", env_kwargs={"max_steps_in_episode": 100})
