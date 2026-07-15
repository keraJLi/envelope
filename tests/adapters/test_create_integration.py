"""Integration tests for envelope.adapters.create().

These require optional adapter dependencies (brax/gymnax/navix). They are kept separate
from the unit tests so a minimal install can still run the suite.
"""

import pytest

from envelope.adapters import create
from envelope.environment import Environment
from envelope.wrappers.truncation_wrapper import TruncationWrapper

pytestmark = pytest.mark.adapters


def _run_init_and_step(env: Environment, key) -> None:
    state, info = env.init(key)
    assert hasattr(info, "obs")
    action = env.action_space.sample(key)
    _state, info = env.step(state, action)
    assert hasattr(info, "obs")


def test_create_brax_smoke(prng_key):
    from envelope.adapters.brax_envelope import BraxEnvelope

    env = create("brax::fast")
    assert isinstance(env, TruncationWrapper)
    assert isinstance(env.env, BraxEnvelope)
    assert isinstance(env, Environment)
    assert env.max_steps == 1000  # Brax default

    _run_init_and_step(env, prng_key)


def test_create_gymnax_smoke(prng_key):
    from envelope.adapters.gymnax_envelope import GymnaxEnvelope

    env = create("gymnax::CartPole-v1")
    assert isinstance(env, TruncationWrapper)
    assert isinstance(env.env, GymnaxEnvelope)
    assert isinstance(env, Environment)
    assert env.max_steps == 500  # CartPole-v1 default

    _run_init_and_step(env, prng_key)


def test_create_navix_smoke(prng_key):
    from envelope.adapters.navix_envelope import NavixEnvelope

    env = create("navix::Navix-Empty-5x5-v0")
    assert isinstance(env, TruncationWrapper)
    assert isinstance(env.env, NavixEnvelope)
    assert isinstance(env, Environment)
    assert env.max_steps == 100  # Navix default

    _run_init_and_step(env, prng_key)


def test_create_jumanji_smoke(prng_key):
    from envelope.adapters.jumanji_envelope import JumanjiEnvelope

    env = create("jumanji::Snake-v1")
    assert isinstance(env, TruncationWrapper)
    assert isinstance(env.env, JumanjiEnvelope)
    assert isinstance(env, Environment)
    assert env.max_steps == 4000  # Snake-v1 default

    _run_init_and_step(env, prng_key)


def test_create_craftax_smoke(prng_key):
    from envelope.adapters.craftax_envelope import CraftaxEnvelope

    env = create("craftax::Craftax-Symbolic-v1")
    assert isinstance(env, TruncationWrapper)
    assert isinstance(env.env, CraftaxEnvelope)
    assert isinstance(env, Environment)
    assert env.max_steps == 100000  # Craftax default

    _run_init_and_step(env, prng_key)


def test_create_mujoco_playground_smoke(prng_key):
    from envelope.adapters.mujoco_playground_envelope import MujocoPlaygroundEnvelope

    env = create("mujoco_playground::CartpoleBalance")
    assert isinstance(env, TruncationWrapper)
    assert isinstance(env.env, MujocoPlaygroundEnvelope)
    assert isinstance(env, Environment)
    assert env.max_steps == 1000  # CartpoleBalance default

    _run_init_and_step(env, prng_key)


def test_create_kinetix_smoke(prng_key):
    from envelope.adapters.kinetix_envelope import KinetixEnvelope

    env = create("kinetix::random")
    assert isinstance(env, TruncationWrapper)
    assert isinstance(env.env, KinetixEnvelope)
    assert isinstance(env, Environment)
    assert env.max_steps == 256  # Kinetix default

    _run_init_and_step(env, prng_key)


def test_create_rejects_max_steps_override():
    """Test that create() raises ValueError when trying to override max_steps."""
    with pytest.raises(ValueError, match="Cannot override"):
        create("gymnax::CartPole-v1", env_kwargs={"max_steps_in_episode": 100})
