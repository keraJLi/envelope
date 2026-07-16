"""Tests for envelope.adapters.navix_envelope module."""

# ruff: noqa: E402

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.adapters

pytest.importorskip("navix")

import navix
from navix.entities import EntityIds

from envelope.adapters.navix_envelope import NavixEnvelope
from envelope.spaces import Continuous, Discrete
from tests.contract import (
    assert_jitted_rollout_contract,
    assert_obs_matches_space,
    assert_reset_step_contract,
)


def _create_navix_env(env_name: str = "Navix-Empty-5x5-v0", **kwargs):
    """Helper to create a NavixEnvelope wrapper."""
    navix_env = navix.make(env_name, **kwargs)
    return NavixEnvelope(navix_env=navix_env)


@pytest.fixture(scope="module")
def navix_env():
    return _create_navix_env()


@pytest.fixture(scope="module", autouse=True)
def _navix_env_warmup(navix_env, prng_key):
    env = navix_env
    key_reset, key_step = jax.random.split(prng_key)
    state, _info = env.init(key_reset)
    action = env.action_space.sample(key_step)
    env.step(state, action)


def test_navix_contract_smoke(prng_key, navix_env):
    assert_reset_step_contract(
        navix_env, key=prng_key, obs_check=assert_obs_matches_space
    )


def test_navix_contract_scan(prng_key, navix_env, scan_num_steps):
    assert_jitted_rollout_contract(navix_env, key=prng_key, num_steps=scan_num_steps)


def test_action_space_conversion(navix_env):
    """Test conversion of navix action spaces to envelope spaces."""
    env = navix_env

    # Check that action space is converted correctly
    assert isinstance(env.action_space, Discrete)
    assert env.action_space.n == env.navix_env.action_space.n
    assert env.action_space.shape == env.navix_env.action_space.shape
    assert env.action_space.dtype == env.navix_env.action_space.dtype


def test_observation_space_contains_real_reset_observation(navix_env, prng_key):
    """The converted cardinalities must include values emitted by Navix."""
    env = navix_env
    assert isinstance(env.observation_space, Discrete)

    _state, info = env.init(prng_key)

    assert env.observation_space.contains(info.obs)
    assert env.observation_space.shape == env.navix_env.observation_space.shape
    assert env.observation_space.dtype == env.navix_env.observation_space.dtype


def test_observation_entity_cardinality_comes_from_authoritative_ids(
    navix_env, prng_key
):
    _state, info = navix_env.init(prng_key)
    entity_cardinality = (
        max(int(value) for name, value in vars(EntityIds).items() if name.isupper()) + 1
    )

    assert jnp.all(navix_env.observation_space.n[..., 0] == entity_cardinality)
    assert jnp.all(info.obs[..., 0] < navix_env.observation_space.n[..., 0])


def test_container_conversion_reset(navix_env, prng_key):
    """Test convert_navix_to_envelope_info on reset timestep."""
    env = navix_env
    key = prng_key

    # Get the raw navix timestep
    navix_timestep = env.navix_env.reset(key)
    state, info = env.init(key)

    # Verify obs, reward, terminated, truncated fields
    assert jnp.array_equal(info.obs, navix_timestep.observation)
    assert info.reward == navix_timestep.reward
    assert info.terminated == (navix_timestep.step_type == navix.StepType.TERMINATION)
    assert info.truncated == (navix_timestep.step_type == navix.StepType.TRUNCATION)

    # Verify terminated=False and truncated=False on reset
    assert not info.terminated
    assert not info.truncated

    # Check that extra timestep fields are preserved at the top level.
    # (if navix timestep has extra fields beyond the standard ones)
    import dataclasses

    timestep_dict = dataclasses.asdict(navix_timestep)
    # Remove standard fields that are handled explicitly
    standard_fields = {"observation", "reward", "step_type"}
    extra_fields = {k: v for k, v in timestep_dict.items() if k not in standard_fields}
    for field_name, field_value in extra_fields.items():
        assert hasattr(info, field_name)


def test_container_conversion_step(navix_env, prng_key):
    """Test convert_navix_to_envelope_info on step timestep."""
    env = navix_env
    key = prng_key

    state, _ = env.init(key)
    action = env.action_space.sample(jax.random.fold_in(prng_key, 1))

    # Get the raw navix timestep
    navix_timestep = env.navix_env.step(state, action)
    next_state, info = env.step(state, action)

    # Verify all fields are correctly converted
    assert jnp.array_equal(info.obs, navix_timestep.observation)
    assert info.reward == navix_timestep.reward
    assert info.terminated == (navix_timestep.step_type == navix.StepType.TERMINATION)
    assert info.truncated == (navix_timestep.step_type == navix.StepType.TRUNCATION)

    # Verify reward values are preserved
    assert info.reward == navix_timestep.reward


def test_episode_truncation(prng_key):
    """Test that episode truncation is correctly detected."""
    from envelope.wrappers.truncation_wrapper import TruncationWrapper

    adapter = _create_navix_env(max_steps=5)
    assert adapter.default_max_steps == 5
    env = TruncationWrapper(env=adapter, max_steps=adapter.default_max_steps)
    key = prng_key

    state, info = env.init(key)

    # Step until truncation
    for _ in range(10):  # More than max_steps
        if info.truncated:
            break
        action = env.action_space.sample(jax.random.fold_in(prng_key, _))
        state, info = env.step(state, action)

    # Verify truncation occurred
    assert info.truncated
    assert not info.terminated


def test_explicit_max_steps_warns(monkeypatch):
    from envelope.adapters import navix_envelope

    class DummyEnv:
        max_steps = 17

    monkeypatch.setattr(
        navix_envelope.navix, "make", lambda *_args, **_kwargs: DummyEnv()
    )

    with pytest.warns(UserWarning, match="max_steps"):
        env = navix_envelope.NavixEnvelope.from_name(
            "Dummy-v0", env_kwargs={"max_steps": 17}
        )

    assert env.default_max_steps == 17


def test_unsupported_space_type():
    """Test that unsupported space types raise ValueError."""
    import jax.numpy as jnp
    from navix import spaces as navix_spaces

    from envelope.adapters.navix_envelope import convert_navix_to_envelope_space

    # Create a mock space that's neither Discrete nor Continuous
    class MockSpace(navix_spaces.Space):
        pass

    # Provide required arguments for Space base class
    mock_space = MockSpace(
        shape=(),
        dtype=jnp.int32,
        minimum=jnp.array(0),
        maximum=jnp.array(10),
    )
    with pytest.raises(ValueError, match="Unsupported space type"):
        convert_navix_to_envelope_space(mock_space)


def test_step_type_conversion(navix_env, prng_key):
    """Test all navix StepType values are correctly converted."""
    import navix

    from envelope.adapters.navix_envelope import convert_navix_to_envelope_info

    env = navix_env
    key = prng_key

    # Test TRANSITION step type (should be on reset)
    reset_timestep = env.navix_env.reset(key)
    reset_info = convert_navix_to_envelope_info(reset_timestep)
    # TRANSITION should map to neither terminated nor truncated
    assert reset_timestep.step_type == navix.StepType.TRANSITION
    assert not reset_info.terminated
    assert not reset_info.truncated

    # Test TRANSITION on normal step (before termination/truncation)
    state, _ = env.init(key)
    action = env.action_space.sample(jax.random.fold_in(prng_key, 1))
    step_timestep = env.navix_env.step(state, action)
    step_info = convert_navix_to_envelope_info(step_timestep)
    # TRANSITION should map to neither terminated nor truncated
    if step_timestep.step_type == navix.StepType.TRANSITION:
        assert not step_info.terminated
        assert not step_info.truncated

    # Verify the conversion logic works correctly for all step types
    assert step_info.terminated == (
        step_timestep.step_type == navix.StepType.TERMINATION
    )
    assert step_info.truncated == (step_timestep.step_type == navix.StepType.TRUNCATION)

    # Test that TERMINATION maps correctly
    # (We can't easily trigger termination, but we verify the logic)
    assert step_info.terminated == (
        int(step_timestep.step_type) == int(navix.StepType.TERMINATION)
    )
    assert step_info.truncated == (
        int(step_timestep.step_type) == int(navix.StepType.TRUNCATION)
    )


def test_discrete_space_conversion():
    """Test conversion of discrete spaces from navix to envelope."""
    from navix import spaces as navix_spaces

    from envelope.adapters.navix_envelope import convert_navix_to_envelope_space

    # Create a navix Discrete space
    navix_discrete = navix_spaces.Discrete.create(10, shape=(3,), dtype=jnp.int32)
    envelope_discrete = convert_navix_to_envelope_space(navix_discrete)
    assert isinstance(envelope_discrete, Discrete)
    # Navix n might be scalar, envelope n can be broadcast to shape.
    navix_n = navix_discrete.n
    envelope_n = envelope_discrete.n
    if jnp.ndim(navix_n) == 0:
        assert jnp.all(envelope_n == navix_n)
    else:
        assert jnp.array_equal(envelope_n, navix_n)
    assert envelope_discrete.shape == navix_discrete.shape
    assert envelope_discrete.dtype == navix_discrete.dtype


def test_continuous_space_conversion():
    """Test conversion of continuous spaces from navix to envelope."""
    from navix import spaces as navix_spaces

    from envelope.adapters.navix_envelope import convert_navix_to_envelope_space

    # Create a navix Continuous space
    navix_continuous = navix_spaces.Continuous.create(
        shape=(3,),
        minimum=jnp.array([-1.0, -2.0, -3.0]),
        maximum=jnp.array([1.0, 2.0, 3.0]),
        dtype=jnp.float32,
    )

    # Convert to envelope space
    envelope_continuous = convert_navix_to_envelope_space(navix_continuous)

    # Verify conversion
    assert isinstance(envelope_continuous, Continuous)
    assert envelope_continuous.shape == navix_continuous.shape
    assert envelope_continuous.dtype == navix_continuous.dtype
    assert jnp.array_equal(envelope_continuous.low, navix_continuous.minimum)
    assert jnp.array_equal(envelope_continuous.high, navix_continuous.maximum)
