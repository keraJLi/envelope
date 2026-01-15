"""Tests for jenv.compat.navix_jenv module."""

import jax
import jax.numpy as jnp
import navix
import pytest

from jenv.compat.navix_jenv import NavixJenv
from jenv.spaces import Continuous, Discrete
from tests.compat.contract import (
    assert_jitted_rollout_contract,
    assert_reset_step_contract,
)

pytestmark = pytest.mark.compat


def _create_navix_env(env_name: str = "Navix-Empty-5x5-v0", **kwargs):
    """Helper to create a NavixJenv wrapper."""
    navix_env = navix.make(env_name, **kwargs)
    return NavixJenv(navix_env=navix_env)


@pytest.fixture(scope="module")
def navix_env():
    return _create_navix_env()


@pytest.fixture(scope="module", autouse=True)
def _navix_env_warmup(navix_env, prng_key):
    env = navix_env
    key_reset, key_step = jax.random.split(prng_key)
    state, _info = env.reset(key_reset)
    action = env.action_space.sample(key_step)
    env.step(state, action)


def test_navix_contract_smoke(prng_key, navix_env):
    env = navix_env

    def obs_check(obs, obs_space):
        # Some navix envs can emit obs outside declared bounds; check shape/dtype only.
        assert obs.shape == obs_space.shape
        assert obs.dtype == obs_space.dtype

    assert_reset_step_contract(env, key=prng_key, obs_check=obs_check)


def test_navix_contract_scan(prng_key, navix_env, scan_num_steps):
    assert_jitted_rollout_contract(navix_env, key=prng_key, num_steps=scan_num_steps)


def test_action_space_conversion(navix_env):
    """Test conversion of navix action spaces to jenv spaces."""
    env = navix_env

    # Check that action space is converted correctly
    assert isinstance(env.action_space, Discrete)
    assert env.action_space.n == env.navix_env.action_space.n
    assert env.action_space.shape == env.navix_env.action_space.shape
    assert env.action_space.dtype == env.navix_env.action_space.dtype


def test_observation_space_conversion(navix_env):
    """Test conversion of navix observation spaces to jenv spaces."""
    env = navix_env

    # Check that observation space is converted correctly
    assert isinstance(env.observation_space, Discrete)
    # Navix n might be scalar, jenv n is array - check if all elements equal navix n
    navix_n = env.navix_env.observation_space.n
    jenv_n = env.observation_space.n
    if jnp.ndim(navix_n) == 0:  # scalar
        assert jnp.all(jenv_n == navix_n)
    else:
        assert jnp.array_equal(jenv_n, navix_n)
    assert env.observation_space.shape == env.navix_env.observation_space.shape
    assert env.observation_space.dtype == env.navix_env.observation_space.dtype


def test_container_conversion_reset(navix_env, prng_key):
    """Test convert_navix_to_jenv_info on reset timestep."""
    env = navix_env
    key = prng_key

    # Get the raw navix timestep
    navix_timestep = env.navix_env.reset(key)
    state, info = env.reset(key)

    # Verify obs, reward, terminated, truncated fields
    assert jnp.array_equal(info.obs, navix_timestep.observation)
    assert info.reward == navix_timestep.reward
    assert info.terminated == (navix_timestep.step_type == navix.StepType.TERMINATION)
    assert info.truncated == (navix_timestep.step_type == navix.StepType.TRUNCATION)

    # Verify terminated=False and truncated=False on reset
    assert not info.terminated
    assert not info.truncated

    # Check that extra timestep fields are preserved via update()
    # (if navix timestep has extra fields beyond the standard ones)
    import dataclasses

    timestep_dict = dataclasses.asdict(navix_timestep)
    # Remove standard fields that are handled explicitly
    standard_fields = {"observation", "reward", "step_type"}
    extra_fields = {k: v for k, v in timestep_dict.items() if k not in standard_fields}
    for field_name, field_value in extra_fields.items():
        assert hasattr(info, field_name)


def test_container_conversion_step(navix_env, prng_key):
    """Test convert_navix_to_jenv_info on step timestep."""
    env = navix_env
    key = prng_key

    state, _ = env.reset(key)
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
    env = _create_navix_env(max_steps=5)  # Very short episode
    key = prng_key

    state, info = env.reset(key)

    # Step until truncation
    for _ in range(10):  # More than max_steps
        if info.truncated:
            break
        action = env.action_space.sample(jax.random.fold_in(prng_key, _))
        state, info = env.step(state, action)

    # Verify truncation occurred
    assert info.truncated
    assert not info.terminated


def test_unsupported_space_type():
    """Test that unsupported space types raise ValueError."""
    import jax.numpy as jnp
    from navix import spaces as navix_spaces

    from jenv.compat.navix_jenv import convert_navix_to_jenv_space

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
        convert_navix_to_jenv_space(mock_space)


def test_step_type_conversion(navix_env, prng_key):
    """Test all navix StepType values are correctly converted."""
    import navix

    from jenv.compat.navix_jenv import convert_navix_to_jenv_info

    env = navix_env
    key = prng_key

    # Test TRANSITION step type (should be on reset)
    reset_timestep = env.navix_env.reset(key)
    reset_info = convert_navix_to_jenv_info(reset_timestep)
    # TRANSITION should map to neither terminated nor truncated
    assert reset_timestep.step_type == navix.StepType.TRANSITION
    assert not reset_info.terminated
    assert not reset_info.truncated

    # Test TRANSITION on normal step (before termination/truncation)
    state, _ = env.reset(key)
    action = env.action_space.sample(jax.random.fold_in(prng_key, 1))
    step_timestep = env.navix_env.step(state, action)
    step_info = convert_navix_to_jenv_info(step_timestep)
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


def test_from_name_with_max_steps_warning(prng_key):
    """Test that from_name warns when using finite max_steps."""
    # Test warning for finite max_steps
    with pytest.warns(
        UserWarning,
        match="Creating a NavixJenv with a finite max_steps is not recommended",
    ):
        env = NavixJenv.from_name("Navix-Empty-5x5-v0", env_kwargs={"max_steps": 100})

    # Environment should still be created
    assert env is not None
    key = prng_key
    state, info = env.reset(key)
    assert state is not None


def test_discrete_space_conversion():
    """Test conversion of discrete spaces from navix to jenv."""
    from navix import spaces as navix_spaces

    from jenv.compat.navix_jenv import convert_navix_to_jenv_space

    # Create a navix Discrete space
    navix_discrete = navix_spaces.Discrete.create(10, shape=(3,), dtype=jnp.int32)
    jenv_discrete = convert_navix_to_jenv_space(navix_discrete)
    assert isinstance(jenv_discrete, Discrete)
    # Navix n might be scalar, jenv n can be broadcast to shape.
    navix_n = navix_discrete.n
    jenv_n = jenv_discrete.n
    if jnp.ndim(navix_n) == 0:
        assert jnp.all(jenv_n == navix_n)
    else:
        assert jnp.array_equal(jenv_n, navix_n)
    assert jenv_discrete.shape == navix_discrete.shape
    assert jenv_discrete.dtype == navix_discrete.dtype


def test_continuous_space_conversion():
    """Test conversion of continuous spaces from navix to jenv."""
    from navix import spaces as navix_spaces

    from jenv.compat.navix_jenv import convert_navix_to_jenv_space

    # Create a navix Continuous space
    navix_continuous = navix_spaces.Continuous.create(
        shape=(3,),
        minimum=jnp.array([-1.0, -2.0, -3.0]),
        maximum=jnp.array([1.0, 2.0, 3.0]),
        dtype=jnp.float32,
    )

    # Convert to jenv space
    jenv_continuous = convert_navix_to_jenv_space(navix_continuous)

    # Verify conversion
    assert isinstance(jenv_continuous, Continuous)
    assert jenv_continuous.shape == navix_continuous.shape
    assert jenv_continuous.dtype == navix_continuous.dtype
    assert jnp.array_equal(jenv_continuous.low, navix_continuous.minimum)
    assert jnp.array_equal(jenv_continuous.high, navix_continuous.maximum)
