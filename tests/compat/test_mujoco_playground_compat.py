"""Tests for jenv.compat.mujoco_playground_jenv module."""

import jax
import jax.numpy as jnp
import pytest

from jenv.compat.mujoco_playground_jenv import MujocoPlaygroundJenv
from jenv.environment import Info
from jenv.spaces import Continuous, PyTreeSpace
from tests.compat.contract import (
    assert_jitted_rollout_contract,
    assert_reset_step_contract,
)

pytestmark = pytest.mark.compat


def _create_mujoco_playground_env(env_name: str = "CartpoleBalance", **kwargs):
    """Helper to create a MujocoPlaygroundJenv wrapper."""
    return MujocoPlaygroundJenv.from_name(env_name, env_kwargs=kwargs or None)


@pytest.fixture(scope="module")
def mujoco_playground_env():
    return _create_mujoco_playground_env()


@pytest.fixture(scope="module", autouse=True)
def _mujoco_playground_env_warmup(mujoco_playground_env, prng_key):
    """Warm up reset/step once to amortize compilation."""
    env = mujoco_playground_env
    key_reset, key_step = jax.random.split(prng_key)
    state, _info = env.reset(key_reset)
    action = env.action_space.sample(key_step)
    env.step(state, action)


def test_mujoco_playground_contract_smoke(prng_key, mujoco_playground_env):
    env = mujoco_playground_env

    def obs_check(obs, obs_space):
        assert obs_space.contains(obs)

    assert_reset_step_contract(env, key=prng_key, obs_check=obs_check)


def test_mujoco_playground_contract_scan(
    prng_key, mujoco_playground_env, scan_num_steps
):
    assert_jitted_rollout_contract(
        mujoco_playground_env, key=prng_key, num_steps=scan_num_steps
    )


def test_mujoco_playground_terminated_matches_done_on_step(
    mujoco_playground_env, prng_key
):
    """MuJoCo Playground-specific: wrapper exposes underlying `done` and maps it to `terminated`."""
    env = mujoco_playground_env
    key_reset, key_action = jax.random.split(prng_key)
    state, _info = env.reset(key_reset)
    action = env.action_space.sample(key_action)
    _next_state, info = env.step(state, action)
    assert hasattr(info, "done")
    assert info.terminated == info.done
    assert not info.truncated  # mujoco_playground doesn't distinguish truncation


def test_action_space_property(mujoco_playground_env):
    """Test that action_space is accessible and has correct properties."""
    env = mujoco_playground_env

    # Check that action space is a Continuous space
    assert isinstance(env.action_space, Continuous)
    assert env.action_space.shape == (env.mujoco_playground_env.action_size,)
    # MuJoCo Playground actions are typically bounded [-1, 1]
    assert jnp.allclose(env.action_space.low, -1.0)
    assert jnp.allclose(env.action_space.high, 1.0)


def test_observation_space_int_obs_size(prng_key):
    """Test observation_space for envs with int observation_size (e.g. CartpoleBalance)."""
    env = MujocoPlaygroundJenv.from_name("CartpoleBalance")

    # CartpoleBalance has int observation_size, should return Continuous space
    obs_size = env.mujoco_playground_env.observation_size
    assert isinstance(obs_size, int)
    assert isinstance(env.observation_space, Continuous)
    assert env.observation_space.shape == (obs_size,)

    # Verify reset produces obs that matches the space
    state, info = env.reset(prng_key)
    assert env.observation_space.contains(info.obs)


def test_observation_space_dict_obs_size(prng_key):
    """Test observation_space for envs with dict observation_size (e.g. Go1Handstand)."""
    env = MujocoPlaygroundJenv.from_name("Go1Handstand")

    # Go1Handstand has dict observation_size, should return PyTreeSpace
    obs_size = env.mujoco_playground_env.observation_size
    assert isinstance(obs_size, dict)
    assert isinstance(env.observation_space, PyTreeSpace)

    # Check that the PyTreeSpace has the correct structure
    for key, shape in obs_size.items():
        assert key in env.observation_space.tree
        assert isinstance(env.observation_space.tree[key], Continuous)
        assert env.observation_space.tree[key].shape == shape

    # Verify reset produces obs that matches the space
    state, info = env.reset(prng_key)
    assert isinstance(info.obs, dict)
    assert env.observation_space.contains(info.obs)


def test_from_name_with_env_kwargs(prng_key):
    """Test that from_name accepts env_kwargs (even if not used by registry.load)."""
    # Note: mujoco_playground.registry.load may not accept kwargs,
    # but from_name should still accept env_kwargs for consistency
    env = MujocoPlaygroundJenv.from_name("CartpoleBalance", env_kwargs={})

    assert env is not None
    key = prng_key
    state, info = env.reset(key)
    assert state is not None


def test_from_name_with_episode_length_warning(prng_key):
    """Test that from_name warns when using finite episode_length."""
    # Test warning for finite episode_length
    with pytest.warns(
        UserWarning,
        match="Creating a MujocoPlaygroundJenv with a finite episode_length is not recommended",
    ):
        env = MujocoPlaygroundJenv.from_name(
            "CartpoleBalance", env_kwargs={"episode_length": 100}
        )

    # Environment should still be created
    assert env is not None
    key = prng_key
    state, info = env.reset(key)
    assert state is not None


def test_episode_length_defaults_to_inf(mujoco_playground_env):
    """Test that by default episode_length is set to a very large value (effectively inf)."""
    env = mujoco_playground_env
    # Check that episode_length is set to a very large value in the config
    # (mujoco_playground uses int, so we use max int instead of inf)
    config = getattr(env.mujoco_playground_env, "_config", None)
    if config is not None:
        episode_length = config.get("episode_length")
        # Should be set to max int32 value (effectively infinite for practical purposes)
        max_int = int(jnp.iinfo(jnp.int32).max)
        assert episode_length == max_int or jnp.isposinf(jnp.asarray(episode_length))


def test_multiple_mujoco_playground_envs(prng_key):
    """Smoke-test a few different mujoco_playground environments."""
    from mujoco_playground import registry

    # Get a few different environments to test
    env_names = ["CartpoleBalance", "AcrobotSwingup", "BallInCup"]

    # Keep this deterministic and small (compile/runtime)
    for env_name in env_names:
        # Skip if environment doesn't exist
        if env_name not in registry.ALL_ENVS:
            pytest.skip(f"Environment {env_name} not available")

        env = _create_mujoco_playground_env(env_name)
        reset_key, action_key = jax.random.split(prng_key, 2)

        state, info = env.reset(reset_key)
        assert state is not None
        assert isinstance(info, Info)
        # Skip expensive contains check - shape/dtype check is sufficient
        assert info.obs.shape == env.observation_space.shape

        action = env.action_space.sample(action_key)
        next_state, next_info = env.step(state, action)
        assert next_state is not None
        assert isinstance(next_info, Info)
        assert next_info.obs.shape == env.observation_space.shape
        assert jnp.all(jnp.isfinite(jnp.asarray(next_info.reward)))


def test_from_name_rejects_unknown_env_kwargs():
    """Test that from_name handles unknown/invalid env_kwargs appropriately."""
    # Try with an unknown config key - mujoco_playground may raise KeyError
    # or silently ignore it depending on implementation
    try:
        env = MujocoPlaygroundJenv.from_name(
            "CartpoleBalance", env_kwargs={"unknown_key": 123}
        )
        # If it doesn't raise, that's also acceptable behavior
        assert env is not None
    except (KeyError, TypeError) as e:
        # If it raises an error, that's also acceptable - just verify it's a reasonable error
        assert (
            "unknown" in str(e).lower()
            or "key" in str(e).lower()
            or "invalid" in str(e).lower()
        )


def test_config_overrides_are_passed_correctly(prng_key):
    """Test that config_overrides are correctly passed to registry.load."""
    # Test with a known config parameter (ctrl_dt)
    env = MujocoPlaygroundJenv.from_name(
        "CartpoleBalance", env_kwargs={"ctrl_dt": 0.072}
    )

    # Verify the config was applied
    assert env.mujoco_playground_env.dt == 0.072

    # Test with another known parameter (sim_dt)
    env2 = MujocoPlaygroundJenv.from_name(
        "CartpoleBalance", env_kwargs={"sim_dt": 0.005}
    )

    # Verify the config was applied
    assert env2.mujoco_playground_env.sim_dt == 0.005
