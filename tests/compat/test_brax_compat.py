"""Tests for jenv.compat.brax_jenv module."""

from copy import deepcopy

import jax
import pytest
from brax.envs import Wrapper as BraxWrapper

from jenv.compat.brax_jenv import BraxJenv
from tests.compat.contract import (
    assert_jitted_rollout_contract,
    assert_reset_step_contract,
)

pytestmark = pytest.mark.compat


@pytest.fixture(scope="module")
def brax_fast_env():
    return BraxJenv.from_name("fast")


@pytest.fixture(scope="module", autouse=True)
def _brax_fast_env_warmup(brax_fast_env, prng_key):
    """Warm up reset/step once to amortize compilation."""
    env = brax_fast_env
    key_reset, key_step = jax.random.split(prng_key)
    state, _info = env.reset(key_reset)
    action = env.action_space.sample(key_step)
    env.step(state, action)


def test_brax_contract_smoke(prng_key, brax_fast_env):
    env = brax_fast_env

    def obs_check(obs, obs_space):
        assert obs_space.contains(obs)

    assert_reset_step_contract(env, key=prng_key, obs_check=obs_check)


def test_brax_contract_scan(prng_key, brax_fast_env, scan_num_steps):
    assert_jitted_rollout_contract(
        brax_fast_env, key=prng_key, num_steps=scan_num_steps
    )


def test_brax_info_preserves_brax_fields_on_reset(brax_fast_env, prng_key):
    """Brax-specific: extra Brax state fields are preserved on reset."""
    env = brax_fast_env
    key = prng_key

    state, info = env.reset(key)

    # Check extra Brax state fields are preserved
    # Brax state typically has: obs, reward, done, metrics, info
    assert hasattr(info, "done")
    assert hasattr(info, "metrics")

    # Verify state fields match what was returned
    assert state is not None
    assert hasattr(state, "obs")


def test_brax_terminated_matches_done_on_step(brax_fast_env, prng_key):
    """Brax-specific: wrapper exposes underlying `done` and maps it to `terminated`."""
    env = brax_fast_env
    key_reset, key_action = jax.random.split(prng_key)
    state, _info = env.reset(key_reset)
    action = env.action_space.sample(key_action)
    _next_state, info = env.step(state, action)
    assert hasattr(info, "done")
    assert info.terminated == info.done


def test_from_name_with_episode_length_warning(prng_key):
    """Test that from_name warns when using finite episode_length."""
    # Test warning for finite episode_length
    with pytest.warns(
        UserWarning,
        match="Creating a BraxJenv with a finite episode_length is not recommended",
    ):
        env = BraxJenv.from_name("fast", env_kwargs={"episode_length": 100})

    # Environment should still be created
    assert env is not None
    key = prng_key
    state, info = env.reset(key)
    assert state is not None


def test_from_name_with_auto_reset_warning(prng_key):
    """Test that from_name warns when using auto_reset=True."""
    # Test warning for auto_reset=True
    with pytest.warns(
        UserWarning, match="Creating a BraxJenv with auto_reset=True is not recommended"
    ):
        env = BraxJenv.from_name("fast", env_kwargs={"auto_reset": True})

    # Environment should still be created
    assert env is not None
    key = prng_key
    state, info = env.reset(key)
    assert state is not None


def test_wrapper_unwrapping():
    """Test that wrapped Brax environments are properly unwrapped."""
    from brax.envs import create as brax_create

    # Create a base Brax environment
    base_env = brax_create("fast", episode_length=None, auto_reset=False)

    # Create a simple wrapper
    class SimpleWrapper(BraxWrapper):
        def reset(self, rng):
            return self.env.reset(rng)

        def step(self, state, action):
            return self.env.step(state, action)

    wrapped_env = SimpleWrapper(base_env)

    # Initialize BraxJenv with wrapped environment
    with pytest.warns(
        UserWarning, match="Environment wrapping should be handled by jenv"
    ):
        env = BraxJenv(brax_env=wrapped_env)

    # Verify environment is properly unwrapped
    assert not isinstance(env.brax_env, BraxWrapper)
    assert env.brax_env is wrapped_env.unwrapped


def test_deepcopy_warning(brax_fast_env, prng_key):
    """Test that deepcopy raises a warning and returns shallow copy."""
    env = brax_fast_env

    # Call deepcopy and verify warning is raised
    with pytest.warns(
        RuntimeWarning, match="Trying to deepcopy.*shallow copy is returned"
    ):
        copied_env = deepcopy(env)

    # Verify shallow copy is returned
    assert copied_env is not None

    # Verify the copied environment is usable
    key = prng_key
    state, info = copied_env.reset(key)
    assert state is not None
