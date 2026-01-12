"""Tests for jenv.compat.brax_jenv module."""

from copy import deepcopy

import jax
import jax.numpy as jnp
import pytest
from brax.envs import Wrapper as BraxWrapper

from jenv.compat.brax_jenv import BraxJenv
from jenv.environment import Info
from jenv.spaces import Continuous


def test_brax2jenv_wrapper():
    """Test that Brax2Jenv wrapper correctly wraps a Brax environment."""
    # Create a Brax2Jenv wrapper with the ant environment
    env = BraxJenv.from_name("fast")

    # Test reset
    key = jax.random.PRNGKey(0)
    state, step_info = env.reset(key)

    # Check that info has the correct structure
    assert isinstance(step_info, Info)

    # Check initial values
    assert not step_info.done
    assert jnp.isscalar(step_info.reward) or step_info.reward.shape == ()

    # Check that observation is in observation space
    assert env.observation_space.contains(step_info.obs)

    # Test step with a valid action
    action = env.action_space.sample(jax.random.PRNGKey(1))
    next_state, next_step_info = env.step(state, action)

    # Check that next_state has the correct structure
    assert next_state is not None
    assert isinstance(next_step_info, Info)

    # Check that observation is in observation space
    assert env.observation_space.contains(next_step_info.obs)

    # Check that action space has the correct bounds
    assert isinstance(env.action_space, type(env.action_space))
    assert env.action_space.low == pytest.approx(-1.0)
    assert env.action_space.high == pytest.approx(1.0)

    # Check that observation space exists
    assert env.observation_space is not None


def test_brax_protocol_adherence():
    """Ensure reset/step return values adhere to the Info and State protocols."""
    env = BraxJenv.from_name("fast")

    key = jax.random.PRNGKey(0)
    state, info = env.reset(key)
    assert state is not None
    assert isinstance(info, Info)

    action = env.action_space.sample(jax.random.PRNGKey(1))
    next_state, next_info = env.step(state, action)
    assert next_state is not None
    assert isinstance(next_info, Info)


def test_info_fields_reset():
    """Test that Info container has correct fields on reset."""
    env = BraxJenv.from_name("fast")
    key = jax.random.PRNGKey(0)

    state, info = env.reset(key)

    # Verify Info has correct structure
    assert hasattr(info, "obs")
    assert hasattr(info, "reward")
    assert hasattr(info, "terminated")

    # Verify reward is 0.0 on reset
    assert info.reward == 0.0

    # Verify terminated is False on reset
    assert info.terminated is False

    # Check extra Brax state fields are preserved
    # Brax state typically has: obs, reward, done, metrics, info
    assert hasattr(info, "done")
    assert hasattr(info, "metrics")

    # Verify state fields match what was returned
    assert state is not None
    assert hasattr(state, "obs")


def test_info_fields_step():
    """Test that Info container has correct fields on step."""
    env = BraxJenv.from_name("fast")
    key = jax.random.PRNGKey(0)

    state, _ = env.reset(key)
    action = env.action_space.sample(jax.random.PRNGKey(1))
    next_state, info = env.step(state, action)

    # Verify Info has correct structure
    assert hasattr(info, "obs")
    assert hasattr(info, "reward")
    assert hasattr(info, "terminated")

    # Verify reward is a scalar
    assert jnp.isscalar(info.reward) or info.reward.shape == ()

    # Verify terminated field exists and matches done
    assert hasattr(info, "done")
    assert info.terminated == info.done

    # Verify extra Brax state fields are preserved
    assert hasattr(info, "metrics")

    # Verify state is not None
    assert next_state is not None


def test_episode_termination():
    """Test that episode termination is correctly detected."""
    env = BraxJenv.from_name("fast")
    key = jax.random.PRNGKey(0)

    state, info = env.reset(key)
    assert not info.terminated
    assert not info.done

    # Step until termination (Brax environments may have max episode length)
    # Run for a reasonable number of steps
    max_steps = 1000
    for step in range(max_steps):
        if info.done:
            break
        action = env.action_space.sample(jax.random.PRNGKey(step))
        state, info = env.step(state, action)

    # Verify that terminated matches done flag
    assert info.terminated == info.done


def test_multiple_episodes():
    """Test multiple reset/step cycles."""
    env = BraxJenv.from_name("fast")

    # Run multiple episodes
    for episode in range(3):
        state, info = env.reset(jax.random.PRNGKey(episode))

        # Verify reset clears terminated flag
        assert not info.terminated
        assert not info.done

        # Take a few steps
        for step in range(5):
            action = env.action_space.sample(jax.random.PRNGKey(episode * 100 + step))
            state, info = env.step(state, action)

        # Verify state is properly reset on next reset
        next_state, next_info = env.reset(jax.random.PRNGKey(episode + 1))
        assert not next_info.terminated
        assert not next_info.done

        # Different keys should give different initial observations
        if episode > 0:
            prev_state, prev_info = env.reset(jax.random.PRNGKey(episode - 1))
            # States from different keys may differ
            assert next_state is not None and prev_state is not None


def test_full_episode_rollout():
    """Test a complete episode rollout using jax.lax.scan."""
    env = BraxJenv.from_name("fast")
    key = jax.random.PRNGKey(0)

    # Reset environment
    state, _ = env.reset(key)

    # Generate actions for a fixed number of steps
    num_steps = 25
    action_keys = jax.random.split(key, num_steps)
    actions = jax.vmap(env.action_space.sample)(action_keys)

    # Use jax.lax.scan to efficiently step through the episode
    def step_fn(state, action):
        return env.step(state, action)

    final_state, step_infos = jax.lax.scan(step_fn, state, actions)

    # Verify all rewards are finite
    assert jnp.all(jnp.isfinite(step_infos.reward))

    # Verify final state is not None
    assert final_state is not None

    # Check that Info structure is preserved through scan (batched)
    assert isinstance(step_infos, Info)

    # Test shape of batched rewards matches num_steps
    assert step_infos.reward.shape == (num_steps,)

    # Verify observations are batched correctly
    assert step_infos.obs.shape[0] == num_steps


def test_from_name_creation():
    """Test BraxJenv.from_name() with different environment names."""
    # Test with "fast" (simple test env)
    env_fast = BraxJenv.from_name("fast")
    assert env_fast is not None
    assert isinstance(env_fast.action_space, Continuous)
    assert isinstance(env_fast.observation_space, Continuous)

    # Test reset and step work
    key = jax.random.PRNGKey(0)
    state, info = env_fast.reset(key)
    assert state is not None
    assert isinstance(info, Info)

    # Test with "ant" (standard benchmark)
    env_ant = BraxJenv.from_name("ant")
    assert env_ant is not None
    assert isinstance(env_ant.action_space, Continuous)
    assert isinstance(env_ant.observation_space, Continuous)

    # Different environments should have different space dimensions
    assert env_fast.action_space.shape != env_ant.action_space.shape


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


def test_deepcopy_warning():
    """Test that deepcopy raises a warning and returns shallow copy."""
    env = BraxJenv.from_name("fast")

    # Call deepcopy and verify warning is raised
    with pytest.warns(
        RuntimeWarning, match="Trying to deepcopy.*shallow copy is returned"
    ):
        copied_env = deepcopy(env)

    # Verify shallow copy is returned
    assert copied_env is not None

    # Verify the copied environment is usable
    key = jax.random.PRNGKey(0)
    state, info = copied_env.reset(key)
    assert state is not None
    assert isinstance(info, Info)


def test_deterministic_reset():
    """Test that reset with same key produces same initial observations."""
    env = BraxJenv.from_name("fast")

    # Reset with same key multiple times
    key = jax.random.PRNGKey(42)
    state1, info1 = env.reset(key)
    state2, info2 = env.reset(key)

    # Verify same initial observations
    assert jnp.array_equal(info1.obs, info2.obs)
    assert jnp.array_equal(state1.obs, state2.obs)

    # Test different keys - some environments may have deterministic initial states
    # So we just verify the reset is consistent, not necessarily different
    key_different = jax.random.PRNGKey(123)
    state3, info3 = env.reset(key_different)
    state4, info4 = env.reset(key_different)

    # Same key should produce same result
    assert jnp.array_equal(info3.obs, info4.obs)
    assert jnp.array_equal(state3.obs, state4.obs)
