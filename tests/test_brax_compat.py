"""Tests for jenv.compat.brax_jenv module."""

import jax
import jax.numpy as jnp
import pytest

from jenv.compat.brax_jenv import BraxJenv
from jenv.environment import Info


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
