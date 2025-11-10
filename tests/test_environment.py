"""Tests for jenv.environment module."""

from functools import cached_property

import jax
import jax.numpy as jnp
import pytest

from jenv.environment import Environment, Info
from jenv.spaces import Continuous, Discrete
from jenv.struct import static_field


def test_environment_is_abstract():
    """Environment is an abstract base class and cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Environment()


def test_info_done_property():
    """Info.done should reflect terminated or truncated flags."""
    running = Info(
        obs=jnp.array([0.0]),
        reward=0.0,
        terminated=False,
        truncated=False,
    )
    terminated = Info(
        obs=jnp.array([1.0]),
        reward=1.0,
        terminated=True,
        truncated=False,
    )
    truncated = Info(
        obs=jnp.array([2.0]),
        reward=1.0,
        terminated=False,
        truncated=True,
    )

    assert running.done is False
    assert terminated.done is True
    assert truncated.done is True


class CounterEnv(Environment):
    """Minimal concrete environment used for exercising the base class contract."""

    limit: int = static_field(default=3)

    def reset(self, key: jax.Array):
        del key
        state = jnp.array(0, dtype=jnp.int32)
        info = Info(
            obs=state,
            reward=0.0,
            terminated=False,
            truncated=False,
        )
        return state, info

    def step(self, key: jax.Array, state: jax.Array, action: jax.Array):
        del key
        next_state = state + 1
        terminated = bool((next_state >= self.limit).item())
        info = Info(
            obs=next_state,
            reward=float(action),
            terminated=terminated,
            truncated=False,
        )
        return next_state, info

    @cached_property
    def observation_space(self) -> Discrete:
        return Discrete(n=self.limit + 1, dtype=jnp.int32)

    @cached_property
    def action_space(self) -> Continuous:
        return Continuous(low=-1.0, high=1.0, dtype=jnp.float32)


def test_environment_concrete_subclass_behaviour():
    """Ensure a concrete Environment subclass satisfies the expected protocol."""
    env = CounterEnv(limit=2)

    state, info = env.reset(jax.random.PRNGKey(0))
    assert env.observation_space.contains(state)
    assert info.done is False

    next_state, next_info = env.step(jax.random.PRNGKey(1), state, jnp.array(0.5))
    assert env.observation_space.contains(next_state)
    assert env.action_space.contains(jnp.array(0.5))
    assert next_info.reward == pytest.approx(0.5)

    final_state, final_info = env.step(
        jax.random.PRNGKey(2), next_state, jnp.array(-0.2)
    )
    assert env.observation_space.contains(final_state)
    assert final_info.done is True
