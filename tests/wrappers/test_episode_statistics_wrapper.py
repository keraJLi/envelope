import jax
import jax.numpy as jnp

import envelope.wrappers as wrappers

from envelope.wrappers.episode_statistics_wrapper import (
    EpisodeStatistics,
    EpisodeStatisticsWrapper,
)
from envelope.wrappers.truncation_wrapper import TruncationWrapper
from envelope.wrappers.vmap_wrapper import VmapWrapper
from tests.wrappers.helpers import StepCounterEnv


def test_init_creates_default_stats():
    env = StepCounterEnv()
    w = EpisodeStatisticsWrapper(env)
    key = jax.random.key(0)
    state, info = w.init(key)
    assert jnp.allclose(state.stats.reward, 0.0)
    assert jnp.allclose(state.stats.length, 0.0)
    assert hasattr(info, "stats")
    assert jnp.allclose(info.stats.reward, 0.0)
    assert jnp.allclose(info.stats.length, 0.0)


def test_step_accumulates_reward():
    env = StepCounterEnv()
    w = EpisodeStatisticsWrapper(env)
    key = jax.random.key(0)
    state, _ = w.init(key)
    actions = [0.1, 0.2, -0.1]
    for a in actions:
        state, info = w.step(state, jnp.asarray(a))
    assert jnp.allclose(state.stats.reward, sum(actions))
    assert jnp.allclose(info.stats.reward, sum(actions))


def test_step_increments_length():
    env = StepCounterEnv()
    w = EpisodeStatisticsWrapper(env)
    key = jax.random.key(0)
    state, _ = w.init(key)
    n_steps = 5
    for _ in range(n_steps):
        state, info = w.step(state, jnp.asarray(0.1))
    assert jnp.allclose(state.stats.length, n_steps)
    assert jnp.allclose(info.stats.length, n_steps)


def test_info_contains_stats_field():
    env = StepCounterEnv()
    w = EpisodeStatisticsWrapper(env)
    key = jax.random.key(0)
    state, info = w.init(key)
    assert hasattr(info, "stats")
    state, info = w.step(state, jnp.asarray(0.5))
    assert hasattr(info, "stats")
    assert isinstance(info.stats, EpisodeStatistics)


def test_reset_clears_episode_stats():
    env = StepCounterEnv()
    w = EpisodeStatisticsWrapper(env)
    key = jax.random.key(0)
    state, _ = w.init(key)
    for _ in range(3):
        state, _ = w.step(state, jnp.asarray(0.2))
    state, info = w.reset(state, key)
    assert jnp.allclose(state.stats.reward, 0.0)
    assert jnp.allclose(state.stats.length, 0)
    assert jnp.allclose(info.stats.reward, 0.0)
    assert jnp.allclose(info.stats.length, 0)
    assert w.observation_space.contains(info.obs)


def test_stats_restart_after_reset():
    env = StepCounterEnv()
    w = EpisodeStatisticsWrapper(env)
    key = jax.random.key(0)
    state, _ = w.init(key)
    for _ in range(3):
        state, _ = w.step(state, jnp.asarray(0.1))
    state, _ = w.reset(state, key)
    for _ in range(2):
        state, _ = w.step(state, jnp.asarray(0.1))
    assert jnp.allclose(state.stats.length, 2)
    assert jnp.allclose(state.stats.reward, 0.2)


def test_cumulative_statistics_are_explicit_and_survive_reset():
    cls = getattr(wrappers, "CumulativeStatisticsWrapper")
    w = cls(StepCounterEnv())
    state, _ = w.init(jax.random.key(0))
    state, _ = w.step(state, jnp.asarray(0.2))
    state, info = w.reset(state=state, key=jax.random.key(1))

    assert jnp.allclose(state.stats.reward, 0.2)
    assert jnp.allclose(state.stats.length, 1)
    assert jnp.allclose(info.stats.reward, 0.2)


def test_negative_rewards_accumulate_correctly():
    env = StepCounterEnv()
    w = EpisodeStatisticsWrapper(env)
    key = jax.random.key(0)
    state, _ = w.init(key)
    state, _ = w.step(state, jnp.asarray(1.0))
    state, _ = w.step(state, jnp.asarray(-0.5))
    assert jnp.allclose(state.stats.reward, 0.5)


def test_state_is_episode_statistics_state_with_inner_state():
    env = StepCounterEnv()
    w = EpisodeStatisticsWrapper(env)
    key = jax.random.key(0)
    state, _ = w.init(key)
    assert hasattr(state, "inner_state")
    assert hasattr(state, "stats")
    assert state.inner_state is not None


def test_state_unwrapped_reaches_inner_env_state():
    env = StepCounterEnv()
    w = EpisodeStatisticsWrapper(env)
    key = jax.random.key(0)
    state, _ = w.init(key)
    assert hasattr(state, "unwrapped")
    inner = state.unwrapped
    assert hasattr(inner, "env_state")


def test_observation_space_action_space_unchanged():
    env = StepCounterEnv()
    w = EpisodeStatisticsWrapper(env)
    assert w.observation_space is env.observation_space
    assert w.action_space is env.action_space


def test_observation_space_contains_after_init_and_step():
    env = StepCounterEnv()
    w = EpisodeStatisticsWrapper(env)
    key = jax.random.key(0)
    state, info = w.init(key)
    assert w.observation_space.contains(info.obs)
    state, info = w.step(state, jnp.asarray(0.1))
    assert w.observation_space.contains(info.obs)


def test_jit_init_step_loop():
    env = StepCounterEnv()
    w = EpisodeStatisticsWrapper(env)
    key = jax.random.key(0)
    n_steps = 4

    @jax.jit
    def run_n_steps(k):
        s, _ = w.init(k)

        def body(carry, _):
            next_s, _ = w.step(carry, jnp.asarray(0.1))
            return next_s, ()

        s, _ = jax.lax.scan(body, s, None, length=n_steps)
        return s.stats

    stats = run_n_steps(key)
    assert jnp.allclose(jnp.asarray(stats.length), n_steps)
    assert jnp.allclose(jnp.asarray(stats.reward), 0.4)


def test_composability_with_vmap_wrapper():
    batch_size = 3
    env = StepCounterEnv()
    w = EpisodeStatisticsWrapper(VmapWrapper(env, batch_size=batch_size))
    key = jax.random.key(0)
    state, info = w.init(key)
    assert jnp.asarray(state.stats.reward).shape == (batch_size,)
    assert jnp.asarray(state.stats.length).shape == (batch_size,)
    assert jnp.asarray(info.stats.reward).shape == (batch_size,)

    actions = jnp.asarray([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]])
    state, _ = jax.lax.scan(
        lambda carry, action: (w.step(carry, action)[0], None), state, actions
    )

    assert jnp.asarray(state.stats.reward).shape == (batch_size,)
    assert jnp.asarray(state.stats.length).shape == (batch_size,)
    assert jnp.allclose(jnp.asarray(state.stats.reward), actions.sum(axis=0))
    assert jnp.all(state.stats.length == 2)


def test_composability_with_truncation_wrapper():
    env = StepCounterEnv(terminate_after=10)
    w = EpisodeStatisticsWrapper(TruncationWrapper(env, max_steps=5))
    key = jax.random.key(0)
    state, _ = w.init(key)
    for _ in range(6):
        state, info = w.step(state, jnp.asarray(0.1))
    # Truncation happens at step 5; stats still accumulate through truncation
    assert jnp.allclose(state.stats.length, 6)
    assert jnp.allclose(state.stats.reward, 0.6)
