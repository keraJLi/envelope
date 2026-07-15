import jax
import jax.numpy as jnp
import pytest

from envelope.spaces import BatchedSpace
from envelope.wrappers.autoreset_wrapper import AutoResetWrapper
from envelope.wrappers.episode_statistics_wrapper import EpisodeStatisticsWrapper
from envelope.wrappers.pooled_init_vmap_wrapper import PooledInitVmapWrapper
from envelope.wrappers.truncation_wrapper import TruncationWrapper
from tests.wrappers.helpers import ScalarToyEnv, StepCounterEnv


def test_init_creates_batched_state():
    batch_size = 4
    env = ScalarToyEnv()
    w = PooledInitVmapWrapper(env, batch_size=batch_size, pool_size=4)
    key = jax.random.key(0)
    state, info = w.init(key)
    assert info.obs.shape == (batch_size,)


def test_init_last_final_is_type_preserving_placeholder():
    batch_size = 2
    env = ScalarToyEnv()
    w = PooledInitVmapWrapper(env, batch_size=batch_size, pool_size=2)
    key = jax.random.key(0)
    state, info = w.init(key)
    assert state.last_final.obs.dtype == info.obs.dtype
    assert state.last_final.terminated.dtype == info.terminated.dtype
    assert jnp.all(info.final_valid == jnp.asarray([False, False]))


def test_init_info_has_final_field():
    env = ScalarToyEnv()
    w = PooledInitVmapWrapper(env, batch_size=2, pool_size=2)
    key = jax.random.key(0)
    state, info = w.init(key)
    assert hasattr(info, "final")
    assert jnp.all(info.final_valid == jnp.asarray([False, False]))
    assert info.final.obs.dtype == info.obs.dtype


def test_step_non_done_envs_continue_normally():
    env = ScalarToyEnv()  # never done
    w = PooledInitVmapWrapper(env, batch_size=2, pool_size=2)
    key = jax.random.key(0)
    state, info = w.init(key)
    action = jnp.array([0.1, 0.2])
    state, info = w.step(state, action)
    # State should have progressed (obs = 0 + action)
    assert jnp.allclose(info.obs, jnp.array([0.1, 0.2]))


def test_step_done_envs_get_pool_states():
    env = StepCounterEnv(terminate_after=1)
    w = PooledInitVmapWrapper(env, batch_size=2, pool_size=4)
    key = jax.random.key(0)
    state, _ = w.init(key)
    action = jnp.array([0.1, 0.1])  # both take one step -> both done
    state, info = w.step(state, action)
    # Done envs get fresh init from pool; obs should be initial (0) from pool
    assert info.obs.shape == (2,)
    # Both envs terminated so both get pool inits (obs 0.0)
    assert jnp.allclose(info.obs, jnp.zeros(2))


def test_step_stores_terminal_info_in_last_final():
    env = StepCounterEnv(terminate_after=1)
    w = PooledInitVmapWrapper(env, batch_size=1, pool_size=1)
    key = jax.random.key(0)
    state, _ = w.init(key)
    action = jnp.array([0.5])
    state, info = w.step(state, action)
    # Terminal step: obs was 0+0.5=0.5, reward=0.5. last_final should have that terminal info
    assert jnp.allclose(state.last_final.obs, jnp.array([0.5]))
    assert jnp.allclose(state.last_final.reward, jnp.array(0.5))


def test_step_continue_envs_preserve_previous_last_final():
    env = StepCounterEnv(terminate_after=2)
    w = PooledInitVmapWrapper(env, batch_size=2, pool_size=2)
    key = jax.random.key(0)
    state, _ = w.init(key)
    # Step once: no env done yet (env_state 0.1 each)
    state, _ = w.step(state, jnp.array([0.1, 0.1]))
    # Step again: both hit terminate_after=2; terminal obs = 0.1+0.1 = 0.2 each
    state, info = w.step(state, jnp.array([0.1, 0.1]))
    # One more step: both get pool inits, neither done; last_final preserved
    state, _ = w.step(state, jnp.array([0.0, 0.0]))
    assert state.last_final.obs.shape == (2,)
    assert jnp.allclose(state.last_final.obs, jnp.array([0.2, 0.2]))
    assert jnp.allclose(state.last_final.reward, jnp.array([0.1, 0.1]))


def test_step_info_final_field():
    env = StepCounterEnv(terminate_after=1)
    w = PooledInitVmapWrapper(env, batch_size=1, pool_size=1)
    key = jax.random.key(0)
    state, info = w.init(key)
    assert hasattr(info, "final")
    state, info = w.step(state, jnp.array([0.5]))
    # For done env: final is terminal info
    assert hasattr(info, "final")
    assert jnp.allclose(info.final.obs, jnp.array([0.5]))


def test_terminal_transition_semantics_match_scalar_autoreset():
    scalar = AutoResetWrapper(StepCounterEnv(terminate_after=1))
    pooled = PooledInitVmapWrapper(
        StepCounterEnv(terminate_after=1), batch_size=1, pool_size=1
    )
    key = jax.random.key(0)
    scalar_state, _ = scalar.init(key)
    pooled_state, _ = pooled.init(key)

    _, scalar_info = scalar.step(scalar_state, jnp.asarray(0.5))
    _, pooled_info = pooled.step(pooled_state, jnp.asarray([0.5]))

    assert bool(jnp.asarray(scalar_info.terminated)) is True
    assert bool(jnp.asarray(pooled_info.terminated[0])) is True
    assert jnp.allclose(scalar_info.reward, pooled_info.reward[0])
    assert jnp.allclose(scalar_info.obs, pooled_info.obs[0])
    assert jnp.allclose(scalar_info.final.obs, pooled_info.final.obs[0])
    assert bool(jnp.asarray(scalar_info.final_valid)) is True
    assert bool(jnp.asarray(pooled_info.final_valid[0])) is True


@pytest.mark.parametrize("batch_size,pool_size", [(0, 1), (1, 0)])
def test_nonpositive_pooling_capability_is_rejected(batch_size, pool_size):
    with pytest.raises(ValueError, match="batch_size|pool_size"):
        PooledInitVmapWrapper(
            ScalarToyEnv(), batch_size=batch_size, pool_size=pool_size
        )


def test_reset_vmaps_inner_reset():
    batch_size = 3
    env = ScalarToyEnv()
    w = PooledInitVmapWrapper(env, batch_size=batch_size, pool_size=3)
    key = jax.random.key(0)
    state, info = w.init(key)
    state, info = w.reset(state, key)
    assert info.obs.shape == (batch_size,)
    assert w.observation_space.contains(info.obs)


def test_observation_space_is_batched_space():
    batch_size = 4
    env = ScalarToyEnv()
    w = PooledInitVmapWrapper(env, batch_size=batch_size, pool_size=4)
    assert isinstance(w.observation_space, BatchedSpace)
    assert w.observation_space.batch_size == batch_size
    assert w.observation_space.space == env.observation_space


def test_action_space_is_batched_space():
    batch_size = 4
    env = ScalarToyEnv()
    w = PooledInitVmapWrapper(env, batch_size=batch_size, pool_size=4)
    assert isinstance(w.action_space, BatchedSpace)
    assert w.action_space.batch_size == batch_size


def test_observation_space_contains_after_init_and_step():
    env = ScalarToyEnv()
    w = PooledInitVmapWrapper(env, batch_size=2, pool_size=2)
    key = jax.random.key(0)
    state, info = w.init(key)
    assert w.observation_space.contains(info.obs)
    state, info = w.step(state, jnp.array([0.1, -0.1]))
    assert w.observation_space.contains(info.obs)


def test_action_space_sample_contains():
    env = ScalarToyEnv()
    w = PooledInitVmapWrapper(env, batch_size=2, pool_size=2)
    key = jax.random.key(0)
    action = w.action_space.sample(key)
    assert w.action_space.contains(action)


@pytest.mark.parametrize("pool_size", [1, 2, 4])
def test_pool_size_parametrized(pool_size):
    env = ScalarToyEnv()
    w = PooledInitVmapWrapper(env, batch_size=2, pool_size=pool_size)
    key = jax.random.key(0)
    state, info = w.init(key)
    state, info = w.step(state, jnp.array([0.0, 0.0]))
    assert info.obs.shape == (2,)


def test_deterministic_given_same_key():
    env = ScalarToyEnv()
    w = PooledInitVmapWrapper(env, batch_size=2, pool_size=2)
    key = jax.random.key(42)
    state1, info1 = w.init(key)
    state2, info2 = w.init(key)
    assert jnp.allclose(info1.obs, info2.obs)
    state1, info1 = w.step(state1, jnp.array([0.1, 0.2]))
    state2, info2 = w.step(state2, jnp.array([0.1, 0.2]))
    assert jnp.allclose(info1.obs, info2.obs)


def test_different_keys_different_pool_states():
    """Different keys produce valid inits; shape matches. ScalarToyEnv init is deterministic (obs 0), so we only assert no crash and shape."""
    env = ScalarToyEnv()
    w = PooledInitVmapWrapper(env, batch_size=2, pool_size=2)
    s1, i1 = w.init(jax.random.key(0))
    s2, i2 = w.init(jax.random.key(1))
    assert s1.inner_state.shape == s2.inner_state.shape
    assert i1.obs.shape == i2.obs.shape == (2,)


def test_all_envs_done_simultaneously():
    env = StepCounterEnv(terminate_after=1)
    w = PooledInitVmapWrapper(env, batch_size=4, pool_size=4)
    key = jax.random.key(0)
    state, _ = w.init(key)
    action = jnp.array([0.1, 0.1, 0.1, 0.1])
    state, info = w.step(state, action)
    assert info.obs.shape == (4,)
    assert state.last_final.obs.shape == (4,)


def test_no_envs_done():
    env = ScalarToyEnv()
    w = PooledInitVmapWrapper(env, batch_size=3, pool_size=2)
    key = jax.random.key(0)
    state, _ = w.init(key)
    state, info = w.step(state, jnp.array([0.1, 0.2, 0.3]))
    assert jnp.allclose(info.obs, jnp.array([0.1, 0.2, 0.3]))
    assert jnp.all(info.final_valid == jnp.asarray([False, False, False]))


def test_batch_size_one_pool_size_one():
    env = ScalarToyEnv()
    w = PooledInitVmapWrapper(env, batch_size=1, pool_size=1)
    key = jax.random.key(0)
    state, info = w.init(key)
    assert info.obs.shape == (1,)
    state, info = w.step(state, jnp.array([0.5]))
    assert info.obs.shape == (1,)


def test_pool_size_greater_than_batch_size():
    env = ScalarToyEnv()
    w = PooledInitVmapWrapper(env, batch_size=2, pool_size=8)
    key = jax.random.key(0)
    state, info = w.init(key)
    state, info = w.step(state, jnp.array([0.0, 0.0]))
    assert info.obs.shape == (2,)


def test_init_key_advances_each_step():
    env = ScalarToyEnv()
    w = PooledInitVmapWrapper(env, batch_size=1, pool_size=1)
    key = jax.random.key(0)
    state, _ = w.init(key)
    key_before = state.init_key
    state, _ = w.step(state, jnp.array([0.0]))
    key_after = state.init_key
    assert not jnp.allclose(key_before, key_after)


def test_jit_init_step():
    env = ScalarToyEnv()
    w = PooledInitVmapWrapper(env, batch_size=2, pool_size=2)
    key = jax.random.key(0)

    @jax.jit
    def run(k):
        s, i = w.init(k)
        a = w.action_space.sample(k)
        s, i = w.step(s, a)
        return i.obs.shape

    shape = run(key)
    assert shape == (2,)


def test_jax_lax_scan_multi_step_loop():
    env = ScalarToyEnv()
    w = PooledInitVmapWrapper(env, batch_size=2, pool_size=2)
    key = jax.random.key(0)
    state, _ = w.init(key)
    keys = jax.random.split(key, 5)
    actions = jnp.stack([w.action_space.sample(k) for k in keys])

    def body(carry, action):
        next_s, info = w.step(carry, action)
        return next_s, info.obs

    final_state, obs_stack = jax.lax.scan(body, state, actions)
    assert obs_stack.shape == (5, 2)


def test_composability_with_episode_statistics_wrapper():
    env = StepCounterEnv(terminate_after=2)
    w = EpisodeStatisticsWrapper(PooledInitVmapWrapper(env, batch_size=2, pool_size=2))
    key = jax.random.key(0)
    state, _ = w.init(key)
    for _ in range(4):
        state, _ = w.step(state, jnp.array([0.1, 0.1]))
    # Stats accumulate across pool resets; reward is batched
    assert jnp.asarray(state.stats.reward).shape == (2,)


def test_composability_with_truncation_wrapper():
    env = StepCounterEnv(truncate_after=2)
    w = PooledInitVmapWrapper(
        TruncationWrapper(env, max_steps=2),
        batch_size=2,
        pool_size=2,
    )
    key = jax.random.key(0)
    state, _ = w.init(key)
    state, info = w.step(state, jnp.array([0.1, 0.1]))
    state, info = w.step(state, jnp.array([0.1, 0.1]))
    # After truncation both envs get pool inits (ScalarToyEnv init obs = 0)
    assert info.obs.shape == (2,)
    assert jnp.allclose(info.obs, jnp.zeros(2))
