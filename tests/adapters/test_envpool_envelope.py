"""Tests for envelope.adapters.envpool_envelope module."""

# ruff: noqa: E402

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.adapters

pytest.importorskip("envpool")

from envelope.adapters.envpool_envelope import EnvPoolEnvelope, EnvPoolState
from envelope.spaces import BatchedSpace, Continuous, Discrete
from tests.contract import (
    assert_info_has_contract_fields,
    assert_obs_matches_space,
)

BATCH_SIZE = 4


@pytest.fixture(scope="module")
def envpool_env():
    return EnvPoolEnvelope.from_name(
        "CartPole-v1", env_kwargs={"batch_size": BATCH_SIZE, "num_envs": BATCH_SIZE}
    )


@pytest.fixture(scope="module", autouse=True)
def _envpool_warmup(envpool_env, prng_key):
    """Warm up init/step once to amortize compilation."""
    state, _info = envpool_env.init(prng_key)
    action = envpool_env.action_space.sample(prng_key)
    envpool_env.step(state, action)


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


def test_envpool_contract_smoke(prng_key, envpool_env):
    """Batched variant of the reset/step contract (envpool is inherently batched)."""
    key_reset, key_action = jax.random.split(prng_key)

    state, info = envpool_env.init(key_reset)
    assert_info_has_contract_fields(info)
    assert jnp.asarray(info.reward).shape == (BATCH_SIZE,)
    assert jnp.all(jnp.isfinite(jnp.asarray(info.reward)))
    assert jnp.all(jnp.asarray(info.reward) == 0.0)
    assert not jnp.any(jnp.asarray(info.terminated))
    assert not jnp.any(jnp.asarray(info.truncated))
    assert_obs_matches_space(info.obs, envpool_env.observation_space)

    action = envpool_env.action_space.sample(key_action)
    assert envpool_env.action_space.contains(action)
    next_state, next_info = envpool_env.step(state, action)
    assert next_state is not None
    assert_info_has_contract_fields(next_info)
    assert jnp.asarray(next_info.reward).shape == (BATCH_SIZE,)
    assert jnp.all(jnp.isfinite(jnp.asarray(next_info.reward)))
    assert_obs_matches_space(next_info.obs, envpool_env.observation_space)

    # Pytree structure and leaf shapes must match between init and step
    assert jax.tree.structure(info) == jax.tree.structure(next_info)
    init_shapes = [jnp.asarray(x).shape for x in jax.tree.leaves(info)]
    step_shapes = [jnp.asarray(x).shape for x in jax.tree.leaves(next_info)]
    assert init_shapes == step_shapes

    assert jax.tree.structure(state) == jax.tree.structure(next_state)
    init_state_shapes = [jnp.asarray(x).shape for x in jax.tree.leaves(state)]
    step_state_shapes = [jnp.asarray(x).shape for x in jax.tree.leaves(next_state)]
    assert init_state_shapes == step_state_shapes


def test_envpool_contract_scan(prng_key, envpool_env, scan_num_steps):
    """Batched variant of the jitted rollout contract."""
    reset_jit = jax.jit(envpool_env.init)
    step_jit = jax.jit(envpool_env.step)

    key_reset, key_rollout = jax.random.split(prng_key)
    state, _ = reset_jit(key_reset)
    action_keys = jax.random.split(key_rollout, scan_num_steps)
    actions = jax.vmap(envpool_env.action_space.sample)(action_keys)
    final_state, infos = jax.lax.scan(step_jit, state, actions)

    assert final_state is not None
    assert_info_has_contract_fields(infos)
    assert infos.reward.shape == (scan_num_steps, BATCH_SIZE)
    assert jnp.all(jnp.isfinite(jnp.asarray(infos.reward)))


# ---------------------------------------------------------------------------
# Space conversion tests
# ---------------------------------------------------------------------------


def test_action_space_is_batched_discrete(envpool_env):
    space = envpool_env.action_space
    assert isinstance(space, BatchedSpace)
    assert space.batch_size == BATCH_SIZE
    assert isinstance(space.space, Discrete)


def test_observation_space_is_batched_continuous(envpool_env):
    space = envpool_env.observation_space
    assert isinstance(space, BatchedSpace)
    assert space.batch_size == BATCH_SIZE
    assert isinstance(space.space, Continuous)


# ---------------------------------------------------------------------------
# State structure tests
# ---------------------------------------------------------------------------


def test_state_is_envpool_state(envpool_env, prng_key):
    state, _info = envpool_env.init(prng_key)
    assert isinstance(state, EnvPoolState)
    assert hasattr(state, "handle")
    assert hasattr(state, "last_final")


def test_state_handle_preserved_across_steps(envpool_env, prng_key):
    """XLA handles are opaque ordering tokens; verify they are present and step progresses."""
    state, info0 = envpool_env.init(prng_key)
    assert state.handle is not None
    action = envpool_env.action_space.sample(prng_key)
    next_state, info1 = envpool_env.step(state, action)
    assert next_state.handle is not None
    # Observations should generally differ between init and a step
    assert not jnp.array_equal(info0.obs, info1.obs)


# ---------------------------------------------------------------------------
# Info and final field tests
# ---------------------------------------------------------------------------


def test_init_info_has_final_field(envpool_env, prng_key):
    _state, info = envpool_env.init(prng_key)
    assert hasattr(info, "final")


def test_init_final_is_nan_placeholder(envpool_env, prng_key):
    _state, info = envpool_env.init(prng_key)
    assert jnp.all(jnp.isnan(jnp.asarray(info.final.reward)))
    assert jnp.all(jnp.isnan(jnp.asarray(info.final.obs)))


def test_step_info_has_final_field(envpool_env, prng_key):
    state, _info = envpool_env.init(prng_key)
    action = envpool_env.action_space.sample(prng_key)
    _state, info = envpool_env.step(state, action)
    assert hasattr(info, "final")


def test_info_extras_forwarded(envpool_env, prng_key):
    _state, info = envpool_env.init(prng_key)
    assert hasattr(info, "elapsed_step")


# ---------------------------------------------------------------------------
# Autoreset / final field semantics
# ---------------------------------------------------------------------------


def test_final_captures_terminal_info_on_done(envpool_env, prng_key):
    """Run steps until at least one env in the batch is done; check final."""
    state, _info = envpool_env.init(prng_key)
    max_steps = 500
    for i in range(max_steps):
        action = envpool_env.action_space.sample(jax.random.fold_in(prng_key, i))
        state, info = envpool_env.step(state, action)
        done = jnp.asarray(info.terminated) | jnp.asarray(info.truncated)
        if jnp.any(done):
            # For envs that are done, final should snapshot the terminal info
            done_mask = done.astype(jnp.float32)
            final_reward = jnp.asarray(info.final.reward)
            # At least one done env should have non-NaN final reward
            assert jnp.any(done_mask * jnp.isfinite(final_reward))
            return
    pytest.fail(f"No episode ended within {max_steps} steps")


def test_final_carries_over_after_done(envpool_env, prng_key):
    """After done, the next step (auto-reset) should still carry final from the terminal step."""
    state, _info = envpool_env.init(prng_key)
    max_steps = 500
    for i in range(max_steps):
        action = envpool_env.action_space.sample(jax.random.fold_in(prng_key, i))
        state, info = envpool_env.step(state, action)
        done = jnp.asarray(info.terminated) | jnp.asarray(info.truncated)
        if jnp.any(done):
            # Take one more step (the auto-reset step)
            action2 = envpool_env.action_space.sample(
                jax.random.fold_in(prng_key, max_steps + i)
            )
            state2, info2 = envpool_env.step(state, action2)
            # final should carry over the terminal info from the previous step
            final_reward = jnp.asarray(info2.final.reward)
            # The envs that were done should have non-NaN final reward
            done_mask = done.astype(jnp.float32)
            assert jnp.any(done_mask * jnp.isfinite(final_reward))
            return
    pytest.fail(f"No episode ended within {max_steps} steps")


# ---------------------------------------------------------------------------
# Constructor test
# ---------------------------------------------------------------------------


def test_from_name_stores_xla_functions(envpool_env):
    assert callable(envpool_env._xla_recv)
    assert callable(envpool_env._xla_step)
    assert envpool_env._xla_handle0 is not None
