"""Shared contract helpers for adapters and wrappers.

These functions enforce a consistent baseline across environments and wrappers:
- reset/step return (state, info) with Info fields present
- reward is scalar-ish and finite
- action sampling is valid for action_space
- observation is validated via a provided callback (contains + dtype)
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from envelope.environment import Info

ObsCheck = Callable[[object, object], None]


def assert_obs_matches_space(obs: object, obs_space: object) -> None:
    assert obs_space.contains(obs)
    space_dtype = getattr(obs_space, "dtype", None)
    if space_dtype is None:
        return

    try:
        matches = jax.tree.map(
            lambda o, dt: jnp.asarray(o).dtype == dt, obs, space_dtype
        )
        assert jnp.all(jnp.array(jax.tree.leaves(matches)))
    except Exception:
        assert jnp.asarray(obs).dtype == space_dtype


def assert_info_has_contract_fields(info: Info) -> None:
    # Protocol adherence: Info is a runtime-checkable Protocol.
    # This is the strongest, most consistent check we can do across wrappers.
    assert isinstance(info, Info)
    assert hasattr(info, "obs")
    assert hasattr(info, "reward")
    assert hasattr(info, "terminated")
    assert hasattr(info, "truncated")


def assert_reset_step_contract(
    env,
    *,
    key: jax.Array,
    obs_check: ObsCheck,
) -> None:
    # Derive subkeys so callers can pass a single fixture key.
    key_reset, key_action = jax.random.split(key)

    state, info = env.init(key_reset)
    assert_info_has_contract_fields(info)
    assert jnp.asarray(info.reward).shape == ()
    assert jnp.isfinite(jnp.asarray(info.reward))
    assert info.reward == 0.0
    assert not bool(jnp.asarray(info.terminated))
    assert not bool(jnp.asarray(info.truncated))

    obs_check(info.obs, env.observation_space)

    action = env.action_space.sample(key_action)
    assert env.action_space.contains(action)
    next_state, next_info = env.step(state, action)
    assert next_state is not None
    assert_info_has_contract_fields(next_info)
    assert jnp.asarray(next_info.reward).shape == ()
    assert jnp.isfinite(jnp.asarray(next_info.reward))
    obs_check(next_info.obs, env.observation_space)


def assert_jitted_rollout_contract(
    env,
    *,
    key: jax.Array,
    num_steps: int,
) -> None:
    """Contract: rollout works under `jax.jit` and produces a valid `Info` container."""
    reset_jit = jax.jit(env.init)
    step_jit = jax.jit(env.step)

    key_reset, key_rollout = jax.random.split(key)
    state, _ = reset_jit(key_reset)
    action_keys = jax.random.split(key_rollout, num_steps)
    actions = jax.vmap(env.action_space.sample)(action_keys)
    final_state, infos = jax.lax.scan(step_jit, state, actions)

    assert final_state is not None
    assert_info_has_contract_fields(infos)
    assert infos.reward.shape == (num_steps,)
    assert jnp.all(jnp.isfinite(jnp.asarray(infos.reward)))
