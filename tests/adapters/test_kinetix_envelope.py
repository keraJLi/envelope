"""Tests for envelope.adapters.kinetix_envelope module."""

# ruff: noqa: E402

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.adapters

pytest.importorskip("kinetix")

from envelope.adapters.kinetix_envelope import KinetixEnvelope
from envelope.environment import Info
from envelope.spaces import Continuous
from tests.contract import (
    assert_jitted_rollout_contract,
    assert_obs_matches_space,
    assert_reset_step_contract,
)


@pytest.fixture(scope="module")
def kinetix_random_env():
    """Create a single random Kinetix env for the whole module."""
    return KinetixEnvelope.create_random()


@pytest.fixture(scope="module", autouse=True)
def kinetix_random_env_warmup(kinetix_random_env, prng_key):
    """Warm up reset/step/scan once to amortize compilation cost."""
    env = kinetix_random_env
    key_reset, key_step, key_scan = jax.random.split(prng_key, 3)
    state, _info = env.init(key_reset)
    action = env.action_space.sample(key_step)
    _state2, _info2 = env.step(state, action)

    # Small scan warmup (kept tiny to avoid adding more compile work than needed).
    num_steps = 3
    action_keys = jax.random.split(key_scan, num_steps)
    actions = jax.vmap(env.action_space.sample)(action_keys)

    def step_fn(s, a):
        return env.step(s, a)

    jax.lax.scan(step_fn, state, actions)


def _create_kinetix_env(name: str = "random", **kwargs):
    """Helper to create a KinetixEnvelope wrapper."""
    if name == "random":
        return KinetixEnvelope.create_random(**kwargs)
    return KinetixEnvelope.create_from_size(name, **kwargs)


def test_kinetix_contract_smoke(prng_key, kinetix_random_env):
    assert_reset_step_contract(
        kinetix_random_env, key=prng_key, obs_check=assert_obs_matches_space
    )


def test_kinetix_contract_scan(prng_key, kinetix_random_env, scan_num_steps):
    assert_jitted_rollout_contract(
        kinetix_random_env, key=prng_key, num_steps=scan_num_steps
    )


def test_action_space_is_continuous_by_default(kinetix_random_env):
    env = kinetix_random_env
    assert isinstance(env.action_space, Continuous)


@pytest.mark.parametrize("size", ["s", "m", "l"])
def test_create_from_size_smoke(prng_key, size):
    env = KinetixEnvelope.create_from_size(size)

    state, info = env.init(prng_key)
    assert state is not None
    assert isinstance(info, Info)
    assert env.observation_space.contains(info.obs)


def test_create_random_with_auto_reset_warning(prng_key):
    with pytest.warns(
        UserWarning,
        match="Creating a KinetixEnvelope with auto_reset=True is not recommended",
    ):
        env = _create_kinetix_env("random", auto_reset=True)

    state, info = env.init(prng_key)
    assert state is not None
    assert isinstance(info, Info)


def test_key_splitting(kinetix_random_env, prng_key):
    env = kinetix_random_env
    key = prng_key
    state, _info = env.init(key)
    assert hasattr(state, "key")
    assert not jnp.array_equal(state.key, key)

    action = env.action_space.sample(jax.random.fold_in(prng_key, 1))
    next_state, _ = env.step(state, action)
    assert not jnp.array_equal(next_state.key, state.key)


def test_from_size_step_produces_finite_reward(prng_key):
    """Test that stepping a size-based env produces finite rewards."""
    env = KinetixEnvelope.create_from_size("s")
    reset_key, action_key = jax.random.split(prng_key, 2)

    state, info = env.init(reset_key)
    assert state is not None
    assert isinstance(info, Info)
    assert info.obs.shape == env.observation_space.shape

    action = env.action_space.sample(action_key)
    next_state, next_info = env.step(state, action)
    assert next_state is not None
    assert isinstance(next_info, Info)
    assert next_info.obs.shape == env.observation_space.shape
    assert jnp.all(jnp.isfinite(jnp.asarray(next_info.reward)))


def test_from_name_rejects_unknown_env_kwargs():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        KinetixEnvelope.from_name("s", env_kwargs={"unknown": 1})


def test_from_name_rejects_invalid_name():
    with pytest.raises(ValueError, match="Invalid env_name"):
        KinetixEnvelope.from_name("s/h4_thrust_aim")
