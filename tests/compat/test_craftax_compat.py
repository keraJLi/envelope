"""Tests for jenv.compat.craftax_jenv module."""

import math

import jax
import jax.numpy as jnp
import pytest

pytest.importorskip("craftax")

from jenv.spaces import Continuous, Discrete
from tests.compat.contract import (
    assert_jitted_rollout_contract,
    assert_reset_step_contract,
)

pytestmark = pytest.mark.compat


@pytest.fixture(
    params=[
        "Craftax-Symbolic-v1",
        "Craftax-Classic-Symbolic-v1",
        "Craftax-Pixels-v1",
        "Craftax-Classic-Pixels-v1",
    ],
    ids=["symbolic", "classic_symbolic", "pixels", "classic_pixels"],
    scope="module",
)
def craftax_env_id(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="module")
def craftax_env(craftax_env_id: str):
    from jenv.compat.craftax_jenv import CraftaxJenv

    return CraftaxJenv.from_name(craftax_env_id)


@pytest.fixture(scope="module", autouse=True)
def _craftax_env_warmup(craftax_env, prng_key):
    """Warm up reset/step once per Craftax variant to amortize compilation."""
    env = craftax_env
    key_reset, key_step = jax.random.split(prng_key)
    state, _info = env.reset(key_reset)
    action = env.action_space.sample(key_step)
    env.step(state, action)


def _assert_obs_matches_space(obs, obs_space: Continuous):
    assert obs.dtype == obs_space.dtype
    assert obs.ndim == len(obs_space.shape)
    # Craftax pixel envs appear to have a width/height swap between returned obs
    # and the declared space. Size-based check is robust to that.
    assert obs.size == math.prod(obs_space.shape)


def _one_step(env, state, key):
    action = env.action_space.sample(key)
    return env.step(state, action)


def test_craftax_contract_smoke(craftax_env, prng_key):
    env = craftax_env

    def obs_check(obs, obs_space):
        assert isinstance(obs_space, Continuous)
        _assert_obs_matches_space(obs, obs_space)

    assert_reset_step_contract(env, key=prng_key, obs_check=obs_check)


def test_craftax_contract_scan(craftax_env, prng_key, scan_num_steps):
    assert_jitted_rollout_contract(craftax_env, key=prng_key, num_steps=scan_num_steps)


def test_spaces_exposed(craftax_env):
    assert craftax_env.action_space is not None
    assert craftax_env.observation_space is not None
    assert isinstance(craftax_env.action_space, Discrete)
    assert isinstance(craftax_env.observation_space, Continuous)


def test_time_limit_overridden_to_inf(craftax_env):
    if hasattr(craftax_env.env_params, "max_timesteps"):
        assert jnp.isposinf(jnp.asarray(craftax_env.env_params.max_timesteps))


def test_key_splitting_reset_and_step(craftax_env, prng_key):
    key = prng_key

    state, _ = craftax_env.reset(key)
    assert not jnp.array_equal(state.key, key)

    _key_step = jax.random.fold_in(prng_key, 1)
    next_state, _ = _one_step(craftax_env, state, _key_step)
    assert not jnp.array_equal(next_state.key, state.key)


class _DummyParams:
    def __init__(self, max_timesteps):
        self.max_timesteps = max_timesteps

    def replace(self, **updates):
        return _DummyParams(updates.get("max_timesteps", self.max_timesteps))


class _DummyEnv:
    def __init__(self, default_params):
        self.default_params = default_params


def test_from_name_warns_on_auto_reset(monkeypatch: pytest.MonkeyPatch):
    from jenv.compat import craftax_jenv

    def fake_make(_name: str, **_kwargs):
        return _DummyEnv(default_params=_DummyParams(max_timesteps=10))

    monkeypatch.setattr(craftax_jenv, "make_craftax_env_from_name", fake_make)

    with pytest.warns(UserWarning, match="auto_reset=True"):
        craftax_jenv.CraftaxJenv.from_name("AnyEnv", env_kwargs={"auto_reset": True})


def test_from_name_warns_on_finite_max_timesteps_when_params_provided(
    monkeypatch: pytest.MonkeyPatch,
):
    from jenv.compat import craftax_jenv

    def fake_make(_name: str, **_kwargs):
        return _DummyEnv(default_params=_DummyParams(max_timesteps=999))

    monkeypatch.setattr(craftax_jenv, "make_craftax_env_from_name", fake_make)

    with pytest.warns(UserWarning, match="finite max_timesteps"):
        provided = _DummyParams(max_timesteps=10)
        env = craftax_jenv.CraftaxJenv.from_name("AnyEnv", env_params=provided)

    # Important: user-provided env_params should not be overwritten; only warn.
    assert env.env_params is provided
    assert env.env_params.max_timesteps == 10
