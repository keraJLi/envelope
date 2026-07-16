"""Tests for envelope.adapters.craftax_envelope module."""

# ruff: noqa: E402

import warnings

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.adapters

pytest.importorskip("craftax")

from envelope.adapters import craftax_envelope
from envelope.spaces import Continuous, Discrete
from envelope.struct import Container
from tests.contract import (
    assert_jitted_rollout_contract,
    assert_obs_matches_space,
    assert_reset_step_contract,
)


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
    from envelope.adapters.craftax_envelope import CraftaxEnvelope

    return CraftaxEnvelope.from_name(craftax_env_id)


@pytest.fixture(scope="module", autouse=True)
def _craftax_env_warmup(craftax_env, prng_key):
    """Warm up reset/step once per Craftax variant to amortize compilation."""
    env = craftax_env
    key_reset, key_step = jax.random.split(prng_key)
    state, _info = env.init(key_reset)
    action = env.action_space.sample(key_step)
    env.step(state, action)


def _one_step(env, state, key):
    action = env.action_space.sample(key)
    return env.step(state, action)


def test_craftax_contract_smoke(craftax_env, prng_key):
    assert_reset_step_contract(
        craftax_env, key=prng_key, obs_check=assert_obs_matches_space
    )


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

    state, _ = craftax_env.init(key)
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


def test_from_name_captures_default_before_disabling_time_limit(monkeypatch):
    dummy_env = _DummyEnv(_DummyParams(max_timesteps=100))
    monkeypatch.setattr(
        craftax_envelope,
        "make_craftax_env_from_name",
        lambda *_args, **_kwargs: dummy_env,
    )
    monkeypatch.setattr(
        craftax_envelope,
        "_probe_gymnaxlike_info_placeholder",
        lambda _env, _params: Container(),
    )

    env = craftax_envelope.CraftaxEnvelope.from_name("AnyEnv")

    assert jnp.isposinf(env.env_params.max_timesteps)
    assert env.default_max_steps == 100


def test_from_name_preserves_explicit_auto_reset(monkeypatch):
    captured_kwargs = {}
    dummy_env = _DummyEnv(_DummyParams(max_timesteps=100))

    def make_env(_name, **kwargs):
        captured_kwargs.update(kwargs)
        return dummy_env

    monkeypatch.setattr(craftax_envelope, "make_craftax_env_from_name", make_env)
    monkeypatch.setattr(
        craftax_envelope,
        "_probe_gymnaxlike_info_placeholder",
        lambda _env, _params: Container(),
    )

    supplied_params = _DummyParams(max_timesteps=17)
    with pytest.warns(UserWarning, match="backend settings"):
        env = craftax_envelope.CraftaxEnvelope.from_name(
            "AnyEnv",
            env_params=supplied_params,
            env_kwargs={"auto_reset": True},
        )

    assert captured_kwargs["auto_reset"] is True
    assert env.craftax_env is dummy_env
    assert env.env_params is supplied_params
    assert env.default_max_steps == 17


def test_from_name_preserves_nonfinite_explicit_horizon_without_warning(monkeypatch):
    dummy_env = _DummyEnv(_DummyParams(max_timesteps=100))
    monkeypatch.setattr(
        craftax_envelope,
        "make_craftax_env_from_name",
        lambda *_args, **_kwargs: dummy_env,
    )
    monkeypatch.setattr(
        craftax_envelope,
        "_probe_gymnaxlike_info_placeholder",
        lambda _env, _params: Container(),
    )
    supplied_params = _DummyParams(max_timesteps=jnp.inf)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        env = craftax_envelope.CraftaxEnvelope.from_name(
            "AnyEnv", env_params=supplied_params
        )

    assert env.env_params is supplied_params
    assert env.default_max_steps is None
