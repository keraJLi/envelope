"""Tests for envelope.adapters.craftax_envelope module."""

# ruff: noqa: E402

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.adapters

pytest.importorskip("craftax")

from craftax.craftax.envs.craftax_pixels_env import (
    CraftaxPixelsEnv,
    CraftaxPixelsEnvNoAutoReset,
)
from craftax.craftax_classic.envs.craftax_pixels_env import (
    CraftaxClassicPixelsEnv,
    CraftaxClassicPixelsEnvNoAutoReset,
)

from envelope.spaces import Continuous, Discrete
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
    failing_envs = (
        CraftaxPixelsEnv,
        CraftaxClassicPixelsEnv,
        CraftaxPixelsEnvNoAutoReset,
        CraftaxClassicPixelsEnvNoAutoReset,
    )
    if isinstance(craftax_env.craftax_env, failing_envs):
        pytest.xfail(
            "Craftax currently returns the wrong observation space. "
            "See https://github.com/MichaelTMatthews/Craftax/pull/49"
        )
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


def test_from_name_errors_on_auto_reset():
    from envelope.adapters.craftax_envelope import CraftaxEnvelope

    with pytest.raises(ValueError, match="Cannot override 'auto_reset' directly"):
        CraftaxEnvelope.from_name("AnyEnv", env_kwargs={"auto_reset": True})
