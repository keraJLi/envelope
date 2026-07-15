"""Tests for envelope.adapters.brax_envelope module."""

# ruff: noqa: E402

from copy import deepcopy

import jax
import pytest

pytestmark = pytest.mark.adapters

pytest.importorskip("brax")

from brax.envs import Wrapper as BraxWrapper

from envelope.adapters.brax_envelope import BraxEnvelope
from tests.contract import (
    assert_jitted_rollout_contract,
    assert_obs_matches_space,
    assert_reset_step_contract,
)


@pytest.fixture(scope="module")
def brax_fast_env():
    return BraxEnvelope.from_name("fast")


@pytest.fixture(scope="module", autouse=True)
def _brax_fast_env_warmup(brax_fast_env, prng_key):
    """Warm up reset/step once to amortize compilation."""
    env = brax_fast_env
    key_reset, key_step = jax.random.split(prng_key)
    state, _info = env.init(key_reset)
    action = env.action_space.sample(key_step)
    env.step(state, action)


def test_brax_contract_smoke(prng_key, brax_fast_env):
    assert_reset_step_contract(
        brax_fast_env, key=prng_key, obs_check=assert_obs_matches_space
    )


def test_brax_contract_scan(prng_key, brax_fast_env, scan_num_steps):
    assert_jitted_rollout_contract(
        brax_fast_env, key=prng_key, num_steps=scan_num_steps
    )


def test_brax_info_preserves_brax_fields_on_reset(brax_fast_env, prng_key):
    """Brax-specific: extra Brax state fields are preserved on reset."""
    env = brax_fast_env
    key = prng_key

    state, info = env.init(key)

    # Check extra Brax state fields are preserved
    # Brax state typically has: obs, reward, done, metrics, info
    assert hasattr(info.backend, "done")
    assert hasattr(info.backend, "metrics")

    # Verify state fields match what was returned
    assert state is not None
    assert hasattr(state, "obs")


def test_brax_terminated_matches_done_on_step(brax_fast_env, prng_key):
    """Brax-specific: wrapper exposes underlying `done` and maps it to `terminated`."""
    env = brax_fast_env
    key_reset, key_action = jax.random.split(prng_key)
    state, _info = env.init(key_reset)
    action = env.action_space.sample(key_action)
    _next_state, info = env.step(state, action)
    assert hasattr(info.backend, "done")
    assert info.terminated == info.backend.done


def test_from_name_with_auto_reset_error():
    """Test that from_name raises ValueError when using auto_reset."""
    with pytest.raises(ValueError, match="Cannot override 'auto_reset' directly"):
        BraxEnvelope.from_name("fast", env_kwargs={"auto_reset": True})


def test_pre_wrapped_brax_environment_is_rejected():
    from brax.envs import create as brax_create

    # Create a base Brax environment
    base_env = brax_create("fast", episode_length=None, auto_reset=False)

    # Create a simple wrapper
    class SimpleWrapper(BraxWrapper):
        def init(self, rng):
            return self.env.init(rng)

        def step(self, state, action):
            return self.env.step(state, action)

    wrapped_env = SimpleWrapper(base_env)

    with pytest.raises(ValueError, match="Pre-wrapped Brax environments"):
        BraxEnvelope(brax_env=wrapped_env)


def test_deepcopy_warning(brax_fast_env, prng_key):
    """Test that deepcopy raises a warning and returns shallow copy."""
    env = brax_fast_env

    # Call deepcopy and verify warning is raised
    with pytest.warns(
        RuntimeWarning, match="Trying to deepcopy.*shallow copy is returned"
    ):
        copied_env = deepcopy(env)

    # Verify shallow copy is returned
    assert copied_env is not None

    # Verify the copied environment is usable
    key = prng_key
    state, info = copied_env.init(key)
    assert state is not None
