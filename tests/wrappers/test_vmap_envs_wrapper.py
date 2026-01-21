import pickle

import jax
import jax.numpy as jnp
import pytest

from envelope.environment import Info
from envelope.spaces import BatchedSpace
from envelope.wrappers.vmap_envs_wrapper import VmapEnvsWrapper
from tests.wrappers.helpers import ParamEnv

# -----------------------------------------------------------------------------
# Core: Space shaping and protocol conformance
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 3])
def test_spaces_are_batched(batch_size):
    # Build a batched pytree of envs via vmap(make_env)(params)
    params = jnp.linspace(-1.0, 1.0, batch_size)
    envs = ParamEnv(offset=params)
    w = VmapEnvsWrapper(env=envs, batch_size=batch_size)
    assert isinstance(w.observation_space, BatchedSpace)
    assert isinstance(w.action_space, BatchedSpace)
    assert w.observation_space.shape == (batch_size,) + (2,)
    assert w.action_space.shape == (batch_size,) + (2,)


def test_protocol_conformance_reset_and_step():
    batch_size = 4
    params = jnp.linspace(0.0, 0.3, batch_size)
    envs = ParamEnv(offset=params)
    w = VmapEnvsWrapper(env=envs, batch_size=batch_size)
    key = jax.random.PRNGKey(0)
    state, info = w.reset(key)
    assert state is not None
    assert isinstance(info, Info)
    action = w.action_space.sample(key)
    next_state, next_info = w.step(state, action)
    assert next_state is not None
    assert isinstance(next_info, Info)


# -----------------------------------------------------------------------------
# Core: Equivalence vs manual vmap over envs and keys
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [2, 5])
def test_reset_and_step_equivalence_to_manual(batch_size):
    params = jnp.linspace(-0.5, 0.5, batch_size)
    envs = ParamEnv(offset=params)
    w = VmapEnvsWrapper(env=envs, batch_size=batch_size)
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, batch_size)
    s_w, i_w = w.reset(key)
    # Manual: vmap over (env, key)
    s_m, i_m = jax.vmap(lambda e, k: e.reset(k))(envs, keys)
    assert jax.tree_util.tree_all(
        jax.tree.map(lambda a, b: jnp.allclose(a, b), s_w, s_m)
    )
    assert jnp.allclose(i_w.obs, i_m.obs)
    action = w.action_space.sample(key)
    s_w2, i_w2 = w.step(s_w, action)
    s_m2, i_m2 = jax.vmap(lambda e, s, a: e.step(s, a))(envs, s_m, action)
    assert jax.tree_util.tree_all(
        jax.tree.map(lambda a, b: jnp.allclose(a, b), s_w2, s_m2)
    )
    assert jnp.allclose(i_w2.obs, i_m2.obs)
    assert jnp.allclose(i_w2.reward, i_m2.reward)


# -----------------------------------------------------------------------------
# Core: Error paths and serialization
# -----------------------------------------------------------------------------


def test_reset_raises_on_wrong_key_batch_dim():
    batch_size = 3
    params = jnp.linspace(0.0, 1.0, batch_size)
    envs = ParamEnv(offset=params)
    w = VmapEnvsWrapper(env=envs, batch_size=batch_size)
    bad_keys = jax.random.split(jax.random.PRNGKey(0), batch_size + 1)
    with pytest.raises(ValueError):
        _ = w.reset(bad_keys)


def test_pickle_serialization_of_state_and_info():
    batch_size = 3
    params = jnp.linspace(0.0, 1.0, batch_size)
    envs = ParamEnv(offset=params)
    w = VmapEnvsWrapper(env=envs, batch_size=batch_size)
    key = jax.random.PRNGKey(0)
    state, info = w.reset(key)
    blob = pickle.dumps((state, info))
    s2, i2 = pickle.loads(blob)
    assert jnp.allclose(s2, state)
    assert jnp.allclose(i2.obs, info.obs)


# -----------------------------------------------------------------------------
# Optional: Property-based sampling (offsets, batch sizes)
# -----------------------------------------------------------------------------


try:
    from hypothesis import given, settings
    from hypothesis import strategies as st
except Exception:  # pragma: no cover - optional dependency
    pytest.skip("hypothesis not installed", allow_module_level=True)


@given(
    batch_size=st.integers(min_value=1, max_value=5),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(deadline=None, max_examples=20)
def test_prop_reset_step_shapes(batch_size, seed):
    params = jnp.linspace(-1.0, 1.0, batch_size)
    envs = ParamEnv(offset=params)
    w = VmapEnvsWrapper(env=envs, batch_size=batch_size)
    key = jax.random.PRNGKey(seed)
    state, info = w.reset(key)
    assert state is not None and info is not None
    assert info.obs.shape == (batch_size, 2)
    action = w.action_space.sample(key)
    s2, i2 = w.step(state, action)
    assert i2.obs.shape == (batch_size, 2)


def test_param_effect_applies_per_env_no_cross_mix():
    batch_size = 4
    offsets = jnp.array([0.0, 1.0, -1.0, 2.0], dtype=jnp.float32)
    envs = ParamEnv(offset=offsets)
    w = VmapEnvsWrapper(env=envs, batch_size=batch_size)
    key = jax.random.PRNGKey(0)
    s, i = w.reset(key)
    s2, i2 = w.step(s, jnp.zeros((batch_size, 2), dtype=jnp.float32))
    expected = i.obs + offsets.reshape((batch_size, 1))
    assert jnp.allclose(i2.obs, expected)


def test_unwrapped_returns_base_unwrapped():
    offsets = jnp.array([0.0, 1.0], dtype=jnp.float32)
    envs = ParamEnv(offset=offsets)
    w = VmapEnvsWrapper(env=envs, batch_size=2)
    assert w.unwrapped is envs.unwrapped
