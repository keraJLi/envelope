import pickle

import jax
import jax.numpy as jnp
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from envelope.environment import Info
from envelope.spaces import BatchedSpace
from envelope.wrappers.observation_normalization_wrapper import (
    ObservationNormalizationWrapper,
)
from envelope.wrappers.vmap_wrapper import VmapWrapper
from tests.wrappers.helpers import FlagDoneEnv, ScalarToyEnv, VectorToyEnv

# -----------------------------------------------------------------------------
# Core: Space shaping and protocol conformance
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_spaces_are_batched(batch_size):
    env = ScalarToyEnv()
    wrapped = VmapWrapper(env, batch_size=batch_size)
    assert isinstance(wrapped.observation_space, BatchedSpace)
    assert isinstance(wrapped.action_space, BatchedSpace)
    assert (
        wrapped.observation_space.shape == (batch_size,) + env.observation_space.shape
    )
    assert wrapped.action_space.shape == (batch_size,) + env.action_space.shape


def test_protocol_conformance_reset_and_step():
    env = ScalarToyEnv()
    wrapped = VmapWrapper(env, batch_size=3)
    key = jax.random.key(0)
    state, info = wrapped.init(key)
    assert isinstance(info, Info)
    assert state is not None
    # Action space contains batched action
    action = wrapped.action_space.sample(key)
    assert wrapped.action_space.contains(action)
    next_state, next_info = wrapped.step(state, action)
    assert isinstance(next_info, Info)
    assert next_state is not None
    assert wrapped.observation_space.contains(next_info.obs)


# -----------------------------------------------------------------------------
# Core: Reset/step equivalence vs manual vmap
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [2, 4])
def test_reset_equivalence_to_manual_vmap(batch_size):
    env = ScalarToyEnv()
    w = VmapWrapper(env, batch_size=batch_size)
    single_key = jax.random.key(0)
    keys = jax.random.split(single_key, batch_size)
    s_wrapped, i_wrapped = w.init(single_key)
    s_manual, i_manual = jax.vmap(env.init)(keys)
    assert jax.tree_util.tree_all(
        jax.tree.map(lambda a, b: jnp.allclose(a, b), s_wrapped, s_manual)
    )
    assert jax.tree_util.tree_all(
        jax.tree.map(lambda a, b: jnp.allclose(a, b), i_wrapped.obs, i_manual.obs)
    )


@pytest.mark.parametrize("dim,batch_size", [(1, 3), (4, 2)])
def test_step_equivalence_to_manual_vmap(dim, batch_size):
    env = VectorToyEnv(dim)
    w = VmapWrapper(env, batch_size=batch_size)
    key = jax.random.key(0)
    s0, _ = w.init(key)
    action = w.action_space.sample(key)
    s1, info1 = w.step(s0, action)
    # Manual baseline
    keys = jax.random.split(key, batch_size)
    s0_m, _ = jax.vmap(env.init)(keys)
    s1_m, info1_m = jax.vmap(env.step)(s0_m, action)
    assert jax.tree_util.tree_all(
        jax.tree.map(lambda a, b: jnp.allclose(a, b), s1, s1_m)
    )
    assert jnp.allclose(info1.reward, info1_m.reward)
    assert jnp.allclose(info1.obs, info1_m.obs)


# -----------------------------------------------------------------------------
# Core: Error paths
# -----------------------------------------------------------------------------


def test_reset_raises_on_wrong_key_batch_dim():
    env = ScalarToyEnv()
    w = VmapWrapper(env, batch_size=4)
    bad_keys = jax.random.split(jax.random.key(0), 3)  # mismatch batch
    with pytest.raises(ValueError):
        _ = w.init(bad_keys)


def test_step_raises_on_action_shape_mismatch():
    env = VectorToyEnv(dim=3)
    w = VmapWrapper(env, batch_size=2)
    key = jax.random.key(0)
    state, _ = w.init(key)
    bad_action = jnp.ones((3,), dtype=jnp.float32)  # missing batch dim
    with pytest.raises(Exception):
        _ = w.step(state, bad_action)


# -----------------------------------------------------------------------------
# Core: Composability with shared normalization
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [2, 5])
def test_normalization_must_wrap_vmap(batch_size):
    base = VectorToyEnv(dim=3)
    with pytest.raises(ValueError, match="ObservationNormalizationWrapper.*wrap"):
        VmapWrapper(
            ObservationNormalizationWrapper(base),
            batch_size=batch_size,
        )

    wrapper = ObservationNormalizationWrapper(
        VmapWrapper(base, batch_size=batch_size)
    )
    state, info = wrapper.init(jax.random.key(42))

    assert info.obs.shape == (batch_size, 3)
    assert state.rmv_state.mean.shape == (3,)


def test_nested_vmaps_equivalence_reset_and_step():
    base = VectorToyEnv(dim=2)
    outer = VmapWrapper(VmapWrapper(base, batch_size=3), batch_size=2)  # (2,3)
    flat = VmapWrapper(base, batch_size=6)  # (6,)
    key = jax.random.key(0)
    s_outer, i_outer = outer.init(key)
    keys = jax.random.split(key, 6)
    s_flat_m, i_flat_m = jax.vmap(base.init)(keys)
    # Reshape outer to flat
    i_outer_flat = i_outer.obs.reshape((6, 2))
    assert jnp.allclose(i_outer_flat, i_flat_m.obs)
    action = flat.action_space.sample(key)
    s_flat, _ = flat.init(key)
    s_flat2, i_flat2 = flat.step(s_flat, action)
    s_outer2, i_outer2 = outer.step(s_outer, action.reshape((2, 3, 2)))
    i_outer2_flat = i_outer2.obs.reshape((6, 2))
    assert jnp.allclose(i_outer2_flat, i_flat2.obs)


def test_termination_truncation_propagation():
    flags = jnp.array([True, False, True], dtype=bool)
    w = VmapWrapper(FlagDoneEnv(flags=flags), batch_size=3)
    s, _ = w.init(jax.random.key(0))
    _, info = w.step(s, jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32))
    assert jnp.all(info.terminated == flags)
    assert jnp.all(info.truncated == ~flags)


def test_state_slicing_matches_single_env_run():
    base = VectorToyEnv(dim=3)
    w = VmapWrapper(base, batch_size=4)
    key = jax.random.key(0)
    s_b, i_b = w.init(key)
    idx = 2
    # Run single env with the same per-env key used by the wrapper for index idx
    keys = jax.random.split(key, 4)
    s0, _ = base.init(keys[idx])
    a_single = base.action_space.sample(keys[idx])
    s1_single, i1_single = base.step(s0, a_single)
    # Run batched and slice
    act = w.action_space.sample(key)
    act = act.at[idx].set(a_single)
    s1_b, i1_b = w.step(s_b, act)
    assert jnp.allclose(i1_b.obs[idx], i1_single.obs)


# -----------------------------------------------------------------------------
# Core: Transform compatibility and serialization
# -----------------------------------------------------------------------------


def test_jit_compatibility_smoke():
    env = VectorToyEnv(dim=2)
    w = VmapWrapper(env, batch_size=3)
    key = jax.random.key(0)

    @jax.jit
    def run_once(k, a):
        s, _ = w.init(k)
        ns, inf = w.step(s, a)
        return ns, inf.obs

    # Sample outside jit to avoid tracing space construction in __post_init__
    act = w.action_space.sample(key)
    ns, obs = run_once(key, act)
    assert ns is not None
    assert obs.shape == (3, 2)


def test_pickle_serialization_of_state_and_info():
    env = ScalarToyEnv()
    w = VmapWrapper(env, batch_size=2)
    key = jax.random.key(0)
    state, info = w.init(key)
    payload = (state, info)
    blob = pickle.dumps(payload)
    loaded_state, loaded_info = pickle.loads(blob)
    # Basic equality on contents
    assert jnp.allclose(loaded_state, state)
    assert jnp.allclose(loaded_info.obs, info.obs)


# -----------------------------------------------------------------------------
# Optional: Property-based sampling
# -----------------------------------------------------------------------------


@given(
    batch_size=st.integers(min_value=1, max_value=4),
    dim=st.integers(min_value=1, max_value=5),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(deadline=None, max_examples=20)
def test_prop_shapes_and_contains(batch_size, dim, seed):
    env = VectorToyEnv(dim=dim)
    w = VmapWrapper(env, batch_size=batch_size)
    key = jax.random.key(seed)
    state, info = w.init(key)
    assert state is not None
    assert isinstance(info, Info)
    assert w.observation_space.contains(info.obs)
    action = w.action_space.sample(key)
    assert w.action_space.contains(action)
    next_state, next_info = w.step(state, action)
    assert w.observation_space.contains(next_info.obs)
