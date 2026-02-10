import jax
import jax.numpy as jnp

from envelope.spaces import BatchedSpace, Continuous, Discrete
from envelope.wrappers.continuous_observation_wrapper import (
    ContinuousObservationWrapper,
    to_continuous,
    to_float,
)
from envelope.wrappers.vmap_wrapper import VmapWrapper
from tests.wrappers.helpers import IntObsEnv, PyTreeObsEnv, ScalarToyEnv


def test_init_reset_step_cast_discrete_obs_to_float32():
    env = IntObsEnv()
    w = ContinuousObservationWrapper(env)
    key = jax.random.key(0)
    state, info = w.init(key)
    assert info.obs.dtype == jnp.float32
    state, info = w.reset(state, key)
    assert info.obs.dtype == jnp.float32
    assert w.observation_space.contains(info.obs)
    state, info = w.step(state, jnp.array(0, dtype=jnp.int32))
    assert info.obs.dtype == jnp.float32


def test_float32_obs_passes_through():
    env = ScalarToyEnv()
    w = ContinuousObservationWrapper(env)
    key = jax.random.key(0)
    state, info = w.init(key)
    assert info.obs.dtype == jnp.float32
    assert jnp.allclose(info.obs, 0.0)
    state, info = w.step(state, jnp.array(0.5))
    assert info.obs.dtype == jnp.float32
    assert jnp.allclose(info.obs, 0.5)


def test_init_equivalence_to_manual_to_float():
    env = IntObsEnv()
    w = ContinuousObservationWrapper(env)
    key = jax.random.key(0)
    _, info_w = w.init(key)
    _, info_raw = env.init(key)
    manual = to_float(info_raw.obs)
    assert jnp.allclose(jnp.asarray(info_w.obs), jnp.asarray(manual))
    assert info_w.obs.dtype == jnp.float32


def test_pytree_obs_all_leaves_cast():
    env = PyTreeObsEnv(shapes={"a": (2,), "b": (3,)})
    w = ContinuousObservationWrapper(env)
    key = jax.random.key(0)
    state, info = w.init(key)
    assert info.obs["a"].dtype == jnp.float32
    assert info.obs["b"].dtype == jnp.float32


def test_observation_space_discrete_to_continuous():
    env = IntObsEnv()
    w = ContinuousObservationWrapper(env)
    space = w.observation_space
    assert isinstance(space, Continuous)
    assert jnp.allclose(space.low, 0.0)
    assert jnp.allclose(space.high, 4.0)
    assert space.dtype == jnp.float32


def test_observation_space_preserves_continuous_bounds_float32():
    env = ScalarToyEnv()
    w = ContinuousObservationWrapper(env)
    space = w.observation_space
    assert isinstance(space, Continuous)
    assert space.dtype == jnp.float32


def test_observation_space_contains_after_init_and_step():
    env = IntObsEnv()
    w = ContinuousObservationWrapper(env)
    key = jax.random.key(0)
    state, info = w.init(key)
    assert w.observation_space.contains(info.obs)
    state, info = w.step(state, jnp.array(0))
    assert w.observation_space.contains(info.obs)


def test_action_space_unchanged():
    env = IntObsEnv()
    w = ContinuousObservationWrapper(env)
    assert w.action_space is env.action_space


def test_pytree_observation_space_all_leaves_continuous():
    """Wrapper maps PyTreeSpace to same structure with all leaves Continuous (to_continuous per leaf)."""
    env = PyTreeObsEnv(shapes={"a": (2,), "b": (3,)})
    w = ContinuousObservationWrapper(env)
    space = w.observation_space
    leaves = jax.tree_util.tree_leaves(
        space, is_leaf=lambda x: isinstance(x, (Continuous, Discrete))
    )
    assert all(isinstance(s, Continuous) for s in leaves)


def test_to_continuous_helper_discrete_scalar():
    space = Discrete(n=5)
    out = to_continuous(space)
    assert isinstance(out, Continuous)
    assert jnp.allclose(out.low, 0.0)
    assert jnp.allclose(out.high, 4.0)
    assert out.dtype == jnp.float32


def test_to_continuous_helper_discrete_array():
    space = Discrete.from_shape(n=3, shape=(2,))
    out = to_continuous(space)
    assert isinstance(out, Continuous)
    assert out.shape == (2,)


def test_to_continuous_helper_float64_continuous():
    space = Continuous(
        low=jnp.array(0.0, dtype=jnp.float64), high=jnp.array(1.0, dtype=jnp.float64)
    )
    out = to_continuous(space)
    assert isinstance(out, Continuous)
    assert out.dtype == jnp.float32


def test_jit_init_step():
    env = IntObsEnv()
    w = ContinuousObservationWrapper(env)
    key = jax.random.key(0)

    @jax.jit
    def run(k):
        s, i = w.init(k)
        return jnp.ones(1, dtype=i.obs.dtype).sum()  # return array, not dtype

    out = run(key)
    assert out.dtype == jnp.float32


def test_composability_with_vmap_wrapper():
    env = IntObsEnv()
    w = ContinuousObservationWrapper(VmapWrapper(env, batch_size=3))
    key = jax.random.key(0)
    state, info = w.init(key)
    assert info.obs.dtype == jnp.float32
    assert info.obs.shape == (3,)
    assert isinstance(w.observation_space, BatchedSpace)
    # Inner space is Continuous (converted from Discrete)
    assert hasattr(w.observation_space.space, "dtype")
    assert w.observation_space.space.dtype == jnp.float32
