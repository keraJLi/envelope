import pickle

import jax
import jax.numpy as jnp
import pytest

from envelope.wrappers.normalization import RunningMeanVar, update_rmv
from envelope.wrappers.observation_normalization_wrapper import (
    ObservationNormalizationWrapper,
)
from envelope.wrappers.vmap_wrapper import VmapWrapper
from tests.wrappers.helpers import (
    ConstantObsEnv,
    IntObsEnv,
    RandomImageEnv,
    VectorObsEnv,
)

# -----------------------------------------------------------------------------
# Core: stats_spec inference and dtype validation
# -----------------------------------------------------------------------------


def test_stats_spec_infers_from_unbatched_space():
    base = VectorObsEnv(dim=5)
    # Wrap with vmap first to add batch dimension, then normalize
    vm = VmapWrapper(env=base, batch_size=7)
    w = ObservationNormalizationWrapper(env=vm)
    # stats_spec should match unbatched obs leaves
    sd = w.stats_spec  # jax.ShapeDtypeStruct for leaf
    assert hasattr(sd, "shape") and hasattr(sd, "dtype")
    assert sd.shape == base.observation_space.shape
    assert sd.dtype == base.observation_space.dtype


def test_non_floating_observation_raises():
    env = IntObsEnv()
    with pytest.raises(ValueError):
        _ = ObservationNormalizationWrapper(env=env)


# -----------------------------------------------------------------------------
# Core: normalization correctness vs manual computation
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size,dim", [(3, 4), (5, 2)])
def test_normalization_matches_manual(batch_size, dim):
    base = VectorObsEnv(dim=dim)
    w = ObservationNormalizationWrapper(
        env=VmapWrapper(env=base, batch_size=batch_size)
    )
    key = jax.random.PRNGKey(0)
    state, info = w.reset(key)
    # Manual rmv update on reshaped obs (-1, *spec.shape)
    reshaped = info.unnormalized_obs.reshape((-1,) + base.observation_space.shape)
    rmv = update_rmv(state.rmv_state, reshaped)
    # Manual normalized obs
    mean = jnp.broadcast_to(rmv.mean, info.unnormalized_obs.shape)
    std = jnp.broadcast_to(jnp.sqrt(rmv.var), info.unnormalized_obs.shape)
    manual = ((info.unnormalized_obs - mean) / (std + 1e-8)).astype(
        base.observation_space.dtype
    )
    assert jnp.allclose(info.obs, manual, atol=1e-6, rtol=1e-6)
    assert state.rmv_state.count == batch_size
    # After one step, counts add batch_size again
    action = w.env.action_space.sample(key)  # sample using vmapped env's action space
    state2, info2 = w.step(state, action)
    assert state2.rmv_state.count == 2 * batch_size
    assert "unnormalized_obs" in info2.__dict__ or hasattr(info2, "unnormalized_obs")


# -----------------------------------------------------------------------------
# Core: batching and nested vmaps
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("b1,b2,dim", [(2, 3, 2), (3, 2, 4)])
def test_nested_vmap_stats_count_and_shapes(b1, b2, dim):
    base = VectorObsEnv(dim=dim)
    inner = VmapWrapper(env=base, batch_size=b2)
    outer = VmapWrapper(env=inner, batch_size=b1)
    w = ObservationNormalizationWrapper(env=outer)
    key = jax.random.PRNGKey(123)
    state, info = w.reset(key)
    assert state.rmv_state.count == b1 * b2
    assert info.obs.shape == (b1, b2, dim)


# -----------------------------------------------------------------------------
# Core: transform compatibility and serialization
# -----------------------------------------------------------------------------


def test_jit_compatibility_smoke():
    base = VectorObsEnv(dim=3)
    w = ObservationNormalizationWrapper(env=VmapWrapper(env=base, batch_size=4))
    key = jax.random.PRNGKey(0)
    print(w)

    @jax.jit
    def run_once(k, a):
        s, i = w.reset(k)
        print(f"s: {s}, a: {a}")
        ns, ni = w.step(s, a)
        return ns.rmv_state.count, ni.obs.shape

    # Sample actions outside of jit to avoid tracing space construction
    action = w.action_space.sample(key)
    cnt, shape = run_once(key, action)
    # Only check shapes; count semantics differ under various compositions
    assert shape == (4, 3)


def test_pickle_running_mean_var_in_state():
    base = VectorObsEnv(dim=2)
    w = ObservationNormalizationWrapper(env=VmapWrapper(env=base, batch_size=3))
    state, info = w.reset(jax.random.PRNGKey(0))
    blob = pickle.dumps(state.rmv_state)
    rmv2: RunningMeanVar = pickle.loads(blob)
    assert jax.tree_util.tree_all(
        jax.tree.map(
            lambda a, b: jnp.allclose(a, b),
            rmv2.mean,
            state.rmv_state.mean,
        )
    )
    assert jax.tree_util.tree_all(
        jax.tree.map(
            lambda a, b: jnp.allclose(a, b),
            rmv2.var,
            state.rmv_state.var,
        )
    )
    assert rmv2.count == state.rmv_state.count


# -----------------------------------------------------------------------------
# Optional: Property-based sampling
# -----------------------------------------------------------------------------


try:
    from hypothesis import given, settings
    from hypothesis import strategies as st
except Exception:  # pragma: no cover - optional dependency
    pytest.skip("hypothesis not installed", allow_module_level=True)


@given(
    batch_size=st.integers(min_value=1, max_value=4),
    dim=st.integers(min_value=1, max_value=6),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(deadline=None, max_examples=20)
def test_prop_normalization_consistency(batch_size, dim, seed):
    base = VectorObsEnv(dim=dim)
    w = ObservationNormalizationWrapper(
        env=VmapWrapper(env=base, batch_size=batch_size)
    )
    key = jax.random.PRNGKey(seed)
    state, info = w.reset(key)
    reshaped = info.unnormalized_obs.reshape((-1,) + base.observation_space.shape)
    rmv = update_rmv(state.rmv_state, reshaped)
    mean = jnp.broadcast_to(rmv.mean, info.unnormalized_obs.shape)
    std = jnp.broadcast_to(jnp.sqrt(rmv.var), info.unnormalized_obs.shape)
    manual = ((info.unnormalized_obs - mean) / (std + 1e-8)).astype(
        base.observation_space.dtype
    )
    assert jnp.allclose(info.obs, manual, atol=1e-5, rtol=1e-5)


# -----------------------------------------------------------------------------
# Additional: constant observations, channel-wise/spec broadcasting, nested PyTree,
# error-paths, and scan-based RMV count
# -----------------------------------------------------------------------------


def test_constant_observations_produce_finite_near_zero_outputs():
    env = ConstantObsEnv(value=7.0, shape=(5,), dtype=jnp.float32)
    w = ObservationNormalizationWrapper(env=env)
    key = jax.random.PRNGKey(0)
    state, info = w.reset(key)
    assert jnp.all(jnp.isfinite(info.obs))
    state, info = w.step(state, jnp.asarray(0.0))
    assert jnp.all(jnp.isfinite(info.obs))
    assert jnp.all(jnp.abs(info.obs) < 1e-3)


def test_image_per_pixel_stats_spec_zero_mean_unit_std():
    H, W, C, B, T = 8, 8, 3, 4, 64

    env = RandomImageEnv(shape=(H, W, C), dtype=jnp.float32)
    v = VmapWrapper(env=env, batch_size=B)
    spec = jax.ShapeDtypeStruct((H, W, C), jnp.float32)
    w = ObservationNormalizationWrapper(env=v, stats_spec=spec)
    key = jax.random.PRNGKey(0)
    state, _ = w.reset(key)

    def scan_fn(s, _):
        s, info = w.step(s, jnp.zeros((B,)))
        return s, info.obs

    _, obs = jax.lax.scan(scan_fn, state, xs=None, length=T)
    mean = jnp.mean(obs, axis=(0, 1))
    std = jnp.std(obs, axis=(0, 1))
    assert jnp.all(jnp.abs(mean) < 0.2)
    assert jnp.all((std > 0.8) & (std < 1.2))


def test_image_channelwise_stats_spec_dtype_cast():
    H, W, C, B = 8, 8, 3, 2
    env = RandomImageEnv(shape=(H, W, C), dtype=jnp.float32)
    v = VmapWrapper(env=env, batch_size=B)
    spec = jax.ShapeDtypeStruct((1, 1, C), jnp.bfloat16)
    w = ObservationNormalizationWrapper(env=v, stats_spec=spec)
    key = jax.random.PRNGKey(0)
    state, info = w.reset(key)
    assert info.obs.dtype == jnp.bfloat16
    state, info = w.step(state, jnp.zeros((B,)))
    assert info.obs.dtype == jnp.bfloat16


def test_scalar_stats_spec_broadcast_to_vector_and_cast():
    D, B = 5, 3
    env = VectorObsEnv(dim=D)
    v = VmapWrapper(env=env, batch_size=B)
    spec = jax.ShapeDtypeStruct((), jnp.float16)
    w = ObservationNormalizationWrapper(env=v, stats_spec=spec)
    key = jax.random.PRNGKey(0)
    state, info = w.reset(key)
    assert jnp.asarray(info.obs).dtype == jnp.float16
    state, info = w.step(state, jnp.zeros((B, D), dtype=jnp.float32))
    assert jnp.asarray(info.obs).dtype == jnp.float16
