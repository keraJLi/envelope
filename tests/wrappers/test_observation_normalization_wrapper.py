import pickle
from functools import cached_property

import jax
import jax.numpy as jnp
import pytest

from envelope.environment import Environment, InfoContainer
from envelope.spaces import Continuous, PyTreeSpace
from envelope.wrappers.autoreset_wrapper import AutoResetWrapper
from envelope.wrappers.normalization import RunningMeanVar, update_rmv
from envelope.wrappers.observation_normalization_wrapper import (
    ObservationNormalizationWrapper,
)
from envelope.wrappers.pooled_init_vmap_wrapper import PooledInitVmapWrapper
from envelope.wrappers.vmap_envs_wrapper import VmapEnvsWrapper
from envelope.wrappers.vmap_wrapper import VmapWrapper
from tests.wrappers.helpers import (
    ConstantObsEnv,
    IntObsEnv,
    PyTreeObsEnv,
    RandomImageEnv,
    StepCounterEnv,
    VectorObsEnv,
)


class RowStructuredObsEnv(Environment):
    """Two rows with distinct means, used to test broadcast-aware statistics."""

    @cached_property
    def observation_space(self):
        return Continuous.from_shape(-jnp.inf, jnp.inf, (2, 3, 1))

    @cached_property
    def action_space(self):
        return Continuous(low=-1.0, high=1.0)

    def init(self, key):
        obs = jnp.asarray([[[0.0], [0.0], [0.0]], [[10.0], [10.0], [10.0]]])
        return obs, InfoContainer(
            obs=obs, reward=0.0, terminated=False, truncated=False
        )

    def reset(self, state, key):
        return self.init(key)

    def step(self, state, action):
        return state, InfoContainer(
            obs=state, reward=0.0, terminated=False, truncated=False
        )


class HeterogeneousStatsEnv(Environment):
    """Leaves whose broadcast statistics consume different sample counts."""

    @cached_property
    def observation_space(self):
        return PyTreeSpace(
            (
                Continuous.from_shape(-jnp.inf, jnp.inf, (2,)),
                Continuous.from_shape(-jnp.inf, jnp.inf, (2, 3)),
            )
        )

    @cached_property
    def action_space(self):
        return Continuous(low=-1.0, high=1.0)

    def init(self, key):
        obs = (
            jnp.asarray([1.0, 3.0]),
            jnp.asarray([[0.0, 2.0, 4.0], [10.0, 12.0, 14.0]]),
        )
        return obs, InfoContainer(
            obs=obs, reward=0.0, terminated=False, truncated=False
        )

    def reset(self, state, key):
        return self.init(key)

    def step(self, state, action):
        return state, InfoContainer(
            obs=state, reward=0.0, terminated=False, truncated=False
        )


class ActionTerminationEnv(Environment):
    """Scalar env whose batch elements can finish on different actions."""

    @cached_property
    def observation_space(self):
        return Continuous(low=-jnp.inf, high=jnp.inf)

    @cached_property
    def action_space(self):
        return Continuous(low=-1.0, high=1.0)

    def init(self, key):
        state = jnp.asarray(0.0, dtype=jnp.float32)
        return state, InfoContainer(
            obs=state,
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            terminated=jnp.asarray(False),
            truncated=jnp.asarray(False),
        )

    def step(self, state, action):
        next_state = state + action
        return next_state, InfoContainer(
            obs=next_state,
            reward=action,
            terminated=action >= 0.75,
            truncated=jnp.asarray(False),
        )


# -----------------------------------------------------------------------------
# Core: stats_spec inference and dtype validation
# -----------------------------------------------------------------------------


def test_stats_spec_infers_from_unbatched_space():
    base = VectorObsEnv(dim=5)
    # Wrap with vmap first to add batch dimension, then normalize
    vm = VmapWrapper(base, batch_size=7)
    w = ObservationNormalizationWrapper(vm)
    # stats_spec should match unbatched obs leaves
    sd = w.stats_spec  # jax.ShapeDtypeStruct for leaf
    assert hasattr(sd, "shape") and hasattr(sd, "dtype")
    assert sd.shape == base.observation_space.shape
    assert sd.dtype == base.observation_space.dtype


def test_non_floating_observation_raises():
    env = IntObsEnv()
    with pytest.raises(ValueError):
        _ = ObservationNormalizationWrapper(env)


@pytest.mark.parametrize("spec_source", ["inferred", "provided"])
def test_pytree_stats_metadata_is_hashable_static_state(spec_source):
    env = PyTreeObsEnv(shapes={"a": (2,), "b": (3,)})
    expected_spec = {
        "a": jax.ShapeDtypeStruct((2,), jnp.float32),
        "b": jax.ShapeDtypeStruct((3,), jnp.float32),
    }
    stats_spec = None if spec_source == "inferred" else expected_spec

    wrapper = ObservationNormalizationWrapper(env, stats_spec=stats_spec)

    assert jax.tree.structure(wrapper.stats_spec) == jax.tree.structure(expected_spec)
    wrapper._validate_static_fields()
    _, treedef = jax.tree.flatten(wrapper)
    hash(treedef)
    rebuilt = jax.tree.map(lambda leaf: leaf, wrapper)
    assert jax.tree.structure(rebuilt.stats_spec) == jax.tree.structure(expected_spec)


# -----------------------------------------------------------------------------
# Core: normalization correctness vs manual computation
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size,dim", [(3, 4), (5, 2)])
def test_normalization_matches_manual(batch_size, dim):
    base = VectorObsEnv(dim=dim)
    w = ObservationNormalizationWrapper(VmapWrapper(base, batch_size=batch_size))
    key = jax.random.key(0)
    state, info = w.init(key)
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
    inner = VmapWrapper(base, batch_size=b2)
    outer = VmapWrapper(inner, batch_size=b1)
    w = ObservationNormalizationWrapper(outer)
    key = jax.random.key(123)
    state, info = w.init(key)
    assert state.rmv_state.count == b1 * b2
    assert info.obs.shape == (b1, b2, dim)


# -----------------------------------------------------------------------------
# Core: transform compatibility and serialization
# -----------------------------------------------------------------------------


def test_jit_compatibility_smoke():
    base = VectorObsEnv(dim=3)
    w = ObservationNormalizationWrapper(VmapWrapper(base, batch_size=4))
    key = jax.random.key(0)
    print(w)

    @jax.jit
    def run_once(k, a):
        s, i = w.init(k)
        print(f"s: {s}, a: {a}")
        ns, ni = w.step(s, a)
        return ns.rmv_state.count, ni.obs.shape

    # Sample actions outside of jit to avoid tracing space construction
    action = w.action_space.sample(key)
    cnt, shape = run_once(key, action)
    # Only check shapes; count semantics differ under various compositions
    assert shape == (4, 3)


def _shared_autoreset_normalizer() -> ObservationNormalizationWrapper:
    termination_steps = jnp.asarray([2, 3])
    envs = jax.vmap(
        lambda terminate_after: StepCounterEnv(terminate_after=terminate_after)
    )(termination_steps)
    return ObservationNormalizationWrapper(
        VmapEnvsWrapper(AutoResetWrapper(envs), batch_size=2)
    )


def _normalized_scalar(raw, state):
    return (raw - state.rmv_state.mean) / (state.rmv_state.std + 1e-8)


def test_shared_normalization_caches_partial_terminal_observations():
    wrapper = _shared_autoreset_normalizer()
    action = jnp.asarray([0.25, 0.75], dtype=jnp.float32)

    state, initial = wrapper.init(jax.random.key(0))

    assert state.rmv_state.count == 2
    assert jnp.all(~initial.final_valid)
    assert jnp.allclose(initial.final.obs, jnp.zeros(2))
    assert jnp.allclose(initial.final.unnormalized_obs, jnp.zeros(2))

    state, _ = wrapper.step(state, action)
    state, first_completion = jax.jit(wrapper.step)(state, action)

    assert jnp.array_equal(first_completion.terminated, jnp.asarray([True, False]))
    assert jnp.array_equal(first_completion.final_valid, jnp.asarray([True, False]))
    assert jnp.allclose(
        first_completion.final.obs[0],
        _normalized_scalar(first_completion.final.unnormalized_obs[0], state),
    )
    assert first_completion.final.obs[1] == 0
    first_cached = first_completion.final.obs

    state, second_completion = wrapper.step(state, action)

    assert jnp.array_equal(second_completion.terminated, jnp.asarray([False, True]))
    assert jnp.all(second_completion.final_valid)
    assert jnp.array_equal(second_completion.final.obs[0], first_cached[0])
    assert jnp.allclose(
        second_completion.final.obs[1],
        _normalized_scalar(second_completion.final.unnormalized_obs[1], state),
    )
    assert state.rmv_state.count == 8


def test_shared_normalization_preserves_final_across_manual_reset_and_scan():
    wrapper = _shared_autoreset_normalizer()
    actions = jnp.full((3, 2), 0.5, dtype=jnp.float32)
    state, _ = wrapper.init(jax.random.key(0))

    @jax.jit
    def run_steps(scan_state, scan_actions):
        return jax.lax.scan(wrapper.step, scan_state, scan_actions)

    state, infos = run_steps(state, actions)
    cached_final = infos.final.obs[-1]
    raw_final = infos.final.unnormalized_obs[-1]
    count_before_reset = state.rmv_state.count

    state, reset_info = jax.jit(wrapper.reset)(state, jax.random.key(1))

    assert jnp.array_equal(reset_info.final.obs, cached_final)
    assert jnp.array_equal(reset_info.final.unnormalized_obs, raw_final)
    assert state.rmv_state.count == count_before_reset + 2


def test_terminal_placeholder_uses_normalized_dtype_and_fixed_schema():
    wrapper = ObservationNormalizationWrapper(
        VmapWrapper(AutoResetWrapper(StepCounterEnv(terminate_after=1)), batch_size=2),
        stats_spec=jax.ShapeDtypeStruct((), jnp.float16),
    )

    state, initial = jax.jit(wrapper.init)(jax.random.key(0))
    state, completed = jax.jit(wrapper.step)(
        state, jnp.asarray([0.25, 0.75], dtype=jnp.float32)
    )

    assert initial.final.obs.dtype == jnp.float16
    assert jnp.all(initial.final.obs == 0)
    assert initial.final.unnormalized_obs.dtype == jnp.float32
    assert jax.tree.structure(initial) == jax.tree.structure(completed)
    assert completed.final.obs.dtype == jnp.float16
    assert completed.final.unnormalized_obs.dtype == jnp.float32
    assert state.rmv_state.count == 4


def test_shared_normalization_can_wrap_pooled_initialization():
    wrapper = ObservationNormalizationWrapper(
        PooledInitVmapWrapper(ActionTerminationEnv(), batch_size=2, pool_size=2)
    )
    state, initial = wrapper.init(jax.random.key(0))

    assert jnp.all(initial.final.obs == 0)
    assert state.rmv_state.count == 2

    state, first = jax.jit(wrapper.step)(
        state, jnp.asarray([1.0, 0.25], dtype=jnp.float32)
    )
    first_cached = first.final.obs

    assert jnp.array_equal(first.terminated, jnp.asarray([True, False]))
    assert jnp.array_equal(first.final_valid, jnp.asarray([True, False]))
    assert first.final.unnormalized_obs[0] == 1.0
    assert first.final.obs[1] == 0

    state, second = wrapper.step(state, jnp.asarray([0.1, 1.0], dtype=jnp.float32))

    assert jnp.array_equal(second.terminated, jnp.asarray([False, True]))
    assert jnp.array_equal(second.final.obs[0], first_cached[0])
    assert second.final.unnormalized_obs[1] == 1.25
    assert state.rmv_state.count == 6


def test_pickle_running_mean_var_in_state():
    base = VectorObsEnv(dim=2)
    w = ObservationNormalizationWrapper(VmapWrapper(base, batch_size=3))
    state, info = w.init(jax.random.key(0))
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
    w = ObservationNormalizationWrapper(VmapWrapper(base, batch_size=batch_size))
    key = jax.random.key(seed)
    state, info = w.init(key)
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
    w = ObservationNormalizationWrapper(env)
    key = jax.random.key(0)
    state, info = w.init(key)
    assert jnp.all(jnp.isfinite(info.obs))
    state, info = w.step(state, jnp.asarray(0.0))
    assert jnp.all(jnp.isfinite(info.obs))
    assert jnp.all(jnp.abs(info.obs) < 1e-3)


def test_image_per_pixel_stats_spec_zero_mean_unit_std():
    H, W, C, B, T = 8, 8, 3, 4, 64

    env = RandomImageEnv(shape=(H, W, C), dtype=jnp.float32)
    v = VmapWrapper(env, batch_size=B)
    spec = jax.ShapeDtypeStruct((H, W, C), jnp.float32)
    w = ObservationNormalizationWrapper(v, stats_spec=spec)
    key = jax.random.key(0)
    state, _ = w.init(key)

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
    v = VmapWrapper(env, batch_size=B)
    spec = jax.ShapeDtypeStruct((1, 1, C), jnp.bfloat16)
    w = ObservationNormalizationWrapper(v, stats_spec=spec)
    key = jax.random.key(0)
    state, info = w.init(key)
    assert info.obs.dtype == jnp.bfloat16
    state, info = w.step(state, jnp.zeros((B,)))
    assert info.obs.dtype == jnp.bfloat16


def test_scalar_stats_spec_broadcast_to_vector_and_cast():
    D, B = 5, 3
    env = VectorObsEnv(dim=D)
    v = VmapWrapper(env, batch_size=B)
    spec = jax.ShapeDtypeStruct((), jnp.float16)
    w = ObservationNormalizationWrapper(v, stats_spec=spec)
    key = jax.random.key(0)
    state, info = w.init(key)
    assert jnp.asarray(info.obs).dtype == jnp.float16
    assert w.observation_space.dtype == jnp.float16
    state, info = w.step(state, jnp.zeros((B, D), dtype=jnp.float32))
    assert jnp.asarray(info.obs).dtype == jnp.float16


def test_broadcast_stats_spec_reduces_the_broadcast_axes():
    spec = jax.ShapeDtypeStruct((2, 1, 1), jnp.float32)
    w = ObservationNormalizationWrapper(RowStructuredObsEnv(), stats_spec=spec)

    state, info = w.init(jax.random.key(0))

    assert jnp.allclose(state.rmv_state.mean[:, 0, 0], jnp.asarray([0.0, 10.0]))
    assert jnp.allclose(info.obs, jnp.zeros((2, 3, 1)), atol=1e-6)


def test_broadcast_statistics_track_per_leaf_effective_counts_under_jit():
    stats_spec = (
        jax.ShapeDtypeStruct((2,), jnp.float32),
        jax.ShapeDtypeStruct((1, 3), jnp.float32),
    )
    wrapper = ObservationNormalizationWrapper(
        HeterogeneousStatsEnv(), stats_spec=stats_spec
    )

    state, _ = wrapper.init(jax.random.key(0))

    assert jax.tree.structure(state.rmv_state.count) == jax.tree.structure(stats_spec)
    assert state.rmv_state.count[0] == 1
    assert state.rmv_state.count[1] == 2

    state, _ = jax.jit(wrapper.step)(state, jnp.asarray(0.0))
    assert state.rmv_state.count[0] == 2
    assert state.rmv_state.count[1] == 4


def test_constant_float16_observations_remain_finite():
    spec = jax.ShapeDtypeStruct((3,), jnp.float16)
    w = ObservationNormalizationWrapper(
        ConstantObsEnv(value=7.0, shape=(3,), dtype=jnp.float16),
        stats_spec=spec,
    )

    _, info = w.init(jax.random.key(0))

    assert info.obs.dtype == jnp.float16
    assert jnp.all(jnp.isfinite(info.obs))
