import jax
import jax.numpy as jnp
import pytest

from envelope.environment import InfoContainer
from envelope.spaces import BatchedSpace, Continuous
from envelope.wrappers.continuous_observation_wrapper import (
    ContinuousObservationWrapper,
)
from envelope.wrappers.flatten_observation_wrapper import (
    FlattenObservationWrapper,
    flatten_x,
)
from envelope.wrappers.vmap_wrapper import VmapWrapper
from tests.wrappers.helpers import PyTreeObsEnv, ScalarToyEnv, VectorObsEnv


def test_init_flattens_pytree_obs():
    env = PyTreeObsEnv(shapes={"a": (2,), "b": (3,)})
    w = FlattenObservationWrapper(env)
    key = jax.random.key(0)
    state, info = w.init(key)
    assert info.obs.shape == (5,)


def test_reset_step_flatten_pytree_obs():
    env = PyTreeObsEnv(shapes={"a": (2,), "b": (3,)})
    w = FlattenObservationWrapper(env)
    key = jax.random.key(0)
    state, info = w.init(key)
    assert info.obs.shape == (5,)
    state, info = w.reset(state, key)
    assert info.obs.shape == (5,)
    assert w.observation_space.contains(info.obs)
    state, info = w.step(state, jnp.array(0.0))
    assert info.obs.shape == (5,)


def test_init_equivalence_to_manual_flatten_x():
    env = PyTreeObsEnv(shapes={"a": (2,), "b": (3,)})
    w = FlattenObservationWrapper(env)
    key = jax.random.key(0)
    _, info_w = w.init(key)
    _, info_raw = env.init(key)
    manual = flatten_x(info_raw.obs)
    assert jnp.allclose(info_w.obs, manual)


def test_single_vector_obs_near_noop():
    env = VectorObsEnv(dim=4)
    w = FlattenObservationWrapper(env)
    key = jax.random.key(0)
    state, info = w.init(key)
    assert info.obs.shape == (4,)
    state_e, info_e = env.init(key)
    assert jnp.allclose(info.obs, info_e.obs)


def test_scalar_obs_becomes_1d():
    env = ScalarToyEnv()
    w = FlattenObservationWrapper(env)
    key = jax.random.key(0)
    state, info = w.init(key)
    assert info.obs.shape == (1,)
    assert jnp.allclose(info.obs, jnp.array([0.0]))


def test_observation_space_flattened():
    env = PyTreeObsEnv(shapes={"a": (2,), "b": (3,)})
    w = FlattenObservationWrapper(env)
    space = w.observation_space
    assert isinstance(space, Continuous)
    assert space.shape == (5,)
    # Concatenated bounds from PyTree leaves (all -inf, inf for PyTreeObsEnv)
    assert jnp.allclose(space.low, jnp.full(5, -jnp.inf))
    assert jnp.allclose(space.high, jnp.full(5, jnp.inf))


def test_observation_space_contains_after_init_and_step():
    env = PyTreeObsEnv(shapes={"a": (2,), "b": (3,)})
    w = FlattenObservationWrapper(env)
    key = jax.random.key(0)
    state, info = w.init(key)
    assert w.observation_space.contains(info.obs)
    state, info = w.step(state, jnp.array(0.0))
    assert w.observation_space.contains(info.obs)


def test_action_space_unchanged():
    env = PyTreeObsEnv(shapes={"a": (2,), "b": (3,)})
    w = FlattenObservationWrapper(env)
    assert w.action_space is env.action_space


def test_multidimensional_leaves_flatten_correctly():
    env = PyTreeObsEnv(shapes={"img": (4, 4), "vec": (3,)})
    w = FlattenObservationWrapper(env)
    key = jax.random.key(0)
    state, info = w.init(key)
    # 4*4 + 3 = 19
    assert info.obs.shape == (19,)


def test_mixed_space_types_raises_value_error():
    from envelope.spaces import Discrete, PyTreeSpace

    class MixedObsEnv:
        @property
        def observation_space(self):
            return PyTreeSpace(
                {
                    "a": Continuous.from_shape(-1.0, 1.0, (2,)),
                    "b": Discrete(n=3),
                }
            )

        action_space = Continuous(low=0.0, high=1.0)

        def init(self, key):
            import jax.numpy as jnp

            from envelope.environment import InfoContainer

            obs = {"a": jnp.zeros(2), "b": jnp.array(0)}
            return obs, InfoContainer(
                obs=obs, reward=0.0, terminated=False, truncated=False
            )

        def reset(self, state, key):
            return self.init(key)

        def step(self, state, action):
            return state, InfoContainer(
                obs=state, reward=0.0, terminated=False, truncated=False
            )

    # FlattenObservationWrapper accesses observation_space in cached_property;
    # building the flattened space triggers the leaf-type check.
    env = MixedObsEnv()
    with pytest.raises(ValueError, match=r"ContinuousObservationWrapper"):
        w = FlattenObservationWrapper(env)
        _ = w.observation_space


def test_jit_init_step():
    # Use VectorObsEnv to avoid tracer in PyTreeObsEnv.init (jnp.arange)
    env = VectorObsEnv(dim=5)
    w = FlattenObservationWrapper(env)
    key = jax.random.key(0)

    @jax.jit
    def run(k):
        s, i = w.init(k)
        s, i = w.step(s, jnp.zeros(5))
        return i.obs.shape

    assert run(key) == (5,)


def test_composability_with_continuous_observation_wrapper():
    env = PyTreeObsEnv(shapes={"a": (2,), "b": (3,)})
    w = FlattenObservationWrapper(ContinuousObservationWrapper(env))
    key = jax.random.key(0)
    state, info = w.init(key)
    assert info.obs.shape == (5,)
    assert info.obs.dtype == jnp.float32


def test_composability_with_vmap_wrapper():
    # Flatten then Vmap: flat obs (5,) then batched -> (batch_size, 5)
    batch_size = 4
    env = PyTreeObsEnv(shapes={"a": (2,), "b": (3,)})
    w = VmapWrapper(FlattenObservationWrapper(env), batch_size=batch_size)
    key = jax.random.key(0)
    state, info = w.init(key)
    assert info.obs.shape == (batch_size, 5)
    assert isinstance(w.observation_space, BatchedSpace)
    assert w.observation_space.space.shape == (5,)


def test_flatten_observation_outside_vmap_preserves_batch_prefix():
    batch_size = 4
    w = FlattenObservationWrapper(
        VmapWrapper(
            PyTreeObsEnv(shapes={"a": (2,), "b": (3,)}),
            batch_size=batch_size,
        )
    )

    state, info = w.init(jax.random.key(0))
    _, step_info = jax.jit(w.step)(state, jnp.zeros((batch_size,)))

    assert w.observation_space.shape == (batch_size, 5)
    assert info.obs.shape == (batch_size, 5)
    assert step_info.obs.shape == (batch_size, 5)

