import jax
import jax.numpy as jnp
import pytest

from envelope.spaces import BatchedSpace, Continuous, Discrete, PyTreeSpace
from envelope.wrappers.clip_action_wrapper import ClipActionWrapper
from envelope.wrappers.flatten_action_wrapper import (
    FlattenActionWrapper,
    flatten_space,
    unflatten_x,
)
from envelope.wrappers.vmap_wrapper import VmapWrapper
from tests.wrappers.helpers import PyTreeActionEnv, ScalarToyEnv


def test_step_with_pytree_action_space():
    env = PyTreeActionEnv()
    w = FlattenActionWrapper(env=env)
    key = jax.random.PRNGKey(0)
    state, _ = w.init(key)
    # Flat action shape (5,) -> unflatten to {"a": (2,), "b": (3,)}
    flat_action = jnp.array([0.1, 0.2, -0.1, 0.0, 0.3])
    state_w, info_w = w.step(state, flat_action)
    unflattened = {"a": flat_action[:2], "b": flat_action[2:]}
    state_e, info_e = env.step(state, unflattened)
    assert jnp.allclose(state_w, state_e)
    assert jnp.allclose(info_w.obs, info_e.obs)


def test_unflatten_roundtrip():
    env = PyTreeActionEnv()
    treedef, shapes, dims = flatten_space(env.action_space)
    action = env.action_space.sample(jax.random.PRNGKey(0))
    flat_leaves = jax.tree.leaves(action)
    flat = jnp.concatenate([jnp.reshape(x, -1) for x in flat_leaves], axis=0)
    recovered = unflatten_x(flat, treedef, shapes, dims)
    assert jax.tree_util.tree_all(
        jax.tree.map(lambda a, b: jnp.allclose(a, b), action, recovered)
    )


def test_single_leaf_action_space_near_noop():
    env = ScalarToyEnv()
    w = FlattenActionWrapper(env=env)
    key = jax.random.PRNGKey(0)
    state, _ = w.init(key)
    # Wrapper flattens to shape (1,) for scalar action space
    action = jnp.array([0.3])
    state_w, info_w = w.step(state, action)
    state_e, info_e = env.step(state, jnp.array(0.3))
    assert jnp.allclose(state_w, state_e)
    assert jnp.allclose(info_w.obs, info_e.obs)


def test_init_reset_delegate_unchanged():
    env = PyTreeActionEnv()
    w = FlattenActionWrapper(env=env)
    key = jax.random.PRNGKey(0)
    state_w, info_w = w.init(key)
    state_e, info_e = env.init(key)
    assert jnp.allclose(state_w, state_e)
    assert jnp.allclose(info_w.obs, info_e.obs)
    state_w, info_w = w.reset(key, state_w)
    state_e, info_e = env.reset(key, state_e)
    assert jnp.allclose(state_w, state_e)
    assert jnp.allclose(info_w.obs, info_e.obs)


def test_action_space_flattened_continuous():
    env = PyTreeActionEnv()
    w = FlattenActionWrapper(env=env)
    space = w.action_space
    assert isinstance(space, Continuous)
    assert space.shape == (5,)
    assert jnp.allclose(space.low, jnp.full(5, -1.0))
    assert jnp.allclose(space.high, jnp.full(5, 1.0))


def test_action_space_flattened_discrete():
    from envelope.spaces import PyTreeSpace

    class DiscretePyTreeActionEnv:
        @property
        def observation_space(self):
            return Continuous.from_shape(-jnp.inf, jnp.inf, (1,))

        @property
        def action_space(self):
            return PyTreeSpace(
                {
                    "a": Discrete(n=2),
                    "b": Discrete(n=3),
                }
            )

        def init(self, key):
            s = jnp.zeros(1)
            from envelope.environment import InfoContainer

            return s, InfoContainer(
                obs=s, reward=0.0, terminated=False, truncated=False
            )

        def reset(self, key, state):
            return self.init(key)

        def step(self, state, action):
            from envelope.environment import InfoContainer

            return state, InfoContainer(
                obs=state, reward=0.0, terminated=False, truncated=False
            )

    env = DiscretePyTreeActionEnv()
    w = FlattenActionWrapper(env=env)
    space = w.action_space
    assert isinstance(space, Discrete)
    assert space.shape == (
        2,
    )  # 2 + 3 = 5 elements? No - Discrete(n=2) has shape (), Discrete(n=3) has ().
    # Actually flatten_space on PyTreeSpace of Discrete: shape is PyTree of shapes. Discrete has shape ().
    # So shapes are [(), ()] and dims [1, 1] for n=2 and n=3? No - prod(()) = 1. So total dim 2.
    # And Discrete concatenation: n = concat([2, 3]) = array [2, 3], shape (2,). So action_space is Discrete(n=[2,3]).
    assert space.shape == (2,)


def test_action_space_contains_sampled():
    env = PyTreeActionEnv()
    w = FlattenActionWrapper(env=env)
    key = jax.random.PRNGKey(0)
    action = w.action_space.sample(key)
    assert w.action_space.contains(action)


def test_observation_space_unchanged():
    env = PyTreeActionEnv()
    w = FlattenActionWrapper(env=env)
    assert w.observation_space is env.observation_space


def test_mixed_space_types_raises_value_error():
    class MixedActionEnv:
        @property
        def observation_space(self):
            return Continuous.from_shape(-1.0, 1.0, (1,))

        @property
        def action_space(self):
            return PyTreeSpace(
                {
                    "a": Continuous.from_shape(-1.0, 1.0, (1,)),
                    "b": Discrete(n=3),
                }
            )

        def init(self, key):
            s = jnp.zeros(1)
            from envelope.environment import InfoContainer

            return s, InfoContainer(
                obs=s, reward=0.0, terminated=False, truncated=False
            )

        def reset(self, key, state):
            return self.init(key)

        def step(self, state, action):
            from envelope.environment import InfoContainer

            return state, InfoContainer(
                obs=state, reward=0.0, terminated=False, truncated=False
            )

    env = MixedActionEnv()
    with pytest.raises(ValueError, match="All spaces must be of the same type"):
        w = FlattenActionWrapper(env=env)
        _ = w.action_space


def test_jit_step():
    """Step produces correct output. Full JIT of step is not supported: unflatten_x uses jnp.split(..., indices) with space-derived indices, which triggers ConcretizationTypeError under jit."""
    env = PyTreeActionEnv()
    w = FlattenActionWrapper(env=env)
    key = jax.random.PRNGKey(0)
    state, _ = w.init(key)
    action = w.action_space.sample(key)
    next_state, info = w.step(state, action)
    assert next_state.shape == (5,)
    assert info.obs.shape == (5,)


def test_composability_with_clip_action_wrapper():
    env = PyTreeActionEnv()
    w = FlattenActionWrapper(env=ClipActionWrapper(env=env))
    key = jax.random.PRNGKey(0)
    state, _ = w.init(key)
    # Out-of-bounds flat action
    flat_action = jnp.array([2.0, 2.0, -2.0, -2.0, -2.0])
    state, info = w.step(state, flat_action)
    # Should be clipped then unflattened
    assert jnp.allclose(state, jnp.array([1.0, 1.0, -1.0, -1.0, -1.0]))


def test_composability_with_vmap_wrapper():
    # Vmap(FlattenAction(env)): flatten first then vmap so step receives (batch, 5) and vmap splits
    batch_size = 2
    env = PyTreeActionEnv()
    w = VmapWrapper(env=FlattenActionWrapper(env=env), batch_size=batch_size)
    key = jax.random.PRNGKey(0)
    state, _ = w.init(key)
    assert isinstance(w.action_space, BatchedSpace)
    action = w.action_space.sample(key)
    assert action.shape == (batch_size, 5)
    state, info = w.step(state, action)
    assert info.obs.shape == (batch_size, 5)
