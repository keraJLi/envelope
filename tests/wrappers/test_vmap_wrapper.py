from functools import cached_property

import jax
import jax.numpy as jnp
import pytest

from jenv.environment import Environment
from jenv.spaces import Continuous, Discrete, PyTreeSpace
from jenv.struct import FrozenPyTreeNode
from jenv.typing import Key
from jenv.wrappers.vmap_wrapper import VmapWrapper
from jenv.wrappers.wrapper import Wrapper


class Info(FrozenPyTreeNode):
    obs: jax.Array
    reward: float
    terminated: bool = False
    truncated: bool = False


class ScalarEnv(Environment):
    @cached_property
    def observation_space(self) -> Continuous:
        return Continuous(low=0.0, high=1.0, shape=(3,))

    @cached_property
    def action_space(self) -> Continuous:
        return Continuous(low=-1.0, high=1.0, shape=())

    def reset(self, key: Key) -> tuple[jax.Array, Info]:
        s = jnp.array(0.0)
        return s, Info(obs=s, reward=0.0)

    def step(self, state: jax.Array, action: jax.Array) -> tuple[jax.Array, Info]:
        ns = state + action
        return ns, Info(obs=ns, reward=action)


class DiscreteActionEnv(ScalarEnv):
    @cached_property
    def action_space(self) -> Discrete:
        return Discrete(n=4)


class TreeEnv(Environment):
    @cached_property
    def observation_space(self) -> PyTreeSpace:
        return PyTreeSpace(
            {
                "x": Continuous(0.0, 1.0, shape=(2,)),
                "a": Discrete(n=3),
            }
        )

    @cached_property
    def action_space(self) -> PyTreeSpace:
        return PyTreeSpace({"u": Continuous(-1.0, 1.0, shape=(2,))})

    def reset(self, key: Key):
        s = {"x": jnp.array([0.0, 0.0])}
        return s, Info(obs=s["x"], reward=0.0)

    def step(self, state, action):
        ns = {"x": state["x"] + action["u"]}
        return ns, Info(obs=ns["x"], reward=jnp.sum(action["u"]))


def test_reset_accepts_single_key_and_splits():
    env = ScalarEnv()
    w = VmapWrapper(env=env, batch_size=4)

    k = jax.random.PRNGKey(0)
    state, info = w.reset(k)

    assert jnp.shape(state) == (4,)
    assert jnp.shape(info.obs) == (4,)


def test_reset_accepts_batched_keys():
    env = ScalarEnv()
    w = VmapWrapper(env=env, batch_size=3)

    k = jax.random.split(jax.random.PRNGKey(0), 3)
    state, info = w.reset(k)

    assert jnp.shape(state) == (3,)
    assert jnp.shape(info.obs) == (3,)


def test_reset_raises_on_wrong_batched_key_dim():
    env = ScalarEnv()
    w = VmapWrapper(env=env, batch_size=3)

    k = jax.random.split(jax.random.PRNGKey(0), 2)
    with pytest.raises(ValueError) as e:
        _ = w.reset(k)
    msg = str(e.value)
    assert "leading dimension (2)" in msg
    assert "batch_size (3)" in msg


@pytest.mark.parametrize(
    "env_factory,actions",
    [
        (ScalarEnv, jnp.array([0.1, -0.2, 0.3, 0.0])),
        (DiscreteActionEnv, jnp.array([1, 2, 0, 3], dtype=jnp.int32)),
    ],
    ids=["continuous-action", "discrete-action"],
)
def test_step_happy_path_vectors_param(env_factory, actions):
    env = env_factory()
    w = VmapWrapper(env=env, batch_size=4)
    k = jax.random.PRNGKey(0)
    s, _ = w.reset(k)
    ns, info = w.step(s, actions)
    assert jnp.allclose(ns, s + actions)
    assert jnp.allclose(info.obs, ns)


@pytest.mark.parametrize(
    "env_factory,actions",
    [
        (ScalarEnv, jnp.array([0.1, -0.2, 0.3, 0.0])),
        (DiscreteActionEnv, jnp.array([1, 2, 0, 3], dtype=jnp.int32)),
    ],
    ids=["continuous-action", "discrete-action"],
)
def test_step_matches_vmap_of_base_env_param(env_factory, actions):
    env = env_factory()
    w = VmapWrapper(env=env, batch_size=4)

    k = jax.random.PRNGKey(0)
    s, _ = w.reset(k)

    ns_w, info_w = w.step(s, actions)
    ns_ref, info_ref = jax.vmap(env.step)(s, actions)

    assert jnp.allclose(ns_w, ns_ref)
    assert jnp.allclose(info_w.obs, info_ref.obs)
    assert jnp.allclose(jnp.asarray(info_w.reward), jnp.asarray(info_ref.reward))


def test_step_raises_on_state_dim_mismatch():
    env = ScalarEnv()
    w = VmapWrapper(env=env, batch_size=3)
    k = jax.random.PRNGKey(0)
    s, _ = w.reset(k)
    a = jnp.array([0.1, 0.2, 0.3])
    with pytest.raises(Exception):
        _ = w.step(s[:2], a)


def test_step_raises_on_action_dim_mismatch():
    env = ScalarEnv()
    w = VmapWrapper(env=env, batch_size=3)
    k = jax.random.PRNGKey(0)
    s, _ = w.reset(k)
    a = jnp.array([0.1, 0.2])
    with pytest.raises(Exception):
        _ = w.step(s, a)


@pytest.mark.parametrize(
    "env_factory,batch_size,expected_obs_shape,expected_act_shape",
    [
        (ScalarEnv, 5, (5, 3), (5,)),
        (DiscreteActionEnv, 7, (7, 3), (7,)),
    ],
    ids=["continuous/continuous", "continuous/discrete"],
)
def test_space_shapes_and_contains_param(
    env_factory, batch_size, expected_obs_shape, expected_act_shape
):
    env = env_factory()
    w = VmapWrapper(env=env, batch_size=batch_size)

    obs_space = w.observation_space
    act_space = w.action_space
    assert obs_space.shape == expected_obs_shape
    assert act_space.shape == expected_act_shape

    obs_sample = obs_space.sample(jax.random.PRNGKey(0))
    act_sample = act_space.sample(jax.random.PRNGKey(1))
    assert obs_sample.shape == expected_obs_shape
    assert act_sample.shape == expected_act_shape
    assert obs_space.contains(obs_sample)
    assert act_space.contains(act_sample)


def test_pytree_space_batched_structure_and_sampling():
    env = TreeEnv()
    w = VmapWrapper(env=env, batch_size=3)

    obs_space = w.observation_space
    act_space = w.action_space
    # Structure preserved; shapes gain leading batch dim
    assert obs_space.shape["x"] == (3, 2)
    assert obs_space.shape["a"] == (3,)
    assert act_space.shape["u"] == (3, 2)

    ok = obs_space.contains(
        {"x": jnp.ones((3, 2)), "a": jnp.ones((3,), dtype=jnp.int32)}
    )
    assert jnp.asarray(ok).item()


def test_step_with_pytree_state_and_action():
    env = TreeEnv()
    w = VmapWrapper(env=env, batch_size=3)

    k = jax.random.PRNGKey(0)
    s, _ = w.reset(k)
    a = {"u": jnp.ones((3, 2))}
    ns, info = w.step(s, a)

    assert jnp.shape(ns["x"]) == (3, 2)
    assert jnp.shape(info.obs) == (3, 2)


@pytest.mark.parametrize(
    "env_factory,actions",
    [
        (ScalarEnv, jnp.array([0.5, -0.5, 0.25, -0.25])),
        (DiscreteActionEnv, jnp.array([0, 1, 2, 3], dtype=jnp.int32)),
    ],
    ids=["continuous-action", "discrete-action"],
)
def test_jit_reset_and_step_param(env_factory, actions):
    env = env_factory()
    w = VmapWrapper(env=env, batch_size=4)

    reset_jit = jax.jit(lambda key: w.reset(key))
    step_jit = jax.jit(lambda s, a: w.step(s, a))

    k = jax.random.PRNGKey(0)
    s, _ = reset_jit(k)
    ns, info = step_jit(s, actions)

    assert jnp.shape(ns) == (4,)
    assert jnp.shape(info.obs) == (4,)


def test_composes_with_base_wrapper():
    env = ScalarEnv()
    base = Wrapper(env=env)
    w = VmapWrapper(env=base, batch_size=2)
    k = jax.random.PRNGKey(0)
    s, info = w.reset(k)
    assert jnp.shape(s) == (2,)
    assert jnp.shape(info.obs) == (2,)
