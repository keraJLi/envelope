"""Tests for jenv.compat.jumanji_jenv module."""

from __future__ import annotations

from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jumanji import specs

import jenv.compat.jumanji_jenv as jumanji_jenv
from jenv.compat.jumanji_jenv import JumanjiJenv, convert_jumanji_spec_to_jenv_space
from jenv.spaces import Continuous, Discrete, PyTreeSpace
from tests.compat.contract import (
    assert_jitted_rollout_contract,
    assert_reset_step_contract,
)

pytestmark = pytest.mark.compat


def _create_jumanji_env(env_name: str = "Snake-v1", **env_kwargs) -> JumanjiJenv:
    """Helper to create a JumanjiJenv wrapper."""
    return JumanjiJenv.from_name(env_name, env_kwargs=env_kwargs or None)


@pytest.fixture(scope="module")
def jumanji_env():
    return _create_jumanji_env()


@pytest.fixture(scope="module", autouse=True)
def _jumanji_env_warmup(jumanji_env, prng_key):
    env = jumanji_env
    key_reset, key_step = jax.random.split(prng_key)
    state, _info = env.reset(key_reset)
    action = env.action_space.sample(key_step)
    env.step(state, action)


def test_jumanji_contract_smoke(prng_key, jumanji_env):
    env = jumanji_env

    def obs_check(obs, _obs_space):
        # Observations are pytrees (often namedtuples); containment is not guaranteed
        # because observation_space is derived from Spec._specs dict.
        jax.tree.structure(obs)

    assert_reset_step_contract(env, key=prng_key, obs_check=obs_check)


def test_jumanji_contract_scan(prng_key, jumanji_env, scan_num_steps):
    assert_jitted_rollout_contract(jumanji_env, key=prng_key, num_steps=scan_num_steps)


def test_observation_space_property_smoke(jumanji_env):
    """Access observation_space cached_property."""
    env = jumanji_env
    space = env.observation_space
    assert isinstance(space, PyTreeSpace)


def test_from_name_with_time_limit_warning():
    """Warn when user passes a finite time_limit, like other compat wrappers."""
    with pytest.warns(
        UserWarning,
        match="Creating a JumanjiJenv with a finite time_limit is not recommended",
    ):
        _create_jumanji_env(time_limit=10)


def test_discrete_spec_conversion(prng_key):
    """Test conversion of DiscreteArray specs to jenv Discrete space."""
    spec = specs.DiscreteArray(num_values=7, dtype=np.int32, name="d")
    space = convert_jumanji_spec_to_jenv_space(spec)
    assert isinstance(space, Discrete)
    assert space.shape == ()
    assert int(jnp.asarray(space.n)) == 7

    sample = space.sample(prng_key)
    assert space.contains(sample)


def test_multidiscrete_spec_conversion(prng_key):
    """Test conversion of MultiDiscreteArray specs to jenv Discrete with array n."""
    md = specs.MultiDiscreteArray(
        num_values=jnp.asarray([2, 3, 4], dtype=jnp.int32), dtype=np.int32, name="md"
    )
    space = convert_jumanji_spec_to_jenv_space(md)

    assert isinstance(space, Discrete)
    assert space.shape == (3,)
    assert jnp.array_equal(space.n, jnp.asarray([2, 3, 4], dtype=jnp.int32))

    sample = space.sample(prng_key)
    assert space.contains(sample)


def test_bounded_array_spec_conversion_broadcasts_bounds(prng_key):
    """Test BoundedArray converts to Continuous with broadcasted bounds."""
    b = specs.BoundedArray(
        shape=(2, 3), dtype=np.float32, minimum=0.0, maximum=1.0, name="b"
    )
    space = convert_jumanji_spec_to_jenv_space(b)
    assert isinstance(space, Continuous)
    assert space.shape == (2, 3)
    assert jnp.all(space.low == 0.0)
    assert jnp.all(space.high == 1.0)

    sample = space.sample(prng_key)
    assert space.contains(sample)


def test_array_spec_conversion_float_is_unbounded_box():
    """Float Array converts to Continuous(-inf, +inf)."""
    spec = specs.Array(shape=(2, 3), dtype=np.float32, name="a")
    space = convert_jumanji_spec_to_jenv_space(spec)
    assert isinstance(space, Continuous)
    assert space.shape == (2, 3)
    assert jnp.all(jnp.isneginf(space.low))
    assert jnp.all(jnp.isposinf(space.high))


def test_array_spec_conversion_non_float_not_supported():
    """Non-float Array specs are intentionally not supported."""
    spec = specs.Array(shape=(2,), dtype=np.int32, name="ai")
    with pytest.raises(NotImplementedError):
        convert_jumanji_spec_to_jenv_space(spec)


def test_deepcopy_warning(jumanji_env):
    env = jumanji_env
    with pytest.warns(RuntimeWarning, match="Trying to deepcopy"):
        copied = deepcopy(env)
    assert copied is not None


def test_namedtuple_observation_preserved_for_info():
    # Current implementation preserves observation as-is.
    import collections

    NT = collections.namedtuple("NT", ["x", "y"])

    class DummyTimestep:
        observation = NT(x=jnp.array([1.0]), y=jnp.array([2.0]))
        reward = 0.0
        extras = {}

        def last(self):
            return False

    info = jumanji_jenv.convert_jumanji_to_jenv_info(DummyTimestep())
    assert isinstance(info.obs, tuple)
    assert hasattr(info.obs, "_asdict")


def test_structured_spec_dict_conversion():
    """Hit Spec._specs dict branch in convert_jumanji_spec_to_jenv_space."""

    class DummySpec:
        _specs = {
            "d": specs.DiscreteArray(num_values=3, dtype=np.int32, name="d"),
            "b": specs.BoundedArray(
                shape=(2,), dtype=np.float32, minimum=0.0, maximum=1.0, name="b"
            ),
        }

    space = convert_jumanji_spec_to_jenv_space(DummySpec())
    assert isinstance(space, PyTreeSpace)
    assert isinstance(space.tree, dict)
    assert set(space.tree.keys()) == {"d", "b"}


def test_spec_discrete_broadcast_branch_exercised():
    # Force the broadcast branch by making spec.shape larger than num_values shape,
    # but still broadcast-compatible.
    md = specs.MultiDiscreteArray(
        num_values=jnp.asarray([2, 3, 4], dtype=jnp.int32), dtype=np.int32, name="md"
    )
    md._shape = (2, 3)

    space = convert_jumanji_spec_to_jenv_space(md)
    assert isinstance(space, Discrete)
    assert space.shape == (2, 3)
    assert jnp.array_equal(space.n, jnp.broadcast_to(jnp.asarray([2, 3, 4]), (2, 3)))


def test_structured_spec_tuple_list_and_unsupported(prng_key):
    d = specs.DiscreteArray(num_values=3, dtype=np.int32, name="d")
    b = specs.BoundedArray(
        shape=(2,), dtype=np.float32, minimum=0.0, maximum=1.0, name="b"
    )

    space = convert_jumanji_spec_to_jenv_space([d, b])
    assert isinstance(space, PyTreeSpace)
    assert isinstance(space.tree, tuple)
    assert isinstance(space.tree[0], Discrete)
    assert isinstance(space.tree[1], Continuous)

    sample = space.sample(prng_key)
    assert space.contains(sample)

    with pytest.raises(ValueError, match="Unsupported spec type"):
        convert_jumanji_spec_to_jenv_space(object())
