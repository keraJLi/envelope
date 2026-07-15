"""Tests for Discrete space."""

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from envelope.spaces import Discrete

# ============================================================================
# Tests: Discrete Space - Basic Functionality
# ============================================================================


@pytest.mark.parametrize(
    ("space_factory", "expected_shape", "lower", "upper", "expected_dtype"),
    [
        pytest.param(
            lambda: Discrete(n=jnp.array(10, dtype=jnp.int32)),
            (),
            0,
            10,
            jnp.int32,
            id="scalar-int32",
        ),
        pytest.param(
            lambda: Discrete(n=jnp.array([5, 10], dtype=jnp.int32)),
            (2,),
            jnp.zeros(2, dtype=jnp.int32),
            jnp.array([5, 10], dtype=jnp.int32),
            jnp.int32,
            id="vector-bounds",
        ),
        pytest.param(
            lambda: Discrete(n=jnp.full((3,), 5, dtype=jnp.int32)),
            (3,),
            jnp.zeros(3, dtype=jnp.int32),
            jnp.full((3,), 5, dtype=jnp.int32),
            jnp.int32,
            id="broadcast-shape",
        ),
        pytest.param(
            lambda: Discrete(n=jnp.array(1, dtype=jnp.int32)),
            (),
            0,
            1,
            jnp.int32,
            id="single-value",
        ),
        pytest.param(
            lambda: Discrete(n=jnp.array(1_000_000, dtype=jnp.int32)),
            (),
            0,
            1_000_000,
            jnp.int32,
            id="large-range",
        ),
    ],
)
def test_discrete_space_sampling(
    space_factory, expected_shape, lower, upper, expected_dtype
):
    """Exercise sampling and containment across Discrete configurations without duplication."""
    space = space_factory()

    key = jax.random.key(0)
    sample = space.sample(key)

    assert sample.shape == expected_shape
    assert sample.dtype == expected_dtype
    assert jnp.all(sample >= lower)
    assert jnp.all(sample < upper)
    assert space.contains(sample)

    keys = jax.random.split(key, 8)
    samples = jax.vmap(space.sample)(keys)
    assert jnp.all(jax.vmap(space.contains)(samples))
    assert jnp.all(samples >= lower)
    assert jnp.all(samples < upper)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(jnp.array([0, 5, 9]), True, id="valid-array"),
        pytest.param(jnp.array([0, 5, 10]), False, id="out-of-range"),
        pytest.param(jnp.array([-1, 0, 5]), False, id="negative"),
    ],
)
def test_discrete_space_contains_array(value, expected):
    """Parameterised coverage for array inputs to Discrete.contains."""
    space = Discrete.from_shape(n=10, shape=(3,))
    assert space.contains(value) == expected


def test_discrete_space_frozen():
    """Test that Discrete space is frozen."""
    space = Discrete(n=10)

    with pytest.raises(dataclasses.FrozenInstanceError):
        space.n = 20


# ============================================================================
# Tests: Discrete Space - JAX Integration
# ============================================================================


def test_discrete_space_jit():
    """Test that Discrete space works with jit."""
    space = Discrete(n=10)

    @jax.jit
    def sample_and_check(key):
        sample = space.sample(key)
        valid = space.contains(sample)
        return sample, valid

    key = jax.random.key(0)
    sample, valid = sample_and_check(key)

    assert 0 <= sample < 10
    assert valid


def test_discrete_space_different_dtypes():
    """Test Discrete space with different dtypes."""
    # int32
    space32 = Discrete(n=10)
    key = jax.random.key(0)
    sample32 = space32.sample(key)
    assert sample32.dtype == jnp.int32

    # Note: int64 may not work consistently, skip it


def test_discrete_tree_operations():
    """Test that Discrete space works with JAX tree operations.

    Tree operations should only transform dynamic fields (n),
    not properties (shape, dtype).
    """
    # Test Discrete space
    discrete = Discrete(n=10)
    result = jax.tree.map(lambda x: x * 2, discrete)

    # Dynamic field (n) should be transformed
    assert result.n == 20
    # Properties should remain unchanged
    assert result.dtype == jnp.int32
    assert result.shape == discrete.shape


# ============================================================================
# Tests: Discrete Space - Edge Cases
# ============================================================================


def test_discrete_contains_uses_integer_dtype_category():
    space = Discrete(n=10)

    assert space.contains(jnp.array(5, dtype=jnp.int32))
    assert space.contains(jnp.array(5, dtype=jnp.int16))
    assert not space.contains(jnp.array(5.0, dtype=jnp.float32))


def test_discrete_space_replace():
    """Test replace method on Discrete space."""
    discrete = Discrete(n=10)
    new_discrete = discrete.replace(n=20)
    assert new_discrete.n == 20
    assert new_discrete.dtype == jnp.int32
    assert discrete.n == 10  # Original unchanged


# ============================================================================
# Tests: Discrete Space - Vmap Creation
# ============================================================================


def test_discrete_space_from_vmapped_function():
    """Test that Discrete space can be returned from a vmapped function.

    When vmap creates spaces, parameters become batched. The space
    will have batched n values, and shape is inferred from n.
    """

    def make_discrete_space(n):
        # When n is batched from vmap, shape will be inferred from n
        return Discrete(n=n)

    # Vmap over function that creates Discrete spaces
    batch_size = 5
    n_values = jnp.array([4, 5, 6, 7, 8], dtype=jnp.int32)
    spaces = jax.vmap(make_discrete_space)(n_values)

    # Verify the vmapped result works - n becomes batched
    assert spaces.n.shape == (batch_size,)
    assert jnp.allclose(spaces.n, n_values)
    # Shape should be inferred from batched n
    assert spaces.shape == (batch_size,)

    # Test sampling from the vmapped space
    key = jax.random.key(0)
    sample = spaces.sample(key)
    assert sample.shape == (batch_size,)
    assert jnp.all(sample >= 0)
    assert jnp.all(sample < n_values)

    # Test contains
    assert spaces.contains(sample)
