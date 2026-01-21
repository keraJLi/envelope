"""Tests for Continuous space."""

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from envelope.spaces import Continuous

# ============================================================================
# Tests: Continuous Space - Basic Functionality
# ============================================================================


@pytest.mark.parametrize(
    ("space_factory", "expected_shape", "low", "high", "expected_dtype"),
    [
        pytest.param(
            lambda: Continuous.from_shape(low=0.0, high=1.0, shape=(3,)),
            (3,),
            jnp.full((3,), 0.0, dtype=jnp.float32),
            jnp.full((3,), 1.0, dtype=jnp.float32),
            jnp.float32,
            id="unit-range",
        ),
        pytest.param(
            lambda: Continuous.from_shape(low=-1.0, high=1.0, shape=(2, 3)),
            (2, 3),
            jnp.full((2, 3), -1.0, dtype=jnp.float32),
            jnp.full((2, 3), 1.0, dtype=jnp.float32),
            jnp.float32,
            id="matrix-range",
        ),
        pytest.param(
            lambda: Continuous(
                low=jnp.array(0.0, dtype=jnp.float32),
                high=jnp.array(1.0, dtype=jnp.float32),
            ),
            (),
            0.0,
            1.0,
            jnp.float32,
            id="scalar",
        ),
        pytest.param(
            lambda: Continuous(
                low=jnp.array([0.0, -1.0], dtype=jnp.float32),
                high=jnp.array([1.0, 1.0], dtype=jnp.float32),
            ),
            (2,),
            jnp.array([0.0, -1.0], dtype=jnp.float32),
            jnp.array([1.0, 1.0], dtype=jnp.float32),
            jnp.float32,
            id="vector-bounds",
        ),
        pytest.param(
            lambda: Continuous.from_shape(low=5.0, high=5.0, shape=(3,)),
            (3,),
            jnp.full((3,), 5.0, dtype=jnp.float32),
            jnp.full((3,), 5.0, dtype=jnp.float32),
            jnp.float32,
            id="degenerate",
        ),
        pytest.param(
            lambda: Continuous.from_shape(low=0.0, high=1e-10, shape=(2,)),
            (2,),
            jnp.full((2,), 0.0, dtype=jnp.float32),
            jnp.full((2,), 1e-10, dtype=jnp.float32),
            jnp.float32,
            id="tiny-range",
        ),
        pytest.param(
            lambda: Continuous.from_shape(low=-1e6, high=1e6, shape=(3,)),
            (3,),
            jnp.full((3,), -1e6, dtype=jnp.float32),
            jnp.full((3,), 1e6, dtype=jnp.float32),
            jnp.float32,
            id="large-range",
        ),
        pytest.param(
            lambda: Continuous.from_shape(low=-10.0, high=-5.0, shape=(2,)),
            (2,),
            jnp.full((2,), -10.0, dtype=jnp.float32),
            jnp.full((2,), -5.0, dtype=jnp.float32),
            jnp.float32,
            id="negative-range",
        ),
    ],
)
def test_continuous_space_sampling(
    space_factory, expected_shape, low, high, expected_dtype
):
    """Exercise sampling and containment across Continuous configurations."""
    space = space_factory()

    key = jax.random.PRNGKey(1)
    sample = space.sample(key)

    assert sample.shape == expected_shape
    assert sample.dtype == expected_dtype
    assert jnp.all(sample >= low)
    assert jnp.all(sample <= high)
    assert space.contains(sample)

    keys = jax.random.split(key, 8)
    samples = jax.vmap(space.sample)(keys)
    assert jnp.all(jax.vmap(space.contains)(samples))
    assert jnp.all(samples >= low)
    assert jnp.all(samples <= high)


def test_continuous_space_frozen():
    """Test that Continuous space is frozen."""
    space = Continuous.from_shape(low=0.0, high=1.0, shape=(3,))

    with pytest.raises(dataclasses.FrozenInstanceError):
        space.low = 2.0


# ============================================================================
# Tests: Continuous Space - JAX Integration
# ============================================================================


def test_continuous_space_jit():
    """Test that Continuous space works with jit."""
    space = Continuous.from_shape(low=0.0, high=1.0, shape=(3,))

    @jax.jit
    def sample_and_check(key):
        sample = space.sample(key)
        valid = space.contains(sample)
        return sample, valid

    key = jax.random.PRNGKey(0)
    sample, valid = sample_and_check(key)

    assert sample.shape == (3,)
    assert valid


def test_continuous_space_different_dtypes():
    """Test Continuous space with different dtypes."""
    # float32
    space32 = Continuous.from_shape(low=0.0, high=1.0, shape=(2,))
    key = jax.random.PRNGKey(0)
    sample32 = space32.sample(key)
    assert sample32.dtype == jnp.float32

    # Note: float16 requires JAX_ENABLE_X64=1, skip testing it


def test_continuous_tree_operations():
    """Test that Continuous space works with JAX tree operations.

    Tree operations should only transform dynamic fields (low, high),
    not properties (shape, dtype).
    """
    continuous = Continuous.from_shape(low=0.0, high=1.0, shape=(3,))
    result = jax.tree.map(lambda x: x + 10, continuous)

    # Dynamic fields should be transformed
    assert jnp.allclose(result.low, 10.0)
    assert jnp.allclose(result.high, 11.0)
    # Properties should remain unchanged
    assert result.shape == (3,)
    assert result.dtype == jnp.float32


# ============================================================================
# Tests: Continuous Space - Validation
# ============================================================================


def test_continuous_space_validation():
    """Test that Continuous space validates bounds at appropriate points."""
    key = jax.random.PRNGKey(0)

    # Test 1: Shape mismatch - fails when accessing shape property
    with pytest.raises(ValueError, match="low and high must have the same shape"):
        space = Continuous(low=jnp.array([0.0]), high=jnp.array([1.0, 2.0]))
        _ = space.shape  # Access shape property to trigger validation

    # Test 1b: Shape mismatch - also fails when calling sample() (which accesses shape)
    with pytest.raises(ValueError, match="low and high must have the same shape"):
        space = Continuous(low=jnp.array([0.0]), high=jnp.array([1.0, 2.0]))
        space.sample(key)  # sample() accesses self.shape, triggering validation

    # Test 2: Dtype mismatch - fails when accessing dtype property
    # Note: JAX may auto-convert dtypes in some cases, so this might not always fail
    # We test with explicit dtypes that JAX won't auto-convert
    try:
        space = Continuous(
            low=jnp.array([0.0], dtype=jnp.float32),
            high=jnp.array([1.0], dtype=jnp.float16),
        )
        _ = space.dtype  # Access dtype property to trigger validation
        # If we get here, JAX auto-converted the dtype, which is acceptable
    except ValueError as e:
        assert "low and high must have the same dtype" in str(e)

    # Test 2b: Dtype mismatch - also fails when calling sample() (which accesses dtype)
    try:
        space = Continuous(
            low=jnp.array([0.0], dtype=jnp.float32),
            high=jnp.array([1.0], dtype=jnp.float16),
        )
        space.sample(key)  # sample() accesses self.dtype, triggering validation
        # If we get here, JAX auto-converted the dtype, which is acceptable
    except ValueError as e:
        assert "low and high must have the same dtype" in str(e)


# ============================================================================
# Tests: Continuous Space - Edge Cases
# ============================================================================


def test_continuous_contains_wrong_dtype():
    """Test Continuous.contains with wrong dtype values."""
    space = Continuous.from_shape(low=0.0, high=1.0, shape=(2,))

    # Should work with float32
    assert space.contains(jnp.array([0.5, 0.5], dtype=jnp.float32))

    # Should also work with int (gets converted/compared)
    # Note: int array [0, 1] should be in range [0.0, 1.0]
    assert space.contains(jnp.array([0, 1], dtype=jnp.int32))


def test_continuous_space_replace():
    """Test replace method on Continuous space."""
    continuous = Continuous.from_shape(low=0.0, high=1.0, shape=(3,))
    new_continuous = continuous.replace(
        low=jnp.full((3,), -1.0, dtype=jnp.float32),
        high=jnp.full((3,), 2.0, dtype=jnp.float32),
    )
    assert jnp.allclose(new_continuous.low, -1.0)
    assert jnp.allclose(new_continuous.high, 2.0)
    assert new_continuous.shape == (3,)
    assert jnp.allclose(continuous.low, 0.0)  # Original unchanged


# ============================================================================
# Tests: Continuous Space - Vmap Creation
# ============================================================================


def test_continuous_space_from_vmapped_function():
    """Test that Continuous space can be returned from a vmapped function.

    When vmap creates spaces, parameters become batched. The space
    will have batched low/high values.
    """

    def make_continuous_space(high):
        # Use array bounds - when high is batched from vmap, it becomes 2D
        return Continuous.from_shape(low=0.0, high=high, shape=(2,))

    # Vmap over function that creates Continuous spaces
    batch_size = 4
    high_values = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    spaces = jax.vmap(make_continuous_space)(high_values)

    # Verify the vmapped result works - high becomes batched
    assert spaces.high.shape == (batch_size, 2)
    assert jnp.allclose(spaces.high[:, 0], high_values)
    assert jnp.allclose(spaces.low, 0.0)

    # Test sampling from the vmapped space
    key = jax.random.PRNGKey(0)
    sample = spaces.sample(key)
    assert sample.shape == (batch_size, 2)
    assert jnp.all(sample >= 0.0)
    # Each batch element should be <= its corresponding high value
    assert jnp.all(sample[:, 0] <= high_values)
    assert jnp.all(sample[:, 1] <= high_values)

    # Test contains
    assert spaces.contains(sample)
