"""Tests for Continuous space."""

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from jenv.spaces import Continuous

# ============================================================================
# Tests: Continuous Space - Basic Functionality
# ============================================================================


@pytest.mark.parametrize(
    ("space_factory", "expected_shape", "low", "high", "expected_dtype"),
    [
        pytest.param(
            lambda: Continuous(low=0.0, high=1.0, shape=(3,), dtype=jnp.float32),
            (3,),
            0.0,
            1.0,
            jnp.float32,
            id="unit-range",
        ),
        pytest.param(
            lambda: Continuous(low=-1.0, high=1.0, shape=(2, 3), dtype=jnp.float32),
            (2, 3),
            -1.0,
            1.0,
            jnp.float32,
            id="matrix-range",
        ),
        pytest.param(
            lambda: Continuous(low=0.0, high=1.0, dtype=jnp.float32),
            (),
            0.0,
            1.0,
            jnp.float32,
            id="scalar",
        ),
        pytest.param(
            lambda: Continuous(
                low=jnp.array([0.0, -1.0]),
                high=jnp.array([1.0, 1.0]),
                dtype=jnp.float32,
            ),
            (2,),
            jnp.array([0.0, -1.0], dtype=jnp.float32),
            jnp.array([1.0, 1.0], dtype=jnp.float32),
            jnp.float32,
            id="vector-bounds",
        ),
        pytest.param(
            lambda: Continuous(low=5.0, high=5.0, shape=(3,), dtype=jnp.float32),
            (3,),
            5.0,
            5.0,
            jnp.float32,
            id="degenerate",
        ),
        pytest.param(
            lambda: Continuous(low=0.0, high=1e-10, shape=(2,), dtype=jnp.float32),
            (2,),
            0.0,
            1e-10,
            jnp.float32,
            id="tiny-range",
        ),
        pytest.param(
            lambda: Continuous(low=-1e6, high=1e6, shape=(3,), dtype=jnp.float32),
            (3,),
            -1e6,
            1e6,
            jnp.float32,
            id="large-range",
        ),
        pytest.param(
            lambda: Continuous(low=-10.0, high=-5.0, shape=(2,), dtype=jnp.float32),
            (2,),
            -10.0,
            -5.0,
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
    space = Continuous(low=0.0, high=1.0, shape=(3,), dtype=jnp.float32)

    with pytest.raises(dataclasses.FrozenInstanceError):
        space.low = 2.0


# ============================================================================
# Tests: Continuous Space - JAX Integration
# ============================================================================


def test_continuous_space_jit():
    """Test that Continuous space works with jit."""
    space = Continuous(low=0.0, high=1.0, shape=(3,), dtype=jnp.float32)

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
    space32 = Continuous(low=0.0, high=1.0, shape=(2,), dtype=jnp.float32)
    key = jax.random.PRNGKey(0)
    sample32 = space32.sample(key)
    assert sample32.dtype == jnp.float32

    # Note: float64 requires JAX_ENABLE_X64=1, skip testing it


def test_continuous_tree_operations():
    """Test that Continuous space works with JAX tree operations.

    Tree operations should only transform dynamic fields (low, high),
    not static fields (shape, dtype).
    """
    continuous = Continuous(low=0.0, high=1.0, shape=(3,), dtype=jnp.float32)
    result = jax.tree.map(lambda x: x + 10, continuous)

    # Dynamic fields should be transformed
    assert result.low == 10.0
    assert result.high == 11.0
    # Static fields should remain unchanged
    assert result.shape == (3,)
    assert result.dtype == jnp.float32


# ============================================================================
# Tests: Continuous Space - Validation
# ============================================================================


def test_continuous_space_validation():
    """Test that Continuous space validates bounds."""
    # Should raise error when low > high
    with pytest.raises(ValueError, match="low must be less than high"):
        Continuous(low=jnp.array([1.0, 0.0]), high=jnp.array([0.0, 1.0]))

    # Should raise error when low and high have different shapes
    with pytest.raises(ValueError, match="low and high must have the same shape"):
        Continuous(low=jnp.array([0.0]), high=jnp.array([1.0, 2.0]))

    # Should raise error when shape is provided with array bounds
    with pytest.raises(
        ValueError, match="shape can only be specified when low and high are scalar"
    ):
        Continuous(low=jnp.array([0.0, 0.0]), high=jnp.array([1.0, 1.0]), shape=(2,))


# ============================================================================
# Tests: Continuous Space - Edge Cases
# ============================================================================


def test_continuous_contains_wrong_dtype():
    """Test Continuous.contains with wrong dtype values."""
    space = Continuous(low=0.0, high=1.0, shape=(2,), dtype=jnp.float32)

    # Should work with float32
    assert space.contains(jnp.array([0.5, 0.5], dtype=jnp.float32))

    # Should also work with int (gets converted/compared)
    # Note: int array [0, 1] should be in range [0.0, 1.0]
    assert space.contains(jnp.array([0, 1], dtype=jnp.int32))


def test_continuous_space_replace():
    """Test replace method on Continuous space."""
    continuous = Continuous(low=0.0, high=1.0, shape=(3,), dtype=jnp.float32)
    new_continuous = continuous.replace(low=-1.0, high=2.0)
    assert new_continuous.low == -1.0
    assert new_continuous.high == 2.0
    assert new_continuous.shape == (3,)
    assert continuous.low == 0.0  # Original unchanged
