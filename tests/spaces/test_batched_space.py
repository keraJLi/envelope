"""Tests for BatchedSpace and batch_space function."""

import jax
import jax.numpy as jnp
import pytest

from envelope.spaces import BatchedSpace, Continuous, Discrete, PyTreeSpace, batch_space

# ============================================================================
# Tests: BatchedSpace - Basic Functionality
# ============================================================================


def test_batched_space_discrete():
    """Test BatchedSpace with Discrete space."""
    base_space = Discrete(n=jnp.array(10, dtype=jnp.int32))
    batch_size = 5
    batched = BatchedSpace(space=base_space, batch_size=batch_size)

    # Test shape property
    assert batched.shape == (batch_size,)
    assert batched.dtype == jnp.int32

    # Test sampling with single key
    key = jax.random.PRNGKey(0)
    sample = batched.sample(key)
    assert sample.shape == (batch_size,)
    assert jnp.all(sample >= 0)
    assert jnp.all(sample < 10)
    assert batched.contains(sample)

    # Test sampling with batched keys
    keys = jax.random.split(key, batch_size)
    sample2 = batched.sample(keys)
    assert sample2.shape == (batch_size,)
    assert batched.contains(sample2)


def test_batched_space_continuous():
    """Test BatchedSpace with Continuous space."""
    base_space = Continuous.from_shape(low=0.0, high=1.0, shape=(3,))
    batch_size = 4
    batched = BatchedSpace(space=base_space, batch_size=batch_size)

    # Test shape property
    assert batched.shape == (batch_size, 3)
    assert batched.dtype == jnp.float32

    # Test sampling
    key = jax.random.PRNGKey(0)
    sample = batched.sample(key)
    assert sample.shape == (batch_size, 3)
    assert jnp.all(sample >= 0.0)
    assert jnp.all(sample <= 1.0)
    assert batched.contains(sample)


def test_batched_space_pytree():
    """Test BatchedSpace with PyTreeSpace."""
    base_space = PyTreeSpace(
        {
            "discrete": Discrete(n=jnp.array(5, dtype=jnp.int32)),
            "continuous": Continuous.from_shape(low=0.0, high=1.0, shape=(2,)),
        }
    )
    batch_size = 3
    batched = BatchedSpace(space=base_space, batch_size=batch_size)

    # Test shape property (should return PyTree of shapes)
    shape = batched.shape
    assert isinstance(shape, dict)
    # PyTree shapes are returned as-is (not wrapped in BatchedSpace)
    assert shape["discrete"] == ()  # scalar discrete space shape
    assert shape["continuous"] == (2,)  # continuous space shape

    # Test sampling
    key = jax.random.PRNGKey(0)
    sample = batched.sample(key)
    assert isinstance(sample, dict)
    assert sample["discrete"].shape == (batch_size,)
    assert sample["continuous"].shape == (batch_size, 2)
    assert batched.contains(sample)


def test_batched_space_sample_key_validation():
    """Test that BatchedSpace validates key shape."""
    base_space = Discrete(n=jnp.array(10, dtype=jnp.int32))
    batch_size = 5
    batched = BatchedSpace(space=base_space, batch_size=batch_size)

    # Single key should work
    key = jax.random.PRNGKey(0)
    batched.sample(key)  # Should not raise

    # Batched keys with correct size should work
    keys = jax.random.split(key, batch_size)
    batched.sample(keys)  # Should not raise

    # Batched keys with wrong size should raise
    wrong_keys = jax.random.split(key, batch_size + 1)
    with pytest.raises(ValueError, match="sample key's leading dimension"):
        batched.sample(wrong_keys)


def test_batched_space_contains():
    """Test BatchedSpace.contains with valid and invalid samples."""
    base_space = Discrete(n=jnp.array(10, dtype=jnp.int32))
    batch_size = 3
    batched = BatchedSpace(space=base_space, batch_size=batch_size)

    # Valid samples
    valid_samples = jnp.array([0, 5, 9], dtype=jnp.int32)
    assert batched.contains(valid_samples)

    # Invalid samples (out of range)
    invalid_samples = jnp.array([0, 5, 10], dtype=jnp.int32)
    assert not batched.contains(invalid_samples)

    # Invalid samples (negative)
    invalid_samples2 = jnp.array([-1, 5, 9], dtype=jnp.int32)
    assert not batched.contains(invalid_samples2)


def test_batched_space_repr():
    """Test BatchedSpace.__repr__."""
    base_space = Discrete(n=jnp.array(10, dtype=jnp.int32))
    batched = BatchedSpace(space=base_space, batch_size=5)
    repr_str = repr(batched)
    assert "BatchedSpace" in repr_str
    assert "batch_size=5" in repr_str


# ============================================================================
# Tests: batch_space function
# ============================================================================


def test_batch_space_discrete():
    """Test batch_space function with Discrete space."""
    base_space = Discrete(n=jnp.array(10, dtype=jnp.int32))
    batch_size = 4
    batched = batch_space(base_space, batch_size)

    assert isinstance(batched, BatchedSpace)
    assert batched.batch_size == batch_size
    assert batched.space == base_space
    assert batched.shape == (batch_size,)


def test_batch_space_continuous():
    """Test batch_space function with Continuous space."""
    base_space = Continuous.from_shape(low=0.0, high=1.0, shape=(2,))
    batch_size = 3
    batched = batch_space(base_space, batch_size)

    assert isinstance(batched, BatchedSpace)
    assert batched.batch_size == batch_size
    assert batched.shape == (batch_size, 2)


def test_batch_space_pytree():
    """Test batch_space function with PyTreeSpace."""
    base_space = PyTreeSpace(
        {
            "discrete": Discrete(n=jnp.array(5, dtype=jnp.int32)),
            "continuous": Continuous.from_shape(low=0.0, high=1.0, shape=(2,)),
        }
    )
    batch_size = 4
    batched = batch_space(base_space, batch_size)

    # Should return PyTreeSpace with BatchedSpace leaves
    assert isinstance(batched, PyTreeSpace)
    assert isinstance(batched.tree["discrete"], BatchedSpace)
    assert isinstance(batched.tree["continuous"], BatchedSpace)
    assert batched.tree["discrete"].batch_size == batch_size
    assert batched.tree["continuous"].batch_size == batch_size

    # Test sampling
    key = jax.random.PRNGKey(0)
    sample = batched.sample(key)
    assert sample["discrete"].shape == (batch_size,)
    assert sample["continuous"].shape == (batch_size, 2)
    assert batched.contains(sample)


def test_batch_space_nested_pytree():
    """Test batch_space function with nested PyTreeSpace."""
    base_space = PyTreeSpace(
        {
            "obs": {
                "position": Continuous.from_shape(low=-1.0, high=1.0, shape=(2,)),
                "velocity": Discrete(n=jnp.array(5, dtype=jnp.int32)),
            },
            "action": Discrete(n=jnp.array(3, dtype=jnp.int32)),
        }
    )
    batch_size = 3
    batched = batch_space(base_space, batch_size)

    assert isinstance(batched, PyTreeSpace)
    assert isinstance(batched.tree["obs"]["position"], BatchedSpace)
    assert isinstance(batched.tree["obs"]["velocity"], BatchedSpace)
    assert isinstance(batched.tree["action"], BatchedSpace)

    # Test sampling
    key = jax.random.PRNGKey(0)
    sample = batched.sample(key)
    assert sample["obs"]["position"].shape == (batch_size, 2)
    assert sample["obs"]["velocity"].shape == (batch_size,)
    assert sample["action"].shape == (batch_size,)
    assert batched.contains(sample)


def test_batch_space_multiple_calls():
    """Test that batch_space can be called multiple times."""
    base_space = Discrete(n=jnp.array(10, dtype=jnp.int32))
    batched1 = batch_space(base_space, 3)
    batched2 = batch_space(batched1, 2)

    # Should create nested BatchedSpace
    assert isinstance(batched2, BatchedSpace)
    assert isinstance(batched2.space, BatchedSpace)
    assert batched2.batch_size == 2
    assert batched2.space.batch_size == 3

    # Test sampling
    key = jax.random.PRNGKey(0)
    sample = batched2.sample(key)
    assert sample.shape == (2, 3)


# ============================================================================
# Tests: BatchedSpace - Edge Cases
# ============================================================================


def test_batched_space_batch_size_one():
    """Test BatchedSpace with batch_size=1."""
    base_space = Discrete(n=jnp.array(10, dtype=jnp.int32))
    batched = BatchedSpace(space=base_space, batch_size=1)

    assert batched.shape == (1,)
    key = jax.random.PRNGKey(0)
    sample = batched.sample(key)
    assert sample.shape == (1,)
    assert batched.contains(sample)


def test_batched_space_scalar_continuous():
    """Test BatchedSpace with scalar Continuous space."""
    base_space = Continuous(
        low=jnp.array(0.0, dtype=jnp.float32),
        high=jnp.array(1.0, dtype=jnp.float32),
    )
    batch_size = 5
    batched = BatchedSpace(space=base_space, batch_size=batch_size)

    assert batched.shape == (batch_size,)
    key = jax.random.PRNGKey(0)
    sample = batched.sample(key)
    assert sample.shape == (batch_size,)
    assert batched.contains(sample)
