"""Tests for Space serialization: repr, pickling, and hashing."""

import jax
import jax.numpy as jnp
import pytest

from envelope.spaces import Continuous, Discrete, PyTreeSpace, Space

# ============================================================================
# Test Data for Repr Tests
# ============================================================================


_REPR_DISCRETE_SPACE = Discrete(n=10)
_REPR_DISCRETE_EXPECTED = (
    _REPR_DISCRETE_SPACE.__class__.__name__,
    str(_REPR_DISCRETE_SPACE.n),
    getattr(_REPR_DISCRETE_SPACE.dtype, "__name__", str(_REPR_DISCRETE_SPACE.dtype)),
)

_REPR_CONTINUOUS_SPACE = Continuous.from_shape(low=0.0, high=1.0, shape=(3,))
_REPR_CONTINUOUS_EXPECTED = (
    _REPR_CONTINUOUS_SPACE.__class__.__name__,
    str(_REPR_CONTINUOUS_SPACE.low),
    str(_REPR_CONTINUOUS_SPACE.high),
    str(_REPR_CONTINUOUS_SPACE.shape),
)

_REPR_PYTREE_SPACE = PyTreeSpace(
    {
        "discrete": _REPR_DISCRETE_SPACE.replace(n=5),
        "continuous": Continuous.from_shape(low=0.0, high=1.0, shape=(2,)),
    }
)
_REPR_PYTREE_EXPECTED = (
    _REPR_PYTREE_SPACE.__class__.__name__,
    "Discrete",
    "Continuous",
)


# ============================================================================
# Tests: Space - Abstract Class
# ============================================================================


def test_space_is_abstract():
    """Test that Space cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Space()


# ============================================================================
# Tests: Repr
# ============================================================================


@pytest.mark.parametrize(
    ("space", "expectations"),
    [
        (_REPR_DISCRETE_SPACE, _REPR_DISCRETE_EXPECTED),
        (_REPR_CONTINUOUS_SPACE, _REPR_CONTINUOUS_EXPECTED),
        (_REPR_PYTREE_SPACE, _REPR_PYTREE_EXPECTED),
    ],
)
def test_space_repr_variants(space, expectations):
    """Ensure repr outputs include human-readable components for each space."""
    repr_str = repr(space)
    for part in expectations:
        assert part in repr_str


def test_pytree_space_repr_nested():
    """Test __repr__ for deeply nested PyTreeSpace."""
    space = PyTreeSpace(
        {
            "outer": {
                "inner": {
                    "discrete": Discrete(n=5),
                }
            }
        }
    )

    repr_str = repr(space)
    assert "PyTreeSpace" in repr_str
    assert "Discrete" in repr_str


def test_space_repr_with_array_parameters():
    """Test __repr__ for spaces with array parameters."""
    discrete = Discrete(n=jnp.array([4, 5, 6], dtype=jnp.int32))
    repr_str = repr(discrete)

    assert "Discrete" in repr_str
    # Should show the array in some form
    assert "[4" in repr_str or "Array" in repr_str or "array" in repr_str.lower()

    continuous = Continuous.from_shape(low=0.0, high=1.0, shape=(2,))
    repr_str = repr(continuous)

    assert "Continuous" in repr_str


# ============================================================================
# Tests: Pickling
# ============================================================================


def test_discrete_space_pickle():
    """Test that Discrete space can be pickled and unpickled."""
    import pickle

    space = Discrete.from_shape(n=10, shape=(2,))

    # Pickle and unpickle
    pickled = pickle.dumps(space)
    unpickled = pickle.loads(pickled)

    # Verify attributes are preserved
    assert jnp.array_equal(unpickled.n, space.n)
    assert unpickled.shape == space.shape
    assert unpickled.dtype == space.dtype

    # Verify functionality works
    key = jax.random.PRNGKey(0)
    sample = unpickled.sample(key)
    assert unpickled.contains(sample)


def test_continuous_space_pickle():
    """Test that Continuous space can be pickled and unpickled."""
    import pickle

    space = Continuous.from_shape(low=-1.0, high=1.0, shape=(3,))

    # Pickle and unpickle
    pickled = pickle.dumps(space)
    unpickled = pickle.loads(pickled)

    # Verify attributes are preserved
    assert jnp.array_equal(unpickled.low, space.low)
    assert jnp.array_equal(unpickled.high, space.high)
    assert unpickled.shape == space.shape
    assert unpickled.dtype == space.dtype

    # Verify functionality works
    key = jax.random.PRNGKey(0)
    sample = unpickled.sample(key)
    assert unpickled.contains(sample)


def test_pytree_space_pickle():
    """Test that PyTreeSpace can be pickled and unpickled."""
    import pickle

    space = PyTreeSpace(
        {
            "discrete": Discrete(n=5),
            "continuous": Continuous.from_shape(low=0.0, high=1.0, shape=(2,)),
        }
    )

    # Pickle and unpickle
    pickled = pickle.dumps(space)
    unpickled = pickle.loads(pickled)

    # Verify structure is preserved
    assert "discrete" in unpickled.tree
    assert "continuous" in unpickled.tree
    assert isinstance(unpickled.tree["discrete"], Discrete)
    assert isinstance(unpickled.tree["continuous"], Continuous)

    # Verify functionality works
    key = jax.random.PRNGKey(0)
    sample = unpickled.sample(key)
    assert unpickled.contains(sample)


def test_space_pickle_with_array_parameters():
    """Test pickling spaces with array parameters."""
    import pickle

    # Discrete with array n
    discrete = Discrete(n=jnp.array([4, 5, 6], dtype=jnp.int32))
    pickled = pickle.dumps(discrete)
    unpickled = pickle.loads(pickled)
    assert jnp.allclose(unpickled.n, discrete.n)

    # Continuous with array bounds
    continuous = Continuous(
        low=jnp.array([0.0, 1.0], dtype=jnp.float32),
        high=jnp.array([1.0, 2.0], dtype=jnp.float32),
    )
    pickled = pickle.dumps(continuous)
    unpickled = pickle.loads(pickled)
    assert jnp.allclose(unpickled.low, continuous.low)
    assert jnp.allclose(unpickled.high, continuous.high)


def test_space_equality_after_pickle():
    """Test that pickled spaces maintain their properties."""
    import pickle

    space1 = Discrete(n=10)
    space2 = pickle.loads(pickle.dumps(space1))

    # While they're different objects, they should have same properties
    assert jnp.array_equal(space1.n, space2.n)
    assert space1.dtype == space2.dtype
    assert space1.shape == space2.shape

    # And should produce valid samples
    key = jax.random.PRNGKey(42)
    assert space1.contains(space2.sample(key))
    assert space2.contains(space1.sample(key))


# ============================================================================
# Tests: Hashing
# ============================================================================


def test_space_not_hashable():
    """Test that Space instances with array fields are not hashable.

    Spaces containing JAX arrays (either as n/low/high or in tree field)
    cannot be hashed. This documents expected behavior.
    """
    # Discrete with array n - not hashable
    discrete_array = Discrete(n=jnp.array([4, 5, 6], dtype=jnp.int32))
    with pytest.raises(TypeError):
        hash(discrete_array)

    # Continuous with array bounds - not hashable
    continuous_array = Continuous(
        low=jnp.array([0.0, 1.0], dtype=jnp.float32),
        high=jnp.array([1.0, 2.0], dtype=jnp.float32),
    )
    with pytest.raises(TypeError):
        hash(continuous_array)

    # PyTreeSpace - not hashable due to tree field
    pytree_space = PyTreeSpace({"a": Discrete(n=10)})
    with pytest.raises(TypeError):
        hash(pytree_space)

    # Note: Discrete/Continuous with scalar parameters ARE hashable
    # because scalars are static fields. This is actually useful behavior.
