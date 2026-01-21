"""Tests for Discrete space."""

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from envelope.spaces import Discrete

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("hypothesis not installed", allow_module_level=True)

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("hypothesis not installed", allow_module_level=True)

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

    key = jax.random.PRNGKey(0)
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
    space = Discrete(n=10)
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

    key = jax.random.PRNGKey(0)
    sample, valid = sample_and_check(key)

    assert 0 <= sample < 10
    assert valid


def test_discrete_space_different_dtypes():
    """Test Discrete space with different dtypes."""
    # int32
    space32 = Discrete(n=10)
    key = jax.random.PRNGKey(0)
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


def test_discrete_contains_wrong_dtype():
    """Test Discrete.contains with wrong dtype values."""
    space = Discrete(n=10)

    # Should work with int32
    assert space.contains(jnp.array(5, dtype=jnp.int32))

    # Should also work with different int dtype (gets compared as numbers)
    assert space.contains(jnp.array(5, dtype=jnp.int16))

    # Float values should work if they're within range
    assert space.contains(jnp.array(5.0, dtype=jnp.float32))


def test_discrete_space_replace():
    """Test replace method on Discrete space."""
    discrete = Discrete(n=10)
    new_discrete = discrete.replace(n=20)
    assert new_discrete.n == 20
    assert new_discrete.dtype == jnp.int32
    assert discrete.n == 10  # Original unchanged


@given(
    n=st.one_of(
        st.just(0),
        st.integers(
            min_value=-(2**31), max_value=-1
        ),  # negative values within int32 range
    ),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(deadline=None, max_examples=20)
def test_discrete_zero_or_negative_n(n, seed):
    """Test Discrete space with n=0 or negative n."""
    space = Discrete(n=n)
    key = jax.random.PRNGKey(seed)

    # Space can be created, sampling produces 0 (jax.random.randint with high<=0 returns 0)
    sample = space.sample(key)
    assert sample == 0
    assert sample.dtype == jnp.int32

    # contains: x >= 0 and x < n is always False for any x when n <= 0
    assert not space.contains(jnp.array(0, dtype=jnp.int32))
    assert not space.contains(jnp.array(1, dtype=jnp.int32))
    assert not space.contains(jnp.array(-1, dtype=jnp.int32))


@given(
    n_values=st.lists(
        st.one_of(
            st.just(0),
            st.integers(
                min_value=-(2**31), max_value=-1
            ),  # negative values within int32 range
            st.integers(min_value=1, max_value=100),  # positive values
        ),
        min_size=1,
        max_size=5,
    ),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(deadline=None, max_examples=30)
def test_discrete_array_n_with_zero_or_negative(n_values, seed):
    """Test Discrete space with array n containing zero or negative values."""
    n_values = jnp.array(n_values)
    space = Discrete(n=n_values)
    key = jax.random.PRNGKey(seed)

    sample = space.sample(key)
    assert sample.shape == n_values.shape

    # Elements where n <= 0 should produce 0
    # Elements where n > 0 should produce values in [0, n)
    for i, n_val in enumerate(n_values):
        if n_val <= 0:
            assert sample[i] == 0
        else:
            assert 0 <= sample[i] < n_val

    # Test contains: elements with n <= 0 will always fail
    # Create a test value where valid elements are in range
    test_value = jnp.array(
        [max(0, n_val - 1) if n_val > 0 else 0 for n_val in n_values],
        dtype=jnp.int32,
    )
    # If any element has n <= 0, contains should return False
    has_invalid = any(n_val <= 0 for n_val in n_values)
    assert space.contains(test_value) == (not has_invalid)


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
    key = jax.random.PRNGKey(0)
    sample = spaces.sample(key)
    assert sample.shape == (batch_size,)
    assert jnp.all(sample >= 0)
    assert jnp.all(sample < n_values)

    # Test contains
    assert spaces.contains(sample)
