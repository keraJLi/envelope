"""Tests for PyTreeSpace."""

import jax
import jax.numpy as jnp
import pytest

from envelope.spaces import Continuous, Discrete, PyTreeSpace

# ============================================================================
# Tests: PyTreeSpace - Basic Functionality
# ============================================================================


def test_pytree_space_basic():
    """Test basic PyTreeSpace functionality."""

    space = PyTreeSpace(
        {
            "discrete": Discrete(n=jnp.array(10, dtype=jnp.int32)),
            "continuous": Continuous.from_shape(low=0.0, high=1.0, shape=(2,)),
        }
    )

    # Test sampling
    key = jax.random.PRNGKey(0)
    sample = space.sample(key)

    assert isinstance(sample, dict)
    assert "discrete" in sample
    assert "continuous" in sample

    # Check discrete sample
    assert sample["discrete"] >= 0
    assert sample["discrete"] < 10

    # Check continuous sample
    assert sample["continuous"].shape == (2,)
    assert jnp.all(sample["continuous"] >= 0.0)
    assert jnp.all(sample["continuous"] <= 1.0)

    # Test contains with valid sample
    assert space.contains(sample)

    # Test contains with invalid sample
    invalid_sample = {"discrete": -1, "continuous": jnp.array([0.5, 0.5])}
    assert not space.contains(invalid_sample)


def test_pytree_space_nested():
    """Test PyTreeSpace with nested structure."""

    space = PyTreeSpace(
        {
            "obs": {
                "position": Continuous.from_shape(low=-10.0, high=10.0, shape=(2,)),
                "velocity": Continuous.from_shape(low=-1.0, high=1.0, shape=(2,)),
            },
            "action": Discrete(n=4),
        }
    )

    # Test sampling
    key = jax.random.PRNGKey(42)
    sample = space.sample(key)

    # Check structure
    assert "obs" in sample
    assert "action" in sample
    assert "position" in sample["obs"]
    assert "velocity" in sample["obs"]

    # Check values
    assert sample["obs"]["position"].shape == (2,)
    assert sample["obs"]["velocity"].shape == (2,)
    assert 0 <= sample["action"] < 4

    # Verify sample is in space
    assert space.contains(sample)


def test_pytree_space_list():
    """Test PyTreeSpace with list structure."""

    space = PyTreeSpace(
        [
            Discrete(n=5),
            Discrete(n=10),
            Continuous.from_shape(low=0.0, high=1.0, shape=(3,)),
        ]
    )

    key = jax.random.PRNGKey(0)
    sample = space.sample(key)

    assert isinstance(sample, list)
    assert len(sample) == 3
    assert 0 <= sample[0] < 5
    assert 0 <= sample[1] < 10
    assert sample[2].shape == (3,)

    # Verify sample is in space
    assert space.contains(sample)


def test_pytree_space_tuple():
    """Test PyTreeSpace with tuple structure."""
    space = PyTreeSpace(
        (
            Discrete(n=5),
            Continuous.from_shape(low=0.0, high=1.0, shape=(2,)),
        )
    )

    key = jax.random.PRNGKey(0)
    sample = space.sample(key)

    assert isinstance(sample, tuple)
    assert len(sample) == 2
    assert 0 <= sample[0] < 5
    assert sample[1].shape == (2,)

    # Verify sample is in space
    assert space.contains(sample)


# ============================================================================
# Tests: PyTreeSpace - Structure Validation
# ============================================================================


@pytest.mark.parametrize(
    ("candidate", "message"),
    [
        pytest.param(
            {"a": 5, "c": jnp.array([0.5, 0.5])},
            "Dict key mismatch",
            id="extra-key",
        ),
        pytest.param(
            {"a": 5},
            "Dict key mismatch",
            id="missing-key",
        ),
        pytest.param(
            {"a": 5, "b": jnp.array([0.5, 0.5]), "z": 10},
            "Dict key mismatch",
            id="extra-and-missing",
        ),
        pytest.param(
            {"x": 5, "y": jnp.array([0.5, 0.5])},
            "Dict key mismatch",
            id="wrong-keys",
        ),
    ],
)
def test_pytree_space_contains_structure_errors(candidate, message):
    """Consolidated coverage for dict-based structure mismatches."""
    space = PyTreeSpace(
        {
            "a": Discrete(n=10),
            "b": Continuous.from_shape(low=0.0, high=1.0, shape=(2,)),
        }
    )

    with pytest.raises(ValueError, match=message):
        space.contains(candidate)


def test_pytree_space_contains_list_structure_mismatch():
    """Test structure validation for list-based PyTreeSpace."""
    space = PyTreeSpace(
        [
            Discrete(n=5),
            Continuous.from_shape(low=0.0, high=1.0, shape=(2,)),
        ]
    )

    # Wrong list length - JAX catches arity mismatch
    with pytest.raises(ValueError, match="arity mismatch"):
        space.contains([5])  # Too short

    with pytest.raises(ValueError, match="arity mismatch"):
        space.contains([5, jnp.array([0.5, 0.5]), 10])  # Too long

    # Wrong type (dict instead of list) - JAX catches this
    with pytest.raises(ValueError):
        space.contains({"a": 5, "b": jnp.array([0.5, 0.5])})


# ============================================================================
# Tests: PyTreeSpace - JAX Integration
# ============================================================================


def test_pytree_space_jit():
    """Test that PyTreeSpace works with jit."""
    space = PyTreeSpace(
        {
            "discrete": Discrete(n=10),
            "continuous": Continuous.from_shape(low=0.0, high=1.0, shape=(2,)),
        }
    )

    @jax.jit
    def sample_and_check(key):
        sample = space.sample(key)
        valid = space.contains(sample)
        return sample, valid

    key = jax.random.PRNGKey(0)
    sample, valid = sample_and_check(key)

    assert isinstance(sample, dict)
    assert 0 <= sample["discrete"] < 10
    assert sample["continuous"].shape == (2,)
    assert valid


def test_pytree_space_many_samples():
    """Test PyTreeSpace sampling with many seeds to ensure all samples are valid."""

    space = PyTreeSpace(
        {
            "discrete": Discrete(n=20),
            "continuous": Continuous.from_shape(low=-5.0, high=5.0, shape=(3,)),
            "nested": {
                "a": Discrete(n=3),
                "b": Continuous.from_shape(low=0.0, high=1.0, shape=(2, 2)),
            },
        }
    )

    # Generate 100 random seeds and sample
    keys = jax.random.split(jax.random.PRNGKey(123), 100)
    samples = jax.vmap(space.sample)(keys)

    # Check shapes
    assert samples["discrete"].shape == (100,)
    assert samples["continuous"].shape == (100, 3)
    assert samples["nested"]["a"].shape == (100,)
    assert samples["nested"]["b"].shape == (100, 2, 2)

    # Verify all samples are valid
    valid = jax.vmap(space.contains)(samples)
    assert jnp.all(valid), f"Found {jnp.sum(~valid)} invalid samples out of 100"


# ============================================================================
# Tests: PyTreeSpace - Array Parameters (Batched)
# ============================================================================


def test_pytree_space_batched_parameters_basic():
    """Test PyTreeSpace with vectorized space parameters (array bounds).

    When spaces have array parameters (like n=jnp.array([4, 5, 6])),
    they define multi-dimensional spaces with per-dimension bounds.
    """
    # Create a space with array parameters
    space = PyTreeSpace(
        {
            "discrete": Discrete(
                n=jnp.array(
                    [4, 5, 6], dtype=jnp.int32
                ),  # 3D space with different bounds per dim
            ),
            "continuous": Continuous(
                low=jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32),
                high=jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
            ),
        }
    )

    # Create keys for batched sampling
    keys = jax.random.split(jax.random.PRNGKey(42), 5)

    # vmap over the batched keys
    samples = jax.vmap(space.sample)(keys)

    # Check shapes - first dimension is vmap batch, second is from array parameters
    assert samples["discrete"].shape == (5, 3)  # 5 samples, 3 dimensions each
    assert samples["continuous"].shape == (5, 3)


def test_pytree_space_batched_parameters_bounds():
    """Test that PyTreeSpace with array parameters respects per-dimension bounds."""
    space = PyTreeSpace(
        {
            "discrete": Discrete(n=jnp.array([4, 5, 6], dtype=jnp.int32)),
            "continuous": Continuous(
                low=jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32),
                high=jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
            ),
        }
    )

    keys = jax.random.split(jax.random.PRNGKey(42), 5)
    samples = jax.vmap(space.sample)(keys)

    # Check discrete bounds per dimension
    assert jnp.all(samples["discrete"][:, 0] >= 0)
    assert jnp.all(samples["discrete"][:, 0] < 4)
    assert jnp.all(samples["discrete"][:, 1] >= 0)
    assert jnp.all(samples["discrete"][:, 1] < 5)
    assert jnp.all(samples["discrete"][:, 2] >= 0)
    assert jnp.all(samples["discrete"][:, 2] < 6)

    # Check continuous bounds per dimension
    assert jnp.all(samples["continuous"][:, 0] >= 0.0)
    assert jnp.all(samples["continuous"][:, 0] <= 1.0)
    assert jnp.all(samples["continuous"][:, 1] >= 1.0)
    assert jnp.all(samples["continuous"][:, 1] <= 2.0)
    assert jnp.all(samples["continuous"][:, 2] >= 2.0)
    assert jnp.all(samples["continuous"][:, 2] <= 3.0)

    # Verify all samples are valid
    valid = jax.vmap(space.contains)(samples)
    assert jnp.all(valid)


# ============================================================================
# Tests: PyTreeSpace - Properties and Methods
# ============================================================================


def test_pytree_space_tree_property():
    """Test that tree property is accessible."""
    tree_dict = {
        "discrete": Discrete(n=10),
        "continuous": Continuous.from_shape(low=0.0, high=1.0, shape=(2,)),
    }
    space = PyTreeSpace(tree_dict)

    # Access tree property
    assert space.tree == tree_dict
    assert "discrete" in space.tree
    assert "continuous" in space.tree


def test_pytree_space_shape_property():
    """Test that the cached shape property mirrors the underlying tree."""
    space = PyTreeSpace(
        {
            "scalar": Discrete(n=3),
            "vector": Continuous.from_shape(low=0.0, high=1.0, shape=(2,)),
            "tuple": (
                Discrete(n=5),
                Continuous.from_shape(low=-1.0, high=1.0, shape=(1,)),
            ),
        }
    )

    shapes_first = space.shape
    shapes_second = space.shape  # cached_property should return the same values

    assert shapes_first == shapes_second
    assert shapes_first["scalar"] == ()
    assert shapes_first["vector"] == (2,)
    assert shapes_first["tuple"][0] == ()
    assert shapes_first["tuple"][1] == (1,)


def test_pytree_space_replace():
    """Test replace method on PyTreeSpace."""
    old_tree = {"a": Discrete(n=5)}
    pytree_space = PyTreeSpace(old_tree)
    new_tree = {"b": Discrete(n=10)}
    new_pytree = pytree_space.replace(tree=new_tree)
    assert "b" in new_pytree.tree
    assert "a" not in new_pytree.tree
    assert "a" in pytree_space.tree  # Original unchanged


# ============================================================================
# Tests: PyTreeSpace - Edge Cases
# ============================================================================


def test_pytree_space_empty_dict():
    """Test PyTreeSpace with empty dictionary."""
    space = PyTreeSpace({})

    key = jax.random.PRNGKey(0)
    sample = space.sample(key)

    assert isinstance(sample, dict)
    assert len(sample) == 0
    assert space.contains(sample)


def test_pytree_space_empty_list():
    """Test PyTreeSpace with empty list."""
    space = PyTreeSpace([])

    key = jax.random.PRNGKey(0)
    sample = space.sample(key)

    assert isinstance(sample, list)
    assert len(sample) == 0
    assert space.contains(sample)


def test_pytree_space_single_element():
    """Test PyTreeSpace with single element."""
    space = PyTreeSpace({"only": Discrete(n=5)})

    key = jax.random.PRNGKey(0)
    sample = space.sample(key)

    assert isinstance(sample, dict)
    assert "only" in sample
    assert 0 <= sample["only"] < 5


def test_pytree_space_deep_nesting():
    """Test PyTreeSpace with deep nesting (5 levels)."""
    space = PyTreeSpace(
        {"level1": {"level2": {"level3": {"level4": {"level5": Discrete(n=10)}}}}}
    )

    key = jax.random.PRNGKey(42)
    sample = space.sample(key)

    # Navigate to deepest level
    value = sample["level1"]["level2"]["level3"]["level4"]["level5"]
    assert 0 <= value < 10

    # Verify contains works with deep nesting
    assert space.contains(sample)


# ============================================================================
# Tests: PyTreeSpace - Vmap Creation
# ============================================================================


def test_pytree_space_from_vmapped_function():
    """Test that PyTreeSpace can be returned from a vmapped function.

    Note: When vmap creates spaces, nested space parameters become batched.
    """

    def make_pytree_space(n):
        return PyTreeSpace(
            {
                "discrete": Discrete(n=n),
                "continuous": Continuous.from_shape(low=0.0, high=1.0, shape=(2,)),
            }
        )

    # Vmap over function that creates PyTreeSpace spaces
    batch_size = 3
    n_values = jnp.array([4, 5, 6], dtype=jnp.int32)
    spaces = jax.vmap(make_pytree_space)(n_values)

    # Verify the vmapped result works - spaces should be a batched pytree
    # The tree structure should be preserved
    key = jax.random.PRNGKey(0)
    sample = spaces.sample(key)

    # Sample should have the same structure as a single PyTreeSpace sample
    assert isinstance(sample, dict)
    assert "discrete" in sample
    assert "continuous" in sample

    # Discrete samples should be batched
    assert sample["discrete"].shape == (batch_size,)
    assert jnp.all(sample["discrete"] >= 0)
    assert jnp.all(sample["discrete"] < n_values)

    # Continuous samples should be batched
    assert sample["continuous"].shape == (batch_size, 2)
    assert jnp.all(sample["continuous"] >= 0.0)
    assert jnp.all(sample["continuous"] <= 1.0)

    # Test contains
    assert spaces.contains(sample)
