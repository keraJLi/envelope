"""Tests for envelope.struct module.

This module tests:
- Immutability (frozen dataclasses)
- JAX pytree integration (tree_flatten/unflatten)
- JAX transformations (vmap, jit, tree.map)
- Dataclass features (defaults, inheritance, replace)
- Serialization (pickle, repr)
"""

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from envelope.struct import FrozenPyTreeNode, field, static_field

# ============================================================================
# Test Fixtures and Helper Classes
# ============================================================================


class SimpleNode(FrozenPyTreeNode):
    """Test class with dynamic and static fields."""

    x: jax.Array  # dynamic field (pytree node)
    y: jax.Array  # dynamic field (pytree node)
    static_val: int = static_field()  # static field


class _NodeWithDefaults(FrozenPyTreeNode):
    """Helper node used to exercise default factories and static fields."""

    x: jax.Array
    y: jax.Array = field(default_factory=lambda: jnp.array([0.0]))
    scale: float = static_field(default=1.0)


class _NodeWithDefaultList(FrozenPyTreeNode):
    """Helper node for default_factory independence checks."""

    x: jax.Array
    items: list = static_field(default_factory=list)


class _BaseNode(FrozenPyTreeNode):
    """Helper base node for inheritance tests."""

    x: jax.Array
    config: str = static_field(default="base")


class _DerivedNode(_BaseNode):
    """Helper derived node for inheritance tests."""

    y: jax.Array = field(default_factory=lambda: jnp.array([0.0]))
    extra: int = static_field(default=0)


class _OrderedNode(FrozenPyTreeNode):
    """Helper node for verifying flatten ordering and stability."""

    a: jax.Array
    b: jax.Array
    static_1: int = static_field()
    c: jax.Array
    static_2: str = static_field()
    d: jax.Array


class _ComplexNode(FrozenPyTreeNode):
    """Helper node for exercising round-trip with diverse static fields."""

    array_field: jax.Array
    int_static: int = static_field()
    another_array: jax.Array
    str_static: str = static_field()
    list_static: list = static_field()


# ============================================================================
# Tests: Basic PyTreeNode Functionality
# ============================================================================


class TestPyTreeNodeBasics:
    """Test basic PyTreeNode functionality: immutability, replace, equality."""

    def test_frozen_attributes(self):
        """Test that PyTreeNode instances are frozen (immutable)."""
        node = SimpleNode(
            x=jnp.array([1.0, 2.0]), y=jnp.array([3.0, 4.0]), static_val=42
        )

        # Attempting to modify any attribute should raise FrozenInstanceError
        with pytest.raises(dataclasses.FrozenInstanceError):
            node.x = jnp.array([5.0, 6.0])

        with pytest.raises(dataclasses.FrozenInstanceError):
            node.y = jnp.array([7.0, 8.0])

        with pytest.raises(dataclasses.FrozenInstanceError):
            node.static_val = 100

    def test_replace_method(self):
        """Test the replace method works correctly."""
        node = SimpleNode(
            x=jnp.array([1.0, 2.0]), y=jnp.array([3.0, 4.0]), static_val=10
        )

        # Replace a dynamic field
        new_node = node.replace(x=jnp.array([5.0, 6.0]))
        assert jnp.allclose(new_node.x, jnp.array([5.0, 6.0]))
        assert jnp.allclose(new_node.y, jnp.array([3.0, 4.0]))  # unchanged
        assert new_node.static_val == 10  # unchanged

        # Replace a static field
        new_node2 = node.replace(static_val=20)
        assert jnp.allclose(new_node2.x, jnp.array([1.0, 2.0]))  # unchanged
        assert jnp.allclose(new_node2.y, jnp.array([3.0, 4.0]))  # unchanged
        assert new_node2.static_val == 20

        # Replace multiple fields
        new_node3 = node.replace(
            x=jnp.array([7.0, 8.0]), y=jnp.array([9.0, 10.0]), static_val=30
        )
        assert jnp.allclose(new_node3.x, jnp.array([7.0, 8.0]))
        assert jnp.allclose(new_node3.y, jnp.array([9.0, 10.0]))
        assert new_node3.static_val == 30

        # Original should be unchanged
        assert jnp.allclose(node.x, jnp.array([1.0, 2.0]))
        assert jnp.allclose(node.y, jnp.array([3.0, 4.0]))
        assert node.static_val == 10

    def test_replace_with_invalid_field(self):
        """Test that replace raises an error for non-existent fields."""
        node = SimpleNode(
            x=jnp.array([1.0, 2.0]), y=jnp.array([3.0, 4.0]), static_val=10
        )

        # Attempting to replace a non-existent field should raise TypeError
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            node.replace(nonexistent_field=42)

        with pytest.raises(TypeError, match="unexpected keyword argument"):
            node.replace(x=jnp.array([5.0, 6.0]), invalid_param="test")

    def test_equality(self):
        """Test that PyTreeNode instances can be compared for equality."""
        node1 = SimpleNode(
            x=jnp.array([1.0, 2.0]), y=jnp.array([3.0, 4.0]), static_val=10
        )
        node2 = SimpleNode(
            x=jnp.array([1.0, 2.0]), y=jnp.array([3.0, 4.0]), static_val=10
        )
        node3 = SimpleNode(
            x=jnp.array([1.0, 2.0]), y=jnp.array([3.0, 4.0]), static_val=20
        )

        # Use tree_leaves to compare the pytree structures element-by-element
        leaves1, treedef1 = jax.tree_util.tree_flatten(node1)
        leaves2, treedef2 = jax.tree_util.tree_flatten(node2)
        leaves3, treedef3 = jax.tree_util.tree_flatten(node3)

        # node1 and node2 should be equal (same static and dynamic values)
        assert treedef1 == treedef2
        assert all(jnp.allclose(l1, l2) for l1, l2 in zip(leaves1, leaves2))

        # node1 and node3 should not be equal (different static_val means different treedef)
        assert treedef1 != treedef3  # Different static values = different treedefs


# ============================================================================
# Tests: JAX Integration
# ============================================================================


class TestPyTreeNodeJAXIntegration:
    """Test JAX-specific operations: vmap, jit, tree operations."""

    def test_vmap_dynamic_attributes(self):
        """Test that vmap works correctly over dynamic (pytree node) attributes."""

        # Create a function that operates on SimpleNode
        def process(node: SimpleNode) -> jax.Array:
            return node.x + node.y

        # Create a batch of nodes
        x_batch = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_batch = jnp.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])

        # Create batched node (static_val should be the same for all)
        batched_node = SimpleNode(x=x_batch, y=y_batch, static_val=42)

        # vmap over the batched dimensions
        result = jax.vmap(process)(batched_node)

        # Expected result
        expected = jnp.array([[11.0, 22.0], [33.0, 44.0], [55.0, 66.0]])

        assert jnp.allclose(result, expected)

    def test_vmap_static_attributes(self):
        """Test that static attributes are broadcast correctly in vmap."""

        # Create a function that uses both dynamic and static fields
        def process(node: SimpleNode) -> jax.Array:
            # Static field should be the same across all batched elements
            return node.x * node.static_val + node.y

        # Create a batch of nodes
        x_batch = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y_batch = jnp.array([[10.0, 20.0], [30.0, 40.0]])

        batched_node = SimpleNode(x=x_batch, y=y_batch, static_val=5)

        # vmap should broadcast the static field across all batch elements
        result = jax.vmap(process)(batched_node)

        # Expected: x * 5 + y for each batch element
        expected = jnp.array([[15.0, 30.0], [45.0, 60.0]])

        assert jnp.allclose(result, expected)

    def test_tree_flatten_unflatten(self):
        """Test that tree_flatten and tree_unflatten work correctly."""
        node = SimpleNode(
            x=jnp.array([1.0, 2.0]), y=jnp.array([3.0, 4.0]), static_val=42
        )

        # Flatten
        children, treedef = jax.tree_util.tree_flatten(node)

        # children should contain dynamic fields (x, y)
        assert len(children) == 2
        assert jnp.allclose(children[0], jnp.array([1.0, 2.0]))
        assert jnp.allclose(children[1], jnp.array([3.0, 4.0]))

        # Unflatten
        reconstructed = jax.tree_util.tree_unflatten(treedef, children)

        assert jnp.allclose(reconstructed.x, node.x)
        assert jnp.allclose(reconstructed.y, node.y)
        assert reconstructed.static_val == node.static_val

    def test_jax_tree_map(self):
        """Test that jax.tree.map works correctly with PyTreeNode."""
        node = SimpleNode(
            x=jnp.array([1.0, 2.0]), y=jnp.array([3.0, 4.0]), static_val=10
        )

        # tree.map should only transform dynamic fields
        result = jax.tree.map(lambda x: x * 2, node)

        assert jnp.allclose(result.x, jnp.array([2.0, 4.0]))
        assert jnp.allclose(result.y, jnp.array([6.0, 8.0]))
        assert result.static_val == 10  # static field unchanged

    def test_pytree_node_with_jit(self):
        """Test that PyTreeNode works with jax.jit."""

        @jax.jit
        def process(node: SimpleNode) -> jax.Array:
            return node.x * node.static_val + node.y

        node = SimpleNode(
            x=jnp.array([1.0, 2.0]), y=jnp.array([3.0, 4.0]), static_val=2
        )
        result = process(node)

        expected = jnp.array([5.0, 8.0])  # [1*2+3, 2*2+4]
        assert jnp.allclose(result, expected)

    def test_tree_flatten_preserves_order_and_is_stable(self):
        """Ensure flatten ordering is deterministic and round-trips."""
        node = _OrderedNode(
            a=jnp.array([1.0]),
            b=jnp.array([2.0]),
            static_1=10,
            c=jnp.array([3.0]),
            static_2="test",
            d=jnp.array([4.0]),
        )

        children1, treedef1 = jax.tree_util.tree_flatten(node)
        children2, treedef2 = jax.tree_util.tree_flatten(node)

        expected_children = (
            jnp.array([1.0]),
            jnp.array([2.0]),
            jnp.array([3.0]),
            jnp.array([4.0]),
        )

        assert len(children1) == len(expected_children) == 4
        for child, expected in zip(children1, expected_children):
            assert jnp.allclose(child, expected)

        assert len(children1) == len(children2)
        assert all(jnp.allclose(c1, c2) for c1, c2 in zip(children1, children2))
        assert treedef1 == treedef2

        reconstructed = jax.tree_util.tree_unflatten(treedef1, children1)
        assert jnp.allclose(reconstructed.a, node.a)
        assert jnp.allclose(reconstructed.b, node.b)
        assert reconstructed.static_1 == node.static_1
        assert jnp.allclose(reconstructed.c, node.c)
        assert reconstructed.static_2 == node.static_2
        assert jnp.allclose(reconstructed.d, node.d)

    @pytest.mark.parametrize(
        "node",
        [
            pytest.param(
                SimpleNode(
                    x=jnp.array([1.0, 2.0, 3.0]),
                    y=jnp.array([4.0, 5.0, 6.0]),
                    static_val=99,
                ),
                id="simple-node",
            ),
            pytest.param(
                _ComplexNode(
                    array_field=jnp.array([1.0, 2.0, 3.0]),
                    int_static=42,
                    another_array=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
                    str_static="metadata",
                    list_static=[1, 2, 3],
                ),
                id="complex-node",
            ),
        ],
    )
    def test_tree_flatten_round_trip(self, node):
        """Flatten/unflatten should be a no-op for representative nodes."""
        children, treedef = jax.tree_util.tree_flatten(node)
        reconstructed = jax.tree_util.tree_unflatten(treedef, children)

        for field_info in dataclasses.fields(node):
            original_value = getattr(node, field_info.name)
            reconstructed_value = getattr(reconstructed, field_info.name)
            if isinstance(original_value, jax.Array):
                assert jnp.allclose(reconstructed_value, original_value)
            else:
                assert reconstructed_value == original_value


# ============================================================================
# Tests: Nested Structures
# ============================================================================


class TestPyTreeNodeNesting:
    """Test nested PyTreeNode structures."""

    def test_nested_pytree_nodes(self):
        """Test nested PyTreeNode structures."""

        class Outer(FrozenPyTreeNode):
            inner: SimpleNode
            z: jax.Array

        inner_node = SimpleNode(x=jnp.array([1.0]), y=jnp.array([2.0]), static_val=5)
        outer_node = Outer(inner=inner_node, z=jnp.array([3.0]))

        # Test that nested structures work with tree.map
        result = jax.tree.map(lambda x: x * 10, outer_node)

        assert jnp.allclose(result.inner.x, jnp.array([10.0]))
        assert jnp.allclose(result.inner.y, jnp.array([20.0]))
        assert result.inner.static_val == 5  # static field unchanged
        assert jnp.allclose(result.z, jnp.array([30.0]))

    def test_nested_defaults(self):
        """Test nodes with defaults containing other nodes with defaults."""

        class InnerNode(FrozenPyTreeNode):
            value: jax.Array = field(default_factory=lambda: jnp.array([0.0]))

        class OuterNode(FrozenPyTreeNode):
            data: jax.Array
            inner: InnerNode = field(default_factory=InnerNode)

        # Create outer node with default inner
        outer = OuterNode(data=jnp.array([5.0]))
        assert jnp.allclose(outer.data, jnp.array([5.0]))
        assert jnp.allclose(outer.inner.value, jnp.array([0.0]))

        # Create outer node with custom inner
        custom_inner = InnerNode(value=jnp.array([10.0]))
        outer2 = OuterNode(data=jnp.array([15.0]), inner=custom_inner)
        assert jnp.allclose(outer2.data, jnp.array([15.0]))
        assert jnp.allclose(outer2.inner.value, jnp.array([10.0]))


# ============================================================================
# Tests: Default Values
# ============================================================================


class TestPyTreeNodeDefaults:
    """Test default values and default factories."""

    def test_node_with_defaults_behavior(self):
        """Ensure default factories and static fields behave as expected."""
        explicit = _NodeWithDefaults(
            x=jnp.array([1.0, 2.0]),
            y=jnp.array([3.0, 4.0]),
            scale=2.0,
        )
        assert jnp.allclose(explicit.x, jnp.array([1.0, 2.0]))
        assert jnp.allclose(explicit.y, jnp.array([3.0, 4.0]))
        assert explicit.scale == 2.0

        # Default values populate when parameters are omitted
        defaulted = _NodeWithDefaults(x=jnp.array([5.0, 6.0]))
        assert jnp.allclose(defaulted.y, jnp.array([0.0]))
        assert defaulted.scale == 1.0

        # tree.map should only transform dynamic fields
        transformed = jax.tree.map(lambda arr: arr * 10, explicit)
        assert jnp.allclose(transformed.x, jnp.array([10.0, 20.0]))
        assert jnp.allclose(transformed.y, jnp.array([30.0, 40.0]))
        assert transformed.scale == 2.0

        # tree_flatten / tree_unflatten should round-trip the node
        children, treedef = jax.tree_util.tree_flatten(explicit)
        assert len(children) == 2
        reconstructed = jax.tree_util.tree_unflatten(treedef, children)
        assert jnp.allclose(reconstructed.x, explicit.x)
        assert jnp.allclose(reconstructed.y, explicit.y)
        assert reconstructed.scale == explicit.scale

        # vmap should work when batching NodeWithDefaults instances
        def process(node: _NodeWithDefaults) -> jax.Array:
            return node.x * node.scale + node.y

        x_batch = jnp.array([[1.0], [2.0], [3.0]])
        y_batch = jnp.zeros_like(x_batch)
        batched_node = _NodeWithDefaults(x=x_batch, y=y_batch)
        result = jax.vmap(process)(batched_node)
        expected = jnp.array([[1.0], [2.0], [3.0]])
        assert jnp.allclose(result, expected)

    def test_static_field_with_default(self):
        """Test that static_field() supports default values."""

        class NodeWithStaticDefault(FrozenPyTreeNode):
            x: jax.Array
            config: dict = static_field(default_factory=dict)
            threshold: float = static_field(default=0.5)

        # Create with defaults
        node1 = NodeWithStaticDefault(x=jnp.array([1.0]))
        assert jnp.allclose(node1.x, jnp.array([1.0]))
        assert node1.config == {}
        assert node1.threshold == 0.5

        # Create with explicit values
        node2 = NodeWithStaticDefault(
            x=jnp.array([2.0]), config={"key": "value"}, threshold=0.9
        )
        assert jnp.allclose(node2.x, jnp.array([2.0]))
        assert node2.config == {"key": "value"}
        assert node2.threshold == 0.9

    def test_mixed_required_and_default_fields(self):
        """Test nodes with mix of required and default fields."""

        class NodeMixedDefaults(FrozenPyTreeNode):
            # Required fields
            a: jax.Array
            b: int = static_field()

            # Optional fields with defaults
            c: jax.Array = field(default_factory=lambda: jnp.array([1.0]))
            d: str = static_field(default="default_value")
            e: jax.Array = field(default_factory=lambda: jnp.array([2.0, 3.0]))

        # Provide only required fields
        node1 = NodeMixedDefaults(a=jnp.array([10.0]), b=5)
        assert jnp.allclose(node1.a, jnp.array([10.0]))
        assert node1.b == 5
        assert jnp.allclose(node1.c, jnp.array([1.0]))
        assert node1.d == "default_value"
        assert jnp.allclose(node1.e, jnp.array([2.0, 3.0]))

        # Override some defaults
        node2 = NodeMixedDefaults(
            a=jnp.array([20.0]), b=10, c=jnp.array([5.0, 6.0]), d="custom"
        )
        assert jnp.allclose(node2.a, jnp.array([20.0]))
        assert node2.b == 10
        assert jnp.allclose(node2.c, jnp.array([5.0, 6.0]))
        assert node2.d == "custom"
        assert jnp.allclose(node2.e, jnp.array([2.0, 3.0]))

    def test_default_factory_independence(self):
        """Test that default_factory creates independent instances."""

        node1 = _NodeWithDefaultList(x=jnp.array([1.0]))
        node2 = _NodeWithDefaultList(x=jnp.array([2.0]))

        # Modify node1's list (if it weren't frozen)
        # Since the node is frozen, we can't modify in place, but we can check
        # that the default_factory created different instances
        assert node1.items is not node2.items
        assert node1.items == []
        assert node2.items == []


# ============================================================================
# Tests: Inheritance
# ============================================================================


class TestPyTreeNodeInheritance:
    """Test inheritance behavior of PyTreeNode."""

    def test_pytree_inheritance_core_behaviour(self):
        """Ensure inherited PyTreeNodes support core dataclass behaviours."""
        node = _DerivedNode(
            x=jnp.array([1.0, 2.0]),
            y=jnp.array([3.0, 4.0]),
            config="derived",
            extra=42,
        )

        assert jnp.allclose(node.x, jnp.array([1.0, 2.0]))
        assert jnp.allclose(node.y, jnp.array([3.0, 4.0]))
        assert node.config == "derived"
        assert node.extra == 42

        with pytest.raises(dataclasses.FrozenInstanceError):
            node.x = jnp.array([5.0])

        children, treedef = jax.tree_util.tree_flatten(node)
        assert len(children) == 2  # dynamic fields

        mapped = jax.tree.map(lambda arr: arr * 2, node)
        assert jnp.allclose(mapped.x, jnp.array([2.0, 4.0]))
        assert jnp.allclose(mapped.y, jnp.array([6.0, 8.0]))
        assert mapped.config == "derived"
        assert mapped.extra == 42

        replaced = node.replace(x=jnp.array([10.0, 20.0]), config="updated")
        assert jnp.allclose(replaced.x, jnp.array([10.0, 20.0]))
        assert jnp.allclose(replaced.y, jnp.array([3.0, 4.0]))
        assert replaced.config == "updated"
        assert replaced.extra == 42

    def test_pytree_inheritance_vmap_and_defaults(self):
        """Check vectorisation and default handling for inherited nodes."""

        node_with_defaults = _DerivedNode(x=jnp.array([5.0]))
        assert jnp.allclose(node_with_defaults.y, jnp.array([0.0]))
        assert node_with_defaults.config == "base"
        assert node_with_defaults.extra == 0

        def process(node: _DerivedNode) -> jax.Array:
            return node.x + node.y

        x_batch = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y_batch = jnp.array([[10.0, 20.0], [30.0, 40.0]])
        batched = _DerivedNode(x=x_batch, y=y_batch, config="batch", extra=99)

        result = jax.vmap(process)(batched)
        expected = jnp.array([[11.0, 22.0], [33.0, 44.0]])
        assert jnp.allclose(result, expected)
        assert batched.config == "batch"
        assert batched.extra == 99


# ============================================================================
# Tests: Serialization
# ============================================================================


class TestPyTreeNodeSerialization:
    """Test serialization: repr, hashing, and pickling."""

    def test_pytree_node_repr(self):
        """Test that PyTreeNode has a readable __repr__."""
        node = SimpleNode(
            x=jnp.array([1.0, 2.0]), y=jnp.array([3.0, 4.0]), static_val=42
        )

        repr_str = repr(node)

        # Should contain class name
        assert "SimpleNode" in repr_str

        # Should be a valid representation (dataclass default repr)
        # Check that it contains field names
        assert "x=" in repr_str or "static_val=" in repr_str

    def test_pytree_node_repr_with_defaults(self):
        """Test __repr__ with nodes that have default values."""

        class NodeWithDefaults(FrozenPyTreeNode):
            x: jax.Array
            y: jax.Array = field(default_factory=lambda: jnp.array([0.0]))
            meta: str = static_field(default="default_meta")

        node = NodeWithDefaults(x=jnp.array([1.0]))
        repr_str = repr(node)

        assert "NodeWithDefaults" in repr_str
        assert "x=" in repr_str

    def test_pytree_node_repr_nested(self):
        """Test __repr__ with nested PyTreeNode structures."""

        class Inner(FrozenPyTreeNode):
            value: jax.Array

        class Outer(FrozenPyTreeNode):
            inner: Inner
            data: jax.Array

        inner = Inner(value=jnp.array([1.0]))
        outer = Outer(inner=inner, data=jnp.array([2.0]))

        repr_str = repr(outer)

        # Should contain both class names
        assert "Outer" in repr_str
        assert "Inner" in repr_str

    def test_pickle_round_trip_basic(self):
        """Basic pickle/unpickle should preserve fields."""
        import pickle

        node = SimpleNode(
            x=jnp.array([1.0, 2.0]), y=jnp.array([3.0, 4.0]), static_val=42
        )

        # Pickle and unpickle
        pickled = pickle.dumps(node)
        unpickled = pickle.loads(pickled)

        # Verify fields are preserved
        assert jnp.allclose(unpickled.x, node.x)
        assert jnp.allclose(unpickled.y, node.y)
        assert unpickled.static_val == node.static_val

    def test_pickle_round_trip_comprehensive(self):
        """Pickling with varied field types; also verify treedef equality.

        Note: We use SimpleNode (module-level class) because local classes
        cannot be pickled in Python.
        """
        import pickle

        # Test with different values
        node = SimpleNode(
            x=jnp.array([1.0, 2.0, 3.0]),
            y=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            static_val=999,
        )

        # Pickle and unpickle
        pickled = pickle.dumps(node)
        unpickled = pickle.loads(pickled)

        # Verify all fields are preserved
        assert jnp.allclose(unpickled.x, node.x)
        assert jnp.allclose(unpickled.y, node.y)
        assert unpickled.static_val == node.static_val

        # Verify functionality still works
        new_node = unpickled.replace(static_val=1000)
        assert new_node.static_val == 1000
        assert jnp.allclose(new_node.x, node.x)

        # Also verify treedef/structure equality post-pickle
        leaves1, treedef1 = jax.tree_util.tree_flatten(node)
        leaves2, treedef2 = jax.tree_util.tree_flatten(unpickled)
        assert treedef1 == treedef2
        assert all(jnp.allclose(l1, l2) for l1, l2 in zip(leaves1, leaves2))

    def test_pytree_node_not_hashable(self):
        """Test that PyTreeNode instances are not hashable (because they're not).

        Frozen dataclasses with mutable fields (like jax.Array) are not hashable.
        This test documents the expected behavior.
        """
        node = SimpleNode(
            x=jnp.array([1.0, 2.0]), y=jnp.array([3.0, 4.0]), static_val=42
        )

        # PyTreeNode with array fields should not be hashable
        with pytest.raises(TypeError):
            hash(node)

        # Also test that we can't use it as a dict key
        with pytest.raises(TypeError):
            {node: "value"}
