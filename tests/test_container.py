"""Robust tests for envelope.struct.Container (JAX pytree dataclass container)."""

import pickle
from dataclasses import KW_ONLY

import jax
import jax.numpy as jnp
import pytest

from envelope.struct import Container
from envelope.typing import PyTree

# ============================================================================
# Container subclasses used across tests
# ============================================================================


class InfoLike(Container):
    """Simple Container subclass to validate pytree behavior and attribute access."""

    _: KW_ONLY
    a: PyTree
    b: PyTree


class BaseContainer(Container):
    """Base Container for subclass auto-registration tests."""

    _: KW_ONLY
    x: PyTree


class DerivedContainer(BaseContainer):
    """Derived Container for subclass auto-registration tests."""

    _: KW_ONLY
    y: PyTree


class SimpleContainer(Container):
    """Generic Container used in tests."""

    _: KW_ONLY
    x: PyTree
    y: PyTree


class OuterContainer(Container):
    """Container that nests another container."""

    _: KW_ONLY
    inner: InfoLike
    z: PyTree


# ============================================================================
# Tests: Attribute semantics and update behavior
# ============================================================================


def test_container_attribute_access_and_missing():
    """Existing attributes are accessible; missing attributes raise AttributeError."""
    info = InfoLike(a=jnp.array([1]), b=jnp.array([2]))

    assert jnp.allclose(info.a, jnp.array([1]))
    assert jnp.allclose(info.b, jnp.array([2]))

    # hasattr should be False for missing (relies on AttributeError)
    assert not hasattr(info, "missing")

    with pytest.raises(AttributeError):
        _ = info.missing


def test_container_update_returns_new_and_preserves_class():
    """update returns a new instance, original is unchanged, subclass preserved."""
    info = InfoLike(a=jnp.array([1.0]), b=jnp.array([2.0]))

    updated = info.update(b=jnp.array([3.0]), c=jnp.array([4.0]))

    # Class preserved and original unchanged
    assert isinstance(updated, InfoLike)
    assert jnp.allclose(info.b, jnp.array([2.0]))
    assert not hasattr(info, "c")

    # Updated fields applied on the new instance
    assert jnp.allclose(updated.a, jnp.array([1.0]))
    assert jnp.allclose(updated.b, jnp.array([3.0]))
    assert jnp.allclose(updated.c, jnp.array([4.0]))


# ============================================================================
# Tests: PyTree protocol (flatten/unflatten) and stability
# ============================================================================


def test_container_tree_flatten_unflatten_round_trip():
    """tree_flatten / tree_unflatten should round-trip a Container accurately."""
    cont = SimpleContainer(x=jnp.array([1.0, 2.0]), y=jnp.array([3.0, 4.0]))

    children, treedef = jax.tree_util.tree_flatten(cont)
    assert len(children) == 2
    assert jnp.allclose(children[0], jnp.array([1.0, 2.0]))
    assert jnp.allclose(children[1], jnp.array([3.0, 4.0]))

    reconstructed = jax.tree_util.tree_unflatten(treedef, children)
    assert isinstance(reconstructed, SimpleContainer)
    assert jnp.allclose(reconstructed.x, cont.x)
    assert jnp.allclose(reconstructed.y, cont.y)


def test_container_treedef_and_leaf_order_behavior():
    """Treedefs depend on declared dataclass fields; keyword order should not matter."""
    c1 = SimpleContainer(x=jnp.array([1]), y=jnp.array([2]))
    c2 = SimpleContainer(y=jnp.array([2]), x=jnp.array([1]))

    children1, treedef1 = jax.tree_util.tree_flatten(c1)
    children2, treedef2 = jax.tree_util.tree_flatten(c2)

    # Treedef equality: JAX treedef comparison ignores aux data for custom nodes
    assert treedef1 == treedef2

    # Leaf order follows class field order (x before y) regardless of kwargs order
    assert jnp.allclose(children1[0], jnp.array([1]))
    assert jnp.allclose(children1[1], jnp.array([2]))
    assert jnp.allclose(children2[0], jnp.array([1]))
    assert jnp.allclose(children2[1], jnp.array([2]))

    # Both still round-trip correctly with their treedefs
    rec1 = jax.tree_util.tree_unflatten(treedef1, children1)
    assert jnp.allclose(rec1.x, c1.x)
    assert jnp.allclose(rec1.y, c1.y)

    rec2 = jax.tree_util.tree_unflatten(treedef2, children2)
    assert jnp.allclose(rec2.x, c2.x)
    assert jnp.allclose(rec2.y, c2.y)


# ============================================================================
# Tests: JAX transformations (vmap, jit, tree.map)
# ============================================================================


def test_container_vmap_over_function_returning_container_subclass():
    """vmap over a function returning a Container subclass."""

    def make_info(x):
        return InfoLike(a=x, b=x + 1)

    xs = jnp.array([1, 2, 3])
    infos = jax.vmap(make_info)(xs)

    assert jnp.allclose(infos.a, xs)
    assert jnp.allclose(infos.b, xs + 1)

    doubled = jax.tree.map(lambda y: y * 2, infos)
    assert jnp.allclose(doubled.a, xs * 2)
    assert jnp.allclose(doubled.b, (xs + 1) * 2)


def test_container_vmap_over_container_input_and_output():
    """vmap over a function taking a Container and returning a Container."""

    def step(c: SimpleContainer) -> SimpleContainer:
        return SimpleContainer(x=c.x + c.y, y=c.x - c.y)

    batched = SimpleContainer(
        x=jnp.array([1.0, 2.0, 3.0]),
        y=jnp.array([10.0, 20.0, 30.0]),
    )
    out = jax.vmap(step)(batched)

    assert jnp.allclose(out.x, jnp.array([11.0, 22.0, 33.0]))
    assert jnp.allclose(out.y, jnp.array([-9.0, -18.0, -27.0]))


def test_container_jit_works_with_input_and_return():
    """jit a function that accepts and returns Container subclasses."""

    @jax.jit
    def process(info: InfoLike) -> InfoLike:
        return InfoLike(a=info.a * 2, b=info.b + 3)

    inp = InfoLike(a=jnp.array([1.0]), b=jnp.array([5.0]))
    out = process(inp)

    assert isinstance(out, InfoLike)
    assert jnp.allclose(out.a, jnp.array([2.0]))
    assert jnp.allclose(out.b, jnp.array([8.0]))


def test_container_subclass_auto_registered_vmap_nested():
    """Subclass auto-registration should work for nested inheritance."""

    def emit(x):
        return DerivedContainer(x=x, y=x + 2)

    xs = jnp.array([0.5, 1.5])
    out = jax.vmap(emit)(xs)

    assert jnp.allclose(out.x, xs)
    assert jnp.allclose(out.y, xs + 2)


def test_container_nested_containers_tree_map_and_roundtrip():
    """Nested Containers should participate in tree operations."""
    inner = InfoLike(a=jnp.array([1.0]), b=jnp.array([2.0]))
    outer = OuterContainer(inner=inner, z=jnp.array([3.0]))

    scaled = jax.tree.map(lambda x: x * 10, outer)
    assert jnp.allclose(scaled.inner.a, jnp.array([10.0]))
    assert jnp.allclose(scaled.inner.b, jnp.array([20.0]))
    assert jnp.allclose(scaled.z, jnp.array([30.0]))

    leaves, td = jax.tree_util.tree_flatten(outer)
    rec = jax.tree_util.tree_unflatten(td, leaves)
    assert jnp.allclose(rec.inner.a, outer.inner.a)
    assert jnp.allclose(rec.inner.b, outer.inner.b)
    assert jnp.allclose(rec.z, outer.z)


# ============================================================================
# Tests: Serialization (pickle)
# ============================================================================


def test_container_pickle_round_trip():
    """Top-level Container subclasses with array leaves should be pickleable."""
    info = InfoLike(a=jnp.array([1.0, 2.0]), b=jnp.array([3.0, 4.0]))

    data = pickle.dumps(info)
    restored = pickle.loads(data)

    assert isinstance(restored, InfoLike)
    assert jnp.allclose(restored.a, info.a)
    assert jnp.allclose(restored.b, info.b)
