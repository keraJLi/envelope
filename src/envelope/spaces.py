from abc import ABC, abstractmethod
from functools import cached_property
from typing import override

import jax
from jax import numpy as jnp

from envelope.struct import FrozenPyTreeNode, static_field
from envelope.typing import Key, PyTree


class Space(ABC, FrozenPyTreeNode):
    @abstractmethod
    def sample(self, key: Key) -> PyTree: ...

    @abstractmethod
    def contains(self, x: PyTree) -> bool: ...


class Discrete(Space):
    """
    A discrete space with a given number of elements. `n` can be a scalar or an array.
    The shape and dtype of the space are inferred from `n`.

    Args:
        n: The number of elements in the space.
    """

    n: int | jax.Array

    @classmethod
    def from_shape(cls, n: int, shape: tuple[int]) -> "Discrete":
        return cls(n=jnp.full(shape, n, dtype=jnp.asarray(n).dtype))

    @property
    def shape(self) -> tuple[int, ...]:
        return jnp.asarray(self.n).shape

    @property
    def dtype(self):
        return jnp.asarray(self.n).dtype

    def sample(self, key: Key) -> jax.Array:
        return jax.random.randint(key, self.shape, 0, self.n, dtype=self.dtype)

    def contains(self, x: int | jax.Array) -> bool:
        return jnp.all(x >= 0) & jnp.all(x < self.n)

    def __repr__(self) -> str:
        return f"Discrete(shape={self.shape}, dtype={self.dtype}, n={self.n})"


class Continuous(Space):
    """
    A continuous space with a given lower and upper bound. `low` and `high` can be
    scalars or arrays. The shape and dtype of the space are inferred from `low` and
    `high`.

    Args:
        low: The lower bound of the space.
        high: The upper bound of the space.
    """

    low: float | jax.Array
    high: float | jax.Array

    @classmethod
    def from_shape(cls, low: float, high: float, shape: tuple[int]) -> "Continuous":
        return cls(
            low=jnp.full(shape, low, dtype=jnp.asarray(low).dtype),
            high=jnp.full(shape, high, dtype=jnp.asarray(high).dtype),
        )

    @property
    def dtype(self):
        if jnp.asarray(self.low).dtype != jnp.asarray(self.high).dtype:
            raise ValueError("low and high must have the same dtype")

        return jnp.asarray(self.low).dtype

    @property
    def shape(self) -> tuple[int, ...]:
        if jnp.asarray(self.low).shape != jnp.asarray(self.high).shape:
            raise ValueError("low and high must have the same shape")

        return jnp.asarray(self.low).shape

    @override
    def sample(self, key: Key) -> jax.Array:
        uniform_sample = jax.random.uniform(key, self.shape, self.dtype)
        return self.low + uniform_sample * (self.high - self.low)

    @override
    def contains(self, x: jax.Array) -> bool:
        return jnp.all((x >= jnp.asarray(self.low)) & (x <= jnp.asarray(self.high)))

    def __repr__(self) -> str:
        dtype_str = getattr(self.dtype, "__name__", str(self.dtype))
        return (
            f"Continuous(shape={self.shape}, dtype={dtype_str}, "
            f"low={self.low}, high={self.high})"
        )


class PyTreeSpace(Space):
    """A Space defined by a PyTree structure of other Spaces.

    Args:
        tree: A PyTree with Discrete or Continuous leaves.

    Usage:
        space = PyTreeSpace({
            "action": Discrete(n=4),
            "obs": Continuous(low=0.0, high=1.0, shape=(2,))
        })
    """

    tree: PyTree

    def __post_init__(self):
        leaves = jax.tree.leaves(self.tree, is_leaf=lambda x: isinstance(x, Space))
        for leaf in leaves:
            if not isinstance(leaf, (Discrete, Continuous)):
                raise TypeError(
                    f"PyTreeSpace leaves must be Discrete or Continuous, got {type(leaf).__name__}"
                )

    @override
    def sample(self, key: Key) -> PyTree:
        leaves, treedef = jax.tree.flatten(
            self.tree, is_leaf=lambda x: isinstance(x, Space)
        )
        keys = jax.random.split(key, len(leaves))
        samples = [space.sample(key) for key, space in zip(keys, leaves)]
        return jax.tree.unflatten(treedef, samples)

    @override
    def contains(self, x: PyTree) -> bool:
        # Use tree.map to check containment for each space-value pair
        contains = jax.tree.map(
            lambda space, xi: space.contains(xi),
            self.tree,
            x,
            is_leaf=lambda node: isinstance(node, Space),
        )
        return jnp.all(jnp.array(jax.tree.leaves(contains)))

    def __repr__(self) -> str:
        """Return a string representation showing the tree structure."""
        return f"{self.__class__.__name__}({self.tree!r})"

    @property
    def shape(self) -> PyTree:
        return jax.tree.map(
            lambda space: space.shape,
            self.tree,
            is_leaf=lambda node: isinstance(node, Space),
        )

    @property
    def dtype(self) -> PyTree:
        return jax.tree.map(
            lambda space: space.dtype,
            self.tree,
            is_leaf=lambda node: isinstance(node, Space),
        )


def _peel_batched(space: "BatchedSpace") -> tuple[tuple[int, ...], Space]:
    """Collect batch dimensions and return (batch_dims_tuple, base_space)."""
    dims: list[int] = []
    s: Space = space
    while isinstance(s, BatchedSpace):
        dims.append(s.batch_size)
        s = s.space
    return tuple(dims), s


class BatchedSpace(Space):
    """
    A view that adds a leading batch dimension to a base Space without
    materializing or broadcasting its parameters.
    """

    space: Space
    batch_size: int = static_field()

    def sample(self, key: Key) -> PyTree:
        # Accept single PRNGKey or a batch of keys shaped (batch_size, 2)
        if getattr(key, "shape", ()) == (2,):
            keys = jax.random.split(key, self.batch_size)
        else:
            if key.shape[0] != self.batch_size:
                raise ValueError(
                    f"sample key's leading dimension ({key.shape[0]}) must match "
                    f"batch_size ({self.batch_size})."
                )
            keys = key
        return jax.vmap(self.space.sample)(keys)

    def contains(self, x: PyTree) -> bool:
        # x is expected to be batched on the leading dimension
        result = jax.vmap(self.space.contains)(x)
        return jnp.all(jnp.asarray(result))

    @cached_property
    def shape(self) -> PyTree:
        batch_dims, base = _peel_batched(self)
        if isinstance(base, PyTreeSpace):
            return jax.tree.map(
                lambda space: batch_dims + space.shape,
                base.tree,
                is_leaf=lambda node: isinstance(node, Space),
            )
        return batch_dims + base.shape

    @property
    def dtype(self) -> PyTree:
        _, base = _peel_batched(self)
        return base.dtype

    def __repr__(self) -> str:
        return f"BatchedSpace(space={self.space!r}, batch_size={self.batch_size})"
