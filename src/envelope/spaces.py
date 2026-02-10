from abc import ABC, abstractmethod
from functools import cached_property
from typing import override

import jax
from jax import numpy as jnp

from envelope.struct import FrozenPyTreeNode, static_field
from envelope.typing import Key, PyTree


class Space(ABC, FrozenPyTreeNode):
    """Base class for all spaces. Spaces are immutable and hashable."""

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...] | PyTree:
        """The shape of the space."""

    @property
    @abstractmethod
    def dtype(self) -> jnp.dtype | PyTree:
        """The dtype of the space."""

    @abstractmethod
    def sample(self, key: Key) -> PyTree:
        """Sample a random element from the space."""

    @abstractmethod
    def contains(self, x: PyTree) -> bool:
        """Check if `self` contains a sample `x`."""


class Discrete(Space):
    """
    A discrete space with a given number of elements. `n` can be a scalar or an array.
    The shape and dtype of the space are inferred from `n`.

    Args:
        n (int | jax.Array): The number of elements in the space.
    """

    n: int | jax.Array

    @classmethod
    def from_shape(cls, n: int, shape: tuple[int, ...]) -> "Discrete":
        """
        Create a Discrete space from a shape and a number of elements. This is
        a shorthand for `Discrete` with `n` being expanded to the shape.
        """
        return cls(n=jnp.full(shape, n, dtype=jnp.asarray(n).dtype))

    @property
    def shape(self) -> tuple[int, ...]:
        return jnp.asarray(self.n).shape

    @property
    def dtype(self) -> jnp.dtype:
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
        low (float | jax.Array): The lower bound of the space.
        high (float | jax.Array): The upper bound of the space.
    """

    low: float | jax.Array
    high: float | jax.Array

    @classmethod
    def from_shape(
        cls, low: float, high: float, shape: tuple[int, ...]
    ) -> "Continuous":
        """
        Create a Continuous space from a shape and a lower and upper bound. This is a
        shorthand for `Continuous` with `low` and `high` being expanded to the shape.
        """
        return cls(
            low=jnp.full(shape, low, dtype=jnp.asarray(low).dtype),
            high=jnp.full(shape, high, dtype=jnp.asarray(high).dtype),
        )

    @property
    def dtype(self) -> jnp.dtype:
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
    """
    A Space defined by a PyTree structure of other Spaces. While `PyTreeSpace.tree`
    might be an aribrarily nested PyTree, it's leaves must only be `Discrete` or
    `Continuous`, and not contain `PyTreeSpace` or `BatchedSpace` as leaves. The shape
    and dtype of the `PyTreeSpace` are PyTrees of the same structure, containing the
    shape and dtype of the leaves.

    Args:
        tree (PyTree): A PyTree with `Discrete` or `Continuous` leaves.
    """

    tree: PyTree

    def __post_init__(self):
        leaves = jax.tree.leaves(self.tree, is_leaf=lambda x: isinstance(x, Space))
        for leaf in leaves:
            if not isinstance(leaf, (Discrete, Continuous)):
                raise TypeError(
                    f"PyTreeSpace leaves must be Discrete or Continuous,"
                    f"got {type(leaf).__name__}"
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
        return f"{self.__class__.__name__}({self.tree!r})"

    @property
    def shape(self) -> PyTree:
        """A PyTree of the same structure as `tree`, containing each leaf's shape."""
        return jax.tree.map(
            lambda space: space.shape,
            self.tree,
            is_leaf=lambda node: isinstance(node, Space),
        )

    @property
    def dtype(self) -> PyTree:
        """A PyTree of the same structure as `tree`, containing each leaf's dtype."""
        return jax.tree.map(
            lambda space: space.dtype,
            self.tree,
            is_leaf=lambda node: isinstance(node, Space),
        )


def peel_batched(space: Space) -> tuple[tuple[int, ...], Space]:
    """Collect batch dimensions and return (batch_dims_tuple, base_space)."""
    dims: list[int] = []
    s: Space = space
    while isinstance(s, BatchedSpace):
        dims.append(s.batch_size)
        s = s.space
    return tuple(dims), s


class BatchedSpace(Space):
    """
    A view that adds a leading batch dimension to a base `Space` without
    materializing or broadcasting its parameters.

    Args:
        space (Space): The underlying base space.
        batch_size (int): The leading batch dimension.
    """

    space: Space
    batch_size: int = static_field()

    @override
    def sample(self, key: Key) -> PyTree:
        """
        Sample a batch of samples from the wrapped `Space`. You may pass a single key
        or a batch of keys shaped `(batch_size, ...)`.
        """
        if not jnp.issubdtype(key.dtype, jax.dtypes.prng_key):
            raise ValueError("key must be a (new-style) `jax.random.key`.")

        # Accept single key or a batch of keys shaped (batch_size, )
        if key.shape == ():
            keys = jax.random.split(key, self.batch_size)
        elif key.shape[0] == self.batch_size:
            keys = key
        else:
            raise ValueError(
                f"sample key's leading dimension ({key.shape[0]}) must match "
                f"batch_size ({self.batch_size})."
            )
        return jax.vmap(self.space.sample)(keys)

    @override
    def contains(self, x: PyTree) -> bool:
        """
        `BatchedSpace.contains` checks if each entry of `x` along the leading dimension
        is contained in the base (unbatched) `Space`.
        """
        result = jax.vmap(self.space.contains)(x)
        return jnp.all(jnp.asarray(result))

    @override
    @cached_property
    def shape(self) -> tuple[int, ...] | PyTree:
        """
        The shape of the `BatchedSpace` is the leading batch dimension prepend the
        shape of the wrapped `Space`. If the wrapped `Space` is a `PyTreeSpace`, the
        shape is a PyTree of the same structure, with the leading batch dimension
        prepended to the shape of each leaf `Space`.
        """
        batch_dims, base = peel_batched(self)
        if isinstance(base, PyTreeSpace):
            return jax.tree.map(
                lambda space: batch_dims + space.shape,
                base.tree,
                is_leaf=lambda node: isinstance(node, Space),
            )
        return batch_dims + base.shape

    @override
    @property
    def dtype(self) -> jnp.dtype | PyTree:
        """The dtype of the base space (batch dimensions don't affect dtype)."""
        _, base = peel_batched(self)
        return base.dtype

    def __repr__(self) -> str:
        return f"BatchedSpace(space={self.space!r}, batch_size={self.batch_size})"
