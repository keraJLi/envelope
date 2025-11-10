from abc import ABC, abstractmethod
from functools import cached_property

import jax
from jax import numpy as jnp

from jenv.struct import PyTreeNode, static_field
from jenv.typing import Key, PyTree


class Space(ABC, PyTreeNode):
    @abstractmethod
    def sample(self, key: Key) -> PyTree: ...

    @abstractmethod
    def contains(self, x: PyTree) -> bool: ...


class Discrete(Space):
    """
    A discrete space with a given number of elements. `n` can be a scalar or an array.
    If `n` is a scalar, you may pass `shape` to produce array-valued samples.

    Args:
        n: The number of elements in the space.
        shape: The shape of the space.
        dtype: The dtype of the space.
    """

    n: int | jax.Array
    shape: tuple[int, ...] = static_field(default=None)
    dtype: jnp.dtype = static_field(default=jnp.int32)

    def __post_init__(self):
        if jnp.any(self.n < 1):
            raise ValueError("n must be at least 1")

        # If shape is None, infer it from n
        if self.shape is None:
            object.__setattr__(self, "shape", jnp.asarray(self.n).shape)
        elif jnp.asarray(self.n).shape != ():
            raise ValueError("shape can only be specified when n is a scalar")

    def sample(self, key: Key) -> jax.Array:
        return jax.random.randint(key, self.shape, 0, self.n, dtype=self.dtype)

    def contains(self, x: int | jax.Array) -> bool:
        return jnp.all(x >= 0) & jnp.all(x < self.n)

    def __repr__(self) -> str:
        dtype_str = getattr(self.dtype, "__name__", str(self.dtype))
        return f"Discrete(n={self.n}, shape={self.shape}, dtype={dtype_str})"


class Continuous(Space):
    """
    A continuous space with a given lower and upper bound. `low` and `high` can be scalars or arrays.
    If `low` and `high` are scalars, you may pass `shape` to produce array-valued samples.

    Args:
        low: The lower bound of the space.
        high: The upper bound of the space.
        shape: The shape of the space.
        dtype: The dtype of the space.
    """

    low: float | jax.Array
    high: float | jax.Array
    shape: tuple[int, ...] = static_field(default=None)
    dtype: jnp.dtype = static_field(default=jnp.float32)

    def __post_init__(self):
        if jnp.asarray(self.low).shape != jnp.asarray(self.high).shape:
            raise ValueError("low and high must have the same shape")
        shape = jnp.asarray(self.low).shape

        if jnp.any(self.low > self.high):
            raise ValueError("low must be less than high")

        # If shape is None, infer it from low/high
        if self.shape is None:
            object.__setattr__(self, "shape", shape)
        elif shape != ():
            raise ValueError("shape can only be specified when low and high are scalar")

    def sample(self, key: Key) -> jax.Array:
        uniform_sample = jax.random.uniform(key, self.shape, self.dtype)
        return self.low + uniform_sample * (self.high - self.low)

    def contains(self, x: jax.Array) -> bool:
        return jnp.all(x >= self.low) & jnp.all(x <= self.high)

    def __repr__(self) -> str:
        dtype_str = getattr(self.dtype, "__name__", str(self.dtype))
        return f"Continuous(low={self.low}, high={self.high}, shape={self.shape}, dtype={dtype_str})"


class PyTreeSpace(Space):
    """A Space defined by a PyTree structure of other Spaces.

    Args:
        tree: A PyTree where all leaves are Space objects.

    Usage:
        space = PyTreeSpace({
            "action": Discrete(n=4, dtype=jnp.int32),
            "obs": Continuous(low=0.0, high=1.0, shape=(2,), dtype=jnp.float32)
        })
    """

    tree: PyTree[Space]

    def __init__(self, tree: PyTree):
        for leaf in jax.tree.leaves(tree, is_leaf=lambda x: isinstance(x, Space)):
            if not isinstance(leaf, Space):
                raise TypeError(f"Leaf {leaf} is not a Space")

        object.__setattr__(self, "tree", tree)

    def sample(self, key: Key) -> PyTree:
        leaves, treedef = jax.tree.flatten(
            self.tree, is_leaf=lambda x: isinstance(x, Space)
        )
        keys = jax.random.split(key, len(leaves))
        samples = [space.sample(key) for key, space in zip(keys, leaves)]
        return jax.tree.unflatten(treedef, samples)

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

    @cached_property
    def shape(self) -> PyTree[tuple[int, ...]]:
        return jax.tree.map(
            lambda space: space.shape,
            self.tree,
            is_leaf=lambda node: isinstance(node, Space),
        )
