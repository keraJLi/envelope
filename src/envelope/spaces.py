from abc import ABC, abstractmethod
from typing import override

import jax
from jax import numpy as jnp

from envelope.struct import FrozenPyTreeNode, static_field
from envelope.typing import Key, PyTree


class Space(ABC, FrozenPyTreeNode):
    """Base class for immutable JAX-compatible spaces.

    Space constructors assume their arguments are internally consistent. In
    particular, callers are responsible for valid bounds and cardinalities, matching
    shapes and dtypes, positive batch sizes, and matching PyTree structures.
    """

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
    def contains(self, x: PyTree) -> jax.Array:
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
    def from_shape(cls, n: int | jax.Array, shape: tuple[int, ...]) -> "Discrete":
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

    def contains(self, x: int | jax.Array) -> jax.Array:
        try:
            candidate = jnp.asarray(x)
        except (TypeError, ValueError):
            return jnp.asarray(False)

        if candidate.shape != self.shape:
            return jnp.asarray(False)
        if not jnp.issubdtype(candidate.dtype, jnp.integer):
            return jnp.asarray(False)
        return jnp.all(candidate >= 0) & jnp.all(candidate < self.n)

    def __repr__(self) -> str:
        return f"Discrete(shape={self.shape}, dtype={self.dtype}, n={self.n})"


class Continuous(Space):
    """A continuous space with elementwise lower and upper bounds.

    ``low`` and ``high`` can be scalars or arrays. Their shared shape and dtype define
    the space. Sampling treats each element independently, so a single space may mix
    finite, one-sided, and unbounded dimensions.

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
        """Sample independently from every dimension of the space.

        The distribution for each element depends on its bounds:

        - Two finite bounds use a uniform distribution over the interval.
        - A finite lower bound uses that bound plus a unit exponential sample.
        - A finite upper bound uses that bound minus a unit exponential sample.
        - Two infinite bounds use a standard normal distribution.

        One-sided and unbounded samples are clipped to the finite range representable
        by the space's dtype.

        Args:
            key: JAX pseudorandom key.

        Returns:
            An array with the space's shape and dtype.
        """
        uniform_key, normal_key, exponential_key = jax.random.split(key, 3)
        low = jnp.asarray(self.low)
        high = jnp.asarray(self.high)
        finite_low = jnp.isfinite(low)
        finite_high = jnp.isfinite(high)

        uniform = jax.random.uniform(uniform_key, self.shape, self.dtype)
        normal = jax.random.normal(normal_key, self.shape, self.dtype)
        exponential = jax.random.exponential(exponential_key, self.shape, self.dtype)

        # A convex combination avoids overflow for wide but finite intervals.
        bounded = (1 - uniform) * low + uniform * high
        dtype_info = jnp.finfo(self.dtype)
        lower_bounded = jnp.minimum(low + exponential, dtype_info.max)
        upper_bounded = jnp.maximum(high - exponential, dtype_info.min)
        unbounded = jnp.clip(normal, dtype_info.min, dtype_info.max)

        return jnp.where(
            finite_low & finite_high,
            bounded,
            jnp.where(
                finite_low,
                lower_bounded,
                jnp.where(finite_high, upper_bounded, unbounded),
            ),
        )

    @override
    def contains(self, x: jax.Array) -> jax.Array:
        try:
            candidate = jnp.asarray(x)
        except (TypeError, ValueError):
            return jnp.asarray(False)

        if candidate.shape != self.shape:
            return jnp.asarray(False)

        is_integer = jnp.issubdtype(candidate.dtype, jnp.integer)
        is_floating = jnp.issubdtype(candidate.dtype, jnp.floating)
        if not (is_integer or is_floating):
            return jnp.asarray(False)

        return jnp.all(
            (candidate >= jnp.asarray(self.low)) & (candidate <= jnp.asarray(self.high))
        )

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
        super().__post_init__()

    @override
    def sample(self, key: Key) -> PyTree:
        leaves, treedef = jax.tree.flatten(
            self.tree, is_leaf=lambda x: isinstance(x, Space)
        )
        keys = jax.random.split(key, len(leaves))
        samples = [space.sample(key) for key, space in zip(keys, leaves)]
        return jax.tree.unflatten(treedef, samples)

    @override
    def contains(self, x: PyTree) -> jax.Array:
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


def rebatch(space: Space, batch_dims: tuple[int, ...]) -> Space:
    """Reapply batch dimensions returned by ``peel_batched``."""
    for batch_dim in reversed(batch_dims):
        space = BatchedSpace(space=space, batch_size=batch_dim)
    return space


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
    def contains(self, x: PyTree) -> jax.Array:
        """
        `BatchedSpace.contains` checks if each entry of `x` along the leading dimension
        is contained in the base (unbatched) `Space`.
        """
        result = jax.vmap(self.space.contains)(x)
        return jnp.all(jnp.asarray(result))

    @property
    @override
    def shape(self) -> PyTree:
        """
        The shape of the `BatchedSpace` is the leading batch dimension prepended to the
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

    @property
    @override
    def dtype(self) -> PyTree:
        _, base = peel_batched(self)
        return base.dtype

    def __repr__(self) -> str:
        return f"BatchedSpace(space={self.space!r}, batch_size={self.batch_size})"
