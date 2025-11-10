from typing import Any, Mapping, Sequence, TypeAlias, TypeVar

import jax

Leaf = TypeVar("Leaf")
PyTree: TypeAlias = Leaf | Sequence["PyTree[Leaf]"] | Mapping[Any, "PyTree[Leaf]"]

Key: TypeAlias = jax.Array
