"""Small adapter-internal helpers for stable backend metadata schemas."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, Callable, TypeVar, cast

import jax
import jax.numpy as jnp

from envelope.struct import Container
from envelope.typing import PyTree

_T = TypeVar("_T")


def backend_container(metadata: Any) -> Container:
    """Normalize suite metadata to a dot-accessible, JAX-compatible namespace."""
    if isinstance(metadata, Container):
        return metadata
    if dataclasses.is_dataclass(metadata) and not isinstance(metadata, type):
        values = {
            item.name: getattr(metadata, item.name)
            for item in dataclasses.fields(metadata)
        }
    elif isinstance(metadata, Mapping):
        values = dict(metadata)
    else:
        values = {"value": metadata}

    if not all(isinstance(key, str) for key in values):
        return Container().update(value=metadata)
    return Container().update(**values)


def placeholder_like(tree: PyTree) -> PyTree:
    """Create a same-structure, same-shape/dtype placeholder for backend metadata."""

    def placeholder_leaf(value: Any):
        return jnp.zeros_like(jnp.asarray(value))

    return jax.tree.map(placeholder_leaf, tree)


def replace_backend_params(value: _T, **changes: Any) -> _T:
    """Call a backend's immutable ``replace`` API without weakening its public type.

    Several adapter dependencies provide dataclass-like parameter objects whose
    runtime ``replace`` method is absent from their published type information.
    Keep that dynamic boundary isolated here while preserving the concrete input type
    for callers.
    """
    replace = cast(Callable[..., _T], getattr(value, "replace"))
    return replace(**changes)
