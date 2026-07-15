"""Small adapter-internal helpers for stable backend metadata schemas."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp

from envelope.struct import Container
from envelope.typing import PyTree


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
