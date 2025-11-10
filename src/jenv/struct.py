from dataclasses import dataclass as _dataclass
from dataclasses import field as _dc_field
from dataclasses import fields as _dc_fields
from dataclasses import replace as _dc_replace
from typing import Any, Dict, Tuple

import jax

__all__ = ["PyTreeNode", "field", "static_field"]


def field(*, pytree_node: bool = True, **kwargs):
    """
    Dataclass field helper.
    Set pytree_node=False for static (non-transformed) fields.
    """
    meta = dict(kwargs.pop("metadata", {}) or {})
    meta["pytree_node"] = pytree_node
    return _dc_field(metadata=meta, **kwargs)


def static_field(**kwargs):
    """Shorthand for field(pytree_node=False, ...)."""
    return field(pytree_node=False, **kwargs)


class PyTreeNode:
    """
    Frozen dataclass base that is a JAX pytree node.

    Usage:
        class Foo(PyTreeNode):
            a: Any                      # pytree leaf
            b: int = static_field()     # static, not a leaf

        x = Foo(a={"w": 1.0}, b=0)
        y = x.replace(b=1)
    """

    # Turn subclasses into frozen dataclasses and register with JAX.
    def __init_subclass__(cls, *, dataclass_kwargs: Dict[str, Any] | None = None, **kw):
        super().__init_subclass__(**kw)
        # Check if this specific class (not parent) has already been processed
        if "__is_jenv_pytreenode__" in cls.__dict__:
            return
        opts = dict(frozen=True, eq=True, repr=True, slots=False)
        if dataclass_kwargs:
            opts.update(dataclass_kwargs)
        _dataclass(cls, **opts)  # modify in place
        cls.__is_jenv_pytreenode__ = True
        jax.tree_util.register_pytree_node_class(cls)

    # pytree protocol
    def tree_flatten(self) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
        children = []
        static = []
        for f in _dc_fields(self):
            v = getattr(self, f.name)
            if f.metadata.get("pytree_node", True):
                children.append(v)
            else:
                static.append(v)
        # aux_data carries static values in dataclass field order
        return tuple(children), tuple(static)

    @classmethod
    def tree_unflatten(cls, aux_data: Tuple[Any, ...], children: Tuple[Any, ...]):
        vals: Dict[str, Any] = {}
        it_children = iter(children)
        it_static = iter(aux_data)
        for f in _dc_fields(cls):
            if f.metadata.get("pytree_node", True):
                vals[f.name] = next(it_children)
            else:
                vals[f.name] = next(it_static)
        return cls(**vals)

    # convenience
    def replace(self, **changes):
        return _dc_replace(self, **changes)
