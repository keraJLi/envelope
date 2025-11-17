import dataclasses
from typing import Any, Dict

import jax

from jenv.typing import PyTree

__all__ = ["FrozenPyTreeNode", "field", "static_field", "Container"]


def field(*, pytree_node: bool = True, **kwargs):
    """
    Dataclass field helper.
    Set pytree_node=False for static (non-transformed) fields.
    """
    meta = dict(kwargs.pop("metadata", {}) or {})
    meta["pytree_node"] = pytree_node
    return dataclasses.field(metadata=meta, **kwargs)


def static_field(**kwargs):
    """Shorthand for field(pytree_node=False, ...)."""
    return field(pytree_node=False, **kwargs)


class FrozenPyTreeNode:
    """
    Frozen dataclass base that is a JAX pytree node.

    Usage:
        class Foo(FrozenPyTreeNode):
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
        dataclasses.dataclass(cls, **opts)  # modify in place
        cls.__is_jenv_pytreenode__ = True
        jax.tree_util.register_pytree_node_class(cls)

    # pytree protocol honoring static_field (metadata["pytree_node"]=False)
    def tree_flatten(self):
        children = []
        static = []
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if f.metadata.get("pytree_node", True):
                children.append(v)
            else:
                static.append(v)
        # aux_data carries static values in dataclass field order
        return tuple(children), tuple(static)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        vals: Dict[str, Any] = {}
        it_children = iter(children)
        it_static = iter(aux_data)
        for f in dataclasses.fields(cls):
            if f.metadata.get("pytree_node", True):
                vals[f.name] = next(it_children)
            else:
                vals[f.name] = next(it_static)
        return cls(**vals)

    # convenience
    def replace(self, **changes):
        return dataclasses.replace(self, **changes)


@jax.tree_util.register_pytree_node_class
class Container:
    def __init__(self, **fields):
        self._fields = fields

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        jax.tree_util.register_pytree_node_class(cls)

    def __getattr__(self, name: str) -> PyTree:
        # Avoid recursion during pickling/unpickling or early init before _fields exists
        fields = self.__dict__.get("_fields", None)
        if fields is not None:
            try:
                return fields[name]
            except KeyError:
                pass
        raise AttributeError(name)

    def update(self, **changes: PyTree) -> "Container":
        new_fields = {**self._fields, **changes}
        return self.__class__(**new_fields)

    def tree_flatten(self):
        return self._fields.values(), self._fields.keys()

    @classmethod
    def tree_unflatten(cls, keys, values):
        return cls(**dict(zip(keys, values)))
