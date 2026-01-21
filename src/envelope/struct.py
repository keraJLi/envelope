import dataclasses
from dataclasses import KW_ONLY
from typing import Any, Iterable, Iterator, Mapping, Self, Tuple

import jax

from envelope.typing import PyTree

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
    def __init_subclass__(cls, *, dataclass_kwargs: dict[str, Any] | None = None, **kw):
        super().__init_subclass__(**kw)
        # Check if this specific class (not parent) has already been processed
        if "__is_envelope_pytreenode__" in cls.__dict__:
            return
        opts = dict(frozen=True, eq=True, repr=True, slots=False)
        if dataclass_kwargs:
            opts.update(dataclass_kwargs)
        dataclasses.dataclass(cls, **opts)  # modify in place
        cls.__is_envelope_pytreenode__ = True

        data = []
        static = []
        for f in dataclasses.fields(cls):
            if f.metadata.get("pytree_node", True):
                data.append(f.name)
            else:
                static.append(f.name)

        jax.tree_util.register_dataclass(cls, data, static)

    # convenience
    def replace(self, **changes):
        return dataclasses.replace(self, **changes)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, eq=True, repr=True, slots=False)
class Container:
    _: KW_ONLY
    _extras: Mapping[str, PyTree] = dataclasses.field(default_factory=dict, repr=False)

    def __init_subclass__(cls, *, dataclass_kwargs: dict[str, Any] | None = None, **kw):
        super().__init_subclass__(**kw)
        if "__is_container_dataclass__" in cls.__dict__:
            return

        opts = dict(frozen=True, eq=True, repr=True, slots=False)
        if dataclass_kwargs:
            opts.update(dataclass_kwargs)

        dataclasses.dataclass(cls, **opts)
        cls.__is_container_dataclass__ = True
        jax.tree_util.register_pytree_node_class(cls)

    def __getattr__(self, name: str) -> PyTree:
        # bypass __getattr__ when accessing _extras to avoid recursion
        extras = object.__getattribute__(self, "_extras")
        if name in extras:
            return extras[name]
        self_name = type(self).__name__
        raise AttributeError(f"'{self_name}' object has no attribute '{name}'")

    def __dir__(self) -> Iterable[str]:
        core_names = {f.name for f in dataclasses.fields(self) if f.name != "_extras"}
        return sorted(set(super().__dir__()) | core_names | set(self._extras.keys()))

    def __iter__(self) -> Iterator[Tuple[str, PyTree]]:
        for f in dataclasses.fields(self):
            if f.name == "_extras":
                continue
            yield (f.name, getattr(self, f.name))
        # extras
        for k, v in self._extras.items():
            yield (k, v)

    def update(self, **changes: PyTree) -> Self:
        core_names = {f.name for f in dataclasses.fields(self) if f.name != "_extras"}
        core_updates: dict[str, PyTree] = {}
        extras_updates: dict[str, PyTree] = {}

        for k, v in changes.items():
            if k in core_names:
                core_updates[k] = v
            else:
                extras_updates[k] = v

        new = dataclasses.replace(self, **core_updates)
        new_extras = {**self._extras, **extras_updates}
        object.__setattr__(new, "_extras", new_extras)
        return new

    def tree_flatten(self) -> Tuple[Tuple[PyTree, ...], Tuple[Any, ...]]:
        core_fields = [f for f in dataclasses.fields(self) if f.name != "_extras"]
        core_keys = tuple(f.name for f in core_fields)
        core_vals = tuple(getattr(self, name) for name in core_keys)

        extras_keys = tuple(self._extras.keys())
        extras_vals = tuple(self._extras[k] for k in extras_keys)

        children = core_vals + extras_vals
        aux_data = (self.__class__, core_keys, extras_keys)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children: Tuple[PyTree, ...]) -> Self:
        actual_cls, core_keys, extras_keys = aux_data
        n_core = len(core_keys)

        core_vals = children[:n_core]
        extras_vals = children[n_core:]

        core_kwargs = dict(zip(core_keys, core_vals))
        extras = dict(zip(extras_keys, extras_vals))

        obj = actual_cls(**core_kwargs)
        object.__setattr__(obj, "_extras", extras)
        return obj
