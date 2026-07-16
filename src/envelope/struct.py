import dataclasses
from dataclasses import KW_ONLY
from typing import (
    Any,
    ClassVar,
    Iterable,
    Iterator,
    Mapping,
    Self,
    Tuple,
    dataclass_transform,
)

import jax
import jax.numpy as jnp
from jax.tree_util import GetAttrKey

from envelope.typing import PyTree

__all__ = ["Container", "FrozenPyTreeNode", "field", "static_field", "zeros_like"]


def _register_pytree_dataclass(
    cls: type[Any], data_fields: Iterable[str], static_fields: Iterable[str]
) -> None:
    """Register a dataclass without routing PyTree reconstruction through ``__init__``.

    JAX may unflatten a PyTree with opaque sentinel objects while determining vmap
    axes.  Those sentinels are structural bookkeeping rather than user-provided
    field values, so constructor and ``__post_init__`` validation must not see them.
    Static values remain PyTree auxiliary data and therefore retain the semantics of
    ``register_dataclass``.
    """
    data_names = tuple(data_fields)
    static_names = tuple(static_fields)

    def flatten_with_keys(
        node: Any,
    ) -> tuple[tuple[tuple[Any, Any], ...], tuple[Any, ...]]:
        children = tuple((GetAttrKey(name), getattr(node, name)) for name in data_names)
        static_values = tuple(getattr(node, name) for name in static_names)
        return children, static_values

    def flatten(node: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        children = tuple(getattr(node, name) for name in data_names)
        static_values = tuple(getattr(node, name) for name in static_names)
        return children, static_values

    def unflatten(static_values: Iterable[Any], children: Iterable[Any]) -> Any:
        # Bypass dataclass initialization only on JAX's reconstruction path. Normal
        # calls to ``cls(...)`` still execute the generated initializer and all
        # ``__post_init__`` validation.
        node = object.__new__(cls)
        for name, value in zip(data_names, children, strict=True):
            object.__setattr__(node, name, value)
        for name, value in zip(static_names, static_values, strict=True):
            object.__setattr__(node, name, value)
        return node

    jax.tree_util.register_pytree_with_keys(
        cls,
        flatten_with_keys,
        unflatten,
        flatten_func=flatten,
    )


def field(*, pytree_node: bool = True, **kwargs: Any) -> Any:
    """
    Dataclass field helper. See `typing.FrozenPyTreeNode` for more details.
    Set `pytree_node=False` for static (non-transformed) fields.
    """
    meta = dict(kwargs.pop("metadata", {}) or {})
    meta["pytree_node"] = pytree_node
    return dataclasses.field(metadata=meta, **kwargs)


def static_field(*, unsafe: bool = False, **kwargs: Any) -> Any:
    """Declare a static (non-transformed) dataclass field.

    Static values become part of a JAX pytree definition, so they must normally be
    hashable. ``unsafe=True`` is an explicit escape hatch for opaque values whose
    stability is managed by the caller.
    """
    meta = dict(kwargs.pop("metadata", {}) or {})
    meta["static_field_unsafe"] = unsafe
    return field(pytree_node=False, metadata=meta, **kwargs)


@dataclass_transform(field_specifiers=(field, static_field), frozen_default=True)
class FrozenPyTreeNode:
    """
    Frozen dataclass base that is a JAX pytree node. Fields can be declared as either
    dynamic (pytree nodes) or static (not pytree nodes) using the `field` and
    `static_field` helpers.

    Usage:
        ```python
        class Foo(FrozenPyTreeNode):
            a: Any                      # pytree leaf
            b: int = static_field()     # static, not a leaf

        x = Foo(a={"w": 1.0}, b=0)
        y = x.replace(b=1)
        ```

    Subclasses that define ``__post_init__`` must call ``super().__post_init__()``
    so static fields are validated.
    """

    # Type checkers can recognize instances as valid inputs to ``dataclasses.fields``
    # and ``dataclasses.replace`` if we define this.
    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[Any]]]

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

        _register_pytree_dataclass(cls, data, static)

    def __post_init__(self):
        self._validate_static_fields()

    def _validate_static_fields(self) -> None:
        for dataclass_field in dataclasses.fields(self):
            if dataclass_field.metadata.get("pytree_node", True):
                continue
            if dataclass_field.metadata.get("static_field_unsafe", False):
                continue

            value = getattr(self, dataclass_field.name)
            try:
                hash(value)
            except TypeError as error:
                raise TypeError(
                    f"static field {dataclass_field.name!r} on "
                    f"{type(self).__name__} must be hashable; use "
                    "static_field(unsafe=True) for caller-managed opaque metadata"
                ) from error

    # convenience
    def replace(self, **changes):
        """Shorthand for `dataclasses.replace(self, **changes)`."""
        return dataclasses.replace(self, **changes)


def zeros_like(tree: PyTree) -> PyTree:
    """Create a same-structure, same-shape/dtype zero tree."""
    return jax.tree.map(lambda value: jnp.zeros_like(jnp.asarray(value)), tree)


@dataclass_transform(frozen_default=True)
@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, eq=True, repr=True, slots=False)
class Container:
    """
    This class implements a container for arbitrary PyTree-valued fields. While
    subclasses can define arbitrary *core* fields, instances of this class can be
    updated to hold any additional *extras* fields.

    Usage example:
        ```python
        class Foo(Container):
            bar: int

        foo = Foo(bar=1)
        foo = foo.update(bar=2, baz=3.0)
        print(foo)  # Foo(bar=2, baz=3.0)
        ```
    """

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
        for key in sorted(self._extras):
            yield (key, self._extras[key])

    def __str__(self) -> str:
        core_str = super().__str__()
        if not self._extras:
            return core_str
        extras_str = (f"{key}={self._extras[key]!r}" for key in sorted(self._extras))
        extras_str = f", {', '.join(extras_str)}"
        return f"{core_str[:-1]}{extras_str})"  # remove closing parenthesis from core

    def update(self, **changes: PyTree) -> Self:
        """Update the container with new values. The `changes` overwrite fields in the
        container, both for core fields and extras. The annotation describes each
        keyword value; Python collects those values into the ``changes`` dictionary.

        Args:
            **changes: A dictionary of field names and values to update.

        Returns:
            A new instance of the container with the updated values.
        """
        core_names = {f.name for f in dataclasses.fields(self) if f.name != "_extras"}
        core_updates: dict[str, PyTree] = {}
        extras_updates: dict[str, PyTree] = {}

        for k, v in changes.items():
            if k in core_names:
                core_updates[k] = v
            else:
                extras_updates[k] = v

        new = dataclasses.replace(self, **core_updates)
        # Dictionaries preserve insertion order. Rebuilding from sorted pairs gives
        # equal schemas the same flattening order regardless of update order.
        new_extras = dict(sorted({**self._extras, **extras_updates}.items()))
        object.__setattr__(new, "_extras", new_extras)
        return new

    def tree_flatten(self) -> Tuple[Tuple[PyTree, ...], Tuple[Any, ...]]:
        core_fields = [f for f in dataclasses.fields(self) if f.name != "_extras"]
        core_keys = tuple(f.name for f in core_fields)
        core_vals = tuple(getattr(self, name) for name in core_keys)

        extras_keys = tuple(self._extras)
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
