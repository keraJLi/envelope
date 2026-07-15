# Struct

## Overview

Similarly to `flax.struct`, the `struct` submodule implements datastructures
that are automatically registered as JAX pytrees with `jax.tree_util`, allowing us to
pass them to JAX transformations. The two datastructures defined fullfil two main
purposes in `envelope`:

- **`FrozenPyTreeNode`** creates a dataclass with fixed fields that you can mark as
  either static or dynamic using `struct.field`.<br>
  For example, environments are instances of
  `FrozenPyTreeNode`, allowing them to have both static (structural, e.g. world size)
  and dynamic params (not affecting array shapes and control flow, e.g. wind speed).
- **`Container`** creates a dataclass with a core set of fields that will always be
  present. Additionally, instances of `Container` can be updated to hold any additional
  fields at runtime (and even within traced methods).<br>
  For example, the `step` function of an environment emits an `Info` object that is a
  `Container`, and holds observation, reward and terminated/truncated flags. Wrappers
  can add information to this, such as current episode statistics.

`Container` sorts additional field names lexicographically, so insertion order does not
change the Container node definition. JAX control-flow branches must still emit the same
field names and value pytrees.

Safe `static_field()` values must be hashable, which rejects arrays and ordinary mutable
containers. Caller-managed opaque metadata may opt out with
`static_field(unsafe=True)`.


## API Reference

::: envelope.struct.FrozenPyTreeNode

::: envelope.struct.field

::: envelope.struct.static_field

::: envelope.struct.Container
