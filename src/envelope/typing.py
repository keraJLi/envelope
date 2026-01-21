from enum import Enum
from typing import Any, TypeAlias

import jax

PyTree: TypeAlias = Any
Key: TypeAlias = jax.Array
Array: TypeAlias = jax.Array


class BatchKind(Enum):
    """
    Batch semantics for environments.
    - VMAP: environment represents instances compatible with `jax.vmap`, for example by
      wrapping it in a `VmapWrapper`.
    - NATIVE_POOL: environment represents a batch of instances via a native pool. This
      is the case when it is wrapping a non-jax-based environment that supports native
      batching, for example those provided by envpool. Environments in this mode cannot
      be vmapped, as it would break the native batching semantics.
    """

    VMAP = "vmap"
    NATIVE_POOL = "native_pool"
