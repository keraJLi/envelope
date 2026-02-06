from typing import Any, TypeAlias

import jax

PyTree: TypeAlias = Any
Key: TypeAlias = jax.Array  # with jnp.issubdtype(key.dtype, jax.dtypes.prng_key)
Array: TypeAlias = jax.Array
