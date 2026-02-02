from functools import cached_property
from typing import override

import jax
import jax.numpy as jnp

from envelope import spaces
from envelope.environment import Info
from envelope.struct import static_field
from envelope.typing import Key, PyTree
from envelope.wrappers.wrapper import WrappedState, Wrapper


def is_single_key(key):
    # New-style typed keys have dtype like key<fry>
    if jnp.issubdtype(key.dtype, jax.dtypes.prng_key):
        return key.ndim == 0
    return key.shape == (2,)


def _split_or_keep_key(key: Key, batch_size: int) -> Key:
    if is_single_key(key):
        return jax.random.split(key, batch_size)
    elif key.shape[0] == batch_size:
        return key
    raise ValueError(
        f"reset key's leading dimension ({key.shape[0]}) must match "
        f"batch_size ({batch_size})."
    )


class VmapWrapper(Wrapper):
    """Does not forward kwargs to the underlying env. Does not wrap the state."""

    batch_size: int = static_field(kw_only=True)

    @override
    def reset(
        self, key: Key, state: PyTree | None = None, **kwargs
    ) -> tuple[WrappedState, Info]:
        keys = _split_or_keep_key(key, self.batch_size)
        state, info = jax.vmap(self.env.reset)(keys, state)
        return state, info

    @override
    def step(
        self, state: WrappedState, action: PyTree, **kwargs
    ) -> tuple[WrappedState, Info]:
        state, info = jax.vmap(self.env.step)(state, action)
        return state, info

    @override
    @cached_property
    def observation_space(self) -> spaces.Space:
        return spaces.BatchedSpace(
            space=self.env.observation_space, batch_size=self.batch_size
        )

    @override
    @cached_property
    def action_space(self) -> spaces.Space:
        return spaces.BatchedSpace(
            space=self.env.action_space, batch_size=self.batch_size
        )
