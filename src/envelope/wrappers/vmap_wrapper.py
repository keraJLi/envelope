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


class VmapWrapper(Wrapper):
    """Does not forward kwargs to the underlying env. Does not wrap the state."""

    batch_size: int = static_field(kw_only=True)

    @override
    def reset(
        self, key: Key, state: PyTree | None = None, **kwargs
    ) -> tuple[WrappedState, Info]:
        # Accept single key or batched keys
        if is_single_key(key):
            keys = jax.random.split(key, self.batch_size)
        else:
            if key.shape[0] != self.batch_size:
                raise ValueError(
                    f"reset key's leading dimension ({key.shape[0]}) must match "
                    f"batch_size ({self.batch_size})."
                )
            keys = key

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
        return spaces.BatchedSpace(space=self.env.observation_space, batch_size=self.batch_size)

    @override
    @cached_property
    def action_space(self) -> spaces.Space:
        return spaces.BatchedSpace(space=self.env.action_space, batch_size=self.batch_size)
