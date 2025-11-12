from functools import cached_property
from typing import override

import jax
import jax.numpy as jnp

from jenv import spaces
from jenv.environment import Info, State
from jenv.typing import Key, PyTree
from jenv.wrappers.wrapper import Wrapper


class VmapWrapper(Wrapper):
    """
    Vectorizes a single (non-batched) environment over a fixed batch size using JAX vmap.

    - reset(key): accepts a single key (split into batch_size) or a batched key
      of shape (batch_size, 2); returns batched state and info.
    - step(state, action): expects batched state and action; returns batched outputs.
    - observation_space / action_space: prepend the batch dimension to the underlying spaces.
    """

    batch_size: int

    @override
    def reset(self, key: Key) -> tuple[State, Info]:
        # Accept single key or batched keys
        if key.shape == (2,):
            keys = jax.random.split(key, self.batch_size)
        else:
            if key.shape[0] != self.batch_size:
                raise ValueError(
                    f"reset key's leading dimension ({key.shape[0]}) must match "
                    f"batch_size ({self.batch_size})."
                )
            keys = key

        # vmap over inputs only, using the same env for all items
        state, info = jax.vmap(self.env.reset)(keys)
        return state, info

    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        next_state, info = jax.vmap(self.env.step)(state, action)
        return next_state, info

    @override
    @cached_property
    def observation_space(self) -> spaces.Space:
        return _batch_space(self.env.observation_space, self.batch_size)

    @override
    @cached_property
    def action_space(self) -> spaces.Space:
        return _batch_space(self.env.action_space, self.batch_size)


def _batch_space(space: spaces.Space, batch_size: int) -> spaces.Space:
    if isinstance(space, spaces.Discrete):
        n = space.n
        shape = (batch_size, *space.shape)

        if jnp.asarray(n).shape != ():
            n = jnp.broadcast_to(n, shape)

        return spaces.Discrete(n=n, shape=shape, dtype=space.dtype)

    if isinstance(space, spaces.Continuous):
        low = space.low
        high = space.high
        shape = (batch_size, *space.shape)

        if jnp.asarray(low).shape != ():
            low = jnp.broadcast_to(jnp.asarray(low), shape)
            high = jnp.broadcast_to(jnp.asarray(high), shape)

        return spaces.Continuous(low=low, high=high, shape=shape, dtype=space.dtype)

    if isinstance(space, spaces.PyTreeSpace):
        batched_tree = jax.tree.map(
            lambda sp: _batch_space(sp, batch_size),
            space.tree,
            is_leaf=lambda node: isinstance(node, spaces.Space),
        )
        return spaces.PyTreeSpace(batched_tree)

    raise TypeError(f"Unsupported Space type for batching: {type(space).__name__}")
