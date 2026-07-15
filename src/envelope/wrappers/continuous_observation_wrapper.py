from functools import cached_property
from typing import override

import jax
import jax.numpy as jnp

from envelope.environment import Info, State
from envelope.spaces import BatchedSpace, Continuous, Discrete, Space, peel_batched
from envelope.typing import Key, PyTree
from envelope.wrappers.wrapper import Wrapper


def to_float(obs: PyTree) -> PyTree:
    return jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float32), obs)


def to_continuous(space: Discrete | Continuous) -> Continuous:
    if isinstance(space, Continuous):
        low = jnp.asarray(space.low, dtype=jnp.float32)
        high = jnp.asarray(space.high, dtype=jnp.float32)
        return Continuous(low=low, high=high)
    if isinstance(space, Discrete):
        low = jnp.zeros_like(space.n, dtype=jnp.float32)
        high = jnp.asarray(space.n - 1, dtype=jnp.float32)
        return Continuous(low=low, high=high)
    raise TypeError(f"Unsupported observation space: {type(space).__name__}")


class ContinuousObservationWrapper(Wrapper):
    @override
    def init(self, key: Key) -> tuple[State, Info]:
        state, info = self.env.init(key)
        info = info.update(obs=to_float(info.obs))
        return state, info

    @override
    def reset(self, state: State, key: Key) -> tuple[State, Info]:
        state, info = self.env.reset(state, key)
        info = info.update(obs=to_float(info.obs))
        return state, info

    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        state, info = self.env.step(state, action)
        info = info.update(obs=to_float(info.obs))
        return state, info

    @cached_property
    @override
    def observation_space(self) -> Space:
        batch_dims, base = peel_batched(self.env.observation_space)

        def is_leaf(x):
            return isinstance(x, (Discrete, Continuous))

        space = jax.tree.map(to_continuous, base, is_leaf=is_leaf)

        for batch_dim in reversed(batch_dims):
            space = BatchedSpace(space, batch_dim)
        return space
