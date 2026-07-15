from functools import cached_property
from typing import override

import jax
import jax.numpy as jnp

from envelope.environment import Info, State
from envelope.spaces import BatchedSpace, Continuous, Discrete, Space, peel_batched
from envelope.typing import Key, PyTree
from envelope.wrappers.wrapper import Wrapper


def flatten_x(x: PyTree, batch_ndim: int = 0):
    leaves = jax.tree.leaves(x)
    xs = [
        jnp.asarray(leaf).reshape(jnp.asarray(leaf).shape[:batch_ndim] + (-1,))
        for leaf in leaves
    ]
    x = jnp.concatenate(xs, axis=-1)
    return x


class FlattenObservationWrapper(Wrapper):
    def _flatten_obs(self, obs: PyTree) -> jax.Array:
        batch_dims, _ = peel_batched(self.env.observation_space)
        return flatten_x(obs, batch_ndim=len(batch_dims))

    @override
    def init(self, key: Key) -> tuple[State, Info]:
        state, info = self.env.init(key)
        info = info.update(obs=self._flatten_obs(info.obs))
        return state, info

    @override
    def reset(self, state: State, key: Key) -> tuple[State, Info]:
        state, info = self.env.reset(state, key)
        info = info.update(obs=self._flatten_obs(info.obs))
        return state, info

    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        state, info = self.env.step(state, action)
        info = info.update(obs=self._flatten_obs(info.obs))
        return state, info

    @cached_property
    @override
    def observation_space(self) -> Space:
        batch_dims, base = peel_batched(self.env.observation_space)

        def is_leaf(x):
            spaces = (Continuous, Discrete)
            return isinstance(x, spaces)

        spaces = jax.tree.leaves(base, is_leaf=is_leaf)
        obs_cls = type(spaces[0])

        if not all(isinstance(space, obs_cls) for space in spaces):
            raise ValueError(
                "Mixed observation trees must be converted with "
                "ContinuousObservationWrapper before flattening"
            )

        if obs_cls == Continuous:
            lows = [jnp.asarray(s.low).reshape(-1) for s in spaces]
            highs = [jnp.asarray(s.high).reshape(-1) for s in spaces]
            low = jnp.concatenate(lows, axis=0)
            high = jnp.concatenate(highs, axis=0)
            space = Continuous(low=low, high=high)
        elif obs_cls == Discrete:
            ns = [jnp.asarray(s.n).reshape(-1) for s in spaces]
            n = jnp.concatenate(ns, axis=0)
            space = Discrete(n=n)
        else:
            raise ValueError(f"Unsupported space type: {obs_cls}")

        for batch_dim in reversed(batch_dims):
            space = BatchedSpace(space, batch_dim)
        return space
