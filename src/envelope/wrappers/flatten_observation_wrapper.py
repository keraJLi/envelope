from functools import cached_property

import jax
import jax.numpy as jnp
from typing_extensions import override

from envelope.environment import Info, State
from envelope.spaces import BatchedSpace, Continuous, Discrete, Space, peel_batched
from envelope.typing import Key, PyTree
from envelope.wrappers.wrapper import Wrapper


def flatten_space(space: Space):
    def is_leaf(x):
        # Tuples containing only integers are shape tuples (leaves)
        # PyTreeSpace can only have tuples that contain at least a Space, so
        # tuples with only integers must be shape tuples from leaf spaces
        return isinstance(x, tuple) and all(isinstance(i, int) for i in x)

    shapes, treedef = jax.tree.flatten(space.shape, is_leaf=is_leaf)
    dims = [jnp.prod(jnp.asarray(shape)) for shape in shapes]
    return treedef, shapes, dims


def flatten_x(x: PyTree):
    leaves = jax.tree.leaves(x)
    xs = jax.tree.map(lambda x: jnp.asarray(x).reshape(-1), leaves)
    x = jnp.concatenate(xs, axis=0)
    return x


class FlattenObservationWrapper(Wrapper):
    @override
    def init(self, key: Key) -> tuple[State, Info]:
        state, info = self.env.init(key)
        info = info.update(obs=flatten_x(info.obs))
        return state, info

    @override
    def reset(self, state: State, key: Key) -> tuple[State, Info]:
        next_state, info = self.env.reset(state, key)
        return next_state, info.update(obs=flatten_x(info.obs))

    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        state, info = self.env.step(state, action)
        info = info.update(obs=flatten_x(info.obs))
        return state, info

    @override
    @cached_property
    def observation_space(self) -> Space:
        batch_dims, base = peel_batched(self.env.observation_space)

        def is_leaf(x):
            spaces = (Continuous, Discrete)
            return isinstance(x, spaces)

        spaces = jax.tree.leaves(base, is_leaf=is_leaf)
        obs_cls = type(spaces[0])

        if not all(isinstance(space, obs_cls) for space in spaces):
            raise ValueError("All spaces must be of the same type")

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

        for batch_dim in batch_dims:
            space = BatchedSpace(space, batch_dim)
        return space
