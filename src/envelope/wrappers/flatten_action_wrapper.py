from functools import cached_property
from math import prod
from typing import override

import jax
import jax.numpy as jnp

from envelope.environment import Info, State
from envelope.spaces import Continuous, Discrete, Space, peel_batched, rebatch
from envelope.typing import PyTree
from envelope.wrappers.wrapper import Wrapper


def flatten_space(space: Space):
    def is_leaf(x):
        # Tuples containing only integers are shape tuples (leaves)
        # PyTreeSpace can only have tuples that contain at least a Space, so
        # tuples with only integers must be shape tuples from leaf spaces
        return isinstance(x, tuple) and all(isinstance(i, int) for i in x)

    shapes, treedef = jax.tree.flatten(space.shape, is_leaf=is_leaf)
    dims = [prod(shape) for shape in shapes]
    return treedef, shapes, dims


def unflatten_x(x: jax.Array, treedef, shapes, dims):
    indices = tuple(sum(dims[:i]) for i in range(1, len(dims)))
    xs = jnp.split(x, indices, axis=-1)
    xs = [p.reshape(p.shape[:-1] + tuple(shape)) for p, shape in zip(xs, shapes)]
    return jax.tree.unflatten(treedef, xs)


class FlattenActionWrapper(Wrapper):
    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        _, base = peel_batched(self.env.action_space)
        treedef, shapes, dims = flatten_space(base)
        action = unflatten_x(action, treedef, shapes, dims)
        return self.env.step(state, action)

    @cached_property
    @override
    def action_space(self) -> Space:
        batch_dims, base = peel_batched(self.env.action_space)

        def is_leaf(x):
            return isinstance(x, (Continuous, Discrete))

        spaces = jax.tree.leaves(base, is_leaf=is_leaf)
        act_cls = type(spaces[0])

        if not all(isinstance(space, act_cls) for space in spaces):
            raise ValueError("All spaces must be of the same type")

        if act_cls == Continuous:
            lows = [jnp.asarray(s.low).reshape(-1) for s in spaces]
            highs = [jnp.asarray(s.high).reshape(-1) for s in spaces]
            low = jnp.concatenate(lows, axis=0)
            high = jnp.concatenate(highs, axis=0)
            space = Continuous(low=low, high=high)
        elif act_cls == Discrete:
            ns = [jnp.asarray(s.n).reshape(-1) for s in spaces]
            n = jnp.concatenate(ns, axis=0)
            space = Discrete(n=n)
        else:
            raise ValueError(f"Unsupported space type: {act_cls}")

        return rebatch(space, batch_dims)
