from functools import cached_property
from typing import override

import jax

from jenv import spaces
from jenv.environment import Info
from jenv.struct import field
from jenv.typing import Key, PyTree
from jenv.wrappers.wrapper import WrappedState, Wrapper


class VmapWrapper(Wrapper):
    batch_size: int = field(kw_only=True)

    @override
    def reset(self, key: Key) -> tuple[WrappedState, Info]:
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

        state, info = jax.vmap(self.env.reset)(keys)
        return state, info

    @override
    def step(self, state: WrappedState, action: PyTree) -> tuple[WrappedState, Info]:
        vmap_axes = _vmap_axes_from_state(state)
        state, info = jax.vmap(self.env.step, in_axes=(vmap_axes, 0))(state, action)
        return state, info

    @override
    @cached_property
    def observation_space(self) -> spaces.Space:
        return spaces.batch_space(self.env.observation_space, self.batch_size)

    @override
    @cached_property
    def action_space(self) -> spaces.Space:
        return spaces.batch_space(self.env.action_space, self.batch_size)


def _vmap_axes_from_state(state: WrappedState) -> PyTree:
    axes = jax.tree.map(lambda _: None, state)
    axes_core = jax.tree.map(lambda _: 0, state.core)
    axes_ep = jax.tree.map(lambda _: 0, state.episodic)
    axes_pers = jax.tree.map(lambda _: None, state.persistent)
    return axes.update(core=axes_core, episodic=axes_ep, persistent=axes_pers)
