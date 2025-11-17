from functools import cached_property
from typing import Callable, override

import jax

from jenv import spaces
from jenv.environment import Environment, Info
from jenv.struct import field
from jenv.typing import Key, PyTree
from jenv.wrappers.wrapper import WrappedState, Wrapper


class VmapEnvsWrapper(Wrapper):
    """
    Vectorizes over a batched collection of environment instances (vmapping over 'self').

    Usage:
        envs = jax.vmap(make_env)(params_batch)     # env pytree batched on leading axis
        wrapped = VmapEnvsWrapper(env=envs, batch_size=B)
        state, info = wrapped.reset(keys)           # keys shape (B, 2) or single key
        next_state, info = wrapped.step(state, action)
    """

    batch_size: int = field(kw_only=True)
    in_axes: Callable[[WrappedState], PyTree] = field(kw_only=True, default=None)
    """Callable that takes a wrapped state and returns an in_axes pytree. Defaults to
    vmapping along axis 0 of core and episodic, and leaves persistent static."""

    def __post_init__(self):
        if self.in_axes is None:
            object.__setattr__(self, "in_axes", _vmap_axes_from_state)

    @override
    def reset(self, key: Key) -> tuple[WrappedState, Info]:
        if key.shape == (2,):
            keys = jax.random.split(key, self.batch_size)
        else:
            if key.shape[0] != self.batch_size:
                raise ValueError(
                    f"reset key's leading dimension ({key.shape[0]}) must match "
                    f"batch_size ({self.batch_size})."
                )
            keys = key
        # vmap over env 'self' and keys
        state, info = jax.vmap(lambda e, k: e.reset(k))(self.env, keys)
        return state, info

    @override
    def step(self, state: WrappedState, action: PyTree) -> tuple[WrappedState, Info]:
        axes = self.in_axes(state)
        next_state, info = jax.vmap(lambda e, s, a: e.step(s, a), in_axes=(0, axes, 0))(
            self.env, state, action
        )
        return next_state, info

    @override
    @cached_property
    def observation_space(self) -> spaces.Space:
        env0 = _index_env(self.env, 0, self.batch_size)
        return spaces.batch_space(env0.observation_space, self.batch_size)

    @override
    @cached_property
    def action_space(self) -> spaces.Space:
        env0 = _index_env(self.env, 0, self.batch_size)
        return spaces.batch_space(env0.action_space, self.batch_size)

    @override
    @property
    def unwrapped(self) -> Environment:
        return self.env.unwrapped


def _index_env(env: Environment, idx: int, batch_size: int) -> Environment:
    def idx_or_keep(x):
        if hasattr(x, "shape") and isinstance(getattr(x, "shape"), tuple):
            if len(x.shape) > 0 and x.shape[0] == batch_size:
                return x[idx]
        return x

    return jax.tree.map(lambda x: idx_or_keep(x), env)


def _vmap_axes_from_state(state: WrappedState) -> PyTree:
    """
    Build an in_axes pytree that maps over core/episodic/persistent (axis 0)
    matching the WrappedState structure.
    """
    axes = jax.tree.map(lambda _: None, state)
    axes_core = jax.tree.map(lambda _: 0, state.core)
    axes_ep = jax.tree.map(lambda _: 0, state.episodic)
    axes_pers = jax.tree.map(lambda _: None, state.persistent)
    return axes.update(core=axes_core, episodic=axes_ep, persistent=axes_pers)
