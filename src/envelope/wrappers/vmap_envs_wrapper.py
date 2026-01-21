from functools import cached_property
from typing import override

import jax

from envelope import spaces
from envelope.environment import Environment, Info
from envelope.struct import field
from envelope.typing import Key, PyTree
from envelope.wrappers.wrapper import WrappedState, Wrapper


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

    @override
    def reset(
        self, key: Key, state: PyTree | None = None, **kwargs
    ) -> tuple[WrappedState, Info]:
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
        state, info = jax.vmap(lambda e, k: e.reset(k, state, **kwargs))(self.env, keys)
        return state, info

    @override
    def step(
        self, state: WrappedState, action: PyTree, **kwargs
    ) -> tuple[WrappedState, Info]:
        next_state, info = jax.vmap(lambda e, s, a: e.step(s, a, **kwargs))(
            self.env, state, action
        )
        return next_state, info

    @override
    @property
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
