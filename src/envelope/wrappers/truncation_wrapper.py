from numbers import Integral
from typing import override

import jax.numpy as jnp

from envelope.environment import Info
from envelope.struct import field
from envelope.typing import Key, PyTree
from envelope.wrappers.wrapper import WrappedState, Wrapper, _find_wrapper


class TruncationWrapper(Wrapper):
    max_steps: int = field(kw_only=True)

    class TruncationState(WrappedState):
        steps: jnp.ndarray | int = field(default=0)

    def __post_init__(self):
        from envelope.wrappers.autoreset_wrapper import AutoResetWrapper
        from envelope.wrappers.pooled_init_vmap_wrapper import PooledInitVmapWrapper

        lifecycle_wrapper = _find_wrapper(
            self.env, (AutoResetWrapper, PooledInitVmapWrapper)
        )
        if lifecycle_wrapper is not None:
            raise ValueError(
                f"TruncationWrapper must be inside {type(lifecycle_wrapper).__name__}"
            )
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, Integral):
            raise TypeError("max_steps must be an integer")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")

    @override
    def init(self, key: Key) -> tuple[TruncationState, Info]:
        inner_state, info = self.env.init(key)
        state = self.TruncationState(inner_state=inner_state, steps=0)
        return state, info

    @override
    def reset(self, state: TruncationState, key: Key) -> tuple[TruncationState, Info]:
        inner_state, info = self.env.reset(state.inner_state, key)
        state = state.replace(inner_state=inner_state, steps=0)
        return state, info

    @override
    def step(
        self, state: TruncationState, action: PyTree
    ) -> tuple[TruncationState, Info]:
        next_inner_state, info = self.env.step(state.inner_state, action)
        steps = state.steps + 1
        state = self.TruncationState(inner_state=next_inner_state, steps=steps)
        truncated = jnp.asarray(info.truncated) | (jnp.asarray(steps) >= self.max_steps)
        return state, info.update(truncated=truncated)
