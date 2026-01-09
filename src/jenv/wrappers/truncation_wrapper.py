import jax.numpy as jnp

from jenv.environment import Info
from jenv.struct import field
from jenv.typing import Key, PyTree
from jenv.wrappers.wrapper import WrappedState, Wrapper


class TruncationWrapper(Wrapper):
    max_steps: int = field(kw_only=True)

    class TruncationState(WrappedState):
        steps: jnp.ndarray | int = field(default=0)

    def reset(
        self, key: Key, state: PyTree | None = None, **kwargs
    ) -> tuple[WrappedState, Info]:
        inner_state = state.inner_state if state else None

        inner_state, info = self.env.reset(key, inner_state, **kwargs)
        state = self.TruncationState(inner_state=inner_state, steps=0)
        return state, info.update(truncated=self.max_steps <= 0)

    def step(
        self, state: WrappedState, action: PyTree, **kwargs
    ) -> tuple[WrappedState, Info]:
        next_inner_state, info = self.env.step(state.inner_state, action, **kwargs)
        next_steps = state.steps + 1
        next_state = self.TruncationState(
            inner_state=next_inner_state, steps=next_steps
        )
        truncated = jnp.asarray(next_steps) >= self.max_steps
        return next_state, info.update(truncated=truncated)
