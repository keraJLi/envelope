import jax.numpy as jnp

from jenv.environment import Info
from jenv.struct import field
from jenv.typing import Key, PyTree
from jenv.wrappers.wrapper import WrappedState, Wrapper


class TruncationWrapper(Wrapper):
    max_steps: int = field(kw_only=True)

    def _get_steps(self, state: WrappedState):
        try:
            return jnp.asarray(state.episodic.steps)
        except AttributeError:
            raise ValueError(
                "TruncationWrapper requires a 'steps' attribute on `state.episodic` "
                "(e.g. via a TimeStepWrapper)."
            )

    def reset(self, key: Key) -> tuple[WrappedState, Info]:
        state, info = self.env.reset(key)
        return state, info.update(truncated=self.max_steps <= 0)

    def step(self, state: WrappedState, action: PyTree) -> tuple[WrappedState, Info]:
        next_state, info = self.env.step(state, action)
        truncated = self._get_steps(next_state) >= self.max_steps
        return next_state, info.update(truncated=truncated)
