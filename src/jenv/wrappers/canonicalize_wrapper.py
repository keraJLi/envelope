from typing import override

from jenv.environment import Info
from jenv.typing import Key, PyTree
from jenv.wrappers.wrapper import WrappedState, Wrapper


class CanonicalizeWrapper(Wrapper):
    """Wraps the underlying env state into `inner_state` on reset/step.
    Apply this first in a wrapper stack.
    """

    @override
    def reset(
        self, key: Key, state: PyTree | None = None, **kwargs
    ) -> tuple[WrappedState, Info]:
        state, info = self.env.reset(key, state, **kwargs)
        state = WrappedState(inner_state=state)
        return state, info

    @override
    def step(
        self, state: WrappedState, action: PyTree, **kwargs
    ) -> tuple[WrappedState, Info]:
        inner_state, info = self.env.step(state.inner_state, action, **kwargs)
        state = state.replace(inner_state=inner_state)
        return state, info
