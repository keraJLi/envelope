from jenv.environment import Info, State
from jenv.typing import Key, PyTree
from jenv.wrappers.wrapper import Wrapper


class TruncationWrapper(Wrapper):
    max_steps: int

    def _get_steps(self, state: State) -> int:
        if hasattr(state, "steps"):
            return state.steps
        raise ValueError(
            "TruncationWrapper requires a 'steps' attribute on the state "
            "(e.g. via a TimeStepWrapper)."
        )

    def reset(self, key: Key) -> tuple[State, Info]:
        state, info = self.env.reset(key)
        return state, info.update(truncated=self.max_steps <= 0)

    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        next_state, info = self.env.step(state, action)
        truncated = self._get_steps(next_state) >= self.max_steps
        return next_state, info.update(truncated=truncated)
