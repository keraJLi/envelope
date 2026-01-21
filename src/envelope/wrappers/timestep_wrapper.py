from envelope.environment import Info
from envelope.struct import field
from envelope.typing import Key, PyTree
from envelope.wrappers.wrapper import WrappedState, Wrapper


class TimeStepWrapper(Wrapper):
    class TimeStepState(WrappedState):
        steps: PyTree = field(default=0)

    def reset(
        self, key: Key, state: PyTree | None = None, **kwargs
    ) -> tuple[WrappedState, Info]:
        inner_state, info = self.env.reset(key, state, **kwargs)
        return self.TimeStepState(inner_state=inner_state, steps=0), info

    def step(
        self, state: WrappedState, action: PyTree, **kwargs
    ) -> tuple[WrappedState, Info]:
        next_inner_state, info = self.env.step(state.inner_state, action, **kwargs)
        next_steps = getattr(state, "steps", 0) + 1
        return self.TimeStepState(inner_state=next_inner_state, steps=next_steps), info
