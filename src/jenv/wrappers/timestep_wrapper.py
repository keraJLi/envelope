from jenv.environment import Info
from jenv.typing import Key, PyTree
from jenv.wrappers.wrapper import WrappedState, Wrapper


class TimeStepWrapper(Wrapper):
    def reset(self, key: Key) -> tuple[WrappedState, Info]:
        state, info = self.env.reset(key)
        episodic = state.episodic.update(steps=0)
        return state.update(episodic=episodic), info

    def step(self, state: WrappedState, action: PyTree) -> tuple[WrappedState, Info]:
        next_state, info = self.env.step(state, action)
        episodic = next_state.episodic.update(steps=state.episodic.steps + 1)
        return next_state.update(episodic=episodic), info
