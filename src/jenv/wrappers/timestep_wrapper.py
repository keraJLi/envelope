from jenv.wrappers.wrapper import Wrapper


class TimeStepWrapper(Wrapper):
    def reset(self, key):
        state, info = self.env.reset(key)
        return state.update(steps=0), info.update

    def step(self, state, action):
        next_state, info = self.env.step(state, action)
        return next_state.update(steps=state.steps + 1), info
