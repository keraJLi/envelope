import jax

from envelope.environment import Info
from envelope.struct import field
from envelope.typing import Key, PyTree
from envelope.wrappers.wrapper import WrappedState, Wrapper


class AutoResetWrapper(Wrapper):
    class AutoResetState(WrappedState):
        reset_key: jax.Array = field()

    def reset(
        self, key: Key, state: PyTree | None = None, **kwargs
    ) -> tuple[WrappedState, Info]:
        key, subkey = jax.random.split(key)
        inner_state = state.inner_state if state else None
        inner_state, info = self.env.reset(key, inner_state, **kwargs)
        state = self.AutoResetState(inner_state=inner_state, reset_key=subkey)
        return state, info.update(next_obs=info.obs)

    def step(
        self, state: WrappedState, action: PyTree, **kwargs
    ) -> tuple[WrappedState, Info]:
        inner_state, info_step = self.env.step(state.inner_state, action, **kwargs)
        done = info_step.terminated | info_step.truncated

        state = self.AutoResetState(inner_state=inner_state, reset_key=state.reset_key)
        info = info_step.update(next_obs=info_step.obs)

        state, info = jax.lax.cond(
            done,
            lambda: self.reset(state.reset_key, state),
            lambda: (state, info),
        )
        return state, info
