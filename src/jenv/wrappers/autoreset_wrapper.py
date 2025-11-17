from typing import override

import jax
from jax import lax

from jenv.environment import Info
from jenv.struct import field
from jenv.typing import Key, PyTree
from jenv.wrappers.wrapper import WrappedState, Wrapper


class AutoResetWrapper(Wrapper):
    """
    Automatically resets episodes on termination/truncation and emits fresh observations
    while preserving the terminal step's reward. Maintains a per-instance RNG in
    state.persistent['reset_key'].
    """

    reset_key_name: str = field(kw_only=True, default="reset_key")

    def _seed_persistent(self, state: WrappedState, key: Key) -> WrappedState:
        k_env, k_next = jax.random.split(key)
        state, _ = self.env.reset(k_env)
        persistent = state.persistent.update(**{self.reset_key_name: k_next})
        return state.update(persistent=persistent)

    @override
    def reset(self, key: Key) -> tuple[WrappedState, Info]:
        k_env, k_next = jax.random.split(key)
        state, info = self.env.reset(k_env)
        persistent = state.persistent.update(**{self.reset_key_name: k_next})
        state = state.update(persistent=persistent)
        info = info.update(terminated=False, truncated=False, just_reset=False)
        return state, info

    def _do_reset(self, state: WrappedState) -> tuple[WrappedState, Info]:
        rk = getattr(state.persistent, self.reset_key_name)
        k_env, k_next = jax.random.split(rk)
        reset_state, reset_info = self.env.reset(k_env)
        persistent = state.persistent.update(**{self.reset_key_name: k_next})
        out_state = reset_state.update(persistent=persistent)
        out_info = reset_info.update(terminated=False, truncated=False, just_reset=True)
        return out_state, out_info

    @override
    def step(self, state: WrappedState, action: PyTree) -> tuple[WrappedState, Info]:
        next_state, info = self.env.step(state, action)
        done = info.terminated | info.truncated

        def do_reset_fn(_):
            rs, ri = self._do_reset(next_state)
            # keep terminal reward; expose final_* for boundaries
            ri = ri.update(
                reward=info.reward,
                final_obs=info.obs,
                final_terminated=info.terminated,
                final_truncated=info.truncated,
            )
            return rs, ri

        def keep_fn(_):
            return next_state, info.update(just_reset=False)

        out_state, out_info = lax.cond(done, do_reset_fn, keep_fn, operand=None)
        return out_state, out_info
