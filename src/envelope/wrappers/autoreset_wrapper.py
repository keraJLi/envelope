from typing import override

import jax
import jax.numpy as jnp

from envelope.environment import Info
from envelope.struct import field
from envelope.typing import Key, PyTree
from envelope.wrappers.wrapper import WrappedState, Wrapper


class AutoResetWrapper(Wrapper):
    """Wrapper that automatically resets the environment when an episode ends.

    When a step results in termination or truncation, this wrapper immediately
    resets the environment. The returned info preserves critical information
    from the terminal step while providing the new episode's initial observation.

    Info fields after a terminal step (terminated=True or truncated=True):
        obs: Initial observation from the new episode (after reset).
        final: Full info snapshot from the terminal step (before reset).
        terminated: True if the episode ended due to termination.
        truncated: True if the episode ended due to truncation.
        reward: Reward from the terminal step.

    Info fields during normal steps (terminated=False and truncated=False):
        obs: Current observation.
        final: Info snapshot from the last completed episode (persisted).
        terminated: False.
        truncated: False.
        reward: Reward from the step.

    This design enables correct value bootstrapping:
        - Use final.obs for value estimation of the true next state
        - On termination: V(s_final) = 0 (episode truly ended)
        - On truncation: bootstrap from V(final.obs) (episode cut off artificially)
        - final persists until the next episode completes, giving easy access
          to last episode's aggregated stats (e.g., final.episode_return)
    """

    class AutoResetState(WrappedState):
        reset_key: jax.Array = field()
        last_final: Info = field()

    @override
    def init(self, key: Key) -> tuple[WrappedState, Info]:
        key, subkey = jax.random.split(key)
        inner_state, info = self.env.init(key)
        # Initialize last_final with the reset info (no previous episode yet)
        last_final = jax.tree.map(lambda x: jnp.full_like(x, jnp.nan), info)
        state = self.AutoResetState(
            inner_state=inner_state, reset_key=subkey, last_final=last_final
        )
        return state, info.update(final=state.last_final)

    @override
    def reset(self, key: Key, state: WrappedState) -> tuple[WrappedState, Info]:
        raise NotImplementedError("Reset is not implemented for AutoResetWrapper")

    @override
    def step(self, state: WrappedState, action: PyTree) -> tuple[WrappedState, Info]:
        key, key_reset = jax.random.split(state.reset_key)
        state = state.replace(reset_key=key)

        inner_state, info = self.env.step(state.inner_state, action)
        reset_inner_state, reset_info = self.env.reset(key_reset, inner_state)

        # Select next state and info based on done
        done = info.terminated | info.truncated
        state = jax.tree.map(
            lambda reset, next: jax.lax.select(done, reset, next),
            state.replace(inner_state=reset_inner_state),
            state.replace(inner_state=inner_state),
        )
        info = jax.tree.map(
            lambda reset, next: jax.lax.select(done, reset, next),
            reset_info.update(final=info),
            info.update(final=state.last_final),
        )
        return state, info
