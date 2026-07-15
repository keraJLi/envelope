from typing import ClassVar, override

import jax
import jax.numpy as jnp

from envelope.environment import Info
from envelope.struct import field
from envelope.typing import Key, PyTree
from envelope.wrappers.wrapper import WrappedState, Wrapper, WrapperStackRule


class AutoResetWrapper(Wrapper):
    """Wrapper that automatically resets the environment when an episode ends.

    When a step results in termination or truncation, this wrapper immediately resets
    the environment. The returned info preserves critical information from the terminal
    step while providing the new episode's initial observation.

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
        final_valid: jax.Array = field()

    wrapper_roles: ClassVar[frozenset[str]] = frozenset({"autoreset", "lifecycle"})
    stack_rules: ClassVar[tuple[WrapperStackRule, ...]] = (
        WrapperStackRule(
            "vectorization",
            (
                "AutoResetWrapper must be inside VmapWrapper/vectorization, "
                "not outside it"
            ),
        ),
    )

    @property
    @override
    def supports_init_pooling(self) -> bool:
        return False

    @override
    def init(self, key: Key) -> tuple[AutoResetState, Info]:
        inner_key, reset_key = jax.random.split(key)
        inner_state, info = self.env.init(inner_key)
        last_final = jax.tree.map(jnp.zeros_like, info)
        final_valid = jnp.asarray(False)
        state = self.AutoResetState(
            inner_state=inner_state,
            reset_key=reset_key,
            last_final=last_final,
            final_valid=final_valid,
        )
        return state, info.update(final=state.last_final, final_valid=final_valid)

    @override
    def reset(self, state: AutoResetState, key: Key) -> tuple[AutoResetState, Info]:
        inner_key, reset_key = jax.random.split(key)
        inner_state, info = self.env.reset(state.inner_state, inner_key)
        state = state.replace(inner_state=inner_state, reset_key=reset_key)
        return state, info.update(final=state.last_final, final_valid=state.final_valid)

    @override
    def step(
        self, state: AutoResetState, action: PyTree
    ) -> tuple[AutoResetState, Info]:
        next_reset_key, reset_key = jax.random.split(state.reset_key)
        transition_state, transition_info = self.env.step(state.inner_state, action)

        done = jnp.asarray(transition_info.terminated) | jnp.asarray(
            transition_info.truncated
        )

        def reset_episode(_):
            reset_state, reset_info = self.env.reset(transition_state, reset_key)
            terminal_info = reset_info.update(
                reward=transition_info.reward,
                terminated=transition_info.terminated,
                truncated=transition_info.truncated,
                final=transition_info,
                final_valid=jnp.asarray(True),
            )
            return reset_state, transition_info, jnp.asarray(True), terminal_info

        def continue_episode(_):
            continuing_info = transition_info.update(
                final=state.last_final,
                final_valid=state.final_valid,
            )
            return (
                transition_state,
                state.last_final,
                state.final_valid,
                continuing_info,
            )

        inner_state, last_final, final_valid, info = jax.lax.cond(
            done, reset_episode, continue_episode, operand=None
        )

        state = state.replace(
            inner_state=inner_state,
            reset_key=next_reset_key,
            last_final=last_final,
            final_valid=final_valid,
        )
        return state, info
