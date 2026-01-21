from envelope.environment import Info, InfoContainer
from envelope.struct import field
from envelope.typing import Key, PyTree
from envelope.wrappers.wrapper import WrappedState, Wrapper


class StateInjectionWrapper(Wrapper):
    """Stores a state that all resets return to.

    For UED: use set_reset_state() to update the injected state, then all resets
    (including auto-reset) return to that state until it's changed again.

    Usage:
        env = AutoResetWrapper(StateInjectionWrapper(env=base_env))
        state, info = env.reset(key)

        for outer_iter in range(num_outer_iters):
            # Sample a new task and set it as the reset state
            task_state, task_obs = sample_task(key)
            state = env.set_reset_state(state, task_state, task_obs)

            # Run episode - auto-resets return to task_state
            for inner_step in range(num_inner_steps):
                state, info = env.step(state, policy(info.obs))
    """

    class InjectedState(WrappedState):
        reset_state: PyTree | None = field(default=None)
        reset_obs: PyTree | None = field(default=None)

    def set_reset_state(
        self, state: WrappedState, reset_state: PyTree, reset_obs: PyTree
    ) -> WrappedState:
        """Update the state that resets will return to.

        This method traverses the wrapped state hierarchy to find and update
        the InjectedState, then reconstructs the full state tree.

        Args:
            state: Current state (can be from any outer wrapper)
            reset_state: The state to reset to (inner environment state)
            reset_obs: The observation to return on reset

        Returns:
            New state with updated reset fields at the appropriate level
        """

        def update_injected(s: WrappedState) -> WrappedState:
            # If this is our InjectedState, update it
            if isinstance(s, self.InjectedState):
                return self.InjectedState(
                    inner_state=reset_state,
                    reset_state=reset_state,
                    reset_obs=reset_obs,
                )
            # Otherwise, recurse into inner_state and rebuild
            if hasattr(s, "inner_state"):
                return s.replace(inner_state=update_injected(s.inner_state))
            raise ValueError("Could not find InjectedState in given state")

        return update_injected(state)

    def reset(
        self, key: Key, state: PyTree | None = None, **kwargs
    ) -> tuple[WrappedState, Info]:
        # Default state has no inner state to reset to
        if state is None:
            state = self.InjectedState(inner_state=None)

        # If no reset state is set, reset wrapped environment
        if state.reset_state is None and state.reset_obs is None:
            inner_state, info = self.env.reset(key, state=state.inner_state, **kwargs)

        # If reset state is set, use it
        elif state.reset_state is not None and state.reset_obs is not None:
            inner_state = state.reset_state
            info = InfoContainer(obs=state.reset_obs, reward=0.0, terminated=False)

        # If only one of reset_state or reset_obs is set, raise error
        else:
            raise ValueError("State must set both reset_state and reset_obs or neither")

        # Return new state with updated inner state
        state = state.replace(inner_state=inner_state)
        return state, info

    def step(
        self, state: WrappedState, action: PyTree, **kwargs
    ) -> tuple[WrappedState, Info]:
        inner_state, info = self.env.step(state.inner_state, action, **kwargs)
        return state.replace(inner_state=inner_state), info
