from typing import ClassVar, override

import jax
import jax.numpy as jnp

from envelope.environment import Info
from envelope.struct import field
from envelope.typing import Key, PyTree
from envelope.wrappers.wrapper import WrappedState, Wrapper, WrapperStackRule


def _validate_leaf_metadata(candidate: PyTree, reference: PyTree, name: str) -> None:
    candidate_leaves = jax.tree.leaves(candidate)
    reference_leaves = jax.tree.leaves(reference)
    for index, (candidate_leaf, reference_leaf) in enumerate(
        zip(candidate_leaves, reference_leaves)
    ):
        candidate_array = jnp.asarray(candidate_leaf)
        reference_array = jnp.asarray(reference_leaf)
        if candidate_array.shape != reference_array.shape:
            raise ValueError(
                f"{name} leaf {index} shape mismatch: expected "
                f"{reference_array.shape}, got {candidate_array.shape}"
            )
        if candidate_array.dtype != reference_array.dtype:
            raise ValueError(
                f"{name} leaf {index} dtype mismatch: expected "
                f"{reference_array.dtype}, got {candidate_array.dtype}"
            )


class StateInjectionWrapper(Wrapper):
    """Stores a state that all resets return to.

    For UED: use set_reset_state() to update the injected state, then all resets
    (including auto-reset) return to that state until it's changed again.

    Usage:
        ```python
        env = AutoResetWrapper(StateInjectionWrapper(env=base_env))
        state, info = env.init(key)

        for outer_iter in range(num_outer_iters):
            # Sample a new task and set it as the reset state
            task_state, task_info = sample_task(key)
            state = env.set_reset_state(
                state, task_state, reset_info=task_info
            )

            # Run episode - auto-resets return to task_state
            for inner_step in range(num_inner_steps):
                state, info = env.step(state, policy(info.obs))
        ```
    """

    wrapper_roles: ClassVar[frozenset[str]] = frozenset(
        {"persistent", "state_injection"}
    )
    stack_rules: ClassVar[tuple[WrapperStackRule, ...]] = (
        WrapperStackRule("lifecycle", "{outer} must be inside {inner}"),
    )

    class InjectedState(WrappedState):
        reset_state: PyTree = field()
        reset_info: Info = field()
        active: jax.Array = field()

    @property
    @override
    def supports_init_pooling(self) -> bool:
        return False

    def set_reset_state(
        self,
        state: WrappedState,
        reset_state: PyTree,
        *,
        reset_info: Info,
    ) -> WrappedState:
        """Update the state that resets will return to.

        This method traverses the wrapped state hierarchy to find and update
        the InjectedState, then reconstructs the full state tree.

        Args:
            state: Current state (can be from any outer wrapper)
            reset_state: The state to reset to (inner environment state)
            reset_info: The complete info value to return on reset. Its PyTree
                structure must match the environment's normal reset info.

        Returns:
            New state with updated reset fields at the appropriate level
        """

        def update_injected(s: WrappedState) -> WrappedState:
            # If this is our InjectedState, update it
            if isinstance(s, self.InjectedState):
                if jax.tree.structure(reset_state) != jax.tree.structure(s.reset_state):
                    raise ValueError(
                        "reset_state must have the same PyTree structure as the "
                        "environment state"
                    )
                if jax.tree.structure(reset_info) != jax.tree.structure(s.reset_info):
                    raise ValueError(
                        "reset_info must have the same PyTree structure as the "
                        "environment's reset info"
                    )
                _validate_leaf_metadata(reset_state, s.reset_state, "reset_state")
                _validate_leaf_metadata(reset_info, s.reset_info, "reset_info")
                return s.replace(
                    inner_state=reset_state,
                    reset_state=reset_state,
                    reset_info=reset_info,
                    active=jnp.asarray(True),
                )
            # Otherwise, recurse into inner_state and rebuild
            if hasattr(s, "inner_state"):
                return s.replace(inner_state=update_injected(s.inner_state))
            raise ValueError("Could not find InjectedState in given state")

        return update_injected(state)

    @override
    def init(self, key: Key) -> tuple[InjectedState, Info]:
        inner_state, info = self.env.init(key)
        state = self.InjectedState(
            inner_state=inner_state,
            reset_state=inner_state,
            reset_info=info,
            active=jnp.asarray(False),
        )
        return state, info

    @override
    def reset(self, state: InjectedState, key: Key) -> tuple[InjectedState, Info]:
        def injected(_):
            return state.reset_state, state.reset_info

        def delegated(_):
            return self.env.reset(state.inner_state, key)

        inner_state, info = jax.lax.cond(
            state.active, injected, delegated, operand=None
        )
        return state.replace(inner_state=inner_state), info

    @override
    def step(self, state: InjectedState, action: PyTree) -> tuple[InjectedState, Info]:
        inner_state, info = self.env.step(state.inner_state, action)
        return state.replace(inner_state=inner_state), info
