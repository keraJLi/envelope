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

    def init(self, key: Key) -> tuple[WrappedState, Info]:
        key, subkey = jax.random.split(key)
<<<<<<< HEAD

        if state is None:
            inner_state, info = self.env.reset(key, **kwargs)
            state = self.AutoResetState(inner_state=inner_state, reset_key=subkey)
        else:
            inner_state, info = self.env.reset(key, state.inner_state, **kwargs)
            state = state.replace(inner_state=inner_state, reset_key=subkey)

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
=======
        inner_state, info = self.env.init(key)
        # Initialize last_final with the reset info (no previous episode yet)
        state = self.AutoResetState(
            inner_state=inner_state, reset_key=subkey, last_final=info
>>>>>>> autoreset-final-info
        )
        return state, info.update(final=state.last_final)

    def reset(self, key: Key, state: WrappedState) -> tuple[WrappedState, Info]:
        key, subkey = jax.random.split(key)
        inner_state, info = self.env.reset(key, state.inner_state)
        # Preserve last_final from previous state (keep last episode's info)
        state = state.replace(inner_state=inner_state, reset_key=subkey)
        return state, info.update(final=state.last_final)

    def step(self, state: WrappedState, action: PyTree) -> tuple[WrappedState, Info]:
        inner_state, info_step = self.env.step(state.inner_state, action)
        done = jnp.asarray(info_step.terminated | info_step.truncated)

        # Derive keys deterministically using fold_in (works with any key shape)
        key_for_reset = jax.random.fold_in(state.reset_key, 0)
        next_reset_key = jax.random.fold_in(state.reset_key, 1)

        # Always compute reset (both branches evaluated for jnp.where)
        reset_inner_state, reset_info = self.env.reset(key_for_reset, inner_state)

        # Build info for continue case: persist last episode's final info
        continue_info = info_step.update(final=state.last_final)

        # Build info for done case: reset obs but preserve terminal step's flags/reward
        # info.final captures the FULL terminal info snapshot (before reset)
        done_info = info_step.update(
            obs=reset_info.obs,  # New episode's initial observation
            final=info_step,  # Full terminal info snapshot
        )

        # Select between done and continue branches using jnp.where
        batch_size = done.shape[0] if done.ndim > 0 else None

        def select(done_val, continue_val):
            # Handle non-array leaves (Python scalars)
            if not hasattr(done_val, "ndim"):
                # Use jnp.where with array conversion (keeps JAX array type)
                return jnp.where(done, done_val, continue_val)

            if batch_size is None:
                # Scalar done (unbatched env): simple where
                return jnp.where(done, done_val, continue_val)

            # Batched done: only select if array is batched (first dim matches)
            if done_val.ndim > 0 and done_val.shape[0] == batch_size:
                # Expand done to broadcast correctly with higher-rank arrays
                cond = done
                for _ in range(done_val.ndim - done.ndim):
                    cond = jnp.expand_dims(cond, axis=-1)
                return jnp.where(cond, done_val, continue_val)
            else:
                # Non-batched array (shared state): keep the continue value
                # These are typically wrapper-level state like running stats
                return continue_val

        # Select inner_state per-env
        final_inner_state = jax.tree.map(select, reset_inner_state, inner_state)

        # Select last_final: on done, store terminal info; on continue, keep previous
        final_last_final = jax.tree.map(select, info_step, state.last_final)

        final_state = self.AutoResetState(
            inner_state=final_inner_state,
            reset_key=next_reset_key,
            last_final=final_last_final,
        )
        final_info = jax.tree.map(select, done_info, continue_info)

        return final_state, final_info
