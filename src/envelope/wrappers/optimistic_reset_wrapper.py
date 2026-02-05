"""Optimistic reset wrapper for vectorized environments.

Instead of resetting every terminated environment individually, this wrapper
pre-computes a small number of resets (num_envs // reset_ratio) and assigns
them to environments that are done. This is more efficient when only a fraction
of environments terminate each step.

Replaces the combination of VmapWrapper + AutoResetWrapper.
"""

from functools import cached_property
from typing import override

import jax
import jax.numpy as jnp

from envelope import spaces
from envelope.environment import Info
from envelope.struct import field, static_field
from envelope.typing import Key, PyTree
from envelope.wrappers.vmap_wrapper import is_single_key
from envelope.wrappers.wrapper import WrappedState, Wrapper


class OptimisticResetWrapper(Wrapper):
    """Vectorized environment wrapper with optimistic resets.

    Args:
        env: Single (non-batched) environment to vectorize.
        batch_size: Number of parallel environments.
        reset_ratio: Number of environment workers per reset. Higher means
            fewer resets computed (more efficient) but higher chance of
            duplicate reset states. Must divide batch_size evenly.
    """

    batch_size: int = static_field(kw_only=True)
    reset_ratio: int = static_field(kw_only=True, default=16)

    class OptimisticResetState(WrappedState):
        reset_key: jax.Array = field()

    @property
    def num_resets(self) -> int:
        return self.batch_size // self.reset_ratio

    def __post_init__(self):
        if self.batch_size % self.reset_ratio != 0:
            raise ValueError(
                f"reset_ratio ({self.reset_ratio}) must evenly divide "
                f"batch_size ({self.batch_size})."
            )

    def _split_keys(self, key: Key) -> Key:
        if is_single_key(key):
            return jax.random.split(key, self.batch_size)
        return key

    @override
    def init(self, key: Key) -> tuple[WrappedState, Info]:
        key, subkey = jax.random.split(key)
        keys = self._split_keys(key)
        inner_state, info = jax.vmap(self.env.init)(keys)
        state = self.OptimisticResetState(inner_state=inner_state, reset_key=subkey)
        return state, info.update(obs_true=info.obs)

    @override
    def reset(self, key: Key, state: WrappedState) -> tuple[WrappedState, Info]:
        key, subkey = jax.random.split(key)
        keys = self._split_keys(key)
        inner_state, info = jax.vmap(self.env.reset)(keys, state.inner_state)
        state = self.OptimisticResetState(inner_state=inner_state, reset_key=subkey)
        return state, info.update(obs_true=info.obs)

    @override
    def step(self, state: WrappedState, action: PyTree) -> tuple[WrappedState, Info]:
        # Step all environments
        inner_state, info_step = jax.vmap(self.env.step)(state.inner_state, action)
        done = jnp.asarray(info_step.terminated | info_step.truncated)
        obs_true = info_step.obs

        # Compute num_resets fresh reset states
        key_for_resets = jax.random.fold_in(state.reset_key, 0)
        next_reset_key = jax.random.fold_in(state.reset_key, 1)
        reset_keys = jax.random.split(key_for_resets, self.num_resets)
        reset_inner_state, reset_info = jax.vmap(self.env.init)(reset_keys)

        # Assign resets to done environments
        # Default: each reset is repeated reset_ratio times
        reset_indexes = jnp.arange(self.num_resets).repeat(self.reset_ratio)

        # Probabilistically assign actual resets to done environments
        rng_choice = jax.random.fold_in(state.reset_key, 2)
        # Add small epsilon to avoid all-zero probabilities when no env is done
        done_probs = done.astype(jnp.float32) + 1e-8
        done_probs = done_probs / done_probs.sum()
        being_reset = jax.random.choice(
            rng_choice,
            jnp.arange(self.batch_size),
            shape=(self.num_resets,),
            p=done_probs,
            replace=False,
        )
        reset_indexes = reset_indexes.at[being_reset].set(jnp.arange(self.num_resets))

        # Expand reset states to match batch_size via indexing
        mapped_reset_state = jax.tree.map(lambda x: x[reset_indexes], reset_inner_state)

        # Select between stepped state and reset state based on done
        def select(reset_val, step_val):
            if not hasattr(reset_val, "ndim"):
                return jnp.where(done, reset_val, step_val)
            if reset_val.ndim > 0 and reset_val.shape[0] == self.batch_size:
                cond = done
                for _ in range(reset_val.ndim - done.ndim):
                    cond = jnp.expand_dims(cond, axis=-1)
                return jnp.where(cond, reset_val, step_val)
            return step_val

        final_inner_state = jax.tree.map(select, mapped_reset_state, inner_state)
        final_state = self.OptimisticResetState(
            inner_state=final_inner_state, reset_key=next_reset_key
        )

        # Build info: for done envs, use reset obs but preserve all step info
        # (episode stats, reward, flags come from the terminal step)
        mapped_reset_obs = jax.tree.map(lambda x: x[reset_indexes], reset_info.obs)
        done_info = info_step.update(
            obs=mapped_reset_obs,
            obs_true=obs_true,
        )
        continue_info = info_step.update(obs_true=obs_true)
        final_info = jax.tree.map(select, done_info, continue_info)

        return final_state, final_info

    @override
    @cached_property
    def observation_space(self) -> spaces.Space:
        return spaces.batch_space(self.env.observation_space, self.batch_size)

    @override
    @cached_property
    def action_space(self) -> spaces.Space:
        return spaces.batch_space(self.env.action_space, self.batch_size)
