from functools import cached_property
from numbers import Integral
from typing import ClassVar, override

import jax
import jax.numpy as jnp

from envelope import spaces
from envelope.environment import Info
from envelope.struct import field, static_field
from envelope.typing import Key, PyTree
from envelope.wrappers.vmap_wrapper import _split_or_keep_key
from envelope.wrappers.wrapper import WrappedState, Wrapper, WrapperStackRule


class PooledInitVmapWrapper(Wrapper):
    batch_size: int = static_field(kw_only=True)
    pool_size: int = static_field(kw_only=True)

    wrapper_roles: ClassVar[frozenset[str]] = frozenset(
        {"lifecycle", "pooled_init_vmap", "vectorization"}
    )
    stack_rules: ClassVar[tuple[WrapperStackRule, ...]] = (
        WrapperStackRule(
            "vectorization",
            "PooledInitVmapWrapper cannot wrap another vectorization wrapper",
        ),
    )

    class PooledInitVmapState(WrappedState):
        init_key: Key = field()
        last_final: Info = field()
        final_valid: jax.Array = field()

    def __post_init__(self):
        if isinstance(self.batch_size, bool) or not isinstance(
            self.batch_size, Integral
        ):
            raise TypeError("batch_size must be a positive concrete static int")
        if isinstance(self.pool_size, bool) or not isinstance(self.pool_size, Integral):
            raise TypeError("pool_size must be a positive concrete static int")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.pool_size <= 0:
            raise ValueError("pool_size must be positive")
        object.__setattr__(self, "batch_size", int(self.batch_size))
        object.__setattr__(self, "pool_size", int(self.pool_size))
        if not self.env.init_can_replace_reset:
            raise ValueError(
                "PooledInitVmapWrapper requires an environment with "
                "init_can_replace_reset=True"
            )
        super().__post_init__()

    @property
    @override
    def init_can_replace_reset(self) -> bool:
        return False

    @override
    def init(self, key: Key) -> tuple[PooledInitVmapState, Info]:
        keys = _split_or_keep_key(key, self.batch_size + 1)
        key_next, keys_pool = keys[0], keys[1:]
        inner_state, info = jax.vmap(self.env.init)(keys_pool)
        last_final = jax.tree.map(jnp.zeros_like, info)
        final_valid = jnp.zeros((self.batch_size,), dtype=jnp.bool_)
        state = self.PooledInitVmapState(
            inner_state=inner_state,
            init_key=key_next,
            last_final=last_final,
            final_valid=final_valid,
        )
        return state, info.update(final=last_final, final_valid=final_valid)

    @override
    def reset(
        self, state: PooledInitVmapState, key: Key
    ) -> tuple[PooledInitVmapState, Info]:
        keys = _split_or_keep_key(key, self.batch_size + 1)
        key_next, keys_pool = keys[0], keys[1:]
        inner_state, info = jax.vmap(self.env.reset)(state.inner_state, keys_pool)
        state = state.replace(inner_state=inner_state, init_key=key_next)
        return state, info.update(final=state.last_final, final_valid=state.final_valid)

    @override
    def step(
        self, state: PooledInitVmapState, action: PyTree
    ) -> tuple[PooledInitVmapState, Info]:
        transition_state, transition_info = jax.vmap(self.env.step)(
            state.inner_state, action
        )
        done = jnp.asarray(transition_info.terminated) | jnp.asarray(
            transition_info.truncated
        )
        next_init_key = jax.random.fold_in(state.init_key, 1)
        continuing_info = transition_info.update(
            final=state.last_final,
            final_valid=state.final_valid,
        )

        def construct_pool(_):
            key_pool = jax.random.fold_in(state.init_key, 0)
            keys_pool = jax.random.split(key_pool, self.pool_size)
            inner_states_pool, infos_pool = jax.vmap(self.env.init)(keys_pool)

            key_idxs = jax.random.fold_in(state.init_key, 2)
            pool_idxs = jax.random.randint(
                key_idxs, (self.batch_size,), 0, self.pool_size
            )
            mapped_init_state = jax.tree.map(lambda x: x[pool_idxs], inner_states_pool)
            mapped_init_info = jax.tree.map(lambda x: x[pool_idxs], infos_pool)

            def select(on_done, on_continue):
                return jax.vmap(jnp.where)(done, on_done, on_continue)

            final_inner_state = jax.tree.map(
                select, mapped_init_state, transition_state
            )
            final_last_final = jax.tree.map(select, transition_info, state.last_final)
            final_valid = state.final_valid | done

            terminal_info = mapped_init_info.update(
                reward=transition_info.reward,
                terminated=transition_info.terminated,
                truncated=transition_info.truncated,
                final=transition_info,
                final_valid=jnp.ones((self.batch_size,), dtype=jnp.bool_),
            )
            final_info = jax.tree.map(select, terminal_info, continuing_info)
            return (
                final_inner_state,
                final_last_final,
                final_valid,
                final_info,
            )

        def keep_transitions(_):
            return (
                transition_state,
                state.last_final,
                state.final_valid,
                continuing_info,
            )

        final_inner_state, final_last_final, final_valid, final_info = jax.lax.cond(
            jnp.any(done), construct_pool, keep_transitions, operand=None
        )

        state = state.replace(
            inner_state=final_inner_state,
            init_key=next_init_key,
            last_final=final_last_final,
            final_valid=final_valid,
        )
        return state, final_info

    @cached_property
    @override
    def observation_space(self) -> spaces.Space:
        return spaces.BatchedSpace(
            space=self.env.observation_space, batch_size=self.batch_size
        )

    @cached_property
    @override
    def action_space(self) -> spaces.Space:
        return spaces.BatchedSpace(
            space=self.env.action_space, batch_size=self.batch_size
        )
