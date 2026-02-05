import warnings
from functools import cached_property
from typing import override

import jax
import jax.numpy as jnp

from envelope import spaces
from envelope.environment import Info
from envelope.struct import field
from envelope.typing import Key, PyTree
from envelope.wrappers.vmap_wrapper import _split_or_keep_key
from envelope.wrappers.wrapper import WrappedState, Wrapper


class PooledInitVmapWrapper(Wrapper):
    batch_size: int = field(kw_only=True)
    pool_size: int = field(kw_only=True)

    class PooledInitVmapState(WrappedState):
        init_key: Key = field()
        last_final: Info = field()

    @override
    def init(self, key: Key) -> tuple[WrappedState, Info]:
        keys = _split_or_keep_key(key, self.batch_size + 1)
        key_next, keys_pool = keys[0], keys[1:]
        inner_state, info = jax.vmap(self.env.init)(keys_pool)
        state = self.PooledInitVmapState(
            inner_state=inner_state,
            init_key=key_next,
            last_final=info,
        )
        return state, info.update(final=state.last_final)

    @override
    def reset(self, key: Key, state: WrappedState) -> tuple[WrappedState, Info]:
        # It's hard to support reset for this wrapper.
        # We would have to init the state of a pool of unwrapped environments, and then
        # somehow inject this into the stack of wrapped states. The current data
        # structure for wrapped states does not make this possible without being super
        # hacky, and violating the assumption that wrapped states are opaque (we would
        # likely have to recursively descend by checking if
        # hasattr(state, "inner_state")).
        # Since there is currently no use case in which we need to carry state across
        # episodes before vmapping, we will implement this later.
        keys = _split_or_keep_key(key, self.batch_size + 1)
        key_next, keys_pool = keys[0], keys[1:]
        inner_state, info = jax.vmap(self.env.reset)(keys_pool, state.inner_state)
        state = state.replace(inner_state=inner_state, init_key=key_next)
        return state, info.update(final=state.last_final)

    @override
    def step(self, state: WrappedState, action: PyTree) -> tuple[WrappedState, Info]:
        inner_state, info = jax.vmap(self.env.step)(state.inner_state, action)
        done = info.terminated | info.truncated

        # Compute pool_size fresh init states
        key_pool = jax.random.fold_in(state.init_key, 0)
        next_init_key = jax.random.fold_in(state.init_key, 1)
        keys_pool = jax.random.split(key_pool, self.pool_size)
        inner_states_pool, infos_pool = jax.vmap(self.env.init)(keys_pool)

        # Randomly assign each env a init state from the pool
        key_idxs = jax.random.fold_in(state.init_key, 2)
        pool_idxs = jax.random.randint(key_idxs, (self.batch_size,), 0, self.pool_size)

        # Expand pool states to batch_size via indexing
        mapped_init_state = jax.tree.map(lambda x: x[pool_idxs], inner_states_pool)
        mapped_init_info = jax.tree.map(lambda x: x[pool_idxs], infos_pool)

        # Select inner_state: init for done envs, continue for others
        final_inner_state = jax.tree.map(
            lambda init, curr: jax.vmap(jnp.where)(done, init, curr),
            mapped_init_state,
            inner_state,
        )

        # Select last_final: on done, store terminal info; on continue, keep previous
        final_last_final = jax.tree.map(
            lambda curr, prev: jax.vmap(jnp.where)(done, curr, prev),
            info,
            state.last_final,
        )

        # Build final_info with final field
        # For done envs: obs is new initial obs, final is terminal info
        # For continue envs: obs is current obs, final is previous last_final
        final_obs = jax.tree.map(
            lambda init, curr: jax.vmap(jnp.where)(done, init, curr),
            mapped_init_info.obs,
            info.obs,
        )
        final_final = jax.tree.map(
            lambda curr, prev: jax.vmap(jnp.where)(done, curr, prev),
            info,  # Terminal info snapshot for done envs
            state.last_final,  # Previous episode's final for continue envs
        )
        final_info = info.update(obs=final_obs, final=final_final)

        state = state.replace(
            inner_state=final_inner_state,
            init_key=next_init_key,
            last_final=final_last_final,
        )
        return state, final_info

    @override
    @cached_property
    def observation_space(self) -> spaces.Space:
        return spaces.BatchedSpace(
            space=self.env.observation_space, batch_size=self.batch_size
        )

    @override
    @cached_property
    def action_space(self) -> spaces.Space:
        return spaces.BatchedSpace(
            space=self.env.action_space, batch_size=self.batch_size
        )
