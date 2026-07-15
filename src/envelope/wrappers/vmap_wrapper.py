from functools import cached_property
from typing import ClassVar, override

import jax
import jax.numpy as jnp

from envelope import spaces
from envelope.environment import Info
from envelope.struct import static_field
from envelope.typing import Key, PyTree
from envelope.wrappers.wrapper import Wrapper, WrapperStackRule


def is_single_key(key):
    # New-style typed keys have dtype like key<fry>
    if jnp.issubdtype(key.dtype, jax.dtypes.prng_key):
        return key.ndim == 0
    return key.shape == (2,)


def _split_or_keep_key(key: Key, batch_size: int) -> Key:
    if is_single_key(key):
        return jax.random.split(key, batch_size)
    elif key.shape[0] == batch_size:
        return key
    raise ValueError(
        f"reset key's leading dimension ({key.shape[0]}) must match "
        f"batch_size ({batch_size})."
    )


class VmapWrapper(Wrapper):
    """Does not wrap the state."""

    batch_size: int = static_field(kw_only=True)
    wrapper_roles: ClassVar[frozenset[str]] = frozenset({"vectorization"})
    stack_rules: ClassVar[tuple[WrapperStackRule, ...]] = (
        WrapperStackRule(
            "normalization",
            "ObservationNormalizationWrapper must wrap {outer}, not be inside it",
        ),
    )

    @override
    def init(self, key: Key) -> tuple[PyTree, Info]:
        keys = _split_or_keep_key(key, self.batch_size)
        state, info = jax.vmap(self.env.init)(keys)
        return state, info

    @override
    def reset(self, state: PyTree, key: Key) -> tuple[PyTree, Info]:
        keys = _split_or_keep_key(key, self.batch_size)
        state, info = jax.vmap(self.env.reset)(state, keys)
        return state, info

    @override
    def step(self, state: PyTree, action: PyTree) -> tuple[PyTree, Info]:
        state, info = jax.vmap(self.env.step)(state, action)
        return state, info

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
