from functools import cached_property
from math import prod
from typing import Any, ClassVar, cast, override

import jax
from jax import numpy as jnp

from envelope.environment import Info
from envelope.spaces import (
    BatchedSpace,
    Continuous,
    PyTreeSpace,
    Space,
    peel_batched,
    rebatch,
)
from envelope.struct import field, static_field
from envelope.typing import Key, PyTree
from envelope.wrappers.normalization import RunningMeanVar, update_rmv
from envelope.wrappers.wrapper import (
    PooledInitializationWrapper,
    WrappedState,
    Wrapper,
    not_inside,
)


class ObservationNormalizationWrapper(Wrapper):
    """Normalize a batch with shared running statistics.

    If the inner stack emits ``final``, the wrapper also normalizes ``final.obs`` while
    retaining the raw terminal observation as ``final.unnormalized_obs``.
    """

    stack_constraints: ClassVar = (not_inside(PooledInitializationWrapper),)

    class ObservationNormalizationState(WrappedState):
        rmv_state: RunningMeanVar = field()

    stats_spec: PyTree | None = static_field(default=None, unsafe=True)
    """Per-leaf normalization statistics spec as a pytree of jax.ShapeDtypeStruct.
    Shapes must be broadcastable to the observation leaves. If None, inferred from
    the observation_space with BatchedSpace ignored; each leaf must have a floating
    dtype."""

    def __post_init__(self):
        if self.stats_spec is None:
            stats_spec = _infer_stats_spec(self.env.observation_space)
            object.__setattr__(self, "stats_spec", stats_spec)
        super().__post_init__()

    def _init_rmv_state(self) -> RunningMeanVar:
        def zeros(sd: jax.ShapeDtypeStruct) -> jax.Array:
            dtype = jnp.result_type(sd.dtype, jnp.float32)
            return jnp.zeros(sd.shape, dtype=dtype)

        def ones(sd: jax.ShapeDtypeStruct) -> jax.Array:
            dtype = jnp.result_type(sd.dtype, jnp.float32)
            return jnp.ones(sd.shape, dtype=dtype)

        mean = jax.tree.map(zeros, self.stats_spec)
        var = jax.tree.map(ones, self.stats_spec)
        count = jax.tree.map(lambda _: jnp.asarray(0), self.stats_spec)

        return RunningMeanVar(mean=mean, var=var, count=count)

    def _normalize_obs(self, obs: PyTree, rmv: RunningMeanVar) -> PyTree:
        def norm_leaf(x, mean, std, spec):
            x = jnp.asarray(x, dtype=mean.dtype)
            mean = jnp.broadcast_to(mean, x.shape)
            std = jnp.broadcast_to(std, x.shape)
            obs = (x - mean) / (std + 1e-8)
            return obs.astype(spec.dtype)

        return jax.tree.map(norm_leaf, obs, rmv.mean, rmv.std, self.stats_spec)

    def _normalize_and_update(
        self, state: ObservationNormalizationState, info: Info
    ) -> tuple[ObservationNormalizationState, Info]:
        raw_obs = info.obs
        reshaped_obs = jax.tree.map(_reshape_for_stats, raw_obs, self.stats_spec)
        rmv_state = update_rmv(state.rmv_state, reshaped_obs)
        norm_obs = self._normalize_obs(raw_obs, rmv_state)

        info = info.update(obs=norm_obs, unnormalized_obs=raw_obs)

        if hasattr(info, "final") and hasattr(info, "final_valid"):
            info_with_final = cast(Any, info)
            raw_final_obs = info_with_final.final.obs
            norm_final_obs = jax.tree.map(
                lambda obs: _mask_valid(info_with_final.final_valid, obs),
                self._normalize_obs(raw_final_obs, rmv_state),
            )

            final = info_with_final.final.update(
                obs=norm_final_obs, unnormalized_obs=raw_final_obs
            )
            info = info.update(final=final)

        state = state.replace(rmv_state=rmv_state)
        return state, info

    @override
    def init(self, key: Key) -> tuple[ObservationNormalizationState, Info]:
        inner_state, info = self.env.init(key)
        rmv_state = self._init_rmv_state()
        next_state = self.ObservationNormalizationState(
            inner_state=inner_state,
            rmv_state=rmv_state,
        )
        return self._normalize_and_update(next_state, info)

    @override
    def reset(
        self, state: ObservationNormalizationState, key: Key
    ) -> tuple[ObservationNormalizationState, Info]:
        inner_state, info = self.env.reset(state.inner_state, key)
        # Preserve running statistics across resets
        state = state.replace(inner_state=inner_state)
        return self._normalize_and_update(state, info)

    @override
    def step(
        self, state: ObservationNormalizationState, action: PyTree
    ) -> tuple[ObservationNormalizationState, Info]:
        inner_state, info = self.env.step(state.inner_state, action)
        state = state.replace(inner_state=inner_state)
        return self._normalize_and_update(state, info)

    @cached_property
    @override
    def observation_space(self) -> Space:
        batch_dims, base = peel_batched(self.env.observation_space)
        stats_spec = cast(Any, self.stats_spec)

        def to_continuous(inner_space: Space, spec: jax.ShapeDtypeStruct) -> Continuous:
            low = jnp.full(inner_space.shape, -jnp.inf, dtype=spec.dtype)
            high = jnp.full(inner_space.shape, jnp.inf, dtype=spec.dtype)
            return Continuous(low=low, high=high)

        if isinstance(base, PyTreeSpace):
            tree = jax.tree.map(
                to_continuous,
                base.tree,
                stats_spec,
                is_leaf=lambda node: isinstance(node, Space),
            )
            space: Space = PyTreeSpace(tree)
        else:
            space = to_continuous(base, stats_spec)
        return rebatch(space, batch_dims)


def _reshape_for_stats(x: jax.Array, spec: jax.ShapeDtypeStruct) -> jax.Array:
    """Turn leading and broadcast axes into samples."""
    x = jnp.asarray(x)
    spec_shape = tuple(spec.shape)
    offset = x.ndim - len(spec_shape)
    sample_axes = list(range(offset))
    statistic_axes: list[int] = []
    for index, spec_dim in enumerate(spec_shape):
        axis = offset + index
        if spec_dim == 1:
            sample_axes.append(axis)
        else:
            statistic_axes.append(axis)

    permutation = tuple(sample_axes + statistic_axes)
    if permutation != tuple(range(x.ndim)):
        x = jnp.transpose(x, permutation)
    sample_count = prod(x.shape[: len(sample_axes)]) if sample_axes else 1
    return x.reshape((sample_count,) + spec_shape)


def _mask_valid(valid: PyTree, obs: jax.Array) -> jax.Array:
    """Zero invalid batch elements without crossing event dimensions."""
    valid = jnp.asarray(valid, dtype=jnp.bool_)
    obs = jnp.asarray(obs)
    mask = valid.reshape(valid.shape + (1,) * (obs.ndim - valid.ndim))
    return jnp.where(mask, obs, jnp.zeros_like(obs))


def _infer_stats_spec(space: Space) -> PyTree:
    """
    Build a PyTree of jax.ShapeDtypeStruct for stats. Strip BatchedSpace layers,
    and for leaf spaces return (shape=space.shape, dtype=inferred).
    """

    def descend(sp: Space):
        if isinstance(sp, BatchedSpace):
            return descend(sp.space)
        if isinstance(sp, PyTreeSpace):
            return jax.tree.map(
                lambda s: descend(s),
                sp.tree,
                is_leaf=lambda n: isinstance(n, Space),
            )
        if not jnp.issubdtype(sp.dtype, jnp.floating):
            raise ValueError(
                f"Space {sp} has dtype {sp.dtype} which is not a floating point dtype"
            )
        return jax.ShapeDtypeStruct(tuple(sp.shape), sp.dtype)

    return descend(space)
