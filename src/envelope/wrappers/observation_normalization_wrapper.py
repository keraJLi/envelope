from typing import override

import jax
from jax import numpy as jnp

from envelope.environment import Info
from envelope.spaces import BatchedSpace, PyTreeSpace, Space
from envelope.struct import field, static_field
from envelope.typing import Key, PyTree
from envelope.wrappers.normalization import RunningMeanVar, update_rmv
from envelope.wrappers.wrapper import WrappedState, Wrapper


class ObservationNormalizationWrapper(Wrapper):
    class ObservationNormalizationState(WrappedState):
        rmv_state: RunningMeanVar = field()

    stats_spec: PyTree | None = static_field(default=None)
    """Per-leaf normalization statistics spec as a pytree of jax.ShapeDtypeStruct.
    Shapes must be broadcastable to the observation leaves. If None, inferred from
    the observation_space with BatchedSpace ignored; each leaf must have a floating
    dtype."""

    def __post_init__(self):
        if self.stats_spec is None:
            stats_spec = _infer_stats_spec(self.env.observation_space)
            object.__setattr__(self, "stats_spec", stats_spec)

    def _init_rmv_state(self) -> RunningMeanVar:
        def zeros(sd: jax.ShapeDtypeStruct) -> jax.Array:
            return jnp.zeros(sd.shape, dtype=sd.dtype)

        def ones(sd: jax.ShapeDtypeStruct) -> jax.Array:
            return jnp.ones(sd.shape, dtype=sd.dtype)

        mean = jax.tree.map(zeros, self.stats_spec)
        var = jax.tree.map(ones, self.stats_spec)

        return RunningMeanVar(mean=mean, var=var, count=0)

    def _normalize_obs(self, obs: PyTree, rmv: RunningMeanVar) -> PyTree:
        def norm_leaf(x, mean, std, spec):
            mean = jnp.broadcast_to(mean, x.shape)
            std = jnp.broadcast_to(std, x.shape)
            obs = (x - mean) / (std + 1e-8)
            return obs.astype(spec.dtype)

        return jax.tree.map(norm_leaf, obs, rmv.mean, rmv.std, self.stats_spec)

    def _normalize_and_update(
        self, state: WrappedState, info: Info
    ) -> tuple[WrappedState, Info]:
        # Ensure each observation leaf is shaped as (-1, *spec.shape)
        reshaped_obs = jax.tree.map(
            lambda x, spec: x.reshape((-1,) + tuple(spec.shape)),
            info.obs,
            self.stats_spec,
        )
        rmv_state = update_rmv(state.rmv_state, reshaped_obs)
        norm_obs = self._normalize_obs(info.obs, rmv_state)

        state = self.ObservationNormalizationState(
            inner_state=state.inner_state, rmv_state=rmv_state
        )
        info = info.update(obs=norm_obs, unnormalized_obs=info.obs)
        return state, info

    @override
    def reset(
        self, key: Key, state: PyTree | None = None, **kwargs
    ) -> tuple[WrappedState, Info]:
        inner_state = None
        rmv_state = self._init_rmv_state()
        if state:
            inner_state = state.inner_state
            rmv_state = state.rmv_state

        inner_state, info = self.env.reset(key, inner_state, **kwargs)
        next_state = self.ObservationNormalizationState(
            inner_state=inner_state, rmv_state=rmv_state
        )
        return self._normalize_and_update(next_state, info)

    @override
    def step(
        self, state: WrappedState, action: PyTree, **kwargs
    ) -> tuple[WrappedState, Info]:
        inner_state, info = self.env.step(state.inner_state, action, **kwargs)
        state = state.replace(inner_state=inner_state)
        return self._normalize_and_update(state, info)


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
