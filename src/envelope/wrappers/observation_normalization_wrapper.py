from dataclasses import InitVar
from functools import cached_property
from math import prod
from typing import Any, ClassVar, Protocol, cast, override

import jax
from jax import numpy as jnp

from envelope.environment import Info
from envelope.spaces import BatchedSpace, Continuous, PyTreeSpace, Space
from envelope.struct import field, static_field
from envelope.typing import Key, PyTree
from envelope.wrappers.normalization import RunningMeanVar, update_rmv
from envelope.wrappers.wrapper import WrappedState, Wrapper, _find_wrapper_by_role


class _InfoWithFinal(Info, Protocol):
    @property
    def final(self) -> Info: ...


class ObservationNormalizationWrapper(Wrapper):
    wrapper_roles: ClassVar[frozenset[str]] = frozenset({"normalization", "persistent"})

    class ObservationNormalizationState(WrappedState):
        rmv_state: RunningMeanVar = field()
        last_normalized_final_obs: PyTree | None = field(default=None)

    stats_spec: InitVar[Any] = None
    """Per-leaf normalization statistics spec as a pytree of jax.ShapeDtypeStruct.
    Shapes must be broadcastable to the observation leaves. If None, inferred from
    the observation_space with BatchedSpace ignored; each leaf must have a floating
    dtype."""

    _stats_treedef: Any = static_field(default=None, kw_only=True)
    _stats_leaves: tuple[jax.ShapeDtypeStruct, ...] = static_field(
        default=(), kw_only=True
    )
    _normalizes_final: bool = static_field(default=False, init=False)

    def __post_init__(self, stats_spec: PyTree | None):
        # JAX reconstruction supplies the already-encoded static fields and leaves
        # the init-only public argument at its default.
        if self._stats_treedef is None:
            if stats_spec is None:
                stats_spec = _infer_stats_spec(self.env.observation_space)
            leaves, treedef = jax.tree.flatten(stats_spec)
            if not leaves or not all(
                isinstance(leaf, jax.ShapeDtypeStruct) for leaf in leaves
            ):
                raise TypeError(
                    "stats_spec leaves must all be jax.ShapeDtypeStruct values"
                )
            object.__setattr__(self, "_stats_treedef", treedef)
            object.__setattr__(self, "_stats_leaves", tuple(leaves))
        object.__setattr__(
            self,
            "_normalizes_final",
            _find_wrapper_by_role(self.env, "final_info") is not None,
        )
        super().__post_init__()

    def _get_stats_spec(self) -> PyTree:
        if self._stats_treedef is None:
            raise RuntimeError("normalization statistics metadata is not initialized")
        return jax.tree.unflatten(self._stats_treedef, self._stats_leaves)

    def __getattribute__(self, name: str):
        # ``stats_spec`` remains the public constructor/view while its actual static
        # dataclass fields are an immutable treedef and tuple of leaves.
        if name == "stats_spec":
            return object.__getattribute__(self, "_get_stats_spec")()
        return super().__getattribute__(name)

    @property
    @override
    def init_can_replace_reset(self) -> bool:
        return False

    def _init_rmv_state(self) -> RunningMeanVar:
        stats_spec = self._get_stats_spec()

        def zeros(sd: jax.ShapeDtypeStruct) -> jax.Array:
            dtype = jnp.result_type(sd.dtype, jnp.float32)
            return jnp.zeros(sd.shape, dtype=dtype)

        def ones(sd: jax.ShapeDtypeStruct) -> jax.Array:
            dtype = jnp.result_type(sd.dtype, jnp.float32)
            return jnp.ones(sd.shape, dtype=dtype)

        mean = jax.tree.map(zeros, stats_spec)
        var = jax.tree.map(ones, stats_spec)
        count = jax.tree.map(lambda _: jnp.asarray(0), stats_spec)

        return RunningMeanVar(mean=mean, var=var, count=count)

    def _normalize_obs(self, obs: PyTree, rmv: RunningMeanVar) -> PyTree:
        def norm_leaf(x, mean, std, spec):
            x = jnp.asarray(x, dtype=mean.dtype)
            mean = jnp.broadcast_to(mean, x.shape)
            std = jnp.broadcast_to(std, x.shape)
            epsilon = jnp.asarray(1e-8, dtype=mean.dtype)
            obs = (x - mean) / (std + epsilon)
            return obs.astype(spec.dtype)

        return jax.tree.map(norm_leaf, obs, rmv.mean, rmv.std, self._get_stats_spec())

    def _normalize_and_update(
        self,
        state: ObservationNormalizationState,
        info: Info,
        *,
        update_final: bool,
    ) -> tuple[ObservationNormalizationState, Info]:
        raw_obs = info.obs
        reshaped_obs = jax.tree.map(
            _reshape_for_stats,
            raw_obs,
            self._get_stats_spec(),
        )
        rmv_state = update_rmv(state.rmv_state, reshaped_obs)
        norm_obs = self._normalize_obs(raw_obs, rmv_state)
        last_normalized_final_obs = state.last_normalized_final_obs

        info = info.update(obs=norm_obs, unnormalized_obs=raw_obs)

        if self._normalizes_final:
            if last_normalized_final_obs is None:
                raise RuntimeError(
                    "normalized final-observation cache is not initialized"
                )

            info_with_final = cast(_InfoWithFinal, info)
            raw_final_obs = info_with_final.final.obs
            if update_final:
                done = jnp.asarray(info.terminated, dtype=jnp.bool_) | jnp.asarray(
                    info.truncated, dtype=jnp.bool_
                )
                candidate = self._normalize_obs(raw_final_obs, rmv_state)
                last_normalized_final_obs = jax.tree.map(
                    lambda new, old: _select_completed(done, new, old),
                    candidate,
                    last_normalized_final_obs,
                )

            final = info_with_final.final.update(
                obs=last_normalized_final_obs,
                unnormalized_obs=raw_final_obs,
            )
            info = info.update(final=final)

        state = self.ObservationNormalizationState(
            inner_state=state.inner_state,
            rmv_state=rmv_state,
            last_normalized_final_obs=last_normalized_final_obs,
        )
        return state, info

    @override
    def init(self, key: Key) -> tuple[ObservationNormalizationState, Info]:
        inner_state, info = self.env.init(key)
        rmv_state = self._init_rmv_state()
        last_normalized_final_obs = None
        if self._normalizes_final:
            final = cast(_InfoWithFinal, info).final
            normalized_template = self._normalize_obs(final.obs, rmv_state)
            last_normalized_final_obs = jax.tree.map(
                jnp.zeros_like, normalized_template
            )
        next_state = self.ObservationNormalizationState(
            inner_state=inner_state,
            rmv_state=rmv_state,
            last_normalized_final_obs=last_normalized_final_obs,
        )
        return self._normalize_and_update(next_state, info, update_final=False)

    @override
    def reset(
        self, state: ObservationNormalizationState, key: Key
    ) -> tuple[ObservationNormalizationState, Info]:
        inner_state, info = self.env.reset(state.inner_state, key)
        # Preserve running statistics across resets
        next_state = self.ObservationNormalizationState(
            inner_state=inner_state,
            rmv_state=state.rmv_state,
            last_normalized_final_obs=state.last_normalized_final_obs,
        )
        return self._normalize_and_update(next_state, info, update_final=False)

    @override
    def step(
        self, state: ObservationNormalizationState, action: PyTree
    ) -> tuple[ObservationNormalizationState, Info]:
        inner_state, info = self.env.step(state.inner_state, action)
        state = state.replace(inner_state=inner_state)
        return self._normalize_and_update(state, info, update_final=True)

    @cached_property
    @override
    def observation_space(self) -> Space:
        return _normalized_observation_space(
            self.env.observation_space, self._get_stats_spec()
        )


def _reshape_for_stats(x: jax.Array, spec: jax.ShapeDtypeStruct) -> jax.Array:
    """Turn broadcast axes into samples while retaining statistic axes."""
    x = jnp.asarray(x)
    spec_shape = tuple(spec.shape)
    if len(spec_shape) > x.ndim:
        raise ValueError(
            f"stats shape {spec_shape} cannot broadcast to observation shape {x.shape}"
        )

    offset = x.ndim - len(spec_shape)
    sample_axes = list(range(offset))
    statistic_axes: list[int] = []
    for index, spec_dim in enumerate(spec_shape):
        axis = offset + index
        obs_dim = x.shape[axis]
        if spec_dim == obs_dim:
            statistic_axes.append(axis)
        elif spec_dim == 1:
            sample_axes.append(axis)
        else:
            raise ValueError(
                f"stats shape {spec_shape} cannot broadcast to observation "
                f"shape {x.shape}"
            )

    permutation = tuple(sample_axes + statistic_axes)
    if permutation != tuple(range(x.ndim)):
        x = jnp.transpose(x, permutation)
    sample_count = prod(x.shape[: len(sample_axes)]) if sample_axes else 1
    return x.reshape((sample_count,) + spec_shape)


def _select_completed(done: jax.Array, new: jax.Array, old: jax.Array) -> jax.Array:
    """Select newly completed batch elements without crossing event dimensions."""
    done = jnp.asarray(done, dtype=jnp.bool_)
    new = jnp.asarray(new)
    old = jnp.asarray(old)
    if new.shape != old.shape:
        raise ValueError(
            "new and cached normalized final observations must have matching shapes"
        )
    if done.ndim > new.ndim or new.shape[: done.ndim] != done.shape:
        raise ValueError(
            f"completion flags with shape {done.shape} must be a batch prefix of "
            f"final observations with shape {new.shape}"
        )
    mask = done.reshape(done.shape + (1,) * (new.ndim - done.ndim))
    return jnp.where(mask, new, old)


def _normalized_observation_space(space: Space, stats_spec: PyTree) -> Space:
    if isinstance(space, BatchedSpace):
        return BatchedSpace(
            space=_normalized_observation_space(space.space, stats_spec),
            batch_size=space.batch_size,
        )
    if isinstance(space, PyTreeSpace):
        tree = jax.tree.map(
            _normalized_observation_space,
            space.tree,
            stats_spec,
            is_leaf=lambda node: isinstance(node, (Space, jax.ShapeDtypeStruct)),
        )
        return PyTreeSpace(tree)

    dtype = stats_spec.dtype
    low = jnp.full(space.shape, -jnp.inf, dtype=dtype)
    high = jnp.full(space.shape, jnp.inf, dtype=dtype)
    return Continuous(low=low, high=high)


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
