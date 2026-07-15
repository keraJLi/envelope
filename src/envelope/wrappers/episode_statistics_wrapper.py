from typing import override

import jax
import jax.numpy as jnp

from envelope.environment import Info
from envelope.struct import FrozenPyTreeNode, field
from envelope.typing import Key, PyTree
from envelope.wrappers.wrapper import WrappedState, Wrapper, _find_wrapper


class EpisodeStatistics(FrozenPyTreeNode):
    reward: jax.Array = field(default=0)
    length: jax.Array = field(default=0)


class EpisodeStatisticsWrapper(Wrapper):
    class EpisodeStatisticsState(WrappedState):
        stats: EpisodeStatistics = field(default=EpisodeStatistics())

    def __post_init__(self):
        from envelope.wrappers.autoreset_wrapper import AutoResetWrapper
        from envelope.wrappers.pooled_init_vmap_wrapper import PooledInitVmapWrapper

        lifecycle_wrapper = _find_wrapper(
            self.env, (AutoResetWrapper, PooledInitVmapWrapper)
        )
        if lifecycle_wrapper is not None:
            raise ValueError(
                "EpisodeStatisticsWrapper must be inside "
                f"{type(lifecycle_wrapper).__name__}"
            )

    @staticmethod
    def _empty_stats(info: Info) -> EpisodeStatistics:
        done = jnp.asarray(info.terminated) | jnp.asarray(info.truncated)
        return EpisodeStatistics(
            reward=jnp.zeros_like(jnp.asarray(info.reward)),
            length=jnp.zeros_like(done, dtype=jnp.int32),
        )

    @override
    def init(self, key: Key) -> tuple[EpisodeStatisticsState, Info]:
        inner_state, info = self.env.init(key)
        state = self.EpisodeStatisticsState(
            inner_state=inner_state, stats=self._empty_stats(info)
        )
        return state, info.update(stats=state.stats)

    @override
    def reset(
        self, state: EpisodeStatisticsState, key: Key
    ) -> tuple[EpisodeStatisticsState, Info]:
        inner_state, info = self.env.reset(state.inner_state, key)
        state = state.replace(inner_state=inner_state, stats=self._empty_stats(info))
        return state, info.update(stats=state.stats)

    @override
    def step(
        self, state: EpisodeStatisticsState, action: PyTree
    ) -> tuple[EpisodeStatisticsState, Info]:
        inner_state, info = self.env.step(state.inner_state, action)
        stats = state.stats.replace(
            reward=state.stats.reward + info.reward,
            length=state.stats.length + 1,
        )
        state = state.replace(inner_state=inner_state, stats=stats)
        return state, info.update(stats=stats)


class CumulativeStatisticsWrapper(Wrapper):
    """Statistics variant that deliberately persists across episode resets."""

    class CumulativeStatisticsState(WrappedState):
        cumulative_stats: EpisodeStatistics = field(default=EpisodeStatistics())

    @property
    @override
    def supports_init_pooling(self) -> bool:
        return False

    @override
    def init(self, key: Key) -> tuple[CumulativeStatisticsState, Info]:
        inner_state, info = self.env.init(key)
        state = self.CumulativeStatisticsState(
            inner_state=inner_state,
            cumulative_stats=EpisodeStatisticsWrapper._empty_stats(info),
        )
        return state, info.update(cumulative_stats=state.cumulative_stats)

    @override
    def reset(
        self, state: CumulativeStatisticsState, key: Key
    ) -> tuple[CumulativeStatisticsState, Info]:
        inner_state, info = self.env.reset(state.inner_state, key)
        state = state.replace(inner_state=inner_state)
        return state, info.update(cumulative_stats=state.cumulative_stats)

    @override
    def step(
        self, state: CumulativeStatisticsState, action: PyTree
    ) -> tuple[CumulativeStatisticsState, Info]:
        inner_state, info = self.env.step(state.inner_state, action)
        cumulative_stats = state.cumulative_stats.replace(
            reward=state.cumulative_stats.reward + info.reward,
            length=state.cumulative_stats.length + 1,
        )
        state = state.replace(
            inner_state=inner_state, cumulative_stats=cumulative_stats
        )
        return state, info.update(cumulative_stats=cumulative_stats)
