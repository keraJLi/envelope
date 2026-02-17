import jax
from typing_extensions import override

from envelope.environment import Info, State
from envelope.struct import FrozenPyTreeNode, field
from envelope.typing import Key, PyTree
from envelope.wrappers.wrapper import WrappedState, Wrapper


class EpisodeStatistics(FrozenPyTreeNode):
    reward: jax.Array = field(default=0.0)
    length: jax.Array = field(default=0.0)


class EpisodeStatisticsWrapper(Wrapper):
    class EpisodeStatisticsState(WrappedState):
        stats: EpisodeStatistics = field(default=EpisodeStatistics())

    @override
    def init(self, key: Key) -> tuple[State, Info]:
        inner_state, info = self.env.init(key)
        state = self.EpisodeStatisticsState(inner_state=inner_state)
        return state, info.update(stats=state.stats)

    @override
    def reset(self, state: State, key: Key) -> tuple[State, Info]:
        inner_state, info = self.env.reset(state.inner_state, key)
        state = state.replace(inner_state=inner_state)
        return state, info.update(stats=state.stats)

    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        inner_state, info = self.env.step(state.inner_state, action)
        stats = state.stats.replace(
            reward=state.stats.reward + info.reward,
            length=state.stats.length + 1,
        )
        state = state.replace(inner_state=inner_state, stats=stats)
        return state, info.update(stats=stats)
