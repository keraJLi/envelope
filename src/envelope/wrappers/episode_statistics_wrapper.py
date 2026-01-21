from dataclasses import field
from envelope.wrappers import Wrapper
from typing import override

from envelope.environment import Environment, Info, State
from envelope.typing import Key, PyTree, Array


class EpisodeStatisticsWrapper(Wrapper):
    class StatisticsState(WappedState):
        episode_reward: Array
        episode_length: Array
        _pointer: int = field(default=0)

    def reset(
        self, key: Key, state: State | None = None, **kwargs
    ) -> tuple[State, Info]:
        state, info = self.env.reset(key, state=state, **kwargs)
        info = 
        return state, info


    @override
    def reset(
        self, key: Key, state: State | None = None, **kwargs
    ) -> tuple[State, Info]:
        state, info = self.env.reset(key, state=state, **kwargs)
        info = 
        return state, info

    @override
    def step(self, state: State, action: PyTree, **kwargs) -> tuple[State, Info]:
        next_state, info = self.env.step(state, action, **kwargs)
        info = self._update_episode_statistics(info)
        return next_state, info

    def _update_episode_statistics(self, info: Info) -> Info:
        """Update episode statistics in the info dictionary."""
        if "episode_statistics" not in info:
            info["episode_statistics"] = {
                "reward": 0.0,
                "length": 0,
            }
        info["episode_statistics"]["reward"] += info.get("reward", 0.0)
        info["episode_statistics"]["length"] += 1
        return info

