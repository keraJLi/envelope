from dataclasses import KW_ONLY
from functools import cached_property
from typing import override

from envelope import spaces
from envelope.environment import Environment, Info, State
from envelope.struct import FrozenPyTreeNode, field
from envelope.typing import Key, PyTree


class WrappedState(FrozenPyTreeNode):
    inner_state: State = field()
    _: KW_ONLY

    @property
    def unwrapped(self) -> State:
        if hasattr(self.inner_state, "unwrapped"):
            return self.inner_state.unwrapped
        return self.inner_state


class Wrapper(Environment):
    """Wrapper for environments."""

    env: Environment = field(kw_only=True)

    @override
    def reset(
        self, key: Key, state: State | None = None, **kwargs
    ) -> tuple[State, Info]:
        return self.env.reset(key, state=state, **kwargs)

    @override
    def step(
        self, state: WrappedState, action: PyTree, **kwargs
    ) -> tuple[WrappedState, Info]:
        return self.env.step(state, action, **kwargs)

    @override
    @cached_property
    def observation_space(self) -> spaces.Space:
        return self.env.observation_space

    @override
    @cached_property
    def action_space(self) -> spaces.Space:
        return self.env.action_space

    @override
    @property
    def unwrapped(self) -> Environment:
        return self.env.unwrapped

    def __getattr__(self, name):
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.env, name)
