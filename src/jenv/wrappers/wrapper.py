from functools import cached_property
from typing import Protocol, override, runtime_checkable

from jenv import spaces
from jenv.environment import Environment, Info, State
from jenv.struct import Container, FrozenPyTreeNode, field
from jenv.typing import Key, PyTree


@runtime_checkable
class WrappedState(Protocol):
    """
    Canonical environment state with explicit semantics:
      - core: base environment's state; replaced on reset
      - episodic: wrapper-owned state that resets on episode boundaries
      - persistent: wrapper-owned state that persists across episodes
    """

    core: PyTree
    episodic: Container
    persistent: Container

    def update(self, **changes: PyTree) -> "WrappedState": ...
    def __getattr__(self, name: str) -> PyTree: ...


class FrozenWrappedState(FrozenPyTreeNode):
    core: PyTree = field()
    episodic: PyTree = field(default_factory=Container)
    persistent: PyTree = field(default_factory=Container)

    def update(self, **changes: PyTree) -> "FrozenWrappedState":
        return self.replace(**changes)


class Wrapper(Environment):
    """Wrapper for environments."""

    env: Environment = field(kw_only=True)

    @override
    def reset(self, key: Key) -> tuple[State, Info]:
        return self.env.reset(key)

    @override
    def step(self, state: WrappedState, action: PyTree) -> tuple[WrappedState, Info]:
        return self.env.step(state, action)

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
