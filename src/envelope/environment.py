from abc import ABC, abstractmethod
from dataclasses import field
from functools import cached_property
from typing import Protocol, runtime_checkable

from envelope import spaces
from envelope.struct import Container, FrozenPyTreeNode
from envelope.typing import Key, PyTree

__all__ = ["Environment", "State", "Info", "InfoContainer"]


@runtime_checkable
class Info(Protocol):
    obs: PyTree
    reward: float
    terminated: bool
    truncated: bool

    def update(self, **changes: PyTree) -> "Info": ...
    def __getattr__(self, name: str) -> PyTree: ...


class InfoContainer(Container):
    obs: PyTree
    reward: float
    terminated: bool
    truncated: bool = field(default=False)


# State remains a general PyTree alias; environments are not forced to WrappedState
State = PyTree


class Environment(ABC, FrozenPyTreeNode):
    """
    Base class for all environments.

    State is an opaque PyTree owned by each environment; wrappers that stack
    environments should expose their wrapped env state as `inner_state` while
    adding any wrapper-specific fields.

    Two distinct lifecycle methods:
        init(key) - Initialize environment and all state from scratch.
        reset(key, state) - Reset the inner environment while preserving
            episode-persistent state.
    """

    @abstractmethod
    def init(self, key: Key) -> tuple[State, Info]:
        """Initialize environment and all state from scratch."""
        ...

    def reset(self, key: Key, state: State) -> tuple[State, Info]:
        """Reset the inner environment while preserving episode-persistent state."""
        return self.init(key)

    @abstractmethod
    def step(self, state: State, action: PyTree) -> tuple[State, Info]: ...

    @abstractmethod
    @cached_property
    def observation_space(self) -> spaces.Space: ...

    @abstractmethod
    @cached_property
    def action_space(self) -> spaces.Space: ...

    @property
    def unwrapped(self) -> "Environment":
        return self
