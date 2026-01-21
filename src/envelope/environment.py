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
    adding any wrapper-specific fields. `reset` may optionally receive a prior
    state (for cross-episode persistence) and arbitrary **kwargs that wrappers
    or environments can use.
    """

    @abstractmethod
    def reset(
        self, key: Key, state: State | None = None, **kwargs
    ) -> tuple[State, Info]: ...

    @abstractmethod
    def step(self, state: State, action: PyTree, **kwargs) -> tuple[State, Info]: ...

    @abstractmethod
    @cached_property
    def observation_space(self) -> spaces.Space: ...

    @abstractmethod
    @cached_property
    def action_space(self) -> spaces.Space: ...

    @property
    def unwrapped(self) -> "Environment":
        return self
