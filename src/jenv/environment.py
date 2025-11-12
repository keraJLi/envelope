from abc import ABC, abstractmethod
from functools import cached_property
from typing import Protocol, runtime_checkable

from jenv import spaces
from jenv.struct import Container, FrozenPyTreeNode
from jenv.typing import Key, PyTree

__all__ = ["Environment", "State"]


@runtime_checkable
class Info(Protocol):
    obs: PyTree
    reward: float
    terminated: bool
    truncated: bool

    def update(self, **changes: PyTree) -> "Info": ...
    def __getattr__(self, name: str) -> PyTree: ...


@runtime_checkable
class State(Protocol):
    env_state: PyTree

    def update(self, **changes: PyTree) -> "State": ...
    def __getattr__(self, name: str) -> PyTree: ...


class InfoContainer(Container):
    obs: PyTree
    reward: float
    terminated: bool
    truncated: bool

    @property
    def obs(self) -> PyTree:
        return self._fields["obs"]

    @property
    def reward(self) -> float:
        return self._fields["reward"]

    @property
    def terminated(self) -> bool:
        return self._fields["terminated"]

    @property
    def truncated(self) -> bool:
        return self._fields.get("truncated", False)


class StateContainer(Container):
    env_state: PyTree

    @property
    def env_state(self) -> PyTree:
        return self._fields["env_state"]


class Environment(ABC, FrozenPyTreeNode):
    """
    Base class for all environments.
    """

    @abstractmethod
    def reset(self, key: Key) -> tuple[State, Info]: ...

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
