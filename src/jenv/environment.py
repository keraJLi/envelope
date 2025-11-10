from abc import ABC, abstractmethod
from functools import cached_property

from jenv import spaces
from jenv.struct import PyTreeNode
from jenv.typing import Key, PyTree

__all__ = ["Environment"]


State = PyTree


class Info(PyTreeNode):
    obs: PyTree
    reward: float
    terminated: bool
    truncated: bool

    @cached_property
    def done(self) -> bool:
        return self.terminated | self.truncated


class Environment(ABC, PyTreeNode):
    """
    Base class for all environments.
    """

    @abstractmethod
    def reset(self, key: Key) -> tuple[State, Info]: ...

    @abstractmethod
    def step(self, key: Key, state: State, action: PyTree) -> tuple[State, Info]: ...

    @abstractmethod
    @cached_property
    def observation_space(self) -> spaces.Space: ...

    @abstractmethod
    @cached_property
    def action_space(self) -> spaces.Space: ...
