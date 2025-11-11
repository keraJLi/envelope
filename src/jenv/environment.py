from abc import ABC, abstractmethod
from functools import cached_property
from typing import Protocol, TypeAlias, runtime_checkable

from jenv import spaces
from jenv.struct import PyTreeNode
from jenv.typing import Key, PyTree

__all__ = ["Environment", "State"]


State: TypeAlias = PyTree


@runtime_checkable
class StepInfo(Protocol):
    obs: PyTree
    reward: float
    done: bool


class Environment(ABC, PyTreeNode):
    """
    Base class for all environments.
    """

    @abstractmethod
    def reset(self, key: Key) -> tuple[State, StepInfo]: ...

    @abstractmethod
    def step(self, state: State, action: PyTree) -> tuple[State, StepInfo]: ...

    @abstractmethod
    @cached_property
    def observation_space(self) -> spaces.Space: ...

    @abstractmethod
    @cached_property
    def action_space(self) -> spaces.Space: ...
