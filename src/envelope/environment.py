from abc import ABC, abstractmethod
from dataclasses import field
from functools import cached_property

from envelope import spaces
from envelope.struct import Container, FrozenPyTreeNode
from envelope.typing import Array, Info, Key, PyTree, State

__all__ = ["Environment", "InfoContainer"]


class InfoContainer(Container):
    """
    Info container for environment emissions, including observation, reward, and
    termination/truncation flags. This container implements the `Info` protocol.

    Attributes:
        obs: The observation from the environment.
        reward: The reward from the environment.
        terminated: Whether the episode has terminated.
        truncated: Whether the episode has truncated.
    """

    obs: PyTree
    reward: float | Array
    terminated: bool
    truncated: bool = field(default=False)


class Environment(ABC, FrozenPyTreeNode):
    """
    Base class for all environments.

    State is an opaque PyTree owned by each environment; wrappers that stack
    environments should expose their wrapped env state as `inner_state` while
    adding any wrapper-specific fields.
    """

    @abstractmethod
    def init(self, key: Key) -> tuple[State, Info]:
        """
        Initialize environment state and sample the initial info.

        This method closely resembles the `reset` method of gymnasium or gymnax.
        However, it is normally called only once per environment lifecycle, as
        subsequent resets should be performed using the `Environment.reset` method,
        which may preserve episode-persistent state.
        """
        ...

    def reset(self, state: State, key: Key) -> tuple[State, Info]:
        """
        Reset the environment while preserving episode-persistent state.

        By default, `state` is ignored and `init` is called, as this most closely
        matches the standard reinforcement learning setting in which the starting state
        is sampled from a fixed distribution.
        """
        return self.init(key)

    @abstractmethod
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        """Step the environment given an action, returning the next state and info."""
        ...

    @abstractmethod
    @cached_property
    def observation_space(self) -> spaces.Space:
        """The space of observations."""
        ...

    @abstractmethod
    @cached_property
    def action_space(self) -> spaces.Space:
        """The space of actions."""
        ...

    @property
    def unwrapped(self) -> "Environment":
        return self
