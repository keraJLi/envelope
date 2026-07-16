from dataclasses import KW_ONLY
from functools import cached_property
from typing import Any, override

from envelope import spaces
from envelope.environment import (
    Environment,
    Info,
    StackConstraint,
    State,
    not_containing,
    not_inside,
)
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


class VectorizingWrapper:
    """Marker mixin for wrappers that introduce a vectorized environment."""


class PooledInitializationWrapper:
    """Marker mixin for wrappers that replace resets with pooled init results."""


class Wrapper(Environment):
    """Base class for environments that delegate to another environment.

    Environments and wrappers may declare directional ``stack_constraints`` using
    ``not_inside`` and ``not_containing``. Construction validates the complete stack,
    so constraints also apply through unrelated intermediate wrappers.
    """

    env: Environment = field()

    def __post_init__(self):
        _validate_stack(self)
        super().__post_init__()

    @override
    def init(self, key: Key) -> tuple[State, Info]:
        return self.env.init(key)

    @override
    def reset(self, state: State, key: Key) -> tuple[State, Info]:
        return self.env.reset(state, key)

    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        return self.env.step(state, action)

    @cached_property
    @override
    def observation_space(self) -> spaces.Space:
        return self.env.observation_space

    @cached_property
    @override
    def action_space(self) -> spaces.Space:
        return self.env.action_space

    @property
    @override
    def unwrapped(self) -> Environment:
        return self.env.unwrapped

    @property
    @override
    def default_max_steps(self) -> int | None:
        return self.env.default_max_steps

    def __getattribute__(self, name: str) -> Any:
        """Forward genuinely missing attributes without hiding wrapper failures.

        ``__getattr__`` is also invoked when a descriptor defined on the wrapper
        raises ``AttributeError``.  Blind delegation in that situation masks the
        original error and makes debugging wrapper properties needlessly hard.
        Distinguish a genuinely absent attribute from a failing descriptor before
        consulting the wrapped environment.
        """
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            cls = object.__getattribute__(self, "__class__")
            if any(name in ancestor.__dict__ for ancestor in cls.__mro__):
                raise
            if name == "__setstate__":
                raise
            env = object.__getattribute__(self, "env")
            return getattr(env, name)


def _validate_stack(root: Environment) -> None:
    layers = list(_iter_stack(root))
    for index, owner in enumerate(layers):
        for constraint in getattr(owner, "stack_constraints", ()):
            candidates = (
                layers[:index]
                if constraint.direction == "outer"
                else layers[index + 1 :]
            )
            for candidate in candidates:
                if not isinstance(candidate, constraint.environment_types):
                    continue
                if constraint.direction == "outer":
                    raise ValueError(
                        f"{type(owner).__name__} cannot be inside "
                        f"{type(candidate).__name__}"
                    )
                raise ValueError(
                    f"{type(owner).__name__} cannot contain {type(candidate).__name__}"
                )


def _iter_stack(env: Environment):
    current = env
    while True:
        yield current
        if not isinstance(current, Wrapper):
            return
        current = current.env


__all__ = [
    "PooledInitializationWrapper",
    "StackConstraint",
    "VectorizingWrapper",
    "WrappedState",
    "Wrapper",
    "not_containing",
    "not_inside",
]
