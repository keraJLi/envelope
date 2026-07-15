from dataclasses import KW_ONLY
from functools import cached_property
from typing import Any, override

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

    env: Environment = field()

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
    def supports_init_pooling(self) -> bool:
        """Propagate reset-equivalence through transparent wrappers."""
        return self.env.supports_init_pooling

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


def _find_wrapper(
    env: Environment, wrapper_types: tuple[type[Wrapper], ...]
) -> Wrapper | None:
    """Return the first matching wrapper in ``env``'s wrapper chain."""
    current = env
    while isinstance(current, Wrapper):
        if isinstance(current, wrapper_types):
            return current
        current = current.env
    return None
