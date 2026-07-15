from dataclasses import KW_ONLY
from functools import cached_property
from typing import Any, ClassVar, NamedTuple, override

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


class WrapperStackRule(NamedTuple):
    """Reject a wrapper stack when a matching role appears inside this wrapper.

    ``inner_role`` is compared with the ``wrapper_roles`` declared by every wrapper in
    the wrapped ``env`` chain. ``message`` becomes the construction-time error and may
    use ``{outer}`` for the wrapper being constructed and ``{inner}`` for the first
    matching inner wrapper.
    """

    inner_role: str
    message: str


class Wrapper(Environment):
    """Base class for environments that delegate to another environment.

    Concrete wrappers may declare composition metadata through ``wrapper_roles`` and
    ``stack_rules``:

    - ``wrapper_roles`` names behavior provided by that wrapper, such as
      ``"vectorization"``, ``"lifecycle"``, or ``"persistent"``. Roles describe the
      wrapper itself; they do not include roles provided by its inner environment.
      Envelope uses them for construction-time validation and for wrappers that need
      to detect behavior elsewhere in their inner chain.
    - ``stack_rules`` describes roles that must not occur inside the wrapper being
      constructed. For each rule, construction walks through ``env`` and its nested
      wrappers. Finding ``rule.inner_role`` raises ``ValueError`` with the rule's
      message. A rule only examines inner wrappers; it does not constrain wrappers
      that may later be placed outside this one.

    For example, a wrapper with
    ``WrapperStackRule("vectorization", "{outer} must be inside {inner}")`` cannot be
    constructed around a vectorization wrapper. The formatted error identifies both
    concrete wrapper classes.

    These labels concern wrapper ordering only. Whether an ``init`` result may replace
    a later ``reset`` result is the separate ``init_can_replace_reset`` capability.
    """

    env: Environment = field()
    wrapper_roles: ClassVar[frozenset[str]] = frozenset()
    stack_rules: ClassVar[tuple[WrapperStackRule, ...]] = ()

    def __post_init__(self):
        for rule in self.stack_rules:
            inner_wrapper = _find_wrapper_by_role(self.env, rule.inner_role)
            if inner_wrapper is not None:
                raise ValueError(
                    rule.message.format(
                        outer=type(self).__name__,
                        inner=type(inner_wrapper).__name__,
                    )
                )
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
    def init_can_replace_reset(self) -> bool:
        """Pass through whether ``init`` can replace ``reset``."""
        return self.env.init_can_replace_reset

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


def _find_wrapper_by_role(env: Environment, role: str) -> Wrapper | None:
    """Return the first wrapper in ``env``'s chain with ``role`` metadata."""
    current = env
    while isinstance(current, Wrapper):
        if role in current.wrapper_roles:
            return current
        current = current.env
    return None
