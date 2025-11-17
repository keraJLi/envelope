from typing import Callable, TypeVar, Union, override

from jenv.environment import Environment, Info
from jenv.struct import Container
from jenv.typing import Key, PyTree
from jenv.wrappers.wrapper import FrozenWrappedState, WrappedState, Wrapper

T = TypeVar("T", bound=Union[Environment, Callable[[], Environment]])


class CanonicalizeWrapper(Wrapper):
    """
    Ensures the externally visible state is a WrappedState:
      - On reset: wraps the inner environment's native state into `core`.
      - On step: passes `state.core` to the inner environment and re-wraps the result,
        preserving `episodic` and `persistent` fields across steps.
    This wrapper should be applied first in a wrapper stack.

    Can be used as a decorator:
        @CanonicalizeWrapper
        class MyEnv(Environment):
            ...

        @CanonicalizeWrapper
        def make_env():
            return SomeEnv()
    """

    def __new__(cls, env: T | None = None, **kwargs):
        # If called as a decorator: env is a class or callable, not an Environment instance
        if env is not None and not isinstance(env, Environment):
            if isinstance(env, type) and issubclass(env, Environment):
                # Decorating a class: @CanonicalizeWrapper class MyEnv: ...
                return cls._decorator(env)
            elif callable(env):
                # Decorating a function: @CanonicalizeWrapper def make_env(): ...
                return cls._decorator(env)
        # Normal instantiation: env is an Environment instance or None (will be set via __init__)
        return super().__new__(cls)

    @classmethod
    def _decorator(cls, target: T) -> T:
        """Decorator that wraps classes or functions returning environments."""
        if isinstance(target, type) and issubclass(target, Environment):
            # Decorate a class: wrap instances
            class WrappedClass(target):
                def __new__(wrapped_cls, *args, **kwargs):
                    # Create instance normally, then wrap it
                    instance = target(*args, **kwargs)
                    return cls(env=instance)

            WrappedClass.__name__ = f"Canonicalized{target.__name__}"
            WrappedClass.__qualname__ = f"Canonicalized{target.__qualname__}"
            return WrappedClass
        elif callable(target):
            # Decorate a function: wrap the returned environment
            def wrapped_fn(*args, **kwargs):
                env = target(*args, **kwargs)
                return cls(env=env)

            wrapped_fn.__name__ = f"canonicalized_{target.__name__}"
            wrapped_fn.__qualname__ = f"canonicalized_{target.__qualname__}"
            return wrapped_fn
        else:
            raise TypeError(
                f"CanonicalizeWrapper decorator can only be applied to Environment classes or callables, got {type(target)}"
            )

    @override
    def reset(self, key: Key) -> tuple[WrappedState, Info]:
        inner_state, info = self.env.reset(key)
        # Wrap if not already a WrappedState
        if (
            hasattr(inner_state, "core")
            and hasattr(inner_state, "episodic")
            and hasattr(inner_state, "persistent")
        ):
            wrapped = inner_state
        else:
            wrapped = FrozenWrappedState(
                core=inner_state, episodic=Container(), persistent=Container()
            )
        return wrapped, info

    @override
    def step(self, state: WrappedState, action: PyTree) -> tuple[WrappedState, Info]:
        # Unwrap for the inner env; CanonicalizeWrapper guarantees WrappedState outward
        inner_next_state, info = self.env.step(state.core, action)
        # Always re-wrap and preserve wrapper fields
        wrapped_next = (
            inner_next_state
            if (
                hasattr(inner_next_state, "core")
                and hasattr(inner_next_state, "episodic")
                and hasattr(inner_next_state, "persistent")
            )
            else FrozenWrappedState(
                core=inner_next_state, episodic=Container(), persistent=Container()
            )
        )
        wrapped_next = wrapped_next.update(
            episodic=state.episodic, persistent=state.persistent
        )
        return wrapped_next, info
