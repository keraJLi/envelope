"""Compatibility wrappers for various RL environment libraries."""

from typing import Any, Protocol, Self

# Lazy imports to avoid requiring all dependencies at once
_env_module_map = {
    "gymnax": ("envelope.compat.gymnax_envelope", "GymnaxEnvelope"),
    "brax": ("envelope.compat.brax_envelope", "BraxEnvelope"),
    "navix": ("envelope.compat.navix_envelope", "NavixEnvelope"),
    "jumanji": ("envelope.compat.jumanji_envelope", "JumanjiEnvelope"),
    "kinetix": ("envelope.compat.kinetix_envelope", "KinetixEnvelope"),
    "craftax": ("envelope.compat.craftax_envelope", "CraftaxEnvelope"),
    "mujoco_playground": (
        "envelope.compat.mujoco_playground_envelope",
        "MujocoPlaygroundEnvelope",
    ),
}


class HasFromNameInit(Protocol):
    @classmethod
    def from_name(
        cls, env_name: str, env_kwargs: dict[str, Any] | None = None, **kwargs
    ) -> Self: ...

    """Creates an environment from a name and keyword arguments. Unless otherwise noted,
    the created environment will have it's default parameters, with truncation and auto
    reset disabled.
    
    Args:
        env_name: Environment name
        env_kwargs: Keyword arguments passed to the environment constructor
        **kwargs: Additional keyword arguments passed to the environment wrapper
    """


def create(env_name: str, env_kwargs: dict[str, Any] | None = None, **kwargs):
    """Create an environment from a prefixed environment ID.

    Args:
        env_name: Environment ID in the format "suite::env_name" (e.g., "brax::ant")
        env_kwargs: Keyword arguments passed to the suite's environment constructor
        **kwargs: Additional keyword arguments passed to the environment wrapper

    Returns:
        An instance of the wrapped environment

    Examples:
        >>> env = create("jumanji::snake")
        >>> env = create("brax::ant", env_kwargs={"backend": "spring"})
        >>> env = create("gymnax::CartPole-v1", env_params=...)
    """
    original_env_id = env_name
    if "::" not in env_name:
        raise ValueError(
            f"Environment ID must be in format 'suite::env_name', got: {original_env_id}"
        )

    suite, env_name = env_name.split("::", 1)
    if not suite or not env_name:
        raise ValueError(
            f"Environment ID must be in format 'suite::env_name', got: {original_env_id}"
        )

    if suite not in _env_module_map:
        raise ValueError(
            f"Unknown environment suite: {suite}. "
            f"Available suites: {list(_env_module_map.keys())}"
        )

    # Lazy import the wrapper class
    module_name, class_name = _env_module_map[suite]
    try:
        import importlib

        module = importlib.import_module(module_name)
        env_class: HasFromNameInit = getattr(module, class_name)
    except ImportError as e:
        raise ImportError(
            f"Failed to import {suite} wrapper. "
            f"Make sure you have installed the '{suite}' dependencies. "
            f"Original error: {e}"
        ) from e

    env = env_class.from_name(env_name, env_kwargs=env_kwargs, **kwargs)

    # Wrap with TruncationWrapper using adapter's default
    default_max_steps = getattr(env, "default_max_steps", None)
    if default_max_steps is not None:
        from envelope.wrappers.truncation_wrapper import TruncationWrapper

        env = TruncationWrapper(env=env, max_steps=int(default_max_steps))

    return env


__all__ = ["create"]
