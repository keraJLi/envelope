"""Compatibility wrappers for various RL environment libraries."""

from typing import Any, Literal, Protocol, Self, cast

from envelope.environment import Environment

# Lazy imports to avoid requiring all dependencies at once
_env_module_map = {
    "gymnax": ("envelope.adapters.gymnax_envelope", "GymnaxEnvelope"),
    "brax": ("envelope.adapters.brax_envelope", "BraxEnvelope"),
    "navix": ("envelope.adapters.navix_envelope", "NavixEnvelope"),
    "jumanji": ("envelope.adapters.jumanji_envelope", "JumanjiEnvelope"),
    "kinetix": ("envelope.adapters.kinetix_envelope", "KinetixEnvelope"),
    "craftax": ("envelope.adapters.craftax_envelope", "CraftaxEnvelope"),
    "mujoco_playground": (
        "envelope.adapters.mujoco_playground_envelope",
        "MujocoPlaygroundEnvelope",
    ),
}

_INSTALL_SPECS = {
    "brax": "jax-envelope[brax]",
    "craftax": "jax-envelope[craftax]",
    "gymnax": (
        "gymnax @ git+https://github.com/RobertTLange/gymnax.git@"
        "18f2e7f3cffafc7042c76fdc538c83957418a9a9"
    ),
    "jumanji": "jax-envelope[jumanji]",
    "kinetix": (
        "kinetix-env @ git+https://github.com/FLAIROx/Kinetix.git@"
        "df4de60cabd42dbd1c35fb5214fdc6728710e33d"
    ),
    "mujoco_playground": "jax-envelope[mujoco-playground]",
    "navix": "jax-envelope[navix]",
}


class HasFromNameInit(Protocol):
    @classmethod
    def from_name(
        cls,
        env_name: str,
        env_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Self:
        """Creates an environment from a name and keyword arguments. Unless otherwise
        noted, the created environment will have its default parameters, with
        truncation and auto-reset disabled.

        Args:
            env_name: Environment name
            env_kwargs: Keyword arguments passed to the environment constructor
            **kwargs: Additional keyword arguments passed to the environment wrapper
        """
        ...


def _validate_max_steps(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        valid_values = (
            "'default', a positive integer, or None"
            if name == "max_episode_steps"
            else "a positive integer or None"
        )
        raise TypeError(f"{name} must be {valid_values}")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def create(
    env_name: str,
    env_kwargs: dict[str, Any] | None = None,
    *,
    max_episode_steps: Literal["default"] | int | None = "default",
    **kwargs: Any,
) -> Environment:
    """Create an environment from a prefixed environment ID.

    Args:
        env_name: Environment ID in the format "suite::env_name" (e.g., "brax::ant")
        env_kwargs: Keyword arguments passed to the suite's environment constructor
        max_episode_steps: Episode horizon. When omitted or set to ``"default"``, use
            the adapter's captured backend default; a positive integer overrides it;
            ``None`` disables truncation.
        **kwargs: Additional keyword arguments passed to the environment wrapper

    Returns:
        An instance of the wrapped environment. If the adapter has a `default_max_steps`
            property, it will be wrapped in a `TruncationWrapper` before returning.

    Examples:
        >>> env = create("jumanji::snake")
        >>> env = create("brax::ant", env_kwargs={"backend": "spring"})
        >>> env = create("gymnax::CartPole-v1", env_params=...)
    """
    original_env_id = env_name

    # Validate explicit values before importing optional dependencies. This argument is
    # reserved by the factory and must never leak into an adapter constructor.
    if isinstance(max_episode_steps, str) and max_episode_steps == "default":
        selected_max_steps: Literal["default"] | int | None = "default"
    else:
        selected_max_steps = _validate_max_steps(
            max_episode_steps, name="max_episode_steps"
        )

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
        install_spec = _INSTALL_SPECS.get(suite)
        install_hint = (
            f' Install with: pip install "{install_spec}".'
            if install_spec is not None
            else ""
        )
        raise ImportError(
            f"Failed to import {suite} wrapper. "
            f"Make sure you have installed the '{suite}' dependencies. "
            f"{install_hint}"
            f"Original error: {e}"
        ) from e

    # Adapters may need to inject or normalize constructor settings. Never let those
    # mutations escape through a caller-owned mapping.
    adapter_env_kwargs = None if env_kwargs is None else dict(env_kwargs)
    env = cast(
        Environment,
        env_class.from_name(env_name, env_kwargs=adapter_env_kwargs, **kwargs),
    )

    if selected_max_steps == "default":
        selected_max_steps = _validate_max_steps(
            getattr(env, "default_max_steps", None), name="default_max_steps"
        )

    if selected_max_steps is not None:
        from envelope.wrappers.truncation_wrapper import TruncationWrapper

        env = TruncationWrapper(env=env, max_steps=selected_max_steps)

    return env


__all__ = ["create"]
