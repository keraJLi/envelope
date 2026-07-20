"""Compatibility wrappers for various RL environment libraries."""

from typing import Any, Literal, Protocol

from envelope.environment import Environment
from envelope.registry import _load_factory


class HasFromNameInit(Protocol):
    @classmethod
    def from_name(
        cls,
        env_name: str,
        env_kwargs: dict[str, Any] | None = None,
        **kwargs: dict[str, Any],
    ) -> Environment:
        """Creates an environment from a name and keyword arguments. Unless otherwise
        noted, the created environment will have its default parameters, with
        truncation and auto-reset disabled.

        Args:
            env_name: Environment name
            env_kwargs: Keyword arguments passed to the environment constructor
            **kwargs: Additional keyword arguments passed to the environment wrapper
        """


def create(
    env_name: str,
    env_kwargs: dict[str, Any] | None = None,
    *,
    max_episode_steps: Literal["default"] | int | None = "default",
    **kwargs: dict[str, Any],
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
    if "::" not in env_name:
        raise ValueError(
            f"Environment ID must be in format 'suite::env_name', got: {original_env_id}"
        )

    suite, env_name = env_name.split("::", 1)
    if not suite or not env_name:
        raise ValueError(
            f"Environment ID must be in format 'suite::env_name', got: {original_env_id}"
        )

    factory = _load_factory(suite)
    env = factory(env_name, env_kwargs=env_kwargs, **kwargs)
    if not isinstance(env, Environment):
        raise TypeError(
            f"Environment provider for suite '{suite}' returned "
            f"{type(env).__name__}, not an Environment"
        )

    if max_episode_steps == "default":
        max_episode_steps = env.default_max_steps

    if max_episode_steps is not None:
        from envelope.wrappers.truncation_wrapper import TruncationWrapper

        env = TruncationWrapper(env=env, max_steps=max_episode_steps)

    return env


__all__ = ["create"]
