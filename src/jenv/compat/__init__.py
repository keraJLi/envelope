"""Compatibility wrappers for various RL environment libraries."""

# Lazy imports to avoid requiring all dependencies at once
_env_module_map = {
    "brax": ("jenv.compat.brax_jenv", "BraxJenv"),
    "navix": ("jenv.compat.navix_jenv", "NavixJenv"),
    "jumanji": ("jenv.compat.jumanji_jenv", "JumanjiJenv"),
    "kinetix": ("jenv.compat.kinetix_jenv", "KinetixJenv"),
    "craftax": ("jenv.compat.craftax_jenv", "CraftaxJenv"),
}


def create(env_id: str, **kwargs):
    """Create an environment from a prefixed environment ID.

    Args:
        env_id: Environment ID in the format "prefix/env_name" (e.g., "brax/ant")
        **kwargs: Additional keyword arguments passed to the environment wrapper

    Returns:
        An instance of the wrapped environment

    Examples:
        >>> env = create("brax/ant")
        >>> env = create("jumanji/snake")
    """
    if "/" not in env_id:
        raise ValueError(
            f"Environment ID must be in format 'prefix/env_name', got: {env_id}"
        )

    prefix, env_name = env_id.split("/", 1)

    if prefix not in _env_module_map:
        raise ValueError(
            f"Unknown environment prefix: {prefix}. "
            f"Available prefixes: {list(_env_module_map.keys())}"
        )

    # Lazy import the wrapper class
    module_name, class_name = _env_module_map[prefix]
    try:
        import importlib

        module = importlib.import_module(module_name)
        env_class = getattr(module, class_name)
    except ImportError as e:
        raise ImportError(
            f"Failed to import {prefix} wrapper. "
            f"Make sure you have installed the '{prefix}' dependencies. "
            f"Original error: {e}"
        ) from e

    return env_class(env_name=env_name, **kwargs)


__all__ = ["create"]
