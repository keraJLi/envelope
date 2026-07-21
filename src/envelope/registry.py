"""Discovery of installed environment suites."""

import importlib.metadata
import warnings
from collections.abc import Callable

from envelope.environment import Environment

_ENTRY_POINT_GROUP = "envelope.environments"

_builtin_factories = {
    "gymnax": "envelope.adapters.gymnax_envelope:GymnaxEnvelope.from_name",
    "brax": "envelope.adapters.brax_envelope:BraxEnvelope.from_name",
    "navix": "envelope.adapters.navix_envelope:NavixEnvelope.from_name",
    "jumanji": "envelope.adapters.jumanji_envelope:JumanjiEnvelope.from_name",
    "kinetix": "envelope.adapters.kinetix_envelope:KinetixEnvelope.from_name",
    "craftax": "envelope.adapters.craftax_envelope:CraftaxEnvelope.from_name",
    "mujoco_playground": (
        "envelope.adapters.mujoco_playground_envelope:"
        "MujocoPlaygroundEnvelope.from_name"
    ),
}


def _entry_points() -> tuple[importlib.metadata.EntryPoint, ...]:
    return tuple(importlib.metadata.entry_points(group=_ENTRY_POINT_GROUP))


def registered_suites() -> tuple[str, ...]:
    """Return the names of built-in and installed environment suites."""
    suites = _builtin_factories.keys() | {entry.name for entry in _entry_points()}
    return tuple(sorted(suites))


def _entry_point(suite: str) -> importlib.metadata.EntryPoint:
    if suite in _builtin_factories:
        return importlib.metadata.EntryPoint(
            name=suite,
            value=_builtin_factories[suite],
            group=_ENTRY_POINT_GROUP,
        )

    matches = [entry for entry in _entry_points() if entry.name == suite]
    if not matches:
        raise ValueError(
            f"Unknown environment suite: {suite}. "
            f"Available suites: {list(registered_suites())}"
        )
    if len(matches) > 1:
        providers = ", ".join(sorted(entry.value for entry in matches))
        raise ValueError(
            f"Multiple environment providers are registered for suite "
            f"'{suite}': {providers}"
        )
    return matches[0]


def _load_factory(suite: str) -> Callable[..., Environment]:
    entry = _entry_point(suite)
    try:
        factory = entry.load()
    except Exception as error:
        raise ImportError(
            f"Failed to load environment provider for suite '{suite}': {error}"
        ) from error

    if isinstance(factory, type) or not callable(factory):
        raise TypeError(
            f"Environment provider for suite '{suite}' must be a factory callable, "
            f"got {factory!r}"
        )

    owner = getattr(factory, "__self__", None)
    if isinstance(owner, type) and not issubclass(owner, Environment):
        raise TypeError(
            f"Environment provider for suite '{suite}' is bound to "
            f"{owner.__name__}, which is not an Environment subclass"
        )

    return factory


def registered_environments(suite: str) -> tuple[str, ...]:
    """Return the environment IDs advertised by an installed suite."""
    if suite not in registered_suites():
        _entry_point(suite)

    try:
        factory = _load_factory(suite)
        owner = getattr(factory, "__self__", None)
        if not isinstance(owner, type):
            return ()

        registered_names = getattr(owner, "registered_names", None)
        if registered_names is None:
            return ()
        if not callable(registered_names):
            raise TypeError("registered_names must be callable")

        names = registered_names()
        if isinstance(names, str):
            raise TypeError("registered_names must return an iterable of strings")
        names = tuple(names)
        if any(not isinstance(name, str) or not name for name in names):
            raise TypeError("registered_names must return non-empty strings")
    except Exception as error:
        if (
            suite in _builtin_factories
            and isinstance(error, ImportError)
            and isinstance(error.__cause__, ImportError)
        ):
            return ()
        warnings.warn(
            f"Could not list environments for suite '{suite}': {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        return ()

    return tuple(f"{suite}::{name}" for name in sorted(set(names)))


__all__ = ["registered_environments", "registered_suites"]
