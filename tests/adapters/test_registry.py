"""Tests for installed environment-suite discovery."""

from __future__ import annotations

import importlib
import importlib.metadata
import sys
import warnings
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar

import pytest

import envelope
import envelope.registry as registry
from envelope.adapters import create
from envelope.environment import Environment
from envelope.spaces import Discrete

_PROVIDER_CALLS: list[tuple[str, dict[str, Any] | None, dict[str, Any]]] = []


class _FakeEnvironment(Environment):
    @cached_property
    def observation_space(self) -> Discrete:
        return Discrete(n=1)

    @cached_property
    def action_space(self) -> Discrete:
        return Discrete(n=1)

    def init(self, key):
        raise NotImplementedError

    def step(self, state, action):
        raise NotImplementedError


class _ClassmethodProvider(_FakeEnvironment):
    @classmethod
    def from_name(cls, env_name, env_kwargs=None, **kwargs):
        _PROVIDER_CALLS.append((env_name, env_kwargs, kwargs))
        return cls()

    @classmethod
    def registered_names(cls):
        return (name for name in ("zeta", "alpha", "zeta", "nested::child"))


class _NoCatalogueProvider(_FakeEnvironment):
    @classmethod
    def from_name(cls, env_name, env_kwargs=None, **kwargs):
        return cls()


class _ForeignProvider:
    @classmethod
    def from_name(cls, env_name, env_kwargs=None, **kwargs):
        return _FakeEnvironment()


class _NonCallableCatalogue(_FakeEnvironment):
    registered_names: ClassVar[tuple[str, ...]] = ("one",)

    @classmethod
    def from_name(cls, env_name, env_kwargs=None, **kwargs):
        return cls()


class _InvalidCatalogue(_FakeEnvironment):
    @classmethod
    def from_name(cls, env_name, env_kwargs=None, **kwargs):
        return cls()

    @classmethod
    def registered_names(cls):
        return ("valid", "", 3)


class _RaisingCatalogue(_FakeEnvironment):
    @classmethod
    def from_name(cls, env_name, env_kwargs=None, **kwargs):
        return cls()

    @classmethod
    def registered_names(cls):
        raise RuntimeError("catalogue exploded")


class _StringCatalogue(_FakeEnvironment):
    @classmethod
    def from_name(cls, env_name, env_kwargs=None, **kwargs):
        return cls()

    @classmethod
    def registered_names(cls):
        return "not-a-catalogue"


class _LiveCatalogue(_FakeEnvironment):
    names: ClassVar[list[str]] = []

    @classmethod
    def from_name(cls, env_name, env_kwargs=None, **kwargs):
        return cls()

    @classmethod
    def registered_names(cls):
        return cls.names


_NOT_CALLABLE = object()
_FACTORY_ERROR = RuntimeError("factory exploded")


def _plain_factory(env_name, env_kwargs=None, **kwargs):
    _PROVIDER_CALLS.append((env_name, env_kwargs, kwargs))
    return _FakeEnvironment()


def _bad_return_factory(env_name, env_kwargs=None, **kwargs):
    return object()


def _raising_factory(env_name, env_kwargs=None, **kwargs):
    raise _FACTORY_ERROR


def _entry_point(name: str, target: str) -> importlib.metadata.EntryPoint:
    return importlib.metadata.EntryPoint(
        name=name,
        value=f"{__name__}:{target}",
        group="envelope.environments",
    )


def _set_providers(monkeypatch, *entry_points, builtins=None) -> None:
    monkeypatch.setattr(registry, "_entry_points", lambda: tuple(entry_points))
    if builtins is not None:
        monkeypatch.setattr(registry, "_builtin_factories", builtins)


def test_registered_suites_is_sorted_and_does_not_load_targets(monkeypatch):
    missing_module = "envelope_registry_metadata_only_target"
    entry_points = (
        importlib.metadata.EntryPoint(
            name="zeta",
            value=f"{missing_module}:factory",
            group="envelope.environments",
        ),
        importlib.metadata.EntryPoint(
            name="alpha",
            value=f"{missing_module}:other_factory",
            group="envelope.environments",
        ),
        importlib.metadata.EntryPoint(
            name="alpha",
            value=f"{missing_module}:third_factory",
            group="envelope.environments",
        ),
    )
    _set_providers(monkeypatch, *entry_points, builtins={"middle": "unused"})

    assert registry.registered_suites() == ("alpha", "middle", "zeta")
    assert missing_module not in sys.modules


def test_real_dist_info_is_discovered_before_its_module_is_imported(
    tmp_path: Path, monkeypatch
):
    suite = "temporary_suite"
    module_name = "temporary_envelope_provider"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        f"""
from {__name__} import _FakeEnvironment

calls = []

def create(env_name, env_kwargs=None, **kwargs):
    calls.append((env_name, env_kwargs, kwargs))
    return _FakeEnvironment()
""".lstrip()
    )
    dist_info = tmp_path / "temporary_envelope_provider-1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: temporary-envelope-provider\nVersion: 1.0\n"
    )
    (dist_info / "entry_points.txt").write_text(
        f"[envelope.environments]\n{suite} = {module_name}:create\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    assert module_name not in sys.modules
    suites = envelope.registered_suites()
    assert suites == tuple(sorted(suites))
    assert suite in suites
    assert module_name not in sys.modules

    env_kwargs = {"difficulty": 2}
    try:
        env = envelope.create(
            f"{suite}::walk",
            env_kwargs=env_kwargs,
            max_episode_steps=None,
            render=False,
        )
        provider = sys.modules[module_name]
        assert isinstance(env, _FakeEnvironment)
        assert provider.calls == [("walk", env_kwargs, {"render": False})]
        assert provider.calls[0][1] is env_kwargs
    finally:
        sys.modules.pop(module_name, None)


def test_classmethod_provider_creates_and_lists_environments(monkeypatch):
    _PROVIDER_CALLS.clear()
    _set_providers(
        monkeypatch, _entry_point("plugin", "_ClassmethodProvider.from_name")
    )

    env_kwargs = {"difficulty": 3}
    env = create(
        "plugin::walk", env_kwargs=env_kwargs, max_episode_steps=None, render=True
    )

    assert isinstance(env, _ClassmethodProvider)
    assert _PROVIDER_CALLS == [("walk", env_kwargs, {"render": True})]
    assert registry.registered_environments("plugin") == (
        "plugin::alpha",
        "plugin::nested::child",
        "plugin::zeta",
    )


def test_plain_function_provider_creates_but_does_not_list(monkeypatch):
    _PROVIDER_CALLS.clear()
    _set_providers(monkeypatch, _entry_point("function", "_plain_factory"))

    env = create("function::walk", max_episode_steps=None)

    assert isinstance(env, _FakeEnvironment)
    assert _PROVIDER_CALLS == [("walk", None, {})]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert registry.registered_environments("function") == ()


def test_builtin_provider_takes_precedence_over_same_named_entry_point(monkeypatch):
    _PROVIDER_CALLS.clear()
    suite = "claimed"
    _set_providers(
        monkeypatch,
        importlib.metadata.EntryPoint(
            name=suite,
            value="provider_that_must_not_be_loaded:factory",
            group="envelope.environments",
        ),
        builtins={suite: f"{__name__}:_ClassmethodProvider.from_name"},
    )

    env = create(f"{suite}::walk", max_episode_steps=None)

    assert isinstance(env, _ClassmethodProvider)
    assert registry.registered_suites() == (suite,)
    assert registry.registered_environments(suite)[0] == f"{suite}::alpha"


def test_duplicate_external_providers_are_listed_once_but_cannot_create(
    monkeypatch,
):
    suite = "duplicate"
    _set_providers(
        monkeypatch,
        _entry_point(suite, "_plain_factory"),
        _entry_point(suite, "_bad_return_factory"),
        builtins={},
    )

    assert registry.registered_suites() == (suite,)
    with pytest.raises(ValueError, match=suite) as excinfo:
        create(f"{suite}::walk", max_episode_steps=None)
    assert "_plain_factory" in str(excinfo.value)
    assert "_bad_return_factory" in str(excinfo.value)

    with pytest.warns(RuntimeWarning, match=suite):
        assert registry.registered_environments(suite) == ()


@pytest.mark.parametrize(
    "target",
    ["_FakeEnvironment", "_NOT_CALLABLE", "_ForeignProvider.from_name"],
)
def test_create_rejects_invalid_provider_targets(monkeypatch, target):
    suite = "invalid_target"
    _set_providers(monkeypatch, _entry_point(suite, target), builtins={})

    with pytest.raises(TypeError, match=suite):
        create(f"{suite}::walk", max_episode_steps=None)


def test_create_validates_provider_result(monkeypatch):
    suite = "invalid_result"
    _set_providers(monkeypatch, _entry_point(suite, "_bad_return_factory"))

    with pytest.raises(TypeError, match=suite):
        create(f"{suite}::walk", max_episode_steps=None)


def test_provider_factory_exception_propagates_unchanged(monkeypatch):
    suite = "raising_factory"
    _set_providers(monkeypatch, _entry_point(suite, "_raising_factory"))

    with pytest.raises(RuntimeError) as excinfo:
        create(f"{suite}::walk", max_episode_steps=None)

    assert excinfo.value is _FACTORY_ERROR


def test_unknown_suite_is_an_error_for_creation_and_listing(monkeypatch):
    _set_providers(monkeypatch, builtins={})

    with pytest.raises(ValueError, match="missing"):
        create("missing::walk", max_episode_steps=None)
    with pytest.raises(ValueError, match="missing"):
        registry.registered_environments("missing")


@pytest.mark.parametrize(
    "target",
    [
        "_NonCallableCatalogue.from_name",
        "_InvalidCatalogue.from_name",
        "_RaisingCatalogue.from_name",
        "_StringCatalogue.from_name",
    ],
)
def test_malformed_external_catalogue_warns_and_returns_empty(monkeypatch, target):
    suite = "bad_catalogue"
    _set_providers(monkeypatch, _entry_point(suite, target), builtins={})

    with pytest.warns(RuntimeWarning, match=suite):
        assert registry.registered_environments(suite) == ()


def test_environment_catalogue_is_read_again_on_each_call(monkeypatch):
    suite = "live_catalogue"
    monkeypatch.setattr(_LiveCatalogue, "names", ["first"])
    _set_providers(
        monkeypatch,
        _entry_point(suite, "_LiveCatalogue.from_name"),
        builtins={},
    )

    assert registry.registered_environments(suite) == (f"{suite}::first",)

    _LiveCatalogue.names[:] = ["second"]
    assert registry.registered_environments(suite) == (f"{suite}::second",)


def test_external_load_failure_warns_and_returns_empty(monkeypatch):
    suite = "broken_import"
    entry_point = importlib.metadata.EntryPoint(
        name=suite,
        value="missing_envelope_provider_module:factory",
        group="envelope.environments",
    )
    _set_providers(monkeypatch, entry_point, builtins={})

    with pytest.warns(RuntimeWarning, match=suite):
        assert registry.registered_environments(suite) == ()


def test_unavailable_builtin_listing_is_silent(monkeypatch):
    suite = "unavailable_builtin"
    _set_providers(
        monkeypatch,
        builtins={suite: "missing_builtin_adapter:Adapter.from_name"},
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert registry.registered_environments(suite) == ()


def test_bound_provider_without_catalogue_returns_empty_without_warning(monkeypatch):
    suite = "no_catalogue"
    _set_providers(
        monkeypatch,
        _entry_point(suite, "_NoCatalogueProvider.from_name"),
        builtins={},
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert registry.registered_environments(suite) == ()
