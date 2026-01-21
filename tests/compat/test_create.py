"""Unit tests for envelope.compat.create() factory function.

These tests are dependency-free (no brax/gymnax/navix imports) and focus on:
- parsing/validation of the env id
- suite dispatch via the module map
- lazy importing via importlib.import_module
- argument forwarding and error wrapping
"""

import importlib
import types

import pytest

import envelope.compat as compat
from envelope.compat import create


def _install_dummy_suite(
    monkeypatch: pytest.MonkeyPatch,
    *,
    suite: str = "dummy",
    module_name: str = "dummy_mod",
    class_name: str = "DummyWrapper",
    return_value: object | None = None,
):
    """Patch the module map and import mechanism to a dummy wrapper."""
    import_calls: list[str] = []
    from_name_calls: list[dict[str, object]] = []

    class DummyWrapper:
        @classmethod
        def from_name(cls, env_name: str, env_kwargs=None, **kwargs):
            from_name_calls.append(
                {"env_name": env_name, "env_kwargs": env_kwargs, "kwargs": kwargs}
            )
            return return_value

    dummy_module = types.SimpleNamespace(**{class_name: DummyWrapper})

    def fake_import_module(name: str):
        import_calls.append(name)
        if name != module_name:
            raise AssertionError(f"Unexpected import: {name}")
        return dummy_module

    monkeypatch.setattr(compat, "_env_module_map", {suite: (module_name, class_name)})
    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    return import_calls, from_name_calls


def test_create_rejects_missing_separator():
    with pytest.raises(ValueError) as excinfo:
        create("brax-ant")
    assert "suite::env_name" in str(excinfo.value)
    assert "brax-ant" in str(excinfo.value)


def test_create_rejects_empty_string():
    with pytest.raises(ValueError) as excinfo:
        create("")
    assert "suite::env_name" in str(excinfo.value)


@pytest.mark.parametrize("env_id", ["::ant", "brax::"])
def test_create_rejects_empty_suite_or_env_name(env_id: str):
    with pytest.raises(ValueError) as excinfo:
        create(env_id)
    assert "suite::env_name" in str(excinfo.value)
    assert env_id in str(excinfo.value)


@pytest.mark.parametrize("invalid_suite", ["unknown", "barx", "invalid"])
def test_create_unknown_suite_mentions_available_suites(
    invalid_suite: str, monkeypatch
):
    # Keep the map deterministic so we can assert it appears in the message.
    monkeypatch.setattr(
        compat, "_env_module_map", {"dummy": ("dummy_mod", "DummyWrapper")}
    )

    with pytest.raises(ValueError) as excinfo:
        create(f"{invalid_suite}::env")

    msg = str(excinfo.value)
    assert f"Unknown environment suite: {invalid_suite}" in msg
    assert "Available suites:" in msg
    assert "dummy" in msg


def test_create_wraps_import_error_and_chains_cause(monkeypatch):
    monkeypatch.setattr(
        compat, "_env_module_map", {"dummy": ("dummy_mod", "DummyWrapper")}
    )

    def fake_import_module(name: str):
        raise ImportError("boom")

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError) as excinfo:
        create("dummy::Env")

    msg = str(excinfo.value)
    assert "Failed to import dummy wrapper" in msg
    assert "Make sure you have installed the 'dummy' dependencies" in msg
    assert excinfo.value.__cause__ is not None
    assert isinstance(excinfo.value.__cause__, ImportError)


def test_create_forwards_env_name_env_kwargs_and_kwargs(monkeypatch):
    sentinel = object()
    import_calls, from_name_calls = _install_dummy_suite(
        monkeypatch, return_value=sentinel
    )

    env_kwargs = {"a": 1}
    out = create("dummy::MyEnv", env_kwargs=env_kwargs, foo=2)

    assert out is sentinel
    assert import_calls == ["dummy_mod"]
    assert from_name_calls == [
        {"env_name": "MyEnv", "env_kwargs": env_kwargs, "kwargs": {"foo": 2}}
    ]


def test_create_preserves_env_kwargs_none_vs_empty_dict(monkeypatch):
    _import_calls, from_name_calls = _install_dummy_suite(
        monkeypatch, return_value=None
    )

    create("dummy::A")
    assert from_name_calls[-1]["env_kwargs"] is None

    empty: dict[str, object] = {}
    create("dummy::B", env_kwargs=empty)
    assert from_name_calls[-1]["env_kwargs"] is empty


def test_create_splits_only_on_first_separator(monkeypatch):
    _import_calls, from_name_calls = _install_dummy_suite(
        monkeypatch, return_value=None
    )

    create("dummy::::ant")
    assert from_name_calls == [{"env_name": "::ant", "env_kwargs": None, "kwargs": {}}]


def test_create_imports_only_the_requested_suite(monkeypatch):
    import_calls: list[str] = []

    class WrapperA:
        @classmethod
        def from_name(cls, env_name: str, env_kwargs=None, **kwargs):
            return "A"

    class WrapperB:
        @classmethod
        def from_name(cls, env_name: str, env_kwargs=None, **kwargs):
            return "B"

    module_a = types.SimpleNamespace(WrapperA=WrapperA)
    module_b = types.SimpleNamespace(WrapperB=WrapperB)

    monkeypatch.setattr(
        compat,
        "_env_module_map",
        {"a": ("a_mod", "WrapperA"), "b": ("b_mod", "WrapperB")},
    )

    def fake_import_module(name: str):
        import_calls.append(name)
        if name == "a_mod":
            return module_a
        if name == "b_mod":
            return module_b
        raise AssertionError(f"Unexpected import: {name}")

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    assert create("a::Env") == "A"
    assert import_calls == ["a_mod"]
