"""Unit tests for envelope.adapters.create() factory function.

These tests are dependency-free (no brax/gymnax/navix imports) and focus on:
- parsing/validation of the env id
- suite dispatch via the module map
- lazy importing via importlib.import_module
- argument forwarding and error wrapping
"""

from functools import cached_property
import importlib
import inspect
import types
from typing import Literal, get_type_hints

import jax
import jax.numpy as jnp
import pytest

import envelope.adapters as adapters
from envelope.adapters import create
from envelope.environment import Environment, InfoContainer
from envelope.spaces import Discrete
from envelope.struct import Container
from envelope.wrappers.truncation_wrapper import TruncationWrapper


class _FakeAdapter(Environment):
    """Dependency-free adapter used to exercise factory/wrapper semantics."""

    default_max_steps: int | None = None

    @cached_property
    def observation_space(self) -> Discrete:
        return Discrete(n=100)

    @cached_property
    def action_space(self) -> Discrete:
        return Discrete(n=2)

    def _info(self, state: jax.Array, *, reward: float) -> InfoContainer:
        return InfoContainer(
            obs=state,
            reward=reward,
            terminated=False,
            truncated=False,
        ).update(backend=Container().update(step=state))

    def init(self, key: jax.Array):
        del key
        state = jnp.asarray(0, dtype=jnp.int32)
        return state, self._info(state, reward=0.0)

    def step(self, state: jax.Array, action: jax.Array):
        del action
        state = state + 1
        return state, self._info(state, reward=1.0)


def test_create_public_max_episode_steps_signature():
    parameter = inspect.signature(create).parameters["max_episode_steps"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default == "default"
    assert get_type_hints(create)["max_episode_steps"] == (
        Literal["default"] | int | None
    )


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
    real_import_module = importlib.import_module

    def fake_import_module(name: str):
        if name == module_name:
            import_calls.append(name)
            return dummy_module
        return real_import_module(name)

    monkeypatch.setattr(adapters, "_env_module_map", {suite: (module_name, class_name)})
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
        adapters, "_env_module_map", {"dummy": ("dummy_mod", "DummyWrapper")}
    )

    with pytest.raises(ValueError) as excinfo:
        create(f"{invalid_suite}::env")

    msg = str(excinfo.value)
    assert f"Unknown environment suite: {invalid_suite}" in msg
    assert "Available suites:" in msg
    assert "dummy" in msg


def test_create_wraps_import_error_and_chains_cause(monkeypatch):
    monkeypatch.setattr(
        adapters, "_env_module_map", {"dummy": ("dummy_mod", "DummyWrapper")}
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


@pytest.mark.parametrize(
    ("suite", "module_name", "class_name", "env_name", "dependency", "install_spec"),
    [
        (
            "brax",
            "envelope.adapters.brax_envelope",
            "BraxEnvelope",
            "fast",
            "brax",
            "jax-envelope[brax]",
        ),
        (
            "craftax",
            "envelope.adapters.craftax_envelope",
            "CraftaxEnvelope",
            "Craftax-Symbolic-v1",
            "craftax",
            "jax-envelope[craftax]",
        ),
        (
            "gymnax",
            "envelope.adapters.gymnax_envelope",
            "GymnaxEnvelope",
            "CartPole-v1",
            "gymnax",
            "gymnax @ git+https://github.com/RobertTLange/gymnax.git@"
            "18f2e7f3cffafc7042c76fdc538c83957418a9a9",
        ),
        (
            "jumanji",
            "envelope.adapters.jumanji_envelope",
            "JumanjiEnvelope",
            "Snake-v1",
            "jumanji",
            "jax-envelope[jumanji]",
        ),
        (
            "kinetix",
            "envelope.adapters.kinetix_envelope",
            "KinetixEnvelope",
            "random",
            "kinetix",
            "kinetix-env @ git+https://github.com/FLAIROx/Kinetix.git@"
            "df4de60cabd42dbd1c35fb5214fdc6728710e33d",
        ),
        (
            "mujoco_playground",
            "envelope.adapters.mujoco_playground_envelope",
            "MujocoPlaygroundEnvelope",
            "CartpoleBalance",
            "mujoco_playground",
            "jax-envelope[mujoco-playground]",
        ),
        (
            "navix",
            "envelope.adapters.navix_envelope",
            "NavixEnvelope",
            "Navix-Empty-5x5-v0",
            "navix",
            "jax-envelope[navix]",
        ),
    ],
)
def test_create_import_error_includes_exact_install_command(
    monkeypatch,
    suite,
    module_name,
    class_name,
    env_name,
    dependency,
    install_spec,
):
    monkeypatch.setattr(
        adapters,
        "_env_module_map",
        {suite: (module_name, class_name)},
    )

    missing = ModuleNotFoundError(f"No module named '{dependency}'", name=dependency)

    def fake_import_module(name: str):
        assert name == module_name
        raise missing

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError) as excinfo:
        create(f"{suite}::{env_name}")

    msg = str(excinfo.value)
    assert "pip install" in msg
    assert install_spec in msg
    assert excinfo.value.__cause__ is missing


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
    assert from_name_calls[-1]["env_kwargs"] == empty
    assert from_name_calls[-1]["env_kwargs"] is not empty


def test_create_defensively_copies_caller_env_kwargs(monkeypatch):
    sentinel = object()
    received: list[dict[str, object]] = []

    class MutatingWrapper:
        @classmethod
        def from_name(cls, env_name: str, env_kwargs=None, **kwargs):
            del env_name, kwargs
            assert env_kwargs is not None
            received.append(env_kwargs)
            env_kwargs["adapter_internal"] = True
            return sentinel

    module = types.SimpleNamespace(MutatingWrapper=MutatingWrapper)
    monkeypatch.setattr(
        adapters,
        "_env_module_map",
        {"dummy": ("dummy_mod", "MutatingWrapper")},
    )
    monkeypatch.setattr(importlib, "import_module", lambda _: module)

    env_kwargs: dict[str, object] = {"user_value": 3}
    out = create("dummy::Env", env_kwargs=env_kwargs)

    assert out is sentinel
    assert env_kwargs == {"user_value": 3}
    assert received[0] is not env_kwargs
    assert received[0] == {"user_value": 3, "adapter_internal": True}


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
        adapters,
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


def test_create_wraps_with_truncation_when_default_max_steps_set(monkeypatch):
    sentinel = types.SimpleNamespace(default_max_steps=500)
    _install_dummy_suite(monkeypatch, return_value=sentinel)

    env = create("dummy::Env")

    assert isinstance(env, TruncationWrapper)
    assert env.max_steps == 500


def test_create_omitted_max_episode_steps_uses_captured_adapter_horizon(monkeypatch):
    adapter = _FakeAdapter(default_max_steps=2)
    _install_dummy_suite(monkeypatch, return_value=adapter)

    env = create("dummy::Env")

    assert isinstance(env, TruncationWrapper)
    assert env.max_steps == 2

    state, init_info = env.init(jax.random.key(0))
    state, first_info = env.step(state, jnp.asarray(0))
    _state, second_info = env.step(state, jnp.asarray(0))

    assert not bool(first_info.truncated)
    assert bool(second_info.truncated)
    assert int(init_info.backend.step) == 0
    assert int(first_info.backend.step) == 1
    assert int(second_info.backend.step) == 2
    assert jax.tree.structure(init_info.backend) == jax.tree.structure(
        first_info.backend
    )
    assert jax.tree.structure(first_info.backend) == jax.tree.structure(
        second_info.backend
    )


def test_create_explicit_default_uses_captured_adapter_horizon(monkeypatch):
    adapter = _FakeAdapter(default_max_steps=9)
    _import_calls, from_name_calls = _install_dummy_suite(
        monkeypatch, return_value=adapter
    )

    env = create("dummy::Env", max_episode_steps="default")

    assert isinstance(env, TruncationWrapper)
    assert env.max_steps == 9
    assert from_name_calls == [{"env_name": "Env", "env_kwargs": None, "kwargs": {}}]


@pytest.mark.parametrize("max_episode_steps", [1, 7, 100])
def test_create_explicit_max_episode_steps_overrides_captured_horizon(
    monkeypatch, max_episode_steps
):
    adapter = _FakeAdapter(default_max_steps=500)
    _import_calls, from_name_calls = _install_dummy_suite(
        monkeypatch, return_value=adapter
    )

    env = create("dummy::Env", max_episode_steps=max_episode_steps)

    assert isinstance(env, TruncationWrapper)
    assert env.max_steps == max_episode_steps
    assert from_name_calls == [{"env_name": "Env", "env_kwargs": None, "kwargs": {}}]


def test_create_explicit_max_episode_steps_works_without_adapter_default(monkeypatch):
    adapter = _FakeAdapter(default_max_steps=None)
    _install_dummy_suite(monkeypatch, return_value=adapter)

    env = create("dummy::Env", max_episode_steps=3)

    assert isinstance(env, TruncationWrapper)
    assert env.max_steps == 3


def test_create_explicit_none_disables_captured_horizon(monkeypatch):
    adapter = _FakeAdapter(default_max_steps=500)
    _import_calls, from_name_calls = _install_dummy_suite(
        monkeypatch, return_value=adapter
    )

    env = create("dummy::Env", max_episode_steps=None)

    assert env is adapter
    assert from_name_calls == [{"env_name": "Env", "env_kwargs": None, "kwargs": {}}]


@pytest.mark.parametrize("max_episode_steps", [True, False, 1.5, "5", object()])
def test_create_rejects_non_integer_max_episode_steps(monkeypatch, max_episode_steps):
    import_calls, _from_name_calls = _install_dummy_suite(
        monkeypatch, return_value=_FakeAdapter(default_max_steps=500)
    )

    with pytest.raises(TypeError, match="max_episode_steps.*positive integer.*None"):
        create("dummy::Env", max_episode_steps=max_episode_steps)
    assert import_calls == []


@pytest.mark.parametrize("max_episode_steps", [0, -1, -100])
def test_create_rejects_non_positive_max_episode_steps(monkeypatch, max_episode_steps):
    import_calls, _from_name_calls = _install_dummy_suite(
        monkeypatch, return_value=_FakeAdapter(default_max_steps=500)
    )

    with pytest.raises(ValueError, match="max_episode_steps.*positive"):
        create("dummy::Env", max_episode_steps=max_episode_steps)
    assert import_calls == []


@pytest.mark.parametrize("captured_horizon", [True, False, 1.5, "5", object()])
def test_create_rejects_non_integer_captured_adapter_horizon(
    monkeypatch, captured_horizon
):
    _install_dummy_suite(
        monkeypatch, return_value=_FakeAdapter(default_max_steps=captured_horizon)
    )

    with pytest.raises(TypeError, match="default_max_steps.*positive integer.*None"):
        create("dummy::Env")


@pytest.mark.parametrize("captured_horizon", [0, -1, -100])
def test_create_rejects_non_positive_captured_adapter_horizon(
    monkeypatch, captured_horizon
):
    _install_dummy_suite(
        monkeypatch, return_value=_FakeAdapter(default_max_steps=captured_horizon)
    )

    with pytest.raises(ValueError, match="default_max_steps.*positive"):
        create("dummy::Env")


@pytest.mark.parametrize("max_episode_steps", [None, 3])
def test_explicit_max_episode_steps_takes_precedence_over_invalid_adapter_default(
    monkeypatch, max_episode_steps
):
    adapter = _FakeAdapter(default_max_steps=0)
    _install_dummy_suite(monkeypatch, return_value=adapter)

    env = create("dummy::Env", max_episode_steps=max_episode_steps)

    if max_episode_steps is None:
        assert env is adapter
    else:
        assert isinstance(env, TruncationWrapper)
        assert env.max_steps == max_episode_steps
