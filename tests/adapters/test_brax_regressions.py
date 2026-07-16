"""Focused Brax adapter regressions using a fake backend environment."""

# ruff: noqa: E402

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.adapters

pytest.importorskip("brax")

import envelope.adapters.brax_envelope as brax_envelope_module
from envelope.adapters.brax_envelope import BraxEnvelope


def test_from_name_disables_default_backend_limit(monkeypatch):
    captured_kwargs = {}

    monkeypatch.setattr(
        brax_envelope_module,
        "brax_create",
        lambda _name, **kwargs: captured_kwargs.update(kwargs) or object(),
    )

    env = BraxEnvelope.from_name("fast")

    assert captured_kwargs == {"episode_length": None, "auto_reset": False}
    assert env.default_max_steps == 1000


def test_from_name_adds_defaults_without_overriding_explicit_backend_controls(
    monkeypatch,
):
    raw_env = object()
    captured_kwargs = {}

    def fake_brax_create(env_name, **kwargs):
        assert env_name == "fast"
        captured_kwargs.update(kwargs)
        return raw_env

    monkeypatch.setattr(brax_envelope_module, "brax_create", fake_brax_create)
    caller_kwargs = {
        "backend": "generalized",
        "episode_length": 17,
        "auto_reset": True,
        "batch_size": 2,
        "action_repeat": 2,
    }

    with pytest.warns(UserWarning, match="backend settings"):
        env = BraxEnvelope.from_name("fast", env_kwargs=caller_kwargs)

    assert env.brax_env is raw_env
    assert captured_kwargs == caller_kwargs
    assert env.default_max_steps == 17


def test_done_is_cast_to_boolean_in_init_and_step():
    class FakeBraxEnv:
        def reset(self, key):
            del key
            return SimpleNamespace(
                obs=jnp.asarray([0.0]),
                reward=jnp.asarray(0.0),
                done=jnp.asarray(0.0),
            )

        def step(self, state, action):
            del state, action
            return SimpleNamespace(
                obs=jnp.asarray([1.0]),
                reward=jnp.asarray(1.0),
                done=jnp.asarray(1.0),
            )

    env = BraxEnvelope(brax_env=FakeBraxEnv())
    state, initial = env.init(jax.random.key(0))
    _, stepped = env.step(state, jnp.asarray(0.0))

    assert initial.terminated.dtype == jnp.bool_
    assert stepped.terminated.dtype == jnp.bool_
    assert not bool(initial.terminated)
    assert bool(stepped.terminated)
