"""Focused Gymnax adapter regressions using a fake backend environment.

The Gymnax package is still an optional dependency, but these tests avoid creating or
compiling a real environment.  They belong to the isolated adapters test job.
"""

# ruff: noqa: E402

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.adapters

pytest.importorskip("gymnax")

import envelope.adapters.gymnax_envelope as gymnax_envelope_module
from envelope.adapters.gymnax_envelope import GymnaxEnvelope
from envelope.struct import Container


class _Params:
    def __init__(self, max_steps_in_episode):
        self.max_steps_in_episode = max_steps_in_episode

    def replace(self, **updates):
        return _Params(updates.get("max_steps_in_episode", self.max_steps_in_episode))


class _ActionSpace:
    def sample(self, key):
        del key
        return jnp.asarray(0, dtype=jnp.int32)


class _FakeGymnaxEnv:
    def __init__(self, default_horizon: int = 500):
        self.default_params = _Params(default_horizon)

    def reset(self, key, params):
        del key, params
        return jnp.asarray([0.0]), jnp.asarray(0, dtype=jnp.int32)

    def step(self, key, state, action, params):
        del key, action, params
        next_state = state + 1
        obs = jnp.asarray([next_state], dtype=jnp.float32)
        backend_info = Container().update(
            score=jnp.asarray(next_state, dtype=jnp.float32)
        )
        return obs, next_state, jnp.asarray(1.0), jnp.asarray(False), backend_info

    def action_space(self, params):
        del params
        return _ActionSpace()


def test_supplied_horizon_is_captured_then_removed_from_backend(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_env = _FakeGymnaxEnv(default_horizon=500)
    constructor_kwargs = []

    def fake_create(env_name, **kwargs):
        assert env_name == "Fake-v0"
        constructor_kwargs.append(kwargs)
        return fake_env, fake_env.default_params

    monkeypatch.setattr(gymnax_envelope_module, "gymnax_create", fake_create)
    supplied_params = _Params(max_steps_in_episode=17)
    caller_kwargs = {"difficulty": "hard"}

    env = GymnaxEnvelope.from_name(
        "Fake-v0", env_params=supplied_params, env_kwargs=caller_kwargs
    )

    assert caller_kwargs == {"difficulty": "hard"}
    assert constructor_kwargs == [{"difficulty": "hard"}]
    assert supplied_params.max_steps_in_episode == 17
    assert jnp.isposinf(jnp.asarray(env.env_params.max_steps_in_episode))
    assert env.default_max_steps == 17


def test_backend_info_namespace_is_stable_between_init_and_step(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_env = _FakeGymnaxEnv(default_horizon=23)
    monkeypatch.setattr(
        gymnax_envelope_module,
        "gymnax_create",
        lambda env_name, **kwargs: (fake_env, fake_env.default_params),
    )
    env = GymnaxEnvelope.from_name("Fake-v0")

    state, init_info = env.init(jax.random.key(0))
    _next_state, step_info = env.step(state, jnp.asarray(0, dtype=jnp.int32))

    assert hasattr(init_info, "backend")
    assert hasattr(step_info, "backend")
    assert init_info.backend.score.shape == step_info.backend.score.shape
    assert init_info.backend.score.dtype == step_info.backend.score.dtype
    assert jax.tree.structure(init_info.backend) == jax.tree.structure(
        step_info.backend
    )
    assert [x.shape for x in jax.tree.leaves(init_info.backend)] == [
        x.shape for x in jax.tree.leaves(step_info.backend)
    ]
    assert [x.dtype for x in jax.tree.leaves(init_info.backend)] == [
        x.dtype for x in jax.tree.leaves(step_info.backend)
    ]
