"""Focused Brax adapter regressions using a fake backend environment."""

# ruff: noqa: E402

import pytest

pytestmark = pytest.mark.adapters

pytest.importorskip("brax")

import envelope.adapters.brax_envelope as brax_envelope_module
from envelope.adapters.brax_envelope import BraxEnvelope


def test_from_name_copies_kwargs_and_avoids_brax_episode_wrapper(monkeypatch):
    raw_env = object()
    captured_kwargs = {}

    def fake_brax_create(env_name, **kwargs):
        assert env_name == "fast"
        captured_kwargs.update(kwargs)
        return raw_env

    monkeypatch.setattr(brax_envelope_module, "brax_create", fake_brax_create)
    caller_kwargs = {"backend": "generalized"}

    env = BraxEnvelope.from_name("fast", env_kwargs=caller_kwargs)

    assert caller_kwargs == {"backend": "generalized"}
    assert env.brax_env is raw_env
    assert captured_kwargs == {
        "backend": "generalized",
        "episode_length": None,
        "auto_reset": False,
    }


@pytest.mark.parametrize(
    ("env_kwargs", "reserved_name"),
    [
        ({"batch_size": 2}, "batch_size"),
        ({"action_repeat": 2}, "action_repeat"),
        ({"action_repeat": True}, "action_repeat"),
    ],
)
def test_from_name_rejects_backend_vectorization_and_action_repeat(
    monkeypatch, env_kwargs, reserved_name
):
    def unexpected_create(*args, **kwargs):
        raise AssertionError("invalid backend controls must be rejected before create")

    monkeypatch.setattr(brax_envelope_module, "brax_create", unexpected_create)

    with pytest.raises(ValueError, match=reserved_name):
        BraxEnvelope.from_name("fast", env_kwargs=env_kwargs)
