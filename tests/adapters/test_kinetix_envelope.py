"""Tests for envelope.adapters.kinetix_envelope module."""

# ruff: noqa: E402

from __future__ import annotations

import pathlib
import warnings

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.adapters

pytest.importorskip("kinetix")

import kinetix
from kinetix.environment import EnvParams, StaticEnvParams
from kinetix.environment import spaces as kinetix_spaces

from envelope.adapters.kinetix_envelope import (
    KinetixEnvelope,
    _convert_space,
    _normalize_level_id,
)
from envelope.environment import Info
from envelope.spaces import Continuous, Discrete
from envelope.struct import Container
from tests.contract import (
    assert_jitted_rollout_contract,
    assert_obs_matches_space,
    assert_reset_step_contract,
)


@pytest.fixture(scope="module")
def kinetix_random_env():
    """Create a single random Kinetix env for the whole module."""
    return KinetixEnvelope.create_random()


@pytest.fixture(scope="module", autouse=True)
def kinetix_random_env_warmup(kinetix_random_env, prng_key):
    """Warm up reset/step/scan once to amortize compilation cost."""
    env = kinetix_random_env
    key_reset, key_step, key_scan = jax.random.split(prng_key, 3)
    state, _info = env.init(key_reset)
    action = env.action_space.sample(key_step)
    _state2, _info2 = env.step(state, action)

    # Small scan warmup (kept tiny to avoid adding more compile work than needed).
    num_steps = 3
    action_keys = jax.random.split(key_scan, num_steps)
    actions = jax.vmap(env.action_space.sample)(action_keys)

    def step_fn(s, a):
        return env.step(s, a)

    jax.lax.scan(step_fn, state, actions)


def _create_kinetix_env(level_id: str = "random", **kwargs):
    """Helper to create a KinetixEnvelope wrapper."""
    if level_id == "random":
        return KinetixEnvelope.create_random(**kwargs)
    return KinetixEnvelope.from_name(level_id, env_kwargs=kwargs)


def test_normalize_level_id_appends_json():
    assert _normalize_level_id("s/h4_thrust_aim") == "s/h4_thrust_aim.json"


def test_normalize_level_id_keeps_json():
    assert _normalize_level_id("s/h4_thrust_aim.json") == "s/h4_thrust_aim.json"


def test_normalize_level_id_strips_leading_slash_and_whitespace():
    assert _normalize_level_id(" /s/h4_thrust_aim  ") == "s/h4_thrust_aim.json"


@pytest.mark.parametrize("level_id", ["", "   "])
def test_normalize_level_id_rejects_empty_values(level_id):
    with pytest.raises(ValueError, match="non-empty"):
        _normalize_level_id(level_id)


def test_normalize_level_id_rejects_trailing_slash():
    with pytest.raises(ValueError, match="must not end"):
        _normalize_level_id("s/")


def _first_packaged_level_id(size: str = "s") -> str:
    """Return a packaged `{size}/{name}` level id, or skip if unavailable."""
    pkg_dir = pathlib.Path(kinetix.__file__).resolve().parent
    levels_dir = pkg_dir / "levels" / size
    if not levels_dir.exists():
        pytest.skip("kinetix package does not contain levels directory")
    jsons = sorted(levels_dir.glob("*.json"))
    if not jsons:
        pytest.skip("kinetix package has no packaged JSON levels")
    return f"{size}/{jsons[0].stem}"


def _packaged_level_ids(sizes: tuple[str, ...] = ("s", "m", "l")) -> list[str]:
    """Return all packaged `{size}/{name}` level ids."""
    pkg_dir = pathlib.Path(kinetix.__file__).resolve().parent
    out: list[str] = []
    for size in sizes:
        levels_dir = pkg_dir / "levels" / size
        if not levels_dir.exists():
            continue
        for p in sorted(levels_dir.glob("*.json")):
            out.append(f"{size}/{p.stem}")
    return out


def test_kinetix_contract_smoke(prng_key, kinetix_random_env):
    assert_reset_step_contract(
        kinetix_random_env, key=prng_key, obs_check=assert_obs_matches_space
    )


def test_kinetix_contract_scan(prng_key, kinetix_random_env, scan_num_steps):
    assert_jitted_rollout_contract(
        kinetix_random_env, key=prng_key, num_steps=scan_num_steps
    )


def test_action_space_is_continuous_by_default(kinetix_random_env):
    env = kinetix_random_env
    assert isinstance(env.action_space, Continuous)


def test_multidiscrete_space_conversion(prng_key):
    cardinalities = jnp.asarray([3, 2, 4], dtype=jnp.int32)
    kinetix_space = kinetix_spaces.MultiDiscrete(
        n=int(cardinalities.sum()),
        number_of_dims_per_distribution=cardinalities,
    )

    space = _convert_space(kinetix_space)

    assert isinstance(space, Discrete)
    assert space.shape == cardinalities.shape
    assert space.dtype == cardinalities.dtype
    assert jnp.array_equal(space.n, cardinalities)
    assert space.contains(space.sample(prng_key))


def test_from_name_premade_level_smoke(prng_key):
    level_id = _first_packaged_level_id("s")
    env = _create_kinetix_env(level_id)

    state, info = env.init(prng_key)
    assert state is not None
    assert isinstance(info, Info)
    assert env.observation_space.contains(info.obs)


def test_create_random_allows_auto_reset():
    with pytest.warns(UserWarning, match="auto_reset"):
        env = _create_kinetix_env("random", auto_reset=True)

    assert env is not None


def test_key_splitting(kinetix_random_env, prng_key):
    env = kinetix_random_env
    key = prng_key
    state, _info = env.init(key)
    assert hasattr(state, "key")
    assert not jnp.array_equal(state.key, key)

    action = env.action_space.sample(jax.random.fold_in(prng_key, 1))
    next_state, _ = env.step(state, action)
    assert not jnp.array_equal(next_state.key, state.key)


def test_random_premade_kinetix_envs(prng_key):
    """Smoke-test a few packaged `{size}/{name}` levels (skip if none are packaged)."""
    level_ids = _packaged_level_ids()
    if not level_ids:
        pytest.skip("kinetix package has no packaged JSON levels")

    # Keep this deterministic and small (compile/runtime).
    for level_id in level_ids[:3]:
        env = _create_kinetix_env(level_id)
        reset_key, action_key = jax.random.split(prng_key, 2)

        state, info = env.init(reset_key)
        assert state is not None
        assert isinstance(info, Info)
        # Skip expensive contains check - shape/dtype check is sufficient
        assert info.obs.shape == env.observation_space.shape

        action = env.action_space.sample(action_key)
        next_state, next_info = env.step(state, action)
        assert next_state is not None
        assert isinstance(next_info, Info)
        assert next_info.obs.shape == env.observation_space.shape
        assert jnp.all(jnp.isfinite(jnp.asarray(next_info.reward)))


def test_from_name_rejects_unknown_env_kwargs():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        KinetixEnvelope.from_name("s/h4_thrust_aim", env_kwargs={"unknown": 1})


@pytest.mark.parametrize("size", ["s", "m", "l"])
def test_from_name_dispatches_size_categories(
    monkeypatch: pytest.MonkeyPatch, size: str
):
    sentinel = object()
    calls = []

    def create_from_size(cls, requested_size, **kwargs):
        calls.append((requested_size, kwargs))
        return sentinel

    monkeypatch.setattr(
        KinetixEnvelope, "create_from_size", classmethod(create_from_size)
    )

    result = KinetixEnvelope.from_name(size, env_kwargs={"auto_reset": True})

    assert result is sentinel
    assert calls == [(size, {"env_params": None, "auto_reset": True})]


def test_create_random_preserves_explicit_finite_horizon_and_warns(
    monkeypatch: pytest.MonkeyPatch,
):
    from envelope.adapters import kinetix_envelope

    supplied_params = EnvParams().replace(max_timesteps=17)
    fake_env = object()
    monkeypatch.setattr(
        kinetix_envelope,
        "make_reset_fn_sample_kinetix_level",
        lambda *_args: object(),
    )
    monkeypatch.setattr(
        kinetix_envelope,
        "make_kinetix_env",
        lambda **_kwargs: fake_env,
    )
    monkeypatch.setattr(
        kinetix_envelope,
        "_probe_gymnaxlike_info_placeholder",
        lambda _env, _params: Container(),
    )

    with pytest.warns(UserWarning, match="reported as termination"):
        env = KinetixEnvelope.create_random(env_params=supplied_params)

    assert env.env_params is supplied_params
    assert env.default_max_steps == 17


def test_create_random_preserves_nonfinite_horizon_without_warning(
    monkeypatch: pytest.MonkeyPatch,
):
    from envelope.adapters import kinetix_envelope

    supplied_params = EnvParams().replace(max_timesteps=jnp.inf)
    monkeypatch.setattr(
        kinetix_envelope,
        "make_reset_fn_sample_kinetix_level",
        lambda *_args: object(),
    )
    monkeypatch.setattr(
        kinetix_envelope,
        "make_kinetix_env",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        kinetix_envelope,
        "_probe_gymnaxlike_info_placeholder",
        lambda _env, _params: Container(),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        env = KinetixEnvelope.create_random(env_params=supplied_params)

    assert env.env_params is supplied_params
    assert env.default_max_steps is None


def test_from_name_rejects_missing_premade_state(monkeypatch: pytest.MonkeyPatch):
    from envelope.adapters import kinetix_envelope

    def mock_load(_level_id: str):
        return None, StaticEnvParams(), EnvParams()

    monkeypatch.setattr(kinetix_envelope, "load_from_json_file", mock_load)
    with pytest.raises(ValueError, match="did not contain a level state"):
        KinetixEnvelope.from_name("s/h4_thrust_aim")


def test_create_premade_replace_failure_raises(monkeypatch: pytest.MonkeyPatch):
    """Premade path does not guard against replace() failures."""
    ep = EnvParams()
    if not hasattr(ep, "max_timesteps") or not hasattr(ep, "replace"):
        pytest.skip("Kinetix EnvParams does not expose max_timesteps/.replace")

    def failing_replace(self, **kwargs):
        raise AttributeError("replace failed")

    monkeypatch.setattr(EnvParams, "replace", failing_replace)
    monkeypatch.setattr(
        "envelope.adapters.kinetix_envelope.load_from_json_file",
        lambda _level_id: (object(), StaticEnvParams(), ep),
    )

    with pytest.raises(AttributeError, match="replace failed"):
        KinetixEnvelope.from_name("s/h4_thrust_aim")


# NOTE: Level-id normalization tests live in tests/adapters/test_kinetix_level_id.py
# to keep this module focused on runtime wrapper behavior.
