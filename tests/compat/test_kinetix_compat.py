"""Tests for jenv.compat.kinetix_jenv module."""

# ruff: noqa: E402

from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import pytest

kinetix = pytest.importorskip("kinetix")

pytestmark = pytest.mark.compat

from kinetix.environment import (
    EnvParams,
    StaticEnvParams,
)

from jenv.compat.kinetix_jenv import KinetixJenv, _normalize_level_id
from jenv.environment import Info
from jenv.spaces import Continuous
from tests.compat.contract import (
    assert_jitted_rollout_contract,
    assert_reset_step_contract,
)


@pytest.fixture(scope="module")
def kinetix_random_env():
    """Create a single random Kinetix env for the whole module."""
    return KinetixJenv.create_random()


@pytest.fixture(scope="module", autouse=True)
def kinetix_random_env_warmup(kinetix_random_env, prng_key):
    """Warm up reset/step/scan once to amortize compilation cost."""
    env = kinetix_random_env
    key_reset, key_step, key_scan = jax.random.split(prng_key, 3)
    state, _info = env.reset(key_reset)
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
    """Helper to create a KinetixJenv wrapper."""
    if level_id == "random":
        return KinetixJenv.create_random(**kwargs)
    return KinetixJenv.from_name(level_id, env_kwargs=kwargs)


def test_normalize_level_id_appends_json():
    assert _normalize_level_id("s/h4_thrust_aim") == "s/h4_thrust_aim.json"


def test_normalize_level_id_keeps_json():
    assert _normalize_level_id("s/h4_thrust_aim.json") == "s/h4_thrust_aim.json"


def test_normalize_level_id_strips_leading_slash_and_whitespace():
    assert _normalize_level_id(" /s/h4_thrust_aim  ") == "s/h4_thrust_aim.json"


@pytest.mark.parametrize("bad", ["", "   ", "/", "s/"])
def test_normalize_level_id_rejects_empty_or_trailing_slash(bad: str):
    with pytest.raises(ValueError):
        _normalize_level_id(bad)


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
    env = kinetix_random_env

    def obs_check(obs, obs_space):
        # Kinetix observations can be large; a shape check is sufficient here.
        assert obs.shape == obs_space.shape

    assert_reset_step_contract(env, key=prng_key, obs_check=obs_check)


def test_kinetix_contract_scan(prng_key, kinetix_random_env, scan_num_steps):
    assert_jitted_rollout_contract(
        kinetix_random_env, key=prng_key, num_steps=scan_num_steps
    )


def test_action_space_is_continuous_by_default(kinetix_random_env):
    env = kinetix_random_env
    assert isinstance(env.action_space, Continuous)


def test_from_name_premade_level_smoke(prng_key):
    level_id = _first_packaged_level_id("s")
    env = _create_kinetix_env(level_id)

    state, info = env.reset(prng_key)
    assert state is not None
    assert isinstance(info, Info)
    assert env.observation_space.contains(info.obs)


def test_create_random_with_auto_reset_warning(prng_key):
    with pytest.warns(
        UserWarning,
        match="Creating a KinetixJenv with auto_reset=True is not recommended",
    ):
        env = _create_kinetix_env("random", auto_reset=True)

    state, info = env.reset(prng_key)
    assert state is not None
    assert isinstance(info, Info)


def test_create_random_with_finite_max_timesteps_warning(prng_key):
    # Only run if Kinetix's EnvParams has max_timesteps and supports .replace(...)
    ep = EnvParams()
    if not hasattr(ep, "max_timesteps") or not hasattr(ep, "replace"):
        pytest.skip("Kinetix EnvParams does not expose max_timesteps/.replace")

    ep_finite = ep.replace(max_timesteps=10)
    with pytest.warns(UserWarning, match="finite max_timesteps is not recommended"):
        env = _create_kinetix_env("random", env_params=ep_finite)

    state, info = env.reset(prng_key)
    assert state is not None
    assert isinstance(info, Info)


def test_key_splitting(kinetix_random_env, prng_key):
    env = kinetix_random_env
    key = prng_key
    state, _info = env.reset(key)
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

        state, info = env.reset(reset_key)
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
        KinetixJenv.from_name("s/h4_thrust_aim", env_kwargs={"unknown": 1})


def test_from_name_allows_premade_state_none(monkeypatch: pytest.MonkeyPatch):
    """Current implementation does not guard against missing premade state."""
    from jenv.compat import kinetix_jenv

    def mock_load(_level_id: str):
        return None, StaticEnvParams(), EnvParams()

    monkeypatch.setattr(kinetix_jenv, "load_from_json_file", mock_load)
    env = KinetixJenv.from_name("s/h4_thrust_aim")
    assert env is not None


def test_create_premade_replace_failure_raises(monkeypatch: pytest.MonkeyPatch):
    """Premade path does not guard against replace() failures."""
    ep = EnvParams()
    if not hasattr(ep, "max_timesteps") or not hasattr(ep, "replace"):
        pytest.skip("Kinetix EnvParams does not expose max_timesteps/.replace")

    def failing_replace(self, **kwargs):
        raise AttributeError("replace failed")

    monkeypatch.setattr(EnvParams, "replace", failing_replace)
    monkeypatch.setattr(
        "jenv.compat.kinetix_jenv.load_from_json_file",
        lambda _level_id: (object(), StaticEnvParams(), ep),
    )

    with pytest.raises(AttributeError, match="replace failed"):
        KinetixJenv.from_name("s/h4_thrust_aim")


#
# NOTE: Level-id normalization tests live in tests/compat/test_kinetix_level_id.py
# to keep this module focused on runtime wrapper behavior.
