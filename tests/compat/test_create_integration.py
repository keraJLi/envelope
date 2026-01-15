"""Integration tests for jenv.compat.create().

These require optional compatibility dependencies (brax/gymnax/navix). They are
kept separate from the unit tests so a minimal install can still run the suite.
"""

import pytest

from jenv.compat import create
from jenv.environment import Environment

pytestmark = pytest.mark.compat


def test_create_brax_smoke(prng_key):
    pytest.importorskip("brax")

    from jenv.compat.brax_jenv import BraxJenv

    env = create("brax::fast")
    assert isinstance(env, BraxJenv)
    assert isinstance(env, Environment)

    _state, info = env.reset(prng_key)
    assert hasattr(info, "obs")


def test_create_gymnax_smoke(prng_key):
    pytest.importorskip("gymnax")

    from jenv.compat.gymnax_jenv import GymnaxJenv

    env = create("gymnax::CartPole-v1")
    assert isinstance(env, GymnaxJenv)
    assert isinstance(env, Environment)

    _state, info = env.reset(prng_key)
    assert hasattr(info, "obs")


def test_create_navix_smoke(prng_key):
    pytest.importorskip("navix")

    from jenv.compat.navix_jenv import NavixJenv

    env = create("navix::Navix-Empty-5x5-v0")
    assert isinstance(env, NavixJenv)
    assert isinstance(env, Environment)

    _state, info = env.reset(prng_key)
    assert hasattr(info, "obs")


def test_create_jumanji_smoke(prng_key):
    pytest.importorskip("jumanji")

    from jenv.compat.jumanji_jenv import JumanjiJenv

    env = create("jumanji::Snake-v1")
    assert isinstance(env, JumanjiJenv)
    assert isinstance(env, Environment)

    _state, info = env.reset(prng_key)
    assert hasattr(info, "obs")


def test_create_craftax_smoke(prng_key):
    pytest.importorskip("craftax")

    from jenv.compat.craftax_jenv import CraftaxJenv

    env = create("craftax::Craftax-Symbolic-v1")
    assert isinstance(env, CraftaxJenv)
    assert isinstance(env, Environment)

    _state, info = env.reset(prng_key)
    assert hasattr(info, "obs")
