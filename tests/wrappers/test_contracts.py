"""Contract smoke tests for wrapper environments."""

import jax
import pytest

from envelope.wrappers.autoreset_wrapper import AutoResetWrapper
from envelope.wrappers.clip_action_wrapper import ClipActionWrapper
from envelope.wrappers.continuous_observation_wrapper import ContinuousObservationWrapper
from envelope.wrappers.episode_statistics_wrapper import EpisodeStatisticsWrapper
from envelope.wrappers.flatten_action_wrapper import FlattenActionWrapper
from envelope.wrappers.flatten_observation_wrapper import FlattenObservationWrapper
from envelope.wrappers.observation_normalization_wrapper import (
    ObservationNormalizationWrapper,
)
from envelope.wrappers.state_injection_wrapper import StateInjectionWrapper
from envelope.wrappers.truncation_wrapper import TruncationWrapper
from envelope.wrappers.wrapper import Wrapper
from tests.contract import assert_obs_matches_space, assert_reset_step_contract
from tests.wrappers.helpers import (
    IntObsEnv,
    PyTreeActionEnv,
    PyTreeObsEnv,
    ScalarToyEnv,
    StepCounterEnv,
    VectorToyEnv,
    WrapperSimpleEnv,
)


def _wrap_base():
    return Wrapper(env=WrapperSimpleEnv())


def _wrap_autoreset():
    return AutoResetWrapper(env=StepCounterEnv())


def _wrap_truncation():
    return TruncationWrapper(env=StepCounterEnv(), max_steps=3)


def _wrap_clip_action():
    return ClipActionWrapper(env=ScalarToyEnv())


def _wrap_continuous_obs():
    return ContinuousObservationWrapper(env=IntObsEnv())


def _wrap_flatten_obs():
    return FlattenObservationWrapper(env=PyTreeObsEnv(shapes={"a": (2,), "b": (3,)}))


def _wrap_flatten_action():
    return FlattenActionWrapper(env=PyTreeActionEnv())


def _wrap_episode_statistics():
    return EpisodeStatisticsWrapper(env=StepCounterEnv())


def _wrap_observation_normalization():
    return ObservationNormalizationWrapper(env=VectorToyEnv(dim=3))


def _wrap_state_injection():
    return StateInjectionWrapper(env=StepCounterEnv())


@pytest.mark.parametrize(
    "make_env",
    [
        _wrap_base,
        _wrap_autoreset,
        _wrap_truncation,
        _wrap_clip_action,
        _wrap_continuous_obs,
        _wrap_flatten_obs,
        _wrap_flatten_action,
        _wrap_episode_statistics,
        _wrap_observation_normalization,
        _wrap_state_injection,
    ],
    ids=[
        "wrapper_base",
        "autoreset",
        "truncation",
        "clip_action",
        "continuous_observation",
        "flatten_observation",
        "flatten_action",
        "episode_statistics",
        "observation_normalization",
        "state_injection",
    ],
)
def test_wrapper_contract_smoke(make_env):
    env = make_env()
    key = jax.random.key(0)
    assert_reset_step_contract(env, key=key, obs_check=assert_obs_matches_space)
