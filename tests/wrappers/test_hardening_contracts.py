"""Focused cross-wrapper contracts for the Envelope 0.5 lifecycle semantics."""

import inspect
from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest

from envelope.wrappers.autoreset_wrapper import AutoResetWrapper
from envelope.wrappers.continuous_observation_wrapper import (
    ContinuousObservationWrapper,
)
from envelope.wrappers.episode_statistics_wrapper import EpisodeStatisticsWrapper
from envelope.wrappers.flatten_observation_wrapper import FlattenObservationWrapper
from envelope.wrappers.pooled_init_vmap_wrapper import PooledInitVmapWrapper
from envelope.wrappers.truncation_wrapper import TruncationWrapper
from envelope.wrappers.vmap_wrapper import VmapWrapper
from tests.wrappers.helpers import PyTreeObsEnv, StepCounterEnv


@pytest.mark.parametrize(
    "make_env",
    [
        lambda: ContinuousObservationWrapper(StepCounterEnv()),
        lambda: FlattenObservationWrapper(
            PyTreeObsEnv(shapes={"a": (2,), "b": (3,)})
        ),
        lambda: EpisodeStatisticsWrapper(StepCounterEnv()),
        lambda: AutoResetWrapper(StepCounterEnv()),
        lambda: PooledInitVmapWrapper(
            StepCounterEnv(), batch_size=2, pool_size=2
        ),
    ],
    ids=[
        "continuous-observation",
        "flatten-observation",
        "episode-statistics",
        "autoreset",
        "pooled-init-vmap",
    ],
)
def test_reset_accepts_canonical_state_key_keywords(make_env: Callable[[], object]):
    env = make_env()
    key = jax.random.key(0)
    state, _ = env.init(key)

    parameter_names = list(inspect.signature(type(env).reset).parameters)
    assert parameter_names[:3] == ["self", "state", "key"]

    reset_state, reset_info = env.reset(state, jax.random.key(1))
    keyword_state, keyword_info = env.reset(state=state, key=jax.random.key(1))

    assert reset_state is not None
    assert reset_info is not None
    assert jax.tree.structure(keyword_state) == jax.tree.structure(reset_state)
    assert jax.tree.structure(keyword_info) == jax.tree.structure(reset_info)


def test_documented_episode_stack_runs_through_boundary_under_jit_scan():
    env = VmapWrapper(
        AutoResetWrapper(
            EpisodeStatisticsWrapper(
                TruncationWrapper(StepCounterEnv(terminate_after=2), max_steps=5)
            )
        ),
        batch_size=2,
    )
    state, _ = env.init(jax.random.key(0))
    actions = jnp.asarray([[0.1, 0.1], [0.2, 0.2]], dtype=jnp.float32)

    @jax.jit
    def rollout(initial_state, xs):
        return jax.lax.scan(
            lambda carry, action: env.step(carry, action), initial_state, xs
        )

    _, infos = rollout(state, actions)

    assert jnp.all(infos.final_valid[0] == jnp.asarray([False, False]))
    assert jnp.all(infos.final_valid[1] == jnp.asarray([True, True]))
    assert jnp.all(infos.terminated[1])
    assert jnp.allclose(infos.reward[1], jnp.asarray([0.2, 0.2]))
    assert jnp.allclose(infos.stats.reward[1], jnp.asarray([0.0, 0.0]))
    assert jnp.allclose(infos.final.stats.reward[1], jnp.asarray([0.3, 0.3]))
    assert jnp.all(infos.final.stats.length[1] == 2)


def test_autoreset_rejects_vectorization_on_the_inside():
    with pytest.raises(ValueError, match="(?i)autoreset.*inside.*vmap|vectorization"):
        AutoResetWrapper(VmapWrapper(StepCounterEnv(), batch_size=2))
