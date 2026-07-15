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
from envelope.wrappers.state_injection_wrapper import StateInjectionWrapper
from envelope.wrappers.truncation_wrapper import TruncationWrapper
from envelope.wrappers.vmap_wrapper import VmapWrapper
from tests.wrappers.helpers import NonzeroInitInfoEnv, PyTreeObsEnv, StepCounterEnv


@pytest.mark.parametrize(
    "make_env",
    [
        lambda: ContinuousObservationWrapper(StepCounterEnv()),
        lambda: FlattenObservationWrapper(PyTreeObsEnv(shapes={"a": (2,), "b": (3,)})),
        lambda: EpisodeStatisticsWrapper(StepCounterEnv()),
        lambda: AutoResetWrapper(StepCounterEnv()),
        lambda: PooledInitVmapWrapper(StepCounterEnv(), batch_size=2, pool_size=2),
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


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("wrapper_kind", ["autoreset", "pooled"])
def test_initial_final_is_a_type_preserving_zero_placeholder(
    wrapper_kind: str, compiled: bool
):
    base = NonzeroInitInfoEnv()
    key = jax.random.key(0)
    if wrapper_kind == "autoreset":
        env = AutoResetWrapper(base)
        _, reference_info = base.init(key)
    else:
        batch_size = 2
        env = PooledInitVmapWrapper(base, batch_size=batch_size, pool_size=batch_size)
        keys = jax.random.split(key, batch_size)
        _, reference_info = jax.vmap(base.init)(keys)

    init = jax.jit(env.init) if compiled else env.init
    state, info = init(key)

    assert jnp.all(jnp.asarray(info.final_valid) == 0)
    for placeholder in (state.last_final, info.final):
        assert jax.tree.structure(placeholder) == jax.tree.structure(reference_info)
        for actual, reference in zip(
            jax.tree.leaves(placeholder), jax.tree.leaves(reference_info)
        ):
            actual = jnp.asarray(actual)
            reference = jnp.asarray(reference)
            assert actual.shape == reference.shape
            assert actual.dtype == reference.dtype
            assert jnp.all(actual == jnp.zeros_like(reference))


@pytest.mark.parametrize(
    "make_env,error_pattern",
    [
        (
            lambda: EpisodeStatisticsWrapper(AutoResetWrapper(StepCounterEnv())),
            r"(?i)EpisodeStatisticsWrapper.*inside.*AutoResetWrapper",
        ),
        (
            lambda: EpisodeStatisticsWrapper(
                PooledInitVmapWrapper(StepCounterEnv(), batch_size=2, pool_size=2)
            ),
            r"(?i)EpisodeStatisticsWrapper.*inside.*PooledInitVmapWrapper",
        ),
        (
            lambda: TruncationWrapper(AutoResetWrapper(StepCounterEnv()), max_steps=5),
            r"(?i)TruncationWrapper.*inside.*AutoResetWrapper",
        ),
        (
            lambda: TruncationWrapper(
                PooledInitVmapWrapper(StepCounterEnv(), batch_size=2, pool_size=2),
                max_steps=5,
            ),
            r"(?i)TruncationWrapper.*inside.*PooledInitVmapWrapper",
        ),
        (
            lambda: StateInjectionWrapper(AutoResetWrapper(StepCounterEnv())),
            r"(?i)StateInjectionWrapper.*inside.*AutoResetWrapper",
        ),
        (
            lambda: StateInjectionWrapper(
                PooledInitVmapWrapper(StepCounterEnv(), batch_size=2, pool_size=2)
            ),
            r"(?i)StateInjectionWrapper.*inside.*PooledInitVmapWrapper",
        ),
    ],
    ids=[
        "episode-stats-outside-autoreset",
        "episode-stats-outside-pooled",
        "truncation-outside-autoreset",
        "truncation-outside-pooled",
        "state-injection-outside-autoreset",
        "state-injection-outside-pooled",
    ],
)
def test_invalid_wrapper_order_is_rejected_at_construction(
    make_env: Callable[[], object], error_pattern: str
):
    with pytest.raises(ValueError, match=error_pattern):
        make_env()


@pytest.mark.parametrize(
    "make_env",
    [
        lambda: AutoResetWrapper(
            EpisodeStatisticsWrapper(
                TruncationWrapper(StateInjectionWrapper(StepCounterEnv()), max_steps=5)
            )
        ),
        lambda: PooledInitVmapWrapper(
            EpisodeStatisticsWrapper(TruncationWrapper(StepCounterEnv(), max_steps=5)),
            batch_size=2,
            pool_size=2,
        ),
    ],
    ids=["autoreset", "pooled"],
)
def test_canonical_wrapper_orders_remain_jittable(make_env: Callable[[], object]):
    env = make_env()
    key = jax.random.key(0)
    state, _ = jax.jit(env.init)(key)
    action = env.action_space.sample(key)
    next_state, info = jax.jit(env.step)(state, action)

    assert next_state is not None
    assert info is not None
