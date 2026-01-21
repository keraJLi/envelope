import jax
import jax.numpy as jnp
import pytest

from envelope.wrappers.truncation_wrapper import TruncationWrapper
from tests.wrappers.helpers import (
    DiscreteStepCounterEnv,
    NoStepsEnv,
    StepCounterEnv,
)


def test_reset_sets_truncated_false():
    env = StepCounterEnv()
    w = TruncationWrapper(env=env, max_steps=3)
    key = jax.random.PRNGKey(0)
    state, info = w.reset(key)
    assert state is not None
    assert info.truncated is False


@pytest.mark.parametrize(
    "env_factory,actions,max_steps,expected_truncated_seq",
    [
        (StepCounterEnv, [0.1, 0.2, 0.3, 0.4], 3, [False, False, True, True]),
        (StepCounterEnv, [0.1], 1, [True]),
        (DiscreteStepCounterEnv, [1, 2, 3], 2, [False, True, True]),
    ],
    ids=["cont_ms3", "cont_ms1", "disc_ms2"],
)
def test_step_truncates_at_threshold(
    env_factory, actions, max_steps, expected_truncated_seq
):
    env = env_factory()
    w = TruncationWrapper(env=env, max_steps=max_steps)
    key = jax.random.PRNGKey(0)
    state, info = w.reset(key)
    truncs = []
    for a in actions:
        state, info = w.step(state, jnp.asarray(a))
        truncs.append(bool(info.truncated))
    assert truncs == expected_truncated_seq


def test_preserves_other_info_fields():
    env = StepCounterEnv()
    w = TruncationWrapper(env=env, max_steps=2)
    key = jax.random.PRNGKey(0)
    state, info = w.reset(key)
    # Step once: not truncated yet
    state, info = w.step(state, jnp.asarray(0.5))
    assert info.terminated is False
    assert jnp.allclose(info.obs, state.unwrapped.env_state)
    # Step twice: hits threshold
    state, info = w.step(state, jnp.asarray(-0.25))
    assert info.terminated is False
    assert jnp.allclose(info.obs, state.unwrapped.env_state)
    assert bool(jnp.asarray(info.truncated)) is True


def test_reset_overrides_underlying_truncated_true():
    env = StepCounterEnv(reset_truncated=True)
    w = TruncationWrapper(env=env, max_steps=5)
    key = jax.random.PRNGKey(0)
    state, info = w.reset(key)
    assert info.truncated is False


@pytest.mark.parametrize("max_steps", [0, 1])
def test_max_steps_edge_values(max_steps):
    env = StepCounterEnv()
    w = TruncationWrapper(env=env, max_steps=max_steps)
    key = jax.random.PRNGKey(0)
    state, info = w.reset(key)
    # First step should truncate immediately when max_steps == 0
    state, info = w.step(state, jnp.asarray(0.0))
    # After first step, steps == 1; wrapper truncates when steps >= max_steps
    expected = 1 >= max_steps
    assert bool(jnp.asarray(info.truncated)) == expected


def test_truncated_remains_true_after_threshold():
    env = StepCounterEnv()
    w = TruncationWrapper(env=env, max_steps=2)
    key = jax.random.PRNGKey(0)
    state, info = w.reset(key)
    # Step 1: steps=1 < 2
    state, info = w.step(state, jnp.asarray(0.1))
    assert bool(jnp.asarray(info.truncated)) is False
    # Step 2: steps=2 == 2 -> truncated
    state, info = w.step(state, jnp.asarray(0.1))
    assert bool(jnp.asarray(info.truncated)) is True
    # Step 3: stays truncated
    state, info = w.step(state, jnp.asarray(0.1))
    assert bool(jnp.asarray(info.truncated)) is True


def test_wrapper_overrides_underlying_truncated_on_step():
    env = StepCounterEnv(step_truncated=True)
    w = TruncationWrapper(env=env, max_steps=10)
    key = jax.random.PRNGKey(0)
    state, info = w.reset(key)
    # Underlying env sets truncated True, but steps < max_steps => wrapper should set False
    state, info = w.step(state, jnp.asarray(0.5))
    assert bool(jnp.asarray(info.truncated)) is False


def test_steps_as_jax_scalar_array_behaves_correctly():
    env = StepCounterEnv(steps_dtype=jnp.int32)
    w = TruncationWrapper(env=env, max_steps=2)
    key = jax.random.PRNGKey(0)
    state, info = w.reset(key)
    # After one step: steps = 1 (jax scalar), not truncated
    state, info = w.step(state, jnp.asarray(0.1))
    assert jnp.asarray(info.truncated).dtype == jnp.bool_
    assert bool(jnp.asarray(info.truncated)) is False
    # After second step: steps = 2 -> truncated
    state, info = w.step(state, jnp.asarray(0.1))
    assert bool(jnp.asarray(info.truncated)) is True


@pytest.mark.parametrize(
    "env_factory,action",
    [
        (StepCounterEnv, jnp.asarray(0.3)),
        (DiscreteStepCounterEnv, jnp.asarray(3, dtype=jnp.int32)),
    ],
    ids=["jit-cont", "jit-disc"],
)
def test_jit_compatibility(env_factory, action):
    env = env_factory()
    w = TruncationWrapper(env=env, max_steps=2)
    key = jax.random.PRNGKey(0)

    # Avoid returning InfoContainer across JIT boundary; return only needed pieces
    reset_jit_state = jax.jit(lambda k: w.reset(k)[0])
    step_jit_state_trunc = jax.jit(
        lambda s, a: (w.step(s, a)[0], w.step(s, a)[1].truncated)
    )

    state = reset_jit_state(key)

    next_state, truncated = step_jit_state_trunc(state, action)
    # `truncated` may be a JAX scalar array after JIT; validate dtype/shape
    assert jnp.asarray(truncated).dtype == jnp.bool_
    assert jnp.shape(truncated) == ()
