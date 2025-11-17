from functools import cached_property

import jax
import jax.numpy as jnp
import pytest

from jenv.environment import Environment, InfoContainer
from jenv.spaces import Continuous, Discrete
from jenv.struct import FrozenPyTreeNode
from jenv.typing import Key
from jenv.wrappers.canonicalize_wrapper import CanonicalizeWrapper
from jenv.wrappers.timestep_wrapper import TimeStepWrapper
from jenv.wrappers.truncation_wrapper import TruncationWrapper


class State(FrozenPyTreeNode):
    env_state: jax.Array
    steps: int


class NoStepsState(FrozenPyTreeNode):
    env_state: jax.Array


class ScalarEnvWithSteps(Environment):
    @cached_property
    def observation_space(self) -> Continuous:
        return Continuous(low=-jnp.inf, high=jnp.inf, shape=())

    @cached_property
    def action_space(self) -> Continuous:
        return Continuous(low=-1.0, high=1.0, shape=())

    def reset(self, key: Key) -> tuple[State, InfoContainer]:
        s = State(env_state=jnp.array(0.0), steps=0)
        return s, InfoContainer(
            obs=s.env_state, reward=0.0, terminated=False, truncated=False
        )

    def step(self, state: State, action: jax.Array) -> tuple[State, InfoContainer]:
        ns = State(env_state=state.env_state + action, steps=state.steps + 1)
        info = InfoContainer(
            obs=ns.env_state,
            reward=jnp.asarray(action),
            terminated=False,
            truncated=False,
        )
        return ns, info


class DiscreteEnvWithSteps(ScalarEnvWithSteps):
    @cached_property
    def action_space(self) -> Discrete:
        return Discrete(n=5)

    def step(self, state: State, action: jax.Array) -> tuple[State, InfoContainer]:
        ns = State(
            env_state=state.env_state + action.astype(jnp.float32),
            steps=state.steps + 1,
        )
        info = InfoContainer(
            obs=ns.env_state,
            reward=jnp.asarray(action),
            terminated=False,
            truncated=False,
        )
        return ns, info


class EnvMissingSteps(Environment):
    @cached_property
    def observation_space(self) -> Continuous:
        return Continuous(low=-jnp.inf, high=jnp.inf, shape=())

    @cached_property
    def action_space(self) -> Continuous:
        return Continuous(low=-1.0, high=1.0, shape=())

    def reset(self, key: Key) -> tuple[NoStepsState, InfoContainer]:
        s = NoStepsState(env_state=jnp.array(0.0))
        return s, InfoContainer(
            obs=s.env_state, reward=0.0, terminated=False, truncated=False
        )

    def step(
        self, state: NoStepsState, action: jax.Array
    ) -> tuple[NoStepsState, InfoContainer]:
        ns = NoStepsState(env_state=state.env_state + action)
        info = InfoContainer(
            obs=ns.env_state, reward=float(action), terminated=False, truncated=False
        )
        return ns, info


class EnvResetTruncated(ScalarEnvWithSteps):
    def reset(self, key: Key) -> tuple[State, InfoContainer]:
        s = State(env_state=jnp.array(0.0), steps=0)
        return s, InfoContainer(
            obs=s.env_state, reward=0.0, terminated=False, truncated=True
        )


class EnvStepAlwaysTruncated(ScalarEnvWithSteps):
    def step(self, state: State, action: jax.Array) -> tuple[State, InfoContainer]:
        ns = State(env_state=state.env_state + action, steps=state.steps + 1)
        info = InfoContainer(
            obs=ns.env_state,
            reward=jnp.asarray(action),
            terminated=False,
            truncated=True,
        )
        return ns, info


class EnvWithArraySteps(Environment):
    @cached_property
    def observation_space(self) -> Continuous:
        return Continuous(low=-jnp.inf, high=jnp.inf, shape=())

    @cached_property
    def action_space(self) -> Continuous:
        return Continuous(low=-1.0, high=1.0, shape=())

    def reset(self, key: Key) -> tuple[State, InfoContainer]:
        s = State(env_state=jnp.array(0.0), steps=jnp.array(0, dtype=jnp.int32))
        return s, InfoContainer(
            obs=s.env_state, reward=0.0, terminated=False, truncated=False
        )

    def step(self, state: State, action: jax.Array) -> tuple[State, InfoContainer]:
        ns = State(
            env_state=state.env_state + action,
            steps=state.steps + jnp.array(1, dtype=jnp.int32),
        )
        info = InfoContainer(
            obs=ns.env_state,
            reward=jnp.asarray(action),
            terminated=False,
            truncated=False,
        )
        return ns, info


def test_reset_sets_truncated_false():
    env = ScalarEnvWithSteps()
    w = TruncationWrapper(
        env=TimeStepWrapper(env=CanonicalizeWrapper(env=env)), max_steps=3
    )
    key = jax.random.PRNGKey(0)
    state, info = w.reset(key)
    assert state is not None
    assert info.truncated is False


@pytest.mark.parametrize(
    "env_factory,actions,max_steps,expected_truncated_seq",
    [
        (ScalarEnvWithSteps, [0.1, 0.2, 0.3, 0.4], 3, [False, False, True, True]),
        (ScalarEnvWithSteps, [0.1], 1, [True]),
        (DiscreteEnvWithSteps, [1, 2, 3], 2, [False, True, True]),
    ],
    ids=["cont_ms3", "cont_ms1", "disc_ms2"],
)
def test_step_truncates_at_threshold(
    env_factory, actions, max_steps, expected_truncated_seq
):
    env = env_factory()
    w = TruncationWrapper(
        env=TimeStepWrapper(env=CanonicalizeWrapper(env=env)), max_steps=max_steps
    )
    key = jax.random.PRNGKey(0)
    state, info = w.reset(key)
    truncs = []
    for a in actions:
        state, info = w.step(state, jnp.asarray(a))
        truncs.append(bool(info.truncated))
    assert truncs == expected_truncated_seq


def test_preserves_other_info_fields():
    env = ScalarEnvWithSteps()
    w = TruncationWrapper(
        env=TimeStepWrapper(env=CanonicalizeWrapper(env=env)), max_steps=2
    )
    key = jax.random.PRNGKey(0)
    state, info = w.reset(key)
    # Step once: not truncated yet
    state, info = w.step(state, jnp.asarray(0.5))
    assert info.terminated is False
    assert jnp.allclose(info.obs, state.core.env_state)
    # Step twice: hits threshold
    state, info = w.step(state, jnp.asarray(-0.25))
    assert info.terminated is False
    assert jnp.allclose(info.obs, state.core.env_state)
    assert bool(jnp.asarray(info.truncated)) is True


def test_missing_steps_attribute_raises():
    env = EnvMissingSteps()
    w = TruncationWrapper(env=CanonicalizeWrapper(env=env), max_steps=1)
    key = jax.random.PRNGKey(0)
    state, info = w.reset(key)
    with pytest.raises(ValueError) as e:
        _ = w.step(state, jnp.asarray(0.1))
    assert "requires a 'steps' attribute" in str(e.value)


def test_reset_overrides_underlying_truncated_true():
    env = EnvResetTruncated()
    w = TruncationWrapper(
        env=TimeStepWrapper(env=CanonicalizeWrapper(env=env)), max_steps=5
    )
    key = jax.random.PRNGKey(0)
    state, info = w.reset(key)
    assert info.truncated is False


@pytest.mark.parametrize("max_steps", [0, 1])
def test_max_steps_edge_values(max_steps):
    env = ScalarEnvWithSteps()
    w = TruncationWrapper(
        env=TimeStepWrapper(env=CanonicalizeWrapper(env=env)), max_steps=max_steps
    )
    key = jax.random.PRNGKey(0)
    state, info = w.reset(key)
    # First step should truncate immediately when max_steps == 0
    state, info = w.step(state, jnp.asarray(0.0))
    # After first step, steps == 1; wrapper truncates when steps >= max_steps
    expected = 1 >= max_steps
    assert bool(jnp.asarray(info.truncated)) == expected


def test_truncated_remains_true_after_threshold():
    env = ScalarEnvWithSteps()
    w = TruncationWrapper(
        env=TimeStepWrapper(env=CanonicalizeWrapper(env=env)), max_steps=2
    )
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
    env = EnvStepAlwaysTruncated()
    w = TruncationWrapper(
        env=TimeStepWrapper(env=CanonicalizeWrapper(env=env)), max_steps=10
    )
    key = jax.random.PRNGKey(0)
    state, info = w.reset(key)
    # Underlying env sets truncated True, but steps < max_steps => wrapper should set False
    state, info = w.step(state, jnp.asarray(0.5))
    assert bool(jnp.asarray(info.truncated)) is False


def test_steps_as_jax_scalar_array_behaves_correctly():
    env = EnvWithArraySteps()
    w = TruncationWrapper(
        env=TimeStepWrapper(env=CanonicalizeWrapper(env=env)), max_steps=2
    )
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
        (ScalarEnvWithSteps, jnp.asarray(0.3)),
        (DiscreteEnvWithSteps, jnp.asarray(3, dtype=jnp.int32)),
    ],
    ids=["jit-cont", "jit-disc"],
)
def test_jit_compatibility(env_factory, action):
    env = env_factory()
    w = TruncationWrapper(
        env=TimeStepWrapper(env=CanonicalizeWrapper(env=env)), max_steps=2
    )
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
