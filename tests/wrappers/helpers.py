"""Shared test helpers for wrapper tests.

These helpers are intentionally **test-only** and are designed to be small,
deterministic, and easy to compose across wrapper unit tests.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any

import jax
import jax.numpy as jnp

from envelope.environment import Environment, Info, InfoContainer, State
from envelope.spaces import Continuous, Discrete, PyTreeSpace
from envelope.struct import FrozenPyTreeNode, static_field
from envelope.typing import Key, PyTree

# ============================================================================
# Step-based envs (used by autoreset/truncation tests)
# ============================================================================


class StepState(FrozenPyTreeNode):
    """Common step-counter state used in multiple wrapper tests."""

    env_state: jax.Array
    # Tests use both python ints and JAX scalar arrays for `steps`.
    steps: int | jax.Array = 0


class NoStepsState(FrozenPyTreeNode):
    """State without a `steps` field (used for negative-path tests)."""

    env_state: jax.Array


class StepCounterEnv(Environment):
    """Scalar env whose `step()` increments `steps` and adds `action` to state.

    Parameters allow emulating a variety of edge cases required by wrapper tests:
    - termination/truncation after thresholds
    - always done / never done / both flags set
    - forced truncated flags on reset/step (underlying env misbehavior)
    - `steps` stored as python int or as a JAX scalar array
    """

    terminate_after: int | None = None
    truncate_after: int | None = None

    always_terminated: bool = False
    always_truncated: bool = False
    both_flags: bool = False

    reset_truncated: bool = False
    step_truncated: bool = False

    steps_dtype: jnp.dtype | None = None

    @cached_property
    def observation_space(self) -> Continuous:
        return Continuous(low=-jnp.inf, high=jnp.inf)

    @cached_property
    def action_space(self) -> Continuous:
        return Continuous(low=-1.0, high=1.0)

    def _threshold_terminated(self, steps: jax.Array) -> jax.Array:
        thr = self.terminate_after
        if thr is None:
            return jnp.asarray(False)
        return steps >= jnp.asarray(thr)

    def _threshold_truncated(self, steps: jax.Array) -> jax.Array:
        thr = self.truncate_after
        if thr is None:
            return jnp.asarray(False)
        return steps >= jnp.asarray(thr)

    def _make_steps(self, value: int) -> int | jax.Array:
        if self.steps_dtype is None:
            return int(value)
        return jnp.array(value, dtype=self.steps_dtype)

    def reset(
        self, key: Key, state: PyTree | None = None, **kwargs: Any
    ) -> tuple[StepState, InfoContainer]:
        s = StepState(env_state=jnp.array(0.0), steps=self._make_steps(0))
        # `reset_truncated` may become a traced boolean when env instances are vmapped.
        # Avoid `bool(tracer)` conversions.
        truncated = (
            bool(self.reset_truncated)
            if isinstance(self.reset_truncated, bool)
            else self.reset_truncated
        )
        return s, InfoContainer(
            obs=s.env_state,
            reward=0.0,
            terminated=False,
            truncated=truncated,
        )

    def step(
        self, state: StepState, action: jax.Array
    ) -> tuple[StepState, InfoContainer]:
        if isinstance(state.steps, int) and not isinstance(state.steps, bool):
            next_steps: int | jax.Array = state.steps + 1
        else:
            steps_arr = jnp.asarray(state.steps)
            next_steps = steps_arr + jnp.asarray(1, dtype=steps_arr.dtype)

        ns = StepState(env_state=state.env_state + action, steps=next_steps)

        # Preserve Python `bool` flags in the common (non-jitted) path.
        # Some tests assert identity (e.g. `info.terminated is False`).
        if isinstance(ns.steps, int) and not isinstance(ns.steps, bool):
            steps_i = int(ns.steps)
            if self.both_flags:
                terminated: bool | jax.Array = True
                truncated: bool | jax.Array = True
            else:
                terminated = bool(self.always_terminated) or (
                    self.terminate_after is not None
                    and steps_i >= int(self.terminate_after)
                )
                truncated = bool(self.always_truncated) or (
                    self.truncate_after is not None
                    and steps_i >= int(self.truncate_after)
                )
            if self.step_truncated:
                truncated = True
        else:
            steps_arr = jnp.asarray(ns.steps)

            both_flags = jnp.asarray(self.both_flags)
            always_terminated = jnp.asarray(self.always_terminated)
            always_truncated = jnp.asarray(self.always_truncated)
            step_truncated = jnp.asarray(self.step_truncated)

            term_thresh = (
                self._threshold_terminated(steps_arr)
                if self.terminate_after is not None
                else jnp.asarray(False)
            )
            trunc_thresh = (
                self._threshold_truncated(steps_arr)
                if self.truncate_after is not None
                else jnp.asarray(False)
            )

            terminated_base = jnp.logical_or(always_terminated, term_thresh)
            truncated_base = jnp.logical_or(always_truncated, trunc_thresh)

            terminated = jnp.where(both_flags, jnp.asarray(True), terminated_base)
            truncated = jnp.where(both_flags, jnp.asarray(True), truncated_base)
            truncated = jnp.where(step_truncated, jnp.asarray(True), truncated)

        info = InfoContainer(
            obs=ns.env_state,
            reward=jnp.asarray(action),
            terminated=terminated,
            truncated=truncated,
        )
        return ns, info


class DiscreteStepCounterEnv(StepCounterEnv):
    """StepCounterEnv with a discrete action space and float state accumulation."""

    action_n: int = 5

    @cached_property
    def action_space(self) -> Discrete:
        return Discrete(n=int(self.action_n))

    def step(
        self, state: StepState, action: jax.Array
    ) -> tuple[StepState, InfoContainer]:
        # Mirror prior tests: cast discrete action to float before accumulating.
        return super().step(state, action.astype(jnp.float32))


class NoStepsEnv(Environment):
    """Env that omits `steps` from state; used to assert wrapper errors."""

    @cached_property
    def observation_space(self) -> Continuous:
        return Continuous(low=-jnp.inf, high=jnp.inf)

    @cached_property
    def action_space(self) -> Continuous:
        return Continuous(low=-1.0, high=1.0)

    def reset(
        self, key: Key, state: PyTree | None = None, **kwargs: Any
    ) -> tuple[NoStepsState, InfoContainer]:
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


class AlternatingTerminationEnv(Environment):
    """Env that sets `terminated=True` on odd steps, `False` on even steps."""

    @cached_property
    def observation_space(self) -> Continuous:
        return Continuous(low=-jnp.inf, high=jnp.inf, shape=(), dtype=jnp.float32)

    @cached_property
    def action_space(self) -> Continuous:
        return Continuous(low=-1.0, high=1.0)

    def reset(
        self, key: Key, state: PyTree | None = None, **kwargs: Any
    ) -> tuple[StepState, InfoContainer]:
        s = StepState(env_state=jnp.array(0.0), steps=0)
        return s, InfoContainer(
            obs=s.env_state, reward=0.0, terminated=False, truncated=False
        )

    def step(
        self, state: StepState, action: jax.Array
    ) -> tuple[StepState, InfoContainer]:
        ns = StepState(env_state=state.env_state + action, steps=int(state.steps) + 1)
        terminated = (ns.steps % 2) == 1
        return ns, InfoContainer(
            obs=ns.env_state,
            reward=jnp.asarray(action),
            terminated=terminated,
            truncated=False,
        )


# ============================================================================
# Vmap helpers (toy envs and flag propagation)
# ============================================================================


class ScalarToyEnv(Environment):
    @cached_property
    def observation_space(self) -> Continuous:
        return Continuous(low=-jnp.inf, high=jnp.inf)

    @cached_property
    def action_space(self) -> Continuous:
        return Continuous(low=-1.0, high=1.0)

    def reset(
        self, key: Key, state: PyTree | None = None, **kwargs: Any
    ) -> tuple[State, Info]:
        s = jnp.asarray(0.0, dtype=jnp.float32)
        return s, InfoContainer(obs=s, reward=0.0, terminated=False, truncated=False)

    def step(self, state: State, action: jax.Array) -> tuple[State, Info]:
        ns = state + action
        info = InfoContainer(
            obs=ns,
            reward=jnp.asarray(action, dtype=jnp.float32),
            terminated=False,
            truncated=False,
        )
        return ns, info


class VectorToyEnv(Environment):
    """Action/obs are vectors of length D."""

    dim: int

    def __init__(self, dim: int):
        object.__setattr__(self, "dim", int(dim))

    @cached_property
    def observation_space(self) -> Continuous:
        return Continuous.from_shape(low=-jnp.inf, high=jnp.inf, shape=(self.dim,))

    @cached_property
    def action_space(self) -> Continuous:
        return Continuous.from_shape(low=-1.0, high=1.0, shape=(self.dim,))

    def reset(
        self, key: Key, state: PyTree | None = None, **kwargs: Any
    ) -> tuple[State, Info]:
        s = jnp.zeros((self.dim,), dtype=jnp.float32)
        return s, InfoContainer(obs=s, reward=0.0, terminated=False, truncated=False)

    def step(self, state: State, action: jax.Array) -> tuple[State, Info]:
        ns = state + action
        reward = jnp.asarray(action, dtype=jnp.float32).sum()
        info = InfoContainer(obs=ns, reward=reward, terminated=False, truncated=False)
        return ns, info


class FlagDoneEnv(Environment):
    """Batched env whose termination/truncation flags are driven by a provided mask."""

    flags: jax.Array

    def __init__(self, flags: jax.Array):
        object.__setattr__(self, "flags", flags)

    @cached_property
    def observation_space(self) -> Continuous:
        return Continuous(low=-jnp.inf, high=jnp.inf)

    @cached_property
    def action_space(self) -> Continuous:
        return Continuous(low=-1.0, high=1.0)

    def reset(self, key: Key, state: PyTree | None = None, **kwargs: Any):
        z = jnp.array(0.0)
        return z, InfoContainer(obs=z, reward=0.0, terminated=False, truncated=False)

    def step(self, state: State, action: jax.Array):
        t = self.flags.astype(bool)
        info = InfoContainer(
            obs=jnp.asarray(action, dtype=jnp.float32),
            reward=jnp.asarray(action),
            terminated=t,
            truncated=~t,
        )
        return state, info


# ============================================================================
# Vmap over env instances (VmapEnvsWrapper) helper
# ============================================================================


class ParamEnv(Environment):
    """Environment parameterized by an offset; used for vmapping over env instances."""

    offset: jax.Array

    @cached_property
    def observation_space(self) -> Continuous:
        return Continuous.from_shape(low=-jnp.inf, high=jnp.inf, shape=(2,))

    @cached_property
    def action_space(self) -> Continuous:
        return Continuous.from_shape(low=-1.0, high=1.0, shape=(2,))

    def reset(
        self, key: Key, state: PyTree | None = None, **kwargs: Any
    ) -> tuple[State, Info]:
        s = jnp.asarray([self.offset, -self.offset], dtype=jnp.float32)
        return s, InfoContainer(obs=s, reward=0.0, terminated=False, truncated=False)

    def step(self, state: State, action: jax.Array) -> tuple[State, Info]:
        ns = state + action + jnp.asarray(self.offset, dtype=jnp.float32)
        reward = jnp.asarray(action, dtype=jnp.float32).sum()
        info = InfoContainer(obs=ns, reward=reward, terminated=False, truncated=False)
        return ns, info


# ============================================================================
# Observation normalization helpers
# ============================================================================


class VectorObsEnv(Environment):
    """Deterministic env: obs equals current env_state; action adds to state."""

    dim: int

    def __init__(self, dim: int):
        object.__setattr__(self, "dim", int(dim))

    @cached_property
    def observation_space(self) -> Continuous:
        return Continuous.from_shape(low=-jnp.inf, high=jnp.inf, shape=(self.dim,))

    @cached_property
    def action_space(self) -> Continuous:
        return Continuous.from_shape(low=-1.0, high=1.0, shape=(self.dim,))

    def reset(self, key: Key, state: PyTree | None = None, **kwargs: Any):
        s = jnp.linspace(0.0, 1.0, self.dim, dtype=jnp.float32)
        return s, InfoContainer(obs=s, reward=0.0, terminated=False, truncated=False)

    def step(self, state: State, action: jax.Array):
        ns = state + action
        info = InfoContainer(
            obs=ns,
            reward=jnp.asarray(action, dtype=jnp.float32).sum(),
            terminated=False,
            truncated=False,
        )
        return ns, info


class PyTreeObsEnv(Environment):
    """Obs is a pytree of arrays."""

    shapes: dict[str, tuple[int, ...]]

    def __init__(self, shapes: dict[str, tuple[int, ...]]):
        object.__setattr__(self, "shapes", dict(shapes))

    @cached_property
    def observation_space(self) -> PyTreeSpace:
        return PyTreeSpace(
            {
                k: Continuous.from_shape(low=-jnp.inf, high=jnp.inf, shape=v)
                for k, v in self.shapes.items()
            }
        )

    @cached_property
    def action_space(self) -> Continuous:
        return Continuous(low=-1.0, high=1.0)

    def reset(self, key: Key, state: PyTree | None = None, **kwargs: Any):
        obs = {
            k: jnp.arange(jnp.prod(jnp.asarray(v)), dtype=jnp.float32).reshape(v)
            for k, v in self.shapes.items()
        }
        s = obs
        return s, InfoContainer(obs=obs, reward=0.0, terminated=False, truncated=False)

    def step(self, state: State, action: jax.Array):
        ns = state
        return ns, InfoContainer(
            obs=state, reward=float(action), terminated=False, truncated=False
        )


class ConstantObsEnv(Environment):
    value: float
    shape: tuple[int, ...]
    dtype: jnp.dtype = jnp.float32

    def __init__(self, value: float, shape: tuple[int, ...], dtype=jnp.float32):
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", dtype)

    @cached_property
    def observation_space(self) -> Continuous:
        return Continuous.from_shape(low=-jnp.inf, high=jnp.inf, shape=self.shape)

    @cached_property
    def action_space(self) -> Continuous:
        return Continuous(low=-1.0, high=1.0)

    def reset(self, key: Key, state: PyTree | None = None, **kwargs: Any):
        obs = jnp.asarray(self.value, self.dtype) * jnp.ones(self.shape, self.dtype)
        return 0, InfoContainer(obs=obs, reward=0.0, terminated=False, truncated=False)

    def step(self, state: State, action: jax.Array):
        obs = jnp.asarray(self.value, self.dtype) * jnp.ones(self.shape, self.dtype)
        return state, InfoContainer(
            obs=obs, reward=float(action), terminated=False, truncated=False
        )


class IntObsEnv(Environment):
    """Non-floating obs env used to assert normalization wrapper raises."""

    @cached_property
    def observation_space(self) -> Discrete:
        return Discrete(n=5)

    @cached_property
    def action_space(self) -> Discrete:
        return Discrete(n=2)

    def reset(self, key: Key, state: PyTree | None = None, **kwargs: Any):
        s = jnp.array(0, dtype=jnp.int32)
        return s, InfoContainer(obs=s, reward=0.0, terminated=False, truncated=False)

    def step(self, state: State, action: jax.Array):
        return state, InfoContainer(
            obs=state, reward=0.0, terminated=False, truncated=False
        )


class RandomImageEnv(Environment):
    """Random image observation env with PRNG key stored in state."""

    shape: tuple[int, ...]
    dtype: jnp.dtype

    def __init__(self, shape: tuple[int, ...], dtype=jnp.float32):
        object.__setattr__(self, "shape", tuple(int(x) for x in shape))
        object.__setattr__(self, "dtype", dtype)

    @cached_property
    def observation_space(self) -> Continuous:
        return Continuous(
            low=-jnp.inf, high=jnp.inf, shape=self.shape, dtype=jnp.float32
        )

    @cached_property
    def action_space(self) -> Continuous:
        return Continuous(low=-1.0, high=1.0)

    def reset(self, key: Key, state: PyTree | None = None, **kwargs: Any):
        k1, k2 = jax.random.split(key)
        obs = jax.random.normal(k1, self.shape, dtype=jnp.float32)
        return k2, InfoContainer(
            obs=obs.astype(self.dtype), reward=0.0, terminated=False, truncated=False
        )

    def step(self, state: State, action: jax.Array):
        k1, k2 = jax.random.split(state)
        obs = jax.random.normal(k1, self.shape, dtype=jnp.float32)
        return k2, InfoContainer(
            obs=obs.astype(self.dtype), reward=0.0, terminated=False, truncated=False
        )


# ============================================================================
# Wrapper base (`envelope.wrappers.wrapper.Wrapper`) delegation helpers
# ============================================================================


class TestInfo(FrozenPyTreeNode):
    """Simple Info implementation for Wrapper tests (includes `.done`)."""

    obs: jax.Array
    reward: float
    terminated: bool = False
    truncated: bool = False

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


class WrapperSimpleEnv(Environment):
    """Simple environment for testing Wrapper delegation."""

    def reset(
        self, key: Key, state: PyTree | None = None, **kwargs: Any
    ) -> tuple[jax.Array, TestInfo]:
        state = jnp.array(0.0)
        info = TestInfo(obs=state, reward=0.0, terminated=False, truncated=False)
        return state, info

    def step(self, state: jax.Array, action: jax.Array) -> tuple[jax.Array, TestInfo]:
        next_state = state + action
        info = TestInfo(
            obs=next_state, reward=float(action), terminated=False, truncated=False
        )
        return next_state, info

    @cached_property
    def observation_space(self) -> Continuous:
        return Continuous(low=-jnp.inf, high=jnp.inf)

    @cached_property
    def action_space(self) -> Continuous:
        return Continuous(low=-1.0, high=1.0)


class WrapperEnvWithFields(Environment):
    """Environment with custom fields for testing."""

    some_field: int = static_field(default=42)
    another_field: str = static_field(default="test")

    def reset(
        self, key: Key, state: PyTree | None = None, **kwargs: Any
    ) -> tuple[jax.Array, TestInfo]:
        state = jnp.array(0.0)
        info = TestInfo(obs=state, reward=0.0, terminated=False, truncated=False)
        return state, info

    def step(self, state: jax.Array, action: jax.Array) -> tuple[jax.Array, TestInfo]:
        next_state = state + action
        info = TestInfo(
            obs=next_state, reward=float(action), terminated=False, truncated=False
        )
        return next_state, info

    @cached_property
    def observation_space(self) -> Continuous:
        return Continuous(low=-jnp.inf, high=jnp.inf)

    @cached_property
    def action_space(self) -> Continuous:
        return Continuous(low=-1.0, high=1.0)


class WrapperEnvWithMethods(Environment):
    """Environment with custom methods for testing."""

    def reset(
        self, key: Key, state: PyTree | None = None, **kwargs: Any
    ) -> tuple[jax.Array, TestInfo]:
        state = jnp.array(0.0)
        info = TestInfo(obs=state, reward=0.0, terminated=False, truncated=False)
        return state, info

    def step(self, state: jax.Array, action: jax.Array) -> tuple[jax.Array, TestInfo]:
        next_state = state + action
        info = TestInfo(
            obs=next_state, reward=float(action), terminated=False, truncated=False
        )
        return next_state, info

    @cached_property
    def observation_space(self) -> Continuous:
        return Continuous(low=-jnp.inf, high=jnp.inf)

    @cached_property
    def action_space(self) -> Continuous:
        return Continuous(low=-1.0, high=1.0)

    def custom_method(self) -> str:
        """Custom method for testing attribute delegation."""
        return "custom_value"

    @property
    def custom_property(self) -> int:
        """Custom property for testing."""
        return 42


def make_wrapper_env_with_private_attr(private_value: int = 10) -> Environment:
    """Factory to create an env with a private attribute for delegation tests."""

    class EnvWithPrivate(WrapperSimpleEnv):
        _private: int = private_value

    return EnvWithPrivate()


def make_wrapper_discrete_env() -> Environment:
    """Factory for wrapper tests needing discrete spaces."""

    class DiscreteEnv(Environment):
        def reset(self, key: Key, state: PyTree | None = None, **kwargs: Any):
            state = jnp.array(0, dtype=jnp.int32)
            info = TestInfo(obs=state, reward=0.0, terminated=False, truncated=False)
            return state, info

        def step(self, state: jax.Array, action: jax.Array):
            next_state = state + 1
            info = TestInfo(
                obs=next_state, reward=1.0, terminated=False, truncated=False
            )
            return next_state, info

        @cached_property
        def observation_space(self) -> Discrete:
            return Discrete(n=10)

        @cached_property
        def action_space(self) -> Discrete:
            return Discrete(n=5)

    return DiscreteEnv()


def make_wrapper_complex_state_env() -> Environment:
    """Factory for wrapper tests that use a dict-like state."""

    class ComplexStateEnv(Environment):
        def reset(self, key: Key, state: PyTree | None = None, **kwargs: Any):
            st = {
                "position": jnp.array([0.0, 0.0]),
                "velocity": jnp.array([1.0, 1.0]),
            }
            info = TestInfo(
                obs=st["position"], reward=0.0, terminated=False, truncated=False
            )
            return st, info

        def step(self, state: dict, action: jax.Array):
            next_state = {
                "position": state["position"] + state["velocity"],
                "velocity": state["velocity"] + action,
            }
            info = TestInfo(
                obs=next_state["position"],
                reward=1.0,
                terminated=False,
                truncated=False,
            )
            return next_state, info

        @cached_property
        def observation_space(self) -> Continuous:
            return Continuous.from_shape(low=-jnp.inf, high=jnp.inf, shape=(2,))

        @cached_property
        def action_space(self) -> Continuous:
            return Continuous.from_shape(low=-1.0, high=1.0, shape=(2,))

    return ComplexStateEnv()
