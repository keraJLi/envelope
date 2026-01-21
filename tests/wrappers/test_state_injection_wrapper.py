"""Tests for envelope.wrappers.state_injection_wrapper.StateInjectionWrapper."""

import jax
import jax.numpy as jnp
import pytest

from envelope.wrappers.autoreset_wrapper import AutoResetWrapper
from envelope.wrappers.state_injection_wrapper import StateInjectionWrapper
from tests.wrappers.helpers import StepCounterEnv, StepState

# ============================================================================
# Tests: Core Functionality
# ============================================================================


class TestStateInjectionCoreFunctionality:
    """Test StateInjectionWrapper core functionality."""

    def test_reset_without_state_delegates_to_inner_env(self):
        """Verify that reset() without prior state calls inner env reset."""
        env = StepCounterEnv()
        w = StateInjectionWrapper(env=env)
        key = jax.random.PRNGKey(0)

        state, info = w.reset(key)

        # Should get fresh state from inner env (StepCounterEnv starts at 0)
        assert state.inner_state is not None
        assert jnp.allclose(state.inner_state.env_state, jnp.array(0.0))
        assert state.inner_state.steps == 0
        assert jnp.allclose(info.obs, jnp.array(0.0))
        # reset_state is None until set_reset_state is called
        assert state.reset_state is None

    def test_set_reset_state_updates_state(self):
        """Verify that set_reset_state() updates the injected state."""
        env = StepCounterEnv()
        w = StateInjectionWrapper(env=env)
        key = jax.random.PRNGKey(0)

        # Get initial state
        state, _ = w.reset(key)

        # Set a custom reset state
        custom_state = StepState(env_state=jnp.array(42.0), steps=0)
        custom_obs = jnp.array(42.0)
        state = w.set_reset_state(state, custom_state, custom_obs)

        # Verify the state was updated
        assert jnp.allclose(state.reset_state.env_state, jnp.array(42.0))
        assert jnp.allclose(state.reset_obs, custom_obs)
        # inner_state should also be updated to match
        assert jnp.allclose(state.inner_state.env_state, jnp.array(42.0))

    def test_subsequent_reset_preserves_injected_state(self):
        """Verify that reset with existing state preserves the injected state."""
        env = StepCounterEnv()
        w = StateInjectionWrapper(env=env)
        key = jax.random.PRNGKey(0)

        # Get initial state and set custom reset state
        state, _ = w.reset(key)
        custom_state = StepState(env_state=jnp.array(42.0), steps=0)
        custom_obs = jnp.array(42.0)
        state = w.set_reset_state(state, custom_state, custom_obs)

        # Reset again, passing the current state (simulates auto-reset)
        key2 = jax.random.PRNGKey(1)
        state2, info2 = w.reset(key2, state)

        # Should preserve the injected state
        assert jnp.allclose(state2.reset_state.env_state, jnp.array(42.0))
        assert jnp.allclose(info2.obs, custom_obs)

    def test_set_reset_state_overrides_existing(self):
        """Verify that set_reset_state() overrides existing injected state."""
        env = StepCounterEnv()
        w = StateInjectionWrapper(env=env)
        key = jax.random.PRNGKey(0)

        # Get initial state and set first custom state
        state, _ = w.reset(key)
        state = w.set_reset_state(
            state,
            StepState(env_state=jnp.array(1.0), steps=0),
            jnp.array(1.0),
        )

        # Set new reset state
        state = w.set_reset_state(
            state,
            StepState(env_state=jnp.array(99.0), steps=0),
            jnp.array(99.0),
        )

        # Should have new injected state
        assert jnp.allclose(state.reset_state.env_state, jnp.array(99.0))
        assert jnp.allclose(state.reset_obs, jnp.array(99.0))

    def test_step_updates_inner_state_but_preserves_reset_state(self):
        """Verify that step updates inner_state but keeps reset_state unchanged."""
        env = StepCounterEnv()
        w = StateInjectionWrapper(env=env)
        key = jax.random.PRNGKey(0)

        # Get initial state and set custom reset state
        state, _ = w.reset(key)
        custom_state = StepState(env_state=jnp.array(0.0), steps=0)
        custom_obs = jnp.array(0.0)
        state = w.set_reset_state(state, custom_state, custom_obs)

        # Take a step
        state, info = w.step(state, jnp.array(0.5))

        # inner_state should have progressed
        assert jnp.allclose(state.inner_state.env_state, jnp.array(0.5))
        assert state.inner_state.steps == 1
        # reset_state should be unchanged
        assert jnp.allclose(state.reset_state.env_state, jnp.array(0.0))
        assert state.reset_state.steps == 0

    def test_reset_with_state_but_no_reset_state_does_normal_reset(self):
        """Verify that reset with state but reset_state=None does normal reset."""
        env = StepCounterEnv()
        w = StateInjectionWrapper(env=env)
        key = jax.random.PRNGKey(0)

        # Get initial state and take a step so inner_state differs from fresh reset
        state, _ = w.reset(key)
        state, _ = w.step(state, jnp.array(5.0))
        assert jnp.allclose(state.inner_state.env_state, jnp.array(5.0))

        # Reset with this state (no reset_state set) - should delegate to inner env
        key2 = jax.random.PRNGKey(1)
        state2, info2 = w.reset(key2, state)

        # Should have done a normal reset - inner_state is fresh from env
        assert jnp.allclose(state2.inner_state.env_state, jnp.array(0.0))
        assert state2.inner_state.steps == 0
        assert jnp.allclose(info2.obs, jnp.array(0.0))
        # reset_state stays None
        assert state2.reset_state is None

    def test_set_reset_state_raises_on_invalid_state(self):
        """Verify that set_reset_state raises when InjectedState not found."""
        env = StepCounterEnv()
        w = StateInjectionWrapper(env=env)

        # Create a state that doesn't contain InjectedState
        invalid_state = StepState(env_state=jnp.array(0.0), steps=0)

        with pytest.raises(ValueError, match="Could not find InjectedState"):
            w.set_reset_state(invalid_state, invalid_state, jnp.array(0.0))

    def test_reset_raises_on_partial_reset_state(self):
        """Verify that reset raises when only one of reset_state/reset_obs is set."""
        env = StepCounterEnv()
        w = StateInjectionWrapper(env=env)
        key = jax.random.PRNGKey(0)

        # Create state with only reset_state set (not reset_obs)
        state_with_only_reset_state = w.InjectedState(
            inner_state=StepState(env_state=jnp.array(0.0), steps=0),
            reset_state=StepState(env_state=jnp.array(42.0), steps=0),
            reset_obs=None,
        )

        with pytest.raises(ValueError, match="must set both"):
            w.reset(key, state_with_only_reset_state)

        # Create state with only reset_obs set (not reset_state)
        state_with_only_reset_obs = w.InjectedState(
            inner_state=StepState(env_state=jnp.array(0.0), steps=0),
            reset_state=None,
            reset_obs=jnp.array(42.0),
        )

        with pytest.raises(ValueError, match="must set both"):
            w.reset(key, state_with_only_reset_obs)


# ============================================================================
# Tests: Composability with AutoResetWrapper
# ============================================================================


class TestStateInjectionWithAutoReset:
    """Test StateInjectionWrapper composability with AutoResetWrapper."""

    def test_auto_reset_returns_to_same_state(self):
        """Verify that auto-reset returns to the same injected state."""
        env = StepCounterEnv(terminate_after=2)
        inner_w = StateInjectionWrapper(env=env)
        w = AutoResetWrapper(env=inner_w)
        key = jax.random.PRNGKey(0)

        # Reset and set a custom reset state
        state, _ = w.reset(key)
        reset_state = StepState(env_state=jnp.array(42.0), steps=0)
        reset_obs = jnp.array(42.0)
        state = inner_w.set_reset_state(state, reset_state, reset_obs)

        # Step until termination triggers auto-reset
        state, _ = w.step(state, jnp.array(0.1))
        state, info = w.step(state, jnp.array(0.2))  # Terminates

        # After auto-reset, should be back to same injected state
        assert jnp.allclose(state.inner_state.reset_state.env_state, jnp.array(42.0))
        assert jnp.allclose(info.obs, reset_obs)

    def test_set_reset_state_with_autoreset_wrapper(self):
        """Verify that set_reset_state works through AutoResetWrapper."""
        env = StepCounterEnv(terminate_after=1)
        inner_w = StateInjectionWrapper(env=env)
        w = AutoResetWrapper(env=inner_w)
        key = jax.random.PRNGKey(0)

        # Get initial state
        state, _ = w.reset(key)

        # Set a custom reset state - just pass the outermost state
        custom_state = StepState(env_state=jnp.array(100.0), steps=0)
        custom_obs = jnp.array(100.0)
        state = inner_w.set_reset_state(state, custom_state, custom_obs)

        # Step to trigger termination → auto-reset
        state, info = w.step(state, jnp.array(0.1))

        # After auto-reset, should return to custom injected state
        assert jnp.allclose(state.inner_state.reset_state.env_state, jnp.array(100.0))
        assert jnp.allclose(info.obs, custom_obs)

    def test_multiple_auto_resets_preserve_state(self):
        """Verify that injected state persists through multiple auto-resets."""
        env = StepCounterEnv(terminate_after=1)
        inner_w = StateInjectionWrapper(env=env)
        w = AutoResetWrapper(env=inner_w)
        key = jax.random.PRNGKey(0)

        # Get initial state and set custom reset state
        state, _ = w.reset(key)
        custom_state = StepState(env_state=jnp.array(50.0), steps=0)
        custom_obs = jnp.array(50.0)
        state = inner_w.set_reset_state(state, custom_state, custom_obs)

        # Trigger multiple auto-resets
        for _ in range(5):
            state, info = w.step(state, jnp.array(0.1))
            # Each auto-reset should return to the same injected state
            assert jnp.allclose(
                state.inner_state.reset_state.env_state, jnp.array(50.0)
            )
            assert jnp.allclose(info.obs, custom_obs)

    def test_ued_style_outer_loop(self):
        """Test the UED-style outer loop pattern with set_reset_state."""
        env = StepCounterEnv(terminate_after=2)
        inner_w = StateInjectionWrapper(env=env)
        w = AutoResetWrapper(env=inner_w)
        key = jax.random.PRNGKey(0)

        # Initial reset
        state, _ = w.reset(key)

        # Simulate UED outer loop
        for outer_iter in range(3):
            # Sample a new "task" (starting state)
            task_value = jnp.array(float(outer_iter * 10))
            task_state = StepState(env_state=task_value, steps=0)
            task_obs = task_value

            # Set the reset state for this outer iteration
            # User just passes the outermost state
            state = inner_w.set_reset_state(state, task_state, task_obs)

            # Inner loop with auto-resets
            for inner_step in range(4):  # More than terminate_after to trigger resets
                state, info = w.step(state, jnp.array(0.1))
                # After auto-reset, obs should be task_obs
                # (on non-reset steps, obs will be different)

            # After inner loop, the reset_state should still be our task
            assert jnp.allclose(state.inner_state.reset_state.env_state, task_value)


# ============================================================================
# Tests: JIT Compatibility
# ============================================================================


class TestStateInjectionJITCompatibility:
    """Test StateInjectionWrapper JIT compatibility."""

    def test_jit_reset_and_step(self):
        """Verify that reset and step can be JIT compiled."""
        env = StepCounterEnv(terminate_after=2)
        w = AutoResetWrapper(env=StateInjectionWrapper(env=env))
        key = jax.random.PRNGKey(0)

        @jax.jit
        def run_episode(k):
            s, _ = w.reset(k)
            for _ in range(5):  # More steps than terminate_after to trigger resets
                s, info = w.step(s, jnp.array(0.1))
            return s, info

        state, info = run_episode(key)
        assert state is not None
        assert info is not None

    def test_jit_set_reset_state(self):
        """Verify JIT works with set_reset_state."""
        env = StepCounterEnv(terminate_after=1)
        inner_w = StateInjectionWrapper(env=env)
        w = AutoResetWrapper(env=inner_w)

        @jax.jit
        def run_with_state(k, reset_state, reset_obs):
            s, _ = w.reset(k)
            # Set the reset state - just pass the outermost state
            s = inner_w.set_reset_state(s, reset_state, reset_obs)
            for _ in range(3):
                s, info = w.step(s, jnp.array(0.1))
            return s, info

        key = jax.random.PRNGKey(0)
        custom_state = StepState(env_state=jnp.array(77.0), steps=0)
        custom_obs = jnp.array(77.0)

        state, info = run_with_state(key, custom_state, custom_obs)

        # Injected state should persist through auto-resets
        assert jnp.allclose(state.inner_state.reset_state.env_state, jnp.array(77.0))
        assert jnp.allclose(info.obs, custom_obs)

    def test_jit_ued_outer_loop(self):
        """Verify JIT works with the UED outer loop pattern."""
        env = StepCounterEnv(terminate_after=1)
        inner_w = StateInjectionWrapper(env=env)
        w = AutoResetWrapper(env=inner_w)

        @jax.jit
        def outer_iteration(state, task_value):
            task_state = StepState(env_state=task_value, steps=0)
            task_obs = task_value

            # Set reset state for this task - pass outermost state
            state = inner_w.set_reset_state(state, task_state, task_obs)

            # Run inner loop
            for _ in range(3):
                state, info = w.step(state, jnp.array(0.1))

            return state, info

        key = jax.random.PRNGKey(0)
        state, _ = w.reset(key)

        # Run multiple outer iterations
        for i in range(3):
            task_value = jnp.array(float(i * 10))
            state, info = outer_iteration(state, task_value)
            assert jnp.allclose(state.inner_state.reset_state.env_state, task_value)
