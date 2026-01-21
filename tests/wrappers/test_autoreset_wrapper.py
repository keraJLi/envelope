"""Tests for envelope.wrappers.autoreset_wrapper.AutoResetWrapper."""

import jax
import jax.numpy as jnp

from envelope.struct import field
from envelope.wrappers.autoreset_wrapper import AutoResetWrapper
from envelope.wrappers.truncation_wrapper import TruncationWrapper
from envelope.wrappers.vmap_envs_wrapper import VmapEnvsWrapper
from envelope.wrappers.vmap_wrapper import VmapWrapper
from envelope.wrappers.wrapper import WrappedState, Wrapper
from tests.wrappers.helpers import (
    AlternatingTerminationEnv,
    StepCounterEnv,
)

# ============================================================================
# Test Fixtures
# ============================================================================

# ============================================================================
# Tests: Core Functionality
# ============================================================================


class TestAutoResetCoreFunctionality:
    """Test AutoResetWrapper core functionality."""

    def test_reset_splits_key_and_stores_reset_key(self):
        """Verify that reset() splits the key and stores reset_key on the state."""
        env = StepCounterEnv()
        w = AutoResetWrapper(env=env)
        key = jax.random.PRNGKey(42)

        state, info = w.reset(key)

        assert state.reset_key is not None
        assert state.reset_key.shape == (2,)
        # Verify it's different from the input key
        assert not jnp.array_equal(state.reset_key, key)

    def test_step_when_not_done_passes_through(self):
        """Verify that when done=False, the wrapper passes through state/info unchanged."""
        env = StepCounterEnv()
        w = AutoResetWrapper(env=env)
        key = jax.random.PRNGKey(0)

        state, _ = w.reset(key)
        # Take a step - should not reset since never done
        next_state, next_info = w.step(state, jnp.array(0.5))

        # State should have progressed
        assert jnp.allclose(next_state.unwrapped.env_state, jnp.array(0.5))
        assert next_state.unwrapped.steps == 1
        # Info should reflect the step
        assert jnp.allclose(next_info.obs, jnp.array(0.5))
        assert jnp.allclose(next_info.reward, 0.5)
        assert bool(jnp.asarray(next_info.terminated)) is False
        assert bool(jnp.asarray(next_info.truncated)) is False

    def test_step_when_terminated_auto_resets(self):
        """Verify that when info.terminated=True, the wrapper automatically calls reset."""
        env = StepCounterEnv(terminate_after=2)
        w = AutoResetWrapper(env=env)
        key = jax.random.PRNGKey(0)

        state, _ = w.reset(key)
        # Step until termination
        state, _ = w.step(state, jnp.array(0.1))
        state, _ = w.step(state, jnp.array(0.2))  # This should trigger termination

        # Next step should auto-reset
        next_state, next_info = w.step(state, jnp.array(0.3))

        # State should be reset (steps back to 0 or 1 after reset+step)
        # After auto-reset, we take a step, so steps should be 1
        assert next_state.unwrapped.steps == 1
        # Env state should be reset (0.0) plus the action (0.3)
        assert jnp.allclose(next_state.unwrapped.env_state, jnp.array(0.3))
        # Info should be from reset+step, not the terminated step
        assert bool(jnp.asarray(next_info.terminated)) is False

    def test_step_when_truncated_auto_resets(self):
        """Verify that when info.truncated=True, the wrapper automatically calls reset."""
        env = StepCounterEnv(truncate_after=2)
        w = AutoResetWrapper(env=env)
        key = jax.random.PRNGKey(0)

        state, _ = w.reset(key)
        # Step until truncation
        state, _ = w.step(state, jnp.array(0.1))
        state, _ = w.step(state, jnp.array(0.2))  # This should trigger truncation

        # Next step should auto-reset
        next_state, next_info = w.step(state, jnp.array(0.3))

        # State should be reset
        assert next_state.unwrapped.steps == 1
        assert jnp.allclose(next_state.unwrapped.env_state, jnp.array(0.3))
        # Info should be from reset+step
        assert bool(jnp.asarray(next_info.truncated)) is False

    def test_step_when_both_terminated_and_truncated(self):
        """Verify behavior when both terminated and truncated are True."""
        env = StepCounterEnv(both_flags=True)
        w = AutoResetWrapper(env=env)
        key = jax.random.PRNGKey(0)

        state, _ = w.reset(key)
        # First step will be done (both flags True)
        next_state, next_info = w.step(state, jnp.array(0.5))

        # Should auto-reset since done=True
        # After reset, state should be fresh (steps=0, env_state=0.0)
        assert next_state.unwrapped.steps == 0
        assert jnp.allclose(next_state.unwrapped.env_state, jnp.array(0.0))
        assert bool(jnp.asarray(next_info.terminated)) is False
        assert bool(jnp.asarray(next_info.truncated)) is False
        # Reset returns reward 0
        assert jnp.allclose(next_info.reward, 0.0)

    def test_reset_key_usage(self):
        """Verify that the stored reset_key is used for auto-reset."""
        env = StepCounterEnv(terminate_after=1)
        w = AutoResetWrapper(env=env)
        key = jax.random.PRNGKey(42)

        state, _ = w.reset(key)
        stored_key = state.reset_key.copy()

        # Step to trigger termination
        state, _ = w.step(state, jnp.array(0.1))

        # After auto-reset, a new reset_key should be stored
        new_key = state.reset_key
        # The new key should be different (reset generates a new one)
        assert not jnp.array_equal(new_key, stored_key)


# ============================================================================
# Tests: State and Info Propagation
# ============================================================================


class TestAutoResetStateInfoPropagation:
    """Test state and info propagation through auto-reset."""

    def test_state_after_auto_reset_is_fresh(self):
        """Verify that auto-reset returns a fresh state from the underlying environment."""
        env = StepCounterEnv(terminate_after=1)
        w = AutoResetWrapper(env=env)
        key = jax.random.PRNGKey(0)

        state, _ = w.reset(key)
        # Step to trigger termination
        state, _ = w.step(state, jnp.array(0.1))

        # After auto-reset, state should be fresh (env_state back to 0.0, steps=0)
        # The reset happens immediately, so state is from reset
        assert state.unwrapped.steps == 0
        assert jnp.allclose(state.unwrapped.env_state, jnp.array(0.0))

    def test_info_after_auto_reset_is_from_reset(self):
        """Verify that info from the reset call is returned (not the done step's info)."""
        env = StepCounterEnv(terminate_after=1)
        w = AutoResetWrapper(env=env)
        key = jax.random.PRNGKey(0)

        state, _ = w.reset(key)
        # Step to trigger termination - this will auto-reset immediately since done=True
        state, reset_info = w.step(state, jnp.array(0.1))

        # Info should be from reset (not the terminated step)
        assert bool(jnp.asarray(reset_info.terminated)) is False
        assert bool(jnp.asarray(reset_info.truncated)) is False
        # Reset returns reward 0
        assert jnp.allclose(reset_info.reward, 0.0)

    def test_reset_key_preservation(self):
        """Verify that reset_key is properly stored and updated on state."""
        env = StepCounterEnv()
        w = AutoResetWrapper(env=env)
        key = jax.random.PRNGKey(0)

        state, _ = w.reset(key)
        initial_key = state.reset_key

        # Take several steps
        for _ in range(5):
            state, _ = w.step(state, jnp.array(0.1))

        assert jnp.array_equal(state.reset_key, initial_key)

    def test_multiple_consecutive_done_steps(self):
        """Verify behavior when environment is done for multiple consecutive steps."""
        env = StepCounterEnv(always_terminated=True)
        w = AutoResetWrapper(env=env)
        key = jax.random.PRNGKey(0)

        state, _ = w.reset(key)
        # Every step will be done, so should auto-reset each time
        for i in range(3):
            state, info = w.step(state, jnp.array(0.1 * (i + 1)))

            # After auto-reset, info should not be done
            assert bool(jnp.asarray(info.terminated)) is False
            assert bool(jnp.asarray(info.truncated)) is False
            # State should reflect the reset (steps=0, env_state=0.0)
            assert state.unwrapped.steps == 0
            assert jnp.allclose(state.unwrapped.env_state, jnp.array(0.0))
            # Reset returns reward 0
            assert jnp.allclose(info.reward, 0.0)


# ============================================================================
# Tests: Edge Cases
# ============================================================================


class TestAutoResetEdgeCases:
    """Test AutoResetWrapper edge cases."""

    def test_done_on_first_step(self):
        """Test when environment terminates/truncates immediately after reset."""
        env = StepCounterEnv(terminate_after=0)  # Terminates immediately
        w = AutoResetWrapper(env=env)
        key = jax.random.PRNGKey(0)

        state, _ = w.reset(key)
        # First step should terminate
        next_state, next_info = w.step(state, jnp.array(0.1))

        # Should auto-reset
        assert next_state.unwrapped.steps == 0
        assert jnp.allclose(next_state.unwrapped.env_state, jnp.array(0.0))
        assert bool(jnp.asarray(next_info.terminated)) is False
        assert jnp.allclose(next_info.reward, 0.0)

    def test_never_done_long_sequence(self):
        """Test long sequence of steps where environment never terminates."""
        env = StepCounterEnv()
        w = AutoResetWrapper(env=env)
        key = jax.random.PRNGKey(0)

        state, _ = w.reset(key)
        initial_env_state = state.unwrapped.env_state

        # Take many steps
        for i in range(100):
            state, info = w.step(state, jnp.array(0.01))

            # Should never reset
            assert bool(jnp.asarray(info.terminated)) is False
            assert bool(jnp.asarray(info.truncated)) is False

        # State should have accumulated
        expected = initial_env_state + 0.01 * 100
        assert jnp.allclose(state.unwrapped.env_state, expected)

    def test_alternating_done_not_done(self):
        """Test rapid alternation between done and not done states."""
        env = AlternatingTerminationEnv()
        w = AutoResetWrapper(env=env)
        key = jax.random.PRNGKey(0)

        state, _ = w.reset(key)
        # Take several steps
        for i in range(5):
            state, info = w.step(state, jnp.array(0.1))

            # After auto-reset, should not be done
            assert bool(jnp.asarray(info.terminated)) is False

    def test_reset_key_regeneration(self):
        """Verify that each reset generates a new reset_key."""
        env = StepCounterEnv(terminate_after=1)
        w = AutoResetWrapper(env=env)
        key = jax.random.PRNGKey(0)

        state, _ = w.reset(key)
        key1 = state.reset_key

        # Trigger auto-reset
        state, _ = w.step(state, jnp.array(0.1))
        key2 = state.reset_key

        # Keys should be different
        assert not jnp.array_equal(key1, key2)

        # Trigger another auto-reset
        state, _ = w.step(state, jnp.array(0.2))
        key3 = state.reset_key

        # All keys should be different
        assert not jnp.array_equal(key1, key3)
        assert not jnp.array_equal(key2, key3)


# ============================================================================
# Tests: Composability
# ============================================================================


class TestAutoResetComposability:
    """Test AutoResetWrapper composability with other wrappers."""

    def test_state_structure(self):
        """Test autoreset wrapper state structure."""
        env = StepCounterEnv(terminate_after=2)
        w = AutoResetWrapper(env=env)
        key = jax.random.PRNGKey(0)

        state, _ = w.reset(key)
        assert state.inner_state is not None
        assert state.reset_key is not None

        # Step to termination
        state, _ = w.step(state, jnp.array(0.1))
        state, _ = w.step(state, jnp.array(0.2))

        # Should auto-reset
        next_state, next_info = w.step(state, jnp.array(0.3))
        assert bool(jnp.asarray(next_info.terminated)) is False
        assert next_state.unwrapped.steps == 1

    def test_with_truncation_wrapper(self):
        """Test that autoreset works correctly when truncation wrapper sets truncated=True."""
        env = StepCounterEnv()
        w = AutoResetWrapper(env=TruncationWrapper(env=env, max_steps=3))
        key = jax.random.PRNGKey(0)

        state, _ = w.reset(key)
        # Step until truncation
        for _ in range(3):
            state, info = w.step(state, jnp.array(0.1))

        # Next step should trigger truncation and auto-reset
        next_state, next_info = w.step(state, jnp.array(0.1))
        # After auto-reset, should not be truncated
        assert bool(jnp.asarray(next_info.truncated)) is False

    def test_with_vmap_wrapper(self):
        """Test autoreset in batched environments."""
        env = StepCounterEnv(terminate_after=2)
        w = VmapWrapper(env=AutoResetWrapper(env=env), batch_size=3)
        key = jax.random.PRNGKey(0)

        state, info = w.reset(key)
        assert info.obs.shape == (3,)

        # Step until some episodes terminate
        state, _ = w.step(state, jnp.ones(3) * 0.1)
        state, _ = w.step(state, jnp.ones(3) * 0.1)

        # Next step should auto-reset terminated episodes
        next_state, next_info = w.step(state, jnp.ones(3) * 0.1)
        assert next_info.obs.shape == (3,)
        # All should be reset, so none should be terminated
        assert jnp.all(~jnp.asarray(next_info.terminated))

    def test_selective_reset_in_batched_envs(self):
        """Verify that when only some episodes terminate, only those are reset."""

        # Create batched envs with different termination steps
        def make_env(terminate_after):
            return StepCounterEnv(terminate_after=terminate_after)

        termination_steps = jnp.array([2, 3, 4])  # Different for each env in batch
        envs = jax.vmap(make_env)(termination_steps)
        env = AutoResetWrapper(env=envs)
        w = VmapEnvsWrapper(env=env, batch_size=3)
        key = jax.random.PRNGKey(0)

        state, _ = w.reset(key)

        # Step 1: none should terminate
        state, info1 = w.step(state, jnp.ones(3) * 0.1)
        assert jnp.all(~info1.terminated)

        # Step 2: first env (index 0) should terminate (steps=2 >= 2)
        state, info2 = w.step(state, jnp.ones(3) * 0.1)
        # After auto-reset, first env should be reset, others continue
        # The first env terminates and gets reset, so obs should be from reset state (0.0)
        # Others should have obs around 0.2 (0.1 + 0.1)
        assert jnp.allclose(info2.obs[0], 0.0, atol=0.01)
        assert jnp.allclose(info2.obs[1], 0.2, atol=0.01)
        assert jnp.allclose(info2.obs[2], 0.2, atol=0.01)
        # First env should not be terminated (was reset), others not terminated yet
        assert bool(jnp.asarray(info2.terminated[0])) is False
        assert bool(jnp.asarray(info2.terminated[1])) is False
        assert bool(jnp.asarray(info2.terminated[2])) is False

        # Step 3: second env (index 1) should terminate (steps=3 >= 3)
        state, info3 = w.step(state, jnp.ones(3) * 0.1)
        # After auto-reset, second env should be reset
        # First env: 0.0 + 0.1 = 0.1 (reset state + action), Second: reset to 0.0, Third: 0.2 + 0.1 = 0.3
        assert jnp.allclose(info3.obs[1], 0.0, atol=0.01)
        assert bool(jnp.asarray(info3.terminated[1])) is False  # Was reset

    def test_with_vmap_envs_wrapper(self):
        """Test autoreset with VmapEnvsWrapper (batched environment instances)."""

        # Create batched envs with different termination steps
        def make_env(terminate_after):
            return StepCounterEnv(terminate_after=terminate_after)

        termination_steps = jnp.array([2, 3, 4])
        envs = jax.vmap(make_env)(termination_steps)
        w = VmapEnvsWrapper(env=AutoResetWrapper(env=envs), batch_size=3)
        key = jax.random.PRNGKey(0)

        state, _ = w.reset(key)

        # Step 1: none terminate
        state, info1 = w.step(state, jnp.ones(3) * 0.1)
        assert jnp.all(~info1.terminated)

        # Step 2: first env terminates
        state, info2 = w.step(state, jnp.ones(3) * 0.1)
        # First env should be reset (obs ~0.0 from reset state), others continue (obs ~0.2)
        assert jnp.allclose(info2.obs[0], 0.0, atol=0.01)
        assert jnp.allclose(info2.obs[1], 0.2, atol=0.01)
        assert bool(jnp.asarray(info2.terminated[0])) is False

    def test_nested_wrappers(self):
        """Test autoreset with multiple wrapper layers."""
        env = StepCounterEnv(terminate_after=2)
        w = AutoResetWrapper(env=TruncationWrapper(env=env, max_steps=10))
        key = jax.random.PRNGKey(0)

        state, _ = w.reset(key)
        # Step until termination
        state, _ = w.step(state, jnp.array(0.1))
        state, _ = w.step(state, jnp.array(0.2))

        # Should auto-reset
        next_state, next_info = w.step(state, jnp.array(0.3))
        assert bool(jnp.asarray(next_info.terminated)) is False


# ============================================================================
# Tests: JIT Compatibility
# ============================================================================


class TestAutoResetJITCompatibility:
    """Test AutoResetWrapper JIT compatibility."""

    def test_jit_reset(self):
        """Verify that reset can be JIT compiled."""
        env = StepCounterEnv()
        w = AutoResetWrapper(env=env)
        key = jax.random.PRNGKey(0)

        @jax.jit
        def reset_fn(k):
            return w.reset(k)

        state, info = reset_fn(key)
        assert state is not None
        assert info is not None

    def test_jit_step(self):
        """Verify that step (including conditional reset) can be JIT compiled."""
        env = StepCounterEnv(terminate_after=2)
        w = AutoResetWrapper(env=env)
        key = jax.random.PRNGKey(0)

        state, _ = w.reset(key)

        @jax.jit
        def step_fn(s, a):
            return w.step(s, a)

        # Step until termination
        state, _ = step_fn(state, jnp.array(0.1))
        state, _ = step_fn(state, jnp.array(0.2))

        # This should trigger auto-reset under JIT
        next_state, next_info = step_fn(state, jnp.array(0.3))
        assert bool(jnp.asarray(next_info.terminated)) is False

    def test_jit_full_episode(self):
        """Test a full episode loop under JIT."""
        env = StepCounterEnv(terminate_after=3)
        w = AutoResetWrapper(env=env)
        key = jax.random.PRNGKey(0)

        @jax.jit
        def episode_fn(k):
            s, _ = w.reset(k)
            rewards = []
            for _ in range(10):  # More steps than max_steps to trigger resets
                s, info = w.step(s, jnp.array(0.1))
                rewards.append(info.reward)
            return jnp.stack(rewards)

        rewards = episode_fn(key)
        assert rewards.shape == (10,)
        # Should have collected rewards from multiple episodes due to auto-reset
        # With max_steps=3: step1 (0.1), step2 (0.1), step3 (done, reset=0), step4 (0.1), step5 (0.1), step6 (done, reset=0), etc.
        # Pattern: [0.1, 0.1, 0.0, 0.1, 0.1, 0.0, 0.1, 0.1, 0.0, 0.1]
        expected_rewards = jnp.array([0.1, 0.1, 0.0, 0.1, 0.1, 0.0, 0.1, 0.1, 0.0, 0.1])
        assert jnp.allclose(rewards, expected_rewards)


# ============================================================================
# Tests: Regression - State Passing on Auto-Reset
# ============================================================================


def test_auto_reset_passes_state_to_inner_wrapper():
    """Verify that auto-reset passes state down to inner wrappers.

    This is a regression test: AutoResetWrapper should pass the current state
    when calling reset() on auto-reset, so inner wrappers can access state
    fields (like state_state for UED).
    """

    # Create a wrapper that tracks whether it receives state on reset
    class StateTrackingWrapper(Wrapper):
        class TrackingState(WrappedState):
            received_state_on_reset: bool = field(default=False)

        def reset(self, key, state=None, **kwargs):
            inner_state, info = self.env.reset(key, state, **kwargs)
            received = state is not None
            return self.TrackingState(
                inner_state=inner_state,
                received_state_on_reset=received,
            ), info

        def step(self, state, action, **kwargs):
            inner_state, info = self.env.step(state.inner_state, action, **kwargs)
            return state.replace(inner_state=inner_state), info

    env = StepCounterEnv(terminate_after=1)  # Terminates after 1 step
    w = AutoResetWrapper(env=StateTrackingWrapper(env=env))
    key = jax.random.PRNGKey(0)

    state, _ = w.reset(key)
    # First reset: state=None (expected)
    assert state.inner_state.received_state_on_reset is False

    # Step to trigger termination → auto-reset
    state, _ = w.step(state, jnp.array(0.1))

    # After auto-reset: inner wrapper should have received state
    assert bool(state.inner_state.received_state_on_reset) is True
