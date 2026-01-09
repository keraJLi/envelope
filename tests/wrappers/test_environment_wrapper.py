"""Tests for jenv.wrappers.environment_wrapper.Wrapper."""

import dataclasses
from functools import cached_property

import jax
import jax.numpy as jnp
import pytest

from jenv.environment import Environment
from jenv.spaces import Continuous, Discrete
from jenv.struct import FrozenPyTreeNode, static_field
from jenv.typing import Key, PyTree
from jenv.wrappers.wrapper import Wrapper

# ============================================================================
# Test Fixtures
# ============================================================================


class Info(FrozenPyTreeNode):
    """Simple StepInfo implementation for testing."""

    obs: jax.Array
    reward: float
    terminated: bool = False
    truncated: bool = False

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


class SimpleEnv(Environment):
    """Simple environment for testing wrappers."""

    def reset(
        self, key: Key, state: PyTree | None = None, **kwargs
    ) -> tuple[jax.Array, Info]:
        state = jnp.array(0.0)
        info = Info(obs=state, reward=0.0, terminated=False, truncated=False)
        return state, info

    def step(self, state: jax.Array, action: jax.Array) -> tuple[jax.Array, Info]:
        next_state = state + action
        info = Info(
            obs=next_state, reward=float(action), terminated=False, truncated=False
        )
        return next_state, info

    @cached_property
    def observation_space(self) -> Continuous:
        return Continuous(low=-jnp.inf, high=jnp.inf)

    @cached_property
    def action_space(self) -> Continuous:
        return Continuous(low=-1.0, high=1.0)


class EnvWithFields(Environment):
    """Environment with custom fields for testing."""

    some_field: int = static_field(default=42)
    another_field: str = static_field(default="test")

    def reset(
        self, key: Key, state: PyTree | None = None, **kwargs
    ) -> tuple[jax.Array, Info]:
        state = jnp.array(0.0)
        info = Info(obs=state, reward=0.0, terminated=False, truncated=False)
        return state, info

    def step(self, state: jax.Array, action: jax.Array) -> tuple[jax.Array, Info]:
        next_state = state + action
        info = Info(
            obs=next_state, reward=float(action), terminated=False, truncated=False
        )
        return next_state, info

    @cached_property
    def observation_space(self) -> Continuous:
        return Continuous(low=-jnp.inf, high=jnp.inf)

    @cached_property
    def action_space(self) -> Continuous:
        return Continuous(low=-1.0, high=1.0)


class EnvWithMethods(Environment):
    """Environment with custom methods for testing."""

    def reset(
        self, key: Key, state: PyTree | None = None, **kwargs
    ) -> tuple[jax.Array, Info]:
        state = jnp.array(0.0)
        info = Info(obs=state, reward=0.0, terminated=False, truncated=False)
        return state, info

    def step(self, state: jax.Array, action: jax.Array) -> tuple[jax.Array, Info]:
        next_state = state + action
        info = Info(
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


# ============================================================================
# Tests: Core Methods
# ============================================================================


class TestWrapperCoreMethods:
    """Test Wrapper core method delegation."""

    def test_reset_delegation(self):
        """Verify reset() delegates to wrapped environment."""
        env = SimpleEnv()
        environment_wrapper = Wrapper(env=env)

        key = jax.random.PRNGKey(0)
        state, info = environment_wrapper.reset(key)

        assert jnp.allclose(state, jnp.array(0.0))
        assert isinstance(info, Info)
        assert info.reward == 0.0

    def test_step_delegation(self):
        """Verify step() delegates to wrapped environment."""
        env = SimpleEnv()
        environment_wrapper = Wrapper(env=env)

        key = jax.random.PRNGKey(0)
        state, info = environment_wrapper.reset(key)
        next_state, next_info = environment_wrapper.step(state, jnp.array(0.5))

        assert jnp.allclose(next_state, jnp.array(0.5))
        assert next_info.reward == 0.5

    def test_reset_return_values(self):
        """Verify reset() returns correct types and values."""
        env = SimpleEnv()
        environment_wrapper = Wrapper(env=env)

        key = jax.random.PRNGKey(0)
        state, info = environment_wrapper.reset(key)

        # Check types
        assert isinstance(state, jax.Array)
        assert isinstance(info, Info)

        # Check values match wrapped env
        env_state, env_info = env.reset(key)
        assert jnp.allclose(state, env_state)
        assert info.reward == env_info.reward

    def test_step_return_values(self):
        """Verify step() returns correct types and values."""
        env = SimpleEnv()
        environment_wrapper = Wrapper(env=env)

        key = jax.random.PRNGKey(0)
        state, _ = environment_wrapper.reset(key)
        next_state, next_info = environment_wrapper.step(state, jnp.array(0.3))

        # Check types
        assert isinstance(next_state, jax.Array)
        assert isinstance(next_info, Info)

        # Check values match wrapped env
        env_state, _ = env.reset(key)
        env_next_state, env_next_info = env.step(env_state, jnp.array(0.3))
        assert jnp.allclose(next_state, env_next_state)
        assert next_info.reward == env_next_info.reward


# ============================================================================
# Tests: Space Properties
# ============================================================================


class TestWrapperSpaceProperties:
    """Test Wrapper space property delegation."""

    def test_observation_space_delegation(self):
        """Verify observation_space delegates correctly."""
        env = SimpleEnv()
        environment_wrapper = Wrapper(env=env)

        assert environment_wrapper.observation_space == env.observation_space
        assert environment_wrapper.observation_space is env.observation_space

    def test_action_space_delegation(self):
        """Verify action_space delegates correctly."""
        env = SimpleEnv()
        environment_wrapper = Wrapper(env=env)

        assert environment_wrapper.action_space == env.action_space
        assert environment_wrapper.action_space is env.action_space

    def test_space_cached_property(self):
        """Verify spaces are cached properties."""
        env = SimpleEnv()
        environment_wrapper = Wrapper(env=env)

        # Access multiple times, should be same object
        obs_space1 = environment_wrapper.observation_space
        obs_space2 = environment_wrapper.observation_space
        assert obs_space1 is obs_space2

        action_space1 = environment_wrapper.action_space
        action_space2 = environment_wrapper.action_space
        assert action_space1 is action_space2

    def test_space_identity(self):
        """Verify space objects maintain identity."""
        env = SimpleEnv()
        environment_wrapper = Wrapper(env=env)

        # Spaces should be the exact same objects as env's spaces
        assert id(environment_wrapper.observation_space) == id(env.observation_space)
        assert id(environment_wrapper.action_space) == id(env.action_space)


# ============================================================================
# Tests: Unwrapping Chain
# ============================================================================


class TestWrapperUnwrapping:
    """Test Wrapper unwrapping chain behavior."""

    def test_unwrapped_single_level(self):
        """Verify unwrapped with one environment_wrapper level."""
        env = SimpleEnv()
        environment_wrapper = Wrapper(env=env)

        assert environment_wrapper.unwrapped is env
        assert environment_wrapper.unwrapped is env.unwrapped

    def test_unwrapped_multiple_levels(self):
        """Verify unwrapped traverses multiple environment_wrapper levels."""
        env = SimpleEnv()
        environment_wrapper1 = Wrapper(env=env)
        environment_wrapper2 = Wrapper(env=environment_wrapper1)
        environment_wrapper3 = Wrapper(env=environment_wrapper2)

        assert environment_wrapper3.unwrapped is env
        assert environment_wrapper2.unwrapped is env
        assert environment_wrapper1.unwrapped is env

    def test_unwrapped_chain_termination(self):
        """Verify unwrapping stops at base environment."""
        env = SimpleEnv()
        environment_wrapper1 = Wrapper(env=env)
        environment_wrapper2 = Wrapper(env=environment_wrapper1)

        unwrapped = environment_wrapper2.unwrapped

        # Should be the base environment, not a environment_wrapper
        assert isinstance(unwrapped, SimpleEnv)
        assert not isinstance(unwrapped, Wrapper)
        assert unwrapped is env

    def test_unwrapped_identity(self):
        """Verify unwrapped returns same object as direct access."""
        env = SimpleEnv()
        environment_wrapper = Wrapper(env=env)

        # unwrapped should be same as env.unwrapped
        assert environment_wrapper.unwrapped is env.unwrapped
        assert environment_wrapper.unwrapped is env


# ============================================================================
# Tests: Attribute Delegation
# ============================================================================


class TestWrapperAttributeDelegation:
    """Test Wrapper attribute delegation via __getattr__."""

    def test_getattr_environment_methods(self):
        """Verify delegation of environment methods."""
        env = EnvWithMethods()
        environment_wrapper = Wrapper(env=env)

        # Should delegate custom methods
        assert environment_wrapper.custom_method() == "custom_value"
        assert hasattr(environment_wrapper, "custom_method")

    def test_getattr_environment_properties(self):
        """Verify delegation of environment properties."""
        env = EnvWithMethods()
        environment_wrapper = Wrapper(env=env)

        # Should delegate custom properties
        assert environment_wrapper.custom_property == 42
        assert hasattr(environment_wrapper, "custom_property")

    def test_getattr_custom_attributes(self):
        """Verify delegation of custom environment attributes."""
        env = EnvWithFields(some_field=100, another_field="hello")
        environment_wrapper = Wrapper(env=env)

        # Should delegate custom fields
        assert environment_wrapper.env.some_field == 100
        assert environment_wrapper.env.another_field == "hello"

    def test_getattr_nested_attributes(self):
        """Verify delegation through nested wrappers."""
        env = EnvWithMethods()
        environment_wrapper1 = Wrapper(env=env)
        environment_wrapper2 = Wrapper(env=environment_wrapper1)

        # Should delegate through nested wrappers
        assert environment_wrapper2.custom_method() == "custom_value"
        assert environment_wrapper2.custom_property == 42

    def test_getattr_missing_attribute(self):
        """Verify AttributeError for non-existent attributes."""
        env = SimpleEnv()
        environment_wrapper = Wrapper(env=env)

        with pytest.raises(AttributeError):
            _ = environment_wrapper.nonexistent_attribute

    def test_getattr_setstate_raises(self):
        """Verify __setstate__ raises AttributeError."""
        env = SimpleEnv()
        environment_wrapper = Wrapper(env=env)

        with pytest.raises(AttributeError):
            _ = environment_wrapper.__setstate__

    def test_getattr_private_attributes(self):
        """Verify behavior with private attributes."""

        # Create env with private attribute
        class EnvWithPrivate(SimpleEnv):
            _private: int = 10

        env = EnvWithPrivate()
        environment_wrapper = Wrapper(env=env)

        # Private attributes should be accessible via __getattr__
        assert environment_wrapper._private == 10


# ============================================================================
# Tests: Nested Wrapper Behavior
# ============================================================================


class TestWrapperNested:
    """Test Wrapper nested environment_wrapper behavior."""

    def test_nested_reset_delegation(self):
        """Verify reset() through nested wrappers."""
        env = SimpleEnv()
        environment_wrapper1 = Wrapper(env=env)
        environment_wrapper2 = Wrapper(env=environment_wrapper1)

        key = jax.random.PRNGKey(0)
        state, info = environment_wrapper2.reset(key)

        assert jnp.allclose(state, jnp.array(0.0))
        assert isinstance(info, Info)

    def test_nested_step_delegation(self):
        """Verify step() through nested wrappers."""
        env = SimpleEnv()
        environment_wrapper1 = Wrapper(env=env)
        environment_wrapper2 = Wrapper(env=environment_wrapper1)

        key = jax.random.PRNGKey(0)
        state, _ = environment_wrapper2.reset(key)
        next_state, next_info = environment_wrapper2.step(state, jnp.array(0.5))

        assert jnp.allclose(next_state, jnp.array(0.5))
        assert next_info.reward == 0.5

    def test_nested_space_access(self):
        """Verify space access through nested wrappers."""
        env = SimpleEnv()
        environment_wrapper1 = Wrapper(env=env)
        environment_wrapper2 = Wrapper(env=environment_wrapper1)

        # All wrappers should return same spaces
        assert environment_wrapper2.observation_space == env.observation_space
        assert environment_wrapper2.action_space == env.action_space
        assert environment_wrapper1.observation_space == env.observation_space
        assert environment_wrapper1.action_space == env.action_space

    def test_nested_unwrapped_chain(self):
        """Verify unwrapped chain with nested wrappers."""
        env = SimpleEnv()
        environment_wrapper1 = Wrapper(env=env)
        environment_wrapper2 = Wrapper(env=environment_wrapper1)
        environment_wrapper3 = Wrapper(env=environment_wrapper2)

        # All should unwrap to same base env
        assert environment_wrapper3.unwrapped is env
        assert environment_wrapper2.unwrapped is env
        assert environment_wrapper1.unwrapped is env

    def test_composition_order_independence(self):
        """Verify environment_wrapper order doesn't break functionality."""
        env = SimpleEnv()
        environment_wrapper1 = Wrapper(env=env)
        environment_wrapper2 = Wrapper(env=environment_wrapper1)

        key = jax.random.PRNGKey(0)

        # Both should work identically
        state1, info1 = environment_wrapper1.reset(key)
        state2, info2 = environment_wrapper2.reset(key)

        assert jnp.allclose(state1, state2)
        assert info1.reward == info2.reward

    def test_composition_state_preservation(self):
        """Verify state preserved through environment_wrapper layers."""
        env = SimpleEnv()
        environment_wrapper1 = Wrapper(env=env)
        environment_wrapper2 = Wrapper(env=environment_wrapper1)

        key = jax.random.PRNGKey(0)
        state, _ = environment_wrapper2.reset(key)
        next_state, next_info = environment_wrapper2.step(state, jnp.array(0.3))

        # State should be correctly propagated
        assert jnp.allclose(next_state, jnp.array(0.3))
        assert jnp.allclose(next_info.reward, 0.3)  # Use allclose for float comparison


# ============================================================================
# Tests: JAX PyTree Compatibility
# ============================================================================


class TestWrapperJAXCompatibility:
    """Test Wrapper JAX PyTree compatibility."""

    def test_jax_tree_flatten(self):
        """Verify JAX tree_flatten works correctly."""
        env = SimpleEnv()
        environment_wrapper = Wrapper(env=env)

        children, aux_data = jax.tree.flatten(environment_wrapper)

        # Wrapper is a pytree node (registered via FrozenPyTreeNode)
        # Children depend on the wrapped env's structure
        # aux_data is a PyTreeDef (from jax.tree_util, which jax.tree uses internally)
        assert aux_data is not None

    def test_jax_tree_unflatten(self):
        """Verify JAX tree_unflatten reconstructs correctly."""
        env = SimpleEnv()
        environment_wrapper = Wrapper(env=env)

        children, aux_data = jax.tree.flatten(environment_wrapper)
        reconstructed = jax.tree.unflatten(aux_data, children)

        assert isinstance(reconstructed, Wrapper)
        assert isinstance(reconstructed.env, SimpleEnv)

    def test_jax_tree_map(self):
        """Verify JAX tree_map operations work."""
        env = SimpleEnv()
        environment_wrapper = Wrapper(env=env)

        # Map a function over the tree
        def double(x):
            if isinstance(x, jax.Array):
                return x * 2
            return x

        result = jax.tree.map(double, environment_wrapper)

        # Should transform arrays in env if any
        assert isinstance(result, Wrapper)
        assert isinstance(result.env, SimpleEnv)

    def test_nested_jax_operations(self):
        """Verify JAX operations with nested wrappers."""
        env = SimpleEnv()
        environment_wrapper1 = Wrapper(env=env)
        environment_wrapper2 = Wrapper(env=environment_wrapper1)

        # Should work with nested wrappers
        children, aux_data = jax.tree.flatten(environment_wrapper2)
        reconstructed = jax.tree.unflatten(aux_data, children)

        assert isinstance(reconstructed, Wrapper)
        assert isinstance(reconstructed.env, Wrapper)
        assert isinstance(reconstructed.env.env, SimpleEnv)


# ============================================================================
# Tests: Edge Cases
# ============================================================================


class TestWrapperEdgeCases:
    """Test Wrapper edge cases and error conditions."""

    def test_wrapper_preserves_dataclass_fields(self):
        """Test that dataclasses.fields() on wrapped env shows unwrapped env's fields."""
        env = EnvWithFields(some_field=100, another_field="hello")
        environment_wrapper = Wrapper(env=env)

        # dataclasses.fields() on environment_wrapper should show environment_wrapper's fields
        wrapper_fields = {f.name for f in dataclasses.fields(environment_wrapper)}
        assert "env" in wrapper_fields  # Wrapper has 'env' field

        # dataclasses.fields() on wrapped env should show unwrapped env's fields
        unwrapped_fields = {f.name for f in dataclasses.fields(environment_wrapper.env)}
        assert "some_field" in unwrapped_fields  # Unwrapped env has 'some_field'
        assert "another_field" in unwrapped_fields

        # Verify we can access the unwrapped env's fields through the environment_wrapper
        assert environment_wrapper.env.some_field == 100
        assert environment_wrapper.env.another_field == "hello"

    def test_wrapper_with_different_space_types(self):
        """Test environment_wrapper with different space types."""

        class DiscreteEnv(Environment):
            def reset(self, key: Key) -> tuple[jax.Array, Info]:
                state = jnp.array(0, dtype=jnp.int32)
                info = Info(obs=state, reward=0.0, terminated=False, truncated=False)
                return state, info

            def step(
                self, state: jax.Array, action: jax.Array
            ) -> tuple[jax.Array, Info]:
                next_state = state + 1
                info = Info(
                    obs=next_state, reward=1.0, terminated=False, truncated=False
                )
                return next_state, info

            @cached_property
            def observation_space(self) -> Discrete:
                return Discrete(n=10)

            @cached_property
            def action_space(self) -> Discrete:
                return Discrete(n=5)

        env = DiscreteEnv()
        environment_wrapper = Wrapper(env=env)

        assert isinstance(environment_wrapper.observation_space, Discrete)
        assert isinstance(environment_wrapper.action_space, Discrete)
        assert environment_wrapper.observation_space.n == 10
        assert environment_wrapper.action_space.n == 5

    def test_wrapper_identity_preservation(self):
        """Verify environment_wrapper preserves environment identity."""
        env = SimpleEnv()
        environment_wrapper = Wrapper(env=env)

        # env attribute should be same reference
        assert environment_wrapper.env is env

        # unwrapped should be same as env.unwrapped
        assert environment_wrapper.unwrapped is env.unwrapped

    def test_wrapper_with_complex_state(self):
        """Test environment_wrapper with complex state structures."""

        class ComplexStateEnv(Environment):
            def reset(self, key: Key, state: PyTree | None = None) -> tuple[dict, Info]:
                state = {
                    "position": jnp.array([0.0, 0.0]),
                    "velocity": jnp.array([1.0, 1.0]),
                }
                info = Info(
                    obs=state["position"], reward=0.0, terminated=False, truncated=False
                )
                return state, info

            def step(self, state: dict, action: jax.Array) -> tuple[dict, Info]:
                next_state = {
                    "position": state["position"] + state["velocity"],
                    "velocity": state["velocity"] + action,
                }
                info = Info(
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

        env = ComplexStateEnv()
        environment_wrapper = Wrapper(env=env)

        key = jax.random.PRNGKey(0)
        state, info = environment_wrapper.reset(key)

        assert isinstance(state, dict)
        assert "position" in state
        assert "velocity" in state
        assert jnp.allclose(state["position"], jnp.array([0.0, 0.0]))

        next_state, next_info = environment_wrapper.step(state, jnp.array([0.1, 0.1]))

        assert isinstance(next_state, dict)
        assert jnp.allclose(next_state["position"], jnp.array([1.0, 1.0]))
