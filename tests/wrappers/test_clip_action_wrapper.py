import jax
import jax.numpy as jnp

from envelope.spaces import Continuous, PyTreeSpace
from envelope.wrappers.clip_action_wrapper import ClipActionWrapper, clip_action
from envelope.wrappers.vmap_wrapper import VmapWrapper
from tests.wrappers.helpers import (
    DiscreteStepCounterEnv,
    ScalarToyEnv,
    StepCounterEnv,
    VectorToyEnv,
)


def test_init_reset_delegate_unchanged():
    env = ScalarToyEnv()
    w = ClipActionWrapper(env=env)
    key = jax.random.PRNGKey(0)
    state_w, info_w = w.init(key)
    state_e, info_e = env.init(key)
    assert jnp.allclose(state_w, state_e)
    assert jnp.allclose(info_w.obs, info_e.obs)

    state_w, info_w = w.reset(state_w, key)
    state_e, info_e = env.reset(state_e, key)
    assert jnp.allclose(state_w, state_e)
    assert jnp.allclose(info_w.obs, info_e.obs)
    assert w.observation_space.contains(info_w.obs)


def test_step_clips_continuous_scalar_action():
    env = StepCounterEnv()
    w = ClipActionWrapper(env=env)
    key = jax.random.PRNGKey(0)
    state, _ = w.init(key)
    # Action 5.0 out of bounds [-1, 1] -> clipped to 1.0
    out_of_bounds = jnp.array(5.0)
    state_clipped, info_clipped = w.step(state, out_of_bounds)
    state_expected, info_expected = env.step(state, jnp.array(1.0))
    assert jnp.allclose(state_clipped.env_state, state_expected.env_state)
    assert jnp.allclose(state_clipped.steps, state_expected.steps)
    assert jnp.allclose(info_clipped.obs, info_expected.obs)


def test_step_clips_continuous_vector_action():
    env = VectorToyEnv(dim=3)
    w = ClipActionWrapper(env=env)
    key = jax.random.PRNGKey(0)
    state, _ = w.init(key)
    # [2, -3, 0.5] -> [1, -1, 0.5] for bounds [-1, 1]
    action = jnp.array([2.0, -3.0, 0.5])
    state_w, info_w = w.step(state, action)
    clipped = jnp.array([1.0, -1.0, 0.5])
    state_e, info_e = env.step(state, clipped)
    assert jnp.allclose(state_w, state_e)
    assert jnp.allclose(info_w.obs, info_e.obs)


def test_in_bounds_action_passes_through_unchanged():
    env = ScalarToyEnv()
    w = ClipActionWrapper(env=env)
    key = jax.random.PRNGKey(0)
    state, _ = w.init(key)
    action = jnp.array(0.3)
    state_w, info_w = w.step(state, action)
    state_e, info_e = env.step(state, action)
    assert jnp.allclose(state_w, state_e)
    assert jnp.allclose(info_w.obs, info_e.obs)


def test_clips_discrete_action():
    env = DiscreteStepCounterEnv(action_n=5)
    w = ClipActionWrapper(env=env)
    key = jax.random.PRNGKey(0)
    state, _ = w.init(key)
    # action=10 with Discrete(5) -> valid [0,4], clip to 4
    action = jnp.array(10, dtype=jnp.int32)
    state_w, info_w = w.step(state, action)
    state_e, info_e = env.step(state, jnp.array(4, dtype=jnp.int32))
    assert jnp.allclose(state_w.env_state, state_e.env_state)
    assert jnp.allclose(info_w.obs, info_e.obs)


def test_clips_negative_discrete_action():
    env = DiscreteStepCounterEnv(action_n=5)
    w = ClipActionWrapper(env=env)
    key = jax.random.PRNGKey(0)
    state, _ = w.init(key)
    action = jnp.array(-3, dtype=jnp.int32)
    state_w, info_w = w.step(state, action)
    state_e, info_e = env.step(state, jnp.array(0, dtype=jnp.int32))
    assert jnp.allclose(state_w.env_state, state_e.env_state)
    assert jnp.allclose(info_w.obs, info_e.obs)


def test_action_at_exact_boundary_passes_through():
    env = ScalarToyEnv()
    w = ClipActionWrapper(env=env)
    key = jax.random.PRNGKey(0)
    state, _ = w.init(key)
    for bound in [jnp.array(-1.0), jnp.array(1.0)]:
        state_w, info_w = w.step(state, bound)
        state_e, info_e = env.step(state, bound)
        assert jnp.allclose(state_w, state_e)
        assert jnp.allclose(info_w.obs, info_e.obs)


def test_inf_bounds_env_large_action_passes_through():
    """When action space has inf bounds, clip_action leaves large actions unchanged. Test the function directly (no helper env has inf action bounds)."""
    space = Continuous(low=-jnp.inf, high=jnp.inf)
    action = jnp.array(1000.0)
    result = clip_action(action, space)
    assert jnp.allclose(result, action)


def test_spaces_unchanged():
    env = ScalarToyEnv()
    w = ClipActionWrapper(env=env)
    assert w.observation_space is env.observation_space
    assert w.action_space is env.action_space


def test_observation_space_contains_after_init_and_step():
    env = ScalarToyEnv()
    w = ClipActionWrapper(env=env)
    key = jax.random.PRNGKey(0)
    state, info = w.init(key)
    assert w.observation_space.contains(info.obs)
    state, info = w.step(state, jnp.array(0.5))
    assert w.observation_space.contains(info.obs)


def test_jit_compatibility():
    env = ScalarToyEnv()
    w = ClipActionWrapper(env=env)
    key = jax.random.PRNGKey(0)

    @jax.jit
    def step_jit(s, a):
        return w.step(s, a)

    state, _ = w.init(key)
    action = jnp.array(0.7)
    next_state, info = step_jit(state, action)
    assert jnp.allclose(next_state, state + action)
    assert jnp.allclose(info.obs, state + action)


def test_composability_with_vmap_wrapper():
    batch_size = 2
    env = ScalarToyEnv()
    w = ClipActionWrapper(env=VmapWrapper(env=env, batch_size=batch_size))
    key = jax.random.PRNGKey(0)
    state, _ = w.init(key)
    # Batched out-of-bounds actions [5, -5] -> clipped to [1, -1]
    action = jnp.array([5.0, -5.0])
    next_state, info = w.step(state, action)
    assert jnp.allclose(info.obs, jnp.array([1.0, -1.0]))


def test_clip_action_function_with_pytree_space():
    space = PyTreeSpace(
        {
            "a": Continuous.from_shape(-1.0, 1.0, (2,)),
            "b": Continuous.from_shape(-1.0, 1.0, (3,)),
        }
    )
    action = {
        "a": jnp.array([2.0, -2.0]),
        "b": jnp.array([0.5, 10.0, -10.0]),
    }
    result = clip_action(action, space)
    assert jnp.allclose(result["a"], jnp.array([1.0, -1.0]))
    assert jnp.allclose(result["b"], jnp.array([0.5, 1.0, -1.0]))
