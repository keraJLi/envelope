"""Profile Rejax PPO — generates a JAX trace for Perfetto/TensorBoard."""

import jax
import jax.numpy as jnp
from rejax import PPO

TOTAL_TIMESTEPS = 100_000  # Shorter run for profiling

algo = PPO.create(
    env="CartPole-v1",
    total_timesteps=TOTAL_TIMESTEPS,
    learning_rate=0.001,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    ent_coef=0.01,
    num_envs=10,
    num_steps=100,
    num_minibatches=5,
    num_epochs=4,
    max_grad_norm=jnp.inf,
    skip_initial_evaluation=True,
    agent_kwargs={"hidden_layer_sizes": (256, 256)},
)

train_fn = jax.jit(algo.train)

# Warmup
key = jax.random.PRNGKey(0)
ts, evaluation = train_fn(key)
jax.block_until_ready((ts, evaluation))
print("Warmup done. Starting profiled run...")

# Profiled run
key = jax.random.PRNGKey(1)
jax.profiler.start_trace("benchmarks/traces/rejax")
ts, evaluation = train_fn(key)
jax.block_until_ready((ts, evaluation))
jax.profiler.stop_trace()
print("Trace saved to benchmarks/traces/rejax/")
print("Open at https://ui.perfetto.dev/")
