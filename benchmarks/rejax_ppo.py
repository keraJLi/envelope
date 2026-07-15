"""Benchmark: Rejax PPO on CartPole-v1"""

import time

import jax
from rejax import PPO

TOTAL_TIMESTEPS = 100_000
NUM_RUNS = 3

# From configs/gymnax/cartpole.yaml
algo = PPO.create(
    env="CartPole-v1",
    total_timesteps=TOTAL_TIMESTEPS,
    learning_rate=0.00075,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    num_envs=5,
    num_steps=100,
    num_minibatches=5,
    num_epochs=5,
    max_grad_norm=0.5,
    skip_initial_evaluation=True,
    eval_freq=TOTAL_TIMESTEPS,
    agent_kwargs={"activation": "tanh"},
)

train_fn = jax.jit(algo.train)

# Warmup (includes compilation)
key = jax.random.PRNGKey(0)
ts, evaluation = train_fn(key)
jax.block_until_ready((ts, evaluation))
print("Warmup done.")

# Timed runs
times = []
for i in range(NUM_RUNS):
    key = jax.random.PRNGKey(i + 1)
    start = time.time()
    ts, evaluation = train_fn(key)
    jax.block_until_ready((ts, evaluation))
    elapsed = time.time() - start
    times.append(elapsed)
    sps = TOTAL_TIMESTEPS / elapsed
    print(f"Run {i + 1}: {elapsed:.3f}s, SPS={sps:,.0f}")

mean_time = sum(times) / len(times)
mean_sps = TOTAL_TIMESTEPS / mean_time
print(f"\nRejax PPO — Mean: {mean_time:.3f}s, SPS={mean_sps:,.0f}")
