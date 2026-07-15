"""Profile per-component timing for rejax PPO."""

import sys
sys.path.insert(0, ".")

import time
import jax
import jax.numpy as jnp
from rejax import PPO

algo = PPO.create(
    env="CartPole-v1",
    total_timesteps=100_000,
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
    eval_freq=100_000,
    agent_kwargs={"activation": "tanh"},
)

# Initialize state
key = jax.random.PRNGKey(0)
ts = algo.init_state(key)

# JIT individual components
jit_collect = jax.jit(algo.collect_trajectories)
jit_gae = jax.jit(algo.calculate_gae)
jit_update = jax.jit(algo.update)

# Warmup
ts, trajectories = jit_collect(ts)
last_val = algo.critic.apply(ts.critic_ts.params, ts.last_obs)
last_val = jnp.where(ts.last_done, 0, last_val)
advantages, targets = jit_gae(trajectories, last_val)
jax.block_until_ready((advantages, targets))

from rejax.algos.ppo import AdvantageMinibatch
batch = AdvantageMinibatch(trajectories, advantages, targets)
minibatches = algo.shuffle_and_split(batch, key)
jit_update(ts, jax.tree.map(lambda x: x[0], minibatches))

N = 20

# Time collect
t0 = time.time()
for _ in range(N):
    ts, trajectories = jit_collect(ts)
    jax.block_until_ready(jax.tree.leaves(trajectories))
collect_time = (time.time() - t0) / N

# Time GAE
ts, trajectories = jit_collect(ts)
jax.block_until_ready(jax.tree.leaves(trajectories))
t0 = time.time()
for _ in range(N):
    last_val = algo.critic.apply(ts.critic_ts.params, ts.last_obs)
    last_val = jnp.where(ts.last_done, 0, last_val)
    advantages, targets = jit_gae(trajectories, last_val)
    jax.block_until_ready((advantages, targets))
gae_time = (time.time() - t0) / N

# Time single update (actor + critic)
batch = AdvantageMinibatch(trajectories, advantages, targets)
minibatches = algo.shuffle_and_split(batch, key)
single_mb = jax.tree.map(lambda x: x[0], minibatches)
t0 = time.time()
for _ in range(N):
    ts_out = jit_update(ts, single_mb)
    jax.block_until_ready(jax.tree.leaves(ts_out))
update_time = (time.time() - t0) / N

print("=== Rejax (64x64 tanh) per train_iteration ===")
print(f"  collect:      {collect_time*1000:.2f} ms")
print(f"  GAE:          {gae_time*1000:.2f} ms")
print(f"  update (1mb): {update_time*1000:.2f} ms  (x{algo.num_minibatches}mb x{algo.num_epochs}ep = {update_time*algo.num_minibatches*algo.num_epochs*1000:.2f} ms)")
total = collect_time + gae_time + update_time * algo.num_minibatches * algo.num_epochs
print(f"  TOTAL:        {total*1000:.2f} ms")
num_updates = algo.total_timesteps // (algo.num_steps * algo.num_envs)
print(f"  Estimated full train: {total * num_updates:.3f}s")
print(f"  Estimated SPS: {algo.total_timesteps / (total * num_updates):,.0f}")

# Full fused train
train_fn = jax.jit(algo.train)
key = jax.random.PRNGKey(0)
train_fn(key)  # warmup
t0 = time.time()
result = train_fn(jax.random.PRNGKey(1))
jax.block_until_ready(jax.tree.leaves(result))
fused_time = time.time() - t0
print(f"\n  Fused train(): {fused_time:.3f}s, SPS={algo.total_timesteps / fused_time:,.0f}")
