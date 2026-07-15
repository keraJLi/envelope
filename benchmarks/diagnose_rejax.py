"""Break down time within a single rejax train_iteration."""

import time

import jax
from rejax import PPO

TOTAL_TIMESTEPS = 100_000

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

# Initialize state
key = jax.random.PRNGKey(0)
ts = algo.init_state(key)


@jax.jit
def timed_collect(ts):
    return algo.collect_trajectories(ts)


@jax.jit
def timed_gae(trajectories, last_val):
    return algo.calculate_gae(trajectories, last_val)


@jax.jit
def timed_update(ts, batch):
    return algo.update(ts, batch)


@jax.jit
def timed_train_iteration(ts):
    return algo.train_iteration(ts)


# Warmup
ts2, trajectories = timed_collect(ts)
jax.block_until_ready((ts2, trajectories))

import jax.numpy as jnp
last_val = algo.critic.apply(ts.critic_ts.params, ts.last_obs)
last_val_done = jnp.where(ts.last_done, 0, last_val)
advantages, targets = timed_gae(trajectories, last_val_done)
jax.block_until_ready((advantages, targets))

from rejax.algos.ppo import AdvantageMinibatch
batch = AdvantageMinibatch(trajectories, advantages, targets)
# Shuffle into minibatches for a single update call
rng_mb = jax.random.PRNGKey(42)
minibatches = algo.shuffle_and_split(batch, rng_mb)
first_mb = jax.tree.map(lambda x: x[0], minibatches)
ts_up = timed_update(ts, first_mb)
jax.block_until_ready(ts_up)

ts_iter = timed_train_iteration(ts)
jax.block_until_ready(ts_iter)

print("Warmup done.\n")

N = 20
import statistics


def bench(name, fn):
    times = []
    for _ in range(N):
        t = time.time()
        result = fn()
        jax.block_until_ready(result)
        times.append(time.time() - t)
    mean = statistics.mean(times) * 1000
    std = statistics.stdev(times) * 1000
    print(f"{name}: {mean:.2f}ms ± {std:.2f}ms")
    return mean


t_collect = bench("collect_trajectories", lambda: timed_collect(ts))
t_gae = bench("calculate_gae", lambda: timed_gae(trajectories, last_val_done))
t_update = bench("update (1 minibatch)", lambda: timed_update(ts, first_mb))
t_iter = bench("train_iteration (full)", lambda: timed_train_iteration(ts))

print(f"\nEstimated per train_iteration: {t_collect + t_gae + t_update * algo.num_minibatches * algo.num_epochs:.1f}ms")
print(f"(collect={t_collect:.1f} + gae={t_gae:.1f} + {algo.num_epochs}×{algo.num_minibatches}×update={t_update * algo.num_minibatches * algo.num_epochs:.1f})")
print(f"Actual train_iteration: {t_iter:.1f}ms")
