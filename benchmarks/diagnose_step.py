"""Break down time within a single envelope train_step."""

import sys
import time

sys.path.insert(0, ".")

import jax
from flax import nnx

from ppo.ppo import (
    Args,
    TrainState,
    collect_trajectories,
    calculate_gae,
    update_epoch,
    shuffle_and_split,
)

TOTAL_TIMESTEPS = 100_000

args = Args(
    env_name="gymnax::CartPole-v1",
    total_timesteps=TOTAL_TIMESTEPS,
    policy_lr=0.00075,
    value_fn_lr=0.00075,
    epsilon=0.2,
    entropy_coef=0.01,
    num_envs=5,
    num_minibatches=5,
    num_epochs=5,
    num_steps=100,
    gamma=0.99,
    gae_lambda=0.95,
    normalize_observations=False,
    seed=0,
)

ts = TrainState(args)


# Time each component individually
@nnx.jit
def timed_collect(ts):
    return collect_trajectories(ts)


@nnx.jit
def timed_gae(ts, info):
    last_value = ts.value_fn(ts.env_info.obs_true)
    return calculate_gae(ts, info, last_value)


@nnx.jit
def timed_update(ts, info):
    minibatches = shuffle_and_split(info, ts.args.num_minibatches, ts.rngs())
    return update_epoch(ts, minibatches)


# Warmup all
info = timed_collect(ts)
jax.block_until_ready(info)

advantages = timed_gae(ts, info)
jax.block_until_ready(advantages)

info_with_adv = info.update(advantages=advantages)
loss = timed_update(ts, info_with_adv)
jax.block_until_ready(loss)

print("Warmup done.\n")

N = 20


def bench(name, fn):
    times = []
    for _ in range(N):
        t = time.time()
        result = fn()
        jax.block_until_ready(result)
        times.append(time.time() - t)
    import statistics

    mean = statistics.mean(times) * 1000
    std = statistics.stdev(times) * 1000
    print(f"{name}: {mean:.2f}ms ± {std:.2f}ms")
    return mean


t_collect = bench("collect_trajectories", lambda: timed_collect(ts))
t_gae = bench("calculate_gae", lambda: timed_gae(ts, info))
t_update = bench(
    "update_epoch (1 epoch)", lambda: timed_update(ts, info_with_adv)
)

print(f"\nEstimated per train_step: {t_collect + t_gae + t_update * args.num_epochs:.1f}ms")
print(f"(collect={t_collect:.1f} + gae={t_gae:.1f} + {args.num_epochs}×update={t_update * args.num_epochs:.1f})")
