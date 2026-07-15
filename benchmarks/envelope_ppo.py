"""Benchmark: Envelope PPO on CartPole-v1"""

import sys
import time

sys.path.insert(0, ".")

from flax import nnx

from ppo.ppo import Args, TrainState, train_step

TOTAL_TIMESTEPS = 100_000
NUM_RUNS = 3

# From configs/gymnax/cartpole.yaml (network arch differs: envelope uses 256x256/swish)
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

steps_per_update = args.num_steps * args.num_envs
num_updates = args.total_timesteps // steps_per_update


def make_train_fn():
    train_state = TrainState(args)

    @nnx.jit
    @nnx.scan(in_axes=nnx.Carry, length=num_updates)
    def train_loop(train_state):
        out_info = train_step(train_state)
        mean_return = out_info.last_return.mean()
        return train_state, mean_return

    return train_state, train_loop


# Warmup (includes compilation)
train_state, train_loop = make_train_fn()
train_state, mean_returns = train_loop(train_state)
mean_returns.block_until_ready()
print("Warmup done.")

# Timed runs (reuse same train_state to avoid NNX recompilation)
times = []
for i in range(NUM_RUNS):
    start = time.time()
    train_state, mean_returns = train_loop(train_state)
    mean_returns.block_until_ready()
    elapsed = time.time() - start
    times.append(elapsed)
    sps = TOTAL_TIMESTEPS / elapsed
    print(f"Run {i + 1}: {elapsed:.3f}s, SPS={sps:,.0f}")

mean_time = sum(times) / len(times)
mean_sps = TOTAL_TIMESTEPS / mean_time
print(f"\nEnvelope PPO — Mean: {mean_time:.3f}s, SPS={mean_sps:,.0f}")
