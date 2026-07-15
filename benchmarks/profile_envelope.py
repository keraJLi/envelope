"""Profile Envelope PPO — generates a JAX trace for Perfetto/TensorBoard."""

import sys

sys.path.insert(0, ".")

import jax
from flax import nnx

from ppo.ppo import Args, TrainState, train_step

TOTAL_TIMESTEPS = 100_000  # Shorter run for profiling

args = Args(
    env_name="gymnax::CartPole-v1",
    total_timesteps=TOTAL_TIMESTEPS,
    policy_lr=0.001,
    value_fn_lr=0.001,
    epsilon=0.2,
    entropy_coef=0.01,
    num_envs=10,
    num_minibatches=5,
    num_epochs=4,
    num_steps=100,
    gamma=0.99,
    gae_lambda=0.95,
    normalize_observations=False,
    seed=0,
)

steps_per_update = args.num_steps * args.num_envs
num_updates = args.total_timesteps // steps_per_update

train_state = TrainState(args)


@nnx.jit
@nnx.scan(in_axes=nnx.Carry, length=num_updates)
def train_loop(train_state):
    out_info = train_step(train_state)
    mean_return = out_info.last_return.mean()
    return train_state, mean_return


# Warmup
train_state, mean_returns = train_loop(train_state)
mean_returns.block_until_ready()
print("Warmup done. Starting profiled run...")

# Profiled run
train_state = TrainState(args)
jax.profiler.start_trace("benchmarks/traces/envelope")
train_state, mean_returns = train_loop(train_state)
mean_returns.block_until_ready()
jax.profiler.stop_trace()
print("Trace saved to benchmarks/traces/envelope/")
print("Open at https://ui.perfetto.dev/")
