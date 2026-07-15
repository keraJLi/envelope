"""Check if nnx.jit+scan recompiles on second call."""

import sys

sys.path.insert(0, ".")

import jax
from flax import nnx

from ppo.ppo import Args, TrainState, train_step

args = Args(
    env_name="gymnax::CartPole-v1",
    total_timesteps=100_000,
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

num_updates = args.total_timesteps // (args.num_steps * args.num_envs)
ts = TrainState(args)


@nnx.jit
@nnx.scan(in_axes=nnx.Carry, length=num_updates)
def train_loop(ts):
    out_info = train_step(ts)
    return ts, out_info.last_return.mean()


# Flush init-time compilations
_ = jax.numpy.zeros(1).block_until_ready()

print("=== First call ===", flush=True)
with jax.log_compiles():
    ts, returns = train_loop(ts)
    returns.block_until_ready()

print("=== Second call ===", flush=True)
ts2 = TrainState(args)
with jax.log_compiles():
    ts2, returns2 = train_loop(ts2)
    returns2.block_until_ready()

print("Done.")
