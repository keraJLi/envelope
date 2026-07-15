"""Precise compilation counting — separate init from train."""

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

# Pre-build everything outside log_compiles
ts1 = TrainState(args)
ts2 = TrainState(args)
ts3 = TrainState(args)

@nnx.jit
@nnx.scan(in_axes=nnx.Carry, length=num_updates)
def train_loop(ts):
    out_info = train_step(ts)
    return ts, out_info.last_return.mean()

# Flush any lazy init
_ = jax.numpy.zeros(1).block_until_ready()

print("=== Train call 1 (first, should compile) ===", flush=True)
with jax.log_compiles():
    ts1, r1 = train_loop(ts1)
    r1.block_until_ready()

print("=== Train call 2 (should be cached) ===", flush=True)
with jax.log_compiles():
    ts2, r2 = train_loop(ts2)
    r2.block_until_ready()

print("=== Train call 3 (should be cached) ===", flush=True)
with jax.log_compiles():
    ts3, r3 = train_loop(ts3)
    r3.block_until_ready()

print("Done.")
