"""Profile per-component timing: envelope vs rejax with matched 64x64 tanh networks."""

import sys
sys.path.insert(0, ".")

import time
import jax
from flax import nnx

from ppo.ppo import Args, TrainState, collect_trajectories, calculate_gae, update_epoch, shuffle_and_split

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

ts = TrainState(args)

# JIT individual components
jit_collect = nnx.jit(collect_trajectories)
jit_gae = nnx.jit(calculate_gae)
jit_update_epoch = nnx.jit(update_epoch)

# Warmup all
info = jit_collect(ts)
last_value = ts.value_fn(ts.env_info.obs_true)
advantages = jit_gae(ts, info, last_value)
info_with_adv = info.update(advantages=advantages)
minibatches = shuffle_and_split(info_with_adv, args.num_minibatches, ts.rngs())
jit_update_epoch(ts, minibatches)
jax.block_until_ready(jax.tree.leaves(nnx.state(ts)))

N = 20

# Time collect
t0 = time.time()
for _ in range(N):
    info = jit_collect(ts)
    jax.block_until_ready(jax.tree.leaves(info))
collect_time = (time.time() - t0) / N

# Time GAE
info = jit_collect(ts)
jax.block_until_ready(jax.tree.leaves(info))
t0 = time.time()
for _ in range(N):
    last_value = ts.value_fn(ts.env_info.obs_true)
    advantages = jit_gae(ts, info, last_value)
    jax.block_until_ready(jax.tree.leaves(advantages))
gae_time = (time.time() - t0) / N

# Time update epoch (single epoch)
info_with_adv = info.update(advantages=advantages)
t0 = time.time()
for _ in range(N):
    minibatches = shuffle_and_split(info_with_adv, args.num_minibatches, ts.rngs())
    loss_info = jit_update_epoch(ts, minibatches)
    jax.block_until_ready(jax.tree.leaves(loss_info))
update_time = (time.time() - t0) / N

print("=== Envelope (64x64 tanh) per train_step ===")
print(f"  collect:      {collect_time*1000:.2f} ms")
print(f"  GAE:          {gae_time*1000:.2f} ms")
print(f"  update_epoch: {update_time*1000:.2f} ms  (x{args.num_epochs} = {update_time*args.num_epochs*1000:.2f} ms)")
total = collect_time + gae_time + update_time * args.num_epochs
print(f"  TOTAL:        {total*1000:.2f} ms")
num_updates = args.total_timesteps // (args.num_steps * args.num_envs)
print(f"  Estimated full train: {total * num_updates:.3f}s")
print(f"  Estimated SPS: {args.total_timesteps / (total * num_updates):,.0f}")

# Now time the full fused train_loop
@nnx.jit
@nnx.scan(in_axes=nnx.Carry, length=num_updates)
def train_loop(ts):
    from ppo.ppo import train_step
    out_info = train_step(ts)
    return ts, out_info.last_return.mean()

ts2 = TrainState(args)
train_loop(ts2)
jax.block_until_ready(jax.tree.leaves(nnx.state(ts2)))

ts3 = TrainState(args)
t0 = time.time()
ts3, returns = train_loop(ts3)
returns.block_until_ready()
fused_time = time.time() - t0
print(f"\n  Fused train_loop: {fused_time:.3f}s, SPS={args.total_timesteps / fused_time:,.0f}")
