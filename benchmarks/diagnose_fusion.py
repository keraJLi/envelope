"""Compare per-iteration cost: scanned loop vs individual steps."""

import sys
import time

sys.path.insert(0, ".")

from flax import nnx

from ppo.ppo import Args, TrainState, train_step

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

steps_per_update = args.num_steps * args.num_envs
num_updates = args.total_timesteps // steps_per_update

# --- Scanned (what benchmark uses) ---
ts1 = TrainState(args)

@nnx.jit
@nnx.scan(in_axes=nnx.Carry, length=num_updates)
def train_loop(ts):
    out_info = train_step(ts)
    return ts, out_info.last_return.mean()

# Warmup
ts1, _ = train_loop(ts1)
_.block_until_ready()

ts1 = TrainState(args)
t0 = time.time()
ts1, returns1 = train_loop(ts1)
returns1.block_until_ready()
t1 = time.time()
scanned_time = t1 - t0
print(f"Scanned ({num_updates} iters): {scanned_time:.3f}s = {scanned_time/num_updates*1000:.2f}ms/iter")

# --- Individual jitted calls ---
ts2 = TrainState(args)
jit_step = nnx.jit(train_step)

# Warmup
info = jit_step(ts2)
info.last_return.mean().block_until_ready()

ts2 = TrainState(args)
t2 = time.time()
for _ in range(num_updates):
    info = jit_step(ts2)
info.last_return.mean().block_until_ready()
t3 = time.time()
loop_time = t3 - t2
print(f"Python loop ({num_updates} iters): {loop_time:.3f}s = {loop_time/num_updates*1000:.2f}ms/iter")

print(f"\nFusion speedup: {loop_time/scanned_time:.2f}x")
print(f"Scanned SPS: {TOTAL_TIMESTEPS/scanned_time:,.0f}")
print(f"Loop SPS: {TOTAL_TIMESTEPS/loop_time:,.0f}")
