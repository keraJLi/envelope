"""Diagnose envelope PPO overhead sources."""

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

# --- Test 1: Compilation time ---
print("=== Test 1: Compilation vs execution ===")
train_state = TrainState(args)

@nnx.jit
@nnx.scan(in_axes=nnx.Carry, length=num_updates)
def train_loop(train_state):
    out_info = train_step(train_state)
    mean_return = out_info.last_return.mean()
    return train_state, mean_return

t0 = time.time()
train_state, mean_returns = train_loop(train_state)
mean_returns.block_until_ready()
t1 = time.time()
print(f"First call (compile + run): {t1 - t0:.3f}s")

# Reuse same train_state (no recompile)
train_state2 = TrainState(args)
t2 = time.time()
train_state2, mean_returns2 = train_loop(train_state2)
mean_returns2.block_until_ready()
t3 = time.time()
print(f"Second call (same fn, new state): {t3 - t2:.3f}s")

# --- Test 2: Does make_train_fn trigger recompile? ---
print("\n=== Test 2: Recompilation from make_train_fn ===")

def make_train_fn():
    ts = TrainState(args)

    @nnx.jit
    @nnx.scan(in_axes=nnx.Carry, length=num_updates)
    def loop(ts):
        out_info = train_step(ts)
        return ts, out_info.last_return.mean()

    return ts, loop

ts_a, loop_a = make_train_fn()
t4 = time.time()
ts_a, _ = loop_a(ts_a)
_.block_until_ready()
t5 = time.time()
print(f"make_train_fn call 1: {t5 - t4:.3f}s")

ts_b, loop_b = make_train_fn()
t6 = time.time()
ts_b, _ = loop_b(ts_b)
_.block_until_ready()
t7 = time.time()
print(f"make_train_fn call 2: {t7 - t6:.3f}s")

# --- Test 3: Single train_step timing ---
print("\n=== Test 3: Single jitted train_step ===")
train_state3 = TrainState(args)
jit_step = nnx.jit(train_step)

# Warmup
info = jit_step(train_state3)
info.last_return.mean().block_until_ready()

times = []
for i in range(20):
    t = time.time()
    info = jit_step(train_state3)
    info.last_return.mean().block_until_ready()
    times.append(time.time() - t)

import statistics
print(f"Single train_step: mean={statistics.mean(times)*1000:.1f}ms, "
      f"std={statistics.stdev(times)*1000:.1f}ms, "
      f"min={min(times)*1000:.1f}ms")
print(f"Projected for {num_updates} updates: {statistics.mean(times) * num_updates:.3f}s")
print(f"Projected SPS: {TOTAL_TIMESTEPS / (statistics.mean(times) * num_updates):,.0f}")
