"""Compare XLA program sizes and shapes."""

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

# Lower to HLO and inspect
print("Lowering envelope train_loop...")
lowered = train_loop.lower(ts)
compiled = lowered.compile()

hlo = lowered.as_text()
print(f"HLO text length: {len(hlo)} chars")
print(f"HLO lines: {hlo.count(chr(10))}")

# Count operations
for op in ["while", "scan", "dot_general", "convolution", "reduce"]:
    count = hlo.lower().count(op)
    print(f"  {op}: {count}")

# Compare with rejax
print("\nLowering rejax train...")
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

rejax_fn = jax.jit(algo.train)
key = jax.random.PRNGKey(0)
lowered_r = rejax_fn.lower(key)

hlo_r = lowered_r.as_text()
print(f"HLO text length: {len(hlo_r)} chars")
print(f"HLO lines: {hlo_r.count(chr(10))}")
for op in ["while", "scan", "dot_general", "convolution", "reduce"]:
    count = hlo_r.lower().count(op)
    print(f"  {op}: {count}")
