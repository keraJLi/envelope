import dataclasses
import time

import jax
import jax.numpy as jnp
import optax
import tyro
from flax import nnx

import envelope
from envelope.typing import PyTree
from ppo_vmap.logger import Logger
from ppo_vmap.networks import DiscretePolicy, GaussianPolicy, ValueFunction


@dataclasses.dataclass(frozen=True)
class Args:
    env_name: str = "gymnax::CartPole-v1"
    total_timesteps: int = 1000000
    policy_lr: float = 0.001
    value_fn_lr: float = 0.001
    epsilon: float = 0.2
    entropy_coef: float = 0.01
    num_envs: int = 10
    pool_size: int = 10
    num_minibatches: int = 5
    num_epochs: int = 4
    num_steps: int = 100
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_observations: bool = False
    seed: int = 0

    # logging
    use_wandb: bool = False
    wandb_entity: str | None = None
    wandb_project: str | None = "envelope-ppo"
    log_every: int = 1

    # parallelism: pmap across devices, vmap within each device
    num_runs: int = 1

    # checkpointing
    num_checkpoints: int = 0


def make_env(args: Args):
    env = envelope.create(args.env_name)
    env = envelope.ContinuousObservationWrapper(env=env)
    env = envelope.FlattenObservationWrapper(env=env)
    env = envelope.FlattenActionWrapper(env=env)
    env = envelope.ClipActionWrapper(env=env)
    env = envelope.EpisodeStatisticsWrapper(env=env)
    vecenv = envelope.PooledInitVmapWrapper(
        env=env, batch_size=args.num_envs, pool_size=args.pool_size
    )
    if args.normalize_observations:
        vecenv = envelope.ObservationNormalizationWrapper(env=vecenv)
    return env, vecenv


class TrainState(nnx.Pytree):
    def __init__(
        self, args: Args, seed: int | None = None, run_idx: int = 0, env_vecenv=None
    ):
        self.args = nnx.static(args)
        self.global_steps = jnp.array(0)
        self.run_idx = jnp.array(run_idx)

        seed = seed if seed is not None else args.seed

        # Initialize environment and rngs
        if env_vecenv is not None:
            env, vecenv = env_vecenv
        else:
            env, vecenv = make_env(args)
        self.vecenv = nnx.data(vecenv)
        self.rngs = nnx.Rngs(seed)

        # Initialize policy and value function
        discrete = isinstance(env.action_space, envelope.Discrete)
        policy_cls = DiscretePolicy if discrete else GaussianPolicy
        self.policy = policy_cls(env.observation_space, env.action_space, self.rngs)
        self.value_fn = ValueFunction(env.observation_space, self.rngs)

        # Initialize optimizers
        self.policy_optimizer = nnx.Optimizer(
            self.policy, optax.adamw(args.policy_lr), wrt=nnx.Param
        )
        self.value_fn_optimizer = nnx.Optimizer(
            self.value_fn, optax.adamw(args.value_fn_lr), wrt=nnx.Param
        )

        # Initialize environment state and info
        env_state, env_info = self.vecenv.init(self.rngs())
        self.env_state = nnx.data(env_state)
        self.env_info = nnx.data(env_info)


def make_train_states(args: Args):
    """Create train states for parallel training.

    Returns (graphdef, batched_state) where batched_state has shape
    (num_devices, runs_per_device, ...) for pmap+vmap composition.
    """
    num_devices = jax.device_count()
    assert args.num_runs % num_devices == 0, (
        f"num_runs ({args.num_runs}) must be divisible by device_count ({num_devices})"
    )

    env_vecenv = make_env(args)
    states = [
        TrainState(args, seed=args.seed + i, run_idx=i, env_vecenv=env_vecenv)
        for i in range(args.num_runs)
    ]

    # Split into graphdef + state arrays, stack, reshape to (num_devices, runs_per_device, ...)
    runs_per_device = args.num_runs // num_devices
    splits = [nnx.split(s) for s in states]
    graphdef = splits[0][0]
    all_states = [s[1] for s in splits]
    batched_state = jax.tree.map(
        lambda *xs: jnp.stack(xs).reshape(num_devices, runs_per_device, *xs[0].shape),
        *all_states,
    )
    return graphdef, batched_state


def shuffle_and_split(data: PyTree, num_minibatches: int, key: jax.Array):
    first_leaf = jax.tree.leaves(data)[0]
    num_steps, num_envs = first_leaf.shape[:2]
    batch_size = num_steps * num_envs
    permutation = jax.random.permutation(key, batch_size)

    def _shuffle_and_split(x):
        x = x.reshape((batch_size, *x.shape[2:]))
        x = jnp.take(x, permutation, axis=0)
        return x.reshape(num_minibatches, -1, *x.shape[1:])

    return jax.tree.map(_shuffle_and_split, data)


def collect_trajectories(ts: TrainState):
    @nnx.scan(in_axes=nnx.Carry, length=ts.args.num_steps)
    def step_env(ts: TrainState):
        obs = ts.env_info.obs
        value = ts.value_fn(obs)

        pi = ts.policy(obs)
        action = pi.sample(seed=ts.rngs())

        env_state, env_info = ts.vecenv.step(ts.env_state, action)
        ts.global_steps += ts.args.num_envs
        ts.env_state = env_state
        ts.env_info = env_info

        out_info = env_info.update(
            obs=obs,
            value=value,
            action=action,
            log_prob=pi.log_prob(action),
            value_next=ts.value_fn(env_info.final.obs),
        )
        return ts, out_info

    ts, out_info = step_env(ts)
    return out_info


def calculate_gae(ts: TrainState, info):
    @nnx.scan(reverse=True)
    def gae_step(carry, transition):
        gae, next_value = carry
        done = transition.terminated | transition.truncated

        next_value = jnp.where(transition.truncated, transition.value_next, next_value)
        next_value = jnp.where(transition.terminated, 0, next_value)

        delta = transition.reward + ts.args.gamma * next_value - transition.value
        gae = delta + ts.args.gamma * ts.args.gae_lambda * (1 - done) * gae
        return (gae, transition.value), gae

    # Get last value for bootstrapping. We can unsqueeze exactly once since obs is flat.
    done = ts.env_info.terminated | ts.env_info.truncated
    last_obs = jnp.where(done[:, None], ts.env_info.final.obs, ts.env_info.obs)
    last_value = ts.value_fn(last_obs)
    init_carry = (jnp.zeros_like(last_value), last_value)
    _, advantages = gae_step(init_carry, info)
    return advantages


def update_policy(ts: TrainState, batch):
    def normalize(x: jax.Array) -> jax.Array:
        return (x - x.mean()) / (x.std() + 1e-8)

    @nnx.value_and_grad(has_aux=True)
    def loss_fn(policy):
        pi = policy(batch.obs)
        log_prob = pi.log_prob(batch.action)
        entropy = pi.entropy().mean()

        ratio = jnp.exp(log_prob - batch.log_prob)
        clip_ratio = jnp.clip(ratio, 1 - ts.args.epsilon, 1 + ts.args.epsilon)
        advantages = normalize(batch.advantages)

        surrogate1 = ratio * advantages
        surrogate2 = clip_ratio * advantages
        policy_loss = -jnp.mean(jnp.minimum(surrogate1, surrogate2))

        loss = policy_loss - ts.args.entropy_coef * entropy
        return loss, (policy_loss, entropy)

    (loss, (policy_loss, entropy)), grads = loss_fn(ts.policy)
    ts.policy_optimizer.update(ts.policy, grads)
    grad_norm = optax.global_norm(grads)
    return {
        "policy_loss": loss,
        "policy_clipped_surrogate_loss": policy_loss,
        "policy_entropy": entropy,
        "policy_grad_norm": grad_norm,
    }


def update_value_fn(ts: TrainState, batch):
    @nnx.value_and_grad
    def loss_fn(value_fn):
        targets = batch.value + batch.advantages
        values = value_fn(batch.obs)
        return 0.5 * jnp.mean((values - targets) ** 2)

    loss, grads = loss_fn(ts.value_fn)
    ts.value_fn_optimizer.update(ts.value_fn, grads)
    grad_norm = optax.global_norm(grads)
    return {"value_loss": loss, "value_grad_norm": grad_norm}


def update_epoch(ts: TrainState, minibatches):
    @nnx.scan
    def update_minibatch(ts: TrainState, batch):
        policy_info = update_policy(ts, batch)
        value_info = update_value_fn(ts, batch)
        return ts, {**policy_info, **value_info}

    _, loss_info = update_minibatch(ts, minibatches)
    return loss_info


def train_step(ts: TrainState):
    # Collect trajectories
    info = collect_trajectories(ts)

    # Compute advantages
    advantages = calculate_gae(ts, info)
    info = info.update(advantages=advantages)

    # Multiple epochs of updates (unrolled since num_epochs is small/static)
    @nnx.scan(in_axes=nnx.Carry, length=ts.args.num_epochs)
    def update_epoch_scan(ts):
        minibatches = shuffle_and_split(info, ts.args.num_minibatches, ts.rngs())
        loss_info = update_epoch(ts, minibatches)
        return ts, loss_info

    ts, loss_infos = update_epoch_scan(ts)
    loss_infos = jax.tree.map(jnp.mean, loss_infos)
    return info.update(**loss_infos)


def make_block_fn(block_size: int, logger: Logger):
    """Create a training block function that runs block_size PPO updates."""

    @nnx.scan(in_axes=nnx.Carry, length=block_size)
    def train_block(ts: TrainState):
        out_info = train_step(ts)
        mean_return = out_info.final.stats.reward.mean()
        mean_episode_length = out_info.final.stats.length.mean()
        jax.debug.callback(
            logger.log,
            ts.global_steps,
            ts.run_idx,
            mean_return,
            mean_episode_length,
            out_info.policy_loss,
            out_info.value_loss,
            out_info.policy_entropy,
        )
        return ts, mean_return

    return train_block


def make_pmap_vmap_step(graphdef, train_block, batched_state):
    vmapped_block = nnx.vmap(train_block)

    @jax.pmap
    def step(state):
        ts = nnx.merge(graphdef, state)
        ts, returns = vmapped_block(ts)
        _, new_state = nnx.split(ts)
        return new_state, returns

    t0 = time.time()
    lowered = step.lower(batched_state)
    lower_time = time.time() - t0

    t0 = time.time()
    compiled = lowered.compile()
    compile_time = time.time() - t0

    return compiled, lower_time, compile_time


def _checkpoint_schedule(num_blocks: int, num_checkpoints: int) -> set[int]:
    if num_checkpoints <= 0:
        return set()
    return {
        round(num_blocks * i / num_checkpoints) for i in range(1, num_checkpoints + 1)
    }


if __name__ == "__main__":
    args = tyro.cli(Args)
    logger = Logger(args)

    # Compute block structure
    steps_per_update = args.num_steps * args.num_envs
    num_updates = args.total_timesteps // steps_per_update
    num_blocks = max(args.num_checkpoints, 1)
    block_size = num_updates // num_blocks
    assert block_size * num_blocks == num_updates, (
        f"num_updates ({num_updates}) must be divisible by num_blocks ({num_blocks})"
    )

    # Create train states: (graphdef, batched_state) with shape
    # (num_devices, runs_per_device, ...)
    graphdef, batched_state = make_train_states(args)

    # Build pmap(vmap(scan)) step function (AOT compiled)
    train_block = make_block_fn(block_size, logger)
    step_fn, lower_time, compile_time = make_pmap_vmap_step(
        graphdef, train_block, batched_state
    )
    logger.log_once({"time/lower": lower_time, "time/compile": compile_time})

    # Determine checkpoint schedule
    checkpoint_at = _checkpoint_schedule(num_blocks, args.num_checkpoints)

    # Run training blocks
    logger.start_time = time.time()
    for block_idx in range(num_blocks):
        batched_state, _ = step_fn(batched_state)
        jax.block_until_ready(batched_state)
        block_num = block_idx + 1
        elapsed = time.time() - logger.start_time
        print(
            f"Block {block_num}/{num_blocks} done "
            f"(update {block_num * block_size}/{num_updates}, {elapsed:.1f}s)"
        )
        if block_num in checkpoint_at:
            global_step = block_num * block_size * steps_per_update
            train_states = nnx.merge(graphdef, batched_state)
            logger.save_checkpoint(global_step, train_states)

    total_time = time.time() - logger.start_time
    logger.log_once({"time/total": total_time})
    print(f"Total training time: {total_time:.2f}s")
    logger.finish()
