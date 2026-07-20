import dataclasses
import time

import jax
import jax.numpy as jnp
import optax
import tyro
import wandb
from flax import nnx

import envelope
from envelope.typing import PyTree
from examples.ppo.networks import DiscretePolicy, GaussianPolicy, ValueFunction

WEIGHT_DECAY = 1e-4


@dataclasses.dataclass(frozen=True)
class Args:
    env_name: str = "gymnax::CartPole-v1"
    total_timesteps: int = 1000000
    policy_lr: float = 0.0003
    value_fn_lr: float = 0.0001
    epsilon: float = 0.2
    entropy_coef: float = 0.01
    num_envs: int = 1024
    pool_size: int = 10
    num_minibatches: int = 8
    num_epochs: int = 4
    num_steps: int = 128
    updates_per_block: int = 1
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_observations: bool = True
    seed: int = 0

    # wandb
    use_wandb: bool = False
    wandb_entity: str | None = "flair"
    wandb_project: str | None = "envelope-ppo"
    wandb_log_every: int = 1


def get_num_updates(args: Args) -> int:
    steps_per_update = args.num_steps * args.num_envs
    if steps_per_update <= 0:
        raise ValueError("num_steps and num_envs must be positive")
    if args.num_epochs <= 0 or args.num_minibatches <= 0:
        raise ValueError("num_epochs and num_minibatches must be positive")
    if args.total_timesteps < steps_per_update:
        raise ValueError("total_timesteps must cover at least one PPO update")
    if steps_per_update % args.num_minibatches:
        raise ValueError("rollout batch size must be divisible by num_minibatches")
    if args.updates_per_block <= 0:
        raise ValueError("updates_per_block must be positive")
    return args.total_timesteps // steps_per_update


def make_lr_schedule(learning_rate: float, args: Args):
    optimizer_steps = get_num_updates(args) * args.num_epochs * args.num_minibatches
    return optax.cosine_decay_schedule(learning_rate, optimizer_steps)


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
    def __init__(self, args: Args):
        self.args = nnx.static(args)
        self.global_steps = jnp.array(0)

        # Initialize environment and rngs
        env, vecenv = make_env(args)
        self.vecenv = nnx.data(vecenv)
        self.rngs = nnx.Rngs(args.seed)

        # Initialize policy and value function
        discrete = isinstance(env.action_space, envelope.Discrete)
        policy_cls = DiscretePolicy if discrete else GaussianPolicy
        self.policy = policy_cls(env.observation_space, env.action_space, self.rngs)
        self.value_fn = ValueFunction(env.observation_space, self.rngs)

        # Initialize optimizers
        self.policy_optimizer = nnx.Optimizer(
            self.policy,
            optax.adamw(
                make_lr_schedule(args.policy_lr, args), weight_decay=WEIGHT_DECAY
            ),
            wrt=nnx.Param,
        )
        self.value_fn_optimizer = nnx.Optimizer(
            self.value_fn,
            optax.adamw(
                make_lr_schedule(args.value_fn_lr, args), weight_decay=WEIGHT_DECAY
            ),
            wrt=nnx.Param,
        )

        # Initialize environment state and info
        env_state, env_info = self.vecenv.init(self.rngs())
        self.env_state = nnx.data(env_state)
        self.env_info = nnx.data(env_info)


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


def calculate_gae(ts: TrainState, info, last_value):
    @nnx.scan(reverse=True)
    def gae_step(carry, transition):
        gae, next_value = carry
        done = transition.terminated | transition.truncated

        next_value = jnp.where(transition.truncated, transition.value_next, next_value)
        next_value = jnp.where(transition.terminated, 0, next_value)

        delta = transition.reward + ts.args.gamma * next_value - transition.value
        gae = delta + ts.args.gamma * ts.args.gae_lambda * (1 - done) * gae
        return (gae, transition.value), gae

    init_carry = (jnp.zeros_like(last_value), last_value)
    _, advantages = gae_step(init_carry, info)
    return advantages


def bootstrap_observation(info):
    done = info.terminated | info.truncated
    mask = done.reshape(done.shape + (1,) * (info.obs.ndim - done.ndim))
    return jnp.where(mask, info.final.obs, info.obs)


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
    last_value = ts.value_fn(bootstrap_observation(ts.env_info))
    advantages = calculate_gae(ts, info, last_value)
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


def make_train_block(num_updates: int):
    @nnx.jit
    @nnx.scan(in_axes=nnx.Carry, length=num_updates)
    def train_block(train_state: TrainState):
        out_info = train_step(train_state)
        return train_state, out_info

    return train_block


def summarize_block(out_info):
    completed = out_info.terminated | out_info.truncated
    num_episodes = completed.sum()

    def completed_mean(values):
        total = jnp.where(completed, values, 0).sum()
        return jnp.where(num_episodes > 0, total / num_episodes, jnp.nan)

    return {
        "num_episodes": num_episodes,
        "mean_return": completed_mean(out_info.final.stats.reward),
        "mean_episode_length": completed_mean(out_info.final.stats.length),
        "policy_loss": out_info.policy_loss.mean(),
        "value_loss": out_info.value_loss.mean(),
        "policy_entropy": out_info.policy_entropy.mean(),
    }


if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=dataclasses.asdict(args),
        )

    train_state = TrainState(args)

    num_updates = get_num_updates(args)
    block_sizes = [args.updates_per_block] * (num_updates // args.updates_per_block)
    if remainder := num_updates % args.updates_per_block:
        block_sizes.append(remainder)

    compiled_blocks = {}
    training_start = time.perf_counter()
    for block_index, block_size in enumerate(block_sizes, start=1):
        if block_size not in compiled_blocks:
            compiled_blocks[block_size] = make_train_block(block_size)

        block_start = time.perf_counter()
        train_state, out_info = compiled_blocks[block_size](train_state)
        stats = jax.device_get(summarize_block(out_info))
        elapsed = time.perf_counter() - block_start
        global_steps = int(jax.device_get(train_state.global_steps))
        sps = block_size * args.num_steps * args.num_envs / elapsed

        print(
            f"block {block_index}/{len(block_sizes)}: steps={global_steps}, "
            f"episodes={int(stats['num_episodes'])}, "
            f"return={float(stats['mean_return']):.4f}, "
            f"length={float(stats['mean_episode_length']):.2f}, "
            f"policy_loss={float(stats['policy_loss']):.4f}, "
            f"value_loss={float(stats['value_loss']):.4f}, sps={sps:.0f}"
        )

        if args.use_wandb and block_index % args.wandb_log_every == 0:
            wandb.log(
                {**{key: float(value) for key, value in stats.items()}, "sps": sps},
                step=global_steps,
            )

    print(f"Total time: {(time.perf_counter() - training_start):.2f} seconds")

    if args.use_wandb:
        wandb.finish()
