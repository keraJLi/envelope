import dataclasses
import time

import jax
import jax.numpy as jnp
import optax
import tyro
from flax import nnx

import envelope
from envelope.typing import PyTree
from examples.ppo.networks import DiscretePolicy, GaussianPolicy, ValueFunction


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

    # wandb
    use_wandb: bool = False
    wandb_entity: str | None = None
    wandb_project: str | None = "envelope-ppo"
    wandb_log_every: int = 1


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
            self.policy, optax.adamw(args.policy_lr), wrt=nnx.Param
        )
        self.value_fn_optimizer = nnx.Optimizer(
            self.value_fn, optax.adamw(args.value_fn_lr), wrt=nnx.Param
        )

        # Initialize environment state and info
        env_state, env_info = self.vecenv.init(self.rngs())
        self.env_state = nnx.data(env_state)
        self.env_info = nnx.data(env_info)


def shuffle_and_split(data: PyTree, num_minibatches: int, key: jax.Array):
    first_leaf = jax.tree.leaves(data)[0]
    num_steps, num_envs = first_leaf.shape[:2]
    batch_size = num_steps * num_envs
    if num_minibatches <= 0:
        raise ValueError("num_minibatches must be positive")
    if batch_size % num_minibatches:
        raise ValueError(
            f"rollout batch size {batch_size} must be divisible by "
            f"num_minibatches={num_minibatches}"
        )
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
        reset_value = ts.value_fn(env_info.obs)
        terminal_value = ts.value_fn(env_info.final.obs)
        bootstrap_value = jnp.where(
            env_info.final_valid & env_info.truncated,
            terminal_value,
            reset_value,
        )
        ts.global_steps += ts.args.num_envs
        ts.env_state = env_state
        ts.env_info = env_info

        out_info = env_info.update(
            obs=obs,
            value=value,
            action=action,
            log_prob=pi.log_prob(action),
            bootstrap_value=bootstrap_value,
        )
        return ts, out_info

    ts, out_info = step_env(ts)
    return out_info


def compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    bootstrap_values: jax.Array,
    terminated: jax.Array,
    truncated: jax.Array,
    *,
    gamma: float,
    gae_lambda: float,
) -> jax.Array:
    """Compute GAE without leaking recurrence across either episode boundary."""
    done = terminated | truncated
    deltas = rewards + gamma * jnp.where(terminated, 0, bootstrap_values) - values

    def gae_step(next_gae, transition):
        delta, transition_done = transition
        gae = delta + gamma * gae_lambda * (~transition_done) * next_gae
        return gae, gae

    _, advantages = jax.lax.scan(
        gae_step,
        jnp.zeros_like(deltas[-1]),
        (deltas, done),
        reverse=True,
    )
    return advantages


def calculate_gae(ts: TrainState, info):
    return compute_gae(
        rewards=info.reward,
        values=info.value,
        bootstrap_values=info.bootstrap_value,
        terminated=info.terminated,
        truncated=info.truncated,
        gamma=ts.args.gamma,
        gae_lambda=ts.args.gae_lambda,
    )


def tiny_cpu_train_step(seed: int = 0) -> dict[str, jax.Array]:
    """Run one self-contained value update for fast example/CI validation."""
    obs_space = envelope.Continuous.from_shape(-1.0, 1.0, shape=(4,))
    value_fn = ValueFunction(obs_space, nnx.Rngs(seed), layer_size=16)
    optimizer = nnx.Optimizer(value_fn, optax.adam(1e-3), wrt=nnx.Param)
    observations = jnp.linspace(-1.0, 1.0, 32).reshape(8, 4)
    targets = jnp.linspace(-0.5, 0.5, 8)

    def loss_fn(model: ValueFunction) -> jax.Array:
        return jnp.mean(jnp.square(model(observations) - targets))

    loss_before = loss_fn(value_fn)
    loss, gradients = nnx.value_and_grad(loss_fn)(value_fn)
    optimizer.update(value_fn, gradients)
    loss_after = loss_fn(value_fn)
    return {
        "loss_before": loss_before,
        "loss_after": loss_after,
        "optimization_loss": loss,
        "updates": jnp.asarray(1, dtype=jnp.int32),
    }


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


if __name__ == "__main__":
    args = tyro.cli(Args)
    wandb = None

    if args.use_wandb:
        import wandb as wandb_module

        wandb = wandb_module
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=dataclasses.asdict(args),
        )

    train_state = TrainState(args)

    steps_per_update = args.num_steps * args.num_envs
    num_updates = args.total_timesteps // steps_per_update

    start = None
    update_count = 0

    def callback(
        global_steps,
        mean_return,
        mean_episode_length,
        policy_loss,
        value_loss,
        policy_entropy,
    ):
        global start, update_count
        update_count += 1

        time_elapsed = time.time() - start
        sps = global_steps / time_elapsed
        print(
            f"global_steps: {global_steps}, "
            f"mean_return: {mean_return:.4f}, "
            f"sps: {sps:.0f}, "
            f"policy_loss: {policy_loss:.4f}, "
            f"value_loss: {value_loss:.4f}, "
        )

        if wandb is not None and update_count % args.wandb_log_every == 0:
            wandb.log(
                {
                    "time/sps": sps,
                    "time/iteration": update_count,
                    "mean_return": float(mean_return),
                    "mean_episode_length": float(mean_episode_length),
                    "policy_loss": float(policy_loss),
                    "value_loss": float(value_loss),
                    "policy_entropy": float(policy_entropy),
                },
                step=int(global_steps),
            )

    @nnx.jit
    @nnx.scan(in_axes=nnx.Carry, length=num_updates)
    def train_loop(train_state):
        out_info = train_step(train_state)
        completed = out_info.final_valid & (out_info.terminated | out_info.truncated)
        completed_count = jnp.maximum(completed.sum(), 1)
        mean_return = jnp.where(
            completed.any(),
            (out_info.final.stats.reward * completed).sum() / completed_count,
            jnp.nan,
        )
        mean_episode_length = jnp.where(
            completed.any(),
            (out_info.final.stats.length * completed).sum() / completed_count,
            jnp.nan,
        )
        jax.debug.callback(
            callback,
            train_state.global_steps,
            mean_return,
            mean_episode_length,
            out_info.policy_loss,
            out_info.value_loss,
            out_info.policy_entropy,
        )
        return train_state, mean_return

    start = time.time()
    train_loop = train_loop.lower(train_state)
    time_lower = time.time() - start
    print(f"Time to lower: {time_lower:.2f} seconds")
    start = time.time()
    train_loop = train_loop.compile()
    time_compile = time.time() - start
    print(f"Time to compile: {time_compile:.2f} seconds")

    if wandb is not None:
        wandb.log({"time/lower": time_lower, "time/compile": time_compile}, step=0)

    start = time.time()
    train_state, mean_returns = train_loop(train_state)
    mean_returns = mean_returns.block_until_ready()
    print(f"Total time: {(time.time() - start):.2f} seconds")

    if wandb is not None:
        wandb.finish()
