"""PPO matching craftax-baselines configuration.

Key differences from ppo/ppo.py:
- Combined actor-critic network with single optimizer
- Combined loss (actor + vf_coef * value_loss - ent_coef * entropy)
- Value clipping
- LR annealing (linear schedule)
- Grad clipping via clip_by_global_norm
- adam with eps=1e-5
"""


# TODO: separate saving info on done and accumulating episode stats/achievements.

import dataclasses
from typing import override

import jax
import jax.numpy as jnp
import optax
from envelope.wrappers.pooled_reset_vmap_wrapper import PooledInitVmapWrapper
from flax import nnx
from ppo.wrappers import (
    ClipActionWrapper,
    EpisodeStatisticsWrapper,
    FlattenActionWrapper,
    FlattenObservationWrapper,
)

import envelope
from envelope.typing import PyTree
from ppo_craftax.networks import ActorCritic


class SaveAchievementsWrapper(envelope.Wrapper):
    class SaveAchievementsState(envelope.WrappedState):
        achievements: PyTree

    def _achievements_from_info(self, info: envelope.WrappedState) -> PyTree:
        craftax_info = info.info
        achievements = {
            a: v
            for a, v in craftax_info.items()
            if a.lower().startswith("achievements")
        }
        return achievements

    @override
    def init(self, key: jax.Array) -> tuple[envelope.WrappedState, envelope.Info]:
        inner_state, info = self.env.init(key)
        ph_achievements = jax.tree.map(
            lambda x: jnp.full_like(x, jnp.nan), self._achievements_from_info(info)
        )
        state = self.SaveAchievementsState(
            inner_state=inner_state, achievements=ph_achievements
        )
        info = info.update(achievements=ph_achievements)
        return state, info

    @override
    def reset(
        self, state: envelope.WrappedState, key: jax.Array
    ) -> tuple[envelope.WrappedState, envelope.Info]:
        inner_state, info = self.env.reset(state.inner_state, key)
        state = state.replace(inner_state=inner_state)
        info = info.update(achievements=state.achievements)
        return state, info

    @override
    def step(
        self, state: envelope.WrappedState, action: jax.Array
    ) -> tuple[envelope.WrappedState, envelope.Info]:
        inner_state, info = self.env.step(state.inner_state, action)

        # On done, use the new achievements, otherwise don't touch
        done = info.terminated | info.truncated
        achievements = jax.tree.map(
            lambda x, y: jnp.where(done, x, y),
            self._achievements_from_info(info),
            state.achievements,
        )

        state = state.replace(inner_state=inner_state, achievements=achievements)
        info = info.update(achievements=achievements)
        return state, info


@dataclasses.dataclass(frozen=True)
class Args:
    env_name: str = "craftax::Craftax-Symbolic-v1"
    total_timesteps: int = 1_000_000_000
    lr: float = 2e-4
    anneal_lr: bool = True
    epsilon: float = 0.2
    entropy_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    num_envs: int = 1024
    num_minibatches: int = 8
    num_epochs: int = 4
    num_steps: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.8
    layer_size: int = 512
    normalize_observations: bool = False
    use_optimistic_resets: bool = True
    optimistic_reset_ratio: int = 16
    seed: int = 0

    # wandb
    use_wandb: bool = False
    wandb_entity: str | None = None
    wandb_project: str | None = "envelope-ppo-craftax"
    wandb_log_every: int = 10


def make_env(args: Args):
    env = envelope.create(args.env_name)
    env = FlattenObservationWrapper(env=env)
    env = FlattenActionWrapper(env=env)
    env = ClipActionWrapper(env=env)
    env = EpisodeStatisticsWrapper(env=env)
    env = SaveAchievementsWrapper(env=env)
    if args.use_optimistic_resets:
        vecenv = PooledInitVmapWrapper(env=env, batch_size=args.num_envs, pool_size=1)
    else:
        vecenv = envelope.VmapWrapper(env=env, batch_size=args.num_envs)
        vecenv = envelope.AutoResetWrapper(env=vecenv)
    if args.normalize_observations:
        vecenv = envelope.ObservationNormalizationWrapper(env=vecenv)
    return env, vecenv


class TrainState(nnx.Pytree):
    def __init__(self, args: Args):
        self.args = nnx.static(args)
        self.global_steps = jnp.array(0)

        env, vecenv = make_env(args)
        self.vecenv = nnx.data(vecenv)
        self.rngs = nnx.Rngs(args.seed)

        # Combined actor-critic network
        self.network = ActorCritic(
            env.observation_space, env.action_space, self.rngs, args.layer_size
        )

        # Single optimizer with grad clipping and optional LR annealing
        num_updates = args.total_timesteps // (args.num_steps * args.num_envs)
        if args.anneal_lr:

            def linear_schedule(count):
                frac = (
                    1.0
                    - (count // (args.num_minibatches * args.num_epochs)) / num_updates
                )
                return args.lr * frac

            tx = optax.chain(
                optax.clip_by_global_norm(args.max_grad_norm),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(args.max_grad_norm),
                optax.adam(args.lr, eps=1e-5),
            )

        self.optimizer = nnx.Optimizer(self.network, tx, wrt=nnx.Param)

        # Initialize environment state
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
        pi, value = ts.network(obs)
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
        )
        return ts, out_info

    ts, out_info = step_env(ts)
    return out_info


def calculate_gae(ts: TrainState, info, last_value):
    @nnx.scan(reverse=True)
    def gae_step(carry, transition):
        gae, next_value = carry
        done = transition.terminated | transition.truncated
        delta = (
            transition.reward
            + ts.args.gamma * next_value * (1 - done)
            - transition.value
        )
        gae = delta + ts.args.gamma * ts.args.gae_lambda * (1 - done) * gae
        return (gae, transition.value), gae

    init_carry = (jnp.zeros_like(last_value), last_value)
    _, advantages = gae_step(init_carry, info)
    return advantages


def update_epoch(ts: TrainState, minibatches):
    @nnx.scan
    def update_minibatch(ts: TrainState, batch):
        targets = batch.value + batch.advantages

        @nnx.value_and_grad(has_aux=True)
        def loss_fn(network):
            pi, value = network(batch.obs)
            log_prob = pi.log_prob(batch.action)
            entropy = pi.entropy().mean()

            # Actor loss
            ratio = jnp.exp(log_prob - batch.log_prob)
            advantages = (batch.advantages - batch.advantages.mean()) / (
                batch.advantages.std() + 1e-8
            )
            clip_ratio = jnp.clip(ratio, 1 - ts.args.epsilon, 1 + ts.args.epsilon)
            actor_loss = -jnp.minimum(
                ratio * advantages, clip_ratio * advantages
            ).mean()

            # Value loss with clipping
            value_pred_clipped = batch.value + (value - batch.value).clip(
                -ts.args.epsilon, ts.args.epsilon
            )
            value_losses = jnp.square(value - targets)
            value_losses_clipped = jnp.square(value_pred_clipped - targets)
            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

            # Combined loss
            total_loss = (
                actor_loss
                + ts.args.vf_coef * value_loss
                - ts.args.entropy_coef * entropy
            )
            return total_loss, (actor_loss, value_loss, entropy)

        (total_loss, (actor_loss, value_loss, entropy)), grads = loss_fn(ts.network)
        ts.optimizer.update(ts.network, grads)

        return ts, {
            "total_loss": total_loss,
            "actor_loss": actor_loss,
            "value_loss": value_loss,
            "entropy": entropy,
        }

    _, loss_info = update_minibatch(ts, minibatches)
    return loss_info


def train_step(ts: TrainState):
    info = collect_trajectories(ts)

    _, last_value = ts.network(ts.env_info.obs)
    last_value = jnp.where(
        ts.env_info.terminated | ts.env_info.truncated, 0, last_value
    )
    advantages = calculate_gae(ts, info, last_value)
    info = info.update(advantages=advantages)

    @nnx.scan(in_axes=nnx.Carry, length=ts.args.num_epochs)
    def update_epoch_scan(ts):
        minibatches = shuffle_and_split(info, ts.args.num_minibatches, ts.rngs())
        loss_info = update_epoch(ts, minibatches)
        return ts, loss_info

    ts, loss_infos = update_epoch_scan(ts)
    loss_infos = jax.tree.map(jnp.mean, loss_infos)
    return info.update(**loss_infos)


if __name__ == "__main__":
    import time

    import tyro

    import wandb

    args = tyro.cli(Args)

    if args.use_wandb:
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
        mean_return,
        global_steps,
        total_loss,
        actor_loss,
        value_loss,
        entropy,
        achievements,
    ):
        global start, update_count
        update_count += 1

        time_elapsed = time.time() - start
        sps = global_steps / time_elapsed
        print(
            f"global_steps: {global_steps}, mean_return: {mean_return}, "
            f"sps: {sps:.2f}, time_elapsed: {time_elapsed:.2f}"
        )

        if args.use_wandb and update_count % args.wandb_log_every == 0:
            wandb.log(
                {
                    "time/sps": sps,
                    "time/iteration": update_count,
                    "mean_return": float(mean_return),
                    "total_loss": float(total_loss),
                    "actor_loss": float(actor_loss),
                    "value_loss": float(value_loss),
                    "entropy": float(entropy),
                    **achievements,
                },
                step=int(global_steps),
            )

    @nnx.jit
    @nnx.scan(in_axes=nnx.Carry, length=num_updates)
    def train_loop(train_state):
        out_info = train_step(train_state)
        mean_return = jnp.nanmean(out_info.last_return)
        achievements = jax.tree.map(jnp.nanmean, out_info.achievements)
        jax.debug.callback(
            callback,
            mean_return,
            train_state.global_steps,
            out_info.total_loss,
            out_info.actor_loss,
            out_info.value_loss,
            out_info.entropy,
            achievements,
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

    if args.use_wandb:
        wandb.log({"time/lower": time_lower, "time/compile": time_compile}, step=0)

    start = time.time()
    train_state, mean_returns = train_loop(train_state)
    mean_returns = mean_returns.block_until_ready()
    print(f"Total time: {(time.time() - start):.2f} seconds")
    wandb.finish()
