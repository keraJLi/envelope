import dataclasses

import jax
import jax.numpy as jnp
import optax
from flax import nnx

import envelope
from envelope.typing import PyTree
from ppo.networks import DiscretePolicy, GaussianPolicy, ValueFunction
from ppo.wrappers import (
    ClipActionWrapper,
    EpisodeStatisticsWrapper,
    FlattenActionWrapper,
    FlattenObservationWrapper,
)


def collect_trajectories(train_state, args):
    @nnx.scan(in_axes=nnx.Carry, length=args.num_steps)
    def step_env(train_state):
        obs = train_state.env_info.obs  # s_t: observation BEFORE action

        value = train_state.value_fn(obs)  # V(s_t)
        pi = train_state.policy(obs)

        action = pi.sample(seed=train_state.rngs())
        env_state, env_info = train_state.vecenv.step(train_state.env_state, action)
        train_state.env_state = env_state
        train_state.env_info = env_info

        out_info = env_info.update(
            obs=obs,  # Store obs BEFORE step (for value function training)
            action=action,
            log_prob=pi.log_prob(action),
            # V(s_t): for TD error
            value=value,
            # V(s_{t+1}^true): for bootstrapping on truncation
            value_next=train_state.value_fn(env_info.obs_true),
        )
        return train_state, out_info

    train_state, out_info = step_env(train_state)
    return out_info


def shuffle_and_split(data: PyTree, num_minibatches: int, rngs: nnx.Rngs):
    def is_numeric(x):
        return hasattr(x, "dtype") and not jnp.issubdtype(x.dtype, jax.dtypes.prng_key)

    first_leaf = next(x for x in jax.tree.leaves(data) if is_numeric(x))
    num_steps, num_envs = first_leaf.shape[:2]
    iteration_size = num_steps * num_envs
    permutation = jax.random.permutation(rngs(), iteration_size)

    def _shuffle_and_split(x):
        if not is_numeric(x):
            return x
        x = x.reshape((iteration_size, *x.shape[2:]))
        x = jnp.take(x, permutation, axis=0)
        return x.reshape(num_minibatches, -1, *x.shape[1:])

    return jax.tree.map(_shuffle_and_split, data)


def calculate_gae(info, last_value, args):
    @nnx.scan(reverse=True)
    def _gae_step(carry, transition):
        gae, next_value = carry
        done = transition.terminated | transition.truncated
        # Bootstrap value for V(s_{t+1}):
        # - Termination: 0 (episode truly ended)
        # - Truncation: value_next (bootstrap from true next state)
        # - Otherwise: next_value from carry (V of next transition's state)
        next_v = jnp.where(
            transition.truncated,
            transition.value_next,
            next_value * (1 - transition.terminated),
        )
        # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        delta = transition.reward + args.gamma * next_v - transition.value
        # Reset GAE on both termination and truncation (new episode starts after either)
        gae = delta + args.gamma * args.gae_lambda * (1 - done) * gae
        return (gae, transition.value), gae

    init_carry = (jnp.zeros_like(last_value), last_value)
    _, advantages = _gae_step(init_carry, info)
    return advantages


def normalize(x: jax.Array) -> jax.Array:
    return (x - x.mean()) / (x.std() + 1e-8)


def update_policy(train_state, args, batch):
    @nnx.value_and_grad
    def loss_fn(policy):
        # Optional todo: add entropy bonus
        pi = policy(batch.obs)
        log_prob = pi.log_prob(batch.action)

        ratio = jnp.exp(log_prob - batch.log_prob)
        clip_ratio = jnp.clip(ratio, 1 - args.epsilon, 1 + args.epsilon)
        advantages = normalize(batch.advantages)

        surrogate1 = ratio * advantages
        surrogate2 = clip_ratio * advantages
        return -jnp.mean(jnp.minimum(surrogate1, surrogate2))

    loss, grads = loss_fn(train_state.policy)
    train_state.policy_optimizer.update(train_state.policy, grads)
    return loss


def update_value_fn(train_state, args, batch):
    @nnx.value_and_grad
    def loss_fn(value_fn):
        # Optional todo: value function clipping
        targets = batch.value + batch.advantages
        values = value_fn(batch.obs)
        return 0.5 * jnp.mean((values - targets) ** 2)

    loss, grads = loss_fn(train_state.value_fn)
    train_state.value_fn_optimizer.update(train_state.value_fn, grads)
    return loss


def update(train_state, args, minibatches):
    @nnx.scan
    def update_on_batch(train_state, batch):
        policy_loss = update_policy(train_state, args, batch)
        value_fn_loss = update_value_fn(train_state, args, batch)
        return train_state, policy_loss + value_fn_loss

    return update_on_batch(train_state, minibatches)


def train(train_state, args):
    out_info = collect_trajectories(train_state, args)
    last_value = train_state.value_fn(train_state.env_info.obs_true)
    advantages = calculate_gae(out_info, last_value, args)
    out_info = out_info.update(advantages=advantages)
    minibatches = shuffle_and_split(out_info, args.num_minibatches, train_state.rngs)
    update(train_state, args, minibatches)
    return out_info


@dataclasses.dataclass(frozen=True)
class Args:
    env_name: str = "gymnax::CartPole-v1"
    policy_lr: float = 0.001
    value_fn_lr: float = 0.001
    epsilon: float = 0.2
    num_envs: int = 10
    num_minibatches: int = 5
    num_updates: int = 5
    num_steps: int = 100
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_observations: bool = False
    seed: int = 0


class TrainState(nnx.Pytree):
    def __init__(self, args: Args):
        # Initialize environment
        env = envelope.create(args.env_name)
        env = FlattenObservationWrapper(env=env)
        env = FlattenActionWrapper(env=env)
        env = ClipActionWrapper(env=env)
        env = EpisodeStatisticsWrapper(env=env)
        vecenv = envelope.VmapWrapper(env=env, batch_size=args.num_envs)
        if args.normalize_observations:
            vecenv = envelope.ObservationNormalizationWrapper(env=vecenv)
        vecenv = envelope.AutoResetWrapper(env=vecenv)

        self.vecenv = nnx.data(vecenv)

        # Initialize policy and value function
        self.rngs = nnx.Rngs(args.seed)
        discrete = isinstance(env.action_space, envelope.Discrete)
        policy_cls = DiscretePolicy if discrete else GaussianPolicy
        self.policy = policy_cls(
            env.observation_space, env.action_space, rngs=self.rngs
        )
        self.value_fn = ValueFunction(env.observation_space, rngs=self.rngs)

        # Initialize optimizers
        self.policy_optimizer = nnx.Optimizer(
            self.policy, optax.adamw(args.policy_lr), wrt=nnx.Param
        )
        self.value_fn_optimizer = nnx.Optimizer(
            self.value_fn, optax.adamw(args.value_fn_lr), wrt=nnx.Param
        )

        # Initialize environment state and info
        env_state, env_info = self.vecenv.reset(self.rngs())
        self.env_state = nnx.data(env_state)
        self.env_info = nnx.data(env_info)


def make_env(args: Args) -> TrainState:
    env = envelope.create(args.env_name)
    env = FlattenObservationWrapper(env=env)
    env = FlattenActionWrapper(env=env)
    env = ClipActionWrapper(env=env)
    env = EpisodeStatisticsWrapper(env=env)
    vecenv = envelope.VmapWrapper(env=env, batch_size=args.num_envs)
    if args.normalize_observations:
        vecenv = envelope.ObservationNormalizationWrapper(env=vecenv)
    vecenv = envelope.AutoResetWrapper(env=vecenv)

    return env, vecenv


if __name__ == "__main__":
    import tyro

    args = tyro.cli(Args)
    train_state = TrainState(args)

    train = nnx.jit(train, static_argnames=("args",))

    last_returns = []
    for i in range(10000):
        out_info = train(train_state, args)
        mean_return = out_info.last_return.mean()
        last_returns.append(mean_return)
        print(
            f"mean_return={mean_return.mean():.4f}, "
            f"mean_value={out_info.value.mean():.4f}"
        )

    import matplotlib.pyplot as plt

    plt.plot(last_returns)
    plt.show()
