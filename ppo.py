from dataclasses import dataclass
from functools import cached_property
from typing import override

import distrax
import jax
import jax.numpy as jnp
import optax
from flax import nnx

import envelope
from envelope.environment import Info, State
from envelope.typing import Key, PyTree


def flatten_space(space: envelope.Space):
    def is_leaf(x):
        return isinstance(x, tuple)

    shapes, treedef = jax.tree.flatten(space.shape, is_leaf=is_leaf)
    dims = jax.tree.map(lambda x: jnp.prod(jnp.asarray(x)), shapes, is_leaf=is_leaf)
    return treedef, shapes, dims


def flatten_x(x: PyTree):
    leaves = jax.tree.leaves(x)
    xs = jax.tree.map(lambda x: jnp.asarray(x).reshape(-1), leaves)
    return jnp.concatenate(xs, axis=0)


def unflatten_x(x: jax.Array, treedef, shapes, dims):
    indices = jnp.cumsum(jnp.array(dims))[:-1]  # last split is the remainder
    xs = jnp.split(x, indices)
    xs = jax.tree.map(lambda x, shape: x.reshape(shape), xs, shapes)
    return jax.tree.unflatten(treedef, xs)


class ClipActionWrapper(envelope.Wrapper):
    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        if isinstance(self.action_space, envelope.Continuous):
            action = jnp.clip(action, self.action_space.low, self.action_space.high)
        elif isinstance(self.action_space, envelope.Discrete):
            action = jnp.clip(action, 0, self.action_space.n - 1)
        else:
            raise ValueError(
                f"Unsupported action space type: {type(self.action_space)}"
            )
        return self.env.step(state, action)


class FlattenObservationWrapper(envelope.Wrapper):
    @override
    def reset(
        self, key: Key, state: State | None = None, **kwargs
    ) -> tuple[State, Info]:
        state, info = self.env.reset(key, state, **kwargs)
        info = info.update(obs=flatten_x(info.obs))
        return state, info

    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        state, info = self.env.step(state, action)
        info = info.update(obs=flatten_x(info.obs))
        return state, info

    @override
    @cached_property
    def observation_space(self) -> envelope.Space:
        def is_leaf(x):
            spaces = (envelope.Continuous, envelope.Discrete, envelope.BatchedSpace)
            return isinstance(x, spaces)

        spaces = jax.tree.leaves(self.env.observation_space, is_leaf=is_leaf)
        obs_cls = type(spaces[0])

        if not all(isinstance(space, obs_cls) for space in spaces):
            raise ValueError("All spaces must be of the same type")

        if obs_cls == envelope.Continuous:
            lows = [jnp.asarray(s.low).reshape(-1) for s in spaces]
            highs = [jnp.asarray(s.high).reshape(-1) for s in spaces]
            low = jnp.concatenate(lows, axis=0)
            high = jnp.concatenate(highs, axis=0)
            return envelope.Continuous(low=low, high=high)
        elif obs_cls == envelope.Discrete:
            ns = [jnp.asarray(s.n).reshape(-1) for s in spaces]
            n = jnp.concatenate(ns, axis=0)
            return envelope.Discrete(n=n)

        raise ValueError(f"Unsupported space type: {obs_cls}")


class FlattenActionWrapper(envelope.Wrapper):
    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        treedef, shapes, dims = flatten_space(self.env.action_space)
        action = unflatten_x(action, treedef, shapes, dims)
        return self.env.step(state, action)

    @override
    @cached_property
    def action_space(self) -> envelope.Space:
        def is_leaf(x):
            spaces = (envelope.Continuous, envelope.Discrete, envelope.BatchedSpace)
            return isinstance(x, spaces)

        spaces = jax.tree.leaves(self.env.action_space, is_leaf=is_leaf)
        obs_cls = type(spaces[0])

        if not all(isinstance(space, obs_cls) for space in spaces):
            raise ValueError("All spaces must be of the same type")

        if obs_cls == envelope.Continuous:
            lows = [jnp.asarray(s.low).reshape(-1) for s in spaces]
            highs = [jnp.asarray(s.high).reshape(-1) for s in spaces]
            low = jnp.concatenate(lows, axis=0)
            high = jnp.concatenate(highs, axis=0)
            return envelope.Continuous(low=low, high=high)
        elif obs_cls == envelope.Discrete:
            ns = [jnp.asarray(s.n).reshape(-1) for s in spaces]
            n = jnp.concatenate(ns, axis=0)
            return envelope.Discrete(n=n)

        raise ValueError(f"Unsupported space type: {obs_cls}")


class EpisodeStatistics(envelope.FrozenPyTreeNode):
    episode_return: jax.Array = envelope.field()
    episode_length: jax.Array = envelope.field()


def nan_stats() -> "EpisodeStatistics":
    return EpisodeStatistics(episode_return=jnp.nan, episode_length=jnp.nan)


def zero_stats() -> "EpisodeStatistics":
    return EpisodeStatistics(episode_return=0.0, episode_length=0)


class EpisodeStatisticsWrapper(envelope.Wrapper):
    class EpisodeStatisticsState(envelope.WrappedState):
        current_stats: EpisodeStatistics = envelope.field(default_factory=zero_stats)
        last_stats: EpisodeStatistics = envelope.field(default_factory=nan_stats)

    def reset(
        self, key: Key, state: State | None = None, **kwargs
    ) -> tuple[State, Info]:
        if state is None:
            inner_state, info = self.env.reset(key, None, **kwargs)
            state = self.EpisodeStatisticsState(inner_state=inner_state)
            info = self._update_info(state, info)
            return state, info

        inner_state, info = self.env.reset(key, state.inner_state, **kwargs)
        state = state.replace(
            inner_state=inner_state,
            current_stats=zero_stats(),
            last_stats=state.current_stats,
        )
        info = self._update_info(state, info)
        return state, info

    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        inner_state, info = self.env.step(state.inner_state, action)
        current_stats = state.current_stats.replace(
            episode_return=state.current_stats.episode_return + info.reward,
            episode_length=state.current_stats.episode_length + 1,
        )
        state = state.replace(
            inner_state=inner_state,
            current_stats=current_stats,
        )
        info = self._update_info(state, info)
        return state, info

    def _update_info(self, state: State, info: Info) -> Info:
        return info.update(
            last_return=state.last_stats.episode_return,
            last_length=state.last_stats.episode_length,
        )


class ValueFunction(nnx.Module):
    def __init__(self, obs_space: envelope.Space, rngs: nnx.Rngs):
        in_dim = jnp.prod(jnp.array(obs_space.shape))
        self.layers = nnx.Sequential(
            nnx.Linear(in_dim, 256, rngs=rngs),
            nnx.swish,
            nnx.Linear(256, 256, rngs=rngs),
            nnx.swish,
            nnx.Linear(256, 1, rngs=rngs),
        )

    def __call__(self, obs: jax.Array) -> jax.Array:
        return self.layers(obs).squeeze(-1)


class GaussianPolicy(nnx.Module):
    def __init__(
        self, obs_space: envelope.Space, action_space: envelope.Space, rngs: nnx.Rngs
    ):
        in_dim = jnp.prod(jnp.array(obs_space.shape))
        out_dim = jnp.prod(jnp.array(action_space.shape))
        self.action_low, self.action_high = action_space.low, action_space.high
        self.layers = nnx.Sequential(
            nnx.Linear(in_dim, 256, rngs=rngs),
            nnx.swish,
            nnx.Linear(256, 256, rngs=rngs),
            nnx.swish,
        )
        self.action_mean = nnx.Linear(256, out_dim, rngs=rngs)
        self.action_log_std = nnx.Linear(256, out_dim, rngs=rngs)

    def __call__(self, obs: jax.Array) -> distrax.Distribution:
        features = self.layers(obs)
        action_mean = self.action_mean(features)
        action_log_std = self.action_log_std(features)
        return distrax.Independent(
            distrax.Clipped(
                distrax.Normal(loc=action_mean, scale=jnp.exp(action_log_std)),
                minimum=self.action_low,
                maximum=self.action_high,
            ),
            reinterpreted_batch_ndims=1,
        )


class DiscretePolicy(nnx.Module):
    def __init__(
        self, obs_space: envelope.Space, action_space: envelope.Space, rngs: nnx.Rngs
    ):
        in_dim = jnp.prod(jnp.array(obs_space.shape))
        out_dim = action_space.n.item()
        self.layers = nnx.Sequential(
            nnx.Linear(in_dim, 256, rngs=rngs),
            nnx.swish,
            nnx.Linear(256, 256, rngs=rngs),
            nnx.swish,
        )
        self.action_logits = nnx.Linear(256, out_dim, rngs=rngs)

    def __call__(self, obs: jax.Array) -> distrax.Distribution:
        features = self.layers(obs)
        action_logits = self.action_logits(features)
        return distrax.Categorical(logits=action_logits)


def collect_trajectories(train_state):
    @nnx.scan(
        in_axes=(nnx.Carry),
        out_axes=(nnx.Carry, 0),
        length=train_state.args.num_steps,
    )
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
            value=value,  # V(s_t): for TD error
            value_next=train_state.value_fn(
                env_info.obs_true
            ),  # V(s_{t+1}^true): for bootstrapping on truncation
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


def calculate_gae(info, last_value, gamma, gae_lambda):
    @nnx.scan(
        in_axes=(nnx.Carry, 0, None, None),
        out_axes=(nnx.Carry, 0),
        reverse=True,
    )
    def _gae_step(carry, transition, gamma, gae_lambda):
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
        delta = transition.reward + gamma * next_v - transition.value
        # Reset GAE on both termination and truncation (new episode starts after either)
        gae = delta + gamma * gae_lambda * (1 - done) * gae
        return (gae, transition.value), gae

    init_carry = (jnp.zeros_like(last_value), last_value)
    _, advantages = _gae_step(init_carry, info, gamma, gae_lambda)
    return advantages


def normalize(x: jax.Array) -> jax.Array:
    return (x - x.mean()) / (x.std() + 1e-8)


def update_policy(train_state, batch):
    @nnx.value_and_grad
    def loss_fn(policy):
        # Optional todo: add entropy bonus
        pi = policy(batch.obs)
        log_prob = pi.log_prob(batch.action)

        eps = train_state.args.epsilon
        ratio = jnp.exp(log_prob - batch.log_prob)
        clip_ratio = jnp.clip(ratio, 1 - eps, 1 + eps)
        advantages = normalize(batch.advantages)

        surrogate1 = ratio * advantages
        surrogate2 = clip_ratio * advantages
        return -jnp.mean(jnp.minimum(surrogate1, surrogate2))

    loss, grads = loss_fn(train_state.policy)
    train_state.policy_optimizer.update(train_state.policy, grads)
    return loss


def update_value_fn(train_state, batch):
    @nnx.value_and_grad
    def loss_fn(value_fn):
        targets = batch.value + batch.advantages
        values = value_fn(batch.obs)
        return 0.5 * jnp.mean((values - targets) ** 2)

    loss, grads = loss_fn(train_state.value_fn)
    train_state.value_fn_optimizer.update(train_state.value_fn, grads)
    return loss


def update(train_state, batch):
    policy_loss = update_policy(train_state, batch)
    value_fn_loss = update_value_fn(train_state, batch)
    return policy_loss + value_fn_loss


def train(train_state):
    out_info = collect_trajectories(train_state)
    last_value = train_state.value_fn(train_state.env_info.obs_true)
    advantages = calculate_gae(out_info, last_value, gamma=0.99, gae_lambda=0.95)
    out_info = out_info.update(advantages=advantages)
    minibatches = shuffle_and_split(out_info, num_minibatches=5, rngs=train_state.rngs)

    for i in range(5):
        batch = jax.tree.map(lambda x: x[i], minibatches)
        update(train_state, batch)

    return out_info


@dataclass
class Args:
    env_name: str = "gymnax::CartPole-v1"
    policy_lr: float = 0.001
    value_fn_lr: float = 0.001
    epsilon: float = 0.2
    num_envs: int = 10
    num_minibatches: int = 5
    num_updates: int = 5
    num_steps: int = 100
    normalize_observations: bool = False
    seed: int = 0


class TrainState(nnx.Pytree):
    def __init__(self, args: Args):
        self.args = nnx.static(args)

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
        if discrete:
            self.policy = DiscretePolicy(
                env.observation_space, env.action_space, rngs=self.rngs
            )
        else:
            self.policy = GaussianPolicy(
                env.observation_space, env.action_space, rngs=self.rngs
            )
        self.value_fn = ValueFunction(env.observation_space, rngs=self.rngs)

        # Initialize optimizers
        self.policy_optimizer = nnx.Optimizer(
            self.policy, optax.adam(args.policy_lr), wrt=nnx.Param
        )
        self.value_fn_optimizer = nnx.Optimizer(
            self.value_fn, optax.adam(args.value_fn_lr), wrt=nnx.Param
        )

        # Initialize environment state and info
        env_state, env_info = self.vecenv.reset(self.rngs())
        self.env_state = nnx.data(env_state)
        self.env_info = nnx.data(env_info)


if __name__ == "__main__":
    import tyro

    args = tyro.cli(Args)
    train_state = TrainState(args)

    train = nnx.jit(train)

    last_returns = []
    for i in range(10000):
        out_info = train(train_state)
        mean_return = out_info.last_return.mean()
        last_returns.append(mean_return)
        print(
            f"mean_return={mean_return.mean():.4f}, mean_value={out_info.value.mean():.4f}"
        )

    import matplotlib.pyplot as plt

    plt.plot(last_returns)
    plt.show()
