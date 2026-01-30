from functools import cached_property
from typing import override

import jax
import jax.numpy as jnp

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
        act_cls = type(spaces[0])

        if not all(isinstance(space, act_cls) for space in spaces):
            raise ValueError("All spaces must be of the same type")

        if act_cls == envelope.Continuous:
            lows = [jnp.asarray(s.low).reshape(-1) for s in spaces]
            highs = [jnp.asarray(s.high).reshape(-1) for s in spaces]
            low = jnp.concatenate(lows, axis=0)
            high = jnp.concatenate(highs, axis=0)
            return envelope.Continuous(low=low, high=high)
        elif act_cls == envelope.Discrete:
            ns = [jnp.asarray(s.n).reshape(-1) for s in spaces]
            n = jnp.concatenate(ns, axis=0)
            return envelope.Discrete(n=n)

        raise ValueError(f"Unsupported space type: {act_cls}")


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
        # Update last_stats when episode ends (done detected in step)
        done = info.terminated | info.truncated
        last_stats = EpisodeStatistics(
            episode_return=jnp.where(done, current_stats.episode_return, state.last_stats.episode_return),
            episode_length=jnp.where(done, current_stats.episode_length, state.last_stats.episode_length),
        )
        # Reset current stats on done
        current_stats = EpisodeStatistics(
            episode_return=jnp.where(done, 0.0, current_stats.episode_return),
            episode_length=jnp.where(done, 0, current_stats.episode_length),
        )
        state = state.replace(inner_state=inner_state, current_stats=current_stats, last_stats=last_stats)
        info = self._update_info(state, info)
        return state, info

    def _update_info(self, state: State, info: Info) -> Info:
        return info.update(
            last_return=state.last_stats.episode_return,
            last_length=state.last_stats.episode_length,
        )
