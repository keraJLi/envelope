import dataclasses
import warnings
from functools import cached_property
from typing import Any, override

from jax import numpy as jnp
from mujoco_playground import registry

from jenv import spaces
from jenv.environment import Environment, Info, InfoContainer, State
from jenv.struct import static_field
from jenv.typing import Key, PyTree

_MAX_INT = int(jnp.iinfo(jnp.int32).max)


class MujocoPlaygroundJenv(Environment):
    """Wrapper to convert a mujoco_playground environment to a jenv environment."""

    mujoco_playground_env: Any = static_field()

    @classmethod
    def from_name(
        cls, env_name: str, env_kwargs: dict[str, Any] | None = None
    ) -> "MujocoPlaygroundJenv":
        """Creates a MujocoPlaygroundJenv from a name and keyword arguments. env_kwargs
        are passed to config_overrides of mujoco_playground.registry.load."""
        env_kwargs = env_kwargs or {}

        # Set episode_length to a very large value by default if not provided
        # (mujoco_playground uses int for episode_length, so we use max int instead of inf)
        episode_length = env_kwargs.setdefault("episode_length", _MAX_INT)
        if episode_length < _MAX_INT:
            warnings.warn(
                "Creating a MujocoPlaygroundJenv with a finite episode_length is not "
                "recommended, use a TruncationWrapper instead."
            )

        # Pass all env_kwargs as config_overrides
        env = registry.load(
            env_name, config_overrides=env_kwargs if env_kwargs else None
        )
        return cls(mujoco_playground_env=env)

    @override
    def reset(self, key: Key) -> tuple[State, Info]:
        state = self.mujoco_playground_env.reset(key)
        info = InfoContainer(obs=state.obs, reward=0.0, terminated=False)
        info = info.update(**dataclasses.asdict(state))
        return state, info

    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        state = self.mujoco_playground_env.step(state, action)
        info = InfoContainer(obs=state.obs, reward=state.reward, terminated=state.done)
        info = info.update(**dataclasses.asdict(state))
        return state, info

    @override
    @cached_property
    def action_space(self) -> spaces.Space:
        # MuJoCo Playground actions are typically bounded [-1, 1]
        return spaces.Continuous.from_shape(
            low=-1.0, high=1.0, shape=(self.mujoco_playground_env.action_size,)
        )

    @override
    @cached_property
    def observation_space(self) -> spaces.Space:
        import jax

        def to_space(size):
            shape = (size,) if isinstance(size, int) else size
            return spaces.Continuous.from_shape(low=-jnp.inf, high=jnp.inf, shape=shape)

        def is_leaf(x):
            return isinstance(x, int) or (isinstance(x, tuple) and all(isinstance(i, int) for i in x))

        space_tree = jax.tree.map(to_space, self.mujoco_playground_env.observation_size, is_leaf=is_leaf)
        if isinstance(space_tree, spaces.Space):
            return space_tree
        return spaces.PyTreeSpace(space_tree)
