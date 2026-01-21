import dataclasses
from functools import cached_property
from typing import Any, override

from jax import numpy as jnp
from mujoco_playground import registry

from envelope import spaces as envelope_spaces
from envelope.environment import Environment, Info, InfoContainer, State
from envelope.struct import static_field
from envelope.typing import Key, PyTree

_MAX_INT = int(jnp.iinfo(jnp.int32).max)


_MUJOCO_PLAYGROUND_DEFAULT_EPISODE_LENGTH = 1000


class MujocoPlaygroundEnvelope(Environment):
    """Wrapper to convert a mujoco_playground environment to a envelope environment."""

    mujoco_playground_env: Any = static_field()
    _default_max_steps: int = static_field(
        default=_MUJOCO_PLAYGROUND_DEFAULT_EPISODE_LENGTH
    )

    @classmethod
    def from_name(
        cls, env_name: str, env_kwargs: dict[str, Any] | None = None
    ) -> "MujocoPlaygroundEnvelope":
        """Creates a MujocoPlaygroundEnvelope from a name and keyword arguments.
        env_kwargs are passed to config_overrides of mujoco_playground.registry.load."""
        env_kwargs = env_kwargs or {}
        if "episode_length" in env_kwargs:
            raise ValueError(
                "Cannot override 'episode_length' directly. "
                "Use TruncationWrapper for episode length control."
            )

        # Get default episode_length from registry config
        default_config = registry.get_default_config(env_name)
        default_max_steps = default_config.episode_length

        # Set episode_length to a very large value
        # (mujoco_playground uses int for episode_length, so we use max int instead of inf)
        env_kwargs["episode_length"] = _MAX_INT

        # Pass all env_kwargs as config_overrides
        env = registry.load(
            env_name, config_overrides=env_kwargs if env_kwargs else None
        )
        return cls(mujoco_playground_env=env, _default_max_steps=default_max_steps)

    @property
    def default_max_steps(self) -> int:
        return self._default_max_steps

    @override
    def reset(self, key: Key) -> tuple[State, Info]:
        env_state = self.mujoco_playground_env.reset(key)
        info = InfoContainer(obs=env_state.obs, reward=0.0, terminated=False)
        info = info.update(**dataclasses.asdict(env_state))
        return env_state, info

    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        state = self.mujoco_playground_env.step(state, action)
        info = InfoContainer(obs=state.obs, reward=state.reward, terminated=state.done)
        info = info.update(**dataclasses.asdict(state))
        return state, info

    @override
    @cached_property
    def action_space(self) -> envelope_spaces.Space:
        # MuJoCo Playground actions are typically bounded [-1, 1]
        return envelope_spaces.Continuous.from_shape(
            low=-1.0, high=1.0, shape=(self.mujoco_playground_env.action_size,)
        )

    @override
    @cached_property
    def observation_space(self) -> envelope_spaces.Space:
        import jax

        def to_space(size):
            shape = (size,) if isinstance(size, int) else size
            return envelope_spaces.Continuous.from_shape(
                low=-jnp.inf, high=jnp.inf, shape=shape
            )

        def is_leaf(x):
            return isinstance(x, int) or (
                isinstance(x, tuple) and all(isinstance(i, int) for i in x)
            )

        space_tree = jax.tree.map(
            to_space, self.mujoco_playground_env.observation_size, is_leaf=is_leaf
        )
        if isinstance(space_tree, envelope_spaces.Space):
            return space_tree
        return envelope_spaces.PyTreeSpace(space_tree)
