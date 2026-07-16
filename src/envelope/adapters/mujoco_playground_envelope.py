from functools import cached_property
from typing import Any, override

from jax import numpy as jnp
from mujoco_playground import MjxEnv, registry

from envelope import spaces as envelope_spaces
from envelope.adapters._common import (
    _capture_horizon,
    backend_container,
    warn_if_wrapper_overlap,
)
from envelope.environment import Environment, Info, InfoContainer, State
from envelope.struct import static_field
from envelope.typing import Key, PyTree

_MAX_INT = int(jnp.iinfo(jnp.int32).max)


class MujocoPlaygroundEnvelope(Environment):
    """
    Wrapper to convert a mujoco_playground environment to a envelope environment.

    Mujoco Playground uses a dataclass as its state. Its fields are preserved under
    ``info.backend``.

    All Mujoco Playground environments have continuous actions and observations, which
    range between `(-1, 1)` and `(-inf, inf)` respectively.

    Args:
        mujoco_playground_env (MjxEnv): the Mujoco Playground environment.
    """

    mujoco_playground_env: MjxEnv = static_field(unsafe=True)
    _default_max_steps: int | None = static_field(default=None)

    @classmethod
    def from_name(
        cls,
        env_name: str,
        env_kwargs: dict[str, Any] | None = None,
    ) -> "MujocoPlaygroundEnvelope":
        """
        Create a `MujocoPlaygroundEnvelope` from a name and keyword arguments.
        `env_kwargs` are passed to `config_overrides` of
        `mujoco_playground.registry.load`.
        """
        warn_if_wrapper_overlap("MuJoCo Playground", env_kwargs, ("episode_length",))

        env_kwargs = env_kwargs or {}
        default_config = registry.get_default_config(env_name)
        default_max_steps = _capture_horizon(
            env_kwargs.get("episode_length", default_config.episode_length)
        )

        # MuJoCo Playground requires an integer episode length.
        config_overrides = {"episode_length": _MAX_INT, **env_kwargs}
        env = registry.load(env_name, config_overrides=config_overrides)
        return cls(mujoco_playground_env=env, _default_max_steps=default_max_steps)

    @property
    def default_max_steps(self) -> int | None:
        return self._default_max_steps

    @override
    def init(self, key: Key) -> tuple[State, Info]:
        state = self.mujoco_playground_env.reset(key)
        info = InfoContainer(
            obs=state.obs,
            reward=state.reward,
            terminated=jnp.asarray(state.done, dtype=bool),
        )
        info = info.update(backend=backend_container(state))
        return state, info

    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        state = self.mujoco_playground_env.step(state, action)
        info = InfoContainer(
            obs=state.obs,
            reward=state.reward,
            terminated=jnp.asarray(state.done, dtype=bool),
        )
        info = info.update(backend=backend_container(state))
        return state, info

    @cached_property
    @override
    def action_space(self) -> envelope_spaces.Space:
        # MuJoCo Playground actions are typically bounded [-1, 1]
        return envelope_spaces.Continuous.from_shape(
            low=-1.0, high=1.0, shape=(self.mujoco_playground_env.action_size,)
        )

    @cached_property
    @override
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
