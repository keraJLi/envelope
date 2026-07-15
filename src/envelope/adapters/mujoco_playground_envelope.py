from functools import cached_property
from typing import Any, cast, override

from jax import numpy as jnp
from mujoco_playground import MjxEnv, registry

from envelope import spaces as envelope_spaces
from envelope.adapters._common import backend_container
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

    Attributes:
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
        env_kwargs = {} if env_kwargs is None else dict(env_kwargs)
        if "episode_length" in env_kwargs:
            raise ValueError(
                "Cannot override 'episode_length' directly. "
                "Use TruncationWrapper for episode length control."
            )

        # Get default episode_length from registry config
        default_config = registry.get_default_config(env_name)
        default_episode_length = default_config.episode_length
        if (
            isinstance(default_episode_length, bool)
            or not isinstance(default_episode_length, int)
            or default_episode_length <= 0
        ):
            raise RuntimeError(
                "MuJoCo Playground must expose a finite positive default horizon"
            )
        default_max_steps = cast(int, default_episode_length)

        # Set episode_length to a very large value
        # (mujoco_playground uses int for episode_length, so we use max int instead of inf)
        env_kwargs["episode_length"] = _MAX_INT

        # Pass all env_kwargs as config_overrides
        config_overrides = env_kwargs if env_kwargs else None
        env = registry.load(env_name, config_overrides=config_overrides)
        return cls(mujoco_playground_env=env, _default_max_steps=default_max_steps)

    @property
    def default_max_steps(self) -> int | None:
        return self._default_max_steps

    @property
    def supports_init_pooling(self) -> bool:
        return True

    @override
    def init(self, key: Key) -> tuple[State, Info]:
        state = self.mujoco_playground_env.reset(key)
        info = InfoContainer(
            obs=state.obs,
            reward=state.reward,
            terminated=jnp.asarray(state.done, dtype=bool),
        ).update(backend=backend_container(state))
        return state, info

    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        state = self.mujoco_playground_env.step(state, action)
        info = InfoContainer(
            obs=state.obs,
            reward=state.reward,
            terminated=jnp.asarray(state.done, dtype=bool),
        ).update(backend=backend_container(state))
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
