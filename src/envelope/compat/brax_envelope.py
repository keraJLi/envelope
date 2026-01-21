import dataclasses
import warnings
from copy import copy
from functools import cached_property
from typing import Any, override

from brax.envs import Env as BraxEnv
from brax.envs import Wrapper as BraxWrapper
from brax.envs import create as brax_create
from jax import numpy as jnp

from envelope import spaces
from envelope.environment import Environment, Info, InfoContainer, State
from envelope.struct import static_field
from envelope.typing import Key, PyTree

# Default episode_length in brax.envs.create()
_BRAX_DEFAULT_EPISODE_LENGTH = 1000


class BraxEnvelope(Environment):
    """Wrapper to convert a Brax environment to a envelope environment."""

    brax_env: BraxEnv = static_field()

    @classmethod
    def from_name(
        cls, env_name: str, env_kwargs: dict[str, Any] | None = None
    ) -> "BraxEnvelope":
        env_kwargs = env_kwargs or {}
        if "episode_length" in env_kwargs:
            raise ValueError(
                "Cannot override 'episode_length' directly. "
                "Use TruncationWrapper for episode length control."
            )
        if "auto_reset" in env_kwargs:
            raise ValueError(
                "Cannot override 'auto_reset' directly. "
                "Use AutoResetWrapper for auto-reset behavior."
            )

        env_kwargs["episode_length"] = jnp.inf
        env_kwargs["auto_reset"] = False
        env = brax_create(env_name, **env_kwargs)
        return cls(brax_env=env)

    @property
    def default_max_steps(self) -> int:
        return _BRAX_DEFAULT_EPISODE_LENGTH

    def __post_init__(self) -> "BraxEnvelope":
        if isinstance(self.brax_env, BraxWrapper):
            warnings.warn(
                "Environment wrapping should be handled by envelope. "
                "Unwrapping brax environment before converting..."
            )
            object.__setattr__(self, "brax_env", self.brax_env.unwrapped)

    @override
    def reset(self, key: Key) -> tuple[State, Info]:
        brax_state = self.brax_env.reset(key)
        info = InfoContainer(obs=brax_state.obs, reward=0.0, terminated=False)
        info = info.update(**dataclasses.asdict(brax_state))
        return brax_state, info

    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        brax_state = self.brax_env.step(state, action)
        info = InfoContainer(
            obs=brax_state.obs, reward=brax_state.reward, terminated=brax_state.done
        )
        info = info.update(**dataclasses.asdict(brax_state))
        return brax_state, info

    @override
    @cached_property
    def action_space(self) -> spaces.Space:
        # All brax environments have action limit of -1 to 1
        return spaces.Continuous.from_shape(
            low=-1.0, high=1.0, shape=(self.brax_env.action_size,)
        )

    @override
    @cached_property
    def observation_space(self) -> spaces.Space:
        # All brax environments have observation limit of -inf to inf
        return spaces.Continuous.from_shape(
            low=-jnp.inf, high=jnp.inf, shape=(self.brax_env.observation_size,)
        )

    def __deepcopy__(self, memo):
        warnings.warn(
            f"Trying to deepcopy {type(self).__name__}, which contains a brax env. "
            "Brax envs throw an error when deepcopying, so a shallow copy is returned.",
            category=RuntimeWarning,
            stacklevel=2,
        )
        return copy(self)
