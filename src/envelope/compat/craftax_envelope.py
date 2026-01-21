from functools import cached_property
from typing import Any, override

import jax
import jax.numpy as jnp
from craftax.craftax.craftax_state import EnvParams as CraftaxEnvParams
from craftax.craftax_classic.envs.craftax_state import (
    EnvParams as CraftaxClassicEnvParams,
)
from craftax.craftax_env import make_craftax_env_from_name

from envelope import spaces as envelope_spaces
from envelope.compat.gymnax_envelope import _convert_space as _convert_gymnax_space
from envelope.environment import Environment, Info, InfoContainer, State
from envelope.struct import Container, static_field
from envelope.typing import Key, PyTree, TypeAlias

EnvParams: TypeAlias = CraftaxEnvParams | CraftaxClassicEnvParams


class CraftaxEnvelope(Environment):
    """Wrapper to convert a Craftax environment to a envelope environment."""

    craftax_env: Any = static_field()
    env_params: PyTree

    @classmethod
    def from_name(
        cls,
        env_name: str,
        env_params: EnvParams | None = None,
        env_kwargs: dict[str, Any] | None = None,
    ) -> "CraftaxEnvelope":
        env_kwargs = env_kwargs or {}
        if "max_timesteps" in env_kwargs:
            raise ValueError(
                "Cannot override 'max_timesteps' directly. "
                "Use TruncationWrapper for episode length control."
            )
        if "auto_reset" in env_kwargs:
            raise ValueError(
                "Cannot override 'auto_reset' directly. "
                "Use AutoResetWrapper for auto-reset behavior."
            )

        env_kwargs["auto_reset"] = False
        env = make_craftax_env_from_name(env_name, **env_kwargs)
        default_params = env.default_params.replace(max_timesteps=jnp.inf)

        env_params = env_params or default_params
        return cls(craftax_env=env, env_params=env_params)

    @property
    def default_max_steps(self) -> int:
        return int(self.craftax_env.default_params.max_timesteps)

    @override
    def reset(self, key: Key) -> tuple[State, Info]:
        key, subkey = jax.random.split(key)
        obs, env_state = self.craftax_env.reset(subkey, self.env_params)
        state = Container().update(key=key, env_state=env_state)
        info = InfoContainer(obs=obs, reward=0.0, terminated=False)
        return state, info

    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        key, subkey = jax.random.split(state.key)
        obs, env_state, reward, done, env_info = self.craftax_env.step(
            subkey, state.env_state, action, self.env_params
        )
        state = state.update(key=key, env_state=env_state)
        info = InfoContainer(obs=obs, reward=reward, terminated=done)
        info = info.update(info=env_info)
        return state, info

    @override
    @cached_property
    def action_space(self) -> envelope_spaces.Space:
        return _convert_gymnax_space(self.craftax_env.action_space(self.env_params))

    @override
    @cached_property
    def observation_space(self) -> envelope_spaces.Space:
        return _convert_gymnax_space(
            self.craftax_env.observation_space(self.env_params)
        )
