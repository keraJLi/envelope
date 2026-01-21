from functools import cached_property
from typing import Any, override

import jax
import jax.numpy as jnp
from gymnax import make as gymnax_create
from gymnax.environments import spaces as gymnax_spaces
from gymnax.environments.environment import Environment as GymnaxEnv
from gymnax.environments.environment import EnvParams as GymnaxEnvParams

from envelope import spaces as envelope_spaces
from envelope.environment import Environment, Info, InfoContainer, State
from envelope.struct import Container, static_field
from envelope.typing import Key, PyTree


class GymnaxEnvelope(Environment):
    """Wrapper to convert a Gymnax environment to a envelope environment."""

    gymnax_env: GymnaxEnv = static_field()
    env_params: PyTree

    @classmethod
    def from_name(
        cls,
        env_name: str,
        env_params: GymnaxEnvParams | None = None,
        env_kwargs: dict[str, Any] | None = None,
    ) -> "GymnaxEnvelope":
        env_kwargs = env_kwargs or {}
        if "max_steps_in_episode" in env_kwargs:
            raise ValueError(
                "Cannot override 'max_steps_in_episode' directly. "
                "Use TruncationWrapper for episode length control."
            )
        gymnax_env, default_params = gymnax_create(env_name, **env_kwargs)
        default_params = default_params.replace(max_steps_in_episode=jnp.inf)

        env_params = env_params or default_params
        return cls(gymnax_env=gymnax_env, env_params=env_params)

    @property
    def default_max_steps(self) -> int:
        return int(self.gymnax_env.default_params.max_steps_in_episode)

    @override
    def reset(self, key: Key) -> tuple[State, Info]:
        key, subkey = jax.random.split(key)
        obs, env_state = self.gymnax_env.reset(subkey, self.env_params)
        state = Container().update(key=key, env_state=env_state)
        info = InfoContainer(obs=obs, reward=0.0, terminated=False)
        info = info.update(info=None)
        return state, info

    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        key, subkey = jax.random.split(state.key)
        obs, env_state, reward, done, env_info = self.gymnax_env.step(
            subkey, state.env_state, action, self.env_params
        )
        state = state.update(key=key, env_state=env_state)
        info = InfoContainer(obs=obs, reward=reward, terminated=done)
        info = info.update(info=env_info)
        return state, info

    @override
    @cached_property
    def action_space(self) -> envelope_spaces.Space:
        return _convert_space(self.gymnax_env.action_space(self.env_params))

    @override
    @cached_property
    def observation_space(self) -> envelope_spaces.Space:
        return _convert_space(self.gymnax_env.observation_space(self.env_params))


def _convert_space(gmx_space: gymnax_spaces.Space) -> envelope_spaces.Space:
    if isinstance(gmx_space, gymnax_spaces.Box):
        low = jnp.broadcast_to(gmx_space.low, gmx_space.shape).astype(gmx_space.dtype)
        high = jnp.broadcast_to(gmx_space.high, gmx_space.shape).astype(gmx_space.dtype)
        return envelope_spaces.Continuous(low=low, high=high)
    elif isinstance(gmx_space, gymnax_spaces.Discrete):
        n = jnp.broadcast_to(gmx_space.n, gmx_space.shape).astype(gmx_space.dtype)
        return envelope_spaces.Discrete(n=n)
    elif isinstance(gmx_space, gymnax_spaces.Tuple):
        spaces = tuple(_convert_space(space) for space in gmx_space.spaces)
        return envelope_spaces.PyTreeSpace(spaces)
    elif isinstance(gmx_space, gymnax_spaces.Dict):
        spaces = {k: _convert_space(space) for k, space in gmx_space.spaces.items()}
        return envelope_spaces.PyTreeSpace(spaces)
    raise ValueError(f"Unsupported space type: {type(gmx_space)}")
