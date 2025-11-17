from functools import cached_property
from typing import override

import jax
import jax.numpy as jnp
from gymnax.environments import spaces as gymnax_spaces
from gymnax.environments.environment import Environment as GymnaxEnv

from jenv import spaces as jenv_spaces
from jenv.environment import Environment, Info, InfoContainer, State
from jenv.struct import static_field
from jenv.typing import Key, PyTree


class GymnaxContainer(InfoContainer):
    @property
    def terminated(self) -> bool:
        return self.done


class GymnaxJenv(Environment):
    """Wrapper to convert a Gymnax environment to a jenv environment."""

    gymnax_env: GymnaxEnv = static_field()
    env_params: PyTree

    @override
    def reset(self, key: Key) -> tuple[State, Info]:
        key, subkey = jax.random.split(key)
        obs, env_state = self.gymnax_env.reset(subkey, self.env_params)
        state = GymnaxContainer(key=key, env_state=env_state)
        return state, InfoContainer(obs=obs, reward=0.0, terminated=False)

    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        key, subkey = jax.random.split(state.key)
        obs, env_state, reward, done, info = self.gymnax_env.step(
            subkey, state.env_state, action, self.env_params
        )
        state = GymnaxContainer(key=key, env_state=env_state)
        info = InfoContainer(obs=obs, reward=reward, terminated=done, info=info)
        return state, info

    @override
    @cached_property
    def action_space(self) -> jenv_spaces.Space:
        return _convert_space(self.gymnax_env.action_space(self.env_params))

    @override
    @cached_property
    def observation_space(self) -> jenv_spaces.Space:
        return _convert_space(self.gymnax_env.observation_space(self.env_params))


def _convert_space(gmnx_space: gymnax_spaces.Space) -> jenv_spaces.Space:
    if isinstance(gmnx_space, gymnax_spaces.Box):
        low = gmnx_space.low
        high = gmnx_space.high
        if jnp.asarray(low).shape == () and jnp.asarray(high).shape == ():
            return jenv_spaces.Continuous(
                low=low, high=high, shape=gmnx_space.shape, dtype=gmnx_space.dtype
            )
        else:
            return jenv_spaces.Continuous(low=low, high=high, dtype=gmnx_space.dtype)
    elif isinstance(gmnx_space, gymnax_spaces.Discrete):
        return jenv_spaces.Discrete(
            n=gmnx_space.n,
            shape=gmnx_space.shape,
            dtype=gmnx_space.dtype,
        )
    elif isinstance(gmnx_space, gymnax_spaces.Tuple):
        spaces = tuple(_convert_space(space) for space in gmnx_space.spaces)
        return jenv_spaces.PyTreeSpace(spaces)
    elif isinstance(gmnx_space, gymnax_spaces.Dict):
        spaces = {
            name: _convert_space(space) for name, space in gmnx_space.spaces.items()
        }
        return jenv_spaces.PyTreeSpace(spaces)
    raise ValueError(f"Unsupported space type: {type(gmnx_space)}")
