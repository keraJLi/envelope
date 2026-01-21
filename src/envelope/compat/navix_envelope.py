import dataclasses
from functools import cached_property
from typing import Any, override

import jax.numpy as jnp
import navix
from navix import spaces as navix_spaces
from navix.environments.environment import Environment as NavixEnv

from envelope import spaces as envelope_spaces
from envelope.environment import Environment, Info, InfoContainer, State
from envelope.typing import Key, PyTree

_NAVIX_DEFAULT_MAX_STEPS = 100


class NavixEnvelope(Environment):
    """Wrapper to convert a Navix environment to a envelope environment."""

    navix_env: NavixEnv

    @classmethod
    def from_name(
        cls, env_name: str, env_kwargs: dict[str, Any] | None = None
    ) -> "NavixEnvelope":
        env_kwargs = env_kwargs or {}
        if "max_steps" in env_kwargs:
            raise ValueError(
                "Cannot override 'max_steps' directly. "
                "Use TruncationWrapper for episode length control."
            )
        env_kwargs["max_steps"] = jnp.inf
        navix_env = navix.make(env_name, **env_kwargs)
        return cls(navix_env=navix_env)

    @property
    def default_max_steps(self) -> int:
        return _NAVIX_DEFAULT_MAX_STEPS

    @override
    def reset(self, key: Key) -> tuple[State, Info]:
        timestep = self.navix_env.reset(key)
        return timestep, convert_navix_to_envelope_info(timestep)

    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        timestep = self.navix_env.step(state, action)
        return timestep, convert_navix_to_envelope_info(timestep)

    @override
    @cached_property
    def action_space(self) -> envelope_spaces.Space:
        return convert_navix_to_envelope_space(self.navix_env.action_space)

    @override
    @cached_property
    def observation_space(self) -> envelope_spaces.Space:
        return convert_navix_to_envelope_space(self.navix_env.observation_space)


def convert_navix_to_envelope_info(nvx_timestep: navix.Timestep) -> InfoContainer:
    timestep_dict = dataclasses.asdict(nvx_timestep)
    step_type = timestep_dict.pop("step_type")
    info = InfoContainer(
        obs=timestep_dict.pop("observation"),
        reward=timestep_dict.pop("reward"),
        terminated=step_type == navix.StepType.TERMINATION,
        truncated=step_type == navix.StepType.TRUNCATION,
    )
    info = info.update(**timestep_dict)
    return info


def convert_navix_to_envelope_space(
    nvx_space: navix_spaces.Space,
) -> envelope_spaces.Space:
    if isinstance(nvx_space, navix_spaces.Discrete):
        n = jnp.asarray(nvx_space.n).astype(nvx_space.dtype)
        return envelope_spaces.Discrete.from_shape(n, shape=nvx_space.shape)

    elif isinstance(nvx_space, navix_spaces.Continuous):
        low = jnp.asarray(nvx_space.minimum).astype(nvx_space.dtype)
        high = jnp.asarray(nvx_space.maximum).astype(nvx_space.dtype)
        return envelope_spaces.Continuous.from_shape(low, high, shape=nvx_space.shape)

    raise ValueError(f"Unsupported space type: {type(nvx_space)}")
