import warnings
from copy import copy
from dataclasses import InitVar
from functools import cached_property
from typing import override

from brax.envs import Env as BraxEnv
from brax.envs import create as brax_create
from jax import numpy as jnp

from jenv import spaces
from jenv.environment import Environment, State, StepInfo
from jenv.struct import static_field
from jenv.typing import Key, PyTree


class BraxJenv(Environment):
    """Wrapper to convert a Brax environment to a jenv environment."""

    env_name: InitVar[str]
    brax_env: BraxEnv = static_field(init=False)

    def __post_init__(self, env_name: str) -> "BraxJenv":
        brax_env = brax_create(env_name, episode_length=None, auto_reset=False)
        object.__setattr__(self, "brax_env", brax_env)

    @override
    def reset(self, key: Key) -> tuple[State, StepInfo]:
        brax_state = self.brax_env.reset(key)
        return brax_state, brax_state

    @override
    def step(self, state: State, action: PyTree) -> tuple[State, StepInfo]:
        brax_state = self.brax_env.step(state, action)
        return brax_state, brax_state

    @override
    @cached_property
    def action_space(self) -> spaces.Space:
        # All brax environments have action limit of -1 to 1
        return spaces.Continuous(low=-1.0, high=1.0, shape=(self.brax_env.action_size,))

    @override
    @cached_property
    def observation_space(self) -> spaces.Space:
        # All brax environments have observation limit of -inf to inf
        return spaces.Continuous(
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
