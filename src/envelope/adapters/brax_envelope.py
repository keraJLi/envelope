import warnings
from copy import copy
from functools import cached_property
from typing import Any, cast, override

import brax.envs as brax_envs
from brax.envs import Env as BraxEnv
from brax.envs import create as brax_create
from jax import numpy as jnp

from envelope import spaces
from envelope.adapters._common import (
    _capture_horizon,
    backend_container,
    warn_if_wrapper_overlap,
)
from envelope.environment import Environment, Info, InfoContainer, State
from envelope.struct import static_field
from envelope.typing import Key, PyTree

# Default episode_length in brax.envs.create()
_BRAX_DEFAULT_EPISODE_LENGTH = 1000
_CONTROLS = ("episode_length", "auto_reset", "batch_size", "action_repeat")


def _brax_state_to_info(brax_state: Any) -> InfoContainer:
    """Convert a brax state to an envelope InfoContainer."""
    info = InfoContainer(
        obs=brax_state.obs,
        reward=brax_state.reward,
        terminated=jnp.asarray(brax_state.done, dtype=jnp.bool_),
    )
    return info.update(backend=backend_container(brax_state))


class BraxEnvelope(Environment):
    """
    Wrapper to convert a Brax environment to a envelope environment.

    Brax' `create` function defaults to an episode length of `1000`. This horizon, or
    an explicitly supplied override, is retained as `default_max_steps` while the
    backend time limit is disabled by default.

    Brax uses a dataclass as its state. Its fields are preserved under ``info.backend``.

    Args:
        brax_env (BraxEnv): the Brax environment.
    """

    brax_env: BraxEnv = static_field(unsafe=True)
    _max_steps: int | None = static_field(default=_BRAX_DEFAULT_EPISODE_LENGTH)

    @classmethod
    def registered_names(cls) -> tuple[str, ...]:
        return tuple(brax_envs._envs)

    @classmethod
    def from_name(
        cls, env_name: str, env_kwargs: dict[str, Any] | None = None
    ) -> "BraxEnvelope":
        """
        Create a `BraxEnvelope` from a name and keyword arguments. `env_kwargs` are
        passed to `brax.envs.create`.
        """
        warn_if_wrapper_overlap("Brax", env_kwargs, _CONTROLS)

        env_kwargs = env_kwargs or {}
        default_max_steps = _capture_horizon(
            env_kwargs.get("episode_length", _BRAX_DEFAULT_EPISODE_LENGTH)
        )
        backend_kwargs = {"episode_length": None, "auto_reset": False, **env_kwargs}
        env = cast(Any, brax_create)(env_name, **backend_kwargs)
        return cls(brax_env=env, _max_steps=default_max_steps)

    @property
    def default_max_steps(self) -> int | None:
        return self._max_steps

    @override
    def init(self, key: Key) -> tuple[State, Info]:
        brax_state = self.brax_env.reset(key)
        info = _brax_state_to_info(brax_state)
        return brax_state, info

    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        brax_state = self.brax_env.step(state, action)
        info = _brax_state_to_info(brax_state)
        return brax_state, info

    @cached_property
    @override
    def action_space(self) -> spaces.Space:
        # All brax environments have action limit of -1 to 1
        shape = (self.brax_env.action_size,)
        return spaces.Continuous.from_shape(low=-1.0, high=1.0, shape=shape)

    @cached_property
    @override
    def observation_space(self) -> spaces.Space:
        # All brax environments have observation limit of -inf to inf
        obs_size = cast(int, self.brax_env.observation_size)
        return spaces.Continuous.from_shape(
            low=-jnp.inf, high=jnp.inf, shape=(obs_size,)
        )

    def __deepcopy__(self, memo):
        warnings.warn(
            f"Trying to deepcopy {type(self).__name__}, which contains a brax env. "
            "Brax envs throw an error when deepcopying, so a shallow copy is returned.",
            category=RuntimeWarning,
            stacklevel=2,
        )
        return copy(self)
