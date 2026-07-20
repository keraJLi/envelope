import dataclasses
from functools import cached_property
from typing import Any, override

import jax.numpy as jnp
import navix
from navix import spaces as navix_spaces
from navix.entities import EntityIds
from navix.environments.environment import Environment as NavixEnv

from envelope import spaces as envelope_spaces
from envelope.adapters._common import _capture_horizon, warn_if_wrapper_overlap
from envelope.environment import Environment, Info, InfoContainer, State
from envelope.struct import static_field
from envelope.typing import Key, PyTree

_NAVIX_DEFAULT_MAX_STEPS = 100
_ENTITY_ID_CARDINALITY = (
    max(int(value) for name, value in vars(EntityIds).items() if name.isupper()) + 1
)


class NavixEnvelope(Environment):
    """
    Wrapper to convert a Navix environment to a envelope environment.

    Navix uses a dataclass timestep, which is preserved under ``info.backend``.

    Args:
        navix_env (NavixEnv): the Navix environment.
    """

    navix_env: NavixEnv = static_field(unsafe=True)
    _max_steps: int | None = static_field(default=None)

    @classmethod
    def registered_names(cls) -> tuple[str, ...]:
        return tuple(navix.registry())

    @classmethod
    def from_name(
        cls, env_name: str, env_kwargs: dict[str, Any] | None = None
    ) -> "NavixEnvelope":
        """
        Create a `NavixEnvelope` from a name and keyword arguments.
        `env_kwargs` are passed to `navix.make`.
        """
        warn_if_wrapper_overlap("Navix", env_kwargs, ("max_steps",))

        env_kwargs = env_kwargs or {}
        default_max_steps = _capture_horizon(
            env_kwargs.get("max_steps", _NAVIX_DEFAULT_MAX_STEPS)
        )
        backend_kwargs: dict[str, Any] = {"max_steps": jnp.inf, **env_kwargs}
        navix_env = navix.make(env_name, **backend_kwargs)
        return cls(navix_env=navix_env, _max_steps=default_max_steps)

    @property
    def default_max_steps(self) -> int | None:
        if self._max_steps is not None:
            return self._max_steps
        return _capture_horizon(self.navix_env.max_steps)

    @override
    def init(self, key: Key) -> tuple[State, Info]:
        timestep = self.navix_env.reset(key)
        return timestep, convert_navix_to_envelope_info(timestep)

    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        timestep = self.navix_env.step(state, action)
        return timestep, convert_navix_to_envelope_info(timestep)

    @cached_property
    @override
    def action_space(self) -> envelope_spaces.Space:
        return convert_navix_to_envelope_space(self.navix_env.action_space)

    @cached_property
    @override
    def observation_space(self) -> envelope_spaces.Space:
        nvx_obs_space = self.navix_env.observation_space
        obs_space = convert_navix_to_envelope_space(nvx_obs_space)
        if not isinstance(obs_space, envelope_spaces.Discrete):
            return obs_space
        n = jnp.asarray(obs_space.n)
        # Navix's first symbolic channel contains authoritative EntityIds, while its
        # declared cardinality currently omits newer ids such as PLAYER. Adjust only
        # that channel; colour/direction channels retain their backend declarations.
        # See https://github.com/epignatelli/navix/issues/109.
        if n.ndim > 0:
            entity_n = jnp.maximum(n[..., 0], _ENTITY_ID_CARDINALITY)
            n = n.at[..., 0].set(entity_n)
        return obs_space.replace(n=n)


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
        shape, dtype = nvx_space.shape, nvx_space.dtype
        low = jnp.broadcast_to(nvx_space.minimum, shape).astype(dtype)
        high = jnp.broadcast_to(nvx_space.maximum, shape).astype(dtype)
        return envelope_spaces.Continuous(low=low, high=high)

    raise ValueError(f"Unsupported space type: {type(nvx_space)}")
