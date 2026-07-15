from functools import cached_property
from typing import Any, override

import jax.numpy as jnp
import navix
from navix import spaces as navix_spaces
from navix.entities import EntityIds
from navix.environments.environment import Environment as NavixEnv

from envelope import spaces as envelope_spaces
from envelope.adapters._common import backend_container
from envelope.environment import Environment, Info, InfoContainer, State
from envelope.struct import static_field
from envelope.typing import Key, PyTree

_MAX_INT = int(jnp.iinfo(jnp.int32).max)
_ENTITY_ID_CARDINALITY = (
    max(int(value) for name, value in vars(EntityIds).items() if name.isupper()) + 1
)


class NavixEnvelope(Environment):
    """
    Wrapper to convert a Navix environment to a envelope environment.

    Navix uses a dataclass timestep, which is preserved under ``info.backend``.

    Attributes:
        navix_env (NavixEnv): the Navix environment.
    """

    navix_env: NavixEnv = static_field(unsafe=True)
    _default_max_steps: int | None = static_field(default=None)

    @classmethod
    def from_name(
        cls, env_name: str, env_kwargs: dict[str, Any] | None = None
    ) -> "NavixEnvelope":
        """
        Create a `NavixEnvelope` from a name and keyword arguments.
        `env_kwargs` are passed to `navix.make`.
        """
        env_kwargs = {} if env_kwargs is None else dict(env_kwargs)
        if "max_steps" in env_kwargs:
            raise ValueError(
                "Cannot override 'max_steps' directly. "
                "Use TruncationWrapper for episode length control."
            )
        navix_env = navix.make(env_name, **env_kwargs)
        return cls(navix_env=navix_env)

    def __post_init__(self):
        default_max_steps = self._default_max_steps
        if default_max_steps is None:
            default_max_steps = int(self.navix_env.max_steps)
            object.__setattr__(
                self, "navix_env", self.navix_env.replace(max_steps=_MAX_INT)
            )
            object.__setattr__(self, "_default_max_steps", default_max_steps)

    @property
    def default_max_steps(self) -> int:
        assert self._default_max_steps is not None
        return self._default_max_steps

    @property
    def init_can_replace_reset(self) -> bool:
        return True

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
        if n.ndim > 0:
            entity_n = jnp.maximum(n[..., 0], _ENTITY_ID_CARDINALITY)
            n = n.at[..., 0].set(entity_n)
        return obs_space.replace(n=n)


def convert_navix_to_envelope_info(nvx_timestep: navix.Timestep) -> InfoContainer:
    info = InfoContainer(
        obs=nvx_timestep.observation,
        reward=nvx_timestep.reward,
        terminated=nvx_timestep.step_type == navix.StepType.TERMINATION,
        truncated=nvx_timestep.step_type == navix.StepType.TRUNCATION,
    ).update(backend=backend_container(nvx_timestep))
    return info


def convert_navix_to_envelope_space(
    nvx_space: navix_spaces.Space,
) -> envelope_spaces.Space:
    if isinstance(nvx_space, navix_spaces.Discrete):
        n = jnp.asarray(nvx_space.n).astype(nvx_space.dtype)
        return envelope_spaces.Discrete(n=jnp.broadcast_to(n, nvx_space.shape))

    elif isinstance(nvx_space, navix_spaces.Continuous):
        low = jnp.asarray(nvx_space.minimum).astype(nvx_space.dtype)
        high = jnp.asarray(nvx_space.maximum).astype(nvx_space.dtype)
        return envelope_spaces.Continuous(
            low=jnp.broadcast_to(low, nvx_space.shape),
            high=jnp.broadcast_to(high, nvx_space.shape),
        )

    raise ValueError(f"Unsupported space type: {type(nvx_space)}")
