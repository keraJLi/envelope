from functools import cached_property
from typing import Any, override

import jax
import jax.numpy as jnp
from craftax.craftax.craftax_state import EnvParams as CraftaxModernEnvParams
from craftax.craftax.envs.craftax_pixels_env import (
    CraftaxPixelsEnv,
    CraftaxPixelsEnvNoAutoReset,
)
from craftax.craftax.envs.craftax_symbolic_env import (
    CraftaxSymbolicEnv,
    CraftaxSymbolicEnvNoAutoReset,
)
from craftax.craftax_classic.envs.craftax_pixels_env import (
    CraftaxClassicPixelsEnv,
    CraftaxClassicPixelsEnvNoAutoReset,
)
from craftax.craftax_classic.envs.craftax_state import (
    EnvParams as CraftaxClassicEnvParams,
)
from craftax.craftax_classic.envs.craftax_symbolic_env import (
    CraftaxClassicSymbolicEnv,
    CraftaxClassicSymbolicEnvNoAutoReset,
)
from craftax.craftax_env import make_craftax_env_from_name
from craftax.environment_base import spaces as craftax_spaces

from envelope import spaces as envelope_spaces
from envelope.adapters._common import (
    _capture_horizon,
    _convert_gymnaxlike_space,
    _probe_gymnaxlike_info_placeholder,
    _warn_preserved_horizon,
    backend_container,
    warn_if_wrapper_overlap,
)
from envelope.environment import Environment, Info, InfoContainer, State
from envelope.struct import Container, field, static_field
from envelope.typing import Key, PyTree, TypeAlias

CraftaxEnvParams: TypeAlias = CraftaxModernEnvParams | CraftaxClassicEnvParams
CraftaxEnv: TypeAlias = (
    CraftaxPixelsEnv
    | CraftaxSymbolicEnv
    | CraftaxClassicPixelsEnv
    | CraftaxClassicSymbolicEnv
    | CraftaxPixelsEnvNoAutoReset
    | CraftaxSymbolicEnvNoAutoReset
    | CraftaxClassicPixelsEnvNoAutoReset
    | CraftaxClassicSymbolicEnvNoAutoReset
)
_CONTROLS = ("auto_reset",)


def _convert_space(space: craftax_spaces.Space) -> envelope_spaces.Space:
    return _convert_gymnaxlike_space(
        space,
        box_type=craftax_spaces.Box,
        discrete_type=craftax_spaces.Discrete,
    )


class CraftaxEnvelope(Environment):
    """
    Wrapper to convert a Craftax environment to a envelope environment.

    Craftax (mostly) uses the Gymnax interface, so backend info is only created on the
    first `step`. Construction probes that schema and stores a type-preserving zero-like
    placeholder; ``info.backend.valid`` distinguishes reset placeholders from real
    step metadata.

    Args:
        craftax_env (CraftaxEnv): the Craftax environment, with baked-in
            `static_env_params`.
        env_params (CraftaxEnvParams): the environment parameters, which are passed to
            the Craftax environment's `reset` and `step` methods.
    """

    craftax_env: CraftaxEnv = static_field(unsafe=True)
    env_params: PyTree = field()
    _max_steps: int | None = static_field()
    _empty_backend_info: Container = field()

    @classmethod
    def registered_names(cls) -> tuple[str, ...]:
        return (
            "Craftax-Symbolic-v1",
            "Craftax-Pixels-v1",
            "Craftax-Classic-Symbolic-v1",
            "Craftax-Classic-Pixels-v1",
        )

    @classmethod
    def from_name(
        cls,
        env_name: str,
        env_params: CraftaxEnvParams | None = None,
        env_kwargs: dict[str, Any] | None = None,
    ) -> "CraftaxEnvelope":
        """
        Create a `CraftaxEnvelope` from a name and keyword arguments.
        `env_kwargs` are passed to `craftax.craftax_env.make_craftax_env_from_name`.
        """
        warn_if_wrapper_overlap("Craftax", env_kwargs, _CONTROLS)

        env_kwargs = env_kwargs or {}
        backend_kwargs = {"auto_reset": False, **env_kwargs}
        env = make_craftax_env_from_name(env_name, **backend_kwargs)
        if env_params is None:
            default_max_steps = _capture_horizon(env.default_params.max_timesteps)
            env_params = env.default_params.replace(max_timesteps=jnp.inf)
        else:
            default_max_steps = _capture_horizon(env_params.max_timesteps)
            if default_max_steps is not None:
                _warn_preserved_horizon("Craftax", "env_params.max_timesteps")

        empty_backend_info = _probe_gymnaxlike_info_placeholder(env, env_params)
        return cls(
            craftax_env=env,
            env_params=env_params,
            _max_steps=default_max_steps,
            _empty_backend_info=empty_backend_info,
        )

    @property
    def default_max_steps(self) -> int | None:
        return self._max_steps

    @override
    def init(self, key: Key) -> tuple[State, Info]:
        key, subkey = jax.random.split(key)
        obs, env_state = self.craftax_env.reset(subkey, self.env_params)
        state = Container().update(key=key, env_state=env_state)
        info = InfoContainer(obs=obs, reward=0.0, terminated=False)
        info = info.update(backend=self._empty_backend_info)
        return state, info

    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        key, subkey = jax.random.split(state.key)
        obs, env_state, reward, done, env_info = self.craftax_env.step(
            subkey, state.env_state, action, self.env_params
        )
        state = state.update(key=key, env_state=env_state)
        backend = backend_container(env_info).update(valid=jnp.asarray(True))
        info = InfoContainer(obs=obs, reward=reward, terminated=done)
        info = info.update(backend=backend)
        return state, info

    @cached_property
    @override
    def action_space(self) -> envelope_spaces.Space:
        return _convert_space(self.craftax_env.action_space(self.env_params))

    @cached_property
    @override
    def observation_space(self) -> envelope_spaces.Space:
        return _convert_space(self.craftax_env.observation_space(self.env_params))
