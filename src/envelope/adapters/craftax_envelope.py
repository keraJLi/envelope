from functools import cached_property
from typing import Any, cast, override

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

from envelope import spaces as envelope_spaces
from envelope.adapters._common import backend_container, placeholder_like
from envelope.adapters.gymnax_envelope import _convert_space as _convert_gymnax_space
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


def _probe_backend_placeholder(
    craftax_env: CraftaxEnv, env_params: PyTree
) -> tuple[Container, PyTree]:
    """Probe Craftax's step-only info schema outside transformed execution."""
    key = jax.random.key(0)
    observation, state = craftax_env.reset(key, env_params)
    _, _, _, _, raw_backend = craftax_env.step(
        key,
        state,
        craftax_env.action_space(env_params).sample(key),
        env_params,
    )
    placeholder = cast(Container, placeholder_like(backend_container(raw_backend)))
    return placeholder.update(valid=jnp.asarray(False)), observation


def _normalize_observation_space(
    declared_space: envelope_spaces.Space, observation: PyTree
) -> envelope_spaces.Space:
    """Repair backend declarations that disagree with emitted pixel dimensions."""
    if bool(declared_space.contains(observation)):
        return declared_space
    if not isinstance(declared_space, envelope_spaces.Continuous):
        raise ValueError("Craftax observation does not match its declared space")

    observation_array = jnp.asarray(observation)
    if not jnp.issubdtype(observation_array.dtype, jnp.floating):
        raise ValueError("Craftax continuous observations must have a floating dtype")

    low = jnp.full(
        observation_array.shape,
        jnp.min(declared_space.low),
        dtype=observation_array.dtype,
    )
    high = jnp.full(
        observation_array.shape,
        jnp.max(declared_space.high),
        dtype=observation_array.dtype,
    )
    normalized = envelope_spaces.Continuous(low=low, high=high)
    if not bool(normalized.contains(observation)):
        raise ValueError("Craftax observation falls outside its declared bounds")
    return normalized


class CraftaxEnvelope(Environment):
    """
    Wrapper to convert a Craftax environment to a envelope environment.

    Craftax (mostly) uses the Gymnax interface, so backend info is only created on the
    first `step`. Construction probes that schema and stores a type-preserving zero-like
    placeholder; ``info.backend.valid`` distinguishes reset placeholders from real
    step metadata.

    Attributes:
        craftax_env (CraftaxEnv): the Craftax environment, with baked-in
            `static_env_params`.
        env_params (CraftaxEnvParams): the environment parameters, which are passed to
            the Craftax environment's `reset` and `step` methods.
    """

    craftax_env: CraftaxEnv = static_field(unsafe=True)
    # TODO: make this dynamic once Craftax's EnvParams is a fully valid pytree.
    env_params: PyTree = static_field(unsafe=True)
    _default_max_steps: int = static_field()
    _backend_placeholder: Container = field()
    _observation_space: envelope_spaces.Space = field()

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
        env_kwargs = {} if env_kwargs is None else dict(env_kwargs)
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
        selected_params = env.default_params if env_params is None else env_params
        default_max_steps = int(selected_params.max_timesteps)
        selected_params = selected_params.replace(max_timesteps=jnp.inf)
        backend_placeholder, initial_observation = _probe_backend_placeholder(
            env, selected_params
        )
        declared_observation_space = _convert_gymnax_space(
            env.observation_space(selected_params)
        )
        observation_space = _normalize_observation_space(
            declared_observation_space, initial_observation
        )
        return cls(
            craftax_env=env,
            env_params=selected_params,
            _default_max_steps=default_max_steps,
            _backend_placeholder=backend_placeholder,
            _observation_space=observation_space,
        )

    @property
    def default_max_steps(self) -> int:
        return self._default_max_steps

    @property
    def supports_init_pooling(self) -> bool:
        return True

    @override
    def init(self, key: Key) -> tuple[State, Info]:
        key, subkey = jax.random.split(key)
        obs, env_state = self.craftax_env.reset(subkey, self.env_params)
        state = Container().update(key=key, env_state=env_state)
        info = InfoContainer(obs=obs, reward=0.0, terminated=False).update(
            backend=self._backend_placeholder
        )
        return state, info

    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        key, subkey = jax.random.split(state.key)
        obs, env_state, reward, done, env_info = self.craftax_env.step(
            subkey, state.env_state, action, self.env_params
        )
        state = state.update(key=key, env_state=env_state)
        info = InfoContainer(obs=obs, reward=reward, terminated=done).update(
            backend=backend_container(env_info).update(valid=jnp.asarray(True))
        )
        return state, info

    @cached_property
    @override
    def action_space(self) -> envelope_spaces.Space:
        return _convert_gymnax_space(self.craftax_env.action_space(self.env_params))

    @cached_property
    @override
    def observation_space(self) -> envelope_spaces.Space:
        return self._observation_space
