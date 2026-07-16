from functools import cached_property
from typing import Any, Callable, cast, override

import jax
import jax.numpy as jnp
from gymnax import make as gymnax_create
from gymnax.environments import spaces as gymnax_spaces
from gymnax.environments.environment import Environment as GymnaxEnv
from gymnax.environments.environment import EnvParams as GymnaxEnvParams

from envelope import spaces as envelope_spaces
from envelope.adapters._common import (
    _capture_horizon,
    _convert_gymnaxlike_space,
    _probe_gymnaxlike_info_placeholder,
    _warn_preserved_horizon,
    backend_container,
)
from envelope.environment import Environment, Info, InfoContainer, State
from envelope.struct import Container, field, static_field
from envelope.typing import Key, PyTree

_GymnaxReset = Callable[
    [Key, GymnaxEnvParams],
    tuple[PyTree, Any],
]
_GymnaxStep = Callable[
    [Key, Any, PyTree, GymnaxEnvParams],
    tuple[PyTree, Any, jnp.ndarray, jnp.ndarray, PyTree],
]


class GymnaxEnvelope(Environment):
    """
    Wrapper to convert a Gymnax environment to a envelope environment.

    Gymnax only creates backend info on the first `step`. To keep structural
    equivalence between `init` and `step`, construction probes that schema and stores a
    type-preserving zero-like placeholder. ``info.backend.valid`` distinguishes the
    unavailable reset metadata from real step metadata.

    Gymnax implements `Tuple` and `Dict` spaces, which are converted to `PyTreeSpace`
    of a `tuple` and `dict` PyTree respectively.

    Args:
        gymnax_env (GymnaxEnv): the Gymnax
            environment.
        env_params (GymnaxEnvParams): the environment
            parameters, which are passed to the Gymnax environment's `reset` and `step`
            methods.
    """

    gymnax_env: GymnaxEnv = static_field(unsafe=True)
    env_params: PyTree = field()
    _max_steps: int | None = static_field()
    _empty_backend_info: Container = field()

    @classmethod
    def from_name(
        cls,
        env_name: str,
        env_params: GymnaxEnvParams | None = None,
        env_kwargs: dict[str, Any] | None = None,
    ) -> "GymnaxEnvelope":
        """Create a `GymnaxEnvelope` from a name and keyword arguments.
        `env_kwargs` are passed to `gymnax.make`.
        """
        env_kwargs = env_kwargs or {}
        gymnax_env, default_params = gymnax_create(env_name, **env_kwargs)
        if env_params is None:
            default_max_steps = _capture_horizon(default_params.max_steps_in_episode)
            env_params = default_params.replace(max_steps_in_episode=jnp.inf)
        else:
            default_max_steps = _capture_horizon(env_params.max_steps_in_episode)
            if default_max_steps is not None:
                _warn_preserved_horizon("Gymnax", "env_params.max_steps_in_episode")

        empty_backend_info = _probe_gymnaxlike_info_placeholder(
            gymnax_env, env_params, step_fn=gymnax_env.step_env
        )
        return cls(
            gymnax_env=gymnax_env,
            env_params=env_params,
            _max_steps=default_max_steps,
            _empty_backend_info=empty_backend_info,
        )

    @property
    def default_max_steps(self) -> int | None:
        return self._max_steps

    @override
    def init(self, key: Key) -> tuple[State, Info]:
        reset_fn = cast(_GymnaxReset, self.gymnax_env.reset)

        key, subkey = jax.random.split(key)
        obs, env_state = reset_fn(subkey, self.env_params)
        state = Container().update(key=key, env_state=env_state)
        info = InfoContainer(obs=obs, reward=0.0, terminated=False)
        info = info.update(backend=self._empty_backend_info)
        return state, info

    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        key, subkey = jax.random.split(state.key)
        # Gymnax's public ``step`` auto-resets on completion. Call the raw
        # transition so Envelope remains the sole owner of reset semantics.
        step_fn = cast(_GymnaxStep, self.gymnax_env.step_env)
        obs, env_state, reward, done, env_info = step_fn(
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
        return _convert_space(self.gymnax_env.action_space(self.env_params))

    @cached_property
    @override
    def observation_space(self) -> envelope_spaces.Space:
        return _convert_space(self.gymnax_env.observation_space(self.env_params))


def _convert_space(gmx_space: gymnax_spaces.Space) -> envelope_spaces.Space:
    if isinstance(gmx_space, (gymnax_spaces.Box, gymnax_spaces.Discrete)):
        return _convert_gymnaxlike_space(
            gmx_space,
            box_type=gymnax_spaces.Box,
            discrete_type=gymnax_spaces.Discrete,
        )
    if isinstance(gmx_space, gymnax_spaces.Tuple):
        spaces = tuple(_convert_space(space) for space in gmx_space.spaces)
        return envelope_spaces.PyTreeSpace(spaces)
    if isinstance(gmx_space, gymnax_spaces.Dict):
        spaces = {k: _convert_space(space) for k, space in gmx_space.spaces.items()}
        return envelope_spaces.PyTreeSpace(spaces)
    raise ValueError(f"Unsupported space type: {type(gmx_space)}")
