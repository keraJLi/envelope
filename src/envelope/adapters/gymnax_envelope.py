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
    backend_container,
    placeholder_like,
    replace_backend_params,
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


def _probe_backend_placeholder(gymnax_env: GymnaxEnv, env_params: PyTree) -> Container:
    """Probe Gymnax's step-only info schema outside transformed execution."""
    reset_fn = cast(_GymnaxReset, gymnax_env.reset)
    step_fn = cast(_GymnaxStep, gymnax_env.step_env)
    key = jax.random.key(0)
    _, state = reset_fn(key, env_params)
    _, _, _, _, raw_backend = step_fn(
        key,
        state,
        gymnax_env.action_space(env_params).sample(key),
        env_params,
    )
    placeholder = cast(Container, placeholder_like(backend_container(raw_backend)))
    return placeholder.update(valid=jnp.asarray(False))


class GymnaxEnvelope(Environment):
    """
    Wrapper to convert a Gymnax environment to a envelope environment.

    Gymnax only creates backend info on the first `step`. To keep structural
    equivalence between `init` and `step`, construction probes that schema and stores a
    type-preserving zero-like placeholder. ``info.backend.valid`` distinguishes the
    unavailable reset metadata from real step metadata.

    Gymnax implements `Tuple` and `Dict` spaces, which are converted to `PyTreeSpace`
    of a `tuple` and `dict` PyTree respectively.

    Attributes:
        gymnax_env (GymnaxEnv): the Gymnax
            environment.
        env_params (GymnaxEnvParams): the environment
            parameters, which are passed to the Gymnax environment's `reset` and `step`
            methods.
    """

    gymnax_env: GymnaxEnv = static_field(unsafe=True)
    env_params: PyTree = field()
    _default_max_steps: int = static_field()
    _backend_placeholder: Container = field()

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
        env_kwargs = {} if env_kwargs is None else dict(env_kwargs)
        if "max_steps_in_episode" in env_kwargs:
            raise ValueError(
                "Cannot override 'max_steps_in_episode' directly. "
                "Use TruncationWrapper for episode length control."
            )
        gymnax_env, default_params = gymnax_create(env_name, **env_kwargs)
        selected_params = default_params if env_params is None else env_params
        default_max_steps = int(selected_params.max_steps_in_episode)
        selected_params = replace_backend_params(
            selected_params, max_steps_in_episode=jnp.inf
        )
        backend_placeholder = _probe_backend_placeholder(gymnax_env, selected_params)
        return cls(
            gymnax_env=gymnax_env,
            env_params=selected_params,
            _default_max_steps=default_max_steps,
            _backend_placeholder=backend_placeholder,
        )

    @property
    def default_max_steps(self) -> int:
        return self._default_max_steps

    @property
    def supports_init_pooling(self) -> bool:
        return True

    @override
    def init(self, key: Key) -> tuple[State, Info]:
        reset_fn = cast(_GymnaxReset, self.gymnax_env.reset)

        key, subkey = jax.random.split(key)
        obs, env_state = reset_fn(subkey, self.env_params)
        state = Container().update(key=key, env_state=env_state)
        info = InfoContainer(obs=obs, reward=0.0, terminated=False).update(
            backend=self._backend_placeholder
        )
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
        info = InfoContainer(obs=obs, reward=reward, terminated=done).update(
            backend=backend_container(env_info).update(valid=jnp.asarray(True))
        )
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
