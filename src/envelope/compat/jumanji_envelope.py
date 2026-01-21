import warnings
from copy import copy
from functools import cached_property
from typing import Any, override

import jax.numpy as jnp
import jumanji
from jumanji.specs import Array, BoundedArray, DiscreteArray, MultiDiscreteArray
from jumanji.types import TimeStep as JumanjiTimeStep

from envelope import spaces as envelope_spaces
from envelope.environment import Environment, Info, InfoContainer, State
from envelope.struct import static_field
from envelope.typing import Key, PyTree

_MAX_INT = jnp.iinfo(jnp.int32).max


class JumanjiEnvelope(Environment):
    """Wrapper to convert a Jumanji environment to a envelope environment."""

    jumanji_env: Any = static_field()
    _default_time_limit: int | None = static_field(default=None)

    @classmethod
    def from_name(
        cls, env_name: str, env_kwargs: dict[str, Any] | None = None
    ) -> "JumanjiEnvelope":
        env_kwargs = env_kwargs or {}
        if "time_limit" in env_kwargs:
            raise ValueError(
                "Cannot override 'time_limit' directly. "
                "Use TruncationWrapper for episode length control."
            )

        # Create env first with defaults to capture default time_limit
        temp_env = jumanji.make(env_name, **env_kwargs)
        default_time_limit = getattr(temp_env, "time_limit", None)

        # Now create env with time_limit=_MAX_INT (if env supports it)
        if default_time_limit is not None:
            env_kwargs["time_limit"] = _MAX_INT
        env = jumanji.make(env_name, **env_kwargs)
        return cls(jumanji_env=env, _default_time_limit=default_time_limit)

    @property
    def default_max_steps(self) -> int | None:
        return self._default_time_limit

    @override
    def reset(self, key: Key) -> tuple[State, Info]:
        env_state, timestep = self.jumanji_env.reset(key)
        info = convert_jumanji_to_envelope_info(timestep)
        return env_state, info

    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        env_state, timestep = self.jumanji_env.step(state, action)
        info = convert_jumanji_to_envelope_info(timestep)
        return env_state, info

    @override
    @cached_property
    def action_space(self) -> envelope_spaces.Space:
        return convert_jumanji_spec_to_envelope_space(self.jumanji_env.action_spec)

    @override
    @cached_property
    def observation_space(self) -> envelope_spaces.Space:
        return convert_jumanji_spec_to_envelope_space(self.jumanji_env.observation_spec)

    def __deepcopy__(self, memo):
        warnings.warn(
            f"Trying to deepcopy {type(self).__name__}, which contains a jumanji env. "
            "Jumanji envs may throw an error when deepcopying, so a shallow copy is "
            "returned.",
            category=RuntimeWarning,
            stacklevel=2,
        )
        return copy(self)


def convert_jumanji_to_envelope_info(timestep: JumanjiTimeStep) -> InfoContainer:
    info = InfoContainer(
        obs=timestep.observation, reward=timestep.reward, terminated=timestep.last()
    ).update(**timestep.extras)
    return info


def convert_jumanji_spec_to_envelope_space(spec: Any) -> envelope_spaces.Space:
    """Convert a Jumanji Spec to a envelope Space."""

    if isinstance(spec, (DiscreteArray, MultiDiscreteArray)):
        n = jnp.asarray(spec.num_values, dtype=spec.dtype)
        if getattr(spec, "shape", ()) not in ((), n.shape):
            n = jnp.broadcast_to(n, spec.shape)
        return envelope_spaces.Discrete(n=n)

    if isinstance(spec, BoundedArray):
        low = jnp.broadcast_to(jnp.asarray(spec.minimum, dtype=spec.dtype), spec.shape)
        high = jnp.broadcast_to(jnp.asarray(spec.maximum, dtype=spec.dtype), spec.shape)
        return envelope_spaces.Continuous(low=low, high=high)

    if isinstance(spec, Array):
        dtype = jnp.dtype(spec.dtype)
        if not jnp.issubdtype(dtype, jnp.floating):
            raise NotImplementedError(
                "Unbounded jumanji Array specs are only supported for floating dtypes. "
                f"Got dtype={dtype} for spec={spec!r}."
            )
        low = jnp.full(spec.shape, -jnp.inf, dtype=dtype)
        high = jnp.full(spec.shape, jnp.inf, dtype=dtype)
        return envelope_spaces.Continuous(low=low, high=high)

    # Structured specs (most Jumanji envs): access private mapping when available.
    subspecs = getattr(spec, "_specs", None)
    if isinstance(subspecs, dict):
        tree = {
            k: convert_jumanji_spec_to_envelope_space(v) for k, v in subspecs.items()
        }
        return envelope_spaces.PyTreeSpace(tree)

    if isinstance(spec, (tuple, list)):
        tree = tuple(convert_jumanji_spec_to_envelope_space(s) for s in spec)
        return envelope_spaces.PyTreeSpace(tree)

    raise ValueError(f"Unsupported spec type: {type(spec)}")
