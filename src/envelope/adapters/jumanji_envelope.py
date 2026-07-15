import warnings
from copy import copy
from functools import cached_property
from typing import Any, override

import jax
import jax.numpy as jnp
import jumanji
from jumanji.env import Environment as JumanjiEnv
from jumanji.specs import Array, BoundedArray, DiscreteArray, MultiDiscreteArray, Spec
from jumanji.types import TimeStep as JumanjiTimeStep

from envelope import spaces as envelope_spaces
from envelope.adapters._common import backend_container
from envelope.environment import Environment, Info, InfoContainer, State
from envelope.struct import static_field
from envelope.typing import Key, PyTree

_MAX_INT = int(jnp.iinfo(jnp.int32).max)


class JumanjiEnvelope(Environment):
    """
    Wrapper to convert a Jumanji environment to a envelope environment.

    Some Jumanji environments support time limits via a `time_limit` attribute of the
    environemnt. If this attribute exists, we overwrite it with the maximum integer
    value and set `default_max_steps` to the original value.

    Attributes:
        jumanji_env (JumanjiEnv): the Jumanji environment.
    """

    jumanji_env: JumanjiEnv = static_field(unsafe=True)
    _default_time_limit: int | None = static_field(default=None)

    @classmethod
    def from_name(
        cls, env_name: str, env_kwargs: dict[str, Any] | None = None
    ) -> "JumanjiEnvelope":
        """
        Create a `JumanjiEnvelope` from a name and keyword arguments.
        `env_kwargs` are passed to `jumanji.make`.
        """
        env_kwargs = {} if env_kwargs is None else dict(env_kwargs)
        if "time_limit" in env_kwargs:
            raise ValueError(
                "Cannot override 'time_limit' directly. "
                "Use TruncationWrapper for episode length control."
            )

        # Create env first with defaults to capture default time_limit
        env = jumanji.make(env_name, **env_kwargs)
        default_time_limit = getattr(env, "time_limit", None)
        if default_time_limit is not None:
            default_time_limit = int(default_time_limit)
            env.time_limit = _MAX_INT

        return cls(jumanji_env=env, _default_time_limit=default_time_limit)

    @property
    def default_max_steps(self) -> int | None:
        return self._default_time_limit

    @property
    def supports_init_pooling(self) -> bool:
        return True

    @override
    def init(self, key: Key) -> tuple[State, Info]:
        env_state, timestep = self.jumanji_env.reset(key)
        info = convert_jumanji_to_envelope_info(timestep)
        return env_state, info

    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        env_state, timestep = self.jumanji_env.step(state, action)
        info = convert_jumanji_to_envelope_info(timestep)
        return env_state, info

    @cached_property
    @override
    def action_space(self) -> envelope_spaces.Space:
        return convert_jumanji_spec_to_envelope_space(self.jumanji_env.action_spec)

    @cached_property
    @override
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
    term = jnp.asarray(timestep.last(), dtype=bool)
    observation = jax.tree.map(_normalize_observation_leaf, timestep.observation)
    info = InfoContainer(
        obs=observation, reward=timestep.reward, terminated=term
    ).update(backend=backend_container(timestep.extras))
    return info


def _normalize_observation_leaf(value: PyTree) -> PyTree:
    """Represent boolean suite observations as integer-valued discrete samples."""
    try:
        array = jnp.asarray(value)
    except (TypeError, ValueError):
        return value
    if jnp.issubdtype(array.dtype, jnp.bool_):
        return array.astype(jnp.int32)
    return value


def convert_jumanji_spec_to_envelope_space(
    spec: Spec | PyTree,
) -> envelope_spaces.Space:
    """Convert a Jumanji Spec to an envelope Space."""
    tree = _spec_to_tree(spec)
    if isinstance(tree, envelope_spaces.Space):
        return tree
    return envelope_spaces.PyTreeSpace(tree=tree)


def _spec_to_tree(spec: Spec | PyTree):
    """Convert a Jumanji Spec to a Space leaf or a raw pytree of Space leaves."""

    if isinstance(spec, (DiscreteArray, MultiDiscreteArray)):
        n = jnp.asarray(spec.num_values, dtype=spec.dtype)
        if getattr(spec, "shape", ()) not in ((), n.shape):
            n = jnp.broadcast_to(n, spec.shape)
        return envelope_spaces.Discrete(n=n)

    if isinstance(spec, BoundedArray):
        dtype = jnp.dtype(spec.dtype)
        if jnp.issubdtype(dtype, jnp.bool_):
            return envelope_spaces.Discrete.from_shape(2, shape=spec.shape)
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
    if isinstance(spec, Spec):
        return spec._constructor(
            **jax.tree.map(
                _spec_to_tree,
                spec._specs,
                is_leaf=lambda x: isinstance(x, Spec),
            )
        )
    if isinstance(spec, (tuple, list)):
        return tuple(_spec_to_tree(s) for s in spec)
    if isinstance(spec, dict):
        return {k: _spec_to_tree(v) for k, v in spec.items()}

    raise ValueError(f"Unsupported spec type: {type(spec)}")
