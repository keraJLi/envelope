import inspect
import warnings
from collections.abc import Mapping
from copy import copy
from functools import cached_property
from typing import Any, override

from brax.envs import Env as BraxEnv
from brax.envs import Wrapper as BraxWrapper
from brax.envs import create as brax_create
from jax import numpy as jnp

from envelope import spaces
from envelope.adapters._common import backend_container
from envelope.environment import Environment, Info, InfoContainer, State
from envelope.struct import static_field
from envelope.typing import Key, PyTree

_BRAX_DEFAULT_EPISODE_LENGTH = (
    inspect.signature(brax_create).parameters["episode_length"].default
)
if (
    isinstance(_BRAX_DEFAULT_EPISODE_LENGTH, bool)
    or not isinstance(_BRAX_DEFAULT_EPISODE_LENGTH, int)
    or _BRAX_DEFAULT_EPISODE_LENGTH <= 0
):
    raise RuntimeError("Brax create() must expose a finite positive default horizon")


class BraxEnvelope(Environment):
    """
    Wrapper to convert a Brax environment to a envelope environment.

    Note that Brax' `create` function has a default value of `1000` for the episode
    length. Thus, the `default_max_steps` property is set to `1000` for all Brax envs.

    Brax uses a dataclass as its state. Its fields are preserved under
    ``info.backend``.

    Attributes:
        brax_env (BraxEnv): the Brax environment.
    """

    brax_env: BraxEnv = static_field(unsafe=True)

    @classmethod
    def from_name(
        cls, env_name: str, env_kwargs: dict[str, Any] | None = None
    ) -> "BraxEnvelope":
        """
        Create a `BraxEnvelope` from a name and keyword arguments.
        `env_kwargs` arepassed to `brax.envs.create`.
        """
        env_kwargs = {} if env_kwargs is None else dict(env_kwargs)
        if "episode_length" in env_kwargs:
            raise ValueError(
                "Cannot override 'episode_length' directly. "
                "Use TruncationWrapper for episode length control."
            )
        if "auto_reset" in env_kwargs:
            raise ValueError(
                "Cannot override 'auto_reset' directly. "
                "Use AutoResetWrapper for auto-reset behavior."
            )
        if "batch_size" in env_kwargs:
            raise ValueError(
                "Cannot set backend 'batch_size'. Use VmapWrapper for vectorization."
            )
        if "action_repeat" in env_kwargs:
            action_repeat = env_kwargs["action_repeat"]
            if (
                isinstance(action_repeat, bool)
                or not isinstance(action_repeat, int)
                or action_repeat != 1
            ):
                raise ValueError(
                    "Cannot set a non-default backend 'action_repeat'; repeat actions "
                    "outside the adapter."
                )

        # ``None`` asks Brax for the raw environment instead of installing its episode
        # wrapper. Envelope owns truncation and auto-reset semantics.
        env_kwargs["episode_length"] = None
        env_kwargs["auto_reset"] = False
        env = brax_create(env_name, **env_kwargs)
        return cls(brax_env=env)

    @property
    def default_max_steps(self) -> int:
        return _BRAX_DEFAULT_EPISODE_LENGTH

    @property
    def init_can_replace_reset(self) -> bool:
        return True

    def __post_init__(self):
        if isinstance(self.brax_env, BraxWrapper):
            raise ValueError(
                "Pre-wrapped Brax environments are unsupported because their "
                "horizon and auto-reset semantics cannot be proven. Pass a raw "
                "Brax environment and apply Envelope wrappers instead."
            )

    @override
    def init(self, key: Key) -> tuple[State, Info]:
        brax_state = self.brax_env.reset(key)
        info = InfoContainer(
            obs=brax_state.obs,
            reward=brax_state.reward,
            terminated=jnp.asarray(brax_state.done, dtype=bool),
        ).update(backend=backend_container(brax_state))
        return brax_state, info

    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        brax_state = self.brax_env.step(state, action)
        info = InfoContainer(
            obs=brax_state.obs,
            reward=brax_state.reward,
            terminated=jnp.asarray(brax_state.done, dtype=bool),
        ).update(backend=backend_container(brax_state))
        return brax_state, info

    @cached_property
    @override
    def action_space(self) -> spaces.Space:
        # All brax environments have action limit of -1 to 1
        return spaces.Continuous.from_shape(
            low=-1.0, high=1.0, shape=(self.brax_env.action_size,)
        )

    @cached_property
    @override
    def observation_space(self) -> spaces.Space:
        # All brax environments have observation limit of -inf to inf
        observation_size = self.brax_env.observation_size
        if isinstance(observation_size, Mapping):
            tree = {
                name: spaces.Continuous.from_shape(
                    low=-jnp.inf,
                    high=jnp.inf,
                    shape=(size,) if isinstance(size, int) else size,
                )
                for name, size in observation_size.items()
            }
            return spaces.PyTreeSpace(tree)
        return spaces.Continuous.from_shape(
            low=-jnp.inf, high=jnp.inf, shape=(observation_size,)
        )

    def __deepcopy__(self, memo):
        warnings.warn(
            f"Trying to deepcopy {type(self).__name__}, which contains a brax env. "
            "Brax envs throw an error when deepcopying, so a shallow copy is returned.",
            category=RuntimeWarning,
            stacklevel=2,
        )
        return copy(self)
