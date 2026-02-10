from functools import cached_property
from typing import TYPE_CHECKING, Any, override

import jax
import jax.numpy as jnp
from craftax.craftax_env import make_craftax_env_from_name

from envelope import spaces as envelope_spaces
from envelope.adapters.gymnax_envelope import _convert_space as _convert_gymnax_space
from envelope.environment import Environment, Info, InfoContainer, State
from envelope.struct import Container, static_field
from envelope.typing import Key, PyTree, TypeAlias

if TYPE_CHECKING:
    from craftax.craftax.craftax_state import EnvParams as CraftaxEnvParamsOriginal
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

    CraftaxEnvParams: TypeAlias = CraftaxEnvParamsOriginal | CraftaxClassicEnvParams
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
else:
    CraftaxEnvParams: TypeAlias = Any
    CraftaxEnv: TypeAlias = Any


class CraftaxEnvelope(Environment):
    """
    Wrapper to convert a Craftax environment to a envelope environment.

    Craftax (mostly) uses the Gymnax interface, so the info object is only created on
    the first `step`. To keep structural equivalence between the `init` and `step`
    infos, we create a placeholder filled with `jnp.nan` that is returned on `init`.

    Args:
        craftax_env (CraftaxEnv): the Craftax environment, with baked-in
            `static_env_params`.
        env_params (CraftaxEnvParams): the environment parameters, which are passed to
            the Craftax environment's `reset` and `step` methods.
    """

    craftax_env: CraftaxEnv = static_field()
    env_params: PyTree = static_field()  # TODO: remove static marker as soon as craftax merges https://github.com/MichaelTMatthews/Craftax/pull/48

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
        env_kwargs = env_kwargs or {}
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
        default_params = env.default_params.replace(max_timesteps=jnp.inf)

        env_params = env_params or default_params
        return cls(craftax_env=env, env_params=env_params)

    @property
    def default_max_steps(self) -> int:
        return int(self.craftax_env.default_params.max_timesteps)

    @cached_property
    def _craftax_info_placeholder(self) -> PyTree:
        key = jax.random.key(0)
        _, state = self.craftax_env.reset(key, self.env_params)
        _, _, _, _, info = self.craftax_env.step(
            key,
            state,
            self.craftax_env.action_space(self.env_params).sample(key),
            self.env_params,
        )
        return jax.tree.map(lambda x: jnp.full_like(x, jnp.nan), info)

    @override
    def init(self, key: Key) -> tuple[State, Info]:
        key, subkey = jax.random.split(key)
        obs, env_state = self.craftax_env.reset(subkey, self.env_params)
        state = Container().update(key=key, env_state=env_state)
        info = InfoContainer(obs=obs, reward=0.0, terminated=False)
        info = info.update(info=self._craftax_info_placeholder)
        return state, info

    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        key, subkey = jax.random.split(state.key)
        obs, env_state, reward, done, env_info = self.craftax_env.step(
            subkey, state.env_state, action, self.env_params
        )
        state = state.update(key=key, env_state=env_state)
        info = InfoContainer(obs=obs, reward=reward, terminated=done)
        info = info.update(info=env_info)
        return state, info

    @override
    @cached_property
    def action_space(self) -> envelope_spaces.Space:
        return _convert_gymnax_space(self.craftax_env.action_space(self.env_params))

    @override
    @cached_property
    def observation_space(self) -> envelope_spaces.Space:
        return _convert_gymnax_space(
            self.craftax_env.observation_space(self.env_params)
        )
