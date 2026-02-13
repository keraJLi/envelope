"""Kinetix compatibility wrapper.

This module exposes Kinetix environments through the `envelope.environment.Environment`
API. It mirrors envelope's adapters philosophy:
- prefer *no* environment-side auto-reset (use `AutoResetWrapper` in envelope)
- prefer *no* fixed episode time-limits (use `TruncationWrapper` in envelope)

`from_name` supports size categories (`"s"`, `"m"`, `"l"`) that reset to a random
level from that size on each reset, or `"random"` for fully procedural UED levels.
"""

from __future__ import annotations

import warnings
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, override

import jax
import jax.numpy as jnp
from kinetix.environment import ActionType, ObservationType, make_kinetix_env
from kinetix.environment.env import EnvParams as KinetixEnvEnvParams
from kinetix.environment.env import KinetixEnv
from kinetix.environment.env import StaticEnvParams as KinetixStaticEnvParams
from kinetix.environment.ued.ued import (
    make_reset_fn_list_of_levels,
    make_reset_fn_sample_kinetix_level,
)
from kinetix.util.saving import BASE_DIR, load_from_json_file

from envelope import field
from envelope import spaces as envelope_spaces
from envelope.adapters.gymnax_envelope import _convert_space as _convert_gymnax_space
from envelope.environment import Environment, Info, InfoContainer, State
from envelope.struct import Container, static_field
from envelope.typing import Key, PyTree


def _warn_auto_reset(auto_reset: bool) -> None:
    if auto_reset:
        warnings.warn(
            "Creating a KinetixEnvelope with auto_reset=True is not recommended, use "
            "an AutoResetWrapper instead.",
            stacklevel=2,
        )


def _list_levels_for_size(size: str) -> list[str]:
    """Return level path strings (e.g. ``"s/h4_thrust_aim.json"``) for a size category."""
    path = Path(BASE_DIR) / size
    return [f"{size}/{file.name}" for file in sorted(path.iterdir())]


class KinetixEnvelope(Environment):
    """Wrapper to convert a Kinetix environment to an envelope environment.

    Kinetix environments are constructed via a `reset_fn` that produces a level on each
    reset, rather than a simple environment name. Two creation modes are provided:
    `create_from_size` and `create_random`.

    Kinetix only produces the `env_info` dict on the first `step`, not on `reset`. To
    keep structural equivalence between the `init` and `step` infos (required for
    `jax.lax.scan`, `jax.vmap`, etc.), a NaN-filled placeholder with the same pytree
    structure is returned on `init`.

    Args:
        kinetix_env (KinetixEnv): the Kinetix environment, with baked-in
            `static_env_params`.
        env_params (KinetixEnvEnvParams): the environment parameters, which are passed
            to the Kinetix environment's `reset` and `step` methods.
    """

    kinetix_env: KinetixEnv = static_field()
    env_params: KinetixEnvEnvParams = field()

    @property
    def default_max_steps(self) -> int:
        return int(KinetixEnvEnvParams().max_timesteps)

    @classmethod
    def from_name(
        cls,
        env_name: Literal["s", "m", "l", "random"],
        env_kwargs: dict[str, Any] | None = None,
    ) -> "KinetixEnvelope":
        """Dispatch to the appropriate creation mode.

        - ``"s"``, ``"m"``, ``"l"``: reset to a random level from that size category
          via `create_from_size`.
        - ``"random"``: fully procedural UED levels via `create_random`.
        """
        env_kwargs = env_kwargs or {}
        if env_name in ("s", "m", "l"):
            return cls.create_from_size(env_name, **env_kwargs)
        if env_name == "random":
            return cls.create_random(**env_kwargs)
        raise ValueError(
            f"Invalid env_name {env_name!r}. Expected one of 's', 'm', 'l', 'random'."
        )

    @classmethod
    def create_from_size(
        cls,
        size: Literal["s", "m", "l"],
        action_type: ActionType = ActionType.CONTINUOUS,
        observation_type: ObservationType = ObservationType.SYMBOLIC_FLAT,
        auto_reset: bool = False,
    ) -> "KinetixEnvelope":
        """Reset to a random level from a size category on each reset.

        Loads all packaged levels for the given *size* (``"s"``, ``"m"``, ``"l"``)
        and builds a reset function that uniformly samples one on each call.
        """
        _warn_auto_reset(auto_reset)

        level_paths = _list_levels_for_size(size)
        # Load one level to obtain static_env_params and env_params.
        _, static_env_params, env_params = load_from_json_file(level_paths[0])
        env_params = env_params.replace(max_timesteps=jnp.inf)

        reset_fn = make_reset_fn_list_of_levels(level_paths, static_env_params)

        kinetix_env = make_kinetix_env(
            action_type=action_type,
            observation_type=observation_type,
            reset_fn=reset_fn,
            env_params=env_params,
            static_env_params=static_env_params,
            auto_reset=auto_reset,
        )
        return cls(kinetix_env=kinetix_env, env_params=env_params)

    @classmethod
    def create_random(
        cls,
        action_type: ActionType = ActionType.CONTINUOUS,
        observation_type: ObservationType = ObservationType.SYMBOLIC_FLAT,
        env_params: KinetixEnvEnvParams | None = None,
        static_env_params: KinetixStaticEnvParams = KinetixStaticEnvParams(),
        auto_reset: bool = False,
    ) -> "KinetixEnvelope":
        """
        Create a random level on each reset using Kinetix's
        `kinetix.environment.ued.ued.make_reset_fn_sample_kinetix_level`.
        """
        _warn_auto_reset(auto_reset)
        if env_params is None:
            env_params = KinetixEnvEnvParams()
        env_params = env_params.replace(max_timesteps=jnp.inf)

        reset_fn = make_reset_fn_sample_kinetix_level(env_params, static_env_params)
        kinetix_env = make_kinetix_env(
            action_type=action_type,
            observation_type=observation_type,
            reset_fn=reset_fn,
            env_params=env_params,
            static_env_params=static_env_params,
            auto_reset=auto_reset,
        )
        return cls(kinetix_env=kinetix_env, env_params=env_params)

    @cached_property
    def _kinetix_info_placeholder(self) -> PyTree:
        # Note that the placeholder that is returned only has nan values where it's
        # dtype is a subdtype of float. TODO: Should we use empty_like?
        key = jax.random.key(0)
        obs, env_state = self.kinetix_env.reset(key, self.env_params)
        action = self.action_space.sample(key)
        _, _, _, _, env_info = self.kinetix_env.step(
            key, env_state, action, self.env_params
        )
        return jax.tree.map(lambda x: jnp.full_like(x, jnp.nan), env_info)

    @override
    def init(self, key: Key) -> tuple[State, Info]:
        key, subkey = jax.random.split(key)
        obs, env_state = self.kinetix_env.reset(subkey, self.env_params)
        state_out = Container().update(key=key, env_state=env_state)
        info = InfoContainer(obs=obs, reward=0.0, terminated=False)
        info = info.update(info=self._kinetix_info_placeholder)
        return state_out, info

    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        key, subkey = jax.random.split(state.key)
        obs, env_state, reward, done, env_info = self.kinetix_env.step(
            subkey, state.env_state, action, self.env_params
        )
        state_out = state.update(key=key, env_state=env_state)
        info = InfoContainer(obs=obs, reward=reward, terminated=done)
        info = info.update(info=env_info)
        return state_out, info

    @override
    @cached_property
    def action_space(self) -> envelope_spaces.Space:
        return _convert_gymnax_space(self.kinetix_env.action_space(self.env_params))

    @override
    @cached_property
    def observation_space(self) -> envelope_spaces.Space:
        return _convert_gymnax_space(
            self.kinetix_env.observation_space(self.env_params)
        )
