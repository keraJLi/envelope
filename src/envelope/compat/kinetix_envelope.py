"""Kinetix compatibility wrapper.

This module exposes Kinetix environments through the `envelope.environment.Environment`
API. It mirrors envelope's compat philosophy:
- prefer *no* environment-side auto-reset (use `AutoResetWrapper` in envelope)
- prefer *no* fixed episode time-limits (use `TruncationWrapper` in envelope)

`from_name` supports premade level ids like `s/h4_thrust_aim` (optionally with
`.json`). For maximum flexibility, users can bypass level handling entirely by
passing a custom `reset_fn`.
"""

from __future__ import annotations

import warnings
from functools import cached_property
from typing import Any, Callable, Literal, override

import jax
import jax.numpy as jnp
from kinetix.environment import (
    ActionType,
    EnvParams,
    ObservationType,
    StaticEnvParams,
    make_kinetix_env,
)
from kinetix.environment.ued.ued import make_reset_fn_sample_kinetix_level
from kinetix.util.saving import load_from_json_file

from envelope import spaces as envelope_spaces
from envelope.compat.gymnax_envelope import _convert_space as _convert_gymnax_space
from envelope.environment import Environment, Info, InfoContainer, State
from envelope.struct import Container, static_field
from envelope.typing import Key, PyTree

LevelResetFn = Callable[[Key], Any]


def _normalize_level_id(level_id: str) -> str:
    """Normalize a path-like level id.

    Examples:
        - ``"s/h4_thrust_aim"`` -> ``"s/h4_thrust_aim.json"``
        - ``"/s/h4_thrust_aim.json"`` -> ``"s/h4_thrust_aim.json"``
    """
    level_id = level_id.strip().lstrip("/")
    if not level_id:
        raise ValueError("level_id must be a non-empty string")
    if level_id.endswith("/"):
        raise ValueError("level_id must not end with '/'")
    if not level_id.endswith(".json"):
        level_id = f"{level_id}.json"
    return level_id


def _warn_auto_reset(auto_reset: bool) -> None:
    if auto_reset:
        warnings.warn(
            "Creating a KinetixEnvelope with auto_reset=True is not recommended, use "
            "an AutoResetWrapper instead.",
            stacklevel=2,
        )


class KinetixEnvelope(Environment):
    """Wrapper to convert a Kinetix environment to a envelope environment."""

    kinetix_env: Any = static_field()
    env_params: Any

    @property
    def default_max_steps(self) -> int:
        return int(EnvParams().max_timesteps)

    @classmethod
    def from_name(
        cls,
        env_name: str | Literal["random"],
        env_params: EnvParams | None = None,
        env_kwargs: dict[str, Any] | None = None,
    ) -> "KinetixEnvelope":
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
        if env_name == "random":
            return cls.create_random(env_params=env_params, **env_kwargs)

        if (
            env_params is not None
            or "env_params" in env_kwargs
            or "static_env_params" in env_kwargs
        ):
            raise ValueError(
                "env_params and static_env_params cannot be passed when creating a "
                "KinetixEnvelope from a premade level."
            )
        return cls.create_premade(env_name, **env_kwargs)

    @classmethod
    def create_premade(
        cls,
        env_name: str,
        action_type: ActionType = ActionType.CONTINUOUS,
        observation_type: ObservationType = ObservationType.SYMBOLIC_FLAT,
        auto_reset: bool = False,
    ) -> "KinetixEnvelope":
        _warn_auto_reset(auto_reset)

        # Load level.
        level_id_json = _normalize_level_id(env_name)
        level, static_env_params, env_params = load_from_json_file(level_id_json)
        env_params = env_params.replace(max_timesteps=jnp.inf) if env_params else None

        def reset_fn(_: Key) -> Any:
            return level

        # Create environment.
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
        env_params: EnvParams | None = None,
        static_env_params: StaticEnvParams = StaticEnvParams(),
        auto_reset: bool = False,
    ) -> "KinetixEnvelope":
        _warn_auto_reset(auto_reset)
        if env_params is None:
            env_params = EnvParams()
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

    @override
    def reset(self, key: Key) -> tuple[State, Info]:
        key, subkey = jax.random.split(key)
        obs, env_state = self.kinetix_env.reset(subkey, self.env_params)
        state_out = Container().update(key=key, env_state=env_state)
        info = InfoContainer(obs=obs, reward=0.0, terminated=False)
        info = info.update(info=None)
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
