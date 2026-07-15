"""Kinetix compatibility wrapper.

This module exposes Kinetix environments through the `envelope.environment.Environment`
API. It mirrors envelope's adapters philosophy:
- prefer *no* environment-side auto-reset (use `AutoResetWrapper` in envelope)
- prefer *no* fixed episode time-limits (use `TruncationWrapper` in envelope)

`from_name` supports premade level ids like `s/h4_thrust_aim` (optionally with
`.json`). For maximum flexibility, users can bypass level handling entirely by
passing a custom `reset_fn`.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any, Callable, Literal, cast, override

import jax
import jax.numpy as jnp
from kinetix.environment import ActionType, ObservationType, make_kinetix_env
from kinetix.environment.env import EnvParams as KinetixEnvEnvParams
from kinetix.environment.env import KinetixEnv
from kinetix.environment.env import StaticEnvParams as KinetixStaticEnvParams
from kinetix.environment.ued.ued import make_reset_fn_sample_kinetix_level
from kinetix.util.saving import load_from_json_file

from envelope import field
from envelope import spaces as envelope_spaces
from envelope.adapters._common import (
    backend_container,
    zeros_like,
    replace_backend_params,
)
from envelope.adapters.gymnax_envelope import _convert_space as _convert_gymnax_space
from envelope.environment import Environment, Info, InfoContainer, State
from envelope.struct import Container, static_field
from envelope.typing import Key, PyTree

LevelResetFn = Callable[[Key], Any]


def _probe_backend_placeholder(
    kinetix_env: KinetixEnv, env_params: KinetixEnvEnvParams
) -> Container:
    """Probe Kinetix's step-only info schema outside transformed execution."""
    key = jax.random.key(0)
    _, state = kinetix_env.reset(key, env_params)
    action = kinetix_env.action_space(env_params).sample(key)
    _, _, _, _, raw_backend = kinetix_env.step(key, state, action, env_params)
    placeholder = cast(Container, zeros_like(backend_container(raw_backend)))
    return placeholder.update(valid=jnp.asarray(False))


def _normalize_level_id(level_id: str) -> str:
    """Normalize a path-like level id.

    Examples:
        - `"s/h4_thrust_aim"` -> `"s/h4_thrust_aim.json"`
        - `"/s/h4_thrust_aim.json"` -> `"s/h4_thrust_aim.json"`
    """
    level_id = level_id.strip().lstrip("/")
    if not level_id:
        raise ValueError("level_id must be a non-empty string")
    if level_id.endswith("/"):
        raise ValueError("level_id must not end with '/'")
    if not level_id.endswith(".json"):
        level_id = f"{level_id}.json"
    return level_id


def _reject_auto_reset(auto_reset: bool) -> None:
    if auto_reset:
        raise ValueError(
            "Cannot enable backend 'auto_reset'. Use AutoResetWrapper instead."
        )


class KinetixEnvelope(Environment):
    """Wrapper to convert a Kinetix environment to an envelope environment.

    Kinetix environments are constructed via a `reset_fn` that produces a level on each
    reset, rather than a simple environment name. Two creation modes are provided:
    `create_random` and `create_premade`.

    Kinetix only produces `env_info` on the first `step`, not on `reset`. Construction
    probes that schema and stores a type-preserving zero-like placeholder so `init` and
    `step` remain structurally equivalent. ``info.backend.valid`` distinguishes reset
    placeholders from real step metadata.

    Attributes:
        kinetix_env (KinetixEnv): the Kinetix environment, with baked-in
            `static_env_params`.
        env_params (KinetixEnvEnvParams): the environment parameters, which are passed
            to the Kinetix environment's `reset` and `step` methods.
    """

    kinetix_env: KinetixEnv = static_field(unsafe=True)
    env_params: KinetixEnvEnvParams = field()
    _default_max_steps: int = static_field()
    _backend_placeholder: Container = field()

    @property
    def default_max_steps(self) -> int:
        return self._default_max_steps

    @property
    def init_can_replace_reset(self) -> bool:
        return True

    @classmethod
    def from_name(
        cls,
        env_name: str | Literal["random"],
        env_params: KinetixEnvEnvParams | None = None,
        env_kwargs: dict[str, Any] | None = None,
    ) -> "KinetixEnvelope":
        """
        The `from_name` method dispatches between the two creation modes:
        - `"random"`: create a random levels on each reset using Kinetix's
          `make_reset_fn_sample_kinetix_level` using `self.create_random`.
        - Any other level id: load a specific level from a packaged JSON file using
          `self.create_premade`.
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
        """
        Load a specific level from a packaged JSON file. The level id has the form
        `"{size}/{name}"` (e.g. `"s/h4_thrust_aim"`); the `.json` suffix is added
        automatically.
        """
        _reject_auto_reset(auto_reset)

        # Load level.
        level_id_json = _normalize_level_id(env_name)
        level, static_env_params, env_params = load_from_json_file(level_id_json)
        if level is None:
            raise ValueError(
                f"Kinetix premade level {level_id_json!r} did not contain a level state"
            )
        selected_params = KinetixEnvEnvParams() if env_params is None else env_params
        default_max_steps = int(selected_params.max_timesteps)
        selected_params = replace_backend_params(selected_params, max_timesteps=jnp.inf)

        def reset_fn(_: Key) -> Any:
            return level

        # Create environment.
        kinetix_env = make_kinetix_env(
            action_type=action_type,
            observation_type=observation_type,
            reset_fn=reset_fn,
            env_params=selected_params,
            static_env_params=static_env_params,
            auto_reset=auto_reset,
        )
        backend_placeholder = _probe_backend_placeholder(kinetix_env, selected_params)
        return cls(
            kinetix_env=kinetix_env,
            env_params=selected_params,
            _default_max_steps=default_max_steps,
            _backend_placeholder=backend_placeholder,
        )

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
        _reject_auto_reset(auto_reset)
        selected_params = KinetixEnvEnvParams() if env_params is None else env_params
        default_max_steps = int(selected_params.max_timesteps)
        selected_params = replace_backend_params(selected_params, max_timesteps=jnp.inf)

        reset_fn = make_reset_fn_sample_kinetix_level(
            selected_params, static_env_params
        )
        kinetix_env = make_kinetix_env(
            action_type=action_type,
            observation_type=observation_type,
            reset_fn=reset_fn,
            env_params=selected_params,
            static_env_params=static_env_params,
            auto_reset=auto_reset,
        )
        backend_placeholder = _probe_backend_placeholder(kinetix_env, selected_params)
        return cls(
            kinetix_env=kinetix_env,
            env_params=selected_params,
            _default_max_steps=default_max_steps,
            _backend_placeholder=backend_placeholder,
        )

    @override
    def init(self, key: Key) -> tuple[State, Info]:
        key, subkey = jax.random.split(key)
        obs, env_state = self.kinetix_env.reset(subkey, self.env_params)
        state_out = Container().update(key=key, env_state=env_state)
        info = InfoContainer(obs=obs, reward=0.0, terminated=False).update(
            backend=self._backend_placeholder
        )
        return state_out, info

    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        key, subkey = jax.random.split(state.key)
        obs, env_state, reward, done, env_info = self.kinetix_env.step(
            subkey, state.env_state, action, self.env_params
        )
        state_out = state.update(key=key, env_state=env_state)
        info = InfoContainer(obs=obs, reward=reward, terminated=done).update(
            backend=backend_container(env_info).update(valid=jnp.asarray(True))
        )
        return state_out, info

    @cached_property
    @override
    def action_space(self) -> envelope_spaces.Space:
        return _convert_gymnax_space(self.kinetix_env.action_space(self.env_params))

    @cached_property
    @override
    def observation_space(self) -> envelope_spaces.Space:
        return _convert_gymnax_space(
            self.kinetix_env.observation_space(self.env_params)
        )
