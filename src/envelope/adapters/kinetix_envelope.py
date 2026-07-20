"""Kinetix compatibility wrapper.

This module exposes Kinetix environments through the `envelope.environment.Environment`
API. It mirrors envelope's adapters philosophy:
- prefer *no* environment-side auto-reset (use `AutoResetWrapper` in envelope)
- prefer *no* fixed episode time-limits (use `TruncationWrapper` in envelope)

`from_name` supports size categories (`"s"`, `"m"`, and `"l"`), procedural
`"random"` levels, and premade level ids like `s/h4_thrust_aim`.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Literal, cast, override

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
from envelope.adapters._common import (
    _capture_horizon,
    _probe_gymnaxlike_info_placeholder,
    _warn_preserved_horizon,
    backend_container,
    warn_if_wrapper_overlap,
)
from envelope.adapters.gymnax_envelope import _convert_space as _convert_gymnax_space
from envelope.environment import Environment, Info, InfoContainer, State
from envelope.struct import Container, static_field
from envelope.typing import Key, PyTree

LevelResetFn = Callable[[Key], Any]


def _warn_if_wrapper_controls(auto_reset: bool) -> None:
    warn_if_wrapper_overlap(
        "Kinetix",
        (),
        ("auto_reset",),
        auto_reset=auto_reset or None,
    )


def _without_time_limit(env_params: KinetixEnvEnvParams) -> KinetixEnvEnvParams:
    return cast(Any, env_params).replace(max_timesteps=jnp.inf)


def _list_levels_for_size(size: str) -> list[str]:
    """Return packaged level paths for one Kinetix size category."""
    path = Path(BASE_DIR) / size
    return [f"{size}/{file.name}" for file in sorted(path.iterdir())]


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


class KinetixEnvelope(Environment):
    """Wrapper to convert a Kinetix environment to an envelope environment.

    Kinetix environments are constructed via a `reset_fn` that produces a level on each
    reset, rather than a simple environment name. Size-category, random, and explicit
    premade-level creation modes are provided.

    Kinetix only produces `env_info` on the first `step`, not on `reset`. Construction
    probes that schema and stores a type-preserving zero-like placeholder so `init` and
    `step` remain structurally equivalent. ``info.backend.valid`` distinguishes reset
    placeholders from real step metadata.

    Args:
        kinetix_env (KinetixEnv): the Kinetix environment, with baked-in
            `static_env_params`.
        env_params (KinetixEnvEnvParams): the environment parameters, which are passed
            to the Kinetix environment's `reset` and `step` methods.
    """

    kinetix_env: KinetixEnv = static_field(unsafe=True)
    env_params: KinetixEnvEnvParams = field()
    _max_steps: int | None = static_field()
    _empty_backend_info: Container = field()

    @classmethod
    def registered_names(cls) -> tuple[str, ...]:
        levels = tuple(
            path.relative_to(BASE_DIR).with_suffix("").as_posix()
            for path in sorted(Path(BASE_DIR).glob("*/*.json"))
        )
        return ("s", "m", "l", "random", *levels)

    @property
    def default_max_steps(self) -> int | None:
        return self._max_steps

    @classmethod
    def from_name(
        cls,
        env_name: str | Literal["s", "m", "l", "random"],
        env_params: KinetixEnvEnvParams | None = None,
        env_kwargs: dict[str, Any] | None = None,
    ) -> "KinetixEnvelope":
        """
        The `from_name` method dispatches between creation modes:
        - `"s"`, `"m"`, or `"l"`: sample packaged levels of that size.
        - `"random"`: create a random levels on each reset using Kinetix's
          `make_reset_fn_sample_kinetix_level` using `self.create_random`.
        - Any other level id: load a specific packaged JSON level using
          `self.create_premade`.
        """
        env_kwargs = env_kwargs or {}
        if env_name in ("s", "m", "l"):
            return cls.create_from_size(env_name, env_params=env_params, **env_kwargs)
        if env_name == "random":
            return cls.create_random(env_params=env_params, **env_kwargs)

        if env_params is not None:
            env_kwargs = {**env_kwargs, "env_params": env_params}
        return cls.create_premade(env_name, **env_kwargs)

    @classmethod
    def create_from_size(
        cls,
        size: Literal["s", "m", "l"],
        action_type: ActionType = ActionType.CONTINUOUS,
        observation_type: ObservationType = ObservationType.SYMBOLIC_FLAT,
        env_params: KinetixEnvEnvParams | None = None,
        auto_reset: bool = False,
    ) -> "KinetixEnvelope":
        """Reset to a random packaged level from a size category."""
        _warn_if_wrapper_controls(auto_reset)
        level_paths = _list_levels_for_size(size)
        _, static_env_params, loaded_env_params = load_from_json_file(level_paths[0])
        default_params = (
            loaded_env_params
            if loaded_env_params is not None
            else KinetixEnvEnvParams()
        )
        if env_params is None:
            default_max_steps = _capture_horizon(default_params.max_timesteps)
            env_params = _without_time_limit(default_params)
        else:
            default_max_steps = _capture_horizon(env_params.max_timesteps)
            if default_max_steps is not None:
                _warn_preserved_horizon("Kinetix", "env_params.max_timesteps")

        reset_fn = make_reset_fn_list_of_levels(level_paths, static_env_params)
        kinetix_env = make_kinetix_env(
            action_type=action_type,
            observation_type=observation_type,
            reset_fn=reset_fn,
            env_params=env_params,
            static_env_params=static_env_params,
            auto_reset=auto_reset,
        )
        empty_backend_info = _probe_gymnaxlike_info_placeholder(kinetix_env, env_params)
        return cls(
            kinetix_env=kinetix_env,
            env_params=env_params,
            _max_steps=default_max_steps,
            _empty_backend_info=empty_backend_info,
        )

    @classmethod
    def create_premade(
        cls,
        env_name: str,
        action_type: ActionType = ActionType.CONTINUOUS,
        observation_type: ObservationType = ObservationType.SYMBOLIC_FLAT,
        env_params: KinetixEnvEnvParams | None = None,
        auto_reset: bool = False,
    ) -> "KinetixEnvelope":
        """
        Load a specific level from a packaged JSON file. The level id has the form
        `"{size}/{name}"` (e.g. `"s/h4_thrust_aim"`); the `.json` suffix is added
        automatically.
        """
        _warn_if_wrapper_controls(auto_reset)

        # Load level
        level_id_json = _normalize_level_id(env_name)
        level, static_env_params, loaded_env_params = load_from_json_file(level_id_json)
        if level is None:
            raise ValueError(
                f"Kinetix premade level {level_id_json!r} did not contain a level state"
            )
        default_params = (
            loaded_env_params
            if loaded_env_params is not None
            else KinetixEnvEnvParams()
        )
        if env_params is None:
            default_max_steps = _capture_horizon(default_params.max_timesteps)
            env_params = _without_time_limit(default_params)
        else:
            default_max_steps = _capture_horizon(env_params.max_timesteps)
            if default_max_steps is not None:
                _warn_preserved_horizon("Kinetix", "env_params.max_timesteps")

        def reset_fn(_: Key) -> Any:
            return level

        # Create environments
        kinetix_env = make_kinetix_env(
            action_type=action_type,
            observation_type=observation_type,
            reset_fn=reset_fn,
            env_params=env_params,
            static_env_params=static_env_params,
            auto_reset=auto_reset,
        )
        empty_backend_info = _probe_gymnaxlike_info_placeholder(kinetix_env, env_params)
        return cls(
            kinetix_env=kinetix_env,
            env_params=env_params,
            _max_steps=default_max_steps,
            _empty_backend_info=empty_backend_info,
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
        _warn_if_wrapper_controls(auto_reset)
        default_params = KinetixEnvEnvParams()
        if env_params is None:
            default_max_steps = _capture_horizon(default_params.max_timesteps)
            env_params = _without_time_limit(default_params)
        else:
            default_max_steps = _capture_horizon(env_params.max_timesteps)
            if default_max_steps is not None:
                _warn_preserved_horizon("Kinetix", "env_params.max_timesteps")

        reset_fn = make_reset_fn_sample_kinetix_level(env_params, static_env_params)
        kinetix_env = make_kinetix_env(
            action_type=action_type,
            observation_type=observation_type,
            reset_fn=reset_fn,
            env_params=env_params,
            static_env_params=static_env_params,
            auto_reset=auto_reset,
        )
        empty_backend_info = _probe_gymnaxlike_info_placeholder(kinetix_env, env_params)
        return cls(
            kinetix_env=kinetix_env,
            env_params=env_params,
            _max_steps=default_max_steps,
            _empty_backend_info=empty_backend_info,
        )

    @override
    def init(self, key: Key) -> tuple[State, Info]:
        key, subkey = jax.random.split(key)
        obs, env_state = self.kinetix_env.reset(subkey, self.env_params)
        state_out = Container().update(key=key, env_state=env_state)
        info = InfoContainer(obs=obs, reward=0.0, terminated=False)
        info = info.update(backend=self._empty_backend_info)
        return state_out, info

    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        key, subkey = jax.random.split(state.key)
        obs, env_state, reward, done, env_info = self.kinetix_env.step(
            subkey, state.env_state, action, self.env_params
        )
        state_out = state.update(key=key, env_state=env_state)
        backend = backend_container(env_info).update(valid=jnp.asarray(True))
        info = InfoContainer(obs=obs, reward=reward, terminated=done)
        info = info.update(backend=backend)
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
