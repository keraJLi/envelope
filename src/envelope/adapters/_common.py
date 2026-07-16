"""Shared adapter-internal helpers."""

from __future__ import annotations

import dataclasses
import math
import warnings
from collections.abc import Iterable, Mapping
from typing import Any

import jax
import jax.numpy as jnp

from envelope import spaces
from envelope.struct import Container, zeros_like


def warn_if_wrapper_overlap(
    adapter_name: str,
    supplied_args: Mapping[str, Any] | Iterable[str] | None,
    wrapper_args: Iterable[str],
    **optional_args: Any,
) -> None:
    """Warn if explicitly supplied backend arguments overlap wrapper controls.

    Named optional arguments count as supplied when their value is not ``None``.
    """
    supplied_args = set(supplied_args or ())
    supplied_args.update(
        name for name, value in optional_args.items() if value is not None
    )
    explicit_settings = sorted(supplied_args.intersection(wrapper_args))
    if not explicit_settings:
        return

    warnings.warn(
        f"Explicit {adapter_name} backend settings may overlap with Envelope "
        f"wrappers: {', '.join(explicit_settings)}.",
        UserWarning,
        stacklevel=3,
    )


def backend_container(metadata: Any) -> Container:
    """Normalize suite metadata to a dot-accessible, JAX-compatible namespace."""
    if isinstance(metadata, Container):
        return metadata
    if dataclasses.is_dataclass(metadata) and not isinstance(metadata, type):
        values = {
            item.name: getattr(metadata, item.name)
            for item in dataclasses.fields(metadata)
        }
    elif isinstance(metadata, Mapping):
        values = dict(metadata)
    else:
        values = {"value": metadata}

    if not all(isinstance(key, str) for key in values):
        return Container().update(value=metadata)
    return Container().update(**values)


def _capture_horizon(value: Any) -> int | None:
    """Convert a finite backend horizon to an integer."""
    if value is None or not math.isfinite(value):
        return None
    return int(value)


def _warn_preserved_horizon(adapter_name: str, parameter_name: str) -> None:
    warnings.warn(
        f"Explicit {adapter_name} {parameter_name} is preserved; backend timeouts "
        "are reported as termination.",
        UserWarning,
        stacklevel=3,
    )


def _convert_gymnaxlike_space(
    backend_space: Any,
    *,
    box_type: type[Any],
    discrete_type: type[Any],
) -> spaces.Space:
    """Convert a Gymnax-like Box or Discrete leaf space."""
    bs = backend_space  # alias for legibility
    if isinstance(backend_space, box_type):
        low = jnp.broadcast_to(bs.low, bs.shape).astype(bs.dtype)
        high = jnp.broadcast_to(bs.high, bs.shape).astype(bs.dtype)
        return spaces.Continuous(low=low, high=high)
    if isinstance(bs, discrete_type):
        n = jnp.broadcast_to(bs.n, bs.shape).astype(bs.dtype)
        return spaces.Discrete(n=n)
    raise ValueError(f"Unsupported space type: {type(bs)}")


def _probe_gymnaxlike_info_placeholder(
    env: Any,
    env_params: Any,
    step_fn: Any | None = None,
) -> Container:
    """Probe a Gymnax-like environment's step-only info schema."""
    key = jax.random.key(0)
    _, state = env.reset(key, env_params)
    action = env.action_space(env_params).sample(key)
    step_fn = env.step if step_fn is None else step_fn
    _, _, _, _, raw_backend = step_fn(key, state, action, env_params)
    return zeros_like(backend_container(raw_backend)).update(valid=jnp.asarray(False))
