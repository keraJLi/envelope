# Migrating from Envelope 0.4 to 0.5

Envelope 0.5 is a correctness-focused beta break. It does not retain shims for behavior
that contradicted the documented lifecycle.

## Lifecycle and episode boundaries

- Every environment and wrapper now has exactly `init(key)`, `reset(state, key)`, and
  `step(state, action)`; positional and keyword invocation use the same names.
- Auto-reset returns the new reset state and observation. The top-level reward and done
  flags still belong to the completing transition.
- Read the complete terminal `Info` from `info.final` and check `info.final_valid` before
  treating it as a completed episode. Non-terminal steps preserve the latest valid
  `info.final`; before the first completion it is a zero-like placeholder.
- `EpisodeStatisticsWrapper` now resets every episode. Use
  `CumulativeStatisticsWrapper` and read `info.cumulative_stats` for totals that survive
  manual and automatic resets.
- `TruncationWrapper` requires a positive limit and preserves inner truncation flags.
- State injection now requires `reset_info=...`, not only a reset observation.

## Wrapper composition

Vectorize outside elementwise auto-reset. Place episode statistics and truncation inside
auto-reset, and cumulative statistics outside it. Pooled initialization accepts only
explicitly pooling-capable stacks and is incompatible with state injection or persistent
inner normalization. Base environments now expose `supports_init_pooling=False`; only
proven adapters and reset-equivalent wrappers opt in or propagate the capability.

## Core data structures

- Space construction rejects invalid concrete bounds, cardinalities, and batch sizes.
- Space membership requires exact structure and shape. Discrete values must be integer;
  continuous values must be real numeric; booleans are never accepted as numbers.
- Sampling an unbounded continuous space now uses finite distributions and therefore
  changes its random stream.
- `Container` extras are ordered lexicographically. Extra names must be fixed across JAX
  branches even though their values may change.
- Mutable or array-valued static fields are rejected. Convert repository-owned metadata
  to immutable values or mark audited third-party objects with `static_field(unsafe=True)`.

## Typing

The wheel now includes `py.typed`. `Info` rewards and done flags are typed as scalar or
array values, fluent container updates return `Self`, and Pyright checks every library
source in standard mode. Downstream projects can therefore type-check against the inline
annotations without installing stubs.

## Adapters and packaging

`create` reserves `max_episode_steps`: use `"default"`, a positive integer, or `None`.
Backend metadata moved to `info.backend`. Published suites use per-adapter extras;
Gymnax and Kinetix retain documented pinned source installs until their required fixes
are released upstream.

Each adapter now keeps one fixed `info.backend` schema across `init`, `reset`, and
`step`. When a backend cannot emit a field during reset, Envelope supplies a zero-like
placeholder and sets `info.backend.valid=False`; consume backend values only when that
flag is true.

Brax no longer accepts backend batching, auto-reset, episode-length overrides, wrapped
backend environments, or non-default `action_repeat`; compose the corresponding Envelope
wrappers instead. Kinetix consistently rejects `auto_reset=True`, including direct random
and premade-level constructors.
