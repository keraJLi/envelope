# Envelope 0.5.0

Envelope 0.5 is a correctness-focused beta break. It does not retain compatibility
shims for behavior that contradicted the lifecycle contract.

## Lifecycle and episode boundaries

- Every environment and wrapper now exposes `init(key)`, `reset(state, key)`, and
  `step(state, action)`, with the same names and ordering for keyword calls.
- Auto-reset returns the reset state and observation. The top-level reward and done
  flags still describe the action that completed the previous episode.
- The complete terminal `Info` is available as `info.final`. Check `info.final_valid`
  before treating it as a completed episode. Later non-terminal steps preserve the
  latest valid final record; before the first completion it is a zero-like placeholder.
- `EpisodeStatisticsWrapper` now resets per episode. Use
  `CumulativeStatisticsWrapper` and `info.cumulative_stats` for totals that survive
  manual and automatic resets.
- `TruncationWrapper` requires `max_steps >= 1`, resets its counter correctly, and
  preserves truncation reported by its inner environment.
- State injection now requires a complete reset `Info` via
  `set_reset_state(state, reset_state, *, reset_info=...)`.

## Wrapper composition

Vectorization belongs outside elementwise auto-reset. Episode statistics and truncation
belong inside auto-reset; cumulative statistics belong outside it. Pooled initialization
only accepts stacks that explicitly support initialization pooling and is incompatible
with state injection or persistent normalization inside the pool.

## Spaces and core structures

- Space construction rejects invalid concrete bounds, cardinalities, and batch sizes.
- `contains` requires exact pytree structure and shape and returns a scalar JAX boolean.
  Discrete candidates must be non-boolean integers; continuous candidates must be
  non-boolean real numbers.
- Continuous sampling supports finite, one-sided, and fully unbounded dimensions.
- `Container` extra keys are ordered lexicographically. The set of keys still has to
  remain fixed across JAX branches.
- Safe static fields reject arrays, mutable containers, and unhashable values. Audited
  opaque third-party objects must use `static_field(unsafe=True)`.

## Adapters and factory

`create` now accepts `max_episode_steps="default"`, a positive integer, or `None`.
Adapters disable native horizons and auto-reset, retain their captured default horizon,
and expose raw suite metadata through a stable `info.backend` container. Backend fields
that are unavailable on reset use zero-like placeholders with a validity flag.

Brax rejects backend batching and unsupported episode wrapping or `action_repeat`
configuration in favor of Envelope wrappers. Kinetix consistently rejects
`auto_reset=True`.

Published adapter suites are available as package extras. Gymnax and Kinetix remain
pinned source-backed development dependencies until their adapter regression suites pass
against indexed releases.

## Typing and packaging

The wheel includes `py.typed`. Rewards and done flags are typed as scalar-or-array,
fluent container updates return `Self`, and the library is checked with Pyright in
standard mode. Experimental PPO code lives under `examples/ppo` and is excluded from
release artifacts.
