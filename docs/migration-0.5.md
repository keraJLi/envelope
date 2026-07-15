# Migrating from Envelope 0.4 to 0.5

Envelope 0.5 is a correctness-focused beta break. It does not retain shims for behavior
that contradicted the documented lifecycle.

## Lifecycle and episode boundaries

- Call every reset as `reset(state, key)`, including keyword calls.
- Auto-reset returns the new reset state and observation. The top-level reward and done
  flags still belong to the completing transition.
- Read the complete terminal `Info` from `info.final` and check `info.final_valid` before
  treating it as a completed episode.
- `EpisodeStatisticsWrapper` now resets every episode. Use
  `CumulativeStatisticsWrapper` for totals that survive resets.
- `TruncationWrapper` requires a positive limit and preserves inner truncation flags.
- State injection now requires `reset_info=...`, not only a reset observation.

## Wrapper composition

Vectorize outside elementwise auto-reset. Place episode statistics and truncation inside
auto-reset, and cumulative statistics outside it. Pooled initialization accepts only
explicitly pooling-capable stacks and is incompatible with state injection or persistent
inner normalization.

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

## Adapters and packaging

`create` reserves `max_episode_steps`: use `"default"`, a positive integer, or `None`.
Backend metadata moved to `info.backend`. Published suites use per-adapter extras;
Gymnax and Kinetix retain documented pinned source installs until their required fixes
are released upstream.
