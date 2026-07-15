# Wrappers

## Wrapper Synopsis

Wrappers compose around environments via nesting:
`Wrapper2(env=Wrapper1(env=base_env))`. Each wrapper may transform observations,
actions, or spaces, and may add its own fields to the state or info.

Wrappers that need to track data across steps (e.g. a step counter, running statistics)
extend `WrappedState`, which nests the inner environment's state as `inner_state`.
Wrappers that only transform observations or actions (like `ClipActionWrapper` or
`ContinuousObservationWrapper`) pass state through without wrapping. The `unwrapped`
property traverses the full nesting to return the base environment's state.

Wrappers communicate additional data to user code by adding fields to the info via
`info.update(...)`. For example, `EpisodeStatisticsWrapper` adds `stats`,
`AutoResetWrapper` adds `final` (a snapshot of the terminal step's info, enabling value
bootstrapping), and `ObservationNormalizationWrapper` adds `unnormalized_obs`.

`AutoResetWrapper` also adds `final_valid`. Before the first episode completes, `final`
is a zero-like placeholder and `final_valid` is false. After a completion, `final`
contains the complete terminal `Info` and remains available on later steps until another
episode completes. Consequently, `final_valid` says whether `final` is a real historical
record; `terminated` and `truncated` say whether the current transition ended an episode.

## Vectorization

Three wrappers add batch dimensions:

- **`VmapWrapper`** vmaps a single environment with `batch_size` parallel instances.
- **`VmapEnvsWrapper`** vmaps over a batched pytree of environment instances, for
  example created via `jax.vmap(make_env)(params)`. This is useful when different
  instances have different configurations.
- **`PooledInitVmapWrapper`** vectorizes like `VmapWrapper`, but lazily generates an
  initialization pool when a step batch contains a completion and samples replacements
  for completed elements. An explicit reset still calls the wrapped environment's
  vectorized reset. It is an alternative to `AutoResetWrapper` + `VmapWrapper`.

## Wrapper Ordering

The key constraint is that `AutoResetWrapper` calls `reset()` on its inner wrapper chain
when an episode ends. Wrappers that need their `reset()` triggered on episode boundaries
(e.g. `TruncationWrapper` resetting its step counter) must therefore be **inside**
`AutoResetWrapper`. Vectorization wrappers must be **outside**, since autoreset operates
per-element.

The standard stack, from outermost to innermost, is:

```text
VmapWrapper
└─ AutoResetWrapper
   └─ ObservationNormalizationWrapper    # optional
      └─ StateInjectionWrapper            # optional
         └─ EpisodeStatisticsWrapper
            └─ TruncationWrapper
               └─ stateless observation/action transforms
                  └─ base environment
```

The pooled alternative is:

```text
PooledInitVmapWrapper
└─ EpisodeStatisticsWrapper
   └─ TruncationWrapper
      └─ reset-equivalent observation/action transforms
         └─ base environment with init_can_replace_reset=True
```

When used with `AutoResetWrapper`, `ObservationNormalizationWrapper` must be inside it,
and therefore inside `VmapWrapper`, so current and terminal observations use the same
statistics and representation. Without auto-reset, normalization may sit on either side
of `VmapWrapper` for per-environment or shared statistics. It cannot be combined with
`PooledInitVmapWrapper`, because its running statistics must survive resets. State
injection is also incompatible with pooled initialization.

| Role | Supported placement and constraint |
| --- | --- |
| Stateless observation transforms | Inside `AutoResetWrapper` or `PooledInitVmapWrapper`, so current and terminal observations use the same representation. Reset-equivalent transforms may be inside a pooled stack. |
| Action transforms | May sit on either side of vectorization when their shape contract permits it. Reset-equivalent transforms may be inside a pooled stack. |
| `TruncationWrapper` | Inside `AutoResetWrapper` or `PooledInitVmapWrapper`, so its counter resets at each episode boundary. |
| `EpisodeStatisticsWrapper` | Inside `AutoResetWrapper` or `PooledInitVmapWrapper`, so `stats` belongs to one episode. |
| `StateInjectionWrapper` | Inside `AutoResetWrapper`; incompatible with pooled initialization. |
| `AutoResetWrapper` | Inside `VmapWrapper`, because reset is elementwise; an alternative to, not a child of, `PooledInitVmapWrapper`. |
| `ObservationNormalizationWrapper` | Inside `AutoResetWrapper` whenever auto-reset is used; incompatible with `PooledInitVmapWrapper`. Without auto-reset it may sit on either side of `VmapWrapper`. |
| `VmapWrapper` / `VmapEnvsWrapper` | Outside elementwise auto-reset; incompatible inside pooled initialization. |
| `PooledInitVmapWrapper` | Replaces `VmapWrapper` plus `AutoResetWrapper` and requires the complete inner stack to report `init_can_replace_reset=True`. |

Known invalid episode-boundary stacks fail during construction with an ordering error.

## API Reference

::: envelope.wrappers.Wrapper

::: envelope.wrappers.WrappedState

::: envelope.wrappers.AutoResetWrapper

::: envelope.wrappers.ClipActionWrapper

::: envelope.wrappers.ContinuousObservationWrapper

::: envelope.wrappers.EpisodeStatisticsWrapper

::: envelope.wrappers.FlattenActionWrapper

::: envelope.wrappers.FlattenObservationWrapper

::: envelope.wrappers.ObservationNormalizationWrapper

::: envelope.wrappers.PooledInitVmapWrapper

::: envelope.wrappers.StateInjectionWrapper

::: envelope.wrappers.TruncationWrapper

::: envelope.wrappers.VmapWrapper

::: envelope.wrappers.VmapEnvsWrapper
