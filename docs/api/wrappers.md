# Wrappers

## Wrapper Synopsis

Wrappers compose around environments via nesting: `Wrapper2(env=Wrapper1(env=base_env))`.
Each wrapper may transform observations, actions, or spaces, and may add its own fields
to the state or info.

Wrappers that need to track data across steps (e.g. a step counter, running statistics)
extend `WrappedState`, which nests the inner environment's state as `inner_state`.
Wrappers that only transform observations or actions (like `ClipActionWrapper` or
`ContinuousObservationWrapper`) pass state through without wrapping. The `unwrapped`
property traverses the full nesting to return the base environment's state.

Wrappers communicate additional data to user code by adding fields to the info via
`info.update(...)`. For example, `EpisodeStatisticsWrapper` adds `stats`,
`AutoResetWrapper` adds `final` (the complete terminal step info) and `final_valid`, and
`ObservationNormalizationWrapper` adds `unnormalized_obs`.

On a completing transition, the returned state and `info.obs` are already reset. Reward,
termination, and truncation still describe the action just taken. Other top-level
metadata describes the reset state; terminal metadata remains in `info.final`. Before
the first completion, `final` is a zero-like structural placeholder and
`final_valid=False`.

## Vectorization

Three wrappers add batch dimensions:

- **`VmapWrapper`** vmaps a single environment with `batch_size` parallel instances.
- **`VmapEnvsWrapper`** vmaps over a batched pytree of environment instances, for example
  created via `jax.vmap(make_env)(params)`. This is useful when different instances have
  different configurations.
- **`PooledInitVmapWrapper`** vmaps like `VmapWrapper`, but pre-computes a pool of initial
  states and samples from them on reset. It includes built-in autoreset logic and only
  accepts environments whose `supports_init_pooling` capability is true.

## Wrapper Ordering

The key constraint is that `AutoResetWrapper` calls `reset()` on its inner wrapper chain
when an episode ends. Wrappers that need their `reset()` triggered on episode boundaries
(e.g. `TruncationWrapper` resetting its step counter) must therefore be **inside**
`AutoResetWrapper`. Vectorization wrappers must be **outside**, since autoreset operates
per-element.

From innermost to outermost:
```
base env → Observation/action transforms → Episode logic → AutoReset → Vectorization
```

A concrete example with all layers:
```
VmapWrapper                              # outermost: adds batch dim
└─ CumulativeStatisticsWrapper           # optional lifetime totals
   └─ AutoResetWrapper                   # resets on done, adds `final`
      └─ StateInjectionWrapper           # optional reset target
         └─ EpisodeStatisticsWrapper     # per-episode reward/length
            └─ TruncationWrapper         # caps episode length
               └─ ContinuousObservationWrapper
                  └─ ClipActionWrapper
                     └─ base env          # innermost
```

Not all wrappers are needed in every pipeline. `ObservationNormalizationWrapper` may be
inside vectorization for per-environment statistics or outside it for shared statistics.
Persistent normalization must remain outside `PooledInitVmapWrapper`; state injection is
not compatible with pooled initialization. Invalid episode-boundary stacks raise during
construction with the supported alternative.

## API Reference

::: envelope.wrappers.Wrapper

::: envelope.wrappers.WrappedState

::: envelope.wrappers.AutoResetWrapper

::: envelope.wrappers.ClipActionWrapper

::: envelope.wrappers.ContinuousObservationWrapper

::: envelope.wrappers.EpisodeStatisticsWrapper

::: envelope.wrappers.CumulativeStatisticsWrapper

::: envelope.wrappers.FlattenActionWrapper

::: envelope.wrappers.FlattenObservationWrapper

::: envelope.wrappers.ObservationNormalizationWrapper

::: envelope.wrappers.PooledInitVmapWrapper

::: envelope.wrappers.StateInjectionWrapper

::: envelope.wrappers.TruncationWrapper

::: envelope.wrappers.VmapWrapper

::: envelope.wrappers.VmapEnvsWrapper
