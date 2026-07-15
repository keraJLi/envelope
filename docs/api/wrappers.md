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
`AutoResetWrapper` adds `final` (a snapshot of the terminal step's info, enabling value
bootstrapping), and `ObservationNormalizationWrapper` adds `unnormalized_obs`.

## Vectorization

Three wrappers add batch dimensions:

- **`VmapWrapper`** vmaps a single environment with `batch_size` parallel instances.
- **`VmapEnvsWrapper`** vmaps over a batched pytree of environment instances, for example
  created via `jax.vmap(make_env)(params)`. This is useful when different instances have
  different configurations.
- **`PooledInitVmapWrapper`** vmaps like `VmapWrapper`, but pre-computes a pool of initial
  states and samples from them on reset. It also includes built-in autoreset logic, making
  it an alternative to `AutoResetWrapper` + `VmapWrapper`.

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
└─ AutoResetWrapper                      # resets on done, adds `final`
   └─ StateInjectionWrapper              # (optional) overrides reset target
      └─ EpisodeStatisticsWrapper        # tracks reward/length
         └─ TruncationWrapper            # caps episode length
            └─ ObservationNormalizationWrapper
               └─ ContinuousObservationWrapper
                  └─ ClipActionWrapper
                     └─ base env         # innermost
```

Not all wrappers are needed in every pipeline. The ordering between wrappers in the same
layer (e.g. the observation/action transforms) is flexible.

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
