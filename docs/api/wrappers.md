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
- **`VmapEnvsWrapper`** vmaps over a batched pytree of environment instances, for
  example created via `jax.vmap(make_env)(params)`. This is useful when different
  instances have different configurations.
- **`PooledInitVmapWrapper`** vectorizes like `VmapWrapper`, but lazily generates a
  small pool of initial states, from which it samples the next state of done
  environments. An explicit reset still calls the wrapped environment's vectorized
  reset. It is an alternative to `AutoResetWrapper` + `VmapWrapper` that is
  computationally efficient.

## Stack constraints

Compatibility uses directional, type-based constraints. A wrapper or environment may
declare:

```python
class MyWrapper(Wrapper):
    stack_constraints = (
        not_inside(SomeOuterType),
        not_containing(SomeInnerType),
    )
```

`not_inside(X)` searches the complete outer chain. `not_containing(X)` searches the
complete inner chain. Matching uses `isinstance`, so subclasses and marker mixins work
without a separate role registry. The entire stack is validated whenever a wrapper is
constructed, including constraints declared by a custom base environment.

The built-in hard constraints are:

| Owner | Cannot be inside | Cannot contain |
| --- | --- | --- |
| `AutoResetWrapper` | `PooledInitializationWrapper` | `VectorizingWrapper` |
| `ObservationNormalizationWrapper` | `PooledInitializationWrapper` | — |
| `StateInjectionWrapper` | `PooledInitializationWrapper` | — |
| `PooledInitVmapWrapper` | — | `VectorizingWrapper` |

`VmapWrapper` and `VmapEnvsWrapper` implement `VectorizingWrapper`.
`PooledInitVmapWrapper` implements both `VectorizingWrapper` and
`PooledInitializationWrapper`. All other built-in wrappers have no hard stack
constraints. Configurations that are runnable but surprising are left to the user.

Conceptually, each rule points in the direction it searches:

```text
outer wrappers  ←  not_inside(...)  [owner]  not_containing(...)  →  inner wrappers
```

For example, autoreset must be vectorized from the outside:

```text
VmapWrapper
└─ AutoResetWrapper       valid
   └─ base

AutoResetWrapper
└─ VmapWrapper            rejected: AutoResetWrapper cannot contain VmapWrapper
   └─ base
```

Observation normalization works on either side of ordinary vectorization. Outside a
`VmapWrapper` it maintains shared statistics; inside it, each vectorized instance has
independent statistics. It may wrap pooled initialization but cannot be placed inside
it. It normalizes terminal observations detected structurally through `info.final` and
`info.final_valid`, while retaining raw values as `info.final.unnormalized_obs`.

Only top-level observations update the running statistics. A terminal observation is
normalized with the same current statistics, but is not counted as another sample.

## API Reference

### Core Wrapper Infrastructure

::: envelope.wrappers.Wrapper

::: envelope.wrappers.WrappedState

::: envelope.wrappers.StackConstraint

::: envelope.wrappers.not_inside

::: envelope.wrappers.not_containing

::: envelope.wrappers.VectorizingWrapper

::: envelope.wrappers.PooledInitializationWrapper


### Wrapper Instances

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
