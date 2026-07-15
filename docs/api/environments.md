# Environment

## Basic interface: `init`, `reset` and `step`

Many RL libraries have `step` and `reset`. Envelope introduces an additional `init`
method, splitting the traditional `reset` into two:

- **`init(key)`** initializes environment state from scratch.
- **`reset(state, key)`** resets the current episode while preserving persistent state.
  For example, `ObservationNormalizationWrapper` keeps its running statistics across
  resets, while only resetting the inner environment.
- **`step(state, action)`** steps the environment.

By default, `reset` simply calls `init`, which is correct for most environments where
the starting state is sampled from a fixed distribution.

Environment methods return `State` and `Info` tuples. `Info` is a structural protocol;
`InfoContainer` is its default implementation, built on `Container`. Wrappers extend the
info with extra fields via `update()` — for example, `EpisodeStatisticsWrapper` adds a
`stats` field, which consumers can then access as `info.stats`.

## Pytree Structure Contract

JAX transformations such as `jax.lax.scan`, `jax.vmap`, and `jax.jit` require that
the pytree structure (treedef) of inputs and outputs remains consistent across
iterations. This imposes a critical requirement on envelope environments:

**`init`, `reset` and `step` should return `State` and `Info` objects with identical
pytree structures and leaf shapes.**

This guarantees that you can map `jnp.where` on the `Info` they produce, or emit the
`Info` as the output of `jax.lax.scan`.

## When `init` can replace `reset`

`Environment.init_can_replace_reset` is `True` by default. It means the result of `init`
can safely be used where a later `reset` result is expected. This matches the base
implementation, where `reset(state, key)` simply calls `init(key)`.

Set this property to `False` when `reset` keeps or changes anything from the current
state. Wrappers pass through the value from their inner environment unless they preserve
their own state across resets. `PooledInitVmapWrapper` requires the complete inner stack
to report `init_can_replace_reset=True`.

## API Reference

::: envelope.environment.Environment

::: envelope.typing.Info

::: envelope.environment.InfoContainer
