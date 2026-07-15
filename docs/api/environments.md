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

## Initialization pooling

`Environment.supports_init_pooling` is `True` by default. It means a state sampled by
`init` may be used in place of a later reset state, which matches the default
implementation of `reset` and most environments with independently sampled episodes.

An environment whose reset depends on its current state must override the property with
`False`. Transparent and reset-equivalent wrappers propagate the inner capability;
reset-persistent, lifecycle, and vectorization wrappers opt out.
`PooledInitVmapWrapper` rejects an opted-out stack during construction.

## API Reference

::: envelope.environment.Environment

::: envelope.typing.Info

::: envelope.environment.InfoContainer
