# Adapters

## Creating adapters

Envelope has a powerful `create` method that can create environments from many different
JAX-based environment suites, and automatically casts them into the envelope
`Environment` interface. The `create` function uses a unique environment identifier of
the format `SUITE::ENVIRONMENT`, and can pass arguments to the environment constructor:
```python
env = create("brax::ant", env_kwargs={"backend": "positional"})
```

Adapters disable native episode horizons and automatic resetting. They capture the
backend's finite default horizon before doing so, preventing double truncation or reset
behavior.

Adapters implement the `HasFromNameInit` protocol, that ensures they can be created via
a `from_name(cls, env_name: str, env_kwargs=None, **kwargs)` function. This is used by
`create`. The `env_kwargs` are passed to the suite's environment constructor (such as
`navix.make` or `brax.envs.create`), and the `kwargs` are passed to the adapter class on
creation. Caller-owned mappings and backend parameter objects are left unchanged.

`create(..., max_episode_steps="default")` applies the captured backend horizon with an
Envelope `TruncationWrapper`. A positive integer overrides that horizon, while `None`
disables outer truncation.

Raw suite metadata is normalized into a stable `info.backend` `Container`, with fields
available through attribute access such as `info.backend.metrics`. Its structure remains
fixed across `init`, `reset`, and `step`; adapters use zero-like placeholders when a
backend does not emit reset-time metadata. For those adapters,
`info.backend.valid=False` identifies a placeholder and real step metadata sets it to
true.

## API Reference (`create`)

::: envelope.adapters.create

::: envelope.adapters.HasFromNameInit

## API Reference (Specific Adapters)

::: envelope.adapters.brax_envelope.BraxEnvelope

::: envelope.adapters.craftax_envelope.CraftaxEnvelope

::: envelope.adapters.gymnax_envelope.GymnaxEnvelope

::: envelope.adapters.jumanji_envelope.JumanjiEnvelope

::: envelope.adapters.kinetix_envelope.KinetixEnvelope

::: envelope.adapters.mujoco_playground_envelope.MujocoPlaygroundEnvelope

::: envelope.adapters.navix_envelope.NavixEnvelope
