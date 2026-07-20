# Adapters

## Creating adapters

Envelope has a powerful `create` method that can create environments from many different
JAX-based environment suites, and automatically casts them into the envelope
`Environment` interface. The `create` function uses a unique environment identifier of
the format `SUITE::ENVIRONMENT`, and can pass arguments to the environment constructor:
```python
env = create("brax::ant", env_kwargs={"backend": "positional"})
```

Adapters choose backend settings that leave episode boundaries to Envelope by default.
Explicit backend options are still passed through. Be careful when enabling native
time limits, auto-reset, batching, or action repetition. Adapters warn when they can
recognize such an explicit setting, but do not reject it.

Adapters implement the `HasFromNameInit` protocol, that ensures they can be created via
a `from_name(cls, env_name: str, env_kwargs=None, **kwargs)` function. This is used by
`create`. The `env_kwargs` are passed to the suite's environment constructor (such as
`navix.make` or `brax.envs.create`), and the `kwargs` are passed to the adapter class on
creation.

With `max_episode_steps="default"`, `create` applies the environment's default time
limit with a `TruncationWrapper`. `None` adds no outer time limit.

Most adapters expose extra suite data under `info.backend`, for example as
`info.backend.metrics`. If a suite has no such data at reset, those fields contain zero
placeholders and `info.backend.valid` is false. Navix retains its established top-level
extra fields.

## API Reference (`create`)

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
