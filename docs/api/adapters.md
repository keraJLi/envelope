# Adapters

## Creating adapters

Envelope has a powerful `create` method that can create environments from many different
JAX-based environment suites, and automatically casts them into the envelope
`Environment` interface. The `create` function uses a unique environment identifier of
the format `SUITE::ENVIRONMENT`, and can pass arguments to the environment constructor:
```python
env = create("brax::ant", env_kwargs={"backend": "positional"})
```

Adapters turn off the suite's own time limit and auto-reset so Envelope can handle
episode boundaries itself. They keep the suite's original time limit for `create`.

Adapters implement the `HasFromNameInit` protocol, that ensures they can be created via
a `from_name(cls, env_name: str, env_kwargs=None, **kwargs)` function. This is used by
`create`. The `env_kwargs` are passed to the suite's environment constructor (such as
`navix.make` or `brax.envs.create`), and the `kwargs` are passed to the adapter class on
creation. The dictionaries and parameter objects passed by the caller are left
unchanged.

With `max_episode_steps="default"`, `create` applies the saved time limit with a
`TruncationWrapper`. A positive integer sets a different limit. `None` adds no outer
time limit.

Extra data returned by the suite is available under `info.backend`, for example as
`info.backend.metrics`. The same fields are present after `init`, `reset`, and `step`.
If a suite has no such data at reset, those fields contain zero placeholders and
`info.backend.valid` is false.

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
