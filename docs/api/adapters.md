# Adapters

## Creating adapters

Envelope has a powerful `create` method that can create environments from many different
JAX-based environment suites, and automatically casts them into the envelope
`Environment` interface. The `create` function uses a unique environment identifier of
the format `SUITE::ENVIRONMENT`, and can pass arguments to the environment constructor:
```python
env = create("brax::ant", env_kwargs={"backend": "positional"})
```

**Note**: Many suites natively support episode truncation and automatic resetting. When
instantiating adapters, users are highly discouraged from using these features. Instead,
they should use the envelope-native wrapper ecosystem. 

Adapters implement the `HasFromNameInit` protocol, that ensures they can be created via
a `from_name(cls, env_name: str, env_kwargs=None, **kwargs)` function. This is used by
`create`. The `env_kwargs` are passed to the suite's environment constructor (such as
`navix.make` or `brax.envs.create`), and the `kwargs` are passed to the adapter class on
creation.

Each adapter may have a `default_episode_length` property that is populated depending on
the suite and specific environment. If it is not `None`, the `create` function will wrap
the adapter in a `TruncationWrapper` before returning it.

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
