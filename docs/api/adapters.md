# Adapters

## Creating adapters

Envelope has a powerful `create` method that can create environments from many different
JAX-based environment suites, and automatically casts them into the envelope
`Environment` interface. The `create` function uses a unique environment identifier of
the format `SUITE::ENVIRONMENT`, and can pass arguments to the environment constructor:
```python
env = create("brax::ant", env_kwargs={"backend": "positional"})
short_env = create("brax::ant", max_episode_steps=200)
unlimited_env = create("brax::ant", max_episode_steps=None)
```

Native time limits and automatic resetting are disabled by adapters. Envelope captures
the backend's original finite horizon before disabling it. The generic factory then uses
that value when `max_episode_steps="default"`, lets a positive integer override it, or
disables outer truncation with `None`.

Adapters implement the `HasFromNameInit` protocol, that ensures they can be created via
a `from_name(cls, env_name: str, env_kwargs=None, **kwargs)` function. This is used by
`create`. The `env_kwargs` are passed to the suite's environment constructor (such as
`navix.make` or `brax.envs.create`), and the `kwargs` are passed to the adapter class on
creation.

Caller-owned keyword mappings and backend parameter objects are copied rather than
mutated. Raw suite metadata is available under `info.backend`, with stable attribute
access such as `info.backend.metrics`. The metadata schema is fixed for the environment
lifecycle. A reset-time zero placeholder is accompanied by `info.backend.valid=False`
when the suite has no real metadata for that emission.

## Installation

Published adapters have bounded extras:

```bash
pip install "jax-envelope[navix]"
pip install "jax-envelope[adapters]"
```

### Source-backed adapters

Gymnax and Kinetix depend on fixes that are not yet part of the supported indexed
releases. Install the pinned revisions explicitly:

```bash
pip install "gymnax @ git+https://github.com/RobertTLange/gymnax.git@18f2e7f3cffafc7042c76fdc538c83957418a9a9"
pip install "kinetix-env @ git+https://github.com/FLAIROx/Kinetix.git@df4de60cabd42dbd1c35fb5214fdc6728710e33d"
pip install jax-envelope
```

The Gymnax pin protects
`tests/adapters/test_gymnax_regressions.py`, especially capture-before-disable of a
caller-supplied horizon and a stable backend-info schema. The Kinetix pin protects
`tests/adapters/test_kinetix_envelope.py`, including random and premade level loading,
auto-reset rejection, and immutable horizon replacement. Move either adapter back to an
indexed release only after that named regression suite passes unchanged.

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
