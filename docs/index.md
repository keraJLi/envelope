# 💌 Envelope

The docs give a high-level overview of the design decisions and semantics of Envelope!


## 👀 Overview

Envelope is a JAX-native environment interface for reinforcement learning. It provides

- a unified `Environment` base class with `init`, `reset`, and `step` methods
- a composable wrapper ecosystem for common RL patterns (automatic resets, episode
  truncation, observation normalization, vectorization, ...)
- adapters that cast popular JAX RL suites (Brax, Craftax, Gymnax, Jumanji, Kinetix,
  MuJoCo Playground, Navix) into the envelope interface

All environments, wrappers, states and infos objects are JAX pytrees, so they work
natively with `jax.jit`, `jax.vmap`, and `jax.lax.scan`.


## ⚡️ Quickstart

1. Install:
    ```bash
    pip install jax-envelope
    ```
2. Use:

```python
import envelope, jax

env = envelope.create("gymnax::CartPole-v1")

key = jax.random.key(0)
state, info = env.init(key)
print(info)  # InfoContainer(obs=Array(...), reward=0.0, terminated=False, ...)
```


## 📖 Docs

- **[Environments](api/environments.md)**: the base `Environment` interface and its
  lifecycle.
- **[Spaces](api/spaces.md)**: observation and action space definitions.
- **[Wrappers](api/wrappers.md)**: composable wrappers and how to order them.
- **[Adapters](api/adapters.md)**: creating environments from third-party suites.
- **[Struct](api/struct.md)**: JAX pytree datastructures used throughout envelope.
- **[Migrating to 0.5](migration-0.5.md)**: intentional contract changes from 0.4.
