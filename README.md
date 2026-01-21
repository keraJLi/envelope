# 💌 Envelope: a JAX-native environment interface
```python
# Create environments from JAX-native suites you have installed, ...
env = envelope.create("gymnax::CartPole-v1")

# ... interact with the environments using a simple interface, ...
state, info = env.reset(key)
states, infos = jax.lax.scan(env.step, state, actions)
plt.plot(infos.reward.cumsum())

# ... and enjoy a powerful ecosystem of wrappers.
env = envelope.wrappers.AutoResetWrapper(env)
env = envelope.wrappers.VmapWrapper(env)
env = envelope.wrappers.ObservationNormalizationWrapper(env)
```

## 🌍 Simple, expressive interaction!
* **Environments are pytrees**. Squish them through JAX transformations and trace their parameters.
* **Idiomatic jax-y interface** of `reset(key: Key) -> State, Info` and `step(state: State, action: PyTree) -> State, Info`. You can directly `jax.scan` over a `step(...)`!
* **Spaces are super simple**. No `Tuple`, `Dict` nonsense! There are two spaces: `Continuous` and `Discrete`, which you can compose into a `PyTreeSpace`.
* **Explicit episode truncation** supports correctly handling bootstrapping for value-function targets.
* **No auto-reset** by default. Resetting every step can be expensive!

## 💪 Powerful, composable wrappers!
* **Carry state across episodes** to track running statistics, for example to normalize observations.
* **Composable wrappers** can be stacked in any order. For example, `ObservationNormalizationWrapper` before vs. after `VmapWrapper` gives per-env vs. global normalization.
<!-- TODO: Add auto-reset behavior (including state injection) and optimistic resets once I implement them. -->

## 🔌 Adapters for existing suites
| 📦 | # 🤖 | # 🌍 |
|------|------|------|
| [gymnax](https://github.com/RobertTLange/gymnax) | 🕺 | 24 |
| [brax](https://github.com/google/brax) | 🕺 | 12 |
| [jumanji](https://github.com/instadeepai/jumanji) | 🕺 / 👯 | 25 / 1 |
| [kinetix](https://github.com/flairox/kinetix) | 🕺 | 74 |
| [craftax](https://github.com/MichaelTMatthews/craftax) | 🕺 | 4 |
| [mujoco_playground](https://github.com/google-deepmind/mujoco_playground) | 🕺 | 54 |
| | |
| Total | 🕺 / 👯 | 193 / 1 |

```python
envelope.create("📦::🌍")
```
let's you create environments from any of the above!

## 📝 Testing
- **Default (no optional compat deps required)**: `uv run pytest -m "not compat"`
- **Compat suite (requires full compat dependency group)**:
  - `uv sync --group compat`
  - `uv run pytest -m compat`
  - If any compat dependency is missing/broken, the run will fail fast with an error telling you what to install.

## 🏗️ Installation
```bash
pip install jax-envelope
```

## 💞 Related projects
* [stoax](https://github.com/EdanToledo/Stoa) is a very similar project that provides adapters and wrappers for the jumanji-like interface.
* Check out all the great suites we have adapters for! [gymnax](https://github.com/RobertTLange/gymnax), [brax](https://github.com/google/brax), [jumanji](https://github.com/instadeepai/jumanji), [kinetix](https://github.com/flairox/kinetix), [craftax](https://github.com/MichaelTMatthews/craftax), [mujoco_playground](https://github.com/google-deepmind/mujoco_playground).
* We will be adding support for [jaxmarl](https://github.com/flairox/jaxmarl) and [pgx](https://github.com/sotetsuk/pgx) in the future, as soon as we figured out the best ever MARL interface for JAX!
