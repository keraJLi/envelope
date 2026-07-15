# 💌 Envelope: a JAX-native environment interface

```python
# Create environments from JAX-native suites you have installed, ...
env = envelope.create("gymnax::CartPole-v1")

# ... interact with the environments using a simple interface, ...
state, info = env.init(key)
states, infos = jax.lax.scan(env.step, state, actions)
plt.plot(infos.reward.cumsum())

# ... and enjoy a powerful ecosystem of wrappers.
env = envelope.wrappers.ObservationNormalizationWrapper(env)
env = envelope.wrappers.AutoResetWrapper(env)
env = envelope.wrappers.VmapWrapper(env)
```

## 🌍 Simple, expressive interaction!

- **Environments are pytrees**. Squish them through JAX transformations and trace their parameters.
- **Idiomatic jax-y interface** of `init(key: Key) -> State, Info` and `step(state: State, action: PyTree) -> State, Info`. You can directly `jax.scan` over a `step(...)`!
- **Spaces are super simple**. No `Tuple`, `Dict` nonsense! There are two spaces: `Continuous` and `Discrete`, which you can compose into a `PyTreeSpace`.
- **Explicit episode truncation** supports correctly handling bootstrapping for value-function targets.
- **No auto-reset** by default. Resetting every step can be expensive!

## 💪 Powerful, composable wrappers!

- **Carry state across episodes** to track running statistics, for example to normalize observations.
- **Explicit wrapper composition** keeps episode boundaries correct. See the
  [wrapper ordering guide](https://github.com/keraJLi/envelope/blob/main/docs/api/wrappers.md#wrapper-ordering)
  for the supported standard and pooled stacks.

## 🔌 Adapters for existing suites


| 📦                                                                        | # 🤖    | # 🌍    |
| ------------------------------------------------------------------------- | ------- | ------- |
| [brax](https://github.com/google/brax)                                    | 🕺      | 12      |
| [craftax](https://github.com/MichaelTMatthews/craftax)                    | 🕺      | 4       |
| [gymnax](https://github.com/RobertTLange/gymnax)                          | 🕺      | 24      |
| [jumanji](https://github.com/instadeepai/jumanji)                         | 🕺 / 👯 | 25 / 1  |
| [kinetix](https://github.com/flairox/kinetix)                             | 🕺      | 4       |
| [mujoco_playground](https://github.com/google-deepmind/mujoco_playground) | 🕺      | 54      |
| [navix](https://github.com/instadeepai/navix)                             | 🕺      | 41      |
|                                                                           |         |         |
| Total                                                                     | 🕺 / 👯 | 164 / 1 |


```python
envelope.create("📦::🌍")
```

let's you create environments from any of the above!

## 📝 Testing

- **Default (no optional adapters deps required)**: `uv run pytest -m "not adapters"`
- **Adapters suite (requires full adapters dependency group)**:
  - `uv sync --group adapters`
  - `uv run pytest -m adapters`
  - Tests for suites unavailable in a partial installation are skipped.

## 🏗️ Installation

```bash
pip install jax-envelope
pip install "jax-envelope[navix]"       # one published adapter
pip install "jax-envelope[adapters]"    # all published adapters
```

Gymnax and Kinetix are temporarily source-backed development adapters rather than
published extras.

## 💞 Related projects

- [stoa](https://github.com/EdanToledo/Stoa) is a very similar project that provides adapters and wrappers for the jumanji-like interface.
- Check out all the great suites we have adapters for! [gymnax](https://github.com/RobertTLange/gymnax), [brax](https://github.com/google/brax), [jumanji](https://github.com/instadeepai/jumanji), [kinetix](https://github.com/flairox/kinetix), [craftax](https://github.com/MichaelTMatthews/craftax), [navix](https://github.com/instadeepai/navix), [mujoco_playground](https://github.com/google-deepmind/mujoco_playground).
- We will be adding support for [jaxmarl](https://github.com/flairox/jaxmarl) and [pgx](https://github.com/sotetsuk/pgx) in the future, as soon as we figured out the best ever MARL interface for JAX!
