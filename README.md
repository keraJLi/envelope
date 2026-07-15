# 💌 Envelope: a JAX-native environment interface

```python
# Create environments from JAX-native suites you have installed, ...
env = envelope.create("gymnax::CartPole-v1")

# ... interact with the environments using a simple interface, ...
state, info = env.init(key)
states, infos = jax.lax.scan(env.step, state, actions)
plt.plot(infos.reward.cumsum())

# ... and enjoy a powerful ecosystem of wrappers.
env = envelope.wrappers.AutoResetWrapper(env)
env = envelope.wrappers.VmapWrapper(env)
env = envelope.wrappers.ObservationNormalizationWrapper(env)
```

## 🌍 Simple, expressive interaction!

- **Environments are pytrees**. Squish them through JAX transformations and trace their parameters.
- **Idiomatic jax-y interface** of `init(key: Key) -> State, Info` and `step(state: State, action: PyTree) -> State, Info`. You can directly `jax.scan` over a `step(...)`!
- **Spaces are super simple**. No `Tuple`, `Dict` nonsense! There are two spaces: `Continuous` and `Discrete`, which you can compose into a `PyTreeSpace`.
- **Explicit episode truncation** supports correctly handling bootstrapping for value-function targets.
- **No auto-reset** by default. Resetting every step can be expensive!

## 💪 Powerful, composable wrappers!

- **Carry state across episodes** to track running statistics, for example to normalize observations.
- **Explicit wrapper composition** keeps episode boundaries correct. Stateless transforms
  are flexible, while autoreset, statistics, truncation, and pooling follow the
  [supported ordering](https://jax-envelope.readthedocs.io/en/latest/api/wrappers/#wrapper-ordering).

## 🔌 Adapters for existing suites


| 📦                                                                        | # 🤖    | # 🌍    |
| ------------------------------------------------------------------------- | ------- | ------- |
| [brax](https://github.com/google/brax)                                    | 🕺      | 12      |
| [craftax](https://github.com/MichaelTMatthews/craftax)                    | 🕺      | 4       |
| [gymnax](https://github.com/RobertTLange/gymnax)                          | 🕺      | 24      |
| [jumanji](https://github.com/instadeepai/jumanji)                         | 🕺 / 👯 | 25 / 1  |
| [kinetix](https://github.com/flairox/kinetix)                             | 🕺      | 74      |
| [mujoco_playground](https://github.com/google-deepmind/mujoco_playground) | 🕺      | 54      |
| [navix](https://github.com/instadeepai/navix)                             | 🕺      | 41      |
|                                                                           |         |         |
| Total                                                                     | 🕺 / 👯 | 234 / 1 |


```python
envelope.create("📦::🌍")
```

let's you create environments from any of the above!

## 📝 Testing

- **Default (no optional adapters deps required)**: `uv run pytest -m "not adapters"`
- **Adapters suite (requires full adapters dependency group)**:
  - `uv sync --group adapters`
  - `uv run pytest -m adapters`
  - If any adapter dependency is missing/broken, the run will fail fast with an error telling you what to install.

## 🏗️ Installation

```bash
pip install jax-envelope
pip install "jax-envelope[brax]"        # one published adapter extra
pip install "jax-envelope[adapters]"    # all published adapter extras
```

Gymnax and Kinetix currently rely on pinned upstream fixes and therefore remain
source-backed development adapters. Their exact install commands are documented in the
[adapter guide](https://jax-envelope.readthedocs.io/en/latest/api/adapters/#source-backed-adapters).

See the [0.5 migration guide](https://jax-envelope.readthedocs.io/en/latest/migration-0.5/)
when upgrading from Envelope 0.4.

## 💞 Related projects

- [stoa](https://github.com/EdanToledo/Stoa) is a very similar project that provides adapters and wrappers for the jumanji-like interface.
- Check out all the great suites we have adapters for! [gymnax](https://github.com/RobertTLange/gymnax), [brax](https://github.com/google/brax), [jumanji](https://github.com/instadeepai/jumanji), [kinetix](https://github.com/flairox/kinetix), [craftax](https://github.com/MichaelTMatthews/craftax), [navix](https://github.com/instadeepai/navix), [mujoco_playground](https://github.com/google-deepmind/mujoco_playground).
- We will be adding support for [jaxmarl](https://github.com/flairox/jaxmarl) and [pgx](https://github.com/sotetsuk/pgx) in the future, as soon as we figured out the best ever MARL interface for JAX!
