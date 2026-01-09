# 🌍 Rejax gets it's own environment API!

## Rationale

Currently, the `rejax.compat` module is built around the popular `gymnax` API, and defines several wrappers that transform other environments to `gymnax` ones. While `gymnax` served as a great stepping stone, it comes with some limitations; both in terms of scope and usability.

- **Scope**: `gymnax` implements popular gym environments, and it's API is sufficient to interact with them. However, RL training often requires more sophisticated environment transformations, such as wrappers that keep state across episodes or rollouts with agent state.
- **Usability**: the `gymnax` API is very verbose. For example, all evironment functions take environment parameters as an argument, even though they should never change during an episode. 

My main goal is to simplify the environment API, by improving on wrapping, typing, baking in parameters, and returning compact objects when stepping.

## Usage example
```python
import jenv

env = jenv.create("gymnax/CartPole-v1")
state, step_info = env.reset(key)
action = env.action_space.sample(key)
state, step_info = env.step(state, action)
print(step_info.obs, step_info.reward)

states, step_infos = jax.lax.scan(env.step, state, actions)
plt.plot(step_infos.reward.cumsum())  # plot cumulative reward
```

## Good wrapper support
Wrappers around gymnax environments can be hacky. Jenv promises to provide explicit support for wrappers that is easy to extend and comprehend. Wrapping semantics can quickly become complicated; let me outline the basic ones here. Consider wrappers
- `AutoResetWrapper` (A): Resets environments on termination and truncation
- `ObservationNormalizationWrapper` (O): normalizes observations, has state that persists across episodes
- `VmapWrapper` (V): Vectorizes an environment using `jax.vmap(step)`

By applying them in different orders, we get different semantics.

## State and reset semantics
- `State` is an opaque PyTree defined by each environment. Wrappers expose the wrapped env state as `inner_state` and may add their own fields.
- `reset(key, state=None, **kwargs)` accepts an optional prior state for cross-episode persistence and arbitrary kwargs for env/wrapper-specific needs. `step(state, action, **kwargs)` mirrors this.
- When using `jit`/`vmap`/`pmap`, pass a concrete state pytree (not `None`) to keep shapes/static structure stable; normalize `None` to a sentinel outside the traced function if needed.
- Wrapper ordering still matters (e.g., `ObservationNormalizationWrapper` before vs. after `VmapWrapper` gives per-env vs. global normalization), but kwargs and optional state are forwarded through the stack.
