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
