import jax
import jax.numpy as jnp
from typing_extensions import override

from envelope.environment import Info, State
from envelope.spaces import BatchedSpace, Continuous, Discrete, PyTreeSpace, Space
from envelope.typing import PyTree
from envelope.wrappers.wrapper import Wrapper


def clip_action(action: PyTree, space: Space) -> PyTree:
    if isinstance(space, BatchedSpace):
        return jax.vmap(clip_action, in_axes=(0, None))(action, space.space)
    elif isinstance(space, PyTreeSpace):
        return jax.tree.map(clip_action, action, space.tree)
    elif isinstance(space, Continuous):
        return jnp.clip(action, space.low, space.high)
    elif isinstance(space, Discrete):
        return jnp.clip(action, 0, space.n - 1)


class ClipActionWrapper(Wrapper):
    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        action = clip_action(action, self.action_space)
        return self.env.step(state, action)
