from functools import cached_property
from typing import override

import jax
import jax.numpy as jnp

from envelope.environment import Info, State
from envelope.spaces import BatchedSpace, Continuous, Discrete, PyTreeSpace, Space
from envelope.typing import PyTree
from envelope.wrappers.wrapper import Wrapper


def _discretize_space(space: Space, num_bins: int) -> Space:
    """Return a Discrete-based space mirroring a Continuous space.

    The resulting `Discrete.n` has the same shape as the original continuous space,
    with each entry equal to `num_bins`. Batched and PyTree spaces are handled
    recursively.
    """
    if isinstance(space, Discrete):
        return space
    if isinstance(space, Continuous):
        return Discrete.from_shape(num_bins, space.shape)
    if isinstance(space, PyTreeSpace):
        tree = jax.tree.map(
            lambda s: _discretize_space(s, num_bins),
            space.tree,
            is_leaf=lambda node: isinstance(node, Space),
        )
        return PyTreeSpace(tree)
    if isinstance(space, BatchedSpace):
        return BatchedSpace(
            space=_discretize_space(space.space, num_bins), batch_size=space.batch_size
        )
    raise TypeError(f"Unsupported action space type for discretization: {type(space)}")


def _undiscretize_action(action: PyTree, space: Space, num_bins: int) -> PyTree:
    # Here `action` is a PyTree of discrete actions, space is non-discrete action space
    if isinstance(space, Discrete):
        return action
    if isinstance(space, Continuous):
        low = jnp.asarray(space.low)
        high = jnp.asarray(space.high)
        a = jnp.asarray(action, dtype=jnp.float32)
        frac = (a + 0.5) / float(num_bins)
        return low + frac * (high - low)
    if isinstance(space, PyTreeSpace):
        return jax.tree.map(
            lambda a, s: _undiscretize_action(a, s, num_bins),
            action,
            space.tree,
            is_leaf=lambda node: isinstance(node, Space),
        )
    if isinstance(space, BatchedSpace):
        return jax.vmap(lambda a: _undiscretize_action(a, space.space, num_bins))(
            action
        )
    raise TypeError(f"Unsupported action space type for discretization: {type(space)}")


class DiscretizeActionWrapper(Wrapper):
    num_bins: int = 51

    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        inner_action = _undiscretize_action(
            action, self.env.action_space, self.num_bins
        )
        return self.env.step(state, inner_action)

    @override
    @cached_property
    def action_space(self) -> Space:
        return _discretize_space(self.env.action_space, self.num_bins)
