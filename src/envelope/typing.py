from typing import Any, Protocol, Self, TypeAlias, runtime_checkable

import jax

__all__ = ["Array", "Info", "Key", "PyTree", "State"]

PyTree: TypeAlias = Any
Key: TypeAlias = jax.Array  # with jnp.issubdtype(key.dtype, jax.dtypes.prng_key)
Array: TypeAlias = jax.Array
State: TypeAlias = PyTree


@runtime_checkable
class Info(Protocol):
    """
    `Info` is a runtime-checkable Protocol that defines required fields and methods for
    environment emissions, including observation, reward, and termination/truncation
    flags.

    Attributes:
        obs: The observation from the environment.
        reward: The reward from the environment.
        terminated: Whether the episode has terminated.
        truncated: Whether the episode has truncated.

    """

    @property
    def obs(self) -> PyTree: ...

    @property
    def reward(self) -> float | Array: ...

    @property
    def terminated(self) -> bool | Array: ...

    @property
    def truncated(self) -> bool | Array: ...

    def update(self, **changes: PyTree) -> Self:
        """Update the info container with new values. This method should return
        a new instance with updated and potentially new values."""
        ...
