from typing import Any, Protocol, TypeAlias, runtime_checkable

import jax

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

    obs: PyTree
    reward: float
    terminated: bool
    truncated: bool

    def update(self, **changes: PyTree) -> "Info":
        """Update the info container with new values. This method should return
        a new instance with updated and potentially new values."""
        ...

    def __getattr__(self, name: str) -> PyTree: ...
