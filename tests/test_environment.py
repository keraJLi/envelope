from functools import cached_property

import jax.numpy as jnp
import pytest

from envelope.environment import Environment, InfoContainer
from envelope.spaces import Continuous


class DefaultPoolingEnvironment(Environment):
    @cached_property
    def observation_space(self) -> Continuous:
        return Continuous(low=-jnp.inf, high=jnp.inf)

    @cached_property
    def action_space(self) -> Continuous:
        return Continuous(low=-1.0, high=1.0)

    def init(self, key):
        del key
        state = jnp.asarray(0.0)
        return state, InfoContainer(
            obs=state, reward=0.0, terminated=False, truncated=False
        )

    def step(self, state, action):
        state = state + action
        return state, InfoContainer(
            obs=state, reward=action, terminated=False, truncated=False
        )


def test_space_descriptors_remain_abstract() -> None:
    class MissingSpaces(Environment):
        def init(self, key):
            raise NotImplementedError

        def step(self, state, action):
            raise NotImplementedError

    assert {"observation_space", "action_space"} <= Environment.__abstractmethods__
    with pytest.raises(TypeError, match="abstract"):
        MissingSpaces()


def test_environments_support_init_pooling_by_default() -> None:
    assert DefaultPoolingEnvironment().supports_init_pooling is True
