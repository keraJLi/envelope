import pytest

from envelope.environment import Environment


def test_space_descriptors_remain_abstract() -> None:
    class MissingSpaces(Environment):
        def init(self, key):
            raise NotImplementedError

        def step(self, state, action):
            raise NotImplementedError

    assert {"observation_space", "action_space"} <= Environment.__abstractmethods__
    with pytest.raises(TypeError, match="abstract"):
        MissingSpaces()
