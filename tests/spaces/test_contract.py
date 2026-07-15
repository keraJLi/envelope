"""Focused behavioral contract for Envelope 0.5 spaces."""

import jax
import jax.numpy as jnp
import pytest

from envelope.spaces import BatchedSpace, Continuous, Discrete


@pytest.mark.parametrize("n", [0, -1, jnp.array([2, 0], dtype=jnp.int32)])
def test_discrete_rejects_non_positive_n(n):
    with pytest.raises(ValueError):
        Discrete(n=n)


@pytest.mark.parametrize("n", [1.5, jnp.array(2.0, dtype=jnp.float32), jnp.array(True)])
def test_discrete_rejects_non_integer_n(n):
    with pytest.raises(TypeError):
        Discrete(n=n)


def test_continuous_validates_bounds_at_construction():
    with pytest.raises(ValueError):
        Continuous(low=jnp.zeros(1), high=jnp.ones(2))
    with pytest.raises(ValueError):
        Continuous(
            low=jnp.zeros(1, dtype=jnp.float16),
            high=jnp.ones(1, dtype=jnp.float32),
        )
    with pytest.raises(TypeError):
        Continuous(low=jnp.array(0), high=jnp.array(1))

    for low, high in [
        (jnp.nan, 1.0),
        (1.0, 0.0),
        (jnp.inf, jnp.inf),
        (-jnp.inf, -jnp.inf),
    ]:
        with pytest.raises(ValueError):
            Continuous(low=low, high=high)


@pytest.mark.parametrize(
    ("batch_size", "error"),
    [(0, ValueError), (-1, ValueError), (True, TypeError), (2.5, TypeError)],
)
def test_batched_space_requires_positive_integer_batch_size(batch_size, error):
    with pytest.raises(error):
        BatchedSpace(space=Discrete(2), batch_size=batch_size)


@pytest.mark.parametrize("dtype", [jnp.int8, jnp.int16, jnp.int32, jnp.uint8])
def test_discrete_contains_any_integer_width(dtype):
    space = Discrete(jnp.array([3, 4], dtype=jnp.int32))
    assert space.contains(jnp.array([1, 2], dtype=dtype))


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32, jnp.int8, jnp.int32])
def test_continuous_contains_any_real_numeric_candidate(dtype):
    space = Continuous.from_shape(0.0, 2.0, (2,))
    assert space.contains(jnp.array([1, 2], dtype=dtype))


@pytest.mark.parametrize("candidate", [jnp.array([1.0, 2.0]), jnp.array([True, False])])
def test_discrete_rejects_non_integer_candidates(candidate):
    assert not Discrete.from_shape(3, (2,)).contains(candidate)


@pytest.mark.parametrize(
    "candidate", [jnp.array([True, False]), jnp.array([1 + 0j, 2 + 0j])]
)
def test_continuous_rejects_non_real_numeric_candidates(candidate):
    assert not Continuous.from_shape(0.0, 2.0, (2,)).contains(candidate)


@pytest.mark.parametrize("candidate", [jnp.array(1), jnp.ones((1, 2), dtype=int)])
def test_discrete_contains_requires_exact_shape(candidate):
    assert not Discrete.from_shape(3, (2,)).contains(candidate)


@pytest.mark.parametrize("candidate", [jnp.array(0.5), jnp.ones((1, 2))])
def test_continuous_contains_requires_exact_shape(candidate):
    assert not Continuous.from_shape(0.0, 1.0, (2,)).contains(candidate)


def test_batched_contains_requires_declared_leading_size():
    space = BatchedSpace(Discrete(3), batch_size=3)
    assert not space.contains(jnp.array([0, 1], dtype=jnp.int32))
    assert not space.contains(jnp.array([0, 1, 2, 0], dtype=jnp.int32))


@pytest.mark.parametrize(
    "space",
    [
        Continuous.from_shape(-jnp.inf, jnp.inf, (3,)),
        Continuous(
            low=jnp.array([-jnp.inf, 0.0, -jnp.inf, 1.0]),
            high=jnp.array([jnp.inf, jnp.inf, 2.0, 1.0]),
        ),
    ],
)
def test_unbounded_continuous_sampling_is_finite_and_contained(space):
    samples = jax.vmap(space.sample)(jax.random.split(jax.random.key(0), 32))
    assert jnp.all(jnp.isfinite(samples))
    assert jnp.all(jax.vmap(space.contains)(samples))


def test_space_validation_does_not_concretize_traced_parameters():
    @jax.jit
    def sample_discrete(n, key):
        return Discrete(n=n).sample(key)

    @jax.jit
    def sample_continuous(low, high, key):
        return Continuous(low=low, high=high).sample(key)

    key = jax.random.key(0)
    assert sample_discrete(jnp.array(3), key).shape == ()
    sample = sample_continuous(jnp.zeros(2), jnp.ones(2), key)
    assert sample.shape == (2,)
