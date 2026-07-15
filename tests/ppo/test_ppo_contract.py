from __future__ import annotations

import importlib
import sys

import jax.numpy as jnp
import pytest


pytestmark = pytest.mark.ppo


def _networks():
    pytest.importorskip("distrax")
    pytest.importorskip("flax")
    return importlib.import_module("examples.ppo.networks")


def _ppo():
    pytest.importorskip("distrax")
    pytest.importorskip("flax")
    pytest.importorskip("optax")
    pytest.importorskip("tyro")
    return importlib.import_module("examples.ppo.ppo")


def test_multidiscrete_bijector_round_trip() -> None:
    bijector = _networks().ReshapeCategoricalBijector(jnp.asarray([2, 3]))
    flat = jnp.arange(6)

    shaped, forward_log_det = bijector.forward_and_log_det(flat)
    restored, inverse_log_det = bijector.inverse_and_log_det(shaped)

    assert shaped.shape == (6, 2)
    assert jnp.array_equal(restored, flat)
    assert jnp.array_equal(forward_log_det, jnp.zeros((6,)))
    assert jnp.array_equal(inverse_log_det, jnp.zeros((6,)))


def test_diagonal_gaussian_entropy_sums_event_dimensions() -> None:
    log_std = jnp.asarray([[0.0, jnp.log(2.0)], [jnp.log(0.5), 0.0]])

    entropy = _networks().diagonal_gaussian_entropy(log_std)
    per_dimension = 0.5 * (1.0 + jnp.log(2.0 * jnp.pi)) + log_std

    assert entropy.shape == (2,)
    assert jnp.allclose(entropy, per_dimension.sum(axis=-1))


@pytest.mark.parametrize(
    ("terminated", "truncated", "bootstrap", "expected"),
    [
        (True, False, 9.0, 0.8),
        (False, True, 2.0, 2.6),
    ],
)
def test_gae_distinguishes_termination_and_truncation(
    terminated: bool, truncated: bool, bootstrap: float, expected: float
) -> None:
    advantages = _ppo().compute_gae(
        rewards=jnp.asarray([1.0]),
        values=jnp.asarray([0.2]),
        bootstrap_values=jnp.asarray([bootstrap]),
        terminated=jnp.asarray([terminated]),
        truncated=jnp.asarray([truncated]),
        gamma=0.9,
        gae_lambda=0.95,
    )

    assert jnp.allclose(advantages, jnp.asarray([expected]))


def test_minibatches_must_divide_the_rollout() -> None:
    data = {"x": jnp.zeros((3, 2, 1))}

    with pytest.raises(ValueError, match="divisible"):
        _ppo().shuffle_and_split(data, num_minibatches=4, key=jnp.asarray([0, 1]))


def test_import_does_not_require_wandb() -> None:
    sys.modules.pop("examples.ppo.ppo", None)
    sys.modules["wandb"] = None
    try:
        _ppo()
    finally:
        sys.modules.pop("wandb", None)


def test_tiny_cpu_training_update_without_wandb() -> None:
    sys.modules.pop("wandb", None)

    metrics = _ppo().tiny_cpu_train_step(seed=0)

    assert int(metrics["updates"]) == 1
    assert bool(jnp.isfinite(metrics["loss_before"]))
    assert bool(jnp.isfinite(metrics["loss_after"]))
    assert "wandb" not in sys.modules
