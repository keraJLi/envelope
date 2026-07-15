from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import jax
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


def test_clipped_gaussian_policy_respects_bounds() -> None:
    flax = pytest.importorskip("flax")
    import envelope

    networks = _networks()
    obs_space = envelope.Continuous.from_shape(-1.0, 1.0, shape=(4,))
    action_space = envelope.Continuous(
        low=jnp.asarray([-2.0, -1.0]),
        high=jnp.asarray([1.0, 3.0]),
    )
    policy = networks.GaussianPolicy(
        obs_space,
        action_space,
        flax.nnx.Rngs(0),
        layer_size=8,
    )

    distribution = policy(jnp.zeros((3, 4)))
    actions = distribution.sample(seed=jax.random.key(1))
    log_prob = distribution.log_prob(actions)
    entropy = distribution.entropy()

    assert actions.shape == (3, 2)
    assert log_prob.shape == (3,)
    assert entropy.shape == (3, 2)
    assert bool(jnp.all(actions >= action_space.low))
    assert bool(jnp.all(actions <= action_space.high))
    assert bool(jnp.all(jnp.isfinite(log_prob)))
    assert bool(jnp.all(jnp.isfinite(entropy)))


@pytest.mark.parametrize(
    ("cardinalities", "expected_action_shape"),
    [
        (jnp.asarray(3), (4,)),
        (jnp.asarray([2, 3]), (4, 2)),
    ],
)
def test_discrete_policy_sample_and_log_prob_shapes(
    cardinalities, expected_action_shape
) -> None:
    flax = pytest.importorskip("flax")
    import envelope

    networks = _networks()
    policy = networks.DiscretePolicy(
        envelope.Continuous.from_shape(-1.0, 1.0, shape=(4,)),
        envelope.Discrete(n=cardinalities),
        flax.nnx.Rngs(0),
        layer_size=8,
    )
    distribution = policy(jnp.zeros((4, 4)))

    actions = distribution.sample(seed=jax.random.key(2))
    log_prob = distribution.log_prob(actions)

    assert actions.shape == expected_action_shape
    assert log_prob.shape == (4,)
    assert bool(jnp.all(jnp.isfinite(log_prob)))


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
    import envelope

    rollout = envelope.InfoContainer(
        obs=jnp.zeros((2, 1, 1)),
        reward=jnp.asarray([[1.0], [100.0]]),
        terminated=jnp.asarray([[terminated], [False]]),
        truncated=jnp.asarray([[truncated], [False]]),
    ).update(
        value=jnp.asarray([[0.2], [0.0]]),
        value_next=jnp.asarray([[bootstrap], [0.0]]),
    )
    train_state = SimpleNamespace(
        args=SimpleNamespace(gamma=0.9, gae_lambda=0.95),
        env_info=SimpleNamespace(
            obs=jnp.zeros((1, 1)),
            terminated=jnp.asarray([False]),
            truncated=jnp.asarray([False]),
            final=SimpleNamespace(obs=jnp.zeros((1, 1))),
        ),
        value_fn=lambda obs: obs[..., 0],
    )

    advantages = _ppo().calculate_gae(train_state, rollout)

    assert jnp.allclose(advantages[0], jnp.asarray([expected]))
    assert jnp.allclose(advantages[1], jnp.asarray([100.0]))


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


def test_observation_normalization_uses_elementwise_autoreset(monkeypatch) -> None:
    import envelope
    from tests.wrappers.helpers import StepCounterEnv

    ppo = _ppo()
    monkeypatch.setattr(ppo.envelope, "create", lambda _name: StepCounterEnv())
    args = ppo.Args(normalize_observations=True, num_envs=2)

    _env, vecenv = ppo.make_env(args)

    assert isinstance(vecenv, envelope.VmapWrapper)
    assert isinstance(vecenv.env, envelope.AutoResetWrapper)
    assert isinstance(vecenv.env.env, envelope.ObservationNormalizationWrapper)


def test_tiny_cpu_training_update_without_wandb() -> None:
    flax = pytest.importorskip("flax")
    optax = pytest.importorskip("optax")
    import envelope

    sys.modules.pop("wandb", None)
    value_fn = _networks().ValueFunction(
        envelope.Continuous.from_shape(-1.0, 1.0, shape=(4,)),
        flax.nnx.Rngs(0),
        layer_size=16,
    )
    optimizer = flax.nnx.Optimizer(value_fn, optax.adam(1e-3), wrt=flax.nnx.Param)
    observations = jnp.linspace(-1.0, 1.0, 32).reshape(8, 4)
    targets = jnp.linspace(-0.5, 0.5, 8)

    def loss_fn(model):
        return jnp.mean(jnp.square(model(observations) - targets))

    loss_before = loss_fn(value_fn)
    _loss, gradients = flax.nnx.value_and_grad(loss_fn)(value_fn)
    optimizer.update(value_fn, gradients)
    loss_after = loss_fn(value_fn)

    assert bool(jnp.isfinite(loss_before))
    assert bool(jnp.isfinite(loss_after))
    assert "wandb" not in sys.modules
