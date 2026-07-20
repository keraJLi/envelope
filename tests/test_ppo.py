from types import SimpleNamespace

import jax.numpy as jnp
import pytest

import envelope

nnx = pytest.importorskip("flax.nnx")
networks = pytest.importorskip("examples.ppo.networks")
ppo = pytest.importorskip("examples.ppo.ppo")

GaussianPolicy = networks.GaussianPolicy
WEIGHT_DECAY = ppo.WEIGHT_DECAY
Args = ppo.Args
bootstrap_observation = ppo.bootstrap_observation
get_num_updates = ppo.get_num_updates
make_lr_schedule = ppo.make_lr_schedule
summarize_block = ppo.summarize_block

pytestmark = pytest.mark.ppo


def test_gaussian_policy_clips_log_standard_deviation():
    obs_space = envelope.Continuous.from_shape(-1.0, 1.0, shape=(3,))
    action_space = envelope.Continuous.from_shape(-1.0, 1.0, shape=(2,))
    policy = GaussianPolicy(obs_space, action_space, nnx.Rngs(0))
    policy.action_log_std.kernel[...] = 0.0

    policy.action_log_std.bias[...] = 100.0
    high_scale = policy(jnp.zeros((1, 3))).distribution.distribution.scale
    policy.action_log_std.bias[...] = -100.0
    low_scale = policy(jnp.zeros((1, 3))).distribution.distribution.scale

    assert jnp.allclose(high_scale, jnp.exp(2.0))
    assert jnp.allclose(low_scale, jnp.exp(-5.0))


def test_publishable_defaults():
    args = Args()

    assert args.policy_lr == 3e-4
    assert args.value_fn_lr == 1e-4
    assert args.num_envs == 1024
    assert args.num_minibatches == 8
    assert args.num_epochs == 4
    assert args.num_steps == 128
    assert args.normalize_observations is True
    assert WEIGHT_DECAY == 1e-4


def test_cosine_schedule_reaches_zero():
    args = Args(
        total_timesteps=32,
        num_envs=4,
        num_steps=4,
        num_epochs=2,
        num_minibatches=2,
    )
    optimizer_steps = get_num_updates(args) * args.num_epochs * args.num_minibatches
    schedule = make_lr_schedule(3e-4, args)

    assert float(schedule(0)) == pytest.approx(3e-4)
    assert float(schedule(optimizer_steps)) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "args",
    [
        Args(total_timesteps=1),
        Args(num_minibatches=0),
        Args(updates_per_block=0),
        Args(total_timesteps=20, num_envs=3, num_steps=2, num_minibatches=4),
    ],
)
def test_invalid_training_shapes_are_rejected(args):
    with pytest.raises(ValueError):
        get_num_updates(args)


def test_bootstrap_observation_uses_final_only_at_episode_boundaries():
    info = SimpleNamespace(
        obs=jnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        terminated=jnp.asarray([False, True, False]),
        truncated=jnp.asarray([False, False, True]),
        final=SimpleNamespace(
            obs=jnp.asarray([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        ),
    )

    actual = bootstrap_observation(info)

    assert jnp.array_equal(
        actual, jnp.asarray([[1.0, 2.0], [30.0, 40.0], [50.0, 60.0]])
    )


def test_block_statistics_only_include_completed_episodes():
    completed = jnp.asarray([[[False, True], [True, False]]])
    out_info = SimpleNamespace(
        terminated=completed,
        truncated=jnp.zeros_like(completed),
        final=SimpleNamespace(
            stats=SimpleNamespace(
                reward=jnp.asarray([[[100.0, 2.0], [6.0, 100.0]]]),
                length=jnp.asarray([[[100, 4], [8, 100]]]),
            )
        ),
        policy_loss=jnp.asarray([1.0, 3.0]),
        value_loss=jnp.asarray([2.0, 4.0]),
        policy_entropy=jnp.asarray([0.5, 1.5]),
    )

    stats = summarize_block(out_info)

    assert int(stats["num_episodes"]) == 2
    assert float(stats["mean_return"]) == pytest.approx(4.0)
    assert float(stats["mean_episode_length"]) == pytest.approx(6.0)
    assert float(stats["policy_loss"]) == pytest.approx(2.0)
    assert float(stats["value_loss"]) == pytest.approx(3.0)
    assert float(stats["policy_entropy"]) == pytest.approx(1.0)
