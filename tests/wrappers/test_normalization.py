import jax
import jax.numpy as jnp
import pytest

from envelope.wrappers.normalization import RunningMeanVar, update_rmv


def _init_rmv_from_example(x_like, dtype=jnp.float32, count=0) -> RunningMeanVar:
    """Initialize RunningMeanVar with zeros/ones matching leaf shapes (no batch dim)."""

    def zeros_no_batch(x):
        return jnp.zeros(x.shape[1:], dtype=dtype)

    def ones_no_batch(x):
        return jnp.ones(x.shape[1:], dtype=dtype)

    mean0 = jax.tree.map(zeros_no_batch, x_like)
    var0 = jax.tree.map(ones_no_batch, x_like)
    return RunningMeanVar(mean=mean0, var=var0, count=count)


@pytest.mark.parametrize(
    "dtype", [jnp.float32, jnp.float16, jnp.bfloat16], ids=["f32", "f16", "bf16"]
)
def test_update_rmv_scalar_batch_matches_sample_stats(dtype):
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (256, 1), dtype=dtype) * jnp.asarray(
        3.0, dtype
    ) + jnp.asarray(2.0, dtype)
    rmv0 = _init_rmv_from_example(x, dtype=dtype, count=0)
    rmv1 = update_rmv(rmv0, x)
    # Expected stats from the batch
    exp_mean = x.mean(axis=0)
    exp_var = x.var(axis=0)
    assert jnp.allclose(rmv1.mean, exp_mean, atol=1e-3, rtol=1e-3)
    assert jnp.allclose(rmv1.var, exp_var, atol=1e-3, rtol=1e-3)
    # Count should equal batch size
    assert rmv1.count == x.shape[0]
    # std property sanity
    assert jnp.allclose(
        jax.tree.map(lambda a, b: a, rmv1.std, jnp.sqrt(rmv1.var)), rmv1.std
    )


@pytest.mark.parametrize(
    "dtype,atol,rtol",
    [(jnp.float32, 1e-6, 1e-6), (jnp.float16, 5e-3, 5e-3)],
    ids=["f32", "f16"],
)
def test_update_rmv_two_batches_equals_single_concat(dtype, atol, rtol):
    k1, k2 = jax.random.split(jax.random.PRNGKey(1))
    x1 = jax.random.normal(k1, (64, 3), dtype=dtype) + jnp.asarray(1.0, dtype)
    x2 = jax.random.normal(k2, (96, 3), dtype=dtype) - jnp.asarray(2.0, dtype)
    x_all = jnp.concatenate([x1, x2], axis=0)

    rmv0 = _init_rmv_from_example(x1, dtype=dtype, count=0)
    rmv_a = update_rmv(rmv0, x1)
    rmv_b = update_rmv(rmv_a, x2)

    rmv_cat = update_rmv(rmv0, x_all)

    assert jnp.allclose(rmv_b.mean, rmv_cat.mean, atol=atol, rtol=rtol)
    assert jnp.allclose(rmv_b.var, rmv_cat.var, atol=atol, rtol=rtol)
    assert rmv_b.count == x_all.shape[0] == rmv_cat.count


def test_update_rmv_batch_size_independence():
    key = jax.random.PRNGKey(2)
    x = jax.random.uniform(key, (128, 4), dtype=jnp.float32) * 10.0
    rmv0 = _init_rmv_from_example(x, dtype=jnp.float32, count=0)

    # One-shot batched update
    rmv_batched = update_rmv(rmv0, x)

    # Per-sample scan: process each sample as a batch of size 1
    def scan_fn(rmv, xi):
        # Treat xi as a batch of size 1 to match update_rmv's batched interface
        rmv_next = update_rmv(rmv, xi[None, ...])
        return rmv_next, None

    rmv_seq, _ = jax.lax.scan(scan_fn, rmv0, x)

    assert jnp.allclose(rmv_seq.mean, rmv_batched.mean, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(rmv_seq.var, rmv_batched.var, atol=1e-6, rtol=1e-6)
    assert rmv_seq.count == x.shape[0] == rmv_batched.count


def test_update_rmv_pytree_leaves():
    key = jax.random.PRNGKey(3)
    x = {
        "a": jax.random.normal(key, (32, 2), dtype=jnp.float32),
        "b": jnp.stack(
            [jnp.arange(32, dtype=jnp.float32), jnp.arange(32, dtype=jnp.float32)],
            axis=1,
        ),  # (32, 2)
    }
    rmv0 = _init_rmv_from_example(x, dtype=jnp.float32, count=0)
    rmv1 = update_rmv(rmv0, x)
    # Shapes: means/vars have no leading batch dim
    assert rmv1.mean["a"].shape == (2,)
    assert rmv1.var["a"].shape == (2,)
    # Values: compare to direct batch stats
    exp_mean_a = x["a"].mean(axis=0)
    exp_var_a = x["a"].var(axis=0)
    exp_mean_b = x["b"].mean(axis=0)
    exp_var_b = x["b"].var(axis=0)
    assert jnp.allclose(rmv1.mean["a"], exp_mean_a, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(rmv1.var["a"], exp_var_a, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(rmv1.mean["b"], exp_mean_b, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(rmv1.var["b"], exp_var_b, atol=1e-6, rtol=1e-6)
    assert rmv1.count == 32


def test_update_rmv_jit_compatibility():
    key = jax.random.PRNGKey(4)
    x = jax.random.normal(key, (50, 5), dtype=jnp.float32)
    rmv0 = _init_rmv_from_example(x, dtype=jnp.float32, count=0)

    @jax.jit
    def jit_update(rmv, x_):
        return update_rmv(rmv, x_)

    rmv1 = jit_update(rmv0, x)
    exp_mean = x.mean(axis=0)
    exp_var = x.var(axis=0)
    assert jnp.allclose(rmv1.mean, exp_mean, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(rmv1.var, exp_var, atol=1e-6, rtol=1e-6)
    assert rmv1.count == x.shape[0]
