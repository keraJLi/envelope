from functools import cached_property

import jax
from jax import numpy as jnp

from envelope.struct import FrozenPyTreeNode
from envelope.typing import Array, PyTree


class RunningMeanVar(FrozenPyTreeNode):
    mean: PyTree
    var: PyTree
    count: PyTree

    @cached_property
    def std(self) -> PyTree:
        return jax.tree.map(lambda var: jnp.sqrt(jnp.maximum(var, 0)), self.var)


def update_rmv(rmv_state: RunningMeanVar, x: PyTree) -> RunningMeanVar:
    """
    Update running mean/variance with new observation batches. Each leaf has its own
    leading sample dimension and count, so broadcast-reduced leaves may consume
    different effective batch sizes.
    """

    def _update(
        mean: Array, var: Array, global_count: Array, x_arr: Array
    ) -> RunningMeanVar:
        x_arr = jnp.asarray(x_arr, dtype=jnp.asarray(mean).dtype)
        global_count = jnp.asarray(global_count)
        batch_count = jnp.asarray(x_arr.shape[0], dtype=global_count.dtype)
        total_count = global_count + batch_count
        batch_mean = x_arr.mean(axis=0)
        batch_var = x_arr.var(axis=0)

        # Combine variances using parallel algorithm
        m_a = var * global_count
        m_b = batch_var * batch_count
        delta = batch_mean - mean
        m2 = m_a + m_b + (delta**2) * (global_count * batch_count) / total_count

        new_mean = mean + delta * (batch_count / total_count)
        new_var = jnp.maximum(m2 / total_count, 0)
        return RunningMeanVar(mean=new_mean, var=new_var, count=total_count)

    def is_result(z):
        return isinstance(z, RunningMeanVar)

    results = jax.tree.map(_update, rmv_state.mean, rmv_state.var, rmv_state.count, x)
    new_mean = jax.tree.map(lambda result: result.mean, results, is_leaf=is_result)
    new_var = jax.tree.map(lambda result: result.var, results, is_leaf=is_result)
    new_count = jax.tree.map(lambda result: result.count, results, is_leaf=is_result)
    return RunningMeanVar(mean=new_mean, var=new_var, count=new_count)
