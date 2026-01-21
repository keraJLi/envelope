from functools import cached_property
from typing import NamedTuple

import jax
from jax import numpy as jnp

from envelope.struct import FrozenPyTreeNode
from envelope.typing import Array, PyTree


class MeanVarPair(NamedTuple):
    mean: Array
    var: Array


class RunningMeanVar(FrozenPyTreeNode):
    mean: PyTree
    var: PyTree
    count: int

    @cached_property
    def std(self) -> PyTree:
        return jax.tree.map(jnp.sqrt, self.var)


def update_rmv(rmv_state: RunningMeanVar, x: PyTree) -> RunningMeanVar:
    """
    Update running mean/variance with a new batch of observations x. We assume x is a
    PyTree of arrays, each with a leading batch dimension (aligned sizes).
    """
    global_count = rmv_state.count
    batch_count = jax.tree.leaves(x)[0].shape[0]
    tot_count = global_count + batch_count

    def _update_arr(mean: Array, var: Array, x_arr: Array) -> MeanVarPair:
        batch_mean = x_arr.mean(axis=0)
        batch_var = x_arr.var(axis=0)

        # Combine variances using parallel algorithm
        m_a = var * global_count
        m_b = batch_var * batch_count
        delta = batch_mean - mean
        m2 = m_a + m_b + (delta**2) * (global_count * batch_count) / tot_count

        new_mean = mean + delta * (batch_count / tot_count)
        new_var = m2 / tot_count
        return MeanVarPair(mean=new_mean, var=new_var)

    def is_pair(z):
        return isinstance(z, MeanVarPair)

    # jax.tree.map returns a PyTree whose leaves are MeanVarPairs
    mean_var_pairs = jax.tree.map(_update_arr, rmv_state.mean, rmv_state.var, x)
    new_mean = jax.tree.map(lambda mv: mv.mean, mean_var_pairs, is_leaf=is_pair)
    new_var = jax.tree.map(lambda mv: mv.var, mean_var_pairs, is_leaf=is_pair)
    return RunningMeanVar(mean=new_mean, var=new_var, count=tot_count)
