import distrax
import jax
import jax.numpy as jnp
import math
import numpy as np
from flax import nnx

import envelope


def ortho_linear(in_dim, out_dim, rngs, scale=jnp.sqrt(2)):
    return nnx.Linear(
        in_dim, out_dim, rngs=rngs, kernel_init=nnx.initializers.orthogonal(scale)
    )


def diagonal_gaussian_entropy(log_std: jax.Array) -> jax.Array:
    """Entropy of an independent diagonal Gaussian, reduced over its event axis."""
    entropy = 0.5 * (1.0 + jnp.log(2.0 * jnp.pi)) + log_std
    return jnp.sum(entropy, axis=-1)


class BoundedGaussian(distrax.Transformed):
    """Tanh-transformed diagonal Gaussian with an explicit base entropy estimate."""

    def __init__(
        self,
        loc: jax.Array,
        log_std: jax.Array,
        minimum: jax.Array,
        maximum: jax.Array,
    ) -> None:
        minimum = jnp.asarray(minimum, dtype=loc.dtype)
        maximum = jnp.asarray(maximum, dtype=loc.dtype)
        if not bool(jnp.all(jnp.isfinite(minimum) & jnp.isfinite(maximum))):
            raise ValueError("PPO Gaussian action bounds must be finite")
        if not bool(jnp.all(maximum > minimum)):
            raise ValueError(
                "PPO Gaussian action upper bounds must exceed lower bounds"
            )

        midpoint = (minimum + maximum) / 2
        half_range = (maximum - minimum) / 2
        base = distrax.Independent(
            distrax.Normal(loc=loc, scale=jnp.exp(log_std)),
            reinterpreted_batch_ndims=1,
        )
        scalar_transform = distrax.Chain(
            [distrax.ScalarAffine(shift=midpoint, scale=half_range), distrax.Tanh()]
        )
        super().__init__(base, distrax.Block(scalar_transform, ndims=1))
        self._log_std = log_std

    def entropy(self, seed: int | jax.Array | None = None) -> jax.Array:
        # The transformed entropy has no closed form. PPO conventionally regularizes
        # the correctly reduced base-Gaussian entropy instead.
        del seed
        return diagonal_gaussian_entropy(self._log_std)


class ValueFunction(nnx.Module):
    def __init__(
        self, obs_space: envelope.Space, rngs: nnx.Rngs, layer_size: int = 256
    ):
        in_dim = math.prod(obs_space.shape)
        self.layers = nnx.Sequential(
            ortho_linear(in_dim, layer_size, rngs),
            nnx.swish,
            ortho_linear(layer_size, layer_size, rngs),
            nnx.swish,
            ortho_linear(layer_size, layer_size, rngs),
            nnx.swish,
            ortho_linear(layer_size, 1, rngs, scale=1.0),
        )

    def __call__(self, obs: jax.Array) -> jax.Array:
        return self.layers(obs).squeeze(-1)


class GaussianPolicy(nnx.Module):
    def __init__(
        self,
        obs_space: envelope.Space,
        action_space: envelope.Space,
        rngs: nnx.Rngs,
        layer_size: int = 256,
    ):
        in_dim = math.prod(obs_space.shape)
        out_dim = math.prod(action_space.shape)
        self.action_low, self.action_high = action_space.low, action_space.high
        self.layers = nnx.Sequential(
            ortho_linear(in_dim, layer_size, rngs),
            nnx.swish,
            ortho_linear(layer_size, layer_size, rngs),
            nnx.swish,
            ortho_linear(layer_size, layer_size, rngs),
            nnx.swish,
        )
        self.action_mean = ortho_linear(layer_size, out_dim, rngs, scale=0.01)
        self.action_log_std = ortho_linear(layer_size, out_dim, rngs, scale=0.01)

    def __call__(self, obs: jax.Array) -> distrax.Distribution:
        features = self.layers(obs)
        action_mean = self.action_mean(features)
        action_log_std = self.action_log_std(features)
        return BoundedGaussian(
            loc=action_mean,
            log_std=action_log_std,
            minimum=self.action_low,
            maximum=self.action_high,
        )


class ReshapeCategoricalBijector(distrax.Bijector):
    """Maps flat categorical indices to multi-dimensional indices.

    Forward: flat index in [0, prod(n)-1] -> multi-index of shape n.shape
    Inverse: multi-index of shape n.shape -> flat index
    """

    def __init__(self, n):
        n_arr = np.asarray(n)
        self._dims = tuple(n_arr.flatten().tolist())
        self._out_shape = n_arr.shape
        super().__init__(
            event_ndims_in=0,
            event_ndims_out=len(self._out_shape),
            is_constant_jacobian=True,
            is_constant_log_det=True,
        )

    def forward_and_log_det(self, x):
        indices = jnp.unravel_index(x, self._dims)
        y = jnp.stack(indices, axis=-1).reshape(*x.shape, *self._out_shape)
        return y, jnp.zeros(x.shape, dtype=jnp.float32)

    def inverse_and_log_det(self, y):
        batch_shape = y.shape[: len(y.shape) - len(self._out_shape)]
        flat = y.reshape(*batch_shape, -1)
        indices = tuple(flat[..., i] for i in range(len(self._dims)))
        x = jnp.ravel_multi_index(indices, self._dims, mode="clip")
        return x, jnp.zeros(batch_shape, dtype=jnp.float32)


class DiscretePolicy(nnx.Module):
    def __init__(
        self,
        obs_space: envelope.Space,
        action_space: envelope.Space,
        rngs: nnx.Rngs,
        layer_size: int = 256,
    ):
        in_dim = math.prod(obs_space.shape)
        out_dim = math.prod(np.asarray(action_space.n).reshape(-1).tolist())
        self.n = nnx.static(jnp.asarray(action_space.n).tolist())
        self.layers = nnx.Sequential(
            ortho_linear(in_dim, layer_size, rngs),
            nnx.swish,
            ortho_linear(layer_size, layer_size, rngs),
            nnx.swish,
            ortho_linear(layer_size, out_dim, rngs, scale=0.01),
        )

    def __call__(self, obs: jax.Array) -> distrax.Distribution:
        action_logits = self.layers(obs)
        dist = distrax.Categorical(logits=action_logits)
        dist = distrax.Transformed(dist, ReshapeCategoricalBijector(self.n))
        return dist
