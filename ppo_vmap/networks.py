import distrax
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import envelope


def ortho_linear(in_dim, out_dim, rngs, scale=jnp.sqrt(2)):
    return nnx.Linear(
        in_dim, out_dim, rngs=rngs, kernel_init=nnx.initializers.orthogonal(scale)
    )


class Identity(nnx.Module):
    def __call__(self, x):
        return x


def get_activation(name: str):
    name = name.lower()
    if name == "relu":
        return nnx.relu
    if name == "tanh":
        return nnx.tanh
    if name == "swish":
        return nnx.swish
    if name == "mish":
        return nnx.mish
    raise ValueError(f"Unknown activation: {name}")


class ValueFunction(nnx.Module):
    def __init__(
        self,
        obs_space: envelope.Space,
        rngs: nnx.Rngs,
        layer_size: int = 256,
        activation: str = "swish",
        layer_norm: bool = False,
    ):
        in_dim = np.prod(obs_space.shape)
        act = get_activation(activation)
        self.layers = nnx.Sequential(
            ortho_linear(in_dim, layer_size, rngs),
            nnx.LayerNorm(layer_size, rngs=rngs) if layer_norm else Identity(),
            act,
            ortho_linear(layer_size, layer_size, rngs),
            nnx.LayerNorm(layer_size, rngs=rngs) if layer_norm else Identity(),
            act,
            ortho_linear(layer_size, layer_size, rngs),
            nnx.LayerNorm(layer_size, rngs=rngs) if layer_norm else Identity(),
            act,
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
        activation: str = "swish",
        layer_norm: bool = False,
    ):
        in_dim = np.prod(obs_space.shape)
        out_dim = np.prod(action_space.shape)
        self.action_low, self.action_high = action_space.low, action_space.high
        self.std_min, self.std_max = -5, 2

        act = get_activation(activation)
        self.layers = nnx.Sequential(
            ortho_linear(in_dim, layer_size, rngs),
            nnx.LayerNorm(layer_size, rngs=rngs) if layer_norm else Identity(),
            act,
            ortho_linear(layer_size, layer_size, rngs),
            nnx.LayerNorm(layer_size, rngs=rngs) if layer_norm else Identity(),
            act,
            ortho_linear(layer_size, layer_size, rngs),
            nnx.LayerNorm(layer_size, rngs=rngs) if layer_norm else Identity(),
            act,
        )
        self.action_mean = ortho_linear(layer_size, out_dim, rngs, scale=0.01)
        self.action_log_std = ortho_linear(layer_size, out_dim, rngs, scale=0.01)

    def __call__(self, obs: jax.Array) -> distrax.Distribution:
        features = self.layers(obs)
        action_mean = self.action_mean(features)
        action_log_std = self.action_log_std(features)
        action_log_std = jnp.clip(action_log_std, self.std_min, self.std_max)
        dist = distrax.Independent(
            distrax.Clipped(
                distrax.Normal(loc=action_mean, scale=jnp.exp(action_log_std)),
                minimum=self.action_low,
                maximum=self.action_high,
            ),
            reinterpreted_batch_ndims=1,
        )

        # Monkey-patch entropy to use the nested distribution for easy access
        # This is mathematically not correct since it ignores clipping, and semantically
        # not correct since it does not sum up the entropies of the independent
        # variables. But it's convenient and we take the mean anyways.
        dist.entropy = dist.distribution.distribution.entropy
        return dist


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
        activation: str = "swish",
        layer_norm: bool = False,
    ):
        in_dim = jnp.prod(jnp.array(obs_space.shape))
        out_dim = jnp.prod(jnp.asarray(action_space.n))
        self.n = nnx.static(jnp.asarray(action_space.n).tolist())
        act = get_activation(activation)
        self.layers = nnx.Sequential(
            ortho_linear(in_dim, layer_size, rngs),
            nnx.LayerNorm(layer_size, rngs=rngs) if layer_norm else Identity(),
            act,
            ortho_linear(layer_size, layer_size, rngs),
            nnx.LayerNorm(layer_size, rngs=rngs) if layer_norm else Identity(),
            act,
            ortho_linear(layer_size, out_dim, rngs, scale=0.01),
        )

    def __call__(self, obs: jax.Array) -> distrax.Distribution:
        action_logits = self.layers(obs)
        dist = distrax.Categorical(logits=action_logits)
        dist = distrax.Transformed(dist, ReshapeCategoricalBijector(self.n))
        return dist
