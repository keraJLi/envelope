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


class ValueFunction(nnx.Module):
    def __init__(
        self, obs_space: envelope.Space, rngs: nnx.Rngs, layer_size: int = 256
    ):
        in_dim = np.prod(obs_space.shape)
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
        in_dim = np.prod(obs_space.shape)
        out_dim = np.prod(action_space.shape)
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
        return distrax.Independent(
            distrax.Clipped(
                distrax.Normal(loc=action_mean, scale=jnp.exp(action_log_std)),
                minimum=self.action_low,
                maximum=self.action_high,
            ),
            reinterpreted_batch_ndims=1,
        )


class DiscretePolicy(nnx.Module):
    def __init__(
        self,
        obs_space: envelope.Space,
        action_space: envelope.Space,
        rngs: nnx.Rngs,
        layer_size: int = 256,
    ):
        in_dim = jnp.prod(jnp.array(obs_space.shape))
        out_dim = action_space.n.item()
        self.layers = nnx.Sequential(
            ortho_linear(in_dim, layer_size, rngs),
            nnx.swish,
            ortho_linear(layer_size, layer_size, rngs),
            nnx.swish,
            ortho_linear(layer_size, layer_size, rngs),
            nnx.swish,
            ortho_linear(layer_size, out_dim, rngs, scale=0.01),
        )

    def __call__(self, obs: jax.Array) -> distrax.Distribution:
        action_logits = self.layers(obs)
        return distrax.Categorical(logits=action_logits)
