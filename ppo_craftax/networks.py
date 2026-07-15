import distrax
import jax
import jax.numpy as jnp
from flax import nnx

import envelope


def ortho_linear(in_dim, out_dim, rngs, scale=jnp.sqrt(2)):
    return nnx.Linear(
        in_dim, out_dim, rngs=rngs, kernel_init=nnx.initializers.orthogonal(scale)
    )


class ActorCritic(nnx.Module):
    """Combined actor-critic matching craftax-baselines ActorCritic architecture.

    3-layer MLP with tanh activation, orthogonal init, separate actor/critic heads.
    """

    def __init__(
        self,
        obs_space: envelope.Space,
        action_space: envelope.Space,
        rngs: nnx.Rngs,
        layer_size: int = 512,
    ):
        in_dim = jnp.prod(jnp.array(obs_space.shape))
        out_dim = action_space.n.item()

        # Actor: 3 hidden layers + output
        self.actor = nnx.Sequential(
            ortho_linear(in_dim, layer_size, rngs),
            nnx.tanh,
            ortho_linear(layer_size, layer_size, rngs),
            nnx.tanh,
            ortho_linear(layer_size, layer_size, rngs),
            nnx.tanh,
            ortho_linear(layer_size, out_dim, rngs, scale=0.01),
        )

        # Critic: 3 hidden layers + output
        self.critic = nnx.Sequential(
            ortho_linear(in_dim, layer_size, rngs),
            nnx.tanh,
            ortho_linear(layer_size, layer_size, rngs),
            nnx.tanh,
            ortho_linear(layer_size, layer_size, rngs),
            nnx.tanh,
            ortho_linear(layer_size, 1, rngs, scale=1.0),
        )

    def __call__(self, obs: jax.Array) -> tuple[distrax.Distribution, jax.Array]:
        action_logits = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        return distrax.Categorical(logits=action_logits), value
