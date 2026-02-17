from functools import cached_property
from typing import Any, Callable, TypeAlias

import envpool
import envpool.python.xla_template as _xla_tpl
import gymnasium.spaces as gymnasium_spaces
import jax
import jax.numpy as jnp
from envpool.python.envpool import EnvPoolMixin
from envpool.python.gymnasium_envpool import GymnasiumEnvPoolMixin
from envpool.python.lax import XlaMixin
from jax.core import ShapedArray as _ShapedArray
from jax.interpreters import mlir as _mlir
from jax.lib import xla_client as _xla_client
from typing_extensions import override

import envelope.spaces as envelope_spaces
from envelope.environment import Environment, InfoContainer
from envelope.struct import FrozenPyTreeNode, field, static_field
from envelope.typing import Info, Key, PyTree

# -- Patch envpool for modern JAX (>=0.4.16) ------------------------------
# The PyPI wheel (0.8.4) uses deprecated/removed JAX symbols in
# envpool.python.xla_template.  Fixes exist in the fork
# pseudo-rnd-thoughts/envpool (branches main & jax-interpreter-mlir) but
# that repo has no binary wheel, so we patch the installed copy instead.
# We replace _make_xla_function entirely with an MLIR-based version.
_xla_tpl.ShapedArray = _ShapedArray
if not hasattr(_xla_client, "register_cpu_custom_call_target"):
    _xla_client.register_cpu_custom_call_target = (
        lambda name, cap: _xla_client.register_custom_call_target(
            name, cap, platform="cpu"
        )
    )
if not hasattr(_xla_tpl.xla, "backend_specific_translations"):
    from functools import partial as _partial

    from jax import core as _core
    from jax._src.interpreters.mlir import custom_call as _custom_call

    def _make_xla_function_mlir(
        obj: Any,
        handle: bytes,
        name: str,
        specs: tuple[tuple[Any], tuple[Any]],
        capsules: tuple[Any, Any],
    ) -> Any:
        """MLIR-based replacement for envpool's _make_xla_function."""
        in_specs, out_specs = specs
        in_specs = _xla_tpl._normalize_specs(in_specs)
        out_specs = _xla_tpl._normalize_specs(out_specs)
        cpu_capsule, gpu_capsule = capsules
        obj_name = f"{type(obj).__name__}_{id(obj)}_{name}"
        _xla_client.register_custom_call_target(
            f"{obj_name}_cpu".encode(), cpu_capsule, platform="cpu"
        )
        _xla_client.register_custom_call_target(
            f"{obj_name}_gpu".encode(), gpu_capsule, platform="gpu"
        )

        def abstract(*args: Any) -> Any:
            if len(out_specs) > 1:
                return tuple(_ShapedArray(*spec) for spec in out_specs)
            return _ShapedArray(*out_specs[0])

        def lowering(ctx: Any, *args: Any, platform: str = "cpu") -> Any:
            result_types = [_mlir.aval_to_ir_type(aval) for aval in ctx.avals_out]
            result_layouts = [
                tuple(range(len(s) - 1, -1, -1)) for s, _ in out_specs
            ]
            operand_layouts = [
                tuple(range(len(s) - 1, -1, -1)) for s, _ in in_specs
            ]
            op = _custom_call(
                f"{obj_name}_{platform}",
                result_types=result_types,
                operands=args,
                backend_config=handle,
                has_side_effect=True,
                operand_layouts=operand_layouts,
                result_layouts=result_layouts,
            )
            return op.results

        prim = _core.Primitive(f"{obj_name}")
        prim.multiple_results = len(out_specs) > 1
        prim.def_impl(_partial(_xla_tpl.xla.apply_primitive, prim))
        prim.def_abstract_eval(abstract)
        _mlir.register_lowering(
            prim, _partial(lowering, platform="cpu"), platform="cpu"
        )
        _mlir.register_lowering(
            prim, _partial(lowering, platform="gpu"), platform="gpu"
        )

        def call(*args: Any) -> Any:
            return prim.bind(*args)

        return call

    _xla_tpl._make_xla_function = _make_xla_function_mlir
# -- End envpool patch ---------------------------------------------------


class _EnvPoolEnvMeta(XlaMixin, GymnasiumEnvPoolMixin, EnvPoolMixin):
    pass


EnvPoolEnv: TypeAlias = _EnvPoolEnvMeta


class EnvPoolState(FrozenPyTreeNode):
    """State for EnvPool environments, holding the XLA handle and last terminal info."""

    handle: Any = field()
    last_final: Info = field()


class EnvPoolEnvelope(Environment):
    """Wrapper to convert an EnvPool environment to an envelope environment.

    EnvPool environments are accessed via the XLA (JAX) interface and are **not
    pure-functional**: environment state lives in a C++ backend outside JAX.
    `init` has a side effect (resetting the backend). The XLA handle is an opaque
    token that ensures correct ordering of operations within JAX's tracing, but
    does not itself hold environment state. The `key` argument to `init` is
    unused — EnvPool manages its own RNG via the `seed` parameter passed at
    construction time.

    EnvPool has a **built-in autoreset** that cannot be disabled. When an episode
    ends at step *t* (`terminated=True` or `truncated=True`), the environment
    returns the terminal observation and reward. On step *t+1* the submitted
    action is silently discarded and the C++ backend performs an internal reset,
    returning the first observation of the new episode with `reward=0` and
    `done=False`. This differs from `AutoResetWrapper` / `PooledInitVmapWrapper`,
    which immediately reset on the same step that `done=True` is emitted. **Do
    not wrap this adapter with `AutoResetWrapper`** — it would cause double
    resets and wasted steps.

    Like `AutoResetWrapper`, this adapter provides an `info.final` field. On
    `done=True`, `info.final` is a snapshot of the current (terminal) info. On
    `done=False`, `info.final` carries over the terminal info from the most
    recently completed episode (NaN-filled before any episode ends).

    Args:
        envpool_env (EnvPoolEnv): the EnvPool environment, created via
            `envpool.make_gymnasium`.
    """

    envpool_env: EnvPoolEnv = static_field()
    _xla_recv: Callable = static_field()
    _xla_step: Callable = static_field()
    _xla_handle0: Any = static_field()

    @classmethod
    def from_name(
        cls, env_name: str, env_kwargs: dict[str, Any] | None = None
    ) -> "EnvPoolEnvelope":
        """Create an `EnvPoolEnvelope` from a name and keyword arguments.

        `env_kwargs` are passed to `envpool.make_gymnasium`.
        """
        env_kwargs = env_kwargs or {}
        env = envpool.make_gymnasium(env_name, **env_kwargs)
        handle, recv, _send, step = env.xla()
        return EnvPoolEnvelope(
            envpool_env=env,
            _xla_recv=recv,
            _xla_step=step,
            _xla_handle0=handle,
        )

    @override
    def init(self, key: Key) -> tuple[EnvPoolState, Info]:
        # Reset the C++ backend (side effect — not traced by JAX)
        self.envpool_env.reset()
        # Receive initial observations via the XLA interface
        handle, (obs, rew, term, trunc, env_info) = self._xla_recv(self._xla_handle0)
        info = InfoContainer(obs=obs, reward=rew, terminated=term, truncated=trunc)
        info = info.update(**env_info)
        # NaN-filled placeholder for last_final (no episode has ended yet)
        last_final = jax.tree.map(lambda x: jnp.full_like(x, jnp.nan), info)
        state = EnvPoolState(handle=handle, last_final=last_final)
        return state, info.update(final=last_final)

    @override
    def step(self, state: EnvPoolState, action: PyTree) -> tuple[EnvPoolState, Info]:
        handle, (obs, rew, term, trunc, env_info) = self._xla_step(state.handle, action)
        info = InfoContainer(obs=obs, reward=rew, terminated=term, truncated=trunc)
        info = info.update(**env_info)

        done = term | trunc
        # Update last_final: on done, snapshot current info; otherwise carry over
        new_last_final = jax.tree.map(
            lambda curr, prev: jax.vmap(jnp.where)(done, curr, prev),
            info,
            state.last_final,
        )
        state = EnvPoolState(handle=handle, last_final=new_last_final)
        return state, info.update(final=new_last_final)

    @override
    @cached_property
    def action_space(self) -> envelope_spaces.Space:
        single_space = _convert_space(self.envpool_env.action_space)
        batch_size = self.envpool_env.config["batch_size"]
        return envelope_spaces.BatchedSpace(single_space, batch_size)

    @override
    @cached_property
    def observation_space(self) -> envelope_spaces.Space:
        single_space = _convert_space(self.envpool_env.observation_space)
        batch_size = self.envpool_env.config["batch_size"]
        return envelope_spaces.BatchedSpace(single_space, batch_size)


def _convert_space(gmn_space: gymnasium_spaces.Space) -> envelope_spaces.Space:
    if isinstance(gmn_space, gymnasium_spaces.Box):
        low = jnp.asarray(gmn_space.low, dtype=gmn_space.dtype)
        high = jnp.asarray(gmn_space.high, dtype=gmn_space.dtype)
        return envelope_spaces.Continuous(low=low, high=high)
    elif isinstance(gmn_space, gymnasium_spaces.Discrete):
        n = jnp.asarray(gmn_space.n, dtype=gmn_space.dtype)
        return envelope_spaces.Discrete(n=n)
    elif isinstance(gmn_space, gymnasium_spaces.MultiDiscrete):
        n = jnp.asarray(gmn_space.nvec, dtype=gmn_space.dtype)
        return envelope_spaces.Discrete(n=n)
    elif isinstance(gmn_space, gymnasium_spaces.Tuple):
        spaces = tuple(_convert_space(space) for space in gmn_space.spaces)
        return envelope_spaces.PyTreeSpace(spaces)
    elif isinstance(gmn_space, gymnasium_spaces.Dict):
        spaces = {k: _convert_space(space) for k, space in gmn_space.spaces.items()}
        return envelope_spaces.PyTreeSpace(spaces)
    raise ValueError(f"Unsupported gymnasium space type: {type(gmn_space)}")


if __name__ == "__main__":
    env = EnvPoolEnvelope.from_name(
        "Pong-v5", env_kwargs={"batch_size": 4, "num_envs": 4}
    )
    print(env.action_space)
    print(env.observation_space)
    print(env.observation_space)
