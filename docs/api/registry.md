# Environment registry

Envelope keeps a global registry of environments, organized into suites. Its main
purpose is to make available suites and environments discoverable through
`registered_suites()` and `registered_environments()`. Envelope's built-in adapters
are registered by default.

An installed package can add a suite to `envelope.create` with an entry point in the
`envelope.environments` group. The entry-point name becomes the suite prefix, and its
target is a callable with the same arguments as `from_name`. Calling
`envelope.create("suite_name::environment_name", env_kwargs=..., **kwargs)` passes the
local ID `"environment_name"`, `env_kwargs`, and `kwargs` to that callable.

A classmethod on an `Environment` subclass can optionally expose the environments it
knows about:

```toml
[project.entry-points."envelope.environments"]
suite_name = "package_name:EnvironmentClass.from_name"
```

```python
from envelope import Environment


class EnvironmentClass(Environment):
    @classmethod
    def from_name(cls, env_name, env_kwargs=None, **kwargs) -> Environment:
        # Construct and return an Envelope Environment.
        ...

    @classmethod
    def registered_names(cls):
        # Return local IDs; Envelope adds the suite prefix.
        return ("environment_a", "environment_b")
```

A plain function is also valid, but does not enumerate environment names:

```toml
[project.entry-points."envelope.environments"]
suite_name = "package_name:create_environment"
```

```python
def create_environment(env_name, env_kwargs=None, **kwargs) -> Environment:
    # Construct and return an Envelope Environment.
    ...
```

The registry API returns sorted tuples:

```python
import envelope

envelope.registered_suites()
envelope.registered_environments("suite_name")
# ("suite_name::environment_a", "suite_name::environment_b")
```

Environment enumeration is best effort. A suite can still support `create` when it
cannot list names or does not implement `registered_names`. Built-in suites take
precedence over entry points with the same name. If multiple installed packages claim
the same otherwise-unknown suite, creation fails; enumeration warns and returns an
empty tuple.

## API Reference

::: envelope.registry.registered_suites

::: envelope.registry.registered_environments
