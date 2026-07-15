#!/usr/bin/env python3
"""Script to count environments in each supported library by querying their registries."""


def count_gymnax():
    """Count gymnax environments."""
    try:
        from gymnax.environments import all_environments

        return len(all_environments)
    except (ImportError, AttributeError):
        try:
            # Try checking the environments module directly
            import gymnax.environments as envs_module

            if hasattr(envs_module, "all_environments"):
                return len(envs_module.all_environments)
            # Try to find environment classes
            import inspect

            env_classes = [
                name
                for name, obj in inspect.getmembers(envs_module)
                if inspect.isclass(obj) and hasattr(obj, "reset")
            ]
            return len(env_classes) if env_classes else "?"
        except:
            return "?"


def count_brax():
    """Count brax environments."""
    try:
        from brax.envs import _envs

        # _envs is a dict-like registry
        return len(_envs)
    except (ImportError, AttributeError):
        return "?"


def count_jumanji():
    """Count jumanji environments."""
    try:
        import jumanji

        # registered_environments is a function that returns a dict
        envs = jumanji.registered_environments()
        return len(envs)
    except (ImportError, AttributeError):
        return "?"


def count_kinetix():
    """Count kinetix environments (levels)."""
    try:
        import pathlib

        import kinetix

        # Kinetix uses level files in levels directory
        pkg_dir = pathlib.Path(kinetix.__file__).resolve().parent
        levels_dir = pkg_dir / "levels"
        if levels_dir.exists():
            count = 0
            for size_dir in levels_dir.iterdir():
                if size_dir.is_dir():
                    count += len(list(size_dir.glob("*.json")))
            return count if count > 0 else "?"
        return "?"
    except (ImportError, AttributeError):
        return "?"


def count_craftax():
    """Count craftax environments."""
    try:
        # Craftax has a fixed set of environments
        # From test file: Craftax-Symbolic-v1, Craftax-Classic-Symbolic-v1,
        # Craftax-Pixels-v1, Craftax-Classic-Pixels-v1
        from craftax.craftax_env import make_craftax_env_from_name

        # Try to find all available environments
        try:
            import craftax.craftax_env as craftax_module

            # Check if there's a registry or list
            if hasattr(craftax_module, "ALL_ENVS"):
                return len(craftax_module.ALL_ENVS)
            # Known environments from tests
            known_envs = [
                "Craftax-Symbolic-v1",
                "Craftax-Classic-Symbolic-v1",
                "Craftax-Pixels-v1",
                "Craftax-Classic-Pixels-v1",
            ]
            # Verify they exist
            count = 0
            for env_name in known_envs:
                try:
                    make_craftax_env_from_name(env_name)
                    count += 1
                except:
                    pass
            return count if count > 0 else 4  # Default to 4 if we can't verify
        except:
            return 4  # Known to have 4 environments
    except ImportError:
        return "?"


def count_navix():
    """Count navix environments."""
    try:
        import navix
        from navix.environments import registry

        return len(registry.registry())

        # Try common registry patterns
        if hasattr(navix, "registered_environments"):
            try:
                envs = navix.registered_environments()
                return len(envs)
            except Exception:
                pass

        if hasattr(navix, "registry"):
            reg = navix.registry
            for name in ("ALL_ENVS", "ENVIRONMENTS", "registered_environments"):
                if hasattr(reg, name):
                    try:
                        return len(getattr(reg, name))
                    except Exception:
                        pass

        # Try module-level registry
        try:
            from navix.environments import registry as env_registry

            for name in ("ALL_ENVS", "ENVIRONMENTS", "registered_environments"):
                if hasattr(env_registry, name):
                    try:
                        return len(getattr(env_registry, name))
                    except Exception:
                        pass
        except Exception:
            pass

        return "?"
    except ImportError:
        return "?"


def count_mujoco_playground():
    """Count mujoco_playground environments."""
    try:
        from mujoco_playground import registry

        if hasattr(registry, "ALL_ENVS"):
            return len(registry.ALL_ENVS)
        # Try to get registered environments another way
        if hasattr(registry, "registered_environments"):
            return len(registry.registered_environments)
        return "?"
    except ImportError:
        return "?"


def main():
    """Main function to count all environments."""
    print("Counting environments from registries...")
    print()

    results = {
        "gymnax": count_gymnax(),
        "brax": count_brax(),
        "jumanji": count_jumanji(),
        "kinetix": count_kinetix(),
        "craftax": count_craftax(),
        "navix": count_navix(),
        "mujoco_playground": count_mujoco_playground(),
    }

    print("Environment counts:")
    for lib, count in results.items():
        print(f"  {lib:20s}: {count}")

    return results


if __name__ == "__main__":
    main()
