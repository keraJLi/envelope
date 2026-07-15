# Experimental PPO example

This is a downstream integration example, not part of Envelope's supported API.

Install its repository-only dependencies with:

```bash
uv sync --group dev --group ppo
```

W&B is optional. Install `--group ppo-wandb` and pass `--use-wandb` to enable it.
The module can otherwise be imported and trained without W&B installed.
