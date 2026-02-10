import dataclasses
import datetime
import json
import os
import time

import h5py
import jax
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx

import wandb


class Logger:
    """Logger that buffers metrics from vmapped/pmapped training runs.

    Since jax.debug.callback inside vmap delivers results in arbitrary order,
    the logger buffers metrics indexed by (step, run_idx). Once all num_runs
    entries for a given step are collected, it averages and emits the log.

    Metrics are stored in HDF5 format with appendable datasets:
        - steps: (num_logged_steps,)
        - {metric}: (num_logged_steps,) averaged across runs
        - {metric}_per_run: (num_logged_steps, num_runs) per-run values

    Local logging structure:
        runs/{env_name}/{timestamp}/
            config.json
            metrics.h5
            checkpoints/step_{global_step}/
    """

    def __init__(self, args):
        self.args = args
        self.num_runs = args.num_runs
        self.log_every = args.log_every
        self.start_time: float | None = None
        self._step_count = 0
        self._prev_flush_time: float | None = None
        self._prev_flush_step: int = 0
        self._sps: float = 0

        # Buffer: step -> {run_idx: metrics_dict}
        self.buffers: dict[int, dict[int, dict[str, float]]] = {}

        # Setup run directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{args.env_name}_{timestamp}"
        self.run_dir = os.path.abspath(os.path.join("runs", args.env_name, timestamp))
        os.makedirs(self.run_dir, exist_ok=True)

        # Save config
        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(dataclasses.asdict(args), f, indent=2)

        # Setup HDF5 metrics file
        self._metrics_path = os.path.join(self.run_dir, "metrics.h5")
        self._h5file = h5py.File(self._metrics_path, "w")

        # Setup wandb
        self._wandb_run = None
        if args.use_wandb:
            self._wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                config=dataclasses.asdict(args),
                name=self.run_name,
            )

        # Orbax checkpointer for saving model state
        self._checkpointer = ocp.StandardCheckpointer()

        print(f"Run directory: {self.run_dir}")

    def log(
        self,
        global_steps,
        run_idx,
        mean_return,
        mean_episode_length,
        policy_loss,
        value_loss,
        policy_entropy,
    ):
        """Called from jax.debug.callback inside vmap. Buffers until all runs report."""
        step = int(global_steps)
        run_idx = int(run_idx)

        metrics = {
            "mean_return": float(mean_return),
            "mean_episode_length": float(mean_episode_length),
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss),
            "policy_entropy": float(policy_entropy),
        }

        if step not in self.buffers:
            self.buffers[step] = {}
        self.buffers[step][run_idx] = metrics

        if len(self.buffers[step]) == self.num_runs:
            self._flush_step(step)

    def _flush_step(self, step: int):
        """Average metrics across all runs and conditionally emit."""
        all_metrics = self.buffers.pop(step)
        self._step_count += 1

        if self._step_count % self.log_every != 0:
            return

        # Update SPS from wall time between flushes
        now = time.time()
        if self._prev_flush_time is not None:
            dt = now - self._prev_flush_time
            if dt > 0:
                self._sps = (step - self._prev_flush_step) / dt
        self._prev_flush_time = now
        self._prev_flush_step = step

        keys = list(next(iter(all_metrics.values())).keys())

        # Collect per-run values and compute averages
        averaged = {}
        per_run = {}
        for key in keys:
            values = [all_metrics[rid][key] for rid in sorted(all_metrics)]
            per_run[key] = values
            averaged[key] = np.mean(values)

        # Append to HDF5
        self._h5_append("steps", step)
        self._h5_append("sps", self._sps)
        for key in keys:
            self._h5_append(key, averaged[key])
            self._h5_append(f"{key}_per_run", per_run[key])
        self._h5file.flush()

        # Print
        print(
            f"step: {step}, "
            f"mean_return: {averaged['mean_return']:.4f}, "
            f"sps: {self._sps:.0f}, "
            f"policy_loss: {averaged['policy_loss']:.4f}, "
            f"value_loss: {averaged['value_loss']:.4f}"
        )

        if self._wandb_run is not None:
            wandb_data = {**averaged, "time/sps": self._sps}
            if self.num_runs > 1:
                for key in keys:
                    wandb_data[f"{key}_std"] = np.std(per_run[key])
            wandb.log(wandb_data, step=step)

    def _h5_append(self, key: str, value):
        """Append a scalar or 1D array to a resizable HDF5 dataset."""
        f = self._h5file
        if key not in f:
            value_arr = np.asarray(value)
            if value_arr.ndim == 0:
                # Scalar: create 1D dataset
                f.create_dataset(key, shape=(0,), maxshape=(None,), dtype="f8")
            else:
                # Array (e.g. per-run values): create 2D dataset
                f.create_dataset(
                    key,
                    shape=(0, *value_arr.shape),
                    maxshape=(None, *value_arr.shape),
                    dtype="f8",
                )
        ds = f[key]
        ds.resize(ds.shape[0] + 1, axis=0)
        ds[-1] = value

    def save_checkpoint(self, global_step, train_states):
        """Save checkpoint of the first run."""
        # Split to get pure array state, take first run (device 0, run 0), move to CPU
        _, state = nnx.split(train_states)
        cpu = jax.devices("cpu")[0]
        state = jax.tree.map(lambda x: jax.device_put(x[0, 0], cpu), state)

        path = os.path.join(self.run_dir, "checkpoints", f"step_{global_step}")
        self._checkpointer.save(path, state)
        print(f"Started saving checkpoint (will finish asynchronously): {path}")

    def log_once(self, data: dict):
        """Log one-time data like compile/lower times."""
        for k, v in data.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
            self._h5file.attrs[k] = v
        self._h5file.flush()

        if self._wandb_run is not None:
            wandb.run.summary.update(data)

    def finish(self):
        self._checkpointer.wait_until_finished()
        self._h5file.close()
        if self._wandb_run is not None:
            wandb.finish()
