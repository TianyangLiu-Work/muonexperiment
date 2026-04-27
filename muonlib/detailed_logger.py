"""
Detailed process-data logger for muonexperiment.
Logs per-step trajectory data (loss, gradient norm, SVD singular values, timing)
to JSONL files organized by experiment/run.
"""
import json
import time
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np


class DetailedLogger:
    """Per-run logger that accumulates step data and flushes to JSONL.

    Parameters
    ----------
    log_dir : str or Path
        Base directory for logs (e.g. PROJECT / "logs_v2")
    eid : str
        Experiment ID (e.g. "E01_detailed")
    algo : str
        Algorithm name
    run_params : dict
        Per-run params: d, seed, lr, etc. — used to generate unique run directory.
    log_interval : int
        Log every N steps (default 1 = every step).
    svd_interval : int
        Compute full SVD for logging every N steps (default 10).
        Set to 0 to disable SVD logging entirely.
    """

    def __init__(
        self,
        log_dir,
        eid: str,
        algo: str,
        run_params: dict,
        log_interval: int = 1,
        svd_interval: int = 10,
    ):
        self.eid = eid
        self.algo = algo
        self.run_params = dict(run_params)
        self.log_interval = log_interval
        self.svd_interval = svd_interval

        # Build unique run directory from ALL params (hash ensures no collisions)
        param_str = json.dumps(self.run_params, sort_keys=True, default=str)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        d = run_params.get("d", run_params.get("m", "?"))
        seed = run_params.get("seed", "?")
        run_dir_name = f"{algo}_d{d}_seed{seed}_{param_hash}"
        self.run_dir = Path(log_dir) / eid / run_dir_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Write params to run directory for reproducibility
        with open(self.run_dir / "params.json", "w") as f:
            json.dump(self.run_params, f, indent=2)

        # Use 'w' mode to avoid append collisions on rerun
        self.traj_file = self.run_dir / "trajectory.jsonl"
        self._buffer = []
        self._start_time = time.time()
        self._completed = False

    def log_step(
        self,
        step: int,
        loss: float,
        grad_norm: Optional[float] = None,
        sv: Optional[np.ndarray] = None,
        **extra,
    ):
        """Log a single training step.

        Parameters
        ----------
        step : int, 0-based iteration index.
        loss : float, current loss value.
        grad_norm : float, optional, Frobenius norm of gradient.
        sv : np.ndarray, optional, singular values of gradient.
        **extra : additional per-step data.
        """
        if step % self.log_interval != 0:
            return

        entry = {
            "step": step,
            "loss": float(loss),
            "elapsed_s": round(time.time() - self._start_time, 4),
        }
        if grad_norm is not None:
            entry["grad_norm"] = float(grad_norm)
        if sv is not None:
            entry["singular_values"] = [float(s) for s in sv]
        if extra:
            for k, v in extra.items():
                if isinstance(v, (np.ndarray, list)):
                    entry[k] = [float(x) if isinstance(x, (np.floating, float, int)) else x for x in v]
                elif isinstance(v, (np.floating, float, int)):
                    entry[k] = float(v)
                else:
                    entry[k] = v

        self._buffer.append(entry)

        if len(self._buffer) >= 100:
            self.flush()

    def flush(self):
        """Write buffered entries to disk. Deduplicates across calls."""
        if not self._buffer:
            return
        with open(self.traj_file, "a") as f:
            for entry in self._buffer:
                f.write(json.dumps(entry) + "\n")
        self._buffer.clear()

    def save_summary(self, summary: dict):
        """Save final run summary."""
        summary["wall_time_s"] = round(time.time() - self._start_time, 2)
        summary["params"] = self.run_params
        summary["completed"] = True
        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        self._completed = True

    def close(self):
        """Flush remaining data and mark completion."""
        self.flush()
        if not self._completed:
            with open(self.run_dir / "summary.json", "w") as f:
                json.dump({"completed": True, "wall_time_s": round(time.time() - self._start_time, 2)}, f)
            self._completed = True
