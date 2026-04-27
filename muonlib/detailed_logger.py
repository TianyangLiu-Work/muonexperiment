"""
Detailed process-data logger for muonexperiment.
Logs per-step trajectory data (loss, gradient norm, SVD singular values, timing)
to JSONL files organized by experiment/run.
"""
import json
import time
import os
from pathlib import Path
from typing import Optional

import numpy as np


class DetailedLogger:
    """Per-run logger that accumulates step data and flushes to JSONL.

    Logs every LOG_INTERVAL steps to JSONL. Final summary saved as summary.json.

    Parameters
    ----------
    log_dir : str or Path
        Base directory for logs (e.g. PROJECT / "logs_v2")
    eid : str
        Experiment ID (e.g. "E01")
    algo : str
        Algorithm name
    run_params : dict
        Per-run params: d, seed, lr, etc. — used to name the run directory.
    log_interval : int
        Log every N steps (default 1 = every step).
    """

    def __init__(
        self,
        log_dir,
        eid: str,
        algo: str,
        run_params: dict,
        log_interval: int = 1,
    ):
        self.eid = eid
        self.algo = algo
        self.run_params = run_params
        self.log_interval = log_interval

        # Build run directory: logs_v2/E01/Muon-Exact_d50_seed0/
        d = run_params.get("d", run_params.get("m", "?"))
        seed = run_params.get("seed", "?")
        run_dir_name = f"{algo}_d{d}_seed{seed}"
        self.run_dir = Path(log_dir) / eid / run_dir_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.traj_file = self.run_dir / "trajectory.jsonl"
        self._buffer = []
        self._start_time = time.time()

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
        step : int
            Iteration index (0-based).
        loss : float
            Current loss value.
        grad_norm : float, optional
            Frobenius norm of gradient.
        sv : np.ndarray, optional
            Singular values of gradient (for Muon variants). Top-20 stored.
        **extra : dict
            Additional per-step data (e.g. eigen_ratio, X_norm, etc.)
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
            # Store full singular values
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

        # Flush every 100 entries to avoid memory buildup
        if len(self._buffer) >= 100:
            self.flush()

    def flush(self):
        """Write buffered entries to disk."""
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
        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    def close(self):
        """Flush remaining data."""
        self.flush()
