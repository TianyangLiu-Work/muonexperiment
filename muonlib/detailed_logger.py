"""
Detailed process-data logger for muonexperiment.
Logs per-step trajectory data to JSONL files organized by experiment/run.

v2.1: Fixed run directory naming to include all distinguishing params,
      overwrite mode by default, completion markers, summary.json support.
"""
import json
import time
import hashlib
from pathlib import Path
from typing import Optional, Dict, List

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
    run_id_params : dict
        All parameters that distinguish this run (d, seed, lr, L, kappa, ...).
        Used to build a unique, collision-free run directory name.
    log_interval : int
        Log every N steps (default 1 = every step).
    mode : str
        'w' = overwrite existing trajectory file (default, safe for new runs).
        'a' = append (use for resuming runs).
    """

    def __init__(
        self,
        log_dir,
        eid: str,
        algo: str,
        run_id_params: Dict,
        log_interval: int = 1,
        mode: str = "w",
    ):
        self.eid = eid
        self.algo = algo
        self.run_id_params = run_id_params
        self.log_interval = log_interval
        self._mode = mode

        # Build collision-free run directory name from ALL params
        param_str = "_".join(
            f"{k}{v}" for k, v in sorted(run_id_params.items())
            if k not in ("iters", "log_interval")
        )
        run_dir_name = f"{algo}__{param_str}"
        # Sanitize: replace path-unsafe chars
        run_dir_name = run_dir_name.replace("/", "-").replace(" ", "_")

        self.run_dir = Path(log_dir) / eid / run_dir_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.traj_file = self.run_dir / "trajectory.jsonl"
        self._buffer: List[Dict] = []
        self._start_time = time.time()
        self._completed = False
        self._traj_fh = None

        # If overwrite mode, truncate existing file
        if mode == "w":
            self.traj_file.write_text("")

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
            Loss value at the START of this step (pre-update state),
            to be consistent with grad_norm and sv from the same state.
        grad_norm : float, optional
            Frobenius norm of gradient.
        sv : np.ndarray, optional
            Full singular values of gradient (for Muon variants).
        **extra : dict
            Additional per-step data (layer_grad_norms, grad_max, X_norm, etc.)
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
        """Save final run summary and completion marker."""
        summary["wall_time_s"] = round(time.time() - self._start_time, 2)
        summary["run_params"] = self.run_id_params
        summary["algo"] = self.algo
        summary["eid"] = self.eid
        summary["completed"] = True
        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        self._completed = True

    def close(self):
        """Flush remaining data and write completion marker."""
        self.flush()
        if not self._completed:
            # Write a minimal completion marker
            minimal = {
                "wall_time_s": round(time.time() - self._start_time, 2),
                "run_params": self.run_id_params,
                "algo": self.algo,
                "eid": self.eid,
                "completed": False,
                "note": "Run did not call save_summary() — may be incomplete or crashed"
            }
            with open(self.run_dir / "summary.json", "w") as f:
                json.dump(minimal, f, indent=2)
