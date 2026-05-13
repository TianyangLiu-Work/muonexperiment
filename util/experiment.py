from __future__ import annotations

import collections.abc
import typing

import joblib
import pandas as pd
import tqdm


RESULT_COLUMNS = (
    "step",
    "initial_loss",
    "loss",
    "grad_norm",
    "best_loss",
    "early_stop_wait",
    "elapsed_s",
    "stop_reason",
)


def run_experiments(
    runs: pd.DataFrame,
    single_run: collections.abc.Callable[[dict[str, typing.Any]], pd.DataFrame],
    *,
    num_workers: int = 1,
    backend: str = "loky",
    algo_order: collections.abc.Sequence[str] | None = None,
    sort_columns: collections.abc.Sequence[str] = ("algo", "d", "seed", "step"),
    desc: str | None = None,
) -> pd.DataFrame:
    run_rows = (
        runs.drop(columns=list(RESULT_COLUMNS), errors="ignore")
        .drop_duplicates("run_id")
        .to_dict("records")
    )
    if not run_rows:
        return runs.iloc[0:0].copy()

    workers = max(1, int(num_workers or 1))

    if workers == 1:
        chunks = (single_run(run) for run in run_rows)
    else:
        chunks = joblib.Parallel(n_jobs=workers, backend=backend, return_as="generator_unordered")(
            joblib.delayed(single_run)(run) for run in run_rows
        )

    result = pd.concat(
        tqdm.tqdm(
            chunks,
            total=len(run_rows),
            desc=desc or f"runs ({workers} joblib)",
            unit="run",
            dynamic_ncols=True,
            mininterval=1.0,
        ),
        ignore_index=True,
    )
    if "algo" in result and algo_order is not None:
        result["algo"] = pd.Categorical(result["algo"], categories=list(algo_order), ordered=True)

    present_sort_columns = [column for column in sort_columns if column in result.columns]
    if present_sort_columns:
        result = result.sort_values(present_sort_columns).reset_index(drop=True)
    if "algo" in result:
        result["algo"] = result["algo"].astype(str)
    return result
