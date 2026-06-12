# E11 Notebook Report

This directory intentionally keeps only the current E11 read-only report.
The maintained implementation lives in `e11_condition_geometry/`, experiment
entry points live in `scripts/e11_*.py`, and generated evidence lives under
`results/`, `figures/`, and `discussion/`.

| Notebook | Purpose | Source Artifacts |
|---|---|---|
| `E11_condition_geometry_report.ipynb` | Executed report for the rebuilt condition-geometry pipeline. | `results/e11*`, `figures/e11*`, `discussion/e11*` |

Do not add exploratory notebook-only implementations here. New experiments
should be implemented as reusable code in `e11_condition_geometry/` plus a
small script under `scripts/`, then summarized in this report or in Markdown.
