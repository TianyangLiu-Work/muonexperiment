from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


OUTPUT_PATH = Path("discussion/e11_activation_perturbation.md")
TABLE_PATH = Path("paper/specgrad_activation_paper/tables/e11_activation_perturbation.tex")


def _row(frame: pd.DataFrame, metric: str, family: str) -> pd.Series:
    rows = frame[
        (frame["metric"] == metric)
        & (frame["problem_family"] == family)
        & (frame["layer_group"] == "all_layers")
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one row for metric={metric}, family={family}, found {len(rows)}")
    return rows.iloc[0]


def _ratio(row: pd.Series) -> str:
    return (
        f"{row['geomean_ratio_muon_over_adam']:.4g} "
        f"[{row['ratio_ci95_low']:.4g}, {row['ratio_ci95_high']:.4g}]"
    )


def _interpret(row: pd.Series, lower_label: str, higher_label: str) -> str:
    if bool(row["ratio_ci95_below_one"]):
        return lower_label
    if bool(row["ratio_ci95_above_one"]):
        return higher_label
    return "inconclusive"


def write_latex_table(equal_update: pd.DataFrame) -> None:
    metrics = [
        ("relative_activation_delta_fro_norm", "Relative $\\|D_iA_i\\|_F$"),
        ("relative_activation_delta_op_norm", "Relative $\\|D_iA_i\\|_{op}$"),
        ("max_relative_sample_activation_delta", "Max per-sample $\\|D_ia_b\\|/\\|a_b\\|$"),
        ("mean_relative_sample_activation_delta", "Mean per-sample $\\|D_ia_b\\|/\\|a_b\\|$"),
    ]
    families = [
        ("All", "All activation-defined tasks"),
        ("MatrixFactorizationInput", "MF-with-input"),
        ("SmallMLPDigits", "Small MLP digits"),
    ]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Equal-update activation perturbation diagnostics. Ratios are Muon/Adam geomeans with 95\\% confidence intervals. Matrix Sensing is excluded because its diagnostic $A$ is a measurement-operator proxy, not a layer activation matrix.}",
        "\\label{tab:e11-activation-perturbation}",
        "\\begin{tabular}{llll}",
        "\\toprule",
        "Metric & Task group & Muon/Adam ratio & Interpretation \\\\",
        "\\midrule",
    ]
    for metric, label in metrics:
        for family, family_label in families:
            row = _row(equal_update, metric, family)
            interpretation = _interpret(row, "Muon lower", "Muon higher")
            lines.append(f"{label} & {family_label} & ${_ratio(row)}$ & {interpretation} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    TABLE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    raw = pd.read_csv("results/e11/activation_perturbation_summary.csv")
    equal = pd.read_csv("results/e11_equal_update/activation_perturbation_summary.csv")
    write_latex_table(equal)

    equal_all_fro = _row(equal, "relative_activation_delta_fro_norm", "All")
    equal_all_op = _row(equal, "relative_activation_delta_op_norm", "All")
    equal_all_max = _row(equal, "max_relative_sample_activation_delta", "All")
    equal_all_mean = _row(equal, "mean_relative_sample_activation_delta", "All")
    equal_mf_fro = _row(equal, "relative_activation_delta_fro_norm", "MatrixFactorizationInput")
    equal_mlp_fro = _row(equal, "relative_activation_delta_fro_norm", "SmallMLPDigits")
    raw_all_fro = _row(raw, "relative_activation_delta_fro_norm", "All")

    text = f"""# E11 Activation Perturbation Diagnostics

This note reports the first direct activation-perturbation diagnostic for the SpecGrad activation-geometry paper draft.

## Definition

For each matrix layer with pre-update parameter `W_i`, actual descent update `D_i = W_i^t - W_i^(t+1)`, and full diagnostic activation matrix `A_i`, the direct pre-activation perturbation is

```text
D_i A_i
```

For neural tasks, optimization still uses the configured mini-batch, while `A_i` is computed on the full sampled dataset for the problem instance. Matrix Sensing is excluded from this diagnostic because its `A` is a measurement-operator proxy rather than a layer activation matrix.

## Main Finding

At matched global update size, Muon has lower operator-norm and worst-case per-sample activation perturbation than Adam, but it does not uniformly reduce batch-Frobenius activation movement.

- Equal-update relative Frobenius perturbation, all activation-defined tasks: `{_ratio(equal_all_fro)}`.
- Equal-update relative operator perturbation, all activation-defined tasks: `{_ratio(equal_all_op)}`.
- Equal-update max per-sample relative perturbation, all activation-defined tasks: `{_ratio(equal_all_max)}`.
- Equal-update mean per-sample relative perturbation, all activation-defined tasks: `{_ratio(equal_all_mean)}`.

The family split matters:

- MF-with-input relative Frobenius perturbation: `{_ratio(equal_mf_fro)}`.
- Small MLP relative Frobenius perturbation: `{_ratio(equal_mlp_fro)}`.

Raw, unmatched runs show lower Muon activation movement overall (`{_ratio(raw_all_fro)}` for relative Frobenius perturbation), but that comparison is confounded by update-size differences. The equal-update control is the paper-facing evidence.

## Interpretation

This supports the paper's norm-geometry distinction. Muon-like polar updates can reduce operator and worst-case per-sample activation perturbation while increasing or decreasing batch-level Frobenius movement depending on the task family.

The result does **not** justify saying that Muon is generally more stable. A safer statement is:

> Muon changes activation perturbation geometry. Under matched update size, it lowers operator/per-sample worst-case perturbation in the current activation-defined tasks, while batch-Frobenius perturbation remains task dependent.

## Evidence

- [Raw activation perturbation summary](../results/e11/activation_perturbation_summary.csv)
- [Equal-update activation perturbation summary](../results/e11_equal_update/activation_perturbation_summary.csv)
- [Raw layer metrics](../results/e11/layer_metrics.csv)
- [Equal-update layer metrics](../results/e11_equal_update/layer_metrics.csv)
- [LaTeX table](../paper/specgrad_activation_paper/tables/e11_activation_perturbation.tex)
"""
    OUTPUT_PATH.write_text(text, encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")
    print(f"wrote {TABLE_PATH}")


if __name__ == "__main__":
    main()
