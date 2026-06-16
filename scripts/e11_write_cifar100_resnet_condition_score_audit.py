from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table
from e11_condition_geometry.statistics import ci95, corr_ci95


INPUT_LAYER_SUMMARY = Path("results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction/layer_summary.csv")
DEFAULT_OUTPUT_DIR = Path("results/e11_cifar100_resnet_condition_score_audit")
DEFAULT_FIGURE_DIR = Path("figures/e11_cifar100_resnet_condition_score_audit")
DEFAULT_DISCUSSION_PATH = Path("discussion/e11_cifar100_resnet_condition_score_audit.md")

RAW_SCORE_DEFINITIONS = {
    "source_observed_drift_positive_control": "log_observed",
    "early_layer_prior": "log_early_layer_prior",
    "gradient_nuclear_rank": "log_gradient_nuclear_rank",
    "alignment_ratio": "log_alignment_ratio",
    "step_size_ratio": "log_step_size_ratio",
    "scaled_jvp_ratio": "log_scaled_jvp_ratio",
    "inverse_scaled_jvp_ratio": "neg_log_scaled_jvp_ratio",
    "early_plus_gradient_rank": "early_plus_gradient_rank",
    "early_minus_scaled_jvp": "early_minus_scaled_jvp",
    "early_plus_rank_minus_scaled_jvp": "early_plus_rank_minus_scaled_jvp",
}

RESIDUAL_SCORE_DEFINITIONS = {
    "source_observed_residual_positive_control": "geomean_observed_tail_drift_sq_ratio_spectral_over_fro",
    "scaled_jvp_residual": "geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro",
    "unit_jvp_residual": "geomean_jvp_tail_drift_sq_ratio_spectral_over_fro",
    "gradient_nuclear_rank_residual": "mean_gradient_nuclear_rank",
    "alignment_ratio_residual": "mean_alignment_ratio_spectral_over_fro",
}


def score_family(score: str) -> str:
    if "positive_control" in score:
        return "positive_control"
    if score == "early_layer_prior":
        return "architecture_prior"
    return "condition_candidate"


def log_positive(values: pd.Series) -> pd.Series:
    return values.astype(float).clip(lower=1e-300).map(math.log)


def add_candidate_scores(layer_summary: pd.DataFrame) -> pd.DataFrame:
    frame = layer_summary.copy()
    frame["early_layer_prior"] = 1.0 / frame["layer_index"].astype(float).clip(lower=1.0)
    frame["log_early_layer_prior"] = log_positive(frame["early_layer_prior"])
    frame["log_observed"] = log_positive(frame["geomean_observed_tail_drift_sq_ratio_spectral_over_fro"])
    frame["log_gradient_nuclear_rank"] = log_positive(frame["mean_gradient_nuclear_rank"])
    frame["log_alignment_ratio"] = log_positive(frame["mean_alignment_ratio_spectral_over_fro"])
    frame["log_step_size_ratio"] = log_positive(frame["mean_step_size_ratio_spectral_over_fro"])
    frame["log_scaled_jvp_ratio"] = log_positive(
        frame["geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro"]
    )
    frame["neg_log_scaled_jvp_ratio"] = -frame["log_scaled_jvp_ratio"]
    frame["early_plus_gradient_rank"] = frame["log_early_layer_prior"] + frame["log_gradient_nuclear_rank"]
    frame["early_minus_scaled_jvp"] = frame["log_early_layer_prior"] - frame["log_scaled_jvp_ratio"]
    frame["early_plus_rank_minus_scaled_jvp"] = (
        frame["log_early_layer_prior"] + frame["log_gradient_nuclear_rank"] - frame["log_scaled_jvp_ratio"]
    )
    return frame


def linear_residual(frame: pd.DataFrame, column: str) -> tuple[float, float, pd.Series]:
    x = log_positive(frame["early_layer_prior"])
    y = log_positive(frame[column])
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    x_centered = x - x_mean
    denominator = float((x_centered * x_centered).sum())
    slope = 0.0 if denominator <= 0.0 else float((x_centered * (y - y_mean)).sum() / denominator)
    intercept = y_mean - slope * x_mean
    return intercept, slope, y - (intercept + slope * x)


def target_residual(frame: pd.DataFrame, column: str, intercept: float, slope: float) -> pd.Series:
    return log_positive(frame[column]) - (float(intercept) + float(slope) * log_positive(frame["early_layer_prior"]))


def raw_score_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    checkpoints = sorted(int(value) for value in frame["warmup_steps"].unique())
    for source_step in checkpoints:
        source = frame[frame["warmup_steps"].eq(source_step)].copy()
        for target_step in checkpoints:
            if source_step == target_step:
                continue
            target = frame[
                frame["warmup_steps"].eq(target_step)
            ][["parameter", "geomean_observed_tail_drift_sq_ratio_spectral_over_fro"]]
            joined = source.merge(target, on="parameter", suffixes=("_source", "_target"))
            target_score = log_positive(joined["geomean_observed_tail_drift_sq_ratio_spectral_over_fro_target"])
            target_top5 = set(
                joined.nlargest(
                    min(5, len(joined)),
                    "geomean_observed_tail_drift_sq_ratio_spectral_over_fro_target",
                )["parameter"]
            )
            for score_name, column in RAW_SCORE_DEFINITIONS.items():
                predictor = joined[column].astype(float)
                spearman, spearman_low, spearman_high, points = corr_ci95(
                    predictor,
                    target_score,
                    method="spearman",
                )
                pearson, pearson_low, pearson_high, _ = corr_ci95(
                    predictor,
                    target_score,
                    method="pearson",
                )
                predicted_top5 = set(joined.nlargest(min(5, len(joined)), column)["parameter"])
                rows.append(
                    {
                        "source_warmup_steps": int(source_step),
                        "target_warmup_steps": int(target_step),
                        "score": score_name,
                        "score_family": score_family(score_name),
                        "points": int(points),
                        "spearman_score_vs_target_observed": spearman,
                        "spearman_ci95_low": spearman_low,
                        "spearman_ci95_high": spearman_high,
                        "pearson_score_vs_target_observed": pearson,
                        "pearson_ci95_low": pearson_low,
                        "pearson_ci95_high": pearson_high,
                        "top5_risk_overlap_fraction": len(predicted_top5 & target_top5)
                        / max(len(target_top5), 1),
                    }
                )
    return pd.DataFrame(rows)


def residual_score_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    checkpoints = sorted(int(value) for value in frame["warmup_steps"].unique())
    for source_step in checkpoints:
        source = frame[frame["warmup_steps"].eq(source_step)].copy()
        obs_intercept, obs_slope, observed_source_residual = linear_residual(
            source,
            "geomean_observed_tail_drift_sq_ratio_spectral_over_fro",
        )
        source_residuals = pd.DataFrame(
            {
                "parameter": source["parameter"].to_numpy(),
                "source_observed_residual_positive_control": observed_source_residual.to_numpy(dtype=float),
            }
        )
        for score_name, column in RESIDUAL_SCORE_DEFINITIONS.items():
            if score_name == "source_observed_residual_positive_control":
                continue
            _intercept, _slope, residual = linear_residual(source, column)
            source_residuals[score_name] = residual.to_numpy(dtype=float)
        for target_step in checkpoints:
            if source_step == target_step:
                continue
            target = frame[frame["warmup_steps"].eq(target_step)].copy()
            target = source[["parameter"]].merge(target, on="parameter", how="inner")
            target_values = target_residual(
                target,
                "geomean_observed_tail_drift_sq_ratio_spectral_over_fro",
                obs_intercept,
                obs_slope,
            )
            joined = source_residuals.merge(
                pd.DataFrame({"parameter": target["parameter"], "target_residual": target_values}),
                on="parameter",
                how="inner",
            )
            target_top5 = set(joined.nlargest(min(5, len(joined)), "target_residual")["parameter"])
            for score_name in RESIDUAL_SCORE_DEFINITIONS:
                predictor = joined[score_name].astype(float)
                target_score = joined["target_residual"].astype(float)
                spearman, spearman_low, spearman_high, points = corr_ci95(
                    predictor,
                    target_score,
                    method="spearman",
                )
                pearson, pearson_low, pearson_high, _ = corr_ci95(
                    predictor,
                    target_score,
                    method="pearson",
                )
                predicted_top5 = set(joined.nlargest(min(5, len(joined)), score_name)["parameter"])
                rows.append(
                    {
                        "source_warmup_steps": int(source_step),
                        "target_warmup_steps": int(target_step),
                        "score": score_name,
                        "score_family": score_family(score_name),
                        "points": int(points),
                        "source_depth_fit_intercept": float(obs_intercept),
                        "source_depth_fit_slope": float(obs_slope),
                        "spearman_residual_score_vs_target_residual": spearman,
                        "spearman_ci95_low": spearman_low,
                        "spearman_ci95_high": spearman_high,
                        "pearson_residual_score_vs_target_residual": pearson,
                        "pearson_ci95_low": pearson_low,
                        "pearson_ci95_high": pearson_high,
                        "top5_residual_risk_overlap_fraction": len(predicted_top5 & target_top5)
                        / max(len(target_top5), 1),
                    }
                )
    return pd.DataFrame(rows)


def summarize_pairs(pairs: pd.DataFrame, *, residual: bool) -> pd.DataFrame:
    rows = []
    spearman_column = (
        "spearman_residual_score_vs_target_residual" if residual else "spearman_score_vs_target_observed"
    )
    pearson_column = "pearson_residual_score_vs_target_residual" if residual else "pearson_score_vs_target_observed"
    top5_column = "top5_residual_risk_overlap_fraction" if residual else "top5_risk_overlap_fraction"
    for score, group in pairs.groupby("score", observed=True, sort=False):
        spearman, spearman_low, spearman_high = ci95(group[spearman_column])
        pearson, pearson_low, pearson_high = ci95(group[pearson_column])
        top5, top5_low, top5_high = ci95(group[top5_column])
        rows.append(
            {
                "score": score,
                "score_family": str(group["score_family"].iloc[0]),
                "checkpoint_transfer_pairs": int(len(group)),
                "mean_spearman": spearman,
                "spearman_ci95_low": spearman_low,
                "spearman_ci95_high": spearman_high,
                "mean_pearson": pearson,
                "pearson_ci95_low": pearson_low,
                "pearson_ci95_high": pearson_high,
                "mean_top5_overlap_fraction": top5,
                "top5_overlap_ci95_low": top5_low,
                "top5_overlap_ci95_high": top5_high,
            }
        )
    return pd.DataFrame(rows)


def write_figure(raw_summary: pd.DataFrame, residual_summary: pd.DataFrame, figure_dir: Path) -> Path:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))
    for ax, summary, title in [
        (axes[0], raw_summary, "Raw source-only scores"),
        (axes[1], residual_summary, "Depth-adjusted residual scores"),
    ]:
        ordered = summary.sort_values("mean_spearman")
        y_positions = list(range(len(ordered)))
        colors = [
            "#CC79A7"
            if family == "positive_control"
            else "#0072B2"
            if family == "architecture_prior"
            else "#009E73"
            for family in ordered["score_family"]
        ]
        estimates = ordered["mean_spearman"].to_numpy(dtype=float)
        low = ordered["spearman_ci95_low"].to_numpy(dtype=float)
        high = ordered["spearman_ci95_high"].to_numpy(dtype=float)
        ax.barh(y_positions, estimates, color=colors, alpha=0.88)
        ax.errorbar(
            estimates,
            y_positions,
            xerr=[estimates - low, high - estimates],
            fmt="none",
            color="black",
            capsize=3,
            linewidth=1,
        )
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(ordered["score"].tolist(), fontsize=8)
        ax.set_xlabel("held-out checkpoint Spearman")
        ax.set_title(title)
    fig.suptitle("CIFAR-100-LT ResNet18 candidate condition-score audit")
    fig.tight_layout()
    path = figure_dir / "cifar100_resnet_condition_score_audit.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_discussion(
    raw_summary: pd.DataFrame,
    residual_summary: pd.DataFrame,
    figure_path: Path,
    output_dir: Path,
    discussion_path: Path,
) -> None:
    raw = raw_summary.set_index("score")
    residual = residual_summary.set_index("score")
    lines = [
        "# E11 CIFAR-100-LT ResNet18 Candidate Condition-Score Audit",
        "",
        "This generated audit evaluates simple source-only candidate scores on the",
        "tail-rich all-layer JVP checkpoint-transfer benchmark. It is a score",
        "discovery guardrail, not a new positive theory claim: the aim is to check",
        "whether obvious head-rank, JVP, or depth-composite scores already solve",
        "the held-out layer-risk ranking problem.",
        "",
        f"![CIFAR-100-LT ResNet18 candidate condition-score audit](../{figure_path.as_posix()})",
        "",
        "## Raw Transfer Summary",
        "",
        markdown_table(
            raw_summary,
            [
                "score",
                "score_family",
                "checkpoint_transfer_pairs",
                "mean_spearman",
                "spearman_ci95_low",
                "spearman_ci95_high",
                "mean_top5_overlap_fraction",
            ],
        ),
        "",
        "## Depth-Adjusted Residual Summary",
        "",
        markdown_table(
            residual_summary,
            [
                "score",
                "score_family",
                "checkpoint_transfer_pairs",
                "mean_spearman",
                "spearman_ci95_low",
                "spearman_ci95_high",
                "mean_top5_overlap_fraction",
            ],
        ),
        "",
        "## Readout",
        "",
        f"- Source observed-drift positive control Spearman: "
        f"{fmt(raw.loc['source_observed_drift_positive_control', 'mean_spearman'])} "
        f"[{fmt(raw.loc['source_observed_drift_positive_control', 'spearman_ci95_low'])}, "
        f"{fmt(raw.loc['source_observed_drift_positive_control', 'spearman_ci95_high'])}].",
        f"- Early-layer architecture prior Spearman: "
        f"{fmt(raw.loc['early_layer_prior', 'mean_spearman'])} "
        f"[{fmt(raw.loc['early_layer_prior', 'spearman_ci95_low'])}, "
        f"{fmt(raw.loc['early_layer_prior', 'spearman_ci95_high'])}].",
        f"- Best simple condition composite is early-minus-scaled-JVP with Spearman "
        f"{fmt(raw.loc['early_minus_scaled_jvp', 'mean_spearman'])} "
        f"[{fmt(raw.loc['early_minus_scaled_jvp', 'spearman_ci95_low'])}, "
        f"{fmt(raw.loc['early_minus_scaled_jvp', 'spearman_ci95_high'])}], below the early-layer prior.",
        f"- Scaled-JVP ratio itself has Spearman "
        f"{fmt(raw.loc['scaled_jvp_ratio', 'mean_spearman'])} "
        f"[{fmt(raw.loc['scaled_jvp_ratio', 'spearman_ci95_low'])}, "
        f"{fmt(raw.loc['scaled_jvp_ratio', 'spearman_ci95_high'])}].",
        f"- After source-fit early-layer residualization, observed residual Spearman is "
        f"{fmt(residual.loc['source_observed_residual_positive_control', 'mean_spearman'])} "
        f"[{fmt(residual.loc['source_observed_residual_positive_control', 'spearman_ci95_low'])}, "
        f"{fmt(residual.loc['source_observed_residual_positive_control', 'spearman_ci95_high'])}], "
        f"while scaled-JVP residual Spearman is "
        f"{fmt(residual.loc['scaled_jvp_residual', 'mean_spearman'])} "
        f"[{fmt(residual.loc['scaled_jvp_residual', 'spearman_ci95_low'])}, "
        f"{fmt(residual.loc['scaled_jvp_residual', 'spearman_ci95_high'])}].",
        "",
        "Interpretation: the obvious source-only score family does not produce a",
        "stronger measurable condition than the architecture-depth prior. This is",
        "useful negative evidence for a top-tier version because it prevents the",
        "paper from presenting a post-hoc composite score as a solved predictor.",
        "The next predictive-condition experiment needs a genuinely new",
        "downstream-aware score and a held-out architecture or dataset split.",
        "",
        "Artifacts:",
        f"- [raw_score_pairs.csv](../{(output_dir / 'raw_score_pairs.csv').as_posix()})",
        f"- [raw_score_summary.csv](../{(output_dir / 'raw_score_summary.csv').as_posix()})",
        f"- [residual_score_pairs.csv](../{(output_dir / 'residual_score_pairs.csv').as_posix()})",
        f"- [residual_score_summary.csv](../{(output_dir / 'residual_score_summary.csv').as_posix()})",
        f"- [config.json](../{(output_dir / 'config.json').as_posix()})",
    ]
    discussion_path.parent.mkdir(parents=True, exist_ok=True)
    discussion_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    output_dir = DEFAULT_OUTPUT_DIR
    figure_dir = DEFAULT_FIGURE_DIR
    discussion_path = DEFAULT_DISCUSSION_PATH
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_summary = add_candidate_scores(pd.read_csv(INPUT_LAYER_SUMMARY))
    raw_pairs = raw_score_pairs(layer_summary)
    residual_pairs = residual_score_pairs(layer_summary)
    raw_summary = summarize_pairs(raw_pairs, residual=False)
    residual_summary = summarize_pairs(residual_pairs, residual=True)
    raw_pairs.to_csv(output_dir / "raw_score_pairs.csv", index=False)
    raw_summary.to_csv(output_dir / "raw_score_summary.csv", index=False)
    residual_pairs.to_csv(output_dir / "residual_score_pairs.csv", index=False)
    residual_summary.to_csv(output_dir / "residual_score_summary.csv", index=False)
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "input_layer_summary": INPUT_LAYER_SUMMARY.as_posix(),
                "raw_score_definitions": RAW_SCORE_DEFINITIONS,
                "residual_score_definitions": RESIDUAL_SCORE_DEFINITIONS,
                "residual_adjustment": "source_checkpoint_log_observed_drift_on_log_early_layer_prior",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    figure_path = write_figure(raw_summary, residual_summary, figure_dir)
    write_discussion(raw_summary, residual_summary, figure_path, output_dir, discussion_path)
    print(f"saved CIFAR-100-LT ResNet18 condition-score audit to {output_dir}")
    print(f"figure: {figure_path}")
    print(f"discussion: {discussion_path}")
    print(raw_summary.to_string(index=False))
    print(residual_summary.to_string(index=False))


if __name__ == "__main__":
    main()
