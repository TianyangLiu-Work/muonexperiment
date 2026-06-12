from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.paper_stats import load_paper_stats
from e11_condition_geometry.reporting import fmt, markdown_table, ratio_ci, write_markdown


OUTPUT_PATH = Path("discussion/e11_result_for_discussion.md")


def read_optional_csv(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def safe_ratio_ci(row: pd.Series) -> str:
    if "ratio_ci95_low" not in row or "ratio_ci95_high" not in row:
        return "N/A"
    return ratio_ci(row)


def main() -> None:
    stats = load_paper_stats()
    cross_task = read_optional_csv("results/e11_cross_task_signature/cross_task_signature_summary.csv")
    mechanism = read_optional_csv("results/e11_mechanism_boundary/mechanism_boundary_map.csv")
    predictor = read_optional_csv("results/e11_boundary_predictor/boundary_predictor_summary.csv")
    predictor_uncertainty = read_optional_csv("results/e11_boundary_predictor/boundary_predictor_uncertainty.csv")

    cross_task_pass = cross_task[cross_task.get("passes_cross_task_screen", False).astype(bool)] if not cross_task.empty else pd.DataFrame()
    boundary_pass_rows = mechanism[mechanism.get("direction", "").isin(["Muon-favorable", "flat/polar-favorable", "own-update-favorable"])] if not mechanism.empty else pd.DataFrame()
    boundary_unfavorable_rows = mechanism[mechanism.get("direction", "").isin(["Adam/GD-favorable", "GD-spectrum-favorable"])] if not mechanism.empty else pd.DataFrame()

    predictor_best = None
    if not predictor.empty:
        pred_focus = predictor[
            (predictor.get("target") == "update_grad_inner_muon_higher")
            & (predictor.get("evaluation") == "leave_setting_out")
        ]
        if not pred_focus.empty:
            predictor_best = pred_focus.sort_values("balanced_accuracy", ascending=False).iloc[0].to_dict()

    predictor_uncertainty_best = None
    if not predictor_uncertainty.empty:
        p = predictor_uncertainty[predictor_uncertainty.get("target") == "update_grad_inner_muon_higher"]
        if not p.empty:
            predictor_uncertainty_best = p.sort_values(
                "mean_balanced_accuracy_chance_filled", ascending=False
            ).iloc[0].to_dict()

    claim_rows = pd.DataFrame(
        [
            {
                "claim": "主结论 1",
                "summary": "Muon 在匹配更新规模下显著改变更新矩阵光谱几何。",
                "evidence": (
                    f"nrUpdate Muon/Adam={fmt(stats.nr_update['geomean_ratio_muon_over_adam'])} "
                    f"{safe_ratio_ci(stats.nr_update)}；stUpdate={fmt(stats.st_update['geomean_ratio_muon_over_adam'])} "
                    f"{safe_ratio_ci(stats.st_update)}。"
                ),
                "caveat": "这是优化器内禀谱几何效应，不是完整泛化/最终损失性能声明。",
            },
            {
                "claim": "主结论 2",
                "summary": "在当前短步长范围里，one-step 减少由梯度-更新对齐度主导。",
                "evidence": (
                    f"Spearman(ΔL, <G,D>)={fmt(stats.calibration_all['spearman_delta_vs_first_order'])} "
                    f"[{fmt(stats.calibration_all['spearman_ci95_low'])}, {fmt(stats.calibration_all['spearman_ci95_high'])}]；"
                    f"within-factor-2={fmt(stats.calibration_all['within_factor_2'])}。"
                ),
                "caveat": "不对应于长程收敛或分类性能；需谨慎外推。",
            },
            {
                "claim": "主结论 3",
                "summary": "谱分配的“有用/无用”有明显的范数边界。",
                "evidence": (
                    f"flat/polar 对比GD：Frobenius 条件下比值={fmt(stats.spectral_fro['geomean_ratio'])} {safe_ratio_ci(stats.spectral_fro)}；"
                    f"operator 条件下比值={fmt(stats.spectral_op['geomean_ratio'])} {safe_ratio_ci(stats.spectral_op)}。"
                ),
                "caveat": "此结论来自局部一阶控制的 probe，对长程训练和不同任务仍有限。",
            },
            {
                "claim": "主结论 4",
                "summary": "优势是条件化而非全局：跨任务可翻转。",
                "evidence": (
                    f"机制表中 Muon/flat 有利={stats.boundary_counts.muon_flat_favorable}，不利={stats.boundary_counts.unfavorable}，"
                    f"混合/不确定={stats.boundary_counts.mixed}。"
                ),
                "caveat": "不宜写成“稳定更优”或“普适更快收敛”。",
            },
        ]
    )

    neural_rows = pd.DataFrame(
        [
            {
                "任务族": "Deep MNIST MLP",
                "nrUpdate (Muon/Adam)": f"{fmt(stats.deep_nr['geomean_ratio_muon_over_adam'])} {safe_ratio_ci(stats.deep_nr)}",
                "stUpdate (Muon/Adam)": f"{fmt(stats.deep_st['geomean_ratio_muon_over_adam'])} {safe_ratio_ci(stats.deep_st)}",
                "one-step对齐 (Muon/Adam)": f"{fmt(stats.deep_first['geomean_ratio_muon_over_adam'])} {safe_ratio_ci(stats.deep_first)}",
            },
            {
                "任务族": "MNIST Patch + 共享权重",
                "nrUpdate (Muon/Adam)": f"{fmt(stats.patch_nr['geomean_ratio_muon_over_adam'])} {safe_ratio_ci(stats.patch_nr)}",
                "stUpdate (Muon/Adam)": f"{fmt(stats.patch_st['geomean_ratio_muon_over_adam'])} {safe_ratio_ci(stats.patch_st)}",
                "one-step对齐 (Muon/Adam)": f"{fmt(stats.patch_first['geomean_ratio_muon_over_adam'])} {safe_ratio_ci(stats.patch_first)}",
            },
            {
                "任务族": "MNIST ConvNet",
                "nrUpdate (Muon/Adam)": f"{fmt(stats.conv_nr['geomean_ratio_muon_over_adam'])} {safe_ratio_ci(stats.conv_nr)}",
                "stUpdate (Muon/Adam)": f"{fmt(stats.conv_st['geomean_ratio_muon_over_adam'])} {safe_ratio_ci(stats.conv_st)}",
                "one-step对齐 (Muon/Adam)": f"{fmt(stats.conv_first['geomean_ratio_muon_over_adam'])} {safe_ratio_ci(stats.conv_first)}",
            },
        ]
    )

    cross_task_text = "目前仅有 update-spectrum 相关指标通过了三任务一致性筛选。"
    if not cross_task.empty:
        pass_count = int(cross_task["passes_cross_task_screen"].sum()) if "passes_cross_task_screen" in cross_task.columns else 0
        total = int(len(cross_task))
        cross_task_text = f"跨任务筛选通过率为 {pass_count}/{total}；通过项可见于下表。"

    text = f"""# E11 讨论稿（可发表版本草案）

## 研究问题

Muon 是否能作为 geometry-shaping 优化器在局部 step 上系统性改变更新矩阵的谱几何？该谱几何是否能解释 one-step 的损失下降方向与幅度？

## 核心结论（按证据优先级）

{markdown_table(claim_rows, ["claim", "summary", "evidence", "caveat"])}

## 神经网络 sanity check（避免“高秩即更好”误读）

{markdown_table(neural_rows, ["任务族", "nrUpdate (Muon/Adam)", "stUpdate (Muon/Adam)", "one-step对齐 (Muon/Adam)"])}

## 跨任务一致性与边界

- {cross_task_text}
- 边界表倾向性：有利 {stats.boundary_counts.muon_flat_favorable + stats.boundary_counts.own_update_positive_control} 条， 不利 {stats.boundary_counts.unfavorable} 条，混合/不确定 {stats.boundary_counts.mixed} 条。
- 预测器能力（保守）：
- Leave-setting-out 最佳特征集: {predictor_best['feature_set'] if predictor_best else 'N/A'}，均值平衡准确率: {fmt(predictor_best['balanced_accuracy']) if predictor_best is not None else 'N/A'}。
  - 机会补齐评估（不确定性控制后）: {fmt(predictor_uncertainty_best['mean_balanced_accuracy_chance_filled']) if predictor_uncertainty_best is not None else 'N/A'}，置信区间 [{fmt(predictor_uncertainty_best['balanced_accuracy_ci95_low']) if predictor_uncertainty_best is not None else 'N/A'}, {fmt(predictor_uncertainty_best['balanced_accuracy_ci95_high']) if predictor_uncertainty_best is not None else 'N/A'}]。
- 结论：目前只支撑“描述性边界”，尚未达到可对新任务稳定预测的强可迁移规则。

## 可直接放入正文的结论顺序（建议）

1. Muon 改变更新谱几何（nrUpdate / stUpdate）；  
2. one-step 主要通过 `<G,D>` 的局部对齐解释；  
3. 光谱分配在范数几何下有边界；  
4. 结果是条件化、跨任务可翻转的，不是普适最优性。

## 开放风险与审稿防线

- 不能把 update-spectrum 的提升直接写成最终优化优势。  
- 不能把神经网络小样本检验扩展为现代架构结论。  
- 长程、充分调优、非 toy 神经任务上的稳定性仍是下一阶段要补的“可泛化预测”证据。
"""

    write_markdown(OUTPUT_PATH, text)
    print(f"saved discussion outline to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
