from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.paper_stats import load_paper_stats
from e11_condition_geometry.reporting import fmt, markdown_table, write_markdown


OUTPUT_PATH = Path("discussion/e11_result_for_discussion.md")


def ci(row: pd.Series, low: str, high: str) -> str:
    return f"[{fmt(row[low])}, {fmt(row[high])}]"


def main() -> None:
    stats = load_paper_stats()

    claim_rows = pd.DataFrame(
        [
            {
                "讨论点": "G 和 A 的当前定义",
                "发现": "主文中严格使用 head gradient 与 downstream tail sensitivity；旧 MF/condition-score 诊断只作背景。",
                "证据": "paper main.tex 的 matched-head-gain / B_T D A_T 定义，以及当前四组 head-to-tail CSV。",
                "解释": "这样可以避免把旧 A_i proxy 和当前 tail activation product 混在一起。",
            },
            {
                "讨论点": "Synthetic boundary",
                "发现": (
                    f"正例 squared drift ratio={fmt(stats.synthetic_positive['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])}，"
                    f"反例 squared drift ratio={fmt(stats.synthetic_negative['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])}。"
                ),
                "证据": "results/e11_head_tail_interference/pair_summary.csv",
                "解释": "nrank(G_H) 与 ssrank(B_T,A_T) 的不等式至少有可 falsify 的符号含义。",
            },
            {
                "讨论点": "One-step long-tail drift",
                "发现": (
                    f"20 seeds 下 spectral/Frobenius squared tail-example logit drift ratio="
                    f"{fmt(stats.long_tail_one_step['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"{ci(stats.long_tail_one_step, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')}。"
                ),
                "证据": "results/e11_long_tail_one_step/pair_summary.csv",
                "解释": "matched head gain 下 spectral/polar 对 held-out tail examples 上的 logits 扰动更小。",
            },
            {
                "讨论点": "Muon-style compatibility",
                "发现": (
                    f"polar(M_t) squared drift ratio={fmt(stats.muon_bridge_polar_momentum['geomean_tail_output_drift_sq_ratio_vs_fro'])} "
                    f"{ci(stats.muon_bridge_polar_momentum, 'tail_output_drift_sq_ratio_vs_fro_ci95_low', 'tail_output_drift_sq_ratio_vs_fro_ci95_high')}；"
                    f"NS(M_t) squared drift ratio={fmt(stats.muon_bridge_ns_momentum['geomean_tail_output_drift_sq_ratio_vs_fro'])} "
                    f"{ci(stats.muon_bridge_ns_momentum, 'tail_output_drift_sq_ratio_vs_fro_ci95_low', 'tail_output_drift_sq_ratio_vs_fro_ci95_high')}。"
                ),
                "证据": "results/e11_long_tail_muon_bridge/pair_summary.csv",
                "解释": "momentum polar 是 selected-state compatibility check；finite Newton-Schulz 近似较弱，不能直接推出完整 Muon training。",
            },
            {
                "讨论点": "Practical Muon trajectory compatibility",
                "发现": (
                    f"120 个 state-step comparisons 下 polar(M_t) squared drift ratio="
                    f"{fmt(stats.practical_muon_bridge_polar_momentum['geomean_tail_output_drift_sq_ratio_vs_fro'])} "
                    f"{ci(stats.practical_muon_bridge_polar_momentum, 'tail_output_drift_sq_ratio_vs_fro_ci95_low', 'tail_output_drift_sq_ratio_vs_fro_ci95_high')}；"
                    f"NS(M_t) squared drift ratio={fmt(stats.practical_muon_bridge_ns_momentum['geomean_tail_output_drift_sq_ratio_vs_fro'])} "
                    f"{ci(stats.practical_muon_bridge_ns_momentum, 'tail_output_drift_sq_ratio_vs_fro_ci95_low', 'tail_output_drift_sq_ratio_vs_fro_ci95_high')}。"
                ),
                "证据": "results/e11_long_tail_practical_muon_bridge/summary.csv",
                "解释": "短 trajectory 上 Muon-style directions 与局部 polar mechanism 兼容，但仍不是 final performance benchmark。",
            },
            {
                "讨论点": "性能 caveat",
                "发现": (
                    f"one-step tail loss diff={fmt(stats.long_tail_one_step['mean_tail_loss_increase_diff_spectral_minus_fro'])} "
                    f"{ci(stats.long_tail_one_step, 'tail_loss_increase_diff_ci95_low', 'tail_loss_increase_diff_ci95_high')}。"
                ),
                "证据": "results/e11_long_tail_one_step/pair_summary.csv",
                "解释": "当前结果支持 function drift claim，不支持 tail accuracy / final performance claim。",
            },
            {
                "讨论点": "8-step forgetting",
                "发现": (
                    f"final squared drift ratio={fmt(stats.long_tail_forgetting['geomean_final_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"{ci(stats.long_tail_forgetting, 'final_tail_output_drift_sq_ratio_ci95_low', 'final_tail_output_drift_sq_ratio_ci95_high')}；"
                    f"area ratio={fmt(stats.long_tail_forgetting['geomean_tail_output_drift_area_ratio_spectral_over_fro'])} "
                    f"{ci(stats.long_tail_forgetting, 'tail_output_drift_area_ratio_ci95_low', 'tail_output_drift_area_ratio_ci95_high')}。"
                ),
                "证据": "results/e11_long_tail_forgetting/summary.csv",
                "解释": "drift reduction 不只是单步现象，但仍然只是短程诊断。",
            },
            {
                "讨论点": "Layerwise mechanism",
                "发现": (
                    f"layer 1 unit/scaled/observed={fmt(stats.layer_one['geomean_jvp_tail_drift_sq_ratio_spectral_over_fro'])}/"
                    f"{fmt(stats.layer_one['geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro'])}/"
                    f"{fmt(stats.layer_one['geomean_observed_tail_drift_sq_ratio_spectral_over_fro'])}；"
                    f"layer 2={fmt(stats.layer_two['geomean_jvp_tail_drift_sq_ratio_spectral_over_fro'])}/"
                    f"{fmt(stats.layer_two['geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro'])}/"
                    f"{fmt(stats.layer_two['geomean_observed_tail_drift_sq_ratio_spectral_over_fro'])}。"
                ),
                "证据": "results/e11_long_tail_layerwise/summary.csv",
                "解释": "机制应写成 matched-head-gain scaling efficiency，而不是“spectral direction 本身更不扰动 tail”。",
            },
        ]
    )

    text = f"""# result for discussion

## 当前讨论主线

当前 paper 应写成 **head-to-tail interference mechanism paper**：在长尾小批量训练中，head-only update 可能在 tail 样本缺席时扰动这些 tail examples 上的 logits；我们研究 idealized spectral/polar direction 是否能在 matched head gain 下减少这种扰动。

旧的 Muon/Adam condition-geometry 结果只作为 guardrail：它提醒我们不能把 spectral/rank geometry 直接写成优化器全局更优、最终 tail accuracy 更好，或完整 Muon 机制已经被解释。

## 会议讨论点

{markdown_table(claim_rows, ["讨论点", "发现", "证据", "解释"])}

## 建议会议结论

1. 主文只讲 head-to-tail function drift，不讲 broad optimizer leaderboard。
2. 定理和图都围绕 matched-head-gain protocol 组织。
3. Muon-style compatibility 已从 fixed checkpoint 推进到 short practical trajectory；可以说 selected-state 兼容性证据更多，但仍不能说完整 practical Muon training 或 final tail accuracy。
4. 如果要更强 empirical paper，下一步不是再画旧 geometry 图，而是加 real long-tail benchmark 和 larger-architecture layerwise diagnostic。
"""
    write_markdown(OUTPUT_PATH, text)
    print(f"saved discussion outline to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
