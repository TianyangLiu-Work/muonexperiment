from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, write_markdown


OUTPUT_DIR = Path("results/e11_mechanism_referee_audit")
OUTPUT_PATH = Path("discussion/e11_mechanism_referee_audit.md")


def ci(row: pd.Series, low: str, high: str) -> str:
    return f"[{fmt(row[low])}, {fmt(row[high])}]"


def read_one(path: str) -> pd.Series:
    return pd.read_csv(path).iloc[0]


def build_evidence() -> dict[str, object]:
    one_step = read_one("results/e11_long_tail_one_step/pair_summary.csv")
    resnet = read_one("results/e11_cifar100_resnet_one_step/pair_summary.csv")
    checkpoint_sweep = pd.read_csv("results/e11_cifar100_resnet_checkpoint_sweep/pair_summary.csv")
    tail_quality = pd.read_csv("results/e11_cifar100_resnet_tail_quality_control/pair_summary.csv")
    imbalance = pd.read_csv("results/e11_cifar100_resnet_imbalance_sweep/pair_summary.csv")
    layerwise = pd.read_csv("results/e11_long_tail_layerwise/summary.csv")
    all_layer_jvp = read_one("results/e11_cifar100_resnet_layer_jvp_tail_quality/overall_summary.csv")
    residual_scores = pd.read_csv(
        "results/e11_cifar100_resnet_condition_score_audit/residual_score_summary.csv"
    ).set_index("score")
    v5_final_gates = pd.read_csv("results/e11_condition_score_v5_protocol/final_score_evaluation/final_gate_report.csv")
    phase1_coverage = pd.read_csv(
        "results/e11_natural_negative_search_protocol/phase1_interim_synthesis/family_coverage.csv"
    )
    phase1_summary = pd.read_csv(
        "results/e11_natural_negative_search_protocol/phase1_interim_synthesis/observed_primary_summary.csv"
    )
    phase1_gates = pd.read_csv(
        "results/e11_natural_negative_search_protocol/phase1_multiplicity_evaluation/gate_report.csv"
    )
    phase2_registry = pd.read_csv(
        "results/e11_natural_negative_search_protocol/phase2_heldout_architecture/settings_registry.csv"
    )
    phase2_decisions = pd.read_csv(
        "results/e11_natural_negative_search_protocol/phase2_multiplicity_evaluation/primary_decisions.csv"
    )
    phase2_gates = pd.read_csv(
        "results/e11_natural_negative_search_protocol/phase2_multiplicity_evaluation/gate_report.csv"
    )
    phase2_mde = pd.read_csv(
        "results/e11_natural_negative_search_protocol/phase2_power_audit/minimum_detectable_effect.csv"
    )
    phase2_state_machine = pd.read_csv(
        "results/e11_natural_negative_search_protocol/phase2_power_audit/outcome_state_machine.csv"
    )
    muon_final_pairs = pd.read_csv(
        "results/e11_cifar100_resnet_lt_muon_final_benchmark/pair_summary.csv"
    ).set_index(["recipe", "frequency_group"])
    local_linear = pd.read_csv("results/e11_local_linearization/summary.csv")
    v5_freeze = pd.read_csv("results/e11_condition_score_v5_protocol/validation_score_freeze/freeze_status.csv")
    ablation_ladder = pd.read_csv("results/e11_condition_score_ablation/term_failure_ladder.csv")

    return {
        "one_step": one_step,
        "resnet": resnet,
        "checkpoint_worst": checkpoint_sweep.loc[
            checkpoint_sweep["tail_output_drift_sq_ratio_ci95_high"].idxmax()
        ],
        "tail_quality_best": tail_quality.loc[tail_quality["mean_tail_accuracy_before"].idxmax()],
        "tail_quality_worst": tail_quality.loc[tail_quality["tail_output_drift_sq_ratio_ci95_high"].idxmax()],
        "imbalance_worst": imbalance.loc[imbalance["tail_output_drift_sq_ratio_ci95_high"].idxmax()],
        "layer_1": layerwise[layerwise["layer"].eq(1)].iloc[0],
        "layer_2": layerwise[layerwise["layer"].eq(2)].iloc[0],
        "all_layer_jvp": all_layer_jvp,
        "residual_scaled_jvp": residual_scores.loc["scaled_jvp_residual"],
        "residual_observed": residual_scores.loc["source_observed_residual_positive_control"],
        "v5_final_gates": v5_final_gates,
        "phase1_coverage": phase1_coverage,
        "phase1_summary": phase1_summary,
        "phase1_gates": phase1_gates,
        "phase2_registry": phase2_registry,
        "phase2_decisions": phase2_decisions,
        "phase2_gates": phase2_gates,
        "phase2_mde": phase2_mde,
        "phase2_state_machine": phase2_state_machine,
        "muon_final_few": muon_final_pairs.loc[("ns_muon_aug_lr1e-4", "few")],
        "local_linear": local_linear,
        "v5_freeze": v5_freeze,
        "ablation_ladder": ablation_ladder,
    }


def build_alternative_explanations(e: dict[str, object]) -> pd.DataFrame:
    one_step = e["one_step"]
    resnet = e["resnet"]
    checkpoint_worst = e["checkpoint_worst"]
    tail_quality_best = e["tail_quality_best"]
    tail_quality_worst = e["tail_quality_worst"]
    imbalance_worst = e["imbalance_worst"]
    layer_1 = e["layer_1"]
    layer_2 = e["layer_2"]
    all_layer_jvp = e["all_layer_jvp"]
    residual_scaled_jvp = e["residual_scaled_jvp"]
    residual_observed = e["residual_observed"]
    phase1_coverage = e["phase1_coverage"]
    phase1_summary = e["phase1_summary"]
    phase1_gates = e["phase1_gates"]
    phase2_registry = e["phase2_registry"]
    phase2_decisions = e["phase2_decisions"]
    phase2_gates = e["phase2_gates"]
    phase2_mde = e["phase2_mde"]
    muon_final_few = e["muon_final_few"]

    coverage = (
        f"{int(phase1_coverage['observed_primary_rows'].sum())}/"
        f"{int(phase1_coverage['expected_settings'].sum())}"
    )
    all_observed = phase1_summary[phase1_summary["scope"].eq("all_observed")].iloc[0]
    gate_status = phase1_gates.set_index("gate_id")["status"].astype(str).to_dict()
    phase2_gate_status = phase2_gates.set_index("gate_id")["status"].astype(str).to_dict()
    phase2_mde_80 = phase2_mde[
        phase2_mde["alpha_scope"].eq("holm_bonferroni_worst_case")
        & phase2_mde["target_power"].eq(0.8)
        & phase2_mde["log_ratio_sd"].eq(0.2)
    ].iloc[0]
    phase2_output_counts = phase2_decisions["output_status"].value_counts().astype(int).to_dict()
    phase2_observed_rows = int(phase2_decisions["output_status"].eq("observed").sum())
    phase2_adjusted_worse_rows = int(
        phase2_decisions["adjusted_primary_decision"].eq("primary_worse_adjusted").sum()
    )
    phase2_head_gain = phase2_decisions["head_gain_gate"].astype(str).str.lower().eq("true")
    phase2_head_gain_fail_rows = int((~phase2_head_gain).sum())
    rows = [
        {
            "audit_id": "MEA-1-head-gain-mismatch",
            "alternative_explanation": "Spectral/polar appears better only because it takes a smaller useful head step.",
            "current_status": "addressed_for_local_claim",
            "decisive_evidence": (
                "The protocol matches head gain before comparing tail drift. "
                f"Digits drift ratio is {fmt(one_step['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                f"{ci(one_step, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')}; "
                f"CIFAR-100-LT ResNet18 drift ratio is {fmt(resnet['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                f"{ci(resnet, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')}; "
                "the validator enforces matched-update/head-gain consistency across committed probes."
            ),
            "remaining_risk": "This remains a matched-head-gain mechanism claim, not an optimizer-schedule claim.",
            "manuscript_action": "State matched head gain before every tail-drift ratio and keep schedule-level claims out of scope.",
            "forbidden_wording": "spectral wins because it always has a smaller raw update norm in real training",
        },
        {
            "audit_id": "MEA-2-tail-unit-sensitivity",
            "alternative_explanation": "The result is just lower unit tail sensitivity for spectral directions.",
            "current_status": "rejected_as_primary_explanation",
            "decisive_evidence": (
                f"Digits layer 1 and 2 unit-JVP ratios are {fmt(layer_1['geomean_jvp_tail_drift_sq_ratio_spectral_over_fro'])} "
                f"and {fmt(layer_2['geomean_jvp_tail_drift_sq_ratio_spectral_over_fro'])}, both above one, "
                f"while observed ratios are {fmt(layer_1['geomean_observed_tail_drift_sq_ratio_spectral_over_fro'])} "
                f"and {fmt(layer_2['geomean_observed_tail_drift_sq_ratio_spectral_over_fro'])}. "
                f"The all-layer ResNet unit-JVP ratio is {fmt(all_layer_jvp['geomean_jvp_tail_drift_sq_ratio_spectral_over_fro'])} "
                f"{ci(all_layer_jvp, 'jvp_tail_drift_sq_ratio_ci95_low', 'jvp_tail_drift_sq_ratio_ci95_high')}, "
                f"but matched-gain observed ratio is {fmt(all_layer_jvp['geomean_observed_tail_drift_sq_ratio_spectral_over_fro'])} "
                f"{ci(all_layer_jvp, 'observed_tail_drift_sq_ratio_ci95_low', 'observed_tail_drift_sq_ratio_ci95_high')}."
            ),
            "remaining_risk": "The effect is a scaled matched-head-gain response, not a universal unit-direction safety result.",
            "manuscript_action": "Use scaled-JVP and observed drift language; do not call unit spectral directions inherently safer.",
            "forbidden_wording": "spectral directions have lower tail sensitivity per unit update",
        },
        {
            "audit_id": "MEA-3-weak-tail-checkpoint",
            "alternative_explanation": "The effect is an artifact of a useless tail classifier.",
            "current_status": "partially_addressed",
            "decisive_evidence": (
                f"The best tail-quality control has pre-update tail accuracy {fmt(tail_quality_best['mean_tail_accuracy_before'])} "
                f"{ci(tail_quality_best, 'tail_accuracy_before_ci95_low', 'tail_accuracy_before_ci95_high')}; "
                f"its worst drift CI upper endpoint is {fmt(tail_quality_worst['tail_output_drift_sq_ratio_ci95_high'])}. "
                f"The imbalance sweep worst drift CI upper endpoint is {fmt(imbalance_worst['tail_output_drift_sq_ratio_ci95_high'])}."
            ),
            "remaining_risk": "Tail-quality controls strengthen drift interpretability but do not prove accuracy improvement.",
            "manuscript_action": "Report tail-quality controls as context and keep accuracy claims separate.",
            "forbidden_wording": "the tail-rich control proves better tail classification",
        },
        {
            "audit_id": "MEA-4-synthetic-only",
            "alternative_explanation": "The mechanism only works in constructed matrix blocks.",
            "current_status": "addressed_as_diagnostic_not_general_theorem",
            "decisive_evidence": (
                f"CIFAR-100-LT ResNet18 one-step ratio is {fmt(resnet['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                f"{ci(resnet, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')}; "
                f"checkpoint sweep worst CI upper endpoint is {fmt(checkpoint_worst['tail_output_drift_sq_ratio_ci95_high'])}; "
                f"all-layer ResNet observed ratio is {fmt(all_layer_jvp['geomean_observed_tail_drift_sq_ratio_spectral_over_fro'])}."
            ),
            "remaining_risk": "Generality beyond the registered architectures/data remains pending.",
            "manuscript_action": "Frame natural evidence as diagnostics and wait for held-out architecture/data gates before generality wording.",
            "forbidden_wording": "the synthetic theorem predicts all natural network settings",
        },
        {
            "audit_id": "MEA-5-rank-only-predictor",
            "alternative_explanation": "The rank condition alone is being overclaimed as a natural residual-risk predictor.",
            "current_status": "quarantined",
            "decisive_evidence": (
                f"Source-observed residual control Spearman is {fmt(residual_observed['mean_spearman'])} "
                f"{ci(residual_observed, 'spearman_ci95_low', 'spearman_ci95_high')}, "
                f"but scaled-JVP residual Spearman is {fmt(residual_scaled_jvp['mean_spearman'])} "
                f"{ci(residual_scaled_jvp, 'spearman_ci95_low', 'spearman_ci95_high')}. "
                "The v5 transport-normalized score is frozen and the completed final gates failed: ResNeXt50 direction threshold failed, while CIFAR-10 residual ranking reversed."
            ),
            "remaining_risk": "A broad predictive-condition claim is blocked by the completed v5 final failures.",
            "manuscript_action": "Keep the theorem as a mechanism boundary and the score as a completed negative final boundary.",
            "forbidden_wording": "nrank(G_H)>ssrank(B_T,A_T) predicts unseen layer risk by itself",
        },
        {
            "audit_id": "MEA-6-post-hoc-score-tuning",
            "alternative_explanation": "The score was tuned on failed held-out finals.",
            "current_status": "controlled_by_registry",
            "decisive_evidence": (
                "The v5 validation-freeze artifact freezes the score before final rows exist; "
                "the final evaluator now reports both registered final split outputs and preserves the failed gates without score repair."
            ),
            "remaining_risk": "Any future score repair must use a new unspent protocol.",
            "manuscript_action": "State spent-final quarantine and no-refit rules in the appendix and rebuttal packet.",
            "forbidden_wording": "final rows influenced the v5 score formula or threshold",
        },
        {
            "audit_id": "MEA-7-anecdotal-natural-negative",
            "alternative_explanation": "Natural positive/negative claims are cherry-picked.",
            "current_status": "finite_phase1_and_phase2_null_candidates_with_caveats",
            "decisive_evidence": (
                f"Current phase1 coverage is {coverage} observed registered settings; "
                f"raw_worse_rows={int(all_observed['raw_worse_rows'])}; "
                f"adjusted primary gate={gate_status['NNS-E4-natural-primary-claim']}; "
                f"quality_gate_pass_rows={int(all_observed['quality_gate_pass_rows'])} and "
                f"quality_gate_fail_rows={int(all_observed['quality_gate_fail_rows'])}. "
                f"Phase2 completed the {len(phase2_registry)}-setting ResNet34 held-out architecture family "
                f"with observed rows={phase2_observed_rows}, adjusted_worse_rows={phase2_adjusted_worse_rows}, "
                f"head_gain_gate_fail_rows={phase2_head_gain_fail_rows}, output_status counts {phase2_output_counts}, and "
                f"NNS-P2-E4={phase2_gate_status['NNS-P2-E4-heldout-architecture-claim']}."
            ),
            "remaining_risk": "The result is finite registered phase1 and phase2 null-candidate evidence with power, head-gain, and tail-quality caveats, not a universal absence claim or mechanism validation.",
            "manuscript_action": "Report no adjusted primary phase1/phase2 counterexample with detectable-effect, head-gain, and tail-quality caveats; do not call the phase2 rows quality-valid mechanism evidence.",
            "forbidden_wording": "no natural counterexample exists, or a natural counterexample has been found",
        },
        {
            "audit_id": "MEA-9-phase2-power-overread",
            "alternative_explanation": "The held-out ResNet34 phase2 finite-null readout will be overread despite limited power and failed head-gain gates.",
            "current_status": "completed_phase2_finite_null_with_power_and_quality_caveats",
            "decisive_evidence": (
                f"The phase2 evaluator reports NNS-P2-E2={phase2_gate_status['NNS-P2-E2-phase2-output-completeness']} "
                f"and NNS-P2-E3={phase2_gate_status['NNS-P2-E3-primary-multiplicity']}; "
                f"primary_decisions.csv keeps all {len(phase2_decisions)} registered rows as "
                f"{phase2_output_counts}, with adjusted_worse_rows={phase2_adjusted_worse_rows} and "
                f"head_gain_gate_fail_rows={phase2_head_gain_fail_rows}. The phase2 power audit fixes the Holm worst-case 80% MDE "
                f"at log-ratio SD 0.2 to ratio {fmt(phase2_mde_80['minimum_detectable_ratio'])}, "
                "so a complete null below that scale must be described as underpowered."
            ),
            "remaining_risk": "A three-seed, eight-setting held-out family can falsify large ResNet34 boundary cases, but it cannot exclude smaller effects or other architectures.",
            "manuscript_action": "Present phase2 as a completed finite-null candidate only within the registered ResNet34 family, with explicit power and head-gain caveats.",
            "forbidden_wording": "phase2 confirms natural generality, validates the mechanism, or rules out held-out architecture boundary cases",
        },
        {
            "audit_id": "MEA-8-performance-proxy",
            "alternative_explanation": "Lower local drift is being used as a proxy for optimizer superiority.",
            "current_status": "not_claimed",
            "decisive_evidence": (
                f"The NS-Muon final-training pilot is negative on few-shot balanced accuracy: diff "
                f"{fmt(muon_final_few['mean_balanced_accuracy_diff'])} "
                f"{ci(muon_final_few, 'balanced_accuracy_diff_ci95_low', 'balanced_accuracy_diff_ci95_high')} "
                "versus augmented AdamW."
            ),
            "remaining_risk": "Performance claims require tuned validation selection and untouched final seeds.",
            "manuscript_action": "Use benchmark pilots only as claim-scope context.",
            "forbidden_wording": "local drift improvements imply final long-tail optimizer superiority",
        },
    ]
    return pd.DataFrame(rows)


def build_theory_measurement_contract(e: dict[str, object]) -> pd.DataFrame:
    local_linear = e["local_linear"]
    v5_freeze = e["v5_freeze"]
    v5_final_gates = e["v5_final_gates"]
    ablation_ladder = e["ablation_ladder"]
    phase1_coverage = e["phase1_coverage"]
    phase1_gates = e["phase1_gates"]
    phase2_registry = e["phase2_registry"]
    phase2_decisions = e["phase2_decisions"]
    phase2_gates = e["phase2_gates"]
    phase2_mde = e["phase2_mde"]
    max_linear_error = local_linear["relative_error_ci95_high"].max()
    freeze_status = "; ".join(
        f"{row.item}={row.status}" for row in v5_freeze[["item", "status"]].itertuples(index=False)
    )
    final_status = "; ".join(
        f"{row.gate_id}={row.status}" for row in v5_final_gates[["gate_id", "status"]].itertuples(index=False)
    )
    ablation_terms = ", ".join(ablation_ladder["term_tested"].astype(str).tolist())
    phase1_gate_status = phase1_gates.set_index("gate_id")["status"].astype(str).to_dict()
    phase2_gate_status = phase2_gates.set_index("gate_id")["status"].astype(str).to_dict()
    phase2_observed_rows = int(phase2_decisions["observed_seeds"].gt(0).sum())
    phase2_mde_80 = phase2_mde[
        phase2_mde["alpha_scope"].eq("holm_bonferroni_worst_case")
        & phase2_mde["target_power"].eq(0.8)
        & phase2_mde["log_ratio_sd"].eq(0.2)
    ].iloc[0]
    rows = [
        {
            "contract_id": "TMC-1-local-linearization",
            "theory_object": "first-order tail-logit response J_T Delta",
            "measurement_proxy": "local_linearization_errors.tex and results/e11_local_linearization/summary.csv",
            "current_evidence": f"maximum reported relative-error CI high is {fmt(max_linear_error)}",
            "supports": "local finite-step interpretation at the tested head-gain scale",
            "does_not_support": "global trajectory or convergence theorem",
            "next_gate": "keep update-scale residuals in the appendix",
        },
        {
            "contract_id": "TMC-2-matched-head-gain",
            "theory_object": "head-gain-normalized comparison of update directions",
            "measurement_proxy": "paired diagnostics and validator head-gain/update-gap checks",
            "current_evidence": "e11_validate_outputs.py enforces committed matched-gain/update contracts",
            "supports": "tail drift comparisons at fixed immediate head progress",
            "does_not_support": "claims about a full optimizer learning-rate schedule",
            "next_gate": "do not remove same-batch and matched-head-gain wording",
        },
        {
            "contract_id": "TMC-3-sandwich-rank-boundary",
            "theory_object": "nrank(G_H)>ssrank(B_T,A_T) for worst-case matched-gain tail drift",
            "measurement_proxy": "matrix-block theorem proof, tightness audit, and condition-score theory map",
            "current_evidence": "exact theorem/proof contract and ratio-identity audit are generated",
            "supports": "constructed mechanism sign boundary",
            "does_not_support": "standalone natural residual-risk prediction",
            "next_gate": "keep theorem proof and tightness audit synchronized with the paper appendix",
        },
        {
            "contract_id": "TMC-4-transport-normalized-score",
            "theory_object": "source-standardized transport residual for unseen architecture/data splits",
            "measurement_proxy": "v5 validation-freeze and final evaluator",
            "current_evidence": f"{freeze_status}; {final_status}",
            "supports": "registered candidate before final rows",
            "does_not_support": "P0 predictive-condition generality before finals pass",
            "next_gate": "consume ResNeXt50-32x4d and CIFAR-10 cross-partition final outputs with no refit",
        },
        {
            "contract_id": "TMC-5-term-ablation-lineage",
            "theory_object": "separate direction guardrail, amplitude, depth, and transport terms",
            "measurement_proxy": "condition-score ablation ladder",
            "current_evidence": ablation_terms,
            "supports": "failure localization and no-leakage score lineage",
            "does_not_support": "v5 final success",
            "next_gate": "map any final failure to the pre-output reviewer failure response",
        },
        {
            "contract_id": "TMC-6-natural-negative-registered-boundary",
            "theory_object": "finite natural boundary search under a fixed primary drift ratio and multiplicity rule",
            "measurement_proxy": "phase1/phase2 natural-negative evaluators plus phase2 detectable-effect audit",
            "current_evidence": (
                f"phase1 expected rows={int(phase1_coverage['expected_settings'].sum())}, "
                f"observed rows={int(phase1_coverage['observed_primary_rows'].sum())}, "
                f"NNS-E4={phase1_gate_status['NNS-E4-natural-primary-claim']}; "
                f"phase2 registered settings={len(phase2_registry)}, observed rows={phase2_observed_rows}, "
                f"NNS-P2-E4={phase2_gate_status['NNS-P2-E4-heldout-architecture-claim']}; "
                f"Holm worst-case 80% MDE at SD 0.2 is {fmt(phase2_mde_80['minimum_detectable_ratio'])}"
            ),
            "supports": "registered finite-family falsification protocol and completed phase2 finite-null interpretation boundary",
            "does_not_support": "universal natural-null wording outside registered phase1/phase2 families or mechanism validation from head-gain-failed rows",
            "next_gate": "register larger-dataset or additional held-out-architecture families before broadening beyond the completed phase1/phase2 boundary",
        },
    ]
    return pd.DataFrame(rows)


def build_falsification_triggers(e: dict[str, object]) -> pd.DataFrame:
    v5_final_gates = e["v5_final_gates"]
    phase1_coverage = e["phase1_coverage"]
    phase1_summary = e["phase1_summary"]
    phase1_gates = e["phase1_gates"]
    phase2_decisions = e["phase2_decisions"]
    phase2_gates = e["phase2_gates"]
    phase2_mde = e["phase2_mde"]
    phase2_state_machine = e["phase2_state_machine"]
    all_layer_jvp = e["all_layer_jvp"]
    v5_status_text = "; ".join(
        f"{row.gate_id}={row.status}" for row in v5_final_gates[["gate_id", "status"]].itertuples(index=False)
    )
    phase1_missing = int(phase1_coverage["missing_primary_rows"].sum())
    phase1_all = phase1_summary[phase1_summary["scope"].eq("all_observed")].iloc[0]
    phase1_gate_status = phase1_gates.set_index("gate_id")["status"].astype(str).to_dict()
    phase2_gate_status = phase2_gates.set_index("gate_id")["status"].astype(str).to_dict()
    phase2_output_counts = phase2_decisions["output_status"].value_counts().astype(int).to_dict()
    phase2_mde_80 = phase2_mde[
        phase2_mde["alpha_scope"].eq("holm_bonferroni_worst_case")
        & phase2_mde["target_power"].eq(0.8)
        & phase2_mde["log_ratio_sd"].eq(0.2)
    ].iloc[0]
    phase2_state_lookup = phase2_state_machine.set_index("state_id")["claim_state"].astype(str).to_dict()
    rows = [
        {
            "trigger_id": "FT-1-v5-final-fails",
            "trigger_condition": "Either registered v5 final split fails residual, direction, baseline-dominance, or reporting gates.",
            "current_status": v5_status_text,
            "required_action": "Downgrade predictive-condition wording under the frozen reviewer failure response.",
            "claim_downgrade": "from registered-not-ready to fixed-split diagnostic or failed predictive condition",
        },
        {
            "trigger_id": "FT-2-natural-family-incomplete",
            "trigger_condition": "Any future registered natural-negative family is incomplete, or finite-null wording drops the detectable-effect, head-gain, and quality caveats.",
            "current_status": (
                f"{phase1_missing} registered primary rows missing in completed coverage table; "
                f"NNS-E4={phase1_gate_status['NNS-E4-natural-primary-claim']}; "
                f"quality_gate_fail_rows={int(phase1_all['quality_gate_fail_rows'])}; "
                f"phase2 output_status counts={phase2_output_counts}; "
                f"NNS-P2-E4={phase2_gate_status['NNS-P2-E4-heldout-architecture-claim']}"
            ),
            "required_action": "Allow only finite registered phase1/phase2 null-candidate wording with detectable-effect, head-gain, and quality caveats.",
            "claim_downgrade": "finite phase1/phase2 null candidates with detectable-effect, head-gain, and quality caveats",
        },
        {
            "trigger_id": "FT-3-unit-jvp-misread",
            "trigger_condition": "Reviewer reads the result as lower unit tail sensitivity rather than matched-gain scaling.",
            "current_status": (
                f"all-layer unit-JVP ratio {fmt(all_layer_jvp['geomean_jvp_tail_drift_sq_ratio_spectral_over_fro'])} "
                f"while observed ratio {fmt(all_layer_jvp['geomean_observed_tail_drift_sq_ratio_spectral_over_fro'])}"
            ),
            "required_action": "Move scaled-JVP and step-size-normalization caveat into main text or caption.",
            "claim_downgrade": "matched-head-gain local mechanism only",
        },
        {
            "trigger_id": "FT-4-performance-overread",
            "trigger_condition": "Any wording links lower local drift directly to better final accuracy.",
            "current_status": "negative NS-Muon final-training pilot and tuned benchmark protocol pending",
            "required_action": "Remove optimizer-superiority wording and cite final-performance separation.",
            "claim_downgrade": "mechanism diagnostic, not benchmark claim",
        },
        {
            "trigger_id": "FT-5-clean-checkout-gap",
            "trigger_condition": "Preferred pdflatex/bibtex/xelatex clean-checkout build is absent.",
            "current_status": "Tectonic-backed server evidence pass; preferred LaTeX gate not_ready",
            "required_action": "Keep artifact-review wording tied to Tectonic fallback until venue-style clean checkout passes.",
            "claim_downgrade": "server artifact reproducibility with toolchain caveat",
        },
        {
            "trigger_id": "FT-6-phase2-power-overread",
            "trigger_condition": "The ResNet34 phase2 family is interpreted as a broad held-out null despite failed head-gain gates or an observed effect scale below the audited detectable-effect floor.",
            "current_status": (
                f"P2-S1 claim_state={phase2_state_lookup['P2-S1-not-run']}; "
                f"NNS-P2-E2={phase2_gate_status['NNS-P2-E2-phase2-output-completeness']}; "
                f"Holm worst-case 80% MDE at SD 0.2 is {fmt(phase2_mde_80['minimum_detectable_ratio'])}"
            ),
            "required_action": "Use the phase2 outcome-state machine and MDE table before any ResNet34 held-out architecture claim; mark small-effect nulls as underpowered.",
            "claim_downgrade": "caveated finite phase2 null candidate or underpowered phase2 null, not broad natural-boundary evidence",
        },
    ]
    return pd.DataFrame(rows)


def write_outputs(
    alternatives: pd.DataFrame,
    contract: pd.DataFrame,
    triggers: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    alternatives.to_csv(OUTPUT_DIR / "alternative_explanation_matrix.csv", index=False)
    contract.to_csv(OUTPUT_DIR / "theory_measurement_contract.csv", index=False)
    triggers.to_csv(OUTPUT_DIR / "falsification_trigger_matrix.csv", index=False)
    config = {
        "purpose": "adversarial referee audit for the E11 mechanism paper",
        "alternative_explanations": int(len(alternatives)),
        "theory_measurement_contracts": int(len(contract)),
        "falsification_triggers": int(len(triggers)),
        "claim_boundary": "local matched-head-gain mechanism; finite registered phase1/phase2 natural-null candidates with power, head-gain, and quality caveats; no final-performance or v5 predictive-condition upgrade",
    }
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    text = f"""# E11 Mechanism Referee Audit

This generated audit takes an adversarial referee stance. It lists the strongest alternative explanations for the current head-to-tail mechanism claim, the committed evidence that answers or limits each one, and the exact wording discipline required to avoid overclaiming. It does not add new empirical results.

## Alternative Explanation Matrix

{markdown_table(alternatives, ["audit_id", "alternative_explanation", "current_status", "decisive_evidence", "remaining_risk", "manuscript_action", "forbidden_wording"])}

## Theory-To-Measurement Contract

{markdown_table(contract, ["contract_id", "theory_object", "measurement_proxy", "current_evidence", "supports", "does_not_support", "next_gate"])}

## Falsification Trigger Matrix

{markdown_table(triggers, ["trigger_id", "trigger_condition", "current_status", "required_action", "claim_downgrade"])}

## Claim Boundary

Allowed now: a local matched-head-gain mechanism paper with explicit theorem assumptions, diagnostic natural evidence, completed caveated phase2 finite-null-candidate wording, artifact-review reproducibility, and locked pending gates.

Blocked now: broad optimizer-performance claims, unqualified natural-null or counterexample wording, and v5 predictive-condition generality before the registered final split gates pass.

Artifacts:
- [alternative_explanation_matrix.csv](../results/e11_mechanism_referee_audit/alternative_explanation_matrix.csv)
- [theory_measurement_contract.csv](../results/e11_mechanism_referee_audit/theory_measurement_contract.csv)
- [falsification_trigger_matrix.csv](../results/e11_mechanism_referee_audit/falsification_trigger_matrix.csv)
- [config.json](../results/e11_mechanism_referee_audit/config.json)
"""
    write_markdown(OUTPUT_PATH, text)


def main() -> None:
    evidence = build_evidence()
    alternatives = build_alternative_explanations(evidence)
    contract = build_theory_measurement_contract(evidence)
    triggers = build_falsification_triggers(evidence)
    write_outputs(alternatives, contract, triggers)
    print(f"saved mechanism referee audit to {OUTPUT_PATH} and {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
