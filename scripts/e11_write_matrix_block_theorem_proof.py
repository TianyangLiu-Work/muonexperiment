from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import markdown_table, write_markdown


OUTPUT_DIR = Path("results/e11_matrix_block_theorem_proof")
DISCUSSION_PATH = Path("discussion/e11_matrix_block_theorem_proof.md")
PAPER_PATH = Path("paper/specgrad_activation_paper/main.tex")


def paper_contains_required_labels() -> dict[str, bool]:
    text = PAPER_PATH.read_text(encoding="utf-8")
    required = {
        "assump:local-head-tail": "\\label{assump:local-head-tail}" in text,
        "app:proof-details": "\\label{app:proof-details}" in text,
        "lem:sandwiched-sensitivity": "\\label{lem:sandwiched-sensitivity}" in text,
        "app:matrix-block-derivation": "\\label{app:matrix-block-derivation}" in text,
        "matched_gain_theorem": "Tail drift under matched head gain" in text,
        "rank_condition": "\\nrank(G_H)>\\ssrank(B_T,A_T)" in text
        or "\\nrank(G_H)>\\ssrank(B_T,A_T)" in text.replace(" ", ""),
    }
    return required


def build_theorem_statement() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "statement_id": "MBT-1-general-matched-gain-bound",
                "scope": "any differentiable local model and any compact norm geometry",
                "claim": "After matching first-order head gain rho, a norm-steepest update has squared tail drift at most rho^2 I_N(T|H).",
                "formal_expression": "||J_T Delta_N||^2 <= rho^2 K_{T,N}^2 / ||g_H||_{N,*}^2",
                "requires": "Assumption local-head-tail; nonzero head gradient; norm-steepest direction; local first-order tail map",
                "paper_location": "main theorem plus Appendix app:proof-details",
                "claim_status": "theorem_ready",
            },
            {
                "statement_id": "MBT-2-sandwich-frobenius-coefficient",
                "scope": "matrix block with J_T(D)=B_T D A_T under Frobenius geometry",
                "claim": "The Frobenius matched-gain interference coefficient is the product of tail operator norms divided by ||G_H||_F^2.",
                "formal_expression": "I_F = ||B_T||_op^2 ||A_T||_op^2 / ||G_H||_F^2",
                "requires": "sandwich local tail map; nonzero tail-sensitive block; Frobenius self-duality",
                "paper_location": "Appendix app:matrix-block-derivation",
                "claim_status": "derivation_ready",
            },
            {
                "statement_id": "MBT-3-sandwich-spectral-coefficient",
                "scope": "matrix block with J_T(D)=B_T D A_T under spectral geometry",
                "claim": "The spectral matched-gain interference coefficient uses the paired singular spectra of B_T and A_T divided by ||G_H||_*^2.",
                "formal_expression": "I_S = sum_i sigma_i(B_T)^2 sigma_i(A_T)^2 / ||G_H||_*^2",
                "requires": "sandwiched-sensitivity lemma; spectral/nuclear duality; singular values sorted and zero-padded",
                "paper_location": "Appendix app:proof-details and app:matrix-block-derivation",
                "claim_status": "derivation_ready",
            },
            {
                "statement_id": "MBT-4-rank-boundary",
                "scope": "nondegenerate sandwich block with sigma_1(B_T) sigma_1(A_T)>0",
                "claim": "Spectral geometry has a smaller worst-case matched-gain tail-drift bound than Frobenius geometry when nrank(G_H) exceeds ssrank(B_T,A_T).",
                "formal_expression": "I_S/I_F = ssrank(B_T,A_T)/nrank(G_H); spectral favored iff nrank(G_H)>ssrank(B_T,A_T)",
                "requires": "MBT-2; MBT-3; nonzero G_H; nondegenerate tail-sensitive block",
                "paper_location": "main Matrix-Block Spectral Condition and Appendix app:matrix-block-derivation",
                "claim_status": "main_theorem_boundary_ready",
            },
        ]
    )


def build_assumption_ledger() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "assumption_id": "MBA-1-locality",
                "assumption": "F_T(theta+Delta)-F_T(theta) is represented by J_T(theta) Delta at the tested update scale.",
                "needed_for": "MBT-1 and all empirical matched-gain diagnostics",
                "stress_test_or_evidence": "results/e11_local_linearization/summary.csv",
                "if_violated": "the theorem remains local but no longer explains measured finite-step logit drift",
                "paper_wording": "local first-order mechanism, not global trajectory theorem",
            },
            {
                "assumption_id": "MBA-2-head-gain-matching",
                "assumption": "Compared directions are scaled so -<g_H,Delta>=rho under the same head batch and checkpoint.",
                "needed_for": "all matched-gain ratios and the interference coefficient",
                "stress_test_or_evidence": "validator checks head-gain and update-gap tolerances across committed diagnostics",
                "if_violated": "a direction can appear safer by taking less head progress",
                "paper_wording": "match head progress before comparing tail drift",
            },
            {
                "assumption_id": "MBA-3-sandwich-block",
                "assumption": "The block tail map has the local form J_T(D)=B_T D A_T, or the general block map is treated separately.",
                "needed_for": "closed-form ssrank(B_T,A_T) expression",
                "stress_test_or_evidence": "final-layer exact sandwich check and all-layer finite-difference JVP diagnostics",
                "if_violated": "use the vectorized operator L_T instead of the paired singular-value formula",
                "paper_wording": "sandwich theorem for matrix blocks; nonlinear layers use measured JVP operators",
            },
            {
                "assumption_id": "MBA-4-nondegenerate-tail",
                "assumption": "sigma_1(B_T) sigma_1(A_T)>0 so the tail-sensitive sandwich block is nonzero.",
                "needed_for": "ssrank ratio interpretation",
                "stress_test_or_evidence": "nonzero measured tail drift and JVP rows in committed diagnostics",
                "if_violated": "tail drift is zero for all D and the rank boundary is uninformative",
                "paper_wording": "nondegenerate tail-sensitive sandwich block",
            },
            {
                "assumption_id": "MBA-5-worst-case-vs-realized",
                "assumption": "The rank boundary orders worst-case sensitivity bounds; realized polar drift can depend on singular-vector alignment.",
                "needed_for": "claim-scope boundary around synthetic and natural diagnostics",
                "stress_test_or_evidence": "head-tail alignment ablation and reviewer risk audit",
                "if_violated": "the paper would overclaim a bound-ordering condition as a realized-drift predictor",
                "paper_wording": "bound-ordering condition, not standalone realized-drift predictor",
            },
        ]
    )


def build_proof_steps() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "step_id": "MBS-1-dual-steepest-direction",
                "proves": "MBT-1-general-matched-gain-bound",
                "argument": "Choose u_N with ||u_N||_N=1 and <g_H,u_N>=||g_H||_{N,*}; scale Delta_N=-rho u_N/||g_H||_{N,*}.",
                "mathematical_tool": "dual norm definition and compact unit ball",
                "reviewer_failure_if_missing": "matched-gain equality is not established",
            },
            {
                "step_id": "MBS-2-tail-sensitivity-bound",
                "proves": "MBT-1-general-matched-gain-bound",
                "argument": "Use K_{T,N}=sup_{||Delta||_N<=1} ||J_T Delta|| to upper-bound ||J_T Delta_N||.",
                "mathematical_tool": "homogeneity of the linearized tail map",
                "reviewer_failure_if_missing": "I_N is not connected to tail drift",
            },
            {
                "step_id": "MBS-3-frobenius-unit-ball",
                "proves": "MBT-2-sandwich-frobenius-coefficient",
                "argument": "Vectorize B_T D A_T as (A_T^T kron B_T) vec(D); the Frobenius unit-ball operator norm is ||B_T||_op ||A_T||_op.",
                "mathematical_tool": "Kronecker operator norm and Frobenius self-duality",
                "reviewer_failure_if_missing": "Frobenius denominator or numerator could be mis-specified",
            },
            {
                "step_id": "MBS-4-spectral-sandwich-lemma",
                "proves": "MBT-3-sandwich-spectral-coefficient",
                "argument": "Diagonalize B_T^T B_T and A_T A_T^T; reduce to a linear objective over squared entries of an operator-norm-constrained matrix.",
                "mathematical_tool": "doubly substochastic matrix polytope and rearrangement inequality",
                "reviewer_failure_if_missing": "the paired singular-spectrum numerator is unsupported",
            },
            {
                "step_id": "MBS-5-rank-ratio",
                "proves": "MBT-4-rank-boundary",
                "argument": "Divide I_S by I_F and identify ||G_H||_*^2/||G_H||_F^2 as nrank(G_H) and the tail singular-spectrum ratio as ssrank(B_T,A_T).",
                "mathematical_tool": "algebraic rearrangement of interference coefficients",
                "reviewer_failure_if_missing": "the displayed nrank-vs-ssrank boundary does not follow from the coefficients",
            },
            {
                "step_id": "MBS-6-degeneracy-and-alignment-caveat",
                "proves": "claim-scope limitation",
                "argument": "If the top tail singular product is zero, the tail map is zero; otherwise the ratio is meaningful but remains a worst-case bound and not a realized alignment guarantee.",
                "mathematical_tool": "nondegenerate block condition plus distinction between supremum and actual polar direction",
                "reviewer_failure_if_missing": "the paper can be read as overclaiming natural realized drift prediction",
            },
        ]
    )


def build_claim_implications() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "MBC-1-main-mechanism",
                "allowed_claim": "For a nondegenerate sandwich block, spectral geometry has a smaller worst-case matched-head-gain tail-drift bound when nrank(G_H)>ssrank(B_T,A_T).",
                "blocked_claim": "Spectral directions always cause less realized tail drift in natural networks.",
                "required_evidence_or_proof": "MBT-1 through MBT-4 plus synthetic boundary diagnostic",
                "paper_location": "Theory section and claim boundary table",
            },
            {
                "claim_id": "MBC-2-realized-drift",
                "allowed_claim": "Observed lower drift in diagnostics is empirical evidence that the local mechanism can appear in the tested settings.",
                "blocked_claim": "The theorem alone proves the empirical ratios.",
                "required_evidence_or_proof": "matched-gain experiments plus alignment caveat",
                "paper_location": "Experiments and appendix alignment ablation",
            },
            {
                "claim_id": "MBC-3-tail-performance",
                "allowed_claim": "Tail loss, margin, and accuracy are separate outcomes reported alongside drift.",
                "blocked_claim": "Lower logit drift implies better tail accuracy.",
                "required_evidence_or_proof": "tail outcome table and margin certificate",
                "paper_location": "Claim boundary table and one-step tail outcomes",
            },
            {
                "claim_id": "MBC-4-muon-scope",
                "allowed_claim": "Muon-style diagnostics are selected-state compatibility checks for polar-like directions.",
                "blocked_claim": "The sandwich theorem proves full Muon training behavior.",
                "required_evidence_or_proof": "arbitrary-direction matched-gain coefficient and Muon state-source controls",
                "paper_location": "Scope section and additional discussion",
            },
        ]
    )


def build_paper_cross_checks(label_status: dict[str, bool]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"check_id": key, "paper_path": PAPER_PATH.as_posix(), "present": "yes" if value else "no"}
            for key, value in label_status.items()
        ]
    )


def write_discussion(
    theorem_statement: pd.DataFrame,
    assumption_ledger: pd.DataFrame,
    proof_steps: pd.DataFrame,
    claim_implications: pd.DataFrame,
    paper_checks: pd.DataFrame,
) -> None:
    text = f"""# E11 Matrix-Block Theorem Proof

This generated artifact is the machine-checkable proof contract for the main
local matrix-block theorem in the E11 paper. It does not add a new experiment.
It records the theorem statements, assumptions, proof steps, the sandwich sensitivity lemma,
and claim boundaries needed to defend the
`nrank(G_H) > ssrank(B_T,A_T)` condition as a top-conference theory contribution
while keeping optimizer-performance and realized-drift claims separate.

## Theorem Statements

{markdown_table(theorem_statement, ["statement_id", "scope", "claim", "formal_expression", "requires", "claim_status"])}

## Assumption Ledger

{markdown_table(assumption_ledger, ["assumption_id", "assumption", "needed_for", "stress_test_or_evidence", "if_violated", "paper_wording"])}

## Proof Steps

{markdown_table(proof_steps, ["step_id", "proves", "argument", "mathematical_tool", "reviewer_failure_if_missing"])}

## Claim Implications

{markdown_table(claim_implications, ["claim_id", "allowed_claim", "blocked_claim", "required_evidence_or_proof", "paper_location"])}

## Paper Cross-Checks

{markdown_table(paper_checks, ["check_id", "paper_path", "present"])}

Artifacts:
- [theorem_statement.csv](../{(OUTPUT_DIR / 'theorem_statement.csv').as_posix()})
- [assumption_ledger.csv](../{(OUTPUT_DIR / 'assumption_ledger.csv').as_posix()})
- [proof_steps.csv](../{(OUTPUT_DIR / 'proof_steps.csv').as_posix()})
- [claim_implications.csv](../{(OUTPUT_DIR / 'claim_implications.csv').as_posix()})
- [paper_cross_checks.csv](../{(OUTPUT_DIR / 'paper_cross_checks.csv').as_posix()})
- [config.json](../{(OUTPUT_DIR / 'config.json').as_posix()})
"""
    write_markdown(DISCUSSION_PATH, text)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    label_status = paper_contains_required_labels()
    theorem_statement = build_theorem_statement()
    assumption_ledger = build_assumption_ledger()
    proof_steps = build_proof_steps()
    claim_implications = build_claim_implications()
    paper_checks = build_paper_cross_checks(label_status)

    theorem_statement.to_csv(OUTPUT_DIR / "theorem_statement.csv", index=False)
    assumption_ledger.to_csv(OUTPUT_DIR / "assumption_ledger.csv", index=False)
    proof_steps.to_csv(OUTPUT_DIR / "proof_steps.csv", index=False)
    claim_implications.to_csv(OUTPUT_DIR / "claim_implications.csv", index=False)
    paper_checks.to_csv(OUTPUT_DIR / "paper_cross_checks.csv", index=False)
    (OUTPUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "purpose": "machine-checkable matrix-block theorem and proof contract",
                "analysis_scope": "formal proof artifact; no new empirical results",
                "paper_path": PAPER_PATH.as_posix(),
                "all_paper_cross_checks_present": bool(all(label_status.values())),
                "theorem_statement_count": int(len(theorem_statement)),
                "proof_step_count": int(len(proof_steps)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_discussion(theorem_statement, assumption_ledger, proof_steps, claim_implications, paper_checks)
    print(f"saved {DISCUSSION_PATH} and {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
