from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import markdown_table, write_markdown


OUTPUT_DIR = Path("results/e11_manuscript_claim_trace")
DISCUSSION_PATH = Path("discussion/e11_manuscript_claim_trace.md")
MAIN_TEX = Path("paper/specgrad_activation_paper/main.tex")
CLAIM_DECISIONS = Path("results/e11_top_conference_claim_decision_audit/claim_decision_matrix.csv")


CLAIM_ANCHORS: dict[str, dict[str, object]] = {
    "TCD-1-main-mechanism-theorem": {
        "main_tex_location": "abstract, introduction, theorem, and conclusion",
        "required_anchors": [
            "Tail drift under matched head gain",
            r"\nrank(G_H)>\ssrank(B_T,A_T)",
            "not a global optimizer theorem",
            "Worst-case and realized coefficients",
        ],
        "trace_status": "present_as_supportable_scoped_claim",
    },
    "TCD-2-natural-drift-diagnostic": {
        "main_tex_location": "abstract, experiment overview, digits/ResNet results, and appendix diagnostics",
        "required_anchors": [
            "tail-example logit drift",
            "tail loss, margin, and accuracy are reported separately",
            "lower logit drift does not imply lower cross-entropy loss",
            "supportable_diagnostic_only",
        ],
        "trace_status": "present_as_diagnostic_only",
    },
    "TCD-3-predictive-condition-generalization": {
        "main_tex_location": "limitations and appendix condition-score audit",
        "required_anchors": [
            "v5 validation-only split freezes a transport-normalized candidate",
            "completed final gates failed",
            "ResNeXt50-32x4d residual ranking survives but direction threshold fails",
            "CIFAR-10 cross-partition residual ranking reverses but direction threshold survives",
        ],
        "trace_status": "present_as_completed_negative_boundary",
    },
    "TCD-4-natural-counterexample-or-finite-null": {
        "main_tex_location": "limitations",
        "required_anchors": [
            r"observed \(26/26\) settings",
            r"observed \(8/8\) phase2 settings",
            r"\texttt{raw\_worse\_rows=0}",
            "finite registered phase1 and phase2 null candidates",
            "detectable-effect, head-gain, and quality caveats",
            "do not prove that no natural counterexample exists outside the registered phase1/phase2 spaces",
        ],
        "trace_status": "present_as_finite_null_candidate_with_caveats",
    },
    "TCD-5-optimizer-performance-benchmark": {
        "main_tex_location": "abstract, experiments, limitations, and appendix benchmark pilots",
        "required_anchors": [
            "not an optimizer-level performance claim",
            "negative benchmark boundary",
            "not evidence for an optimizer-performance advantage",
            "not a complete long-tailed classification optimizer benchmark",
            "partial tuned-validation observations are progress accounting only",
            "selection rule, launch order, and final seed quarantine remain frozen",
        ],
        "trace_status": "present_as_blocked_protocol_context",
    },
    "TCD-6-artifact-reproducibility": {
        "main_tex_location": "limitations and submission reproducibility audit",
        "required_anchors": [
            "server validation and rendered-PDF fallback",
            "preferred LaTeX clean-checkout reproduction remains an explicit gate",
            "supportable_with_toolchain_caveat",
        ],
        "trace_status": "present_as_toolchain_caveated_reproducibility",
    },
}


def line_for(text: str, phrase: str) -> int:
    index = text.find(phrase)
    if index < 0:
        return -1
    return text[:index].count("\n") + 1


def normalize_anchor(text: str, claim_id: str, anchor: str, decision: str) -> str:
    if anchor == "supportable_diagnostic_only":
        return decision
    if anchor == "supportable_with_toolchain_caveat":
        return decision
    return text


def main() -> None:
    main_text = MAIN_TEX.read_text(encoding="utf-8")
    decisions = pd.read_csv(CLAIM_DECISIONS).set_index("claim_id")
    rows: list[dict[str, object]] = []
    blocked_rows: list[dict[str, object]] = []

    for claim_id, spec in CLAIM_ANCHORS.items():
        decision_row = decisions.loc[claim_id]
        decision = str(decision_row["current_decision"])
        required_anchors = list(spec["required_anchors"])
        missing: list[str] = []
        found_lines: list[str] = []
        for anchor in required_anchors:
            haystack = normalize_anchor(main_text, claim_id, anchor, decision)
            line = line_for(haystack, anchor)
            if line < 0:
                missing.append(anchor)
            else:
                found_lines.append(f"{anchor}@{line if haystack is main_text else 'decision'}")
        blocked_phrases = [
            phrase.strip()
            for phrase in str(decision_row["author_blocked_wording"]).split(";")
            if phrase.strip()
        ]
        present_blocked = [phrase for phrase in blocked_phrases if phrase in main_text]
        for phrase in blocked_phrases:
            blocked_rows.append(
                {
                    "claim_id": claim_id,
                    "blocked_phrase": phrase,
                    "present_in_main_tex": "yes" if phrase in main_text else "no",
                    "audit_status": "fail" if phrase in main_text else "pass",
                }
            )
        rows.append(
            {
                "claim_id": claim_id,
                "current_decision": decision,
                "main_tex_location": spec["main_tex_location"],
                "trace_status": spec["trace_status"],
                "required_anchor_count": len(required_anchors),
                "found_anchor_count": len(required_anchors) - len(missing),
                "anchor_lines": "; ".join(found_lines),
                "missing_anchors": "; ".join(missing) if missing else "none",
                "blocked_phrase_audit": "pass" if not present_blocked else "fail",
                "blocked_phrase_hits": "; ".join(present_blocked) if present_blocked else "none",
            }
        )

    claim_trace = pd.DataFrame(rows)
    blocked_audit = pd.DataFrame(blocked_rows)
    if claim_trace["missing_anchors"].ne("none").any():
        missing = claim_trace[claim_trace["missing_anchors"].ne("none")]
        raise AssertionError(f"manuscript claim trace missing anchors: {missing.to_dict('records')}")
    if blocked_audit["audit_status"].ne("pass").any():
        hits = blocked_audit[blocked_audit["audit_status"].ne("pass")]
        raise AssertionError(f"main.tex contains blocked claim wording: {hits.to_dict('records')}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    claim_trace.to_csv(OUTPUT_DIR / "claim_trace.csv", index=False)
    blocked_audit.to_csv(OUTPUT_DIR / "blocked_phrase_audit.csv", index=False)
    config = {
        "main_tex": str(MAIN_TEX),
        "claim_decision_source": str(CLAIM_DECISIONS),
        "claim_count": int(len(claim_trace)),
        "blocked_phrase_checks": int(len(blocked_audit)),
        "all_anchors_present": bool(claim_trace["missing_anchors"].eq("none").all()),
        "blocked_phrases_absent": bool(blocked_audit["audit_status"].eq("pass").all()),
    }
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    text = f"""# E11 Manuscript Claim Trace

This generated trace links the paper-level top-conference claim decisions to explicit `main.tex` anchors. It does not add empirical evidence; it checks that the manuscript presents supportable claims as scoped claims, completed negative boundaries as negative boundaries, and blocked claims only as limitations or protocols.

## Claim Trace

{markdown_table(claim_trace, ["claim_id", "current_decision", "main_tex_location", "trace_status", "required_anchor_count", "found_anchor_count", "anchor_lines", "blocked_phrase_audit"])}

## Blocked Phrase Audit

{markdown_table(blocked_audit, ["claim_id", "blocked_phrase", "present_in_main_tex", "audit_status"])}

## Operating Rule

Every `supportable` decision must have a local scoped manuscript anchor. Every completed-negative-boundary or `blocked` decision must have incomplete, negative-boundary, protocol, or caveat wording in `main.tex`, and the blocked positive wording from the top-conference claim decision matrix must be absent.

Artifacts:
- [claim_trace.csv](../results/e11_manuscript_claim_trace/claim_trace.csv)
- [blocked_phrase_audit.csv](../results/e11_manuscript_claim_trace/blocked_phrase_audit.csv)
- [config.json](../results/e11_manuscript_claim_trace/config.json)
"""
    write_markdown(DISCUSSION_PATH, text)
    print(f"saved manuscript claim trace to {DISCUSSION_PATH} and {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
