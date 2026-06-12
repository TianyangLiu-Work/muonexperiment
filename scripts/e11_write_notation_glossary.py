from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import markdown_table, write_markdown


OUTPUT_PATH = Path("discussion/e11_notation_glossary.md")


def main() -> None:
    symbols = pd.DataFrame(
        [
            {
                "symbol": r"\(W_i\)",
                "definition": r"The trainable matrix parameter at layer or factor \(i\).",
                "scope": "All matrix-parameter problems.",
            },
            {
                "symbol": r"\(G_i = \nabla_{W_i} L\)",
                "definition": r"The gradient matrix of the loss with respect to \(W_i\) at the current step.",
                "scope": "All diagnostics and one-step analyses.",
            },
            {
                "symbol": r"\(\Delta W_i\)",
                "definition": r"The signed parameter change \(W_i^+ - W_i\) after one optimizer step.",
                "scope": "Adam, Muon, stateless controls, and update-spectrum diagnostics.",
            },
            {
                "symbol": r"\(D_i=-\Delta W_i=W_i-W_i^+\)",
                "definition": r"The positive descent update convention used by `update_grad_inner` and update-spectrum diagnostics.",
                "scope": "One-step decrease calibration, boundary comparisons, and theory note.",
            },
            {
                "symbol": r"\(\langle G_i, D_i\rangle\)",
                "definition": r"The positive first-order decrease proxy for layer \(i\), equal to \(-\langle G_i,\Delta W_i\rangle\).",
                "scope": "One-step decrease calibration and boundary comparisons.",
            },
            {
                "symbol": r"\(\Delta L\)",
                "definition": "Observed one-step loss decrease, recorded as loss before the optimizer step minus loss after that step on the same training batch.",
                "scope": "Step metrics and first-order calibration.",
            },
            {
                "symbol": r"\(nr(M)=\|M\|_*^2/\|M\|_F^2\)",
                "definition": "Numerical rank / nuclear-rank diagnostic used for gradients and updates.",
                "scope": "`nrG`, `nrUpdate`, and normalized rank-fraction plots.",
            },
            {
                "symbol": r"\(sr(M)=\|M\|_F^2/\|M\|_{op}^2\)",
                "definition": "Stable-rank diagnostic used for activation/product matrices and updates.",
                "scope": "`stA`, `stUpdate`, and condition-score diagnostics.",
            },
            {
                "symbol": r"\(A_i\)",
                "definition": r"Problem-specific activation/product matrix paired with \(G_i\). In MF-with-input, \(A_i=W_{i+1}\cdots W_{10}Z\), with \(A_{10}=Z\). For MLPs, \(A_i\) is computed from the full dataset even when training uses a smaller mini-batch.",
                "scope": "Strict activation-product interpretation only in MF-with-input; other tasks use problem-specific full-diagnostic spectral quantities.",
            },
            {
                "symbol": r"\(C_i=nr(G_i)/sr(A_i)\)",
                "definition": "Layerwise condition score used in the original condition-geometry plots.",
                "scope": "Condition-score diagnostics, not the final main mechanism claim by itself.",
            },
            {
                "symbol": "`nrUpdate`, `stUpdate`",
                "definition": r"Numerical-rank and stable-rank diagnostics computed from the singular values of \(D_i\), equivalently \(\Delta W_i\).",
                "scope": "Main update-spectrum signature.",
            },
            {
                "symbol": r"\(D_F^\star=rG/\|G\|_F\)",
                "definition": "The locally optimal positive descent update under a Frobenius update budget.",
                "scope": "Theory note and spectral-allocation probe.",
            },
            {
                "symbol": r"\(D_{op}^\star=\eta U_qV_q^\top\)",
                "definition": "A locally optimal positive descent update under an operator-norm update budget.",
                "scope": "Theory note and spectral-allocation probe.",
            },
        ]
    )

    diagnostics = pd.DataFrame(
        [
            {
                "quantity": "`loss`",
                "meaning": "Objective value at the recorded step.",
                "interpretation_rule": "Lower is better for optimization, but loss alone does not identify the mechanism.",
            },
            {
                "quantity": "`recovery_error`",
                "meaning": "Problem-specific normalized reconstruction/recovery error when defined.",
                "interpretation_rule": "Use as performance context, not as the primary geometry metric.",
            },
            {
                "quantity": "`update_grad_inner`",
                "meaning": r"Positive first-order decrease proxy, equal to \(\sum_i \langle G_i,D_i\rangle=-\sum_i \langle G_i,\Delta W_i\rangle\).",
                "interpretation_rule": "At matched update size, larger values mean better local one-step alignment.",
            },
            {
                "quantity": "`delta_loss`",
                "meaning": "Observed one-step loss decrease on the same training batch used for the gradient/update.",
                "interpretation_rule": "Compare with `update_grad_inner` to test local first-order calibration.",
            },
            {
                "quantity": "`condition_score`",
                "meaning": r"Aggregated form of \(nr(G_i)/sr(A_i)\).",
                "interpretation_rule": "Useful exploratory geometry, but not sufficient to claim progress without loss/decrease evidence.",
            },
        ]
    )

    text = f"""# E11 Notation Glossary

This generated glossary fixes the notation used across the E11 paper-facing artifacts. It should be treated as the source of truth for symbols in figure captions, discussion text, and the final report.

## Symbols

{markdown_table(symbols, ["symbol", "definition", "scope"])}

## Recorded Diagnostics

{markdown_table(diagnostics, ["quantity", "meaning", "interpretation_rule"])}

## Scope Notes

1. The strict activation-product definition of \\(A_i\\) applies to MF-with-input. Matrix Sensing and MLP diagnostics still report an \\(A_i\\)-like spectral quantity, but those are task-specific diagnostics rather than the same theoretical object.
2. The default core E11 settings use noisy mini-batch optimization (`train_batch_size < num_samples`, `noise_std > 0`) while activation diagnostics use the full sampled problem instance.
3. `delta_loss` is a same-batch pre/post-update quantity, not the difference between losses evaluated on two independently sampled training batches.
4. The main paper claim is about update-spectrum shaping through the descent update \\(D_i\\), not about condition score alone.
5. Higher \\(nr\\) or \\(sr\\) should not be read as better optimization without the one-step decrease or performance evidence tied to it.
"""
    write_markdown(OUTPUT_PATH, text)
    print(f"saved notation glossary to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
