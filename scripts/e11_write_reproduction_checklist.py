from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.artifacts import APPENDIX_RUNNER_DESCRIPTIONS, APPENDIX_RUNNER_SCRIPTS, MAIN_EVIDENCE_STAGES
from e11_condition_geometry.reporting import markdown_table, write_markdown


OUTPUT_PATH = Path("discussion/e11_reproduction_checklist.md")


def main() -> None:
    main_sequence = pd.DataFrame([dict(item) for item in MAIN_EVIDENCE_STAGES])

    appendix_sequence = pd.DataFrame(
        [
            {
                "command": f"python3 {script}",
                "why": APPENDIX_RUNNER_DESCRIPTIONS[script],
            }
            for script in APPENDIX_RUNNER_SCRIPTS
        ]
    )

    paper_artifacts = pd.DataFrame(
        [
            {
                "artifact": "discussion/e11_main_paper_package.md",
                "role": "Smallest main figure/table package.",
            },
            {
                "artifact": "discussion/e11_quantitative_claim_ledger.md",
                "role": "Allowed wording, forbidden wording, quantitative anchors, and evidence links.",
            },
            {
                "artifact": "discussion/e11_paper_numbers.tex",
                "role": "LaTeX macros generated from the current result CSVs.",
            },
            {
                "artifact": "discussion/e11_paper_skeleton.md",
                "role": "Current section-level paper skeleton and figure/table plan.",
            },
            {
                "artifact": "discussion/e11_reviewer_risk_audit.md",
                "role": "Known reviewer risks and safe claim decisions.",
            },
        ]
    )

    text = f"""# E11 Reproduction Checklist

This generated checklist separates the minimal main-paper evidence from appendix controls and generated writing assets. It is not a new experiment; it is the current reproducibility map for the E11 paper direction.

## Make Targets

```bash
make e11-main-results      # core trajectories, base figures, equal-update control, and spectral-allocation probe
make e11-appendix-results  # current appendix/guardrail probes
make e11-all-results       # main plus appendix/guardrail result generation
make e11-paper-assets      # generated Markdown/TeX paper assets after results exist
make e11-check             # validation, tests, and whitespace check
make e11-full              # paper assets plus e11-check
```

## Minimal Main-Paper Evidence

{markdown_table(main_sequence, ["stage", "command", "produces", "paper_role"])}

## Appendix / Guardrail Evidence

{markdown_table(appendix_sequence, ["command", "why"])}

## Generated Paper-Facing Assets

After the result CSVs and figures exist, regenerate paper-facing assets with:

```bash
make e11-paper-assets
```

{markdown_table(paper_artifacts, ["artifact", "role"])}

## Batch / Activation Contract

The validation gate enforces the current training-versus-diagnostics contract:

- Non-MLP optimization is full-batch, so `train_batch_size == num_samples`.
- MLP optimization uses a strict mini-batch, so `train_batch_size < num_samples`.
- MLP activation diagnostics use the full sampled dataset and must record `diagnostic_A_definition == full_layer_input_activation`.
- `delta_loss` is the same-batch pre/post-update decrease, evaluated on the training batch used for the gradient/update.

## Validation Gate

Before using the numbers in a draft, run:

```bash
make e11-check
```

The stronger local gate is:

```bash
make e11-full
```

`make e11-full` regenerates discussion/paper-facing artifacts and then runs validation, tests, and whitespace checks. It assumes the longer experiment result CSVs already exist unless their writer script reruns the relevant probe.
"""
    write_markdown(OUTPUT_PATH, text)
    print(f"saved reproduction checklist to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
