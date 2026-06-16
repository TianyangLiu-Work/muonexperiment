# Head-to-Tail Interference Paper Draft

This directory now contains the replacement paper draft:

**Head-to-Tail Interference in Long-Tailed Small-Batch Training**

The paper reframes spectral-gradient and Muon-style spectral updates through a long-tailed small-batch training problem: head-only updates can damage tail predictions before tail samples reappear. The main object is the head-to-tail interference coefficient

```tex
\mathcal I_N(T\mid H)=K_{T,N}^2/\|g_H\|_{N,*}^2.
```

For a sandwich matrix block with tail perturbation `J_T(D)=B_T D A_T`, the draft derives the worst-case sensitivity-bound condition

```tex
\nrank(G_H) > \ssrank(B_T,A_T)
```

as a sufficient condition for spectral geometry to have a smaller tail-drift upper bound than Frobenius geometry under matched head gain. The draft separately reports observed tail drift because actual polar directions also depend on singular-vector alignment with the tail Jacobian.

## Files

```text
specgrad_activation_paper/
├── main.tex
├── main.pdf
├── two_page.tex
├── two_page.pdf
├── iclr2025_conference.sty
├── iclr2025_conference.bst
├── references.bib
├── Makefile
├── README.md
├── figures/
│   ├── head_tail_drift_ratio.png
│   ├── head_tail_alignment_ablation.png
│   ├── long_tail_head_only_forgetting.png
│   ├── long_tail_layerwise_drift.png
│   ├── long_tail_muon_bridge.png
│   ├── long_tail_checkpoint_sweep.png
│   ├── long_tail_class_partition_sweep.png
│   ├── long_tail_imbalance_ablation.png
│   ├── long_tail_one_step_tail_response.png
│   ├── long_tail_practical_muon_bridge.png
│   ├── long_tail_muon_state_source_control.png
│   ├── long_tail_rho_sweep.png
│   ├── cifar100_resnet_layer_jvp_tail_quality.png
│   ├── cifar100_resnet_layer_jvp_checkpoint_prediction.png
│   ├── cifar100_resnet_practical_muon_bridge.png
│   └── long_tail_practical_training.png
├── tables/
│   ├── e11_paper_numbers.tex
│   ├── head_tail_empirical_results.tex
│   └── local_linearization_errors.tex
└── notes/
    └── experiment_evaluation.md
```

The old section/appendix/table source files were removed so that the directory contains only this replacement paper source. The current tables are generated from CSV evidence by `scripts/e11_write_head_tail_paper_results.py` and `scripts/e11_write_local_linearization_table.py`.

The main paper uses the official ICLR 2025 style files,
`iclr2025_conference.sty` and `iclr2025_conference.bst`, with a local
title/header override in `main.tex` that suppresses the anonymous-author block
and the review-status header while keeping the ICLR page geometry. Keep the
official style file unchanged unless the target venue or template requirement
changes. Do not uncomment `\iclrfinalcopy` for submission review builds.

The draft uses paper-local copies of the paper-facing figures so that the paper
directory can be compiled and packaged without reaching into the repo-level
evidence directories. Regenerate those copies with `scripts/e11_write_paper_figures.py`
or the higher-level `make e11-paper-assets` target. The source figures remain:

- `figures/e11_long_tail_one_step/long_tail_one_step_tail_response.png`
- `figures/e11_head_tail_alignment_ablation/head_tail_alignment_ablation.png`
- `figures/e11_long_tail_imbalance_ablation/long_tail_imbalance_ablation.png`
- `figures/e11_long_tail_checkpoint_sweep/long_tail_checkpoint_sweep.png`
- `figures/e11_long_tail_class_partition_sweep/long_tail_class_partition_sweep.png`
- `figures/e11_long_tail_rho_sweep/long_tail_rho_sweep.png`
- `figures/e11_head_tail_interference/head_tail_drift_ratio.png`
- `figures/e11_long_tail_muon_bridge/long_tail_muon_bridge.png`
- `figures/e11_long_tail_practical_muon_bridge/long_tail_practical_muon_bridge.png`
- `figures/e11_long_tail_muon_state_source_control/long_tail_muon_state_source_control.png`
- `figures/e11_long_tail_practical_training/long_tail_practical_training.png`
- `figures/e11_long_tail_forgetting/long_tail_head_only_forgetting.png`
- `figures/e11_long_tail_layerwise/long_tail_layerwise_drift.png`
- `figures/e11_cifar100_resnet_layer_jvp_tail_quality/cifar100_resnet_layer_jvp_tail_quality.png`
- `figures/e11_cifar100_resnet_layer_jvp_checkpoint_prediction/cifar100_resnet_layer_jvp_checkpoint_prediction.png`
- `figures/e11_cifar100_resnet_practical_muon_bridge/cifar100_resnet_practical_muon_bridge.png`

`scripts/e11_validate_outputs.py` checks that the `\includegraphics` paths in
`main.tex` resolve to paper-local files.

## Citation Audit

The bibliography in `references.bib` was checked against primary public
sources. Preprint entries use arXiv IDs and URLs; the submitted Muon
long-tailed paper uses its OpenReview forum URL; the digits dataset cites the
scikit-learn JMLR paper; classical long-tailed recognition entries use CVF,
NeurIPS, or OpenReview pages. Do not list a preprint as conference proceedings
unless the venue page confirms the accepted proceedings metadata.

## Reproduce From Repo Root

From the repository root, use the root Makefile when you want to refresh the
generated table, generated number macros, paper-local figure copies, PDF, and
validation gates together:

```bash
make e11-paper-assets
make e11-paper-pdf
make e11-check
```

For the full current paper artifact path in one command:

```bash
make e11-full
```

`make e11-paper-assets` regenerates `tables/e11_paper_numbers.tex`,
`tables/head_tail_empirical_results.tex`, `tables/local_linearization_errors.tex`,
and the paper-local `figures/*.png`
copies from the current CSV evidence. `make e11-check` then validates that the
paper-local figures exist, the generated number macros are used by `main.tex`,
the bibliography is synchronized, and the E11 test suite passes.

## Local Build

Use the local Makefile to build both the main paper and the two-page report:

```bash
make
```

To build only the main paper, run:

```bash
make main
```

To build only the two-page report, run:

```bash
make two-page
```

or run the ICLR-style LaTeX/BibTeX sequence manually:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
