# E11 Activation Perturbation Diagnostics

This note reports the first direct activation-perturbation diagnostic for the SpecGrad activation-geometry paper draft.

## Definition

For each matrix layer with pre-update parameter `W_i`, actual descent update `D_i = W_i^t - W_i^(t+1)`, and full diagnostic activation matrix `A_i`, the direct pre-activation perturbation is

```text
D_i A_i
```

For neural tasks, optimization still uses the configured mini-batch, while `A_i` is computed on the full sampled dataset for the problem instance. Matrix Sensing is excluded from this diagnostic because its `A` is a measurement-operator proxy rather than a layer activation matrix.

## Main Finding

At matched global update size, Muon has lower operator-norm and worst-case per-sample activation perturbation than Adam, but it does not uniformly reduce batch-Frobenius activation movement.

- Equal-update relative Frobenius perturbation, all activation-defined tasks: `1.142 [1.13, 1.153]`.
- Equal-update relative operator perturbation, all activation-defined tasks: `0.839 [0.831, 0.8471]`.
- Equal-update max per-sample relative perturbation, all activation-defined tasks: `0.7279 [0.7198, 0.7361]`.
- Equal-update mean per-sample relative perturbation, all activation-defined tasks: `1.206 [1.19, 1.222]`.

The family split matters:

- MF-with-input relative Frobenius perturbation: `1.163 [1.152, 1.175]`.
- Small MLP relative Frobenius perturbation: `0.714 [0.6767, 0.7532]`.

Raw, unmatched runs show lower Muon activation movement overall (`0.5244 [0.5178, 0.5311]` for relative Frobenius perturbation), but that comparison is confounded by update-size differences. The equal-update control is the paper-facing evidence.

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
