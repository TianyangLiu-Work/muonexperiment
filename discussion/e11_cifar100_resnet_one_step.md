# E11 CIFAR-100-LT ResNet18 One-Step Diagnostic

This generated probe repeats the matched-head-gain head-to-tail diagnostic on
CIFAR-100-LT using a ResNet18 architecture with a CIFAR-style stem. The
checkpoint is warmed up on the imbalanced train split, then BatchNorm is fixed
with `eval()` for the one-step diagnostic. The intervention updates only
Conv/Linear matrix weights: Conv kernels are flattened to
`out_channels x (in_channels * kernel_height * kernel_width)` for the
spectral/polar direction; BatchNorm and bias parameters are left unchanged.

- Seeds: 3
- Head classes: 0--49 (50 classes)
- Tail classes: 50--99 (50 classes)
- Head train examples per class: 300
- Tail train examples per class: 30
- Tail eval examples per class: 40 from CIFAR-100 test split
- Warmup steps: 1000
- Warmup/head batch sizes: 256/256
- AdamW lr/weight decay: 0.0003/0.0001
- Dtype/device request: float32/cuda
- Target head first-order gain: 0.005 * head loss

![ResNet18 CIFAR-100-LT tail response](../figures/e11_cifar100_resnet_one_step/cifar100_resnet_one_step_tail_response.png)

| quantity | value |
|---|---:|
| squared tail-example logit drift ratio, spectral/Fro | 0.5369 [0.3943, 0.7312] |
| centered-logit squared drift ratio, spectral/Fro | 0.5386 [0.3944, 0.7353] |
| margin-delta squared ratio, spectral/Fro | 0.5086 [0.374, 0.6917] |
| spectral lower squared tail-example logit drift fraction | 1 |
| tail loss increase diff, spectral - Fro | -0.0006038 [-0.000737, -0.0004706] |
| tail margin drop diff, spectral - Fro | -0.000793 [-0.0009722, -0.0006137] |
| tail accuracy drop diff, spectral - Fro | -0.0003333 [-0.001259, 0.000592] |
| actual head-gain relative error, Fro/GD | 0.06523 [-0.02923, 0.1597] |
| actual head-gain relative error, spectral | 0.08741 [0.04108, 0.1337] |
| tail CE before | 4.734 [4.618, 4.849] |
| tail accuracy before | 0.06867 [0.06585, 0.07148] |
| mean matrix gradient nuclear rank | 35.57 |

Interpretation should remain conservative. This is a larger architecture
diagnostic for local function drift, not a full practical Muon optimizer
benchmark. It is stronger than the two-layer MLP check because it tests the
same matched-gain readout with Conv blocks and BatchNorm state fixed during
measurement.

Artifacts:
- [step_metrics.csv](../results/e11_cifar100_resnet_one_step/step_metrics.csv)
- [pair_summary.csv](../results/e11_cifar100_resnet_one_step/pair_summary.csv)
- [layer_metrics.csv](../results/e11_cifar100_resnet_one_step/layer_metrics.csv)
- [config.json](../results/e11_cifar100_resnet_one_step/config.json)
