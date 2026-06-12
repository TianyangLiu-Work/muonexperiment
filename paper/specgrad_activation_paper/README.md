# 谱梯度更新的激活扰动几何（中文 LaTeX 论文草稿）

这是一个中文 LaTeX 论文项目，主题是：

> 将 SpecGrad / Muon-like 极分解更新解释为 activation perturbation geometry，而不是简单的“高秩更新更好”。

## 文件树

```text
specgrad_activation_paper/
├── main.tex
├── references.bib
├── Makefile
├── README.md
├── sections/
│   ├── 00_abstract.tex
│   ├── 01_introduction.tex
│   ├── 02_related_work.tex
│   ├── 03_setup.tex
│   ├── 04_local_geometry.tex
│   ├── 05_activation_perturbation.tex
│   ├── 06_downstream_stable_rank.tex
│   ├── 07_existing_e11_evidence.tex
│   ├── 08_experiments_tbd.tex
│   ├── 09_discussion.tex
│   └── 10_conclusion.tex
├── appendices/
│   ├── A_proofs.tex
│   ├── B_metrics_protocol.tex
│   └── C_claims_and_caveats.tex
├── tables/
│   ├── e11_core_results.tex
│   └── e11_neural_negative_controls.tex
├── figures/
│   └── .gitkeep
└── notes/
    └── experiment_evaluation.md
```

## 编译

建议使用 XeLaTeX，并需要安装中文 LaTeX 支持（例如 TeX Live 的
`ctex` / `xeCJK` 宏包）：

```bash
make
```

或手动编译：

```bash
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

## 当前状态

- 已写入完整中文论文结构。
- 已包含核心数学推导：Frobenius/operator-norm 一阶最优性、activation perturbation bound、sandwiched stable rank。
- 已整理现有 E11 局部几何证据。
- 已加入 direct activation perturbation diagnostics 的初步结果。
- 未做的 downstream Jacobian、长尾小 batch 与现代架构实验已在 `sections/08_experiments_tbd.tex` 中以 TBD 形式保留。
- `notes/experiment_evaluation.md` 评估了哪些实验必须先做，才能把本文从理论框架推进到项目论文。

## 安全表述

本文当前支持：

> SpecGrad / Muon-like 更新是 update-spectrum shaping 方法，其局部收益依赖任务、层、activation 与 downstream geometry。

本文当前不支持：

> Muon/SpecGrad 一般更稳定或一般优于 Adam。
