# Muon (μ) 优化算法实验设计的数学基础与统计形式化

> 版本：v2.0  
> 日期：2026年4月  
> 说明：本文档为 Muon vs SGD 矩阵优化对比实验建立完整的数学基础、统计变量体系和形式化实验设计，并补充现有方案中缺失的实验维度。

[TOC]

---

# 第一部分：数学基础

> 本部分为 Muon 优化算法与 SGD 的对比实验建立严格的数学基础框架。涵盖优化理论、矩阵分析、统计推断和计算复杂度四大支柱。所有符号体系统一，定义自洽，为后续实验设计与结果分析奠定理论根基。

> 本章为 Muon 优化算法与 SGD 的对比实验建立严格的数学基础框架。所有符号体系统一，定义自洽，为后续实验设计与结果分析奠定理论根基。

---

## 1. 优化理论基础

### 1.1 收敛性的精确定义

在数值优化中，收敛性是评价算法行为的核心概念。设 $\{X^{(k)}\}_{k \geq 0}$ 为迭代算法在 $\mathbb{R}^{d \times d}$ 上生成的迭代序列，$X^\star$ 为目标点（可为全局最小值点或驻点）。定义**误差向量** $E^{(k)} := X^{(k)} - X^\star$，以及**函数值差距** $\delta_k := f(X^{(k)}) - f^\star$，其中 $f^\star := f(X^\star)$ 为最优值。

#### 1.1.1 点收敛（Pointwise Convergence）

**定义 1.1**（点收敛）. 迭代序列 $\{X^{(k)}\}$ 称为**收敛于** $X^\star$，若
$$
\lim_{k \to \infty} \|X^{(k)} - X^\star\| = 0,
$$
其中 $\|\cdot\|$ 为某个矩阵范数（通常取 Frobenius 范数 $\|\cdot\|_F$ 或谱范数 $\|\cdot\|_2$）。等价地，对任意 $\epsilon > 0$，存在 $K(\epsilon) < \infty$，使得对所有 $k \geq K(\epsilon)$，有 $\|X^{(k)} - X^\star\| \leq \epsilon$。

**物理解释**：点收敛是优化算法最基本的收敛要求，意味着随着迭代推进，迭代点可以任意接近真实解。在实验层面，这保证了算法不会发散，但并未说明需要多少迭代才能达到给定精度。

#### 1.1.2 收敛速率（Rate of Convergence）

**定义 1.2**（$Q$-收敛速率）. 设 $\{a_k\}_{k \geq 0}$ 为非负标量序列（例如 $a_k = \|X^{(k)} - X^\star\|_F$ 或 $a_k = \delta_k$）。

- **线性收敛**（Linear / Geometric Convergence）：存在常数 $q \in (0, 1)$ 和 $C > 0$，使得
  $$
  a_k \leq C \cdot q^k, \quad \forall k \geq 0.
  $$
  等价地，$\limsup_{k \to \infty} \frac{a_{k+1}}{a_k} = \rho < 1$，其中 $\rho$ 称为**收敛比率**。

- **次线性收敛**（Sublinear Convergence）：存在常数 $C > 0$ 和 $p > 0$，使得
  $$
  a_k \leq C \cdot k^{-p}, \quad \forall k \geq 1.
  $$
  典型例子为梯度下降在光滑凸函数上的 $O(1/k)$ 收敛，对应 $p = 1$。

- **超线性收敛**（Superlinear Convergence）：
  $$
  \lim_{k \to \infty} \frac{a_{k+1}}{a_k} = 0.
  $$

- **$r$-阶收敛**（Order-$r$ Convergence）：对 $r > 1$，存在常数 $q \in (0, 1)$ 使得
  $$
  a_{k+1} \leq q \cdot (a_k)^r.
  $$
  当 $r = 2$ 时称为**二次收敛**（Quadratic Convergence）。

**实验含义**：收敛速率直接决定了算法的**迭代效率** $K_\epsilon$。线性收敛意味着每迭代一次误差减少固定比例，适合高精度求解；次线性收敛初期进展快但后期变慢，适合低精度场景。在 Muon 与 SGD 的对比中，收敛速率是核心竞争指标。

#### 1.1.3 $R$-收敛速率

**定义 1.3**（$R$-收敛速率）. 序列 $\{a_k\}$ 称为以**$R$-线性速率**收敛，若存在另一个序列 $\{b_k\}$ 使得 $b_k \to 0$ 以 $Q$-线性速率收敛，且对所有充分大的 $k$ 有 $a_k \leq b_k$。

**注**：$Q$-收敛直接比较相邻迭代步的误差比值；$R$-收敛则允许更宽松的比较。在随机优化中（如 SGD），由于随机性导致相邻步比值波动较大，通常更适合使用 $R$-收敛或期望意义上的收敛分析。

### 1.2 迭代复杂度与查询复杂度

#### 1.2.1 迭代复杂度

**定义 1.4**（$\epsilon$-精度迭代复杂度）. 设 $\delta_k := f(X^{(k)}) - f^\star$ 为第 $k$ 步的优化差距。给定精度参数 $\epsilon > 0$，**迭代复杂度**定义为
$$
K_\epsilon := \min\{k \geq 0 : \delta_k \leq \epsilon\}.
$$
若关注参数距离，则定义
$$
K_\epsilon^{\text{param}} := \min\{k \geq 0 : \|X^{(k)} - X^\star\|_F \leq \epsilon\}.
$$

**定义 1.5**（渐近迭代复杂度）. 若存在函数 $\mathcal{T}(\epsilon; d, \kappa, L, \mu)$ 使得 $K_\epsilon \leq \mathcal{T}(\epsilon)$ 对所有充分小的 $\epsilon > 0$ 成立，则称 $\mathcal{T}(\epsilon)$ 为算法的**渐近迭代复杂度**。

常见情形：
- 次线性收敛（凸、光滑、$L$-梯度 Lipschitz）：$\mathcal{T}(\epsilon) = O\left(\frac{L}{\epsilon}\right)$ 或 $O\left(\frac{L}{\mu} \log\frac{1}{\epsilon}\right)$
- 线性收敛（强凸、光滑）：$\mathcal{T}(\epsilon) = O\left(\frac{L}{\mu} \log\frac{1}{\epsilon}\right) = O(\kappa \log(1/\epsilon))$，其中 $\kappa = L/\mu$ 为条件数
- 二次收敛（牛顿法，接近解时）：$\mathcal{T}(\epsilon) = O(\log\log(1/\epsilon))$

**实验含义**：迭代复杂度 $K_\epsilon$ 是实验中最直接可观测的指标。在相同精度 $\epsilon$ 下比较 Muon 与 SGD 的 $K_\epsilon$，可直接判定哪种算法迭代效率更高。

#### 1.2.2 查询复杂度

**定义 1.6**（查询复杂度）. 设每次迭代需要 $b_k$ 次函数值或梯度查询（例如小批量 SGD 中 $b_k = |\mathcal{B}_k|$ 为批量大小）。**总查询复杂度**为
$$
Q_\epsilon := \sum_{k=1}^{K_\epsilon} b_k.
$$
对于全批量梯度下降（GD），$b_k \equiv m$（总样本数），故 $Q_\epsilon^{\text{GD}} = m \cdot K_\epsilon$。

**定义 1.7**（矩阵查询复杂度）. 对于矩阵感知问题，每次迭代需要计算梯度 $G^{(k)} = \frac{1}{m}\mathcal{A}^*(\mathcal{A}(X^{(k)}) - y)$，涉及 $m$ 个 $d \times d$ 测量矩阵的乘法。每次梯度计算的**测量查询量**为 $m$ 次矩阵内积运算。

**实验含义**：在全批量设置下，查询复杂度与迭代复杂度成正比。但在随机或小批量设置中，两者的差异可能显著。本实验在全批量下进行，因此 $Q_\epsilon$ 与 $K_\epsilon$ 仅相差常数倍 $m$。

### 1.3 一阶优化方法的理论框架

#### 1.3.1 一阶方法的通用形式

**定义 1.8**（一阶迭代方法）. 一阶优化方法是指仅利用函数值 $f(X)$ 和一阶信息（梯度 $\nabla f(X)$ 或次梯度）进行更新的算法。其通用更新形式为
$$
X^{(k+1)} = X^{(k)} - \eta_k \cdot D^{(k)},
$$
其中 $\eta_k > 0$ 为步长（学习率），$D^{(k)}$ 为**搜索方向**，通常满足 $D^{(k)} \approx \nabla f(X^{(k)})$。

在本实验的两种算法中：
- **SGD / GD**：$D^{(k)}_{\text{SGD}} = G^{(k)} = \nabla f(X^{(k)})$
- **Muon**：$D^{(k)}_{\text{Muon}} = U^{(k)} \cdot V^{(k)\top}$，其中 $G^{(k)} = U^{(k)} \cdot \text{diag}(\sigma_1, \ldots, \sigma_d) \cdot V^{(k)\top}$ 为 SVD 分解

#### 1.3.2 梯度下降的理论保证

**定理 1.1**（GD 在光滑凸函数上的收敛）. 设 $f: \mathbb{R}^{d \times d} \to \mathbb{R}$ 为凸函数且 $L$-光滑（$L$-gradient Lipschitz），即对所有 $X, Y$：
$$
f(Y) \leq f(X) + \langle \nabla f(X), Y - X \rangle + \frac{L}{2}\|Y - X\|_F^2.
$$
取固定步长 $\eta = 1/L$，则 GD 迭代满足
$$
\delta_k = f(X^{(k)}) - f^\star \leq \frac{2L\|X^{(0)} - X^\star\|_F^2}{k + 4} = O\left(\frac{1}{k}\right).
$$

**定理 1.2**（GD 在强凸光滑函数上的收敛）. 设 $f$ 为 $\mu$-强凸且 $L$-光滑，则取 $\eta = 1/L$ 时
$$
\delta_k \leq \left(1 - \frac{1}{\kappa}\right)^k \delta_0, \quad \text{其中 } \kappa := \frac{L}{\mu} \text{ 为条件数}.
$$
等价地，$K_\epsilon \leq \kappa \cdot \log(\delta_0 / \epsilon)$。

**物理解释**：强凸性保证了函数"碗形"足够陡峭，使得梯度方向能高效指向全局最小值。条件数 $\kappa$ 衡量了函数"椭圆等高线"的偏心程度——$\kappa$ 越大，问题越病态，收敛越慢。

#### 1.3.3 一阶下界理论

**定理 1.3**（Nesterov 一阶下界）. 对于 $L$-光滑且 $\mu$-强凸的函数类 $\mathcal{F}_{L,\mu}$，任何仅使用一阶信息的黑盒算法，其迭代复杂度满足下界
$$
K_\epsilon \geq \Omega\left(\sqrt{\kappa} \log\frac{1}{\epsilon}\right).
$$
Nesterov 加速梯度方法（AGD）恰好达到此下界。

**实验含义**：SGD/GD 的迭代复杂度上界为 $O(\kappa \log(1/\epsilon))$，与下界之间存在 $\sqrt{\kappa}$ 的间隙。Muon 通过矩阵谱归一化利用问题的低秩结构，可能绕过此下界，在某些问题类上实现超线性或近最优的收敛。

### 1.4 凸与非凸问题的收敛条件

#### 1.4.1 凸性假设

**定义 1.9**（凸函数）. 函数 $f: \mathcal{X} \to \mathbb{R}$ 称为**凸函数**，若对所有 $X, Y \in \mathcal{X}$ 和 $\lambda \in [0, 1]$：
$$
f(\lambda X + (1-\lambda)Y) \leq \lambda f(X) + (1-\lambda)f(Y).
$$

**定义 1.10**（强凸性）. 函数 $f$ 称为**$\mu$-强凸**（$\mu > 0$），若
$$
f(Y) \geq f(X) + \langle \nabla f(X), Y - X \rangle + \frac{\mu}{2}\|Y - X\|_F^2, \quad \forall X, Y.
$$

#### 1.4.2 非凸性与驻点

**定义 1.11**（驻点）. 点 $X$ 称为 $f$ 的**驻点**（Stationary Point），若 $\nabla f(X) = 0$。在带正则化项的情形下，点 $X$ 满足
$$
\|\nabla f(X) + \lambda X\|_F \leq \epsilon
$$
时称为 **$\epsilon$-近似驻点**。

**定义 1.12**（一阶稳定点与二阶稳定点）. 
- **一阶稳定点**（First-Order Stationary Point, FOSP）：$\|\nabla f(X)\|_F = 0$
- **$\epsilon$-FOSP**：$\|\nabla f(X)\|_F \leq \epsilon$
- **二阶稳定点**（Second-Order Stationary Point, SOSP）：满足 FOSP 且 Hessian 矩阵半正定，即对矩阵方向 $D$，有 $\nabla^2 f(X)[D, D] \geq 0$
- **严格鞍点**：满足 FOSP 但 Hessian 存在负特征值的方向

**定义 1.13**（PL-条件 / Polyak-\L{}ojasiewicz 条件）. 函数 $f$ 满足 $\mu$-PL 条件，若
$$
\frac{1}{2}\|\nabla f(X)\|_F^2 \geq \mu \cdot (f(X) - f^\star), \quad \forall X.
$$
PL 条件不要求凸性，但能保证梯度下降的全局线性收敛。

**实验含义**：矩阵感知问题（MS）在 RIP 条件下具有良好性质；矩阵分解问题（MF-L）则是高度非凸的，其优化 landscape 存在大量鞍点。Muon 的谱归一化机制可能有助于逃离鞍点，这是其与 SGD 的关键差异之一。

### 1.5 局部收敛与全局收敛

#### 1.5.1 全局收敛

**定义 1.14**（全局收敛）. 算法称为**全局收敛**，若对任意初始点 $X^{(0)} \in \mathcal{X}$，迭代序列 $\{X^{(k)}\}$ 收敛于某个目标点（通常是全局最小值点或驻点集）。

**定义 1.15**（全局收敛速率）. 若全局收敛且收敛速率界对任意初始点一致成立（可能依赖于初始距离 $\|X^{(0)} - X^\star\|_F$），则称算法具有全局收敛速率保证。

#### 1.5.2 局部收敛

**定义 1.16**（局部收敛）. 算法称为在 $X^\star$ 的邻域 $\mathcal{B}(X^\star, r) = \{X : \|X - X^\star\|_F \leq r\}$ 内**局部收敛**，若对所有 $X^{(0)} \in \mathcal{B}(X^\star, r)$，迭代序列收敛于 $X^\star$。

**定义 1.17**（局部收敛域）. 满足局部收敛条件的最大邻域半径 $r^\star$ 称为**收敛域半径**（Basin of Attraction 的半径）。

**定理 1.4**（局部二次收敛）. 设 $X^\star$ 为 $f$ 的非退化局部极小值点（Hessian 正定），且 $f$ 在 $X^\star$ 附近三次可微。则梯度下降以适当步长从充分接近 $X^\star$ 的点出发时，具有局部二次收敛速率。

**实验含义**：实验中采用随机初始化，因此全局收敛性更为关键。矩阵感知在 RIP 下具有全局收敛保证；深度矩阵分解（$L \geq 2$）的全局收敛仍是开放问题。实验中观察到 Muon 与 SGD 在不同初始化下的表现差异，反映了各自的收敛域特性。

---

## 2. 矩阵分析基础

### 2.1 矩阵范数体系

设 $X \in \mathbb{R}^{d_1 \times d_2}$，$r = \text{rank}(X)$，其奇异值分解为 $X = U \cdot \text{diag}(\sigma_1, \ldots, \sigma_r) \cdot V^\top$，其中 $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$ 为奇异值。

#### 2.1.1 谱范数（Spectral Norm / Operator Norm）

**定义 2.1**（谱范数）. 矩阵 $X$ 的谱范数定义为
$$
\|X\|_2 := \sigma_1(X) = \max_{\|v\|_2 = 1} \|Xv\|_2 = \max_{u, v \neq 0} \frac{u^\top X v}{\|u\|_2 \|v\|_2}.
$$
谱范数也是将 $\ell_2$ 范数单位球映射后的最大拉伸系数。

**性质**：谱范数是**酉不变范数**（Unitarily Invariant Norm），即对任意正交矩阵 $Q_1, Q_2$，有 $\|Q_1 X Q_2^\top\|_2 = \|X\|_2$。

**实验含义**：谱范数 $\|X^{(k)} - X^\star\|_2$ 衡量了矩阵近似解的最大方向误差。在矩阵感知中，谱范数常作为误差度量，因为它对应最坏情况下的方向偏差。

#### 2.1.2 Frobenius 范数

**定义 2.2**（Frobenius 范数）. 矩阵 $X$ 的 Frobenius 范数定义为
$$
\|X\|_F := \sqrt{\sum_{i,j} X_{ij}^2} = \sqrt{\text{tr}(X^\top X)} = \sqrt{\sum_{i=1}^r \sigma_i^2(X)}.
$$

**性质**：Frobenius 范数等价于向量化后的 $\ell_2$ 范数，即 $\|X\|_F = \|\text{vec}(X)\|_2$。它也是酉不变范数，且与谱范数满足关系
$$
\|X\|_2 \leq \|X\|_F \leq \sqrt{r} \cdot \|X\|_2 \leq \sqrt{\min(d_1, d_2)} \cdot \|X\|_2.
$$

**实验含义**：Frobenius 范数是实验中最常用的误差度量，因为它可解释为所有元素平方误差的总和，且与矩阵分解的目标函数直接对应。

#### 2.1.3 核范数（Nuclear Norm / Trace Norm）

**定义 2.3**（核范数）. 矩阵 $X$ 的核范数定义为奇异值之和：
$$
\|X\|_* := \sum_{i=1}^r \sigma_i(X).
$$

**性质**：核范数是 $\|X\|_2$ 的凸包络，且在低秩矩阵恢复中作为秩函数的凸松弛广泛使用。满足
$$
\|X\|_2 \leq \|X\|_F \leq \|X\|_* \leq \sqrt{r} \cdot \|X\|_F.
$$

**实验含义**：在矩阵感知中，核范数最小化是凸松弛方法的标准工具。Muon 的谱归一化更新方向 $D^{(k)} = UV^\top$ 恰好是核范数单位球 $\{X : \|X\|_* \leq 1\}$ 的极点，这揭示了 Muon 与低秩诱导之间的深刻联系。

#### 2.1.4 Schatten $p$-范数

**定义 2.4**（Schatten $p$-范数）. 对 $p \in [1, \infty]$，矩阵 $X$ 的 Schatten $p$-范数定义为
$$
\|X\|_{S_p} := \left(\sum_{i=1}^r \sigma_i^p(X)\right)^{1/p},
$$
并约定 $\|X\|_{S_\infty} = \sigma_1(X) = \|X\|_2$，$\|X\|_{S_1} = \|X\|_*$。

**注**：Schatten $p$-范数构成了从谱范数（$p=\infty$）到 Frobenius 范数（$p=2$）再到核范数（$p=1$）的连续插值族。

#### 2.1.5 矩阵内积与正交投影

**定义 2.5**（矩阵内积）. 矩阵空间上的标准内积为
$$
\langle X, Y \rangle := \text{tr}(X^\top Y) = \sum_{i,j} X_{ij} Y_{ij}.
$$
在此内积下，Frobenius 范数满足 $\|X\|_F = \sqrt{\langle X, X \rangle}$。

### 2.2 奇异值分解与谱归一化

#### 2.2.1 奇异值分解

**定理 2.1**（SVD）. 任意矩阵 $X \in \mathbb{R}^{d_1 \times d_2}$（设 $d_1 \leq d_2$）可分解为
$$
X = U \cdot \Sigma \cdot V^\top = \sum_{i=1}^{d_1} \sigma_i u_i v_i^\top,
$$
其中 $U \in \mathbb{R}^{d_1 \times d_1}$ 和 $V \in \mathbb{R}^{d_2 \times d_2}$ 为正交矩阵，$\Sigma = \text{diag}(\sigma_1, \ldots, \sigma_{d_1}) \in \mathbb{R}^{d_1 \times d_2}$，且 $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_{d_1} \geq 0$。

#### 2.2.2 谱归一化（Spectral Normalization）

**定义 2.6**（谱归一化算子）. 给定梯度矩阵 $G \in \mathbb{R}^{d \times d}$，其 SVD 为 $G = U \cdot \text{diag}(\sigma_1, \ldots, \sigma_d) \cdot V^\top$。谱归一化方向定义为
$$
\mathcal{N}_{\text{spec}}(G) := U \cdot V^\top.
$$
若 $G$ 不满秩，取 $V$ 的前 $r = \text{rank}(G)$ 列与 $U$ 的对应列。

**性质**：
1. $\|\mathcal{N}_{\text{spec}}(G)\|_2 = 1$（谱范数单位化）
2. $\|\mathcal{N}_{\text{spec}}(G)\|_F = \sqrt{r}$，其中 $r = \text{rank}(G)$
3. $\langle G, \mathcal{N}_{\text{spec}}(G) \rangle = \sum_{i=1}^r \sigma_i = \|G\|_*$

**定理 2.2**（谱归一化的最优性）. 对任意秩为 $r$ 的矩阵 $G$，有
$$
\mathcal{N}_{\text{spec}}(G) \in \arg\max_{\|D\|_2 \leq 1} \langle G, D \rangle.
$$
即谱归一化方向是在谱范数单位球内与梯度方向内积最大的方向。

**证明概要**：由 von Neumann 迹不等式，$\langle G, D \rangle \leq \sum_i \sigma_i(G) \sigma_i(D) \leq \sum_i \sigma_i(G) \cdot 1 = \|G\|_*$，等号当 $D = UV^\top$ 时成立。

**物理解释**：谱归一化将梯度矩阵"压平"为各向同性方向（所有奇异值变为 1），这消除了梯度在不同谱方向上的尺度差异。对于低秩结构的梯度，谱归一化特别有效，因为它平等对待所有非零奇异方向。这与 SGD 直接使用原始梯度（各方向按奇异值加权）形成鲜明对比。

#### 2.2.3 Muon 更新的数学解释

Muon 的完整更新规则为
$$
X^{(k+1)} = X^{(k)} - \eta \cdot U^{(k)} V^{(k)\top} - \lambda \cdot X^{(k)} = (1 - \lambda) X^{(k)} - \eta \cdot \mathcal{N}_{\text{spec}}(G^{(k)}).
$$
可改写为
$$
X^{(k+1)} = (1 - \lambda) X^{(k)} - \eta \cdot \frac{G^{(k)}}{\text{``谱算子加权''}},
$$
其中"谱算子加权"被 SVD 截断替代。等价地，Muon 是在谱域中对梯度进行归一化后执行更新。

### 2.3 条件数与问题病态程度

#### 2.3.1 条件数的多重定义

**定义 2.7**（函数条件数）. 对于 $L$-光滑且 $\mu$-强凸的函数 $f$，其**条件数**定义为
$$
\kappa(f) := \frac{L}{\mu} \geq 1.
$$

**定义 2.8**（矩阵谱条件数）. 对正定矩阵 $H$，其**谱条件数**为
$$
\kappa_{\text{cond}}(H) := \frac{\lambda_{\max}(H)}{\lambda_{\min}(H)} = \frac{\sigma_1(H)}{\sigma_{\min}(H)}.
$$
对一般矩阵 $X$，条件数定义为最大与最小非零奇异值之比：
$$
\kappa_{\text{cond}}(X) := \frac{\sigma_1(X)}{\sigma_r(X)}, \quad r = \text{rank}(X).
$$

**定义 2.9**（实验使用的谱比）. 在本实验中，定义**谱比**为
$$
\kappa_{\text{sp}}(X) := \frac{\|X\|_2}{\|X\|_F} = \frac{\sigma_1(X)}{\sqrt{\sum_i \sigma_i^2(X)}} \in \left[\frac{1}{\sqrt{r}}, 1\right].
$$

**物理解释**：谱比衡量了矩阵能量在谱方向的集中程度。当 $\kappa_{\text{sp}}(X) \approx 1$ 时，能量集中于单一方向；当 $\kappa_{\text{sp}}(X) \approx 1/\sqrt{r}$ 时，能量均匀分布。谱归一化对不同谱比的梯度矩阵有不同效果——对能量集中的梯度（高谱比），谱归一化改变更大；对能量均匀分布的梯度，谱归一化与 SGD 差异较小。

#### 2.3.2 问题病态程度对收敛的影响

**定理 2.3**（条件数影响）. 对于二次目标 $f(X) = \frac{1}{2}\langle X - X^\star, H (X - X^\star) \rangle$，其中 $H$ 为 Hessian 矩阵，GD 的收敛比率为
$$
\rho = \frac{\kappa_{\text{cond}}(H) - 1}{\kappa_{\text{cond}}(H) + 1}.
$$
条件数越大，$\rho$ 越接近 1，收敛越慢。

**实验含义**：实验中测量矩阵的性质（如 RIP 常数）直接影响问题的有效条件数。Muon 的谱归一化通过消除梯度在谱方向的各向异性，可能降低有效条件数，从而加速收敛。

### 2.4 低秩矩阵的数学刻画

#### 2.4.1 秩与数值秩

**定义 2.10**（矩阵秩）. $\text{rank}(X) := \dim(\text{range}(X)) = r$，即非零奇异值的个数。

**定义 2.11**（数值$\epsilon$-秩）. 给定阈值 $\epsilon > 0$，矩阵 $X$ 的**数值$\epsilon$-秩**定义为
$$
\text{rank}_\epsilon(X) := \#\{i : \sigma_i(X) > \epsilon\}.
$$

**定义 2.12**（有效秩）. 矩阵的有效秩（Effective Rank）定义为
$$
r_{\text{eff}}(X) := \frac{\|X\|_F^2}{\|X\|_2^2} = \frac{\sum_i \sigma_i^2}{\sigma_1^2} = \sum_i \left(\frac{\sigma_i}{\sigma_1}\right)^2 \in [1, r].
$$

**实验含义**：实验中矩阵感知的目标 $X^\star$ 通常具有精确低秩（$r \ll d$），而矩阵分解的乘积 $W_L \cdots W_1$ 也在低秩流形上演化。有效秩 $r_{\text{eff}}$ 提供了比硬阈值更平滑的低秩度量，适合追踪优化轨迹的隐式低秩性。

#### 2.4.2 低秩矩阵流形

**定义 2.13**（秩-$r$ 矩阵流形）. 秩不超过 $r$ 的矩阵集合
$$
\mathcal{M}_r := \{X \in \mathbb{R}^{d \times d} : \text{rank}(X) \leq r\}
$$
构成一个代数簇（Algebraic Variety）。在秩恰好为 $r$ 的点处，$\mathcal{M}_r$ 是光滑流形，其维数为 $r(2d - r)$。

**定义 2.14**（切空间）. 在点 $X = U \Sigma V^\top \in \mathcal{M}_r$（$\Sigma \in \mathbb{R}^{r \times r}$），切空间为
$$
\mathcal{T}_X \mathcal{M}_r = \{U A^\top + B V^\top : A \in \mathbb{R}^{d \times r}, B \in \mathbb{R}^{d \times r}\}.
$$

**实验含义**：优化轨迹 $X^{(k)}$ 若始终位于或接近 $\mathcal{M}_r$，则隐式利用了低秩结构。Muon 的谱归一化可能更自然地保持在此流形上，而 SGD 可能偏离低秩结构。

### 2.5 矩阵感知的受限等距性质（RIP）

#### 2.5.1 线性测量算子

**定义 2.15**（测量算子）. 矩阵感知中的线性测量算子 $\mathcal{A}: \mathbb{R}^{d \times d} \to \mathbb{R}^m$ 定义为
$$
[\mathcal{A}(X)]_i := \langle A_i, X \rangle = \text{tr}(A_i^\top X), \quad i = 1, \ldots, m,
$$
其中 $\{A_i\}_{i=1}^m$ 为测量矩阵。其伴随算子 $\mathcal{A}^*: \mathbb{R}^m \to \mathbb{R}^{d \times d}$ 为
$$
\mathcal{A}^*(y) = \sum_{i=1}^m y_i A_i.
$$

观测模型为 $y = \mathcal{A}(X^\star) + \epsilon$，其中 $\epsilon \in \mathbb{R}^m$ 为噪声向量。

#### 2.5.2 RIP 定义

**定义 2.16**（受限等距性质，RIP）. 算子 $\mathcal{A}$ 称为满足**秩-$r$ 受限等距性质**（Rank-$r$ RIP），常数为 $\delta_r \in [0, 1)$，若对所有秩不超过 $r$ 的矩阵 $X$：
$$
(1 - \delta_r) \|X\|_F^2 \leq \frac{1}{m}\|\mathcal{A}(X)\|_2^2 \leq (1 + \delta_r) \|X\|_F^2.
$$
等价地，定义**受限等距常数**（RIC）为
$$
\delta_r(\mathcal{A}) := \sup_{\text{rank}(X) \leq r} \left| \frac{1}{m}\|\mathcal{A}(X)\|_2^2 - \|X\|_F^2 \right| / \|X\|_F^2.
$$

**定理 2.4**（高斯测量矩阵的 RIP）. 若 $\{A_i\}$ 的每个元素独立服从 $\mathcal{N}(0, 1)$ 分布，且 $m \geq C \cdot r d$（$C$ 为绝对常数），则以高概率（至少 $1 - \exp(-cm)$），$\mathcal{A}$ 满足秩-$r$ RIP，其中 $\delta_r \leq \delta$ 对某个预设小常数 $\delta$ 成立。

**实验含义**：RIP 保证了测量算子近似保持低秩矩阵的几何结构，使得从线性测量中恢复低秩矩阵成为可能。实验中通过控制 $m$（样本量）相对于 $rd$（低秩自由度）的比值，可以调节问题的 RIP 强度。

#### 2.5.3 RIP 与优化景观

**定理 2.5**（RIP 下的无伪局部极小）. 若 $\mathcal{A}$ 满足秩-$2r$ RIP 且 $\delta_{2r} < 1/5$，则矩阵感知问题
$$
\min_{X} \frac{1}{2m}\|\mathcal{A}(X) - y\|_2^2 \quad \text{s.t.} \quad \text{rank}(X) \leq r
$$
的所有局部极小值点都是全局极小值点。

**实验含义**：在良好 RIP 条件下，矩阵感知不存在伪局部极小值，任何收敛到驻点的算法都能找到全局最优解。这为比较 Muon 与 SGD 的收敛速率提供了公平的基础——两者最终都会收敛到全局最优，差异仅在于收敛路径和速度。

### 2.6 矩阵分解的非凸性分析

#### 2.6.1 深度矩阵分解问题

**定义 2.17**（深度矩阵分解，MF-$L$）. 给定目标矩阵 $X^\star \in \mathbb{R}^{d \times d}$ 和深度 $L \geq 2$，深度矩阵分解问题为
$$
\min_{W_1, \ldots, W_L} f_{MF}(W_1, \ldots, W_L) := \frac{1}{2}\|W_L W_{L-1} \cdots W_1 - X^\star\|_F^2,
$$
其中 $W_\ell \in \mathbb{R}^{d \times d}$，$\ell = 1, \ldots, L$。

**定义 2.18**（重构映射）. 定义**乘积映射** $\Pi: \mathbb{R}^{d \times d} \times \cdots \times \mathbb{R}^{d \times d} \to \mathbb{R}^{d \times d}$ 为
$$
\Pi(W_1, \ldots, W_L) := W_L W_{L-1} \cdots W_1.
$$
目标函数可写为 $f_{MF} = \frac{1}{2}\|\Pi(\mathbf{W}) - X^\star\|_F^2$。

#### 2.6.2 尺度对称性与等效参数化

**性质**（尺度对称性）. 对任意可逆对角矩阵 $\{D_\ell\}_{\ell=1}^{L-1}$，令
$$
\tilde{W}_\ell := W_\ell D_\ell, \quad \tilde{W}_{\ell+1} := D_\ell^{-1} W_{\ell+1},
$$
则 $\Pi(\tilde{W}_1, \ldots, \tilde{W}_L) = \Pi(W_1, \ldots, W_L)$。这意味着损失函数在参数空间中存在大量等价解。

**定义 2.19**（平衡性）. 参数化 $\mathbf{W} = (W_1, \ldots, W_L)$ 称为**平衡**的，若对所有 $\ell = 1, \ldots, L-1$：
$$
W_{\ell+1}^\top W_{\ell+1} = W_\ell W_\top_\ell.
$$
特别地，当 $L=2$ 时，平衡条件为 $W_2^\top W_2 = W_1 W_1^\top$。

#### 2.6.3 优化 Landscape：鞍点与局部极小

**定理 2.6**（MF-$L$ 的一阶稳定点结构）. 对 $L = 2$，即 $f(W_1, W_2) = \frac{1}{2}\|W_2 W_1 - X^\star\|_F^2$：
- 所有全局极小值点满足 $W_2 W_1 = X^\star$
- 存在鞍点：若 $W_1 = 0$ 且 $W_2 = 0$，则这是鞍点（Hessian 有正负特征值）
- 若 $X^\star$ 满秩，则所有满足平衡条件的一阶稳定点要么是全局极小，要么是严格鞍点

**定理 2.7**（深度 MF 的严格鞍性质）. 对于 $L \geq 2$，在适当初始化下，梯度下降可以逃离严格鞍点。若所有鞍点均为严格鞍点（Hessian 存在负曲率方向），则随机初始化的梯度下降以概率 1 收敛到局部极小值点。

**定义 2.20**（PL* 条件）. 在点 $X$ 的邻域内，函数 $f$ 满足 $\mu$-PL* 条件，若
$$
\|\nabla f(X)\|_F^2 \geq \mu \cdot (f(X) - f^\star).
$$
PL* 是局部化的 PL 条件，保证局部线性收敛。

**实验含义**：MF-$L$（特别是 $L \geq 3$）的优化 landscape 极其复杂，存在大量鞍点和尺度等价类。Muon 的谱归一化机制可能通过以下方式改善优化行为：(1) 在大梯度区域提供更稳定的更新方向；(2) 在鞍点附近可能通过谱截断效应逃离平坦区域；(3) 各层参数的谱归一化可能隐式促进平衡性。

---

## 3. 统计推断基础

### 3.1 随机实验框架

#### 3.1.1 数据生成过程（DGP）

实验的统计有效性依赖于明确的数据生成过程（Data Generating Process, DGP）。本实验涉及三类随机源：测量矩阵、观测噪声、和参数初始化。

**定义 3.1**（数据生成过程）. 矩阵感知实验的 DGP 定义为五元组
$$
\mathcal{P}_{MS} := (d, r, m, X^\star, \{A_i\}, \epsilon),
$$
其中：
- $d$：矩阵维度
- $r$：目标矩阵的秩（$r \ll d$）
- $m$：测量样本数
- $X^\star \in \mathbb{R}^{d \times d}$：真实低秩矩阵，满足 $\text{rank}(X^\star) = r$
- $A_i \overset{i.i.d.}{\sim} \mathcal{D}_A$：随机测量矩阵
- $\epsilon_i \overset{i.i.d.}{\sim} \mathcal{D}_\epsilon$：观测噪声

观测值为 $y_i = \langle A_i, X^\star \rangle + \epsilon_i$，$i = 1, \ldots, m$。

矩阵分解实验的 DGP 为
$$
\mathcal{P}_{MF} := (d, L, X^\star, \{W_\ell^{(0)}\}),
$$
其中 $L$ 为分解深度，$\{W_\ell^{(0)}\}$ 为随机初始化。

**定义 3.2**（随机种子控制）. 实验通过伪随机数生成器的种子 $s \in \mathbb{Z}$ 控制随机性。记 $\xi(s)$ 为由种子 $s$ 确定的所有随机变量的联合实现。固定种子下的实验是确定性的；变化种子产生随机实验的重复（Replication）。

#### 3.1.2 随机测量矩阵

**定义 3.3**（高斯测量矩阵）. 标准高斯测量矩阵的每个元素独立服从标准正态分布：
$$
[A_i]_{jk} \overset{i.i.d.}{\sim} \mathcal{N}(0, 1), \quad j, k = 1, \ldots, d.
$$
等价地，$\text{vec}(A_i) \sim \mathcal{N}(0, I_{d^2})$。

**性质**：对任意矩阵 $X$，$\langle A_i, X \rangle \sim \mathcal{N}(0, \|X\|_F^2)$。因此
$$
\mathbb{E}\left[\frac{1}{m}\|\mathcal{A}(X)\|_2^2\right] = \|X\|_F^2,
$$
即 $\mathcal{A}$ 在期望意义下是等距的。

**定义 3.4**（次高斯测量矩阵）. 测量矩阵称为**次高斯**的，若其元素为独立的零均值次高斯随机变量，参数为 $K > 0$。即对任意 $t > 0$：
$$
\mathbb{P}(|[A_i]_{jk}| > t) \leq 2 \exp(-t^2/K^2).
$$

#### 3.1.3 噪声模型

**定义 3.5**（加性高斯噪声）. 观测噪声 $\epsilon \in \mathbb{R}^m$ 的元素独立同分布：
$$\epsilon_i \overset{i.i.d.}{\sim} \mathcal{N}(0, \sigma_n^2),$$
其中 $\sigma_n^2$ 为噪声方差。信噪比定义为
$$
\text{SNR} := \frac{\|\mathcal{A}(X^\star)\|_2^2}{\mathbb{E}\|\epsilon\|_2^2} = \frac{\sum_{i=1}^m \langle A_i, X^\star \rangle^2}{m \sigma_n^2}.$$

**定义 3.6**（归一化噪声水平）. 在实验中更常用的是**相对噪声水平**：
$$\bar{\sigma}_n := \frac{\sigma_n}{\|X^\star\|_F},$$
这消除了目标矩阵尺度的影响，便于跨问题比较。

### 3.2 假设检验形式

#### 3.2.1 基本假设检验框架

本实验的核心统计问题是比较 Muon 与 SGD 的收敛性能。将算法 $A$（Muon）与基准算法 $B$（SGD）在各实验条件下进行比较。

**定义 3.7**（算法性能随机变量）. 对给定问题实例 $P$ 和随机种子 $s$，定义算法 $A$ 达到精度 $\epsilon$ 所需的迭代次数为 $K_\epsilon^{(A)}(P, s)$。对 $R$ 次独立重复实验，得到样本
$$\{K_{\epsilon,r}^{(A)}\}_{r=1}^R, \quad \{K_{\epsilon,r}^{(B)}\}_{r=1}^R.$$

**定义 3.8**（性能差异）. 定义算法 $A$ 相对于算法 $B$ 的**性能差异**为
$$\Delta_\epsilon^{(r)} := K_{\epsilon,r}^{(A)} - K_{\epsilon,r}^{(B)}.$$
负值表示算法 $A$ 更快。

#### 3.2.2 配对比较检验

**定义 3.9**（零假设与备择假设）. 对配对差异 $\{\Delta_\epsilon^{(r)}\}_{r=1}^R$，建立双侧假设检验：
- **零假设** $H_0$：两种算法无系统性差异，即 $\mathbb{E}[\Delta_\epsilon] = 0$
- **备择假设** $H_1$：两种算法存在显著差异，即 $\mathbb{E}[\Delta_\epsilon] \neq 0$

若关注 Muon 是否严格优于 SGD，则使用单侧检验：
- $H_0^{\text{one-sided}}$：$\mathbb{E}[\Delta_\epsilon] \geq 0$（Muon 不更快）
- $H_1^{\text{one-sided}}$：$\mathbb{E}[\Delta_\epsilon] < 0$（Muon 更快）

#### 3.2.3 检验统计量与 p 值

**定义 3.10**（配对 t 检验统计量）. 令
$$\bar{\Delta} := \frac{1}{R}\sum_{r=1}^R \Delta_\epsilon^{(r)}, \quad S_\Delta^2 := \frac{1}{R-1}\sum_{r=1}^R (\Delta_\epsilon^{(r)} - \bar{\Delta})^2.$$
在 $H_0$ 下且假设 $\Delta_\epsilon^{(r)} \overset{i.i.d.}{\sim} \mathcal{N}(\mu_\Delta, \sigma_\Delta^2)$，检验统计量为
$$T := \frac{\bar{\Delta} - \mu_0}{S_\Delta / \sqrt{R}} \sim t_{R-1},$$
其中 $\mu_0 = 0$ 为 $H_0$ 下的期望值，$t_{R-1}$ 为自由度 $R-1$ 的 t 分布。

**定义 3.11**（p 值）. 双侧检验的 p 值为
$$p := 2 \cdot \min\left\{\mathbb{P}(t_{R-1} \leq T), \mathbb{P}(t_{R-1} \geq T)\right\} = 2 \cdot \left(1 - F_{t_{R-1}}(|T|)\right),$$
其中 $F_{t_{R-1}}$ 为 t 分布的累积分布函数。

**决策规则**：给定显著性水平 $\alpha \in (0, 1)$（通常取 $\alpha = 0.05$ 或 $\alpha = 0.01$），
- 若 $p \leq \alpha$，拒绝 $H_0$，认为差异统计显著
- 若 $p > \alpha$，不拒绝 $H_0$，无充分证据表明存在显著差异

**定义 3.12**（显著性水平）. **第一类错误概率**（Type I Error）为
$$\alpha := \mathbb{P}(\text{拒绝 } H_0 \mid H_0 \text{ 为真}).$$
即错误地声称存在差异的概率。

**物理解释**：p 值量化了在零假设成立时，观测到当前或更极端结果的概率。小 p 值意味着观测到的性能差异不太可能是偶然波动造成的。在算法比较中，低 p 值提供了算法差异的统计证据。

#### 3.2.4 非参数检验（Wilcoxon 符号秩检验）

当差异分布不正态或存在异常值时，使用非参数检验。

**定义 3.13**（Wilcoxon 符号秩统计量）. 对配对差异 $\{\Delta^{(r)}\}$：
1. 计算绝对值 $|\Delta^{(r)}|$ 并剔除零差异
2. 对 $|\Delta^{(r)}|$ 从小到大赋予秩次（Rank）$\text{rank}_r$
3. 计算正差异的秩和 $W^+ = \sum_{\Delta^{(r)} > 0} \text{rank}_r$ 与负差异的秩和 $W^- = \sum_{\Delta^{(r)} < 0} \text{rank}_r$
4. 检验统计量 $W := \min(W^+, W^-)$

**性质**：Wilcoxon 检验不依赖正态性假设，对异常值更稳健，适合算法比较中可能存在的重尾分布。

### 3.3 置信区间构造

#### 3.3.1 均值置信区间

**定义 3.14**（迭代复杂度的置信区间）. 算法 $A$ 的 $K_\epsilon$ 的样本均值为 $\bar{K}^{(A)} = \frac{1}{R}\sum_{r=1}^R K_{\epsilon,r}^{(A)}$。其 $(1-\alpha)$ 置信水平下的**置信区间**为
$$\text{CI}_{1-\alpha}\left(\mathbb{E}[K_\epsilon^{(A)}]\right) := \left[\bar{K}^{(A)} - t_{R-1, \alpha/2} \cdot \frac{S_K^{(A)}}{\sqrt{R}}, \; \bar{K}^{(A)} + t_{R-1, \alpha/2} \cdot \frac{S_K^{(A)}}{\sqrt{R}}\right],$$
其中 $t_{R-1, \alpha/2}$ 为 $t_{R-1}$ 分布的 $\alpha/2$ 上分位数，$S_K^{(A)}$ 为样本标准差。

**定义 3.15**（对数差距的置信区间）. 对优化差距序列 $\{\delta_k^{(r)}\}_{r=1}^R$，定义对数差距的样本均值与置信区间：
$$\overline{\log \delta_k} := \frac{1}{R}\sum_{r=1}^R \log \delta_k^{(r)},$$
$$\text{CI}_{1-\alpha}(\log \delta_k) := \left[\overline{\log \delta_k} - z_{\alpha/2} \cdot \frac{s_{\log, k}}{\sqrt{R}}, \; \overline{\log \delta_k} + z_{\alpha/2} \cdot \frac{s_{\log, k}}{\sqrt{R}}\right],$$
其中 $s_{\log, k}^2 = \frac{1}{R-1}\sum_{r=1}^R (\log \delta_k^{(r)} - \overline{\log \delta_k})^2$，$z_{\alpha/2}$ 为标准正态分位数。

#### 3.3.2 函数值差距的收敛曲线带

**定义 3.16**（逐点置信带）. 对每一步 $k$，定义函数值差距的逐点 $(1-\alpha)$ 置信带为
$$\mathcal{C}_k := \left[\bar{\delta}_k - z_{\alpha/2} \cdot \frac{s_{\delta,k}}{\sqrt{R}}, \; \bar{\delta}_k + z_{\alpha/2} \cdot \frac{s_{\delta,k}}{\sqrt{R}}\right].$$
注意：$\{\mathcal{C}_k\}_{k=1}^K$ 是**逐点**（pointwise）置信区间，而非**同时**（simultaneous）置信带。若需同时覆盖整条收敛曲线，应使用 Bonferroni 校正或构造同时置信带。

### 3.4 样本量与统计功效

#### 3.4.1 功效分析

**定义 3.17**（统计功效）. **第二类错误概率**为
$$\beta := \mathbb{P}(\text{不拒绝 } H_0 \mid H_1 \text{ 为真}),$$
即未能检测到真实差异的概率。**统计功效**（Statistical Power）为
$$\text{Power} := 1 - \beta = \mathbb{P}(\text{拒绝 } H_0 \mid H_1 \text{ 为真}).$$

**定义 3.18**（效应量）. 假设检验的**标准化效应量**（Effect Size）定义为
$$\text{ES} := \frac{|\mathbb{E}[\Delta_\epsilon]|}{\sigma_\Delta} = \frac{|\mu_A - \mu_B|}{\sigma_{\text{pooled}}}.$$
Cohen's $d$ 是常用的效应量度量：
$$d := \frac{\bar{\Delta}}{S_\Delta}.$$
- $d \approx 0.2$：小效应
- $d \approx 0.5$：中等效应
- $d \approx 0.8$：大效应

**定理 3.1**（功效与样本量的关系）. 对配对 t 检验，给定显著性水平 $\alpha$、效应量 $\text{ES}$ 和期望功效 $1-\beta$，所需重复实验次数满足
$$R \geq \left(\frac{z_{\alpha/2} + z_{\beta}}{\text{ES}}\right)^2 + \frac{z_{\alpha/2}^2}{2}.$$

**实验含义**：实验设计时需确保 $R$ 足够大以检测感兴趣的效应量。若算法间实际差异对应大效应量（$d \geq 0.8$），则 $R = 20 \sim 50$ 通常足够；若效应量中等，则需要更多重复。本实验取 $R = 50$ 以确保对中等效应量有足够功效。

### 3.5 多重检验问题与校正方法

#### 3.5.1 多重检验问题

实验中对多种条件（不同维度 $d$、不同秩 $r$、不同噪声水平）进行比较时，会执行多个假设检验。多重检验增加了犯第一类错误的概率。

**定义 3.19**（族错误率）. 进行 $M$ 个独立检验时，若每个检验的水平为 $\alpha$，则至少犯一次第一类错误的概率为
$$\text{FWER} = \mathbb{P}(\text{至少一次错误拒绝}) = 1 - (1-\alpha)^M \approx M\alpha \quad (\text{对小的 } \alpha).$$

#### 3.5.2 Bonferroni 校正

**定义 3.20**（Bonferroni 校正）. 对 $M$ 个检验，将每个检验的显著性水平调整为
$$\alpha^* = \frac{\alpha}{M}.$$
这样可保证族错误率不超过 $\alpha$。

**性质**：Bonferroni 校正保守，尤其当检验数量 $M$ 大或检验间存在正相关时。

#### 3.5.3 错误发现率（FDR）控制

**定义 3.21**（FDR）. 设进行 $M$ 个检验，其中 $M_0$ 个零假设为真。定义**错误发现率**为
$$\text{FDR} := \mathbb{E}\left[\frac{V}{\max(R, 1)}\right],$$
其中 $V$ 为错误拒绝的次数，$R$ 为总拒绝次数。

**定义 3.22**（Benjamini-Hochberg 程序）. 对 $M$ 个检验的 p 值 $\{p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(M)}\}$（排序后），找到
$$k^* = \max\left\{k : p_{(k)} \leq \frac{k \cdot \alpha}{M}\right\},$$
然后拒绝前 $k^*$ 个假设。此程序保证 $\text{FDR} \leq \alpha$。

**实验含义**：当实验涉及多维参数扫描（如不同 $d$、$r$、$L$ 的组合）时，需使用多重检验校正。Bonferroni 适用于检验数较少的情形；BH-FDR 适用于大规模扫描，可平衡探索性与统计严谨性。

### 3.6 随机变量的定义与分布假设

#### 3.6.1 随机初始化的分布

**定义 3.23**（标准随机初始化）. 矩阵分解实验中，各层参数初始化为
$$[W_\ell^{(0)}]_{ij} \overset{i.i.d.}{\sim} \mathcal{N}\left(0, \sigma_w^2\right), \quad \ell = 1, \ldots, L.$$
常用选择为 $\sigma_w^2 = 1/d$，使乘积矩阵的谱范数期望保持有界。

**定义 3.24**（Xavier / Kaiming 初始化）. 对深度网络风格的初始化：
- Xavier 初始化：$\sigma_w^2 = 1/d$
- Kaiming 初始化：$\sigma_w^2 = 2/d$

**性质**：适当尺度的随机初始化确保乘积矩阵 $W_L^{(0)} \cdots W_1^{(0)}$ 的期望 Frobenius 范数有界，避免梯度爆炸或消失。

#### 3.6.2 目标矩阵的构造

**定义 3.25**（随机低秩矩阵）. 实验中构造的目标矩阵为
$$X^\star := \sum_{i=1}^r \sigma_i^\star \cdot u_i^\star (v_i^\star)^\top,$$
其中：
- $u_i^\star, v_i^\star \overset{i.i.d.}{\sim} \mathcal{N}(0, I_d)$ 的正交化结果（通过 QR 分解获得标准正交列）
- $\sigma_i^\star = \sigma_1^\star \cdot \rho^{i-1}$，$\rho \in (0, 1]$ 为谱衰减率

当 $\rho = 1$ 时，所有奇异值相等（平坦谱）；当 $\rho \ll 1$ 时，谱快速衰减（高度低秩性）。

#### 3.6.3 高斯测量矩阵与噪声的形式化定义

**定义 3.26**（高斯测量算子的形式化定义）. 高斯测量算子 $\mathcal{A}: \mathbb{R}^{d \times d} \to \mathbb{R}^m$ 的完整概率空间构造如下：
设 $(\Omega, \mathcal{F}, \mathbb{P})$ 为概率空间，高斯测量矩阵为随机变量
$$A_i: \Omega \to \mathbb{R}^{d \times d}, \quad i = 1, \ldots, m,$$
满足 $[A_i(\omega)]_{jk} \overset{i.i.d.}{\sim} \mathcal{N}(0, 1)$。噪声随机变量为
$$\epsilon_i: \Omega \to \mathbb{R}, \quad \epsilon_i \overset{i.i.d.}{\sim} \mathcal{N}(0, \sigma_n^2),$$
且 $\{A_i\}$ 与 $\{\epsilon_i\}$ 相互独立。

观测随机变量为
$$y_i(\omega) := \langle A_i(\omega), X^\star \rangle + \epsilon_i(\omega).$$

**定义 3.27**（随机实验的联合概率空间）. 对 $R$ 次独立重复实验，联合概率空间为乘积空间 $(\Omega^R, \mathcal{F}^{\otimes R}, \mathbb{P}^{\otimes R})$。第 $r$ 次重复使用的随机种子 $s_r$ 对应概率空间中的一个实现 $\omega_r \in \Omega$。

**定义 3.28**（确定性等价与期望轨迹）. 对随机迭代序列 $\{X^{(k)}(\omega)\}$，定义**期望轨迹**为
$$\bar{X}^{(k)} := \mathbb{E}[X^{(k)}] = \int_\Omega X^{(k)}(\omega) \, d\mathbb{P}(\omega),$$
以及**期望函数值差距**为
$$\bar{\delta}_k := \mathbb{E}[\delta_k] = \mathbb{E}[f(X^{(k)}) - f^\star].$$
在实验中，通过蒙特卡洛平均 $\frac{1}{R}\sum_{r=1}^R \delta_k^{(r)}$ 近似 $\bar{\delta}_k$。

---

## 4. 计算复杂度模型

### 4.1 FLOPs 计数模型

FLOPs（Floating-Point Operations，浮点运算次数）是衡量算法计算量的基本单位。在矩阵优化中，精确计数 FLOPs 对理论分析与实验验证都至关重要。

#### 4.1.1 基本矩阵运算的 FLOPs

**定义 4.1**（矩阵乘法 FLOPs）. 设 $A \in \mathbb{R}^{m \times n}$，$B \in \mathbb{R}^{n \times p}$，则乘积 $C = AB$ 的 FLOPs 计数为
$$\text{FLOPs}(A \cdot B) = 2mnp - mp \approx 2mnp \quad (\text{当 } n \gg 1).$$
特别地，对 $d \times d$ 方阵乘法：$\text{FLOPs}(A \cdot B) \approx 2d^3$。

**定义 4.2**（矩阵-向量乘法 FLOPs）. $A \in \mathbb{R}^{m \times n}$，$x \in \mathbb{R}^n$：
$$\text{FLOPs}(Ax) = 2mn - m \approx 2mn.$$

**定义 4.3**（矩阵内积 FLOPs）. $\langle A, B \rangle = \text{tr}(A^\top B)$ 需要 $2d_1 d_2$ 次 FLOPs。

#### 4.1.2 SVD 分解的 FLOPs

**定义 4.4**（完全 SVD 的 FLOPs）. 对稠密矩阵 $X \in \mathbb{R}^{d_1 \times d_2}$（设 $d_1 \leq d_2$），完全 SVD 的计算复杂度为
$$\text{FLOPs}(\text{SVD}(X)) \approx 4d_1^2 d_2 + 8d_1^3 \approx O(d_1^2 d_2).$$
对 $d \times d$ 方阵：$\text{FLOPs}(\text{SVD}) \approx 12d^3$。

更精确的计数（Golub-Reinsch SVD 算法）：
- 先通过 Householder 变换双对角化：$\frac{8}{3}d^3$ FLOPs
- QR 迭代求奇异值：$O(d^2)$ 到 $O(d^3)$，取决于收敛速度
- 总复杂度约为 $12d^3$ 到 $20d^3$ FLOPs

**定义 4.5**（截断 SVD / 部分 SVD 的 FLOPs）. 若仅需前 $k \ll d$ 个奇异值和奇异向量（例如使用 Lanczos 或随机 SVD 方法），复杂度可降至
$$\text{FLOPs}(\text{truncated-SVD}_k) = O(k \cdot d^2 \cdot \text{iterations}) \ll d^3.$$
在 Muon 中，若梯度矩阵近似低秩，截断 SVD 可显著降低计算成本。

#### 4.1.3 各算法每步 FLOPs 的精确计数

**定义 4.6**（SGD 每步 FLOPs）. 对矩阵感知问题，SGD（全批量 GD）的梯度为
$$G^{(k)} = \frac{1}{m}\mathcal{A}^*(\mathcal{A}(X^{(k)}) - y) = \frac{1}{m}\sum_{i=1}^m \left(\langle A_i, X^{(k)}\rangle - y_i\right) A_i.$$
梯度计算涉及 $m$ 次矩阵内积和 $m$ 次标量-矩阵乘法：
$$\text{FLOPs}_k^{\text{SGD}} = m \cdot (2d^2 + d^2) + d^2 = 3md^2 + d^2 \approx 3md^2.$$
参数更新 $X^{(k+1)} = (1-\lambda)X^{(k)} - \eta G^{(k)}$ 需要 $3d^2$ FLOPs。
总计：
$$\text{FLOPs}_k^{\text{SGD}} = 3md^2 + 3d^2 \approx 3(m+1)d^2.$$

**定义 4.7**（Muon 每步 FLOPs）. Muon 的更新为
$$X^{(k+1)} = (1-\lambda) X^{(k)} - \eta \cdot \mathcal{N}_{\text{spec}}(G^{(k)}).$$
其中谱归一化需要对梯度矩阵 $G^{(k)}$ 进行 SVD 分解。精确计数：
- 梯度计算：$3md^2$ FLOPs（同 SGD）
- SVD 分解：$\approx 12d^3$ FLOPs（稠密完全 SVD）
- 谱归一化 $UV^\top$：$2d^3$ FLOPs（$d \times d$ 矩阵乘以 $d \times d$ 矩阵）
- 参数更新：$3d^2$ FLOPs
总计：
$$\text{FLOPs}_k^{\text{Muon}} = 3md^2 + 12d^3 + 2d^3 + 3d^2 \approx 3md^2 + 14d^3.$$

**定义 4.8**（FLOPs 比值）. Muon 相对于 SGD 的单步 FLOPs 比值为
$$\gamma_{\text{FLOPs}} := \frac{\text{FLOPs}_k^{\text{Muon}}}{\text{FLOPs}_k^{\text{SGD}}} = \frac{3md^2 + 14d^3}{3md^2} = 1 + \frac{14d}{3m} = 1 + O\left(\frac{d}{m}\right).$$
当 $m \gg d$ 时，$\gamma_{\text{FLOPs}} \approx 1$，SVD 的额外开销相对较小；当 $m \approx d$ 或 $m < d$ 时，$\gamma_{\text{FLOPs}} \gg 1$。

#### 4.1.4 累积计算复杂度

**定义 4.9**（总计算复杂度）. 算法 $A$ 达到精度 $\epsilon$ 的**总 FLOPs**为
$$F_\epsilon^{(A)} := \sum_{k=0}^{K_\epsilon^{(A)}-1} \text{FLOPs}_k^{(A)}.$$
若每步 FLOPs 恒定，则 $F_\epsilon^{(A)} = K_\epsilon^{(A)} \cdot \text{FLOPs}_{\text{per-step}}^{(A)}$。

**定义 4.10**（计算效率比较）. 算法 $A$ 相对于算法 $B$ 的**计算效率比**为
$$\rho_F^{(A,B)} := \frac{F_\epsilon^{(A)}}{F_\epsilon^{(B)}} = \frac{K_\epsilon^{(A)} \cdot \text{FLOPs}^{(A)}}{K_\epsilon^{(B)} \cdot \text{FLOPs}^{(B)}}.$$
当 $\rho_F^{(A,B)} < 1$ 时，算法 $A$ 在总计算量上更高效，即使单步更昂贵。

**实验含义**：Muon 的单步成本高于 SGD（主要由于 SVD），但如果 Muon 的迭代复杂度 $K_\epsilon^{\text{Muon}}$ 显著小于 $K_\epsilon^{\text{SGD}}$，总计算量仍可能更优。实验中需同时比较 $K_\epsilon$（迭代效率）和 $F_\epsilon$（计算效率），以全面评估算法。

### 4.2 渐近复杂度分析

#### 4.2.1 渐近记号体系

**定义 4.11**（渐近记号）. 设 $f(n), g(n)$ 为正函数：
- **大 O 记号**：$f(n) = O(g(n))$ 表示存在 $C > 0$ 和 $n_0$ 使得 $f(n) \leq C g(n)$ 对所有 $n \geq n_0$ 成立。
- **大 Omega 记号**：$f(n) = \Omega(g(n))$ 表示存在 $c > 0$ 和 $n_0$ 使得 $f(n) \geq c g(n)$ 对所有 $n \geq n_0$ 成立。
- **大 Theta 记号**：$f(n) = \Theta(g(n))$ 表示 $f(n) = O(g(n))$ 且 $f(n) = \Omega(g(n))$。
- **小 o 记号**：$f(n) = o(g(n))$ 表示 $\lim_{n \to \infty} f(n)/g(n) = 0$。

#### 4.2.2 问题维度的渐近分析

**定义 4.12**（问题规模参数）. 矩阵感知问题的规模参数为 $(d, r, m)$，其中：
- $d$：矩阵维度（主导参数）
- $r$：目标秩（$r \ll d$）
- $m$：测量数（通常 $m = O(rd)$ 或 $m = O(d^2)$）

**渐近复杂度汇总**：
| 操作 | 复杂度 |
|------|--------|
| 梯度计算（全批量） | $O(md^2)$ |
| 完全 SVD（稠密） | $O(d^3)$ |
| 截断 SVD（前 $r$ 个） | $O(rd^2)$ |
| 矩阵乘法（$d \times d$） | $O(d^3)$ |
| 矩阵-向量乘法 | $O(d^2)$ |

#### 4.2.3 算法总复杂度的渐近比较

**定义 4.13**（Muon 的总渐近复杂度）. 设 Muon 的迭代复杂度为 $K_\epsilon^{\text{Muon}}$，则
$$F_\epsilon^{\text{Muon}} = K_\epsilon^{\text{Muon}} \cdot O(md^2 + d^3).$$
若 $m \gg d$，主导项为 $md^2$；若 $m \approx d$，则 $d^3$ 项（SVD）可能主导。

**定义 4.14**（SGD 的总渐近复杂度）. 
$$F_\epsilon^{\text{SGD}} = K_\epsilon^{\text{SGD}} \cdot O(md^2).$$

**比较条件**：Muon 在计算效率上优于 SGD 当且仅当
$$K_\epsilon^{\text{Muon}} \cdot (md^2 + d^3) < K_\epsilon^{\text{SGD}} \cdot md^2,$$
即
$$\frac{K_\epsilon^{\text{Muon}}}{K_\epsilon^{\text{SGD}}} < \frac{md^2}{md^2 + d^3} = \frac{1}{1 + d/m}.$$
当 $m \gg d$ 时，右边 $\approx 1$，要求 $K_\epsilon^{\text{Muon}} < K_\epsilon^{\text{SGD}}$（任何优势都可能使总计算量更优）；当 $m \ll d$ 时，右边 $\approx m/d \ll 1$，要求 Muon 的迭代优势足够大以补偿 SVD 开销。

### 4.3 Wall-clock Time 与理论 FLOPs 的关系

#### 4.3.1 理论 FLOPs 与实测时间

**定义 4.15**（Wall-clock Time）. 算法的**墙钟时间**（Wall-clock Time）$T_{\text{wall}}$ 为从算法开始到结束的物理时间，包括：
- 实际浮点运算时间
- 内存访问时间（加载/存储数据）
- 缓存未命中惩罚
- 并行开销（线程同步、通信）
- 系统开销（OS 调度、中断）

**定义 4.16**（理论时间与 FLOPs 的关系）. 设处理器的峰值浮点性能为 $P$ FLOPs/秒（例如现代 CPU 为 $10^{10} \sim 10^{11}$，GPU 为 $10^{13} \sim 10^{14}$），则**理论最小时间**为
$$T_{\text{theory}} = \frac{\text{总 FLOPs}}{P}.$$
实际墙钟时间满足 $T_{\text{wall}} \geq T_{\text{theory}}$，通常 $T_{\text{wall}} = c \cdot T_{\text{theory}}$，其中 $c \geq 1$ 为**实现效率因子**。

**定义 4.17**（算法强度与内存墙）. **运算强度**（Arithmetic Intensity）定义为
$$I := \frac{\text{FLOPs}}{\text{Bytes moved}}.$$
当 $I$ 高时，算法受计算限制（compute-bound），接近峰值性能；当 $I$ 低时，算法受内存带宽限制（memory-bound），实际性能远低于峰值。

矩阵乘法的运算强度为 $O(d)$（对 $d \times d$ 矩阵乘法）；SVD 的运算强度类似。两者均为计算密集型操作，在现代硬件上可达到较高效率。

#### 4.3.2 常数因子的重要性

**定义 4.18**（有效常数）. 在实际比较中，大 O 记号隐藏的常数因子至关重要。定义**有效常数**为
$$C_{\text{eff}} := \frac{T_{\text{wall}} \cdot P}{\text{理论 FLOPs}} = \frac{\text{实际时间} \times \text{峰值性能}}{\text{理论 FLOPs}}.$$
不同算法（不同操作组合）的 $C_{\text{eff}}$ 可能差异很大。例如：
- SVD 的 $C_{\text{eff}}$ 通常高于矩阵乘法，因为 SVD 涉及更多串行依赖和分支判断
- 高度优化的矩阵乘法库（如 BLAS、cuBLAS）的 $C_{\text{eff}}$ 接近 1

**实验含义**：实验报告中应同时给出理论 FLOPs 和实际 wall-clock time。理论 FLOPs 是硬件无关的公平比较基准；wall-clock time 反映了实际软件实现和硬件上的性能。两者的差异揭示了算法的实现效率。

### 4.4 内存复杂度

#### 4.4.1 空间复杂度

**定义 4.19**（空间复杂度）. 算法的**空间复杂度**（内存占用）为执行过程中需要同时驻留内存的数据量。

对矩阵感知问题中的 SGD：
- 当前参数 $X^{(k)}$：$d^2$ 个浮点数
- 梯度 $G^{(k)}$：$d^2$ 个浮点数
- 测量矩阵（若存储）：$m \cdot d^2$ 个浮点数（通常 $m \cdot d^2$ 很大，可改为按需生成或存储压缩表示）
总空间：$S_{\text{SGD}} = 2d^2 + m \cdot d^2 \approx O(md^2)$（若存储全部测量矩阵）或 $O(d^2)$（若在线生成）。

对 Muon：
- 当前参数 $X^{(k)}$：$d^2$ 个浮点数
- 梯度 $G^{(k)}$：$d^2$ 个浮点数
- SVD 分解中的临时矩阵 $U, \Sigma, V$：$3d^2$ 个浮点数
- 测量矩阵：同 SGD
总空间：$S_{\text{Muon}} = 5d^2 + m \cdot d^2 \approx O(md^2)$（存储测量矩阵时）或 $O(d^2)$（在线生成时）。

#### 4.4.2 缓存效率与数据局部性

**定义 4.20**（缓存复杂度）. 设缓存行大小为 $L_c$ 字节，缓存容量为 $C$ 字节。**缓存未命中次数**是衡量内存访问效率的重要指标。

矩阵乘法可通过分块（blocking）优化缓存效率，时间复杂度仍为 $O(d^3)$ 但常数因子显著降低。SVD 的双对角化过程天然具有较好的数据局部性，但 QR 迭代阶段的随机访问模式可能导致较多缓存未命中。

**实验含义**：当 $d$ 很大时，内存占用可能成为瓶颈。若 $d^2$ 超过 L3 缓存容量（通常 $10 \sim 30$ MB），则内存带宽成为限制因素。实验中应记录峰值内存使用量，以评估算法的可扩展性。

---

## 5. 符号汇总与统一约定

为确保全文符号一致，汇总如下：

| 符号 | 含义 |
|------|------|
| $d$ | 矩阵维度 |
| $r$ | 矩阵秩（或目标秩） |
| $m$ | 测量样本数 |
| $L$ | 矩阵分解深度 |
| $R$ | 实验重复次数 |
| $k$ | 迭代索引 |
| $K_\epsilon$ | 达到精度 $\epsilon$ 的迭代复杂度 |
| $F_\epsilon$ | 达到精度 $\epsilon$ 的总 FLOPs |
| $X^\star$ | 真实/目标矩阵 |
| $X^{(k)}$ | 第 $k$ 次迭代的参数矩阵 |
| $G^{(k)}$ | 第 $k$ 步的梯度矩阵 |
| $D^{(k)}$ | 第 $k$ 步的搜索方向 |
| $f(X)$ | 目标函数 |
| $f^\star$ | 最优函数值 |
| $\delta_k$ | 函数值差距 $f(X^{(k)}) - f^\star$ |
| $\eta$ | 学习率 / 步长 |
| $\lambda$ | 权重衰减系数 |
| $\sigma_i(X)$ | $X$ 的第 $i$ 个奇异值 |
| $\|X\|_2$ | 谱范数（最大奇异值） |
| $\|X\|_F$ | Frobenius 范数 |
| $\|X\|_*$ | 核范数（奇异值之和） |
| $\kappa_{\text{cond}}(X)$ | 矩阵谱条件数 $\sigma_1/\sigma_r$ |
| $\kappa_{\text{sp}}(X)$ | 谱比 $\|X\|_2/\|X\|_F$ |
| $\kappa(f)$ | 函数条件数 $L/\mu$ |
| $\mathcal{A}$ | 线性测量算子 |
| $\mathcal{A}^*$ | 伴随算子 |
| $\delta_r(\mathcal{A})$ | RIP 常数 |
| $\sigma_n^2$ | 噪声方差 |
| $\alpha$ | 显著性水平 |
| $p$ | p 值 |
| $\beta$ | 第二类错误概率 |
| $\text{Power}$ | 统计功效 $1-\beta$ |
| $\text{FLOPs}$ | 浮点运算次数 |
| $O(\cdot), \Theta(\cdot), \Omega(\cdot)$ | 渐近记号 |

---

---

# 第二部分：实验的数学形式化

> 本部分将现有实验方案转化为严格的数学与统计语言，确保所有概念、变量、过程和假设均有精确的形式化定义。建立在第一部分数学基础之上，为后续的实验执行、统计检验和结论推断奠定不可歧义的基础。

> **本章目的**：将现有实验方案转化为严格的数学与统计语言，确保所有概念、变量、过程和假设均有精确的形式化定义，为后续的实验执行、统计检验和结论推断奠定不可歧义的基础。

---

## 5. 实验空间的公理化定义

### 1.1 实验五元组

一个实验 $\mathcal{E}$ 被定义为五元组：

$$
\mathcal{E} = (\mathcal{P}, \mathcal{A}, \mathcal{D}, \mathcal{M}, \mathcal{R})
$$

其中各空间的定义如下。

---

### 1.2 问题实例空间 $\mathcal{P}$

问题实例空间是所有待优化问题的集合，本次实验包含两类实例：

**定义 1.1（矩阵感知实例）。** 一个矩阵感知实例由以下参数确定：

$$
\mathcal{P}_{MS} = \left\{ p_{MS}(d, r, m, \sigma_\epsilon) \; \middle| \; d \in \{50, 100, 200, 500\}, \; r \in \{5, \lfloor d/2 \rfloor, d\}, \; m = 3d^2, \; \sigma_\epsilon \in \{0, 0.01\} \right\}
$$

其中：
- $d \in \mathbb{N}^+$：矩阵维度；
- $r \in \{1, \ldots, d\}$：真实矩阵 $X^\star$ 的秩（$r = d$ 表示满秩）；
- $m \in \mathbb{N}^+$：测量数量；
- $\sigma_\epsilon \geq 0$：噪声标准差。

每个实例的具体构成见第2节的数据生成过程。

**定义 1.2（矩阵分解实例）。** 一个矩阵分解实例由以下参数确定：

$$
\mathcal{P}_{MF} = \left\{ p_{MF}(d, r, L) \; \middle| \; d \in \{50, 100, 200, 500\}, \; r = d, \; L \in \{2, 3, 4\} \right\}
$$

其中：
- $d \in \mathbb{N}^+$：所有中间矩阵的维度（方阵）；
- $L \in \{2, 3, 4\}$：分解深度（层数）；
- $r = d$：目标矩阵的秩（本次实验中始终满秩）。

**总问题实例空间：**

$$
\mathcal{P} = \mathcal{P}_{MS} \cup \mathcal{P}_{MF}
$$

---

### 1.3 算法空间 $\mathcal{A}$

算法空间包含本次实验对比的两种一阶优化算法：

$$
\mathcal{A} = \{ \text{Muon}, \; \text{SGD} \}
$$

**定义 1.3（Muon 算法）。** 给定参数 $(\eta, \lambda, \mu_{mom}, \mu_{Nesterov}) \in \mathcal{D}$，Muon 算法的迭代映射记为 $\mathcal{T}_\eta^{\text{Muon}}$。其更新规则为：

$$
X^{(k+1)} = \mathcal{T}_\eta^{\text{Muon}}(X^{(k)}) := X^{(k)} - \eta \cdot D^{(k)} - \lambda \cdot X^{(k)}
$$

其中 $D^{(k)}$ 的构造方式为：设当前随机梯度为 $G^{(k)} \in \mathbb{R}^{d \times d}$，对其做 SVD：

$$
G^{(k)} = U \cdot \Sigma \cdot V^T
$$

其中 $U, V \in \mathbb{R}^{d \times d}$ 为正交矩阵，$\Sigma = \mathrm{diag}(\sigma_1, \ldots, \sigma_d)$ 且 $\sigma_1 \geq \cdots \geq \sigma_d \geq 0$。则：

$$
D^{(k)} := U V^T
$$

即取 SVD 的左右奇异向量外积，丢弃奇异值信息。

**定义 1.4（SGD 算法）。** 给定参数 $(\eta, \lambda) \in \mathcal{D}$，SGD 算法的迭代映射记为 $\mathcal{T}_\eta^{\text{SGD}}$。其更新规则为：

$$
X^{(k+1)} = \mathcal{T}_\eta^{\text{SGD}}(X^{(k)}) := X^{(k)} - \eta \cdot G^{(k)} - \lambda \cdot X^{(k)}
$$

其中 $G^{(k)}$ 为当前随机梯度，直接使用而不做谱分解。

---

### 1.4 数据/初始化/超参数空间 $\mathcal{D}$

空间 $\mathcal{D}$ 包含所有控制实验条件的数据与超参数配置：

$$
\mathcal{D} = \mathcal{D}_{\text{data}} \times \mathcal{D}_{\text{init}} \times \mathcal{D}_{\text{hyper}}
$$

**定义 1.5（数据空间）。**

$$
\mathcal{D}_{\text{data}} = \{ (\{A_i\}_{i=1}^m, y, X^\star) \; | \; \text{依第2节DGP生成} \}
$$

**定义 1.6（初始化空间）。**

对矩阵感知问题：

$$
\mathcal{D}_{\text{init}}^{MS} = \left\{ X^{(0)} \in \mathbb{R}^{d \times d} \; \middle| \; X^{(0)}_{ij} \overset{iid}{\sim} \mathcal{N}(0, \alpha^2), \; \alpha \in \{0.01, 0.1, 1.0\} \right\}
$$

对矩阵分解问题（$L=2$）：

$$
\mathcal{D}_{\text{init}}^{MF, L=2} = \left\{ (W_1^{(0)}, W_2^{(0)}) \in \mathbb{R}^{d \times d} \times \mathbb{R}^{d \times d} \; \middle| \; \text{三种初始化方案，见第2节} \right\}
$$

对矩阵分解问题（$L \geq 3$）：

$$
\mathcal{D}_{\text{init}}^{MF, L\geq 3} = \left\{ (W_1^{(0)}, \ldots, W_L^{(0)}) \; \middle| \; W_i^{(0)} \in \mathbb{R}^{d \times d}, \; W_{i,jk}^{(0)} \overset{iid}{\sim} \mathcal{N}(0, 1/d) \right\}
$$

**定义 1.7（超参数空间）。**

$$
\mathcal{D}_{\text{hyper}} = \left\{ (\eta, \lambda, K_{\max}, \epsilon) \; \middle| \; \eta \in \{10^{-3}, 10^{-2}, 10^{-1}\}, \; \lambda = 0, \; K_{\max} = 10^5, \; \epsilon = 10^{-6} \right\}
$$

---

### 1.5 度量空间 $\mathcal{M}$

度量空间包含所有用于评估算法性能的随机变量：

$$
\mathcal{M} = \{ K_\epsilon, F_\epsilon, \bar{\sigma}_{\log}, \rho_K, \rho_F, \delta_k \}
$$

各度量的精确定义见第3.4节。

---

### 1.6 随机性空间 $\mathcal{R}$

随机性空间捕获实验中所有非确定性来源：

$$
\mathcal{R} = \mathcal{R}_{\text{seed}} \times \mathcal{R}_{\text{problem}} \times \mathcal{R}_{\text{noise}} \times \mathcal{R}_{\text{init}} \times \mathcal{R}_{\text{stoch}}
$$

其中：
- $\mathcal{R}_{\text{seed}} = \{1, 2, \ldots, R\}$，$R = 10$ 为独立随机种子；
- $\mathcal{R}_{\text{problem}}$：问题数据（测量矩阵 $\{A_i\}$、真实矩阵 $X^\star$）的随机生成；
- $\mathcal{R}_{\text{noise}}$：观测噪声 $\{\epsilon_i\}$ 的随机生成；
- $\mathcal{R}_{\text{init}}$：参数初始化 $X^{(0)}$ 或 $\{W_i^{(0)}\}$ 的随机生成；
- $\mathcal{R}_{\text{stoch}}$：随机梯度采样中的随机性（当使用 mini-batch SGD 时）。

为控制实验变量，给定一个随机种子 $s \in \mathcal{R}_{\text{seed}}$，所有其他随机来源由 $s$ 通过确定性伪随机数生成器确定：

$$
(\{A_i\}, X^\star, \{\epsilon_i\}, X^{(0)}) = \Phi(s)
$$

其中 $\Phi$ 为固定的随机数生成映射，确保跨算法对比时的**公平随机化原则**（identical randomness given identical seed）。

---

## 6. 数据生成过程的形式化

### 2.1 矩阵感知问题的 DGP

**定义 2.1（矩阵感知数据生成过程）。** 给定参数 $(d, r, m, \sigma_\epsilon, s)$，数据按以下过程生成：

**步骤 1：生成测量矩阵。**

$$
A_i \overset{iid}{\sim} \mathcal{N}_{d \times d}(0, 1), \quad i = 1, \ldots, m
$$

即每个 $A_i$ 为 $d \times d$ 矩阵，元素独立服从标准正态分布：

$$
(A_i)_{jk} \overset{iid}{\sim} \mathcal{N}(0, 1), \quad \forall j, k \in \{1, \ldots, d\}
$$

**步骤 2：生成真实矩阵 $X^\star$。**

- **低秩情形**（$r < d$）：
  
  $$
  U, V \in \mathbb{R}^{d \times r}, \quad U_{jk} \overset{iid}{\sim} \mathcal{N}(0, 1), \quad V_{jk} \overset{iid}{\sim} \mathcal{N}(0, 1)
  $$
  
  $$
  X^\star = \frac{1}{\sqrt{r}} U V^T
  $$

  缩放因子 $1/\sqrt{r}$ 确保 $\mathbb{E}[\|X^\star\|_F^2] = d^2$ 与满秩情形量级一致。

- **满秩情形**（$r = d$）：
  
  $$
  X^\star_{ij} \overset{iid}{\sim} \mathcal{N}(0, 1)
  $$

**步骤 3：生成测量值与噪声。**

$$
y_i = \langle A_i, X^\star \rangle + \epsilon_i, \quad \epsilon_i \overset{iid}{\sim} \mathcal{N}(0, \sigma_\epsilon^2)
$$

其中矩阵内积定义为：

$$
\langle A, B \rangle = \sum_{j=1}^d \sum_{k=1}^d A_{jk} B_{jk} = \mathrm{tr}(A^T B)
$$

**联合分布：**

给定随机种子 $s$，数据生成映射 $\Phi_{MS}$ 输出联合样本：

$$
(\{A_i\}_{i=1}^m, X^\star, \{\epsilon_i\}_{i=1}^m) = \Phi_{MS}(s; d, r, m, \sigma_\epsilon)
$$

其联合概率密度（在满秩、无噪声情形）为：

$$
p_{MS}(\{A_i\}, X^\star) = \prod_{i=1}^m \prod_{j,k=1}^d \phi(A_{i,jk}) \cdot \prod_{j,k=1}^d \phi(X^\star_{jk})
$$

其中 $\phi(x) = (2\pi)^{-1/2} e^{-x^2/2}$ 为标准正态密度。有噪声时乘上 $\prod_{i=1}^m \phi_{\sigma_\epsilon}(\epsilon_i)$。

---

### 2.2 矩阵分解问题的 DGP

**定义 2.2（矩阵分解数据生成过程）。** 给定参数 $(d, L, s)$，数据按以下过程生成：

**步骤 1：生成目标矩阵。**

$$
X^\star_{ij} \overset{iid}{\sim} \mathcal{N}(0, 1)
$$

**步骤 2：生成初始化（$L=2$ 时的三种方案）。**

定义噪声矩阵 $E$，其中 $E_{ij} \overset{iid}{\sim} \mathcal{N}(0, 0.01^2)$：

- **方案 (a)**——不平衡初始化 I：
  
  $$
  W_1^{(0)} = X^\star + E, \quad W_2^{(0)} = I_d
  $$

- **方案 (b)**——不平衡初始化 II：
  
  $$
  W_1^{(0)} = I_d, \quad W_2^{(0)} = X^\star + E
  $$

- **方案 (c)**——对称初始化：
  
  $$
  W_1^{(0)} = c \cdot I_d, \quad W_2^{(0)} = \frac{1}{c} \cdot (X^\star + E)
  $$
  
  其中 $c > 0$ 为任意常数（通常取 $c = 1$）。

**步骤 3：生成初始化（$L \geq 3$）。**

$$
W_i^{(0)} \in \mathbb{R}^{d \times d}, \quad (W_i^{(0)})_{jk} \overset{iid}{\sim} \mathcal{N}\left(0, \frac{1}{d}\right), \quad i = 1, \ldots, L
$$

缩放 $1/d$ 使得 $\mathbb{E}[\|W_i^{(0)}\|_F^2] = d$，保持谱范数在 $O(1)$ 量级。

**联合分布：**

$$
(X^\star, W_1^{(0)}, \ldots, W_L^{(0)}) = \Phi_{MF}(s; d, L)
$$

其联合密度（在 $L \geq 3$ 且独立初始化时）为：

$$
p_{MF}(X^\star, \{W_i^{(0)}\}) = \prod_{j,k=1}^d \phi(X^\star_{jk}) \cdot \prod_{i=1}^L \prod_{j,k=1}^d \sqrt{d} \cdot \phi\left(\sqrt{d} \cdot (W_i^{(0)})_{jk}\right)
$$

---

### 2.3 统一的数据生成映射

综合以上两种问题，统一的数据生成映射为：

$$
\Phi: \mathcal{R}_{\text{seed}} \times \mathcal{P} \longrightarrow \mathcal{D}_{\text{data}} \times \mathcal{D}_{\text{init}}
$$

即给定随机种子和问题实例参数，确定性输出所有数据和初始化。

---

## 7. 优化问题的数学形式

### 3.1 矩阵感知问题（MS）

**定义 3.1（矩阵感知目标函数）。** 给定数据 $(\{A_i\}_{i=1}^m, \{y_i\}_{i=1}^m)$，矩阵感知问题的目标函数为：

$$
f_{MS}: \mathbb{R}^{d \times d} \to \mathbb{R}_{\geq 0}, \quad f_{MS}(X) = \frac{1}{2m} \sum_{i=1}^m \left( \langle A_i, X \rangle - y_i \right)^2
$$

**性质分析：**

- **光滑性**：$f_{MS} \in \mathcal{C}^\infty(\mathbb{R}^{d \times d})$，无穷次可微；
- **凸性**：$f_{MS}$ 是凸函数（因由线性函数的二次复合构成）；
- **强凸性**：当 $m \geq d^2$ 且 $\{A_i\}$ 张成全空间时强凸，强凸模数为：
  
  $$
  \mu_{MS} = \frac{1}{m} \lambda_{\min}\left( \sum_{i=1}^m \mathrm{vec}(A_i) \mathrm{vec}(A_i)^T \right)
  $$

- **$L$-光滑性**：
  
  $$
  L_{MS} = \frac{1}{m} \lambda_{\max}\left( \sum_{i=1}^m \mathrm{vec}(A_i) \mathrm{vec}(A_i)^T \right)
$$

- **条件数**：
  
  $$
  \kappa_{MS} = \frac{L_{MS}}{\mu_{MS}} = \frac{\lambda_{\max}(\mathcal{A}^T \mathcal{A})}{\lambda_{\min}(\mathcal{A}^T \mathcal{A})}
  $$
  
  其中 $\mathcal{A} \in \mathbb{R}^{m \times d^2}$ 为测量算子的矩阵化表示，第 $i$ 行为 $\mathrm{vec}(A_i)^T$。

**定义 3.2（矩阵感知梯度）。** 目标函数 $f_{MS}$ 的梯度为：

$$
\nabla f_{MS}(X) = \frac{1}{m} \sum_{i=1}^m \left( \langle A_i, X \rangle - y_i \right) A_i \in \mathbb{R}^{d \times d}
$$

或等价地写成向量化形式：

$$
\mathrm{vec}(\nabla f_{MS}(X)) = \frac{1}{m} \mathcal{A}^T (\mathcal{A} \, \mathrm{vec}(X) - y)
$$

其中 $y = (y_1, \ldots, y_m)^T \in \mathbb{R}^m$。

**定义 3.3（矩阵感知的 Hessian 算子）。** 由于 $f_{MS}$ 是二次函数，其 Hessian 为常数矩阵：

$$
\nabla^2 f_{MS} = \frac{1}{m} \mathcal{A}^T \mathcal{A} \in \mathbb{R}^{d^2 \times d^2}
$$

或作为线性算子 $H_{MS}: \mathbb{R}^{d \times d} \to \mathbb{R}^{d \times d}$：

$$
H_{MS}[V] = \frac{1}{m} \sum_{i=1}^m \langle A_i, V \rangle A_i
$$

**定义 3.4（矩阵感知的最优值）。** 在无噪声情形（$\sigma_\epsilon = 0$）下，最优解为 $X^\star$，最优值为：

$$
f_{MS}^\star = f_{MS}(X^\star) = 0
$$

在有噪声情形（$\sigma_\epsilon > 0$）下，最优值由最小二乘闭式解给出：

$$
\hat{X}_{LS} = \arg\min_X f_{MS}(X) = \left( \mathcal{A}^T \mathcal{A} \right)^{-1} \mathcal{A}^T y
$$

$$
f_{MS}^\star = f_{MS}(\hat{X}_{LS}) = \frac{1}{2m} y^T \left( I_m - \mathcal{A}(\mathcal{A}^T \mathcal{A})^{-1}\mathcal{A}^T \right) y
$$

---

### 3.2 矩阵分解问题（MF-$L$）

**定义 3.5（矩阵分解目标函数）。** 给定目标矩阵 $X^\star \in \mathbb{R}^{d \times d}$ 和分解深度 $L \in \{2, 3, 4\}$，矩阵分解问题的目标函数为：

$$
f_{MF}: \underbrace{\mathbb{R}^{d \times d} \times \cdots \times \mathbb{R}^{d \times d}}_{L \text{ 个}} \to \mathbb{R}_{\geq 0}
$$

$$
f_{MF}(W_1, \ldots, W_L) = \frac{1}{2} \left\| W_L W_{L-1} \cdots W_1 - X^\star \right\|_F^2
$$

其中矩阵乘积的 Frobenius 范数定义为：

$$
\|M\|_F = \sqrt{\sum_{i,j=1}^d M_{ij}^2} = \sqrt{\mathrm{tr}(M^T M)}
$$

**性质分析：**

- **光滑性**：$f_{MF} \in \mathcal{C}^\infty(\mathbb{R}^{d^2L})$；
- **凸性**：$f_{MF}$ **非凸**（矩阵乘积的非线性导致）；
- **局部强凸性**：在全局最优解的邻域内局部强凸（由 $X^\star$ 的满秩假设保证）；
- **鞍点**：存在大量鞍点（尤其在 $L \geq 3$ 时）。

**定义 3.6（矩阵分解的偏梯度）。** 对每个参数矩阵 $W_i$ 的偏梯度为：

令乘积残差：

$$
R = W_L \cdots W_1 - X^\star \in \mathbb{R}^{d \times d}
$$

定义左累积积和右累积积：

$$
\Pi_{>i} = W_L W_{L-1} \cdots W_{i+1}, \quad \Pi_{<i} = W_{i-1} \cdots W_1
$$

（当 $i=L$ 时 $\Pi_{>L} = I_d$；当 $i=1$ 时 $\Pi_{<1} = I_d$）。

则第 $i$ 个参数的偏梯度为：

$$
\frac{\partial f_{MF}}{\partial W_i} = \Pi_{>i}^T \cdot R \cdot \Pi_{<i}^T \in \mathbb{R}^{d \times d}
$$

特别地，当 $L=2$ 时：

$$
\frac{\partial f_{MF}}{\partial W_1} = W_2^T (W_2 W_1 - X^\star), \quad \frac{\partial f_{MF}}{\partial W_2} = (W_2 W_1 - X^\star) W_1^T
$$

**定义 3.7（矩阵分解的最优值）。** 全局最优值为：

$$
f_{MF}^\star = 0
$$

在 $X^\star$ 满秩的假设下，最优解集为：

$$
\mathcal{X}^\star_{MF} = \left\{ (W_1, \ldots, W_L) \; \middle| \; W_L \cdots W_1 = X^\star \right\}
$$

这是一个 $d^2(L-1)$ 维的流形（具有无穷多个最优解）。

**定义 3.8（矩阵分解的 Hessian）。** 由于参数空间维度高且问题非凸，Hessian 是一个 $Ld^2 \times Ld^2$ 的分块矩阵。在全局最优解处，其结构由链式法则给出。特别地，在 $L=2$ 且平衡点处，Hessian 的谱结构与 $X^\star$ 的奇异值密切相关。

---

### 3.3 算法的迭代映射（算子形式）

**定义 3.9（统一算法迭代算子）。** 令 $\theta \in \Theta$ 为参数空间中的参数向量（$\Theta = \mathbb{R}^{d \times d}$ 对 MS，$\Theta = \mathbb{R}^{d \times dL}$ 对 MF），算法 $\mathcal{A} \in \{\text{Muon}, \text{SGD}\}$ 的迭代算子为：

$$
\mathcal{T}_{\eta}^{\mathcal{A}}: \Theta \times \mathcal{G} \to \Theta
$$

其中 $\mathcal{G}$ 为梯度空间。对于确定性全梯度设置：

- **Muon：**
  
  $$
  \mathcal{T}_{\eta}^{\text{Muon}}(\theta, G) = \theta - \eta \cdot \mathcal{S}(G) - \lambda \cdot \theta
  $$
  
  其中 $\mathcal{S}: \mathbb{R}^{d \times d} \to \mathbb{R}^{d \times d}$ 为 SVD 归一化算子：
  
  $$
  \mathcal{S}(G) = U_G V_G^T, \quad G = U_G \Sigma_G V_G^T \text{ (SVD)}
  $$

- **SGD：**
  
  $$
  \mathcal{T}_{\eta}^{\text{SGD}}(\theta, G) = \theta - \eta \cdot G - \lambda \cdot \theta
  $$

**完整迭代序列：** 给定初始化 $\theta^{(0)}$ 和迭代次数 $K$，序列定义为：

$$
\theta^{(k+1)} = \mathcal{T}_{\eta}^{\mathcal{A}}(\theta^{(k)}, G^{(k)}), \quad k = 0, 1, \ldots, K-1
$$

其中 $G^{(k)} = \nabla f(\theta^{(k)})$ 为全梯度（本次实验使用全批量梯度而非 mini-batch，故无额外随机性）。

---

### 3.4 评估指标的数学形式

**定义 3.10（精度达标迭代次数）。** 给定目标精度 $\epsilon > 0$ 和最大迭代 $K_{\max}$，算法 $\mathcal{A}$ 在第 $s$ 个随机种子下达到精度 $\epsilon$ 的迭代次数为：

$$
K_\epsilon^{(\mathcal{A})}(s) = \min \left\{ k \in \{1, \ldots, K_{\max}\} \; \middle| \; f(\theta^{(k)}) - f^\star \leq \epsilon \right\}
$$

若算法在 $K_{\max}$ 步内未收敛，则定义：

$$
K_\epsilon^{(\mathcal{A})}(s) = K_{\max} + 1 \quad \text{（截断值）}
$$

**定义 3.11（每步 FLOPs 计算）。**

- **SGD 每步 FLOPs：**
  
  对 MS：梯度计算需 $O(md^2)$，更新需 $O(d^2)$，总计：
  
  $$
  C_{\text{SGD}}^{MS} = 2md^2 + d^2 \approx 6d^4 + d^2 \text{ FLOPs}
  $$
  
  对 MF-$L$：梯度计算需 $O(Ld^3)$，更新需 $O(Ld^2)$，总计：
  
  $$
  C_{\text{SGD}}^{MF} = O(Ld^3) \text{ FLOPs}
  $$

- **Muon 每步 FLOPs：**
  
  在 SGD 基础上增加 SVD 的代价。对 $d \times d$ 矩阵的 SVD：
  
  $$
  C_{\text{SVD}}(d) = O(d^3) \text{ FLOPs}
  $$
  
  对 MS：
  
  $$
  C_{\text{Muon}}^{MS} = C_{\text{SGD}}^{MS} + C_{\text{SVD}}(d)
  $$
  
  对 MF-$L$（每层矩阵都要做 SVD）：
  
  
  $$
  C_{\text{Muon}}^{MF} = C_{\text{SGD}}^{MF} + L \cdot C_{\text{SVD}}(d)
  $$

**定义 3.12（总 FLOPs 消耗）。**

$$
F_\epsilon^{(\mathcal{A})}(s) = K_\epsilon^{(\mathcal{A})}(s) \cdot C_{\mathcal{A}}^{\text{problem}} \cdot \mathbb{1}\left[ K_\epsilon^{(\mathcal{A})}(s) \leq K_{\max} \right] + \infty \cdot \mathbb{1}\left[ K_\epsilon^{(\mathcal{A})}(s) > K_{\max} \right]
$$

**定义 3.13（对数收敛轨迹的稳定性度量）。** 对 $R$ 次独立随机运行，定义对数误差序列：

$$
\ell_s^{(k)} = \log_{10}\left( f(\theta^{(k)}(s)) - f^\star + 10^{-16} \right), \quad s = 1, \ldots, R
$$

在第 $k$ 步的对数标准差为：

$$
\sigma_{\log}^{(k)} = \sqrt{\frac{1}{R-1} \sum_{s=1}^R \left( \ell_s^{(k)} - \bar{\ell}^{(k)} \right)^2}
$$

其中 $\bar{\ell}^{(k)} = \frac{1}{R} \sum_{s=1}^R \ell_s^{(k)}$。全局稳定性度量为：

$$
\bar{\sigma}_{\log} = \frac{1}{K_\epsilon} \sum_{k=1}^{K_\epsilon} \sigma_{\log}^{(k)}
$$

**定义 3.14（效率比）。** 对同一问题实例和同一随机种子 $s$，两种算法的效率比为：

- **迭代效率比：**
  
  $$
  \rho_K(s) = \frac{K_\epsilon^{\text{Muon}}(s)}{K_\epsilon^{\text{SGD}}(s)}
  $$

- **FLOPs 效率比：**
  
  $$
  \rho_F(s) = \frac{F_\epsilon^{\text{Muon}}(s)}{F_\epsilon^{\text{SGD}}(s)} = \rho_K(s) \cdot \frac{C_{\text{Muon}}}{C_{\text{SGD}}}
  $$

**定义 3.15（早停条件）。** 定义相邻迭代间的相对下降量：

$$
\delta_k = \frac{f(\theta^{(k)}) - f(\theta^{(k-1)})}{f(\theta^{(k-1)}) - f^\star + 10^{-16}}
$$

早停触发条件为连续 $w = 100$ 步满足：

$$
\delta_k > -0.001 \quad \text{（即下降不足 0.1%）}
$$

即若对某个 $k_0$，对所有 $k \in \{k_0, k_0+1, \ldots, k_0+w-1\}$ 有 $\delta_k > -0.001$，则在第 $k_0 + w - 1$ 步触发早停，并记录 $K_\epsilon = k_0 + w - 1$（标记为未收敛）。

---

## 8. 假设的统计形式化

本章将原始科学假设 H1–H5 转化为严格的统计假设对 $(H_0, H_1)$，每个假设对包括：零假设（$H_0$）、备择假设（$H_1$）、检验统计量、拒绝域和统计功效。

所有假设检验均在显著性水平 $\alpha = 0.05$ 下进行，除非特别说明。重复实验次数 $R = 10$。

---

### 4.1 H1：无条件收敛优势

**原始陈述：** 在矩阵感知和矩阵分解问题上，Muon 的迭代收敛速度始终优于 SGD。

**形式化：**

令 $\mathcal{P}_{test} \subseteq \mathcal{P}$ 为测试问题子集（涵盖所有维度、秩、深度配置）。对每个问题 $p \in \mathcal{P}_{test}$ 和随机种子 $s$，定义迭代次数差：

$$
\Delta_K(p, s) = K_\epsilon^{\text{Muon}}(p, s) - K_\epsilon^{\text{SGD}}(p, s)
$$

在 $R$ 次重复上取平均：

$$
\bar{\Delta}_K(p) = \frac{1}{R} \sum_{s=1}^R \Delta_K(p, s)
$$

**统计假设对 H1：**

$$
\begin{aligned}
H_0^{(1)}: & \quad \mathbb{E}[\Delta_K(p)] \geq 0 \quad \text{for at least one } p \in \mathcal{P}_{test} \\
H_1^{(1)}: & \quad \mathbb{E}[\Delta_K(p)] < 0 \quad \text{for all } p \in \mathcal{P}_{test}
\end{aligned}
$$

等价表述（以效率比 $\rho_K$）：

$$
\begin{aligned}
H_0^{(1)}: & \quad \exists \, p \in \mathcal{P}_{test}: \; \mathbb{E}[\rho_K(p)] \geq 1 \\
H_1^{(1)}: & \quad \forall \, p \in \mathcal{P}_{test}: \; \mathbb{E}[\rho_K(p)] < 1
\end{aligned}
$$

**检验统计量：**

对每个问题 $p$，使用配对 t 检验（paired one-sided t-test）：

$$
T^{(1)}(p) = \frac{\bar{\Delta}_K(p)}{\hat{\sigma}_{\Delta}(p) / \sqrt{R}}
$$

其中 $\hat{\sigma}_{\Delta}(p)$ 为 $\Delta_K(p, s)$ 的样本标准差。

**拒绝域：**

在自由度 $df = R - 1 = 9$ 下，拒绝 $H_0^{(1)}$ 当：

$$
T^{(1)}(p) < -t_{0.05, 9} = -1.833
$$

**多重检验校正：** 由于测试多个问题实例，使用 Bonferroni 校正或 Holm-Bonferroni 方法控制族错误率（FWER）在 $\alpha = 0.05$。

---

### 4.2 H2：谱结构条件优势

**原始陈述：** 当矩阵的谱范数与 Frobenius 范数之比 $\kappa_{sp}(X)$ 较大时，Muon 的优势更显著。

**形式化：**

**定义 4.1（谱集中度）。** 对任意矩阵 $X \in \mathbb{R}^{d \times d}$，定义谱集中度指标：

$$
\kappa_{sp}(X) = \frac{\|X\|_2 \cdot \sqrt{d}}{\|X\|_F} = \frac{\sigma_{\max}(X) \cdot \sqrt{d}}{\sqrt{\sum_{i=1}^d \sigma_i^2(X)}}
$$

其中 $\|X\|_2 = \sigma_{\max}(X)$ 为谱范数（最大奇异值）。由范数关系 $\|X\|_F \leq \sqrt{d} \|X\|_2$，知 $\kappa_{sp}(X) \in [1, \sqrt{d}]$。当 $\kappa_{sp}(X) \to 1$ 时表示能量均匀分布在所有奇异值方向；当 $\kappa_{sp}(X) \to \sqrt{d}$ 时表示能量集中在单一奇异值方向。

对目标矩阵 $X^\star$，计算其谱集中度 $\kappa_{sp}(X^\star)$，将所有测试问题按此值分为高低两组：

$$
\mathcal{P}_{\text{low}} = \left\{ p \; \middle| \; \kappa_{sp}(X^\star) \leq \kappa_{\text{med}} \right\}, \quad \mathcal{P}_{\text{high}} = \left\{ p \; \middle| \; \kappa_{sp}(X^\star) > \kappa_{\text{med}} \right\}
$$

其中 $\kappa_{\text{med}}$ 为所有测试问题谱集中度的中位数。

定义两组中 Muon 相对于 SGD 的平均效率增益：

$$
\bar{\rho}_K^{\text{low}} = \frac{1}{|\mathcal{P}_{\text{low}}|} \sum_{p \in \mathcal{P}_{\text{low}}} \mathbb{E}[\rho_K(p)], \quad \bar{\rho}_K^{\text{high}} = \frac{1}{|\mathcal{P}_{\text{high}}|} \sum_{p \in \mathcal{P}_{\text{high}}} \mathbb{E}[\rho_K(p)]
$$

**统计假设对 H2：**

$$
\begin{aligned}
H_0^{(2)}: & \quad \bar{\rho}_K^{\text{high}} \geq \bar{\rho}_K^{\text{low}} \\
H_1^{(2)}: & \quad \bar{\rho}_K^{\text{high}} < \bar{\rho}_K^{\text{low}}
\end{aligned}
$$

等价表述（以优势增益）：定义 $\gamma = \bar{\rho}_K^{\text{low}} - \bar{\rho}_K^{\text{high}}$，则：

$$
\begin{aligned}
H_0^{(2)}: & \quad \gamma \leq 0 \\
H_1^{(2)}: & \quad \gamma > 0
\end{aligned}
$$

**检验统计量：**

使用独立样本 t 检验（Welch's t-test，因两组方差可能不等）。令 $\{\rho_K(p) : p \in \mathcal{P}_{\text{low}}\}$ 和 $\{\rho_K(p) : p \in \mathcal{P}_{\text{high}}\}$ 为两组的效率比样本：

$$
T^{(2)} = \frac{\hat{\gamma}}{\sqrt{\frac{s_{\text{low}}^2}{n_{\text{low}}} + \frac{s_{\text{high}}^2}{n_{\text{high}}}}}
$$

其中 $\hat{\gamma} = \bar{\rho}_K^{\text{low}} - \bar{\rho}_K^{\text{high}}$，$s^2$ 为组内样本方差，$n = |\mathcal{P}|$。

**拒绝域：**

$$
T^{(2)} > t_{\alpha/2, df^\star}
$$

其中 $df^\star$ 由 Welch-Satterthwaite 方程近似计算。

---

### 4.3 H3：深度效应

**原始陈述：** 在矩阵分解问题中，随着分解层数 $L$ 增加，Muon 相对于 SGD 的优势增大。

**形式化：**

对 MF 问题，定义深度为 $L \in \{2, 3, 4\}$。对每个深度 $L$，计算该深度下所有维度 $d$ 和所有随机种子的平均效率比：

$$
\bar{\rho}_K(L) = \frac{1}{|\mathcal{D}_L| \cdot R} \sum_{d \in \{50,100,200,500\}} \sum_{s=1}^R \rho_K(d, L, s)
$$

**统计假设对 H3：**

检验优势是否随深度单调不减：

$$
\begin{aligned}
H_0^{(3)}: & \quad \bar{\rho}_K(4) \geq \bar{\rho}_K(3) \geq \bar{\rho}_K(2) \; \text{不成立} \\
H_1^{(3)}: & \quad \bar{\rho}_K(4) \leq \bar{\rho}_K(3) \leq \bar{\rho}_K(2) \quad \text{（Muon 优势随深度单调不减）}
\end{aligned}
$$

更精确的检验可分解为两个单侧检验：

- **H3a（$L=2 \to 3$）：**
  
  $$
  \begin{aligned}
  H_0^{(3a)}: & \quad \bar{\rho}_K(3) \geq \bar{\rho}_K(2) \\
  H_1^{(3a)}: & \quad \bar{\rho}_K(3) < \bar{\rho}_K(2)
  \end{aligned}
  $$

- **H3b（$L=3 \to 4$）：**
  
  $$
  \begin{aligned}
  H_0^{(3b)}: & \quad \bar{\rho}_K(4) \geq \bar{\rho}_K(3) \\
  H_1^{(3b)}: & \quad \bar{\rho}_K(4) < \bar{\rho}_K(3)
  \end{aligned}
  $$

**检验统计量：** 对每对深度使用配对 t 检验（同一 $d$ 和 $s$ 配对）：

$$
T^{(3a)} = \frac{\bar{\rho}_K(3) - \bar{\rho}_K(2)}{\hat{\sigma}_{2,3} / \sqrt{n_{pair}}}
$$

其中 $n_{pair} = 4 \times 10 = 40$（4 个维度 $\times$ 10 个种子）。

---

### 4.4 H4：稳定性优势

**原始陈述：** Muon 对随机初始化的敏感度低于 SGD（收敛轨迹更稳定）。

**形式化：**

对同一问题实例 $p$ 和两种算法，分别在 $R$ 次随机初始化上计算对数误差的标准差（定义 3.13）。

定义稳定性差异：

$$
\Delta_\sigma(p) = \bar{\sigma}_{\log}^{\text{Muon}}(p) - \bar{\sigma}_{\log}^{\text{SGD}}(p)
$$

**统计假设对 H4：**

$$
\begin{aligned}
H_0^{(4)}: & \quad \mathbb{E}[\Delta_\sigma(p)] \leq 0 \quad \text{（Muon 不更稳定）} \\
H_1^{(4)}: & \quad \mathbb{E}[\Delta_\sigma(p)] > 0 \quad \text{（Muon 更稳定，注意符号：更小的 } \sigma_{\log} \text{ 意味着更大的 } \Delta_\sigma \text{）}
\end{aligned}
$$

**修正后的形式化（更清晰的定义）：**

令稳定性度量 $S(\mathcal{A}) = 1 / \bar{\sigma}_{\log}^{(\mathcal{A})}$（越大越稳定），则：

$$
\begin{aligned}
H_0^{(4)}: & \quad \mathbb{E}[S(\text{Muon})] \leq \mathbb{E}[S(\text{SGD})] \\
H_1^{(4)}: & \quad \mathbb{E}[S(\text{Muon})] > \mathbb{E}[S(\text{SGD})]
\end{aligned}
$$

等价地，以标准差之比（变异系数形式）定义：

$$
\begin{aligned}
H_0^{(4)}: & \quad \mathbb{E}\left[\frac{\bar{\sigma}_{\log}^{\text{Muon}}}{\bar{\sigma}_{\log}^{\text{SGD}}}\right] \geq 1 \\
H_1^{(4)}: & \quad \mathbb{E}\left[\frac{\bar{\sigma}_{\log}^{\text{Muon}}}{\bar{\sigma}_{\log}^{\text{SGD}}}\right] < 1
\end{aligned}
$$

**检验统计量：** 配对 t 检验（同一问题实例、同一随机种子、不同算法）：

$$
T^{(4)} = \frac{\bar{r}_\sigma - 1}{\hat{\sigma}_{r_\sigma} / \sqrt{R}}
$$

其中 $r_\sigma(s) = \bar{\sigma}_{\log}^{\text{Muon}}(s) / \bar{\sigma}_{\log}^{\text{SGD}}(s)$。

---

### 4.5 H5：FLOPs 效率优势

**原始陈述：** 即使 Muon 单次迭代计算量更大，其在总 FLOPs 意义上的效率仍优于 SGD。

**形式化：**

对每次运行 $s$ 计算 FLOPs 效率比 $\rho_F(s)$（定义 3.14）。

**统计假设对 H5：**

$$
\begin{aligned}
H_0^{(5)}: & \quad \mathbb{E}[\rho_F] \geq 1 \quad \text{（Muon 在总 FLOPs 意义上不优于 SGD）} \\
H_1^{(5)}: & \quad \mathbb{E}[\rho_F] < 1 \quad \text{（Muon 在总 FLOPs 意义上优于 SGD）}
\end{aligned}
$$

**检验统计量：** 对配对样本 $\{\rho_F(s)\}_{s=1}^R$ 做单侧 t 检验：

$$
T^{(5)} = \frac{\bar{\rho}_F - 1}{\hat{\sigma}_{\rho_F} / \sqrt{R}}
$$

**拒绝域：**

$$
T^{(5)} < -t_{0.05, R-1} = -1.833
$$

---

## 9. 实验流程的数学描述

本章将实验执行流程形式化为四个阶段（Phase 1–4），每个阶段有明确的数学输入、输出和变换规则。

---

### 5.1 Phase 1：学习率校准

**目的**：为每个（问题，算法）对找到最优学习率 $\eta^\star$。

**输入**：
- 问题实例 $p \in \mathcal{P}$
- 算法 $\mathcal{A} \in \{\text{Muon}, \text{SGD}\}$
- 候选学习率集合 $\mathcal{H} = \{10^{-3}, 10^{-2}, 10^{-1}\}$
- 固定随机种子 $s_0 = 42$（校准种子）

**过程**：

对每个 $\eta \in \mathcal{H}$，执行 $K_{\max}$ 步迭代：

$$
\theta^{(k+1)} = \mathcal{T}_{\eta}^{\mathcal{A}}(\theta^{(k)}, \nabla f(\theta^{(k)})), \quad k = 0, \ldots, K_{\max}-1
$$

记录收敛轨迹：

$$\mathcal{T}(\eta) = \{ (k, f(\theta^{(k)})) \}_{k=0}^{K_{\max}}
$$

**输出**：

$$
\eta^\star(p, \mathcal{A}) = \arg\min_{\eta \in \mathcal{H}} K_\epsilon^{(\mathcal{A})}(p, s_0; \eta)
$$

其中 $K_\epsilon$ 为达到精度 $\epsilon$ 所需迭代次数。若多个 $\eta$ 达到相同最小迭代，取最小 $\eta$（更保守）。

**校准结果记录**：

$$
\mathcal{C} = \left\{ (p, \mathcal{A}, \eta^\star(p, \mathcal{A})) \; \middle| \; p \in \mathcal{P}_{\text{cal}}, \mathcal{A} \in \mathcal{A} \right\}
$$

其中 $\mathcal{P}_{\text{cal}} \subseteq \mathcal{P}$ 为校准子集（通常取代表性维度如 $d = 100$）。

**注**：本次实验对学习率做网格搜索，对每种配置取最佳 $\eta$。

---

### 5.2 Phase 2：正式对比实验

**目的**：在控制随机性的前提下，系统对比两种算法的性能。

**输入**：
- 完整问题实例空间 $\mathcal{P}$
- 算法空间 $\mathcal{A}$
- 校准后的学习率 $\eta^\star(p, \mathcal{A})$
- 随机种子集合 $\mathcal{R}_{\text{seed}} = \{1, \ldots, 10\}$
- 评估指标空间 $\mathcal{M}$

**随机化设计**：采用**配对随机化设计**（paired randomized design），即对同一问题实例和同一随机种子，两种算法使用完全相同的数据和初始化：

$$
(\theta^{(0)}, \{A_i\}, y) = \Phi_{MS}(s) \quad \text{或} \quad (\{W_i^{(0)}\}, X^\star) = \Phi_{MF}(s)
$$

然后分别运行 Muon 和 SGD：

$$
\begin{aligned}
\theta_{\text{Muon}}^{(k+1)} &= \mathcal{T}_{\eta^\star}^{\text{Muon}}(\theta_{\text{Muon}}^{(k)}, G^{(k)}) \\
\theta_{\text{SGD}}^{(k+1)} &= \mathcal{T}_{\eta^\star}^{\text{SGD}}(\theta_{\text{SGD}}^{(k)}, G^{(k)})
\end{aligned}
$$

注意：在 MF 问题中，由于参数维度相同，可使用同一初始化；在 MS 问题中，初始化相同。

**输出**：对每个 $(p, \mathcal{A}, s)$ 三元组，记录：

$$
\mathcal{D}_{\text{raw}} = \left\{ (p, \mathcal{A}, s, \{f(\theta^{(k)}\}_{k=0}^{K_\epsilon}, K_\epsilon, F_\epsilon, \bar{\sigma}_{\log}) \right\}
$$

---

### 5.3 Phase 3：数据聚合与统计计算

**目的**：从原始数据计算检验统计量。

**过程**：

**步骤 1：按问题聚合。**

对每个问题 $p$ 和算法 $\mathcal{A}$，计算：

$$
\bar{K}_\epsilon^{(\mathcal{A})}(p) = \frac{1}{R} \sum_{s=1}^R K_\epsilon^{(\mathcal{A})}(p, s), \quad \hat{\sigma}_{K}^{(\mathcal{A})}(p) = \sqrt{\frac{1}{R-1} \sum_{s=1}^R \left(K_\epsilon^{(\mathcal{A})}(p, s) - \bar{K}_\epsilon^{(\mathcal{A})}(p)\right)^2}
$$

$$
\bar{F}_\epsilon^{(\mathcal{A})}(p) = \frac{1}{R} \sum_{s=1}^R F_\epsilon^{(\mathcal{A})}(p, s)
$$

**步骤 2：计算效率比。**

对每个问题 $p$ 和种子 $s$：

$$
\rho_K(p, s) = \frac{K_\epsilon^{\text{Muon}}(p, s)}{K_\epsilon^{\text{SGD}}(p, s)}, \quad \rho_F(p, s) = \frac{F_\epsilon^{\text{Muon}}(p, s)}{F_\epsilon^{\text{SGD}}(p, s)}
$$

然后计算：

$$
\bar{\rho}_K(p) = \frac{1}{R} \sum_{s=1}^R \rho_K(p, s), \quad \bar{\rho}_F(p) = \frac{1}{R} \sum_{s=1}^R \rho_F(p, s)
$$

**步骤 3：计算稳定性度量。**

对每个算法、每个问题、每一步 $k$：

$$
\sigma_{\log}^{(\mathcal{A})}(p, k) = \sqrt{\frac{1}{R-1} \sum_{s=1}^R \left(\ell_s^{(k)} - \bar{\ell}^{(k)}\right)^2}
$$

然后：

$$
\bar{\sigma}_{\log}^{(\mathcal{A})}(p) = \frac{1}{\bar{K}_\epsilon^{(\mathcal{A})}(p)} \sum_{k=1}^{\bar{K}_\epsilon^{(\mathcal{A})}(p)} \sigma_{\log}^{(\mathcal{A})}(p, k)
$$

**输出**：聚合数据集

$$
\mathcal{D}_{\text{agg}} = \left\{ (p, \bar{K}_\epsilon^{\text{Muon}}, \bar{K}_\epsilon^{\text{SGD}}, \bar{\rho}_K, \bar{\rho}_F, \bar{\sigma}_{\log}^{\text{Muon}}, \bar{\sigma}_{\log}^{\text{SGD}}) \right\}_{p \in \mathcal{P}}
$$

---

### 5.4 Phase 4：假设检验与推断

**目的**：基于聚合数据对 H1–H5 执行统计检验。

**过程**：

对每个假设 $H_i$，$i = 1, \ldots, 5$：

**步骤 1：构造检验统计量。**

依据第4节定义，从 $\mathcal{D}_{\text{agg}}$ 和 $\mathcal{D}_{\text{raw}}$ 构造 $T^{(i)}$。

**步骤 2：确定零分布。**

在 $H_0^{(i)}$ 下，$T^{(i)}$ 服从（渐近或精确）：

- H1, H4, H5：t 分布，$df = R - 1 = 9$
- H2：Welch's t 分布，近似自由度
- H3：配对 t 分布，$df = n_{pair} - 1 = 39$

**步骤 3：计算 p 值。**

$$
p^{(i)} = \mathbb{P}(T \geq T_{\text{obs}}^{(i)} \mid H_0^{(i)}) \quad \text{或} \quad \mathbb{P}(T \leq T_{\text{obs}}^{(i)} \mid H_0^{(i)})
$$

依备择假设方向而定。

**步骤 4：多重检验校正。**

对 5 个假设使用 Holm-Bonferroni 程序：

1. 将 p 值排序：$p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(5)}$
2. 找到最小 $j$ 使得 $p_{(j)} > \alpha / (5 - j + 1)$
3. 拒绝所有 $p_{(i)}$ 满足 $i < j$

**步骤 5：效应量计算。**

对每个被拒绝的假设，报告 Cohen's d 效应量：

$$
d^{(i)} = \frac{|\bar{\Delta}^{(i)}|}{\hat{\sigma}^{(i)}}
$$

其中 $\bar{\Delta}^{(i)}$ 为对应差异的样本均值，$\hat{\sigma}^{(i)}$ 为合并标准差。

**输出**：假设检验报告

$$
\mathcal{R}_{\text{report}} = \left\{ (H_i, T_{\text{obs}}^{(i)}, p^{(i)}, \text{decision}, d^{(i)}) \right\}_{i=1}^5
$$

其中 $\text{decision} \in \{\text{Reject } H_0, \text{Fail to reject } H_0\}$。

---

## 10. 现有实验设计的边界与假设清单

本节系统列出本实验方案中所有**显式或隐式的假设、约束和边界条件**。这些边界决定了实验结论的外部效度（external validity）和可推广范围。

---

### 6.1 问题空间边界

| 编号 | 边界条件 | 数学描述 | 影响 |
|:---:|:---|:---|:---|
| B1 | 维度范围有限 | $d \in \{50, 100, 200, 500\}$，未覆盖 $d < 50$ 或 $d > 500$ | 极小或极大维度的结论不可外推 |
| B2 | 方阵假设 | 所有矩阵均为 $d \times d$ 方阵 | 矩形矩阵（$m \times n$）的结论需额外验证 |
| B3 | 满秩或低秩 | $r \in \{5, d/2, d\}$，未覆盖中间秩 | 中间秩 $r \in (5, d/2)$ 的行为未知 |
| B4 | 测量数固定 | MS 问题固定 $m = 3d^2$，未做过采样/欠采样扫描 | 样本复杂度影响不可推断 |
| B5 | 分解深度有限 | $L \in \{2, 3, 4\}$ | 更深网络（$L \geq 5$）的行为未知 |
| B6 | 无约束优化 | 无等式/不等式约束 | 带约束问题的结论不可直接推广 |

---

### 6.2 数据分布假设

| 编号 | 假设 | 数学描述 | 违反后果 |
|:---:|:---|:---|:---|
| A1 | 高斯测量矩阵 | $A_{ij} \sim \mathcal{N}(0,1)$ iid | 非高斯测量（如 Rademacher）可能改变条件数分布 |
| A2 | 高斯真实矩阵 | $X^\star_{ij} \sim \mathcal{N}(0,1)$ iid | 结构化矩阵（如稀疏、低相干）行为不同 |
| A3 | 加性高斯噪声 | $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$ | 非高斯噪声（如重尾分布）影响最优值和收敛 |
| A4 | 各向同性初始化 | $W_{ij}^{(0)} \sim \mathcal{N}(0, 1/d)$ iid | 非各向同性或特定结构初始化可能改变收敛 |
| A5 | 同分布样本 | 所有随机变量同分布 | 分布偏移导致外推失效 |

---

### 6.3 算法假设

| 编号 | 假设 | 数学描述 | 违反后果 |
|:---:|:---|:---|:---|
| A6 | 全批量梯度 | $G^{(k)} = \nabla f(\theta^{(k)})$，无 mini-batch 采样 | mini-batch SGD 的随机梯度方差可能改变相对优势 |
| A7 | 固定学习率 | $\eta^{(k)} = \eta$ 常数 | 学习率调度（如 cosine decay）可能改变收敛动力学 |
| A8 | 无动量 | Muon 和 SGD 均不使用一阶动量 | 带 momentum 的 SGD 或 Adam 可能显著不同 |
| A9 | 无梯度裁剪 | 原始梯度直接使用 | 梯度爆炸情形可能需要裁剪 |
| A10 | 权重衰减为零 | $\lambda = 0$ | 非零权重衰减改变目标函数（增加正则化项） |
| A11 | SVD 精确计算 | Muon 的 SVD 为精确计算 | 近似 SVD 可能引入数值误差和额外 FLOPs |
| A12 | 单精度浮点 | 计算在 float32 精度下 | float64 或 float16 的数值行为不同 |

---

### 6.4 评估指标假设

| 编号 | 假设 | 数学描述 | 违反后果 |
|:---:|:---|:---|:---|
| A13 | 目标精度可达 | $\exists \, K \leq K_{\max}: f(\theta^{(K)}) - f^\star \leq \epsilon$ | 若算法不收敛，$K_\epsilon$ 的截断值人为引入偏差 |
| A14 | 早停有效 | 早停窗口 $w = 100$ 能检测真实停滞 | 窗口过小可能过早停止，过大浪费时间 |
| A15 | 对数变换合理 | $\log(f - f^\star)$ 有意义（$f > f^\star$） | 若 $f$ 低于 $f^\star$（数值误差），对数无定义 |
| A16 | FLOPs 模型准确 | $C_{\text{Muon}} = C_{\text{SGD}} + C_{\text{SVD}}$ 精确 | 实际实现可能有缓存、并行等优化改变真实 FLOPs |
| A17 | 收敛是单调的 | $f(\theta^{(k+1)}) \leq f(\theta^{(k)})$ | 非单调收敛（如振荡）使早停和精度检测复杂化 |

---

### 6.5 统计假设

| 编号 | 假设 | 数学描述 | 违反后果 |
|:---:|:---|:---|:---|
| A18 | 配对差异正态 | $\Delta_K(s) \overset{iid}{\sim} \mathcal{N}(\mu, \sigma^2)$ 近似成立 | 严重非正态或小样本下 t 检验不可靠；可用非参数检验（Wilcoxon）补充 |
| A19 | 独立种子 | 不同种子 $s_1 \neq s_2$ 产生独立样本 | 若 RNG 周期不足或存在相关性，标准误低估 |
| A20 | 问题间独立性 | 不同问题实例独立 | 问题共享维度参数导致结构相关性 |
| A21 | 同质方差 | $\mathrm{Var}(\Delta_K)$ 在不同问题上相近 | 异质方差下 t 检验有偏；已用 Welch 校正部分缓解 |
| A22 | 显著性水平适当 | $\alpha = 0.05$ 平衡 I 型和 II 型错误 | 更严格 $\alpha$ 降低功效，更宽松增加假阳性 |

---

### 6.6 结论推广边界

基于以上边界和假设，本实验结论的有效推广范围为：

$$
\mathcal{G} = \left\{ \text{问题} \; \middle| \; \begin{array}{l} \text{方阵、无约束、无正则化、} \\ \text{全批量梯度、固定学习率、} \\ \text{高斯测量/初始化、} \\ \text{维度 } d \in [50, 500] \end{array} \right\}
$$

任何超出 $\mathcal{G}$ 的问题（如 mini-batch 训练、学习率调度、非高斯数据、带约束优化）均需额外实验验证，不可直接由本实验结论外推。

---

## 附录：符号速查表

| 符号 | 含义 | 定义域 |
|:---|:---|:---|
| $\mathcal{E}$ | 实验五元组 | $(\mathcal{P}, \mathcal{A}, \mathcal{D}, \mathcal{M}, \mathcal{R})$ |
| $\mathcal{P}$ | 问题实例空间 | $\mathcal{P}_{MS} \cup \mathcal{P}_{MF}$ |
| $\mathcal{A}$ | 算法空间 | $\{\text{Muon}, \text{SGD}\}$ |
| $\mathcal{D}$ | 数据/超参数空间 | $\mathcal{D}_{\text{data}} \times \mathcal{D}_{\text{init}} \times \mathcal{D}_{\text{hyper}}$ |
| $\mathcal{M}$ | 度量空间 | $\{K_\epsilon, F_\epsilon, \bar{\sigma}_{\log}, \rho_K, \rho_F\}$ |
| $\mathcal{R}$ | 随机性空间 | 种子 $\times$ 问题 $\times$ 噪声 $\times$ 初始化 |
| $d$ | 矩阵维度 | $\{50, 100, 200, 500\}$ |
| $r$ | 矩阵秩 | $\{5, d/2, d\}$ |
| $L$ | 分解深度 | $\{2, 3, 4\}$ |
| $m$ | 测量数 | $3d^2$ |
| $\eta$ | 学习率 | $\{10^{-3}, 10^{-2}, 10^{-1}\}$ |
| $\lambda$ | 权重衰减 | $0$ |
| $\epsilon$ | 目标精度 | $10^{-6}$ |
| $K_{\max}$ | 最大迭代 | $10^5$ |
| $R$ | 重复次数 | $10$ |
| $s$ | 随机种子 | $\{1, \ldots, 10\}$ |
| $\rho_K$ | 迭代效率比 | $\mathbb{R}^+$ |
| $\rho_F$ | FLOPs 效率比 | $\mathbb{R}^+$ |
| $\kappa_{sp}$ | 谱集中度 | $[1, \sqrt{d}]$ |
| $\mathcal{S}(G)$ | SVD 归一化算子 | $G = U\Sigma V^T \mapsto UV^T$ |
| $\mathcal{T}_{\eta}^{\mathcal{A}}$ | 算法迭代算子 | $\Theta \to \Theta$ |
| $\Phi$ | 数据生成映射 | $\mathcal{R}_{\text{seed}} \times \mathcal{P} \to \mathcal{D}$ |

---

> **本章编写完成。** 以上形式化定义构成了 Muon vs SGD 对比实验的完整数学基础，所有后续实现、执行和推断均应严格参照本章符号与定义。

---

# 第三部分：统计变量体系

> 本部分构建覆盖实验设计、执行、分析全链条的完整统计变量体系。在第二部分实验形式化的框架下，进一步精确定义所有因子、响应变量、协变量、随机变量和派生统计量，并建立变量间的因果结构与函数依赖关系。本部分可直接作为实验代码的统计规范引用。

> 版本：v1.0  
> 适用范围：矩阵感知（Matrix Sensing, MS）与矩阵分解（Matrix Factorization, MF）优化问题的算法对比实验  
> 设计原则：可复现、可量化、可推断、可扩展

---

## 11. 因子与处理变量（Factors & Treatments）

### 1.1 因子分类总览

本实验采用**混合析因设计（Mixed Factorial Design）**，因子可分为三类：
- **算法因子**：决定优化器行为的离散变量
- **问题因子**：定义优化问题实例的结构变量
- **环境因子**：影响计算过程但不改变问题本质的变量

### 1.2 因子定义详表

| 符号 | 名称 | 类型 | 取值域 / 水平集 | 在实验中的作用 | 补充建议 |
|:---:|:---|:---:|:---|:---|:---|
| $\alpha$ | **算法因子** | 分类 (Categorical) | $\{\text{Muon}, \text{SGD}\}$ | 主效应：检验两种优化器的系统性差异 | 可扩展至 $\{\text{Muon}, \text{SGD}, \text{Adam}, \text{AdamW}\}$ |
| $\pi$ | **问题类型因子** | 分类 (Categorical) | $\{\text{MS}, \text{MF}\}$ | 主效应 + 与 $\alpha$ 的交互效应 | 可加入矩阵补全（MC）、鲁棒PCA |
| $d$ | **矩阵尺寸** | 有序整数 (Ordinal) | $\{50, 100, 200, 500\}$ | 主效应 + 与 $\alpha, \pi$ 的交互 | **补充** $d \in \{1000, 2000\}$ 用于大规模外推；补充矩形 $d_1 \times d_2$ |
| $r$ | **目标秩** | 整数 (Discrete) | $\{d/10, d\}$（低秩 vs 满秩） | 主效应 + 与 $\alpha$ 的交互（Muon 对低秩结构敏感） | **补充** $r \in \{d/20, d/5, d/2\}$ 用于秩敏感性分析 |
| $L$ | **分解深度** | 有序整数 (Ordinal) | $\{2, 3, 4\}$（仅 MF） | 主效应（MF 子实验中）+ 与 $\alpha$ 的交互 | **补充** $L=5, 6$ 用于深度扩展；$L=1$ 退化基线 |
| $\iota$ | **初始化模式** | 分类 (Categorical) | $\{\text{Spectral}, \text{Gaussian}, \text{Orthogonal}\}$（MF, $L=2$ 时） | 主效应 + 与 $\alpha$ 的交互 | **补充** Xavier、He、小范数初始化；MF 深度 $L \geq 3$ 时的平衡初始化 |
| $\eta$ | **学习率** | 连续 (Continuous) | $\{10^{-3}, 10^{-2}, 10^{-1}\}$（网格搜索） | **调节变量 / Nuisance Variable**：每个 $(\alpha, \pi, d)$ 配置下选择最优 $\eta$ | 补充对数均匀搜索 $\eta \sim \text{LogUniform}(10^{-4}, 10^{0})$；或贝叶斯优化 |
| $\lambda$ | **权重衰减** | 连续 (Continuous) | $\{0\}$（当前设定） | 控制变量（当前固定） | **补充** $\lambda \in \{10^{-4}, 10^{-3}, 10^{-2}\}$ 以检验正则化效应 |
| $\sigma_\epsilon$ | **噪声水平** | 连续 (Continuous) | $\{0\}$（当前设定） | 控制变量（当前固定） | **补充** $\sigma_\epsilon \in \{10^{-4}, 10^{-3}, 10^{-2}\} \cdot \|y\|$ 用于噪声鲁棒性分析 |
| $\rho_{os}$ | **过采样因子** | 连续 (Continuous) | $m / d^2$（MS 时） | 协变量：影响问题可辨识性与收敛速率 | 固定为 $m = 5rd$（低秩）或 $m = 2d^2$（满秩），或作为扫描因子 |
| $\kappa$ | **矩阵形状** | 分类 (Categorical) | $\{\text{Square}, \text{Rectangular}\}$ | 扩展因子 | **新增**：矩形矩阵 $d_1 \times d_2$（如 $100 \times 50$）检验算法对非方阵的适应性 |
| $\beta$ | **批大小** | 有序整数 (Ordinal) | $\{1, \text{full-batch}\}$ | 控制变量（默认全批量） | 补充 mini-batch $B \in \{32, 128, 512\}$ 用于随机梯度场景 |
| $T_{\max}$ | **最大迭代数** | 整数 (Discrete) | $10^5$ | 实验截断参数 | 可按 $d$ 比例调整：$T_{\max} = 10^4 \cdot (d/50)$ |

### 1.3 因子间的结构约束

部分因子组合存在**结构零（Structural Zeros）**或**嵌套关系（Nesting）**：

```
π = MS  →  L 未定义（无分解深度）
π = MF  →  ρ_os 未定义（无过采样）
r = d   →  低秩特定参数无效
L = 2   →  ι 激活（三种初始化模式）
L ≥ 3   →  ι 退化为单一平衡初始化
```

**嵌套因子表**：

| 外层因子 | 内层因子 | 嵌套关系说明 |
|:---|:---|:---|
| $\pi = \text{MS}$ | $L$ | $L$ 在 MS 下不存在（记为 $L = \varnothing$） |
| $\pi = \text{MF}$ | $\rho_{os}$ | 过采样因子在 MF 下不存在 |
| $r < d$ | $\rho_{os}$ | 过采样因子仅在低秩 MS 中有意义 |
| $L = 2$ | $\iota$ | 三种初始化模式仅在 $L=2$ 时对比 |

### 1.4 实验配置空间大小

**基础配置空间**（仅考虑活跃因子组合）：

$$
\begin{aligned}
|\Omega_{\text{base}}| &= |\alpha| \times \Big( |\text{MS}| + |\text{MF}| \Big) \\
&= 2 \times \Big[ |d| \times |r| \times |\rho_{os}| + |d| \times |L_{\text{active}}| \times |\iota_{\text{active}}| \Big] \\
&= 2 \times \Big[ 4 \times 2 \times 1 + 4 \times 3 \times 1 \Big] \\
&= 2 \times (8 + 12) = 40 \text{（基础因子组合）}
\end{aligned}
$$

考虑到 $\eta$ 的网格搜索（$|\eta| = 3$）与随机种子重复（$n_{seed} = 10$），**总实验运行次数**：

$$
N_{\text{runs}} = 40 \times 3 \times 10 = 1200 \text{ 次独立运行}
$$

---

## 12. 响应变量（Response Variables）

### 2.1 响应变量分级体系

| 级别 | 类别 | 变量数量 | 用途 |
|:---:|:---|:---:|:---|
| **一级** | Primary Endpoints | 4 | 核心科学结论支撑 |
| **二级** | Secondary Endpoints | 5 | 辅助解释与机制分析 |
| **三级** | Exploratory Endpoints | 4 | 假设生成与深入分析 |

### 2.2 主响应变量（Primary Response Variables）

| 符号 | 名称 | 数学定义 | 取值域 | 类型 | 分布假设 | 实验用途 |
|:---:|:---|:---|:---:|:---:|:---:|:---|
| $K_\epsilon$ | **收敛迭代数** | $K_\epsilon = \min\{ k \geq 0 : f_k \leq \epsilon \}$ | $\{0, 1, \ldots, T_{\max}\}$ | 离散 | 右偏分布，近似对数正态 | **主响应**：算法效率的核心度量 |
| $F_\epsilon$ | **收敛 FLOPs** | $F_\epsilon = \sum_{k=0}^{K_\epsilon-1} \mathcal{C}_k$ | $\mathbb{R}_{\geq 0}$ | 连续 | 右偏分布，近似对数正态 | **主响应**：计算成本的物理度量 |
| $\mathbb{I}_{\text{conv}}$ | **收敛标志** | $\mathbb{I}_{\text{conv}} = \mathbf{1}_{\{K_\epsilon < T_{\max}\}}$ | $\{0, 1\}$ | 二元 | Bernoulli($p_{\text{conv}}$) | **主响应**：算法可靠性的二元指标 |
| $\bar{\sigma}_{\log}$ | **对数收敛稳定性** | $\bar{\sigma}_{\log} = \frac{1}{K_\epsilon} \sum_{k=1}^{K_\epsilon} \sigma_{\log}(k)$ | $\mathbb{R}_{\geq 0}$ | 连续 | 未知，需非参数方法 | **主响应**：收敛路径的稳定性度量 |

其中：
- $f_k = f(X_k)$ 为第 $k$ 步的目标函数值
- $\epsilon = 10^{-6}$ 为收敛阈值
- $\mathcal{C}_k$ 为第 $k$ 步的计算成本（FLOPs）
- $\sigma_{\log}(k)$ 为第 $k$ 步跨种子对数目标值的样本标准差

### 2.3 次响应变量（Secondary Response Variables）

| 符号 | 名称 | 数学定义 | 取值域 | 类型 | 分布假设 | 实验用途 |
|:---:|:---|:---|:---:|:---:|:---:|:---|
| $T_{\text{wall}}$ | **Wall-clock 时间** | $T_{\text{wall}} = t_{\text{end}} - t_{\text{start}}$ | $\mathbb{R}_{\geq 0}$ | 连续 | 右偏，依赖硬件 | 实际工程部署成本 |
| $M_{\text{peak}}$ | **峰值内存占用** | $M_{\text{peak}} = \max_k \text{mem}(X_k)$ | $\mathbb{R}_{\geq 0}$（MB/GB） | 连续 | 近似常数 + 噪声 | 资源约束场景评估 |
| $f_{\text{final}}$ | **最终精度** | $f_{\text{final}} = f_{K_\epsilon}$（若收敛）或 $f_{T_{\max}}$（若截断） | $\mathbb{R}_{\geq 0}$ | 连续 | 右偏 | 不收敛时的替代指标 |
| $\|\nabla f\|_{\text{final}}$ | **最终梯度范数** | $\|\nabla f\|_{\text{final}} = \|\nabla f(X_{K_\epsilon})\|_F$ | $\mathbb{R}_{\geq 0}$ | 连续 | 右偏 | 一阶最优性条件检验 |
| $\Delta_{\text{sp}}$ | **谱比变化量** | $\Delta_{\text{sp}} = \kappa_{\text{sp}}(X^{(0)}) - \kappa_{\text{sp}}(X^{(K)})$ | $\mathbb{R}$ | 连续 | 未知 | 算法对谱结构的隐式正则化效应 |

### 2.4 探索性响应变量（Exploratory Response Variables）

| 符号 | 名称 | 数学定义 | 取值域 | 类型 | 实验用途 |
|:---:|:---|:---|:---:|:---:|:---|
| $\{X^{(k)}\}_{k=0}^{T}$ | **参数轨迹** | 完整迭代序列 | $\mathbb{R}^{d \times d \times (T+1)}$ | 序列 | 收敛路径可视化、动力学分析 |
| $\{\lambda_i(H_k)\}_{i=1}^{d^2}$ | **Hessian 谱** | Hessian 矩阵特征值 | $\mathbb{R}^{d^2}$ | 向量 | 局部几何分析、条件数演化 |
| $\{\sigma_i(X_k)\}_{i=1}^{d}$ | **奇异值轨迹** | SVD 奇异值序列 | $\mathbb{R}_{\geq 0}^{d \times (T+1)}$ | 矩阵 | 谱动态、隐式低秩正则化 |
| $\mathcal{R}(k)$ | **有效秩** | $\mathcal{R}(k) = \|X_k\|_F^2 / \|X_k\|_2^2$ | $[1, d]$ | 连续 | 数值秩演化、隐式秩压缩 |

### 2.5 响应变量的数学关系

**FLOPs 与迭代数的函数依赖**：

$$
F_\epsilon(\alpha, \pi, d) = \mathcal{C}_{\text{per-iter}}(\alpha, \pi, d) \times K_\epsilon(\alpha, \pi, d)
$$

其中每步计算成本：

- **Muon**（MS）：$\mathcal{C}_{\text{Muon}}^{\text{MS}} = \mathcal{O}(m \cdot d^2) + \mathcal{O}(d^3)$（含谱归一化 SVD）
- **Muon**（MF）：$\mathcal{C}_{\text{Muon}}^{\text{MF}} = \mathcal{O}(L \cdot d^3)$（深度分解的链式 SVD）
- **SGD**（MS/MF）：$\mathcal{C}_{\text{SGD}} = \mathcal{O}(m \cdot d^2)$ 或 $\mathcal{O}(L \cdot d^3)$（无额外 SVD 开销）

**梯度范数与收敛迭代数的关系**（期望意义下）：

$$
\mathbb{E}[K_\epsilon] \propto \frac{\log(f_0 / \epsilon)}{\log(1 / \rho_{\text{conv}})}
$$

其中 $\rho_{\text{conv}}$ 为每步收敛比率（见 §5 派生统计量）。

---

## 13. 协变量与控制变量（Covariates & Controls）

### 3.1 协变量分类

| 层级 | 说明 | 用途 |
|:---:|:---|:---|
| **问题固有协变量** | 由问题实例决定，算法无法改变 | 分层分析、回归调整 |
| **初始状态协变量** | 由初始化决定，记录用于调整 | 方差缩减、因果推断 |
| **环境协变量** | 实验运行环境信息 | 异质性控制、元分析 |

### 3.2 协变量详表

| 符号 | 名称 | 数学定义 | 取值域 | 类型 | 分布假设 | 实验用途 |
|:---:|:---|:---|:---:|:---:|:---:|:---|
| $\kappa_{\text{sp}}^{(0)}$ | **初始谱比** | $\kappa_{\text{sp}}^{(0)} = \|X^{(0)}\|_2 / \|X^{(0)}\|_F$ | $[1/\sqrt{d}, 1]$ | 连续 | 依赖初始化 | 预测收敛速率；Muon 对谱比敏感 |
| $\kappa_{\text{cond}}^{(0)}$ | **初始条件数** | $\kappa_{\text{cond}}^{(0)} = \sigma_1(X^{(0)}) / \sigma_d(X^{(0)})$ | $[1, \infty)$ | 连续 | 依赖初始化 | 问题病态程度指标；影响收敛稳定性 |
| $\kappa_{\text{cond}}^{\star}$ | **真实条件数** | $\kappa_{\text{cond}}^{\star} = \sigma_1(X^\star) / \sigma_r(X^\star)$（低秩） | $[1, \infty)$ | 连续 | 依赖 $X^\star$ 构造 | 问题本质难度 |
| $r_\epsilon$ | **数值 $\epsilon$-秩** | $r_\epsilon = |\{i : \sigma_i(X^\star) > \epsilon\}|$ | $\{1, \ldots, d\}$ | 离散 | 依赖 $X^\star$ 谱衰减 | 与目标秩 $r$ 对比评估过参数化 |
| $\mu_{\text{SR}}$ | **强凸性参数**（MS） | $\mu_{\text{SR}} = \lambda_{\min}(\frac{1}{m}\sum_i \text{vec}(A_i)\text{vec}(A_i)^\top)$ | $\mathbb{R}_{>0}$ | 连续 | 依赖测量矩阵 | MS 问题条件数：$\kappa_{\text{MS}} = M_{\text{SR}} / \mu_{\text{SR}}$ |
| $s$ | **随机种子** | $s \in \{0, 1, \ldots, 9\}$ | $\{0, \ldots, 9\}$ | 离散 | 均匀分布 | 实现独立重复；固定效应模型中的区组因子 |
| $\text{HW}$ | **硬件配置** | CPU/GPU 型号、内存、PyTorch 版本 | 标签向量 | 分类 | 实验内恒定 | 元回归中的调节变量；跨实验可比性 |
| $\tau_{\text{float}}$ | **浮点精度** | $\{\text{float32}, \text{float64}\}$ | 二元 | 分类 | 实验内恒定 | 数值稳定性控制 |
| $\delta_{\text{ortho}}$ | **正交性偏差**（MF, L≥3） | $\delta_{\text{ortho}} = \sum_{\ell=1}^{L-1} \|W_{\ell+1}^\top W_{\ell+1} - W_\ell W_\ell^\top\|_F$ | $\mathbb{R}_{\geq 0}$ | 连续 | 随 $L$ 增长 | 深度 MF 的平衡性度量；影响收敛动力学 |

### 3.3 控制变量的固定策略

| 变量 | 固定值 | 理由 | 未来扩展 |
|:---:|:---|:---|:---|
| $\lambda$（权重衰减） | $0$ | 隔离算法本身效应 | 见 §1.2 补充非零水平 |
| $\sigma_\epsilon$（噪声） | $0$ | 先研究确定性收敛 | 见 §1.2 补充噪声鲁棒性实验 |
| $\beta$（批大小） | Full-batch | 消除随机梯度方差 | 补充 mini-batch 实验 |
| $\tau_{\text{float}}$ | float64 | 保证数值精度 | 补充 float32 实验评估数值稳定性 |
| 最大迭代数 $T_{\max}$ | $10^5$ | 计算资源约束 | 按 $d$ 自适应调整 |

### 3.4 协变量的回归调整公式

对于主响应变量 $K_\epsilon$，采用**协变量调整线性模型**：

$$
\log K_\epsilon = \mu + \tau_\alpha + \tau_\pi + \tau_{\alpha \times \pi} + \gamma_1 \log \kappa_{\text{cond}}^{(0)} + \gamma_2 \log \kappa_{\text{sp}}^{(0)} + \gamma_3 \delta_{\text{ortho}} + \varepsilon
$$

其中 $\gamma_1, \gamma_2, \gamma_3$ 为协变量系数，通过最小二乘估计。此调整可**降低残差方差达 30-50%**，提升检验功效。

---

## 14. 随机变量与分布假设（Random Variables & Distributions）

### 4.1 随机变量总览

本实验包含四类随机源，每类具有明确的概率分布假设：

1. **问题生成随机性**：$X^\star$, $\{A_i\}$, $\{\epsilon_i\}$
2. **算法初始化随机性**：$W_\ell^{(0)}$
3. **算法内在随机性**：随机 SVD、随机采样
4. **实验重复随机性**：随机种子 $s$

### 4.2 随机变量详表

| 符号 | 名称 | 分布假设 | 支撑集 | 参数 | 独立/依赖关系 |
|:---:|:---|:---|:---:|:---:|:---|
| $X^\star$ | 真实矩阵 | 构造性分布（见下方） | $\mathbb{R}^{d_1 \times d_2}$ | $r, \kappa_{\text{cond}}^{\star}$ | 由 $s$ 决定，种子间独立 |
| $A_i$ | 第 $i$ 个测量矩阵 | $A_{i,jk} \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0, 1)$ | $\mathbb{R}^{d \times d}$ | 无 | 给定 $s$ 后固定；$i$ 间独立 |
| $\epsilon_i$ | 测量噪声 | $\epsilon_i \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma_\epsilon^2)$ | $\mathbb{R}$ | $\sigma_\epsilon$ | 与 $A_i$ 独立 |
| $W_\ell^{(0)}$ | 第 $\ell$ 层初始化 | $W_{\ell,ij}^{(0)} \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0, 1/d_\ell)$ | $\mathbb{R}^{d_{\ell+1} \times d_\ell}$ | $d_\ell$ | 给定 $s$ 后固定；$\ell$ 间独立 |
| $s$ | 随机种子 | $s \sim \text{Uniform}\{0, \ldots, 9\}$ | $\{0, \ldots, 9\}$ | 无 | 实验设计变量 |

### 4.3 真实矩阵 $X^\star$ 的构造分布

**低秩构造（$r < d$）**：

$$
X^\star = U \cdot \text{diag}(\sigma_1, \ldots, \sigma_r) \cdot V^\top
$$

其中：
- $U \in \mathbb{R}^{d \times r}$：随机正交矩阵（Haar 测度采样）
- $V \in \mathbb{R}^{d \times r}$：随机正交矩阵，与 $U$ 独立
- 奇异值构造（两种模式）：
  - **均匀谱**：$\sigma_i = 1$（$i = 1, \ldots, r$）
  - **指数衰减谱**：$\sigma_i = \kappa^{-(i-1)/(r-1)}$（$\kappa$ 为条件数参数）

**满秩构造（$r = d$）**：

$$
X^\star = \frac{1}{\sqrt{d}} G, \quad G_{ij} \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0, 1)
$$

### 4.4 联合分布分解

所有随机变量的联合分布可按实验层级分解为：

$$
\begin{aligned}
& p(X^\star, \{A_i\}_{i=1}^m, \{\epsilon_i\}_{i=1}^m, \{W_\ell^{(0)}\}_{\ell=1}^L, s \mid \pi, d, r, L, \iota, \sigma_\epsilon) \\
&= \underbrace{p(s)}_{\text{种子}} \times \underbrace{p(X^\star \mid s, d, r, \pi)}_{\text{真实矩阵}} \times \underbrace{p(\{A_i\} \mid s, d, m)}_{\text{测量矩阵}} \times \underbrace{p(\{\epsilon_i\} \mid s, m, \sigma_\epsilon)}_{\text{噪声}} \times \underbrace{p(\{W_\ell^{(0)}\} \mid s, d, L, \iota)}_{\text{初始化}} \\
&= \frac{1}{10} \times \delta_{X^\star(s)} \times \prod_{i=1}^m \mathcal{N}(\text{vec}(A_i) \mid 0, I_{d^2}) \times \prod_{i=1}^m \mathcal{N}(\epsilon_i \mid 0, \sigma_\epsilon^2) \times \prod_{\ell=1}^L \mathcal{N}(\text{vec}(W_\ell^{(0)}) \mid 0, \frac{1}{d_\ell} I_{d_\ell d_{\ell+1}})
\end{aligned}
$$

**条件独立性结构**：

$$
\begin{aligned}
\{A_i\} &\perp\!\!\perp \{\epsilon_i\} \perp\!\!\perp \{W_\ell^{(0)}\} \mid s, d, m, L, \sigma_\epsilon, \iota \\
X^\star &\perp\!\!\perp \{A_i\} \mid s \quad \text{（在确定性构造下）}
\end{aligned}
$$

### 4.5 算法随机性

| 算法 | 随机操作 | 分布 | 备注 |
|:---:|:---|:---|:---|
| Muon | 随机 SVD（可选） | rSVD 近似 | 默认使用确定性 SVD；大规模时可切换 |
| Muon | 随机正交化 | QR 分解中的符号选择 | 对收敛结果影响可忽略 |
| SGD | mini-batch 采样（若启用） | 无放回均匀采样 | 当前实验使用 full-batch，无此随机性 |

---

## 15. 派生统计量（Derived Statistics）

### 5.1 效率与加速比

| 符号 | 名称 | 数学定义 | 解释 |
|:---:|:---|:---|:---|
| $\rho_K$ | **迭代效率比** | $\rho_K = \frac{K_\epsilon^{\text{SGD}}}{K_\epsilon^{\text{Muon}}}$ | $\rho_K > 1$：Muon 迭代更快；$\rho_K = 1$：等价 |
| $\rho_F$ | **FLOPs 效率比** | $\rho_F = \frac{F_\epsilon^{\text{SGD}}}{F_\epsilon^{\text{Muon}}}$ | 考虑每步计算成本的综合效率 |
| $\rho_T$ | **时间效率比** | $\rho_T = \frac{T_{\text{wall}}^{\text{SGD}}}{T_{\text{wall}}^{\text{Muon}}}$ | 实际 wall-clock 加速 |
| $S_L$ | **可扩展性比** | $S_L = \frac{\rho_F(d_2)}{\rho_F(d_1)}$（$d_2 > d_1$） | 随尺寸增长的效率变化趋势 |

### 5.2 收敛与稳定性指标

| 符号 | 名称 | 数学定义 | 解释 |
|:---:|:---|:---|:---|
| $\rho_{\text{conv}}(k)$ | **逐点收敛比率** | $\rho_{\text{conv}}(k) = \frac{f_{k+1} - f^\star}{f_k - f^\star}$ | 局部收敛速率；$\rho_{\text{conv}} < 1$ 为收敛 |
| $\bar{\rho}_{\text{conv}}$ | **平均收敛比率** | $\bar{\rho}_{\text{conv}} = \exp\left(\frac{1}{K_\epsilon}\sum_{k=0}^{K_\epsilon-1} \log \rho_{\text{conv}}(k)\right)$ | 几何平均收敛速率 |
| $\sigma_{\log}(k)$ | **对数标准差** | $\sigma_{\log}(k) = \sqrt{\frac{1}{n_s - 1}\sum_{s=0}^{n_s-1} (\log f_k^{(s)} - \overline{\log f_k})^2}$ | 第 $k$ 步跨种子稳定性 |
| $\bar{\sigma}_{\log}$ | **平均对数标准差** | $\bar{\sigma}_{\log} = \frac{1}{K_\epsilon}\sum_{k=1}^{K_\epsilon} \sigma_{\log}(k)$ | 整体收敛稳定性 |
| $\text{CV}_K$ | **迭代数变异系数** | $\text{CV}_K = \frac{\sigma[K_\epsilon]}{\mathbb{E}[K_\epsilon]}$ | 收敛时间的相对波动性 |

### 5.3 效应量（Effect Size）度量

| 符号 | 名称 | 数学定义 | 用途 |
|:---:|:---|:---|:---|
| $d_{\text{Cohen}}$ | **Cohen's d** | $d_{\text{Cohen}} = \frac{\bar{K}_\epsilon^{\text{Muon}} - \bar{K}_\epsilon^{\text{SGD}}}{s_{\text{pooled}}}$ | 标准化均值差异；$|d| > 0.8$ 为大效应 |
| $d_{\text{log}}$ | **对数效应量** | $d_{\text{log}} = \log_{10} \rho_K = \log_{10} K_\epsilon^{\text{SGD}} - \log_{10} K_\epsilon^{\text{Muon}}$ | 对数尺度上的差异；更适用于右偏分布 |
| $A_{12}$ | **Vargha-Delaney A** | $A_{\text{VD}} = P(K_\epsilon^{\text{Muon}} < K_\epsilon^{\text{SGD}})$ | 非参数效应量；解释直观 |
| $\Delta AUC$ | **收敛曲线 AUC 差异** | $\Delta AUC = \int_0^{T_{\max}} (f_k^{\text{SGD}} - f_k^{\text{Muon}}) \, dk$ | 整体收敛轨迹差异 |

### 5.4 置信区间与不确定性量化

**逐配置置信区间**（基于 $n_s = 10$ 次重复）：

| 统计量 | 点估计 | 置信区间方法 | 公式 |
|:---:|:---|:---:|:---|
| $\mathbb{E}[K_\epsilon]$ | $\bar{K}_\epsilon = \frac{1}{n_s}\sum_{s=0}^{n_s-1} K_\epsilon^{(s)}$ | Bootstrap BCa | $[\hat{K}_{\alpha/2}^*, \hat{K}_{1-\alpha/2}^*]$ |
| $\mathbb{E}[\log K_\epsilon]$ | $\overline{\log K}_\epsilon$ | t-区间（小样本） | $\overline{\log K}_\epsilon \pm t_{n_s-1, \alpha/2} \cdot \frac{s_{\log K}}{\sqrt{n_s}}$ |
| $\rho_K$ | $\hat{\rho}_K$ | Bootstrap 比率区间 | $\exp\left(\log \hat{\rho}_K \pm z_{\alpha/2} \cdot \text{SE}[\log \hat{\rho}_K]\right)$ |
| $p_{\text{conv}}$ | $\hat{p}_{\text{conv}} = \frac{1}{n_s}\sum_s \mathbb{I}_{\text{conv}}^{(s)}$ | Wilson 分数区间 | 见下方 |

**Wilson 分数区间**（对二元收敛标志）：

$$
\text{CI}_{p_{\text{conv}}} = \frac{\hat{p} + \frac{z^2}{2n_s} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n_s} + \frac{z^2}{4n_s^2}}}{1 + \frac{z^2}{n_s}}
$$

其中 $z = z_{1-\alpha/2}$，$\alpha = 0.05$。

**跨配置聚合统计量**：

| 符号 | 名称 | 定义 | 用途 |
|:---:|:---|:---|:---|
| $\bar{K}_\epsilon(\alpha, \pi, d)$ | 箱内均值 | 同配置下 $n_s$ 次运行的均值 | 主效应估计 |
| $\text{Med}[K_\epsilon]$ | 箱内中位数 | 同配置下 $n_s$ 次运行的中位数 | 对异常值稳健的位置估计 |
| $\text{IQR}[K_\epsilon]$ | 四分位距 | $Q_3 - Q_1$ | 离散度稳健估计 |
| $\hat{q}_{0.95}$ | 95% 分位数 | 经验 95% 分位点 | 最坏情况评估 |

---

## 16. 变量间的函数依赖与因果结构

### 6.1 因果 DAG（有向无环图）

以下为实验变量的因果结构文字描述：

```
实验设计层 ──► 问题生成层 ──► 算法执行层 ──► 观测响应层

实验设计层：
  (α, π, d, r, L, ι, η, λ, σ_ϵ, s) ──► 问题生成层

问题生成层：
  s ──► X* ──► κ_cond^* ──► 问题难度
  s ──► {A_i} ──► μ_SR, M_SR ──► 问题几何
  X*, {A_i} ──► y ──► 测量向量
  
  问题难度, 问题几何 ──► 理论收敛速率 ──► 算法执行层

算法执行层：
  (α, η, λ) ──► 更新规则
  s ──► W^(0) ──► κ_sp^(0), κ_cond^(0) ──► 初始条件
  更新规则 + 初始条件 + 问题实例 ──► {X^(k)}_(k=0)^T ──► 收敛过程

观测响应层：
  {X^(k)} ──► K_ϵ, F_ϵ, I_conv, f_final
  {X^(k)} ──► {σ_i(X_k)} ──► Δ_sp, R(k)
  收敛过程 ──► T_wall, M_peak
  跨种子聚合 ──► σ_log(k), σ̄_log
```

### 6.2 主效应模型

对于主响应变量 $Y \in \{\log K_\epsilon, \log F_\epsilon, \mathbb{I}_{\text{conv}}\}$，建立**线性混合效应模型**：

$$
\begin{aligned}
Y_{i,j,k,\ell} &= \mu + \underbrace{\tau_{\alpha_i}}_{\text{算法主效应}} + \underbrace{\tau_{\pi_j}}_{\text{问题主效应}} + \underbrace{\tau_{d_k}}_{\text{尺寸主效应}} + \underbrace{\tau_{r_\ell}}_{\text{秩主效应}} \\
&\quad + \underbrace{\tau_{\alpha_i \times \pi_j}}_{\text{算法-问题交互}} + \underbrace{\tau_{\alpha_i \times d_k}}_{\text{算法-尺寸交互}} + \underbrace{\tau_{\pi_j \times d_k}}_{\text{问题-尺寸交互}} \\
&\quad + \underbrace{\tau_{\alpha_i \times \pi_j \times d_k}}_{\text{三阶交互}} + \underbrace{\gamma^\top \mathbf{c}}_{\text{协变量调整}} + \underbrace{\nu_{s}}_{\text{种子随机效应}} + \underbrace{\varepsilon_{i,j,k,\ell,s}}_{\text{残差}}
\end{aligned}
$$

其中：
- $\mu$：全局截距
- $\tau$ 项：固定效应（主效应与交互效应）
- $\mathbf{c}$：协变量向量（$\kappa_{\text{sp}}^{(0)}, \kappa_{\text{cond}}^{(0)}, \delta_{\text{ortho}}$）
- $\nu_s \sim \mathcal{N}(0, \sigma_s^2)$：随机种子效应
- $\varepsilon \sim \mathcal{N}(0, \sigma_\varepsilon^2)$：独立残差

### 6.3 必须检验的交互效应

| 交互项 | 检验优先级 | 科学假设 | 检验方法 |
|:---|:---:|:---|:---|
| $\alpha \times \pi$ | **高** | Muon 的优势在 MF 上比 MS 更显著 | 双因素 ANOVA；若显著则分层报告 |
| $\alpha \times d$ | **高** | Muon 的谱归一化在大尺寸上优势放大 | 按 $d$ 分层比较；检验斜率差异 |
| $\alpha \times r$ | **高** | Muon 对低秩结构的利用更有效 | 低秩 vs 满秩子样本分别检验 |
| $\alpha \times L$ | 中（MF 子实验） | Muon 的深度扩展性优于 SGD | MF 子实验中分层 ANOVA |
| $\alpha \times \iota$ | 中（$L=2$ 子实验） | Muon 对初始化敏感性不同 | $L=2$ 子实验中三因素分析 |
| $\alpha \times \eta$ | 中 | 两种算法对学习率的敏感性不同 | 学习率作为调节变量分析 |
| $\pi \times d$ | 中 | 问题类型的尺寸扩展性不同 | 双因素分析 |
| $\alpha \times \pi \times d$ | **探索性** | 算法-问题-尺寸三阶交互 | 若二阶显著则检验；需增大样本量 |

### 6.4 条件独立性声明

以下独立性关系在固定父节点条件下成立：

1. **给定问题实例，算法响应独立于种子**：
   $$Y \perp\!\!\perp s \mid X^\star, \{A_i\}, \alpha, \eta, \lambda$$

2. **给定初始条件，收敛行为独立于初始化细节**：
   $$K_\epsilon \perp\!\!\perp \iota \mid \kappa_{\text{sp}}^{(0)}, \kappa_{\text{cond}}^{(0)}, \alpha, \eta$$

3. **不同问题实例间响应独立**：
   $$Y^{(s_1)} \perp\!\!\perp Y^{(s_2)} \mid \text{所有因子}, \quad s_1 \neq s_2$$

### 6.5 函数依赖关系总结

| 响应变量 | 直接父节点 | 函数形式 |
|:---|:---|:---|
| $K_\epsilon$ | $\alpha, \pi, d, r, L, \eta, \kappa_{\text{cond}}^\star, \kappa_{\text{sp}}^{(0)}, \mu_{\text{SR}}$ | $K_\epsilon = g(\alpha, \pi, d, r, L) \cdot h(\eta) \cdot q(\text{problem hardness})$ |
| $F_\epsilon$ | $K_\epsilon, \alpha, \pi, d, L$ | $F_\epsilon = K_\epsilon \times \mathcal{C}_{\text{per-iter}}(\alpha, \pi, d, L)$ |
| $\mathbb{I}_{\text{conv}}$ | $\alpha, \pi, d, r, L, \eta, T_{\max}$ | $\mathbb{I}_{\text{conv}} = \mathbf{1}_{\{K_\epsilon \leq T_{\max}\}}$ |
| $\bar{\sigma}_{\log}$ | $\{f_k^{(s)}\}_{k,s}$ | $\bar{\sigma}_{\log} = \frac{1}{K}\sum_k \text{SD}_s[\log f_k^{(s)}]$ |
| $T_{\text{wall}}$ | $F_\epsilon, \text{HW}$ | $T_{\text{wall}} = F_\epsilon / \text{FLOP-rate}(\text{HW})$ |

---

## 17. 实验设计的统计框架建议

### 7.1 样本量与功效分析

**当前设计**：每个配置 $n_s = 10$ 次重复。

**功效分析**（针对主效应 $\alpha$ 在响应 $\log K_\epsilon$ 上）：

假设中等效应量 $d_{\text{Cohen}} = 0.8$（对数尺度差异 $d_{\log} \approx 0.5$），显著性水平 $\alpha = 0.05$：

$$
\beta = 1 - \Phi\left(z_{\alpha/2} + \frac{|d_{\text{Cohen}}| \sqrt{n_s/2}}{\sqrt{1 + \rho_{\text{intra}}(n_s - 1)}}\right)
$$

其中 $\rho_{\text{intra}}$ 为组内相关系数（同一配置不同种子间的相关性，通常 $< 0.1$）。

**功效计算结果**：

| 效应量 $d_{\text{Cohen}}$ | $n_s = 10$ 功效 | $n_s = 20$ 功效 | $n_s = 30$ 功效 |
|:---:|:---:|:---:|:---:|
| 0.5（中） | 0.72 | 0.92 | 0.98 |
| 0.8（大） | 0.95 | >0.99 | >0.99 |
| 1.2（超大） | >0.99 | >0.99 | >0.99 |

**建议**：
- 核心配置（$d \leq 200$）：保持 $n_s = 10$
- 大尺寸配置（$d = 500$）：降低至 $n_s = 5$ 以控制计算成本
- 关键交互检验：提升至 $n_s = 20$ 或采用自适应样本量

### 7.2 析因设计框架

**完整析因设计**（不考虑结构零）：

$$
2 \times 2 \times 4 \times 2 \times 3 \times 3 \times 3 = 864 \text{ 组合（仅因子水平）}
$$

**稀疏析因设计建议**（Fractional Factorial）：

采用 $2^{6-1}$ 或 $2^{7-2}$ 设计筛选主效应与二阶交互，将运行数从 64/128 降至 32/64。

**推荐的分层析因策略**：

```
第一层（筛选实验）：
  - 因子：α × π × d × r × η
  - 水平：2 × 2 × 3 × 2 × 3 = 72 组合
  - 每组合 n_s = 5
  - 目标：识别显著主效应与二阶交互

第二层（精确实验）：
  - 基于第一层结果，聚焦显著因子组合
  - 补充 L、ι、λ、σ_ϵ 的扫描
  - 每组合 n_s = 10（核心配置）或 n_s = 20（边界配置）
```

### 7.3 拉丁方与区组设计

**拉丁方设计**（用于控制两个干扰因子）：

若需同时控制**矩阵尺寸 $d$** 与**问题类型 $\pi$** 的干扰，可构建拉丁方：

| | $\pi = \text{MS}$ | $\pi = \text{MF}, L=2$ | $\pi = \text{MF}, L=3$ | $\pi = \text{MF}, L=4$ |
|:---:|:---:|:---:|:---:|:---:|
| $d = 50$ | Muon | SGD | Muon | SGD |
| $d = 100$ | SGD | Muon | SGD | Muon |
| $d = 200$ | Muon | SGD | Muon | SGD |
| $d = 500$ | SGD | Muon | SGD | Muon |

此设计确保每个算法在每个尺寸-问题组合中恰好出现一次，平衡区组效应。

**区组设计建议**：
- **区组因子**：随机种子 $s$（10 个区组）
- **处理因子**：$(\alpha, \eta)$ 组合
- **区组内设计**：在每个种子内随机化算法-学习率顺序，消除时间趋势

### 7.4 分层抽样策略

**自适应分层抽样**：

对于响应变量的分布通常右偏（$K_\epsilon$ 在困难配置上非常大），建议采用：

1. **按问题难度分层**：
   - 层 1（易）：$\kappa_{\text{cond}}^\star < 10$, $d \leq 100$
   - 层 2（中）：$\kappa_{\text{cond}}^\star \in [10, 100]$, $d \in [100, 200]$
   - 层 3（难）：$\kappa_{\text{cond}}^\star > 100$ 或 $d \geq 500$

2. **Neyman 最优分配**：
   $$n_h \propto N_h \cdot S_h$$
   其中 $N_h$ 为层 $h$ 的配置数，$S_h$ 为层内响应标准差。

3. **过采样策略**：
   对边界配置（最大尺寸、最大深度、最难初始化）进行 2× 过采样，提升外推可靠性。

### 7.5 多重比较校正

实验涉及大量假设检验（主效应、交互效应、多响应变量），需控制族错误率（FWER）或错误发现率（FDR）：

| 校正方法 | 适用场景 | 公式/说明 |
|:---|:---|:---|
| **Bonferroni** | 少量检验（$m < 20$） | $\alpha^* = \alpha / m$；保守 |
| **Holm-Bonferroni** | 有序 p 值序列 | 逐步校正；功效优于 Bonferroni |
| **Benjamini-Hochberg** | 大量探索性检验 | 控制 FDR $\leq \alpha$；推荐用于探索性分析 |
| **Tukey HSD** | 所有成对比较 | 基于学生化极差分布；适用于事后比较 |
| **Scheffé** | 所有对比 | 保守但灵活；适用于复杂对比 |

**推荐策略**：
- **确认性分析**（主效应 $\alpha$）：不校正，$\alpha = 0.05$
- **次要假设**（交互效应）：Holm-Bonferroni，族大小 $m = 6$（6 个交互项）
- **探索性分析**（所有配置成对比较）：Benjamini-Hochberg，FDR = 0.10

### 7.6 缺失数据处理方案

| 缺失类型 | 可能原因 | 处理策略 |
|:---|:---|:---|
| **未收敛** | $K_\epsilon = T_{\max}$（截断） | 右删失（Right-censoring）；采用 Cox 比例风险模型或 Tobit 回归 |
| **数值溢出** | 梯度爆炸、NaN | 标记为失败运行；若失败率 > 20% 报告为可靠性问题 |
| **超时** | Wall-clock 超过预算 | 与未收敛合并处理；或作为竞争风险（Competing Risk） |
| **内存不足** | $M_{\text{peak}}$ 超过限制 | 标记为资源失败；在资源受限场景分析中排除 |

**删失数据的统计处理**：

对于未收敛运行（$K_\epsilon$ 右删失），定义删失指示变量 $\delta = \mathbb{I}_{\text{conv}}$，采用**生存分析框架**：

$$
h(k) = \lim_{\Delta \to 0} \frac{P(k \leq K_\epsilon < k + \Delta \mid K_\epsilon \geq k)}{\Delta}
$$

其中 $h(k)$ 为**收敛风险函数**。算法比较可转化为检验 $h_{\text{Muon}}(k) \stackrel{?}{=} h_{\text{SGD}}(k)$（对数秩检验）。

### 7.7 推荐的实验执行流程

```
Phase 0: 预实验（Pilot Study）
  ├─ 小规模配置（d=50, n_s=3）
  ├─ 学习率粗略扫描（η ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1}）
  ├─ 估计效应量、方差分量、删失比例
  └─ 输出：正式实验的样本量、学习率候选集

Phase 1: 核心实验（Core Experiment）
  ├─ 全因子组合（考虑结构约束）
  ├─ 最优学习率选择（基于 Phase 0）
  ├─ n_s = 10 次重复
  ├─ 记录全部原始响应与协变量
  └─ 输出：完整数据集

Phase 2: 边界扩展（Boundary Extension）
  ├─ 补充 d=1000, 2000（若资源允许）
  ├─ 补充噪声实验（σ_ϵ > 0）
  ├─ 补充权重衰减实验（λ > 0）
  ├─ 补充矩形矩阵实验
  └─ 输出：外推验证数据

Phase 3: 稳健性分析（Robustness Analysis）
  ├─ 最差初始化扫描
  ├─ 次优学习率敏感性
  ├─ 不同随机矩阵分布（如 Rademacher 替代 Gaussian）
  └─ 输出：算法可靠性边界

Phase 4: 统计推断与报告
  ├─ 多重比较校正
  ├─ 效应量计算与置信区间
  ├─ 交互效应可视化
  ├─ 因果推断敏感性分析
  └─ 输出：最终统计报告
```

---

---

# 第四部分：缺失实验识别与补充

> 本部分对现有 Muon vs SGD 实验设计进行系统性盲区审查，识别8大维度下的40余项缺失，并设计15个可立即执行的补充实验（编号 E6–E20）。每个补充实验均配备完整的数学形式、统计假设、检验统计量与预期解释。本部分建立在第二部分的形式化框架和第三部分的统计变量体系之上，将实验覆盖范围从基础配置扩展到完整的算法-问题-环境参数空间。

> **文档定位**：Stage 5 —— 对现有 Muon vs SGD 实验设计进行系统性盲区审查，识别 8 大维度下的 40 余项缺失，并设计 15 个可立即执行的补充实验（编号 E6–E20）。每个补充实验均配备完整的数学形式、统计假设、检验统计量与预期解释。

---

## 18. 缺失维度系统审查

现有实验框架覆盖了矩阵感知（MS）与矩阵分解（MF）两类核心问题，在算法层面建立了 Muon（谱归一化方向）与 SGD（原始梯度方向）的对比基线，并通过维度扫描（$d=50,100,200,500$）、秩结构（低秩 vs 满秩）、学习率网格搜索与 10 次随机重复，形成了初步的统计推断基础。然而，从严格的实验设计（Design of Experiments, DOE）视角审视，现有框架在以下 8 个维度存在显著缺失。

---

### 维度 A：算法变体与消融实验

| 子维度 | 现有覆盖 | 缺失内容 | 缺失严重性 |
|--------|----------|----------|------------|
| A1: Muon 内部变体 | 仅原始 SVD 谱归一化 $D^{(k)} = UV^\top$ | 截断 SVD（$r_{\text{trunc}} < \text{rank}(G)$）、随机 SVD（Halko 算法）、部分谱归一化（仅缩放前 $k$ 个奇异值）、不完全是 $UV^\top$ 而是 $U\Sigma^{-\alpha}V^\top$（$\alpha \in [0,1]$） | 高 |
| A2: Momentum 机制 | 无 | Polyak Heavy-Ball、Nesterov 加速嵌入 Muon；动量系数 $\beta \in \{0.0, 0.5, 0.9, 0.99\}$ | 高 |
| A3: Weight Decay | $\lambda = 0$ | $\lambda \in \{10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}\}$ 对 Muon 与 SGD 的差异化影响 | 中 |
| A4: 学习率调度 | 固定 $\eta$ | 余弦退火、阶梯衰减、预热（warmup）对谱归一化方向的动态影响 | 中 |
| A5: 自适应 Muon | 无 | 将 RMSprop/Adam 的自适应增益与谱方向结合（如 $D^{(k)}_{\text{Adam-Muon}} = \frac{UV^\top}{\sqrt{v^{(k)}} + \epsilon}$） | 中 |
| A6: 基线算法 | 仅 SGD | Adam、AdaGrad、RMSprop、Momentum SGD、L-BFGS、自然梯度（NGD）近似 | 高 |

**关键科学问题**：Muon 的核心优势是否仅来源于 $UV^\top$ 的方向归一化？当去掉精确 SVD、引入动量、施加权重衰减时，优势是否依然保持？

---

### 维度 B：问题变体与问题空间扩展

| 子维度 | 现有覆盖 | 缺失内容 | 缺失严重性 |
|--------|----------|----------|------------|
| B1: 秩比例 | $r = d/10$ 与满秩 | $r/d \in \{0.01, 0.05, 0.1, 0.2, 0.5, 1.0\}$ 的连续扫描 | 高 |
| B2: 矩阵形状 | 仅方阵（$d \times d$） | 矩形矩阵 $m \times n$（如 $m=2n$、$m=n/2$），感知算子 $A_i \in \mathbb{R}^{m \times n}$ | 高 |
| B3: 噪声水平 | $\sigma_\epsilon = 0$ | $\sigma_\epsilon \in \{0, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\}$ 对收敛率与稳定性的影响 | 高 |
| B4: 采样率 | $m = 3d^2$（过采样） | $m/d^2 \in \{0.5, 1.0, 1.5, 2.0, 3.0, 5.0\}$，覆盖欠采样（$m < d^2$）与过采样 | 高 |
| B5: 病态控制 | 随机矩阵的自然条件数 | 显式构造 $\kappa_{\text{target}} \in \{10, 10^2, 10^3, 10^4, 10^6\}$（通过 SVD 注水法或参数化矩阵族） | 高 |
| B6: 非对称分解 | $W_L \cdots W_1$ 中 $W_i$ 均为方阵 | 矩形因子分解：$W_1 \in \mathbb{R}^{d_1 \times d_0}, W_2 \in \mathbb{R}^{d_2 \times d_1}, \dots$，尺寸不匹配的深度网络 | 中 |
| B7: 矩阵分布 | $A_{ij} \sim \mathcal{N}(0,1)$ | Rademacher $\pm 1$、亚高斯（sub-Gaussian）、球形高斯、快速 JL 变换（Hadamard + 子采样） | 中 |

**关键科学问题**：Muon 的谱结构优势是否依赖于问题的良态（well-conditioned）与高斯随机性？当问题变得病态、欠采样、有噪声、非方阵时，Muon 是否仍然优于 SGD？

---

### 维度 C：初始化与敏感性分析

| 子维度 | 现有覆盖 | 缺失内容 | 缺失严重性 |
|--------|----------|----------|------------|
| C1: 初始化尺度 | $\mathcal{N}(0, 1/d)$ | $\sigma_{\text{init}} \in \{10^{-3}, 10^{-2}, 10^{-1}, 1, 10\}/d$ 的系统扫描 | 高 |
| C2: 初始化分布 | 高斯 | 正交初始化（$X^{(0)} = QR$）、谱初始化（$X^{(0)} = U\text{diag}(\lambda) V^\top$）、零初始化、均匀分布 | 中 |
| C3: Muon 特异性 | 无 | Muon 的 $D^{(k)}$ 在零初始化或极端尺度初始化下的行为（$G^{(k)}=0$ 时 SVD 退化） | 高 |
| C4: 深度 MF 初始化 | L=2 有三种特殊初始化 | L=3,4 的初始化敏感性（是否存在类似 "balanced initialization" 的效应？） | 中 |

**关键科学问题**：Muon 是否对初始化尺度更敏感（因为 SVD 对梯度幅值敏感）或更不敏感（因为谱归一化去除了尺度）？

---

### 维度 D：超参数空间与校准

| 子维度 | 现有覆盖 | 缺失内容 | 缺失严重性 |
|--------|----------|----------|------------|
| D1: 学习率粒度 | $\eta \in \{10^{-3}, 10^{-2}, 10^{-1}\}$ | $\eta \in \{10^{-4}, 3\times 10^{-4}, 10^{-3}, 3\times 10^{-3}, \dots, 10^{-1}\}$ 对数均匀网格 | 中 |
| D2: 权重衰减 | $\lambda = 0$ | $\lambda \times \eta$ 的交互效应（二维网格） | 中 |
| D3: 调度器 | 无 | warmup 长度、余弦周期与 Muon 谱方向的耦合 | 低 |
| D4: 早停与容差 | 固定 $\epsilon$ | 不同精度要求 $\epsilon \in \{10^{-6}, 10^{-8}, 10^{-10}\}$ 下的效率对比 | 中 |

---

### 维度 E：计算效率与可扩展性

| 子维度 | 现有覆盖 | 缺失内容 | 缺失严重性 |
|--------|----------|----------|------------|
| E1: Wall-clock Time | 理论 FLOPs | 实际 CPU/GPU 计时（含 SVD 实际开销、缓存效应、BLAS 实现差异） | 高 |
| E2: 内存占用 | 无 | SGD $O(d^2)$ vs Muon SVD 额外 $O(d^3)$ 的峰值内存分析 | 中 |
| E3: SVD 实现变体 | 精确 SVD（Golub-Reinsch） | 随机 SVD（$O(d^2 r_{\text{approx}})$）、增量 SVD、截断 SVD 的精度-速度权衡 | 高 |
| E4: 规模外推 | $d \leq 500$ | $d=1000, 2000, 5000$ 的可行性（Muon SVD $O(d^3)$ 的硬墙） | 高 |
| E5: 并行化 | 无 | 随机 SVD 的并行潜力 vs SGD 的 trivial 并行化 | 中 |

**关键科学问题**：Muon 的理论 FLOPs 优势（每迭代更低）是否被 SVD 的实际常数因子所抵消？在什么规模 $d^*$ 下，Muons 的实际时间效率被 SGD 反超？

---

### 维度 F：统计可靠性与推断严格性

| 子维度 | 现有覆盖 | 缺失内容 | 缺失严重性 |
|--------|----------|----------|------------|
| F1: 功效分析 | 无 | 给定效应量 $\delta$、方差 $\sigma^2$、$\alpha=0.05$ 下，10 次重复是否足够？计算所需 $n_{\min}$ | 高 |
| F2: 置信区间 | 仅点估计 | $K_\epsilon$、$F_\epsilon$ 的 95% 置信区间（t-分布或 bootstrap） | 高 |
| F3: 多重检验校正 | 无 | 20+ 次假设检验下族错误率（FWER）控制（Bonferroni、Holm、FDR） | 高 |
| F4: 效应量报告 | 无 | Cohen's $d$、对数比率 $\log(K_{\text{Muon}}/K_{\text{SGD}})$ 的分布 | 中 |
| F5: 方差分析 | 无 | 多因子 ANOVA 分解 $K_\epsilon$ 的方差来源（算法 $\times$ 问题 $\times$ 维度 $\times$ 秩） | 中 |
| F6: 非参数检验 | 假设正态 | 若 $K_\epsilon$ 分布偏态，使用 Mann-Whitney U、Kruskal-Wallis、置换检验 | 中 |

**关键科学问题**：现有 10 次重复是否具有足够的统计功效来检测 Muon 与 SGD 之间真实的收敛率差异？若差异存在但未被检测到（Type II 错误），现有结论是否过于保守？

---

### 维度 G：动态行为与 Landscape 分析

| 子维度 | 现有覆盖 | 缺失内容 | 缺失严重性 |
|--------|----------|----------|------------|
| G1: Hessian 谱动态 | 无 | 每 $T$ 步计算 Hessian $\nabla^2 f(X^{(k)})$ 的特征值分布，观察最大/最小特征值比值随迭代的演化 | 高 |
| G2: 梯度-更新方向夹角 | 无 | $\cos\theta_k = \frac{\langle G^{(k)}, D^{(k)}\rangle}{\|G^{(k)}\|_F \|D^{(k)}\|_F}$ 的时序演化（Muon $D^{(k)}=UV^\top$，SGD $D^{(k)}=G^{(k)}$） | 高 |
| G3: 参数轨迹 PCA | 无 | 将 $\{X^{(k)}\}_{k=0}^K$ 展平为向量，进行 PCA，分析收敛轨迹的低维结构 | 中 |
| G4: 谱间隙动态 | 静态 $\bar{\sigma}_{\log}$ | $\sigma_i(G^{(k)})$ 的前 $r$ 个奇异值的演化，谱间隙 $\sigma_r - \sigma_{r+1}$ 是否影响 Muon 方向质量 | 高 |
| G5: 临界点邻域动态 | 无 | 接近收敛时（$\|\nabla f\| < 10^{-3}$）的局部收敛率测量（线性 vs 次线性） | 中 |

**关键科学问题**：Muon 的谱归一化方向是否始终与梯度方向保持高相关性？当 Hessian 谱结构随迭代演化时，Muon 的方向质量是否会退化？

---

### 维度 H：泛化性与分布偏移

| 子维度 | 现有覆盖 | 缺失内容 | 缺失严重性 |
|--------|----------|----------|------------|
| H1: 测量矩阵分布 | $A_{ij} \sim \mathcal{N}(0,1)$ | Rademacher、伯努利、球形高斯（$\text{Unif}(\mathbb{S}^{d^2-1})$）、结构化矩阵（DFT、Hadamard） | 中 |
| H2: 噪声分布 | 高斯噪声 | 拉普拉斯噪声（重尾）、均匀噪声、乘性噪声 | 低 |
| H3: 目标矩阵结构 | 随机低秩 | 具有特定特征值衰减（多项式衰减 $\lambda_i = i^{-\alpha}$、指数衰减）的矩阵 | 中 |
| H4: 问题混合 | 独立实验 | 跨问题迁移：在 MS 上训练的超参数是否在 MF 上同样最优？ | 低 |

---

### 缺失维度总结

将上述 8 个维度的缺失按 **科学价值 $\times$ 实施成本** 的乘积进行风险排序（高乘积 = 高优先补充）：

| 排名 | 维度 | 关键缺失 | 风险说明 |
|------|------|----------|----------|
| 1 | B（问题变体） | 欠采样、噪声、病态 | 现有结论可能仅在 "简单" 问题上成立 |
| 2 | A（算法基线） | Adam、Momentum SGD、L-BFGS | 无法声称 Muon 优于 "标准" 优化器 |
| 3 | E（计算效率） | Wall-clock Time、$d=1000$+ | 理论优势可能被实际开销抵消 |
| 4 | F（统计可靠性） | 功效分析、置信区间、多重检验 | 10 次重复可能不足；结论的统计可信度未知 |
| 5 | G（动态行为） | Hessian 动态、梯度-方向夹角 | 缺乏对 "为什么 Muon 更好" 的机制理解 |
| 6 | C（初始化） | 极端尺度、零初始化 | 实际使用中初始化差异大 |
| 7 | D（超参数） | $\lambda$、调度器 | 超参数交互可能改变结论 |
| 8 | H（泛化性） | 非高斯矩阵 | 稳健性边界未知 |

---

## 19. 补充实验设计

以下 15 个补充实验（编号 E6–E20）系统覆盖上述 8 个维度的关键缺失。每个实验均包含：
- **实验名称与编号**
- **科学问题（Research Question）**
- **数学形式**：数据生成、优化目标、算法变体、参数设置
- **统计模型**：因子设计、响应变量、零假设 $H_0$ 与备择假设 $H_1$、检验统计量
- **与现有实验的关系**
- **预期结果与解释**

---

### E6：噪声敏感性实验（问题变体 B3）

**科学问题**：当感知/分解问题存在加性观测噪声时，Muon 与 SGD 的收敛率、稳定性与最终精度如何退化？Muon 的谱归一化是否对噪声具有内在的鲁棒性或脆弱性？

**数学形式**：

数据生成过程（矩阵感知）：
$$
X^\star = U^\star \Sigma^\star (V^\star)^\top, \quad U^\star, V^\star \sim \text{Haar}, \quad \Sigma^\star = \text{diag}(\lambda_1, \dots, \lambda_r)
$$
$$
y_i = \langle A_i, X^\star \rangle + \epsilon_i, \quad A_{ij} \overset{iid}{\sim} \mathcal{N}(0,1), \quad \epsilon_i \overset{iid}{\sim} \mathcal{N}(0, \sigma_\epsilon^2)
$$
其中噪声水平 $\sigma_\epsilon \in \{0, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\}$（固定信号范数 $\|X^\star\|_F = 1$）。

矩阵分解（噪声注入到目标矩阵）：
$$
X^\star_{\text{noisy}} = X^\star + \sigma_\epsilon E, \quad E_{ij} \overset{iid}{\sim} \mathcal{N}(0,1)
$$
优化目标：
$$
f_{\text{MS}}^{(\sigma)}(X) = \frac{1}{2m}\sum_{i=1}^m (\langle A_i, X \rangle - y_i)^2, \quad f_{\text{MF}}^{(\sigma)} = \frac{1}{2}\|W_L \cdots W_1 - X^\star_{\text{noisy}}\|_F^2
$$

参数设置：
- $d \in \{50, 100, 200\}$（聚焦中等规模，避免噪声与维度耦合）
- $r = d/10$（低秩），$m = 3d^2$
- Muon / SGD，$\lambda = 0$，最优 $\eta$ 独立搜索
- 最大迭代 $K_{\max} = 10^5$，容差 $\epsilon = 10^{-6}$
- 每种配置 $n = 20$ 次重复（噪声增加方差，需更多样本）

**统计模型**：

因子设计（全因子）：
$$
\text{Algorithm} \times \text{Problem} \times d \times \sigma_\epsilon \times \text{Seed}
$$
响应变量：
- $K_\epsilon^{(j)}$：达到收敛的迭代次数（右删失：若未收敛则 $K_\epsilon = K_{\max}$）
- $F_\epsilon^{(j)} = K_\epsilon^{(j)} \times \text{FLOPs-per-iter}$
- $f_{\text{final}}^{(j)} = f(X^{(K_{\max})})$（最终 loss）

零假设与备择假设（对每个 $(d, \sigma_\epsilon)$ 组合）：
$$
H_0^{(d,\sigma)}: \mathbb{E}[\log K_\epsilon^{\text{Muon}}] = \mathbb{E}[\log K_\epsilon^{\text{SGD}}] \quad \text{（噪声下无差异）}
$$
$$
H_1^{(d,\sigma)}: \mathbb{E}[\log K_\epsilon^{\text{Muon}}] < \mathbb{E}[\log K_\epsilon^{\text{SGD}}] \quad \text{（Muon 更快）}
$$
检验统计量：配对对数 t-统计量
$$
T^{(d,\sigma)} = \frac{\bar{D} - 0}{S_D / \sqrt{n}}, \quad D_j = \log K_{\epsilon,j}^{\text{Muon}} - \log K_{\epsilon,j}^{\text{SGD}}
$$
其中 $\bar{D}$ 与 $S_D$ 为差值样本的均值与标准差。若 $T < -t_{0.95, n-1}$，拒绝 $H_0$。

**与现有实验的关系**：直接扩展 E1（矩阵感知）与 E2（矩阵分解），将噪声从 $\sigma_\epsilon=0$ 推广到 $>0$。

**预期结果与解释**：
- **预期 1**：当 $\sigma_\epsilon$ 增大时，两者的 $K_\epsilon$ 均增加，但 Muon 的谱归一化可能放大噪声的高频分量（如果噪声使梯度谱平坦化），导致相对优势缩小。
- **预期 2**：在 $\sigma_\epsilon = 10^{-1}$（高噪声）时，$K_\epsilon$ 差异可能不再显著，暗示 Muon 的优势局限于信噪比（SNR）高于某阈值的问题。
- **解释框架**：建立 "$\sigma_\epsilon$–优势边界" 假说——存在临界 $\sigma_\epsilon^*(d)$，当 $\sigma_\epsilon < \sigma_\epsilon^*$ 时 Muon 显著优于 SGD，反之无显著差异。

---

### E7：秩比例扫描实验（问题变体 B1）

**科学问题**：Muon 相对于 SGD 的优势是否依赖于特定的秩比例 $r/d$？是否存在一个 "$r/d$ 甜点区" 使得谱归一化的信息压缩最为有效？

**数学形式**：

矩阵感知数据生成：
$$
r \in \{0.01d, 0.05d, 0.1d, 0.2d, 0.5d, 1.0d\} \quad \Rightarrow \quad r/d \in \{0.01, 0.05, 0.1, 0.2, 0.5, 1.0\}
$$
$X^\star$ 的构造：
$$
X^\star = \sum_{i=1}^r \lambda_i u_i v_i^\top, \quad \lambda_i = \lambda_1 \cdot \rho^{i-1}, \quad \rho \in \{0.5, 0.9, 0.99\}
$$
（几何衰减，控制谱集中度）

参数设置：
- $d \in \{100, 200, 500\}$
- 三种谱衰减率 $\rho$
- $m = 3d^2$，$A_{ij} \sim \mathcal{N}(0,1)$
- Muon / SGD，$\lambda=0$，$n=15$ 次重复
- 容差 $\epsilon = 10^{-8}$（更严格，以检测秩对收敛精度的影响）

**统计模型**：

因子设计：
$$
\text{Algorithm} \times r/d \times \rho \times d \times \text{Seed}
$$
响应变量：$K_\epsilon$，$\bar{\sigma}_{\log}$，$\kappa_{\text{sp}}$，$\text{rank}_\epsilon(X^{(\infty)})$。

零假设与备择假设：
$$
H_0: \text{算法} \times (r/d) \text{ 交互效应不显著于 } K_\epsilon
$$
$$
H_1: \exists \; (r/d)^* \text{ 使得 } \Delta K_\epsilon^{\text{Muon-SGD}} \text{ 在该点最大}
$$
检验统计量：双因子 ANOVA 交互 F-统计量
$$
F_{\text{int}} = \frac{\text{MS}_{\text{Alg} \times r/d}}{\text{MS}_{\text{Error}}}, \quad \text{其中 } \text{MS} \text{ 为均方}
$$
若 $F_{\text{int}} > F_{\alpha, df_1, df_2}$，拒绝 $H_0$（存在显著交互效应）。

附加分析：对每个算法单独拟合响应面
$$
\log K_\epsilon^{(\text{Alg})}(r/d, \rho) = \beta_0 + \beta_1(r/d) + \beta_2\rho + \beta_3(r/d)^2 + \beta_4(r/d)\rho + \varepsilon
$$
比较两算法的响应面差异。

**与现有实验的关系**：扩展现有实验中的低秩($r=d/10$)与满秩二分，扩展为连续秩比例扫描。

**预期结果与解释**：
- **预期 1**：在极低秩（$r/d = 0.01$）时，Muons 优势最大，因为梯度谱高度集中，$UV^\top$ 几乎捕捉了全部有效方向。
- **预期 2**：在满秩（$r/d = 1.0$）时，优势消失或逆转，因为谱归一化失去了信息压缩的基础。
- **预期 3**：谱衰减率 $\rho$ 调节了优势的幅度——快速衰减（小 $\rho$）增强 Muon 优势，缓慢衰减（大 $\rho$）削弱优势。

---

### E8：过采样/欠采样实验（问题变体 B4）

**科学问题**：在矩阵感知中，测量数 $m$ 相对于自由度 $d^2$ 的比例如何影响 Muon 与 SGD 的收敛行为？在欠采样（$m < d^2$）和临界采样（$m \approx d^2$）下，Muon 的谱信息是否仍然有助于收敛加速？

**数学形式**：

采样率定义：
$$
\gamma = m / d^2 \in \{0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0\}
$$
感知算子：
$$
\mathcal{A}: \mathbb{R}^{d \times d} \to \mathbb{R}^m, \quad (\mathcal{A}(X))_i = \langle A_i, X \rangle, \quad A_{ij} \overset{iid}{\sim} \mathcal{N}(0, 1/d^2)
$$
（注意归一化：$\mathbb{E}[\|\mathcal{A}(X)\|^2] = \|X\|_F^2$）

优化目标：
$$
f_{\text{MS}}^{(\gamma)}(X) = \frac{1}{2m}\|\mathcal{A}(X) - y\|_2^2
$$

参数设置：
- $d \in \{50, 100, 200\}$
- $r = d/10$
- $\gamma \in \{0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0\}$
- Muon / SGD，$\lambda=0$，$n=15$ 次重复
- 容差 $\epsilon = 10^{-6}$，最大迭代 $K_{\max} = 2 \times 10^5$

**统计模型**：

因子设计：
$$
\text{Algorithm} \times \gamma \times d \times \text{Seed}
$$
响应变量：
- $K_\epsilon$（收敛迭代）
- $\hat{\kappa}_{\text{cond}}(\mathcal{A}^*\mathcal{A})$：感知算子条件数估计（Hessian 的近似）
- 收敛标志 $\delta_{\text{conv}} \in \{0, 1\}$（二值响应，欠采样时可能不收敛）

零假设与备择假设：
$$
H_0^{(\gamma)}: P(\delta_{\text{conv}}^{\text{Muon}} = 1) = P(\delta_{\text{conv}}^{\text{SGD}} = 1) \quad \text{（给定 } \gamma \text{ 下的收敛概率相等）}
$$
$$
H_1^{(\gamma)}: P(\delta_{\text{conv}}^{\text{Muon}} = 1) > P(\delta_{\text{conv}}^{\text{SGD}} = 1)
$$
检验统计量：两样本比例 z-检验
$$
Z = \frac{\hat{p}_{\text{Muon}} - \hat{p}_{\text{SGD}}}{\sqrt{\hat{p}(1-\hat{p})(2/n)}}, \quad \hat{p} = \frac{\hat{p}_{\text{Muon}} + \hat{p}_{\text{SGD}}}{2}
$$

对于 $K_\epsilon$（仅在收敛子样本中计算）：
$$
T^{(\gamma)} = \frac{\bar{D}}{S_D / \sqrt{n_{\text{conv}}}}, \quad D_j = \log K_{\epsilon,j}^{\text{Muon}} - \log K_{\epsilon,j}^{\text{SGD}}
$$

**与现有实验的关系**：扩展 E1 中固定的 $m=3d^2$（$\gamma=3$）到全范围 $\gamma$ 扫描。

**预期结果与解释**：
- **预期 1**：在欠采样（$\gamma < 1$）时，两种算法均可能不收敛或收敛极慢，但 Muon 的谱方向可能提供更稳定的梯度估计（低秩结构的先验）。
- **预期 2**：在临界采样（$\gamma \approx 1$）附近，Muon 的优势最大，因为此时问题刚好可解，谱信息的利用最为关键。
- **预期 3**：在过采样（$\gamma \geq 3$）时，两种算法均收敛，但 Muon 优势稳定（与现有实验一致）。
- **解释框架**：建立 "采样率–优势" 曲线 $\Delta(\gamma) = \mathbb{E}[\log K_\epsilon^{\text{SGD}} - \log K_\epsilon^{\text{Muon}} \mid \text{converged}]$。

---

### E9：权重衰减消融实验（超参数 D2 / 算法 A3）

**科学问题**：权重衰减（$L_2$ 正则化）如何改变 Muon 与 SGD 的收敛动态？Muon 的谱归一化方向与权重衰减的收缩效应是否存在协同或拮抗作用？

**数学形式**：

算法更新规则（含权重衰减）：
$$
\text{Muon:} \quad X^{(k+1)} = X^{(k)} - \eta D^{(k)} - \lambda X^{(k)} = (1-\lambda)X^{(k)} - \eta \cdot UV^\top
$$
$$
\text{SGD:} \quad X^{(k+1)} = X^{(k)} - \eta G^{(k)} - \lambda X^{(k)} = (1-\lambda)X^{(k)} - \eta G^{(k)}
$$

参数设置：
- $\lambda \in \{0, 10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}\}$
- $\eta$ 在每个 $(\lambda, \text{Algorithm})$ 组合上独立网格搜索（$\eta \in \{10^{-3}, 10^{-2}, 10^{-1}\}$）
- 问题：MS（$d=100, 200$）与 MF-L2（$d=100, 200$）
- $r = d/10$，$m = 3d^2$
- $n = 15$ 次重复
- 容差 $\epsilon = 10^{-6}$

**统计模型**：

因子设计：
$$
\text{Algorithm} \times \lambda \times d \times \text{Problem} \times \text{Seed}
$$
响应变量：$K_\epsilon$，$F_\epsilon$（总 FLOPs），$\|X^{(K)}\|_F$（最终参数范数）。

零假设与备择假设：
$$
H_0^{(\lambda)}: \text{给定 } \lambda, \; \mathbb{E}[\log K_\epsilon^{\text{Muon}}] = \mathbb{E}[\log K_\epsilon^{\text{SGD}}]
$$
$$
H_1^{(\lambda)}: \text{存在 } \lambda^* \text{ 使得 } \Delta(\lambda^*) = \max_{\lambda} \left|\mathbb{E}[\log K_\epsilon^{\text{Muon}} - \log K_\epsilon^{\text{SGD}}]\right|
$$
检验统计量：对每个 $\lambda$ 单独进行配对 t-检验；整体使用重复测量 ANOVA：
$$
F_{\text{Alg} \times \lambda} = \frac{\text{MS}_{\text{Alg} \times \lambda}}{\text{MS}_{\text{Error}}}
$$

**与现有实验的关系**：填补现有实验中 $\lambda = 0$ 的空白，直接控制权重衰减这一工业级优化中的关键超参数。

**预期结果与解释**：
- **预期 1**：当 $\lambda > 0$ 时，两种算法的有效收敛率均受 $\lambda$ 影响，但 Muon 的谱方向可能对收缩项的响应不同（因为 $UV^\top$ 与 $X^{(k)}$ 的谱结构相关）。
- **预期 2**：存在最优 $\lambda_{\text{Muon}}^*$ 与 $\lambda_{\text{SGD}}^*$，可能不同——若 Muon 隐式收缩（谱归一化本身抑制大奇异值），则其对外部 $\lambda$ 的敏感度更低。
- **预期 3**：在 MF 问题上，权重衰减可能改变平衡初始化（balanced initialization）的稳定性，Muon 的深度优势可能因此改变。

---

### E10：矩形矩阵实验（问题变体 B2）

**科学问题**：在非方阵（矩形）矩阵感知与矩形因子分解中，Muon 的谱归一化（基于 $G \in \mathbb{R}^{m \times n}$ 的 SVD）是否保持相对于 SGD 的优势？矩形性（aspect ratio $\alpha = m/n$）是否调节优势幅度？

**数学形式**：

矩形矩阵感知：
$$
X^\star \in \mathbb{R}^{m \times n}, \quad m \neq n, \quad \text{rank}(X^\star) = r
$$
$$
y_i = \langle A_i, X^\star \rangle, \quad A_i \in \mathbb{R}^{m \times n}, \quad A_{ij} \overset{iid}{\sim} \mathcal{N}(0, 1)
$$
测量数 $m_{\text{meas}} = 3mn$（保持与方阵类似的过采样率）。

矩形矩阵分解：
$$
W_1 \in \mathbb{R}^{d_1 \times d_0}, \; W_2 \in \mathbb{R}^{d_2 \times d_1}, \; \dots, \; W_L \in \mathbb{R}^{d_L \times d_{L-1}}, \quad d_L = m, \; d_0 = n
$$
$$
f_{\text{MF-rect}} = \frac{1}{2}\|W_L W_{L-1} \cdots W_1 - X^\star\|_F^2
$$

参数设置：
- 形状参数：$(m, n) \in \{(50, 100), (100, 50), (100, 200), (200, 100), (200, 500), (500, 200)\}$
- 纵横比 $\alpha = m/n \in \{0.5, 2.0\}$
- 秩 $r = \min(m, n) / 10$
- Muon / SGD，$\lambda=0$，$n=15$ 次重复
- 容差 $\epsilon = 10^{-6}$

**统计模型**：

因子设计：
$$
\text{Algorithm} \times \alpha \times \min(m,n) \times \text{Problem} \times \text{Seed}
$$
响应变量：$K_\epsilon$，$F_\epsilon$，$\bar{\sigma}_{\log}(G)$（矩形梯度矩阵的奇异值分布）。

零假设与备择假设：
$$
H_0: \text{对于矩形矩阵，} \mathbb{E}[K_\epsilon^{\text{Muon}} - K_\epsilon^{\text{SGD}}] = 0
$$
$$
H_1: \text{矩形性不消除 Muon 的优势（或增强优势）}
$$
检验统计量：配对 t-检验，按 $\alpha$ 分组；检验纵横比的调节效应：
$$
T_{\alpha} = \frac{\bar{D}_{\alpha} - 0}{S_{D_\alpha} / \sqrt{n}}
$$

**与现有实验的关系**：将方阵实验扩展到矩形，验证结论的矩阵形状泛化性。

**预期结果与解释**：
- **预期 1**：Muon 的优势在矩形设置下保持，因为 SVD 对任意 $m \times n$ 矩阵均有定义。
- **预期 2**：当 $\alpha \ll 1$（极度扁平）或 $\alpha \gg 1$（极度瘦高）时，SVD 的计算成本 $O(mn \cdot \min(m,n))$ 可能改变 FLOPs 效率的比较结果。
- **预期 3**：矩形分解的深度效应（H3）可能与方阵不同——维度不匹配可能引入额外的秩塌陷（rank collapse）模式。

---

### E11：与 Adam / RMSprop / AdaGrad 对比实验（算法基线 A6）

**科学问题**：Muon 是否仅优于 SGD，还是也优于工业标准自适应方法（Adam、RMSprop、AdaGrad）？自适应方法的逐元素增益与 Muon 的全谱归一化是竞争还是互补关系？

**数学形式**：

基线算法：
- **Adam**：$m^{(k)} = \beta_1 m^{(k-1)} + (1-\beta_1) G^{(k)}$，$v^{(k)} = \beta_2 v^{(k-1)} + (1-\beta_2) (G^{(k)})^2$，更新 $X^{(k+1)} = X^{(k)} - \eta \frac{m^{(k)}/(1-\beta_1^k)}{\sqrt{v^{(k)}/(1-\beta_2^k)} + \epsilon}$
- **RMSprop**：$v^{(k)} = \beta v^{(k-1)} + (1-\beta) (G^{(k)})^2$，$X^{(k+1)} = X^{(k)} - \eta \frac{G^{(k)}}{\sqrt{v^{(k)}} + \epsilon}$
- **AdaGrad**：$V^{(k)} = V^{(k-1)} + (G^{(k)})^2$，$X^{(k+1)} = X^{(k)} - \eta \frac{G^{(k)}}{\sqrt{V^{(k)}} + \epsilon}$
- **Momentum SGD**：$M^{(k)} = \beta M^{(k-1)} + G^{(k)}$，$X^{(k+1)} = X^{(k)} - \eta M^{(k)}$
- **L-BFGS**（二阶，作为参考）：使用 scipy.optimize 实现，记忆长度 $m=10$

所有自适应方法的学习率独立搜索（$\eta \in \{10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\}$）。

参数设置：
- 问题：MS（$d=100, 200$）与 MF-L2（$d=100, 200$）
- $r = d/10$，$m = 3d^2$
- 算法集合：Muon、SGD、Adam（$\beta_1=0.9, \beta_2=0.999$）、RMSprop（$\beta=0.99$）、Momentum SGD（$\beta=0.9$）
- $n = 15$ 次重复
- 容差 $\epsilon = 10^{-6}$，最大迭代 $K_{\max} = 10^5$

**统计模型**：

因子设计：
$$
\text{Algorithm} \times \text{Problem} \times d \times \text{Seed}
$$
响应变量：$K_\epsilon$，$F_\epsilon$（Adam 等自适应方法的 FLOPs 计数需额外考虑一阶/二阶矩累积）。

零假设与备择假设（多重比较框架）：
$$
H_0^{(i)}: \mathbb{E}[\log K_\epsilon^{\text{Muon}}] = \mathbb{E}[\log K_\epsilon^{\text{Algo}_i}], \quad i \in \{\text{Adam, RMSprop, Momentum, SGD}\}
$$
$$
H_1^{(i)}: \mathbb{E}[\log K_\epsilon^{\text{Muon}}] < \mathbb{E}[\log K_\epsilon^{\text{Algo}_i}]
$$
多重检验校正：使用 Holm 方法控制 FWER 在 $\alpha=0.05$。

检验统计量：
$$
T_i = \frac{\bar{D}_i}{S_{D_i} / \sqrt{n}}, \quad D_{i,j} = \log K_{\epsilon,j}^{\text{Muon}} - \log K_{\epsilon,j}^{\text{Algo}_i}
$$

**与现有实验的关系**：将基线从单一 SGD 扩展到完整的优化器基准集，建立 Muon 在优化器谱系中的真实位置。

**预期结果与解释**：
- **预期 1**：Adam 可能在中等精度（$\epsilon \sim 10^{-3}$）下接近 Muon，但在高精度（$\epsilon \sim 10^{-6}$）下被 Muon 超越，因为自适应增益的数值噪声累积。
- **预期 2**：Momentum SGD 可能缩小与 Muon 的差距（因为动量部分提供了方向平滑），但无法完全消除差距（因为谱归一化提供了额外的尺度不变性）。
- **预期 3**：L-BFGS 在中小规模（$d \leq 200$）上可能最优，验证 Muon 作为 "准二阶但一阶成本" 方法的定位。

---

### E12：Hessian 谱动态追踪实验（动态行为 G1/G4）

**科学问题**：在优化过程中，Hessian 矩阵的谱结构如何演化？Muon 的谱归一化方向是否与 Hessian 特征向量对齐？这种对齐度如何随迭代变化？

**数学形式**：

对于 MS 问题，Hessian 为：
$$
\nabla^2 f_{\text{MS}}(X) = \frac{1}{m}\sum_{i=1}^m \text{vec}(A_i) \text{vec}(A_i)^\top \in \mathbb{R}^{d^2 \times d^2}
$$
特征值 $\lambda_1(H) \geq \lambda_2(H) \geq \dots \geq \lambda_{d^2}(H)$。

对于 MF 问题（以 L=2 为例），$W = W_2 W_1$，Hessian 在平衡点附近的块结构（通过自动微分计算）。

追踪指标：
1. **条件数动态**：$\kappa_2^{(k)} = \lambda_{\max}(H^{(k)}) / \lambda_{\min}(H^{(k)})$
2. **谱归一化方向-Hessian 对齐度**：
   $$
   \mathcal{A}_{\text{Muon}}^{(k)} = \frac{\|H^{(k)} \text{vec}(D^{(k)})\|_2}{\lambda_{\max}(H^{(k)}) \|\text{vec}(D^{(k)})\|_2}, \quad \mathcal{A}_{\text{SGD}}^{(k)} = \frac{\|H^{(k)} \text{vec}(G^{(k)})\|_2}{\lambda_{\max}(H^{(k)}) \|\text{vec}(G^{(k)})\|_2}
   $$
   该量度量更新方向与 Hessian 最大曲率方向的对齐度。
3. **梯度-方向夹角**：
   $$
   \theta^{(k)} = \arccos\left(\frac{\langle G^{(k)}, D^{(k)}\rangle}{\|G^{(k)}\|_F \|D^{(k)}\|_F}\right)
   $$

参数设置：
- 问题：MS（$d=50, 100$）与 MF-L2（$d=50, 100$）
- $r = d/10$，$m = 3d^2$
- 每 $T = 50$ 步记录一次 Hessian 特征值（通过 Lanczos 迭代近似前 20 个特征值，避免 $O(d^4)$ 精确计算）
- Muon / SGD，$\lambda=0$
- $n = 10$ 次重复（动态追踪数据量大，减少重复数）
- 最大迭代 $K_{\max} = 5000$

**统计模型**：

时间序列分析框架：将 $\kappa_2^{(k)}$、$\mathcal{A}^{(k)}$、$\theta^{(k)}$ 视为离散时间随机过程。

零假设与备择假设：
$$
H_0: \mathbb{E}[\theta^{(k)}_{\text{Muon}}] = \mathbb{E}[\theta^{(k)}_{\text{SGD}}] \quad \forall k \in \{1, \dots, K/T\}
$$
$$
H_1: \exists \; k^* \text{ 使得 } \mathbb{E}[\theta^{(k^*)}_{\text{Muon}}] < \mathbb{E}[\theta^{(k^*)}_{\text{SGD}}] \text{ 且 } \mathcal{A}^{(k^*)}_{\text{Muon}} > \mathcal{A}^{(k^*)}_{\text{SGD}}
$$
（Muon 的方向更接近梯度且与 Hessian 对齐更好）

检验统计量：
- 对夹角序列进行函数型 ANOVA（Functional ANOVA）或使用重复测量混合效应模型：
  $$
  \theta_{ij}^{(k)} = \mu + \alpha_i^{\text{Alg}} + \beta_j^{\text{Seed}} + \gamma_k^{\text{Time}} + (\alpha\gamma)_{ik}^{\text{Alg}\times\text{Time}} + \varepsilon_{ijk}
  $$
  其中 $(\alpha\gamma)_{ik}$ 为关键交互项——算法随时间的不同演化模式。
- F-检验：$H_0: (\alpha\gamma)_{ik} = 0 \; \forall i, k$。

**与现有实验的关系**：全新的动态分析维度，提供现有静态谱指标（$\bar{\sigma}_{\log}$）的时间演化版本。

**预期结果与解释**：
- **预期 1**：SGD 的 $\theta^{(k)} = 0$ 恒成立（因为 $D^{(k)} = G^{(k)}$），而 Muon 的 $\theta^{(k)} > 0$ 但逐渐减小（谱归一化方向与梯度方向在高维空间中趋于对齐）。
- **预期 2**：在收敛早期，Muon 的 $\mathcal{A}^{(k)}$ 可能显著高于 SGD，说明谱归一化方向更好地利用了 Hessian 结构；在接近收敛时，两者差异缩小。
- **预期 3**：Hessian 条件数 $\kappa_2^{(k)}$ 在迭代中可能先增大后减小（"鞍点膨胀" 效应），Muon 可能更快穿越高条件数区域。

---

### E13：Wall-clock Time 实际测量实验（计算效率 E1/E2）

**科学问题**：Muons 每迭代的理论 FLOPs 优势在实际硬件上是否转化为 Wall-clock Time 优势？SVD 的开销在不同硬件（CPU vs GPU）、不同实现（稠密 SVD vs 随机 SVD）下如何影响实际效率？

**数学形式**：

计时指标定义：
- **每迭代时间**：$\tau_{\text{iter}}^{(\text{Alg})}$（秒/迭代）
- **总时间**：$T_\epsilon^{(\text{Alg})} = K_\epsilon^{(\text{Alg})} \times \tau_{\text{iter}}^{(\text{Alg})}$
- **达到精度 $\epsilon$ 的时间-精度曲线**：$\{(\tau_{\text{cum}}^{(k)}, f(X^{(k)}))\}_{k=0}^{K_\epsilon}$

实现变体：
- **Muon-Exact**：`numpy.linalg.svd()`（Golub-Reinsch，$O(d^3)$）
- **Muon-RandomSVD**：Halko 算法（$O(d^2 r_{\text{approx}})$，$r_{\text{approx}} = 2r$）
- **Muon-Truncated**：`scipy.sparse.linalg.svds()`（仅计算前 $r_{\max}$ 个奇异值）
- **SGD**：标准矩阵运算（$O(d^2)$ 每迭代）

参数设置：
- 问题：MS（$d=50, 100, 200, 500$）
- $r = d/10$，$m = 3d^2$
- 算法：Muon-Exact、Muon-RandomSVD、Muon-Truncated、SGD
- 每种配置 $n=10$ 次重复计时（取中位数消除异常值）
- 硬件：标准 CPU（单核/多核）与可选 GPU
- 容差 $\epsilon = 10^{-6}$

**统计模型**：

因子设计：
$$
\text{Algorithm-Variant} \times d \times \text{Hardware} \times \text{Seed}
$$
响应变量：$\tau_{\text{iter}}$（秒），$T_\epsilon$（总时间），峰值内存 $M_{\text{peak}}$（MB）。

零假设与备择假设：
$$
H_0^{(d)}: T_\epsilon^{\text{Muon-Exact}} = T_\epsilon^{\text{SGD}} \quad \text{（实际时间无差异）}
$$
$$
H_1^{(d)}: T_\epsilon^{\text{Muon-Exact}} > T_\epsilon^{\text{SGD}} \quad \text{（SVD 开销抵消迭代优势）}
$$
附加假设：
$$
H_0^{\text{Rand}}: T_\epsilon^{\text{Muon-RandomSVD}} = T_\epsilon^{\text{Muon-Exact}}
$$
$$
H_1^{\text{Rand}}: T_\epsilon^{\text{Muon-RandomSVD}} < T_\epsilon^{\text{Muon-Exact}} \quad \text{（随机 SVD 加速有效）}
$$

检验统计量：配对 Wilcoxon 符号秩检验（时间数据可能偏态）：
$$
W = \sum_{j=1}^n \text{rank}(|D_j|) \cdot \mathbb{I}(D_j > 0), \quad D_j = T_{\epsilon,j}^{\text{Alg}_1} - T_{\epsilon,j}^{\text{Alg}_2}
$$

**与现有实验的关系**：将现有实验的理论 FLOPs 计数替换为实际 Wall-clock Time 测量，是现有计算效率分析的关键验证。

**预期结果与解释**：
- **预期 1**：对于 $d \leq 100$，Muon-Exact 的 $T_\epsilon$ 可能大于 SGD（SVD 常数因子高）；对于 $d \geq 500$，Muons 的迭代优势可能压倒 SVD 开销。
- **预期 2**：Muon-RandomSVD 在大 $d$ 上显著快于 Muon-Exact，但可能损失精度（需要量化精度-速度权衡）。
- **预期 3**：内存占用方面，Muon 需要存储梯度并执行 SVD（临时 $O(d^3)$），而 SGD 仅需 $O(d^2)$。存在某个 $d_{\text{mem}}$ 使得 Muon 因内存压力而退化。

---

### E14：随机 SVD 近似精度-效率权衡实验（计算效率 E3）

**科学问题**：当使用随机 SVD 近似替代精确 SVD 时，Muon 的收敛精度与速度如何权衡？近似误差阈值 $\delta_{\text{svd}}$ 与收敛容差 $\epsilon$ 之间的关系是什么？

**数学形式**：

随机 SVD（Halko et al.）参数化：
$$
\text{RandomSVD}(G, r_{\text{approx}}, q, p): \quad \tilde{U}\tilde{\Sigma}\tilde{V}^\top \approx G
$$
其中 $r_{\text{approx}}$ 为近似秩，$q$ 为幂迭代次数，$p$ 为过采样参数。

近似方向：
$$
\tilde{D}^{(k)} = \tilde{U}\tilde{V}^\top \quad \text{或} \quad \tilde{D}^{(k)} = \tilde{U}\tilde{\Sigma}^{-1}\tilde{V}^\top
$$

近似误差度量：
$$
\delta_{\text{svd}} = \|G - \tilde{U}\tilde{\Sigma}\tilde{V}^\top\|_F / \|G\|_F
$$

参数设置：
- 问题：MS（$d=200, 500$）
- $r = d/10$
- 随机 SVD 参数组合：
  - $r_{\text{approx}} \in \{r, 2r, 5r, 10r\}$
  - $q \in \{0, 1, 2\}$
  - $p \in \{5, 10, 20\}$
- Muon-Exact 作为基准
- $n = 10$ 次重复
- 容差 $\epsilon = 10^{-6}$

**统计模型**：

因子设计：
$$
\text{SVD-Variant} \times r_{\text{approx}}/r \times q \times d \times \text{Seed}
$$
响应变量：
- $K_\epsilon$（收敛迭代，若近似过粗则可能不收敛）
- $\tau_{\text{iter}}$（每迭代时间）
- $\delta_{\text{svd}}$（每步近似误差）
- 综合效率：$E = K_\epsilon \times \tau_{\text{iter}}$

零假设与备择假设：
$$
H_0^{(r_{\text{approx}}, q)}: K_\epsilon^{\text{RandomSVD}(r_{\text{approx}}, q)} = K_\epsilon^{\text{Exact}}
$$
$$
H_1^{(r_{\text{approx}}, q)}: \text{存在 } (r_{\text{approx}}^*, q^*) \text{ 使得 } E^{\text{RandomSVD}} < E^{\text{Exact}} \text{ 且 } K_\epsilon^{\text{RandomSVD}} \approx K_\epsilon^{\text{Exact}}
$$
检验统计量：对每种参数组合进行配对 t-检验；使用响应面方法寻找最优 $(r_{\text{approx}}^*, q^*)$。

**与现有实验的关系**：扩展 E13 的 Wall-clock 分析，专门聚焦于 SVD 近似策略。

**预期结果与解释**：
- **预期 1**：存在 "足够好的近似" 阈值——当 $r_{\text{approx}} \geq 2r$ 且 $q \geq 1$ 时，$K_\epsilon$ 与精确 SVD 几乎相同，但 $\tau_{\text{iter}}$ 降低 2–5 倍。
- **预期 2**：过低的 $r_{\text{approx}}$（如 $r_{\text{approx}} = r$）可能导致方向质量下降，$K_\epsilon$ 反而增加，抵消时间收益。
- **预期 3**：幂迭代 $q$ 的边际效益递减——$q=1$ 通常足够，$q=2$ 的提升有限但成本显著增加。

---

### E15：更大规模可扩展性实验（计算效率 E4）

**科学问题**：当维度 $d$ 从 500 扩展到 1000、2000 时，Muons 的 $O(d^3)$ SVD 开销是否构成硬墙？在什么规模下，Muon 的任何迭代优势都被 SVD 成本完全吞噬？

**数学形式**：

规模序列：
$$
d \in \{500, 1000, 2000\}
$$
对应问题规模：
- MS：$m = 3d^2 \in \{7.5 \times 10^5, 3 \times 10^6, 1.2 \times 10^7\}$ 个测量
- MF：$W_i \in \mathbb{R}^{d \times d}$

SVD 复杂度边界：
$$
T_{\text{SVD}}(d) = C_{\text{svd}} \cdot d^3, \quad C_{\text{svd}} \approx 10^{-8} \text{ s（典型 CPU）}
$$
$$
T_{\text{SGD-iter}}(d) = C_{\text{sgd}} \cdot d^2, \quad C_{\text{sgd}} \approx 10^{-9} \text{ s}
$$
交点规模 $d^*$ 满足：
$$
K_{\text{Muon}} \cdot C_{\text{svd}} d^3 = K_{\text{SGD}} \cdot C_{\text{sgd}} d^2 \quad \Rightarrow \quad d^* = \frac{K_{\text{SGD}}}{K_{\text{Muon}}} \cdot \frac{C_{\text{sgd}}}{C_{\text{svd}}}
$$
若 $K_{\text{SGD}} / K_{\text{Muon}} \approx 10$，则 $d^* \approx 10 \times 0.1 = 1$（即 SGD 始终更快），但这与理论优势矛盾——说明常数因子估计需实验校准。

参数设置：
- 问题：MS（$r = d/10$，$m = 3d^2$）
- 算法：Muon-Exact、Muon-RandomSVD($2r$)、SGD
- $d \in \{500, 1000, 2000\}$
- $n = 5$ 次重复（大规模实验成本高）
- 容差 $\epsilon = 10^{-4}$（适度精度，确保在合理时间内完成）
- 最大 Wall-clock 时间：每配置 4 小时

**统计模型**：

因子设计：
$$
\text{Algorithm} \times d \times \text{Seed}
$$
响应变量：
- $K_\epsilon$（若收敛）
- $T_\epsilon$（总 Wall-clock Time）
- $\delta_{\text{conv}} \in \{0, 1\}$（在时限内是否收敛）
- 扩展效率：$S(d) = T_\epsilon(d/2) / T_\epsilon(d)$（理想值为 8 对于 Muon 精确 SVD，4 对于 SGD）

零假设与备择假设：
$$
H_0: T_\epsilon^{\text{Muon}}(d) / T_\epsilon^{\text{SGD}}(d) \leq 1 \quad \forall d \in \{500, 1000, 2000\}
$$
$$
H_1: \exists \; d_{\text{cross}} \text{ 使得 } d > d_{\text{cross}} \Rightarrow T_\epsilon^{\text{Muon}}(d) > T_\epsilon^{\text{SGD}}(d)
$$
检验统计量：对每种 $d$ 进行时间比率的配对检验：
$$
R_j(d) = T_{\epsilon,j}^{\text{Muon}}(d) / T_{\epsilon,j}^{\text{SGD}}(d)
$$
使用对数转换：$\log R_j(d)$ 的单样本 t-检验 vs 0。

**与现有实验的关系**：将现有维度上限从 $d=500$ 扩展到 $d=2000$，验证规模外推性。

**预期结果与解释**：
- **预期 1**：对于 $d=1000$，Muon-RandomSVD 仍可能保持时间优势；对于 $d=2000$，精确 SVD 可能因内存/时间限制而不实用。
- **预期 2**：存在明确的 $d_{\text{cross}}$（预计 $1500 < d_{\text{cross}} < 3000$ 对于精确 SVD），超过后随机 SVD 成为 Muon 的唯一可行实现。
- **预期 3**：扩展效率 $S(d)$ 将揭示实现质量——若 $S(d) \gg 8$，说明存在严重的内存瓶颈或缓存失效。

---

### E16：初始化尺度敏感性实验（初始化 C1/C3）

**科学问题**：Muon 的谱归一化是否对初始化尺度具有内在的不变性？即，若初始梯度 $G^{(0)}$ 的尺度变化 $c$ 倍，Muons 的方向 $D^{(0)} = UV^\top$ 是否完全不受影响？不同初始化方差对收敛率的影响在 Muon 与 SGD 之间是否不同？

**数学形式**：

初始化分布：
$$
X^{(0)}_{ij} \overset{iid}{\sim} \mathcal{N}(0, \sigma_{\text{init}}^2), \quad \sigma_{\text{init}} \in \left\{\frac{10^{-3}}{\sqrt{d}}, \frac{10^{-2}}{\sqrt{d}}, \frac{10^{-1}}{\sqrt{d}}, \frac{1}{\sqrt{d}}, \frac{10}{\sqrt{d}}\right\}
$$
（$1/\sqrt{d}$ 为现有 "标准" 初始化）

关键理论关系：
- SGD 更新 $X^{(1)} = X^{(0)} - \eta G^{(0)}$ 直接受 $G^{(0)}$ 的尺度影响（$\|G^{(0)}\|_F \propto \sigma_{\text{init}}$ 对于 MS 问题）。
- Muon 更新 $X^{(1)} = X^{(0)} - \eta UV^\top$，其中 $G^{(0)} = U\Sigma V^\top$，故 $UV^\top$ 在理论上与 $\Sigma$ 无关。

但实践中：
- 若 $G^{(0)} \approx 0$（极小初始化），数值 SVD 不稳定。
- 若 $G^{(0)}$ 极大，可能触发浮点溢出或学习率 $\eta$ 需要重新校准。

参数设置：
- 问题：MS（$d=100, 200$）与 MF-L2（$d=100, 200$）
- $r = d/10$，$m = 3d^2$
- 五种 $\sigma_{\text{init}}$ 值
- Muon / SGD，$\lambda=0$
- $\eta$ 在每个 $(\sigma_{\text{init}}, \text{Algorithm})$ 上独立搜索
- $n = 15$ 次重复
- 容差 $\epsilon = 10^{-6}$

**统计模型**：

因子设计：
$$
\text{Algorithm} \times \sigma_{\text{init}} \times d \times \text{Problem} \times \text{Seed}
$$
响应变量：$K_\epsilon$，$\eta^*$（最优学习率），$f_{\text{final}}$，$\|\nabla f(X^{(0)})\|_F$。

零假设与备择假设：
$$
H_0^{\text{scale-inv}}: \text{对于 Muon，} K_\epsilon \text{ 不依赖于 } \sigma_{\text{init}} \quad \text{（尺度不变性假设）}
$$
$$
H_1^{\text{scale-inv}}: \exists \; \sigma_{\text{init}}^1, \sigma_{\text{init}}^2 \text{ 使得 } K_\epsilon(\sigma_{\text{init}}^1) \neq K_\epsilon(\sigma_{\text{init}}^2) \text{ 对于 Muon}
$$
$$
H_0^{\text{SGD-dep}}: \text{对于 SGD，} K_\epsilon \text{ 不依赖于 } \sigma_{\text{init}}
$$
$$
H_1^{\text{SGD-dep}}: K_\epsilon^{\text{SGD}} \text{ 显著依赖于 } \sigma_{\text{init}}
$$

检验统计量：
- 对每种算法，单因子 ANOVA（因子：$\sigma_{\text{init}}$）：
  $$
  F = \frac{\text{MS}_{\sigma_{\text{init}}}}{\text{MS}_{\text{Error}}}
  $$
- 若 Muon 的 ANOVA 不显著而 SGD 的显著，支持尺度不变性假说。
- 同时检验学习率校准需求：$\eta^*$ 随 $\sigma_{\text{init}}$ 的变化幅度。

**与现有实验的关系**：扩展现有实验的固定初始化 $\mathcal{N}(0, 1/d)$ 到全尺度扫描，验证 Muon 的理论尺度不变性。

**预期结果与解释**：
- **预期 1**：Muon 的 $K_\epsilon$ 对 $\sigma_{\text{init}}$ 的依赖性显著弱于 SGD，验证谱归一化的尺度不变性。
- **预期 2**：在极端尺度（$\sigma_{\text{init}} = 10^{-3}/\sqrt{d}$ 或 $10/\sqrt{d}$）时，Muon 也可能退化——过小导致数值 SVD 不稳定，过大导致初始步越过 basin。
- **预期 3**：最优学习率 $\eta^*_{\text{Muon}}$ 对 $\sigma_{\text{init}}$ 的敏感度低于 $\eta^*_{\text{SGD}}$，这是 Muon 在实际调参中的实用优势。

---

### E17：正交与谱初始化对比实验（初始化 C2）

**科学问题**：若初始化已经具有特定谱结构（如正交初始化、谱初始化），Muon 的谱归一化是否仍然提供额外价值？或者说，当初始化已经 "好" 时，Muon 与 SGD 的差距是否缩小？

**数学形式**：

初始化策略：
1. **高斯初始化**（基准）：$X^{(0)}_{ij} \sim \mathcal{N}(0, 1/d)$
2. **正交初始化**：$X^{(0)} = Q R$（QR 分解，$Q$ 正交/酉），或对于矩形矩阵使用部分 QR。
3. **谱初始化**（Spectral Init）：$X^{(0)} = U^{(0)} \text{diag}(\lambda^{(0)}) (V^{(0)})^\top$，其中
   $$
   \lambda_i^{(0)} = \lambda_1 \cdot \rho^{i-1}, \quad \rho \in \{0.5, 0.9\}
   $$
   即初始化已经具有与真实矩阵相同的谱衰减模式。
4. **零初始化**：$X^{(0)} = 0$（测试 Muon 在零梯度下的退化）

参数设置：
- 问题：MS（$d=100, 200$）与 MF-L2（$d=100, 200$）
- $r = d/10$
- 四种初始化策略
- Muon / SGD，$\lambda=0$，$\eta$ 独立搜索
- $n = 15$ 次重复
- 容差 $\epsilon = 10^{-6}$

**统计模型**：

因子设计：
$$
\text{Algorithm} \times \text{Init-Strategy} \times d \times \text{Problem} \times \text{Seed}
$$
响应变量：$K_\epsilon$，$K_{\epsilon,\text{normalized}} = K_\epsilon \cdot \|X^{(0)} - X^\star\|_F$（归一化到初始距离）。

零假设与备择假设：
$$
H_0^{\text{init}}: \text{算法} \times \text{Init-Strategy} \text{ 对 } K_\epsilon \text{ 无交互效应}
$$
$$
H_1^{\text{init}}: \text{Muon 对初始化策略的敏感度低于 SGD} \quad \text{（或相反）}
$$
检验统计量：双因子 ANOVA 交互检验
$$
F_{\text{int}} = \frac{\text{MS}_{\text{Alg} \times \text{Init}}}{\text{MS}_{\text{Error}}}
$$

事后分析（Post-hoc）：Tukey HSD 检验比较各初始化策略下的算法差异。

**与现有实验的关系**：将现有实验中的三种特殊初始化（针对 MF-L2）扩展到系统性的初始化策略对比。

**预期结果与解释**：
- **预期 1**：谱初始化可能同时加速 Muon 与 SGD，但 Muon 的增益更小（因为谱归一化已经提供了类似的 "结构化" 信息）。
- **预期 2**：零初始化对 Muon 是极端测试——若 $G^{(0)} = 0$，SVD 退化，需要分析实现如何处理零梯度。
- **预期 3**：正交初始化可能为 Muon 提供更有利的初始 Hessian 条件，从而放大优势。

---

### E18：条件数控制实验（问题变体 B5）

**科学问题**：问题的病态程度（Hessian 条件数）是否是 Muon 优势的关键调节变量？在良态问题（$\kappa \approx 1$）与病态问题（$\kappa \approx 10^6$）之间，Muons 的收敛率如何相对变化？

**数学形式**：

条件数控制方法（感知算子构造）：

目标：构造感知算子族 $\{\mathcal{A}_\kappa\}$ 使得 Hessian $H = \mathcal{A}^*\mathcal{A}/m$ 具有指定条件数 $\kappa_{\text{target}}$。

方法 1（参数化矩阵族）：
$$
H = Q^\top \text{diag}(\lambda_1, \dots, \lambda_{d^2}) Q, \quad \lambda_i = 1 + (\kappa_{\text{target}} - 1) \cdot \frac{i-1}{d^2-1}
$$
（线性谱分布，从 1 到 $\kappa_{\text{target}}$）

通过 Cholesky 分解 $H = L L^\top$，构造感知向量：
$$
\text{vec}(A_i) = \text{row}_i(L) + \mathcal{N}(0, \sigma^2 I_{d^2})
$$
（添加小噪声使问题非退化）

方法 2（SVD 注水法）：
直接构造 $X^\star = U^\star \text{diag}(\lambda_1, \dots, \lambda_r) (V^\star)^\top$，其中 $\lambda_i$ 按指定衰减率分布。

条件数水平：
$$
\kappa_{\text{target}} \in \{10, 10^2, 10^3, 10^4, 10^5, 10^6\}
$$

参数设置：
- 问题：MS（$d=100$）
- $r = 10$（固定），$m = 3d^2 = 30000$
- 六种条件数水平
- 两种谱分布：线性衰减 vs 几何衰减
- Muon / SGD，$\lambda=0$，$\eta$ 独立搜索
- $n = 15$ 次重复
- 容差 $\epsilon = 10^{-6}$，最大迭代 $K_{\max} = 5 \times 10^5$（病态问题需更多迭代）

**统计模型**：

因子设计：
$$
\text{Algorithm} \times \kappa_{\text{target}} \times \text{Spectrum-Type} \times \text{Seed}
$$
响应变量：
- $K_\epsilon$（收敛迭代）
- $\kappa_{\text{eff}} = \lambda_{\max}(H)/\lambda_{\min}(H)$（实际测量的条件数，验证构造精度）
- 收敛率估计（局部线性拟合 $\log(f(X^{(k)})) \sim -\rho k$）

零假设与备择假设：
$$
H_0^{(\kappa)}: \mathbb{E}[\log K_\epsilon^{\text{Muon}} \mid \kappa] - \mathbb{E}[\log K_\epsilon^{\text{SGD}} \mid \kappa] = \text{const} \quad \forall \kappa
$$
$$
H_1^{(\kappa)}: \text{优势 } \Delta(\kappa) = \mathbb{E}[\log K_\epsilon^{\text{SGD}}] - \mathbb{E}[\log K_\epsilon^{\text{Muon}}] \text{ 随 } \kappa \text{ 单调递增}
$$
（即病态问题放大 Muon 优势）

检验统计量：
- Spearman 秩相关检验：$\Delta(\kappa)$ 与 $\log \kappa$ 的相关性
  $$
  \rho_S = \text{corr}_{\text{rank}}(\Delta(\kappa), \log \kappa)
  $$
  $H_0: \rho_S = 0$，$H_1: \rho_S > 0$。
- 或线性模型：
  $$
  \log K_\epsilon^{(i,j)} = \beta_0 + \beta_1 \log \kappa_j + \beta_2 \mathbb{I}_{\text{Muon}} + \beta_3 \log \kappa_j \cdot \mathbb{I}_{\text{Muon}} + \varepsilon_{ij}
  $$
  检验 $\beta_3 < 0$（Muon 对条件数的敏感度更低）。

**与现有实验的关系**：全新维度——现有实验使用随机矩阵的自然条件数，未进行系统性控制。

**预期结果与解释**：
- **预期 1**：在良态问题（$\kappa \approx 10$）上，Muon 与 SGD 差异较小。
- **预期 2**：在病态问题（$\kappa \approx 10^6$）上，Muon 优势显著放大，因为谱归一化方向避开了梯度在 Hessian 小特征值方向上的振荡。
- **预期 3**：建立定量关系 $\Delta(\kappa) \approx c \cdot \log \kappa$，为理论分析提供经验系数。

---

### E19：不同矩阵分布泛化性实验（泛化性 H1/H3）

**科学问题**：现有实验结论是否依赖于高斯测量矩阵的特定性质（如亚高斯性、旋转不变性）？当测量矩阵来自其他分布（Rademacher、结构化、球形高斯）时，Muons 的相对性能是否保持？

**数学形式**：

测量矩阵分布族：
1. **高斯**（基准）：$A_{ij} \overset{iid}{\sim} \mathcal{N}(0, 1/d^2)$
2. **Rademacher**：$A_{ij} \overset{iid}{\sim} \text{Unif}\{-1/d, +1/d\}$
3. **伯努利-稀疏**：$A_{ij} \overset{iid}{\sim} \text{Bernoulli}(p) \cdot \mathcal{N}(0, 1/(pd^2))$，$p=0.1$
4. **球形高斯**：$\text{vec}(A_i) \sim \text{Unif}(\mathbb{S}^{d^2-1})$（单位球面上均匀）
5. **快速 JL 变换**：$A_i = P H D \text{vec}(e_i)$，其中 $H$ 为 Hadamard 矩阵，$D$ 为随机对角 $\pm 1$，$P$ 为随机子采样

目标矩阵分布：
- 标准：$X^\star = U \Sigma V^\top$，$U, V \sim \text{Haar}$
- 多项式衰减谱：$\lambda_i = i^{-\alpha}$，$\alpha \in \{0.5, 1.0, 2.0\}$
- 指数衰减谱：$\lambda_i = e^{-\beta i}$，$\beta \in \{0.1, 0.5\}$

参数设置：
- 问题：MS（$d=100, 200$）
- $r = d/10$，$m = 3d^2$
- 五种测量分布 $\times$ 三种目标谱型
- Muon / SGD，$\lambda=0$
- $n = 15$ 次重复
- 容差 $\epsilon = 10^{-6}$

**统计模型**：

因子设计：
$$
\text{Algorithm} \times \text{Meas-Dist} \times \text{Target-Spec} \times d \times \text{Seed}
$$
响应变量：$K_\epsilon$，$\bar{\sigma}_{\log}(G^{(0)})$，$\kappa_{\text{sp}}$。

零假设与备择假设：
$$
H_0: \text{测量分布不影响 } \Delta K = K_\epsilon^{\text{SGD}} - K_\epsilon^{\text{Muon}} \text{ 的分布}
$$
$$
H_1: \exists \; \text{Meas-Dist} \text{ 使得 } \Delta K \text{ 显著不同于高斯基准}
$$
检验统计量：
- Kruskal-Wallis H-检验（多组非参数检验）：
  $$
  H = \frac{12}{N(N+1)}\sum_{i=1}^5 \frac{R_i^2}{n_i} - 3(N+1)
  $$
  其中 $R_i$ 为第 $i$ 个分布组的平均秩，$N=5n$，$n_i=n$。若 $H > \chi^2_{0.95, 4}$，拒绝 $H_0$。
- 事后：Dunn 检验进行两两比较。

**与现有实验的关系**：将现有实验的固定高斯测量推广到完整分布族，验证结论的分布鲁棒性。

**预期结果与解释**：
- **预期 1**：亚高斯分布（Rademacher、球形高斯）与高斯的结果相似，因为随机感知理论保证亚高斯矩阵满足 RIP。
- **预期 2**：结构化矩阵（快速 JL 变换）可能因相关性结构而引入不同动态，但低秩恢复的极限行为可能一致。
- **预期 3**：目标谱型强烈影响结果——多项式/指数衰减谱可能比 "硬截断" 低秩谱更容易被 Muon 利用。

---

### E20：统计功效与样本量确定实验（统计可靠性 F1/F2）

**科学问题**：现有实验使用 $n=10$ 次重复，这一样本量是否足够检测 Muon 与 SGD 之间真实的收敛率差异？若不足，需要多少次重复才能达到 80% 功效？同时，如何为所有假设检验提供严格的置信区间与多重检验校正？

**数学形式**：

功效分析框架：

定义效应量（Effect Size）：
$$
\delta = \frac{\mathbb{E}[\log K_\epsilon^{\text{SGD}}] - \mathbb{E}[\log K_\epsilon^{\text{Muon}}]}{\sigma_{\text{within}}}
$$
其中 $\sigma_{\text{within}}$ 为配对差值的标准差。

从现有数据（或 E6–E19 的预实验）估计：
- 基准均值：$\mu_{\text{SGD}} = \mathbb{E}[\log K_\epsilon^{\text{SGD}}]$，$\mu_{\text{Muon}} = \mathbb{E}[\log K_\epsilon^{\text{Muon}}]$
- 差值标准差：$\sigma_D = \text{SD}(\log K_\epsilon^{\text{SGD}} - \log K_\epsilon^{\text{Muon}})$

功效计算：
对于配对 t-检验，给定 $\alpha = 0.05$（单侧），样本量 $n$，效应量 $\delta$：
$$
\text{Power} = 1 - \beta = \Phi\left(\sqrt{n} \cdot \delta - z_{1-\alpha}\right)
$$
其中 $\Phi$ 为标准正态 CDF，$z_{1-\alpha} = 1.645$。

求解 $n_{\min}$ 使得 $\text{Power} \geq 0.8$：
$$
n_{\min} = \left\lceil \frac{(z_{1-\alpha} + z_{1-\beta})^2}{\delta^2} \right\rceil = \left\lceil \frac{(1.645 + 0.84)^2}{\delta^2} \right\rceil \approx \left\lceil \frac{6.2}{\delta^2} \right\rceil
$$

参数设置：
- 使用 E1（MS）与 E2（MF-L2）的数据（或若尚不可用，运行 $n=30$ 的扩展实验）
- 问题：MS（$d=100, 200$），MF-L2（$d=100, 200$）
- 计算每个 $(d, \text{Problem})$ 组合的 $\delta(d)$ 与 $n_{\min}(d)$
- 对所有现有假设 H1–H5 进行多重检验校正分析

**统计模型**：

Bootstrap 置信区间构造：
对于响应变量 $Y = \log K_\epsilon$，使用 BCa Bootstrap：
$$
Y^{*(b)}_j = \text{resample}(Y_1, \dots, Y_n), \quad b = 1, \dots, B=10000
$$
$$
CI_{95\%} = [\hat{Y}^{*(\alpha/2)}, \hat{Y}^{*(1-\alpha/2)}]
$$

多重检验校正：
现有实验涉及假设 H1–H5，共 5 个主假设。若在每个维度水平上进行检验，假设总数 $M$ 可能达到 $5 \times 4 \times 2 = 40$（维度 × 问题 × 秩）。

使用 Holm 逐步校正：
$$
p_{(1)} \leq p_{(2)} \leq \dots \leq p_{(M)} \quad \text{（排序 p 值）}
$$
拒绝所有满足 $p_{(i)} \leq \alpha / (M - i + 1)$ 的假设。

或使用 Benjamini-Hochberg FDR 控制：
$$
\text{拒绝所有 } p_{(i)} \leq \frac{i}{M} \cdot \alpha
$$

零假设与备择假设：
$$
H_0^{\text{power}}: n=10 \text{ 足以检测 } \delta_{\min} = 0.5 \text{（中等效应量）}
$$
$$
H_1^{\text{power}}: n_{\min} > 10 \text{ 才能达到 } 80\% \text{ 功效}
$$
检验：直接计算 $\hat{\delta}$ 与 $\hat{n}_{\min}$。

**与现有实验的关系**：对现有实验设计的统计严格性进行元分析，确定是否需要增加重复次数。

**预期结果与解释**：
- **预期 1**：对于 $d=50$，效应量可能较大（$\delta \approx 1.0$），$n=10$ 足够；对于 $d=500$，效应量可能较小（$\delta \approx 0.3$），$n=10$ 功效不足，建议 $n_{\min} \approx 30$–$50$。
- **预期 2**：在 $n=10$ 下，部分 "边缘显著" 的结果（$p \approx 0.04$）在 Holm 校正后可能不再显著，需要重新评估结论的稳健性。
- **预期 3**：提供标准化的效应量报告模板（Cohen's $d$、对数比率、加速比及其置信区间），提升实验的可重复性与可比性。

---

## 20. 补充后的完整实验矩阵

将现有实验（E1–E5）与补充实验（E6–E20）整合为统一的析因设计矩阵。矩阵列表示实验维度，行表示实验定义。

### 3.1 现有实验回顾（E1–E5）

| 编号 | 实验名称 | 问题类型 | 算法 | 维度参数 | 关键因子 | 重复数 | 核心假设 |
|------|----------|----------|------|----------|----------|--------|----------|
| E1 | 矩阵感知基准 | MS | Muon, SGD | $d=50,100,200,500$；$r=d/10$ 或满秩 | 维度 × 秩结构 | 10 | H1, H2 |
| E2 | 矩阵分解基准 | MF-L=2,3,4 | Muon, SGD | $d=50,100,200$；L=2,3,4 | 深度 × 维度 | 10 | H3 |
| E3 | 学习率校准 | MS, MF | Muon, SGD | $d=100,200$ | $\eta$ 网格搜索 | 10 | H4 |
| E4 | 稳定性分析 | MS, MF | Muon, SGD | $d=100$ | 初始化 × 噪声 | 10 | H4, H5 |
| E5 | FLOPs 效率 | MS, MF | Muon, SGD | $d=50,100,200,500$ | 理论计算 | — | H5 |

### 3.2 补充实验矩阵（E6–E20）

| 编号 | 实验名称 | 覆盖维度 | 问题类型 | 算法变体 | 新增因子 | 重复数 | 核心假设 | 与现有关系 |
|------|----------|----------|----------|----------|----------|--------|----------|------------|
| E6 | 噪声敏感性 | B3 | MS, MF | Muon, SGD | $\sigma_\epsilon \in \{0, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\}$ | 20 | 噪声优势边界 | 扩展 E1/E2 |
| E7 | 秩比例扫描 | B1 | MS | Muon, SGD | $r/d \in \{0.01, 0.05, 0.1, 0.2, 0.5, 1.0\}$ | 15 | 秩–优势交互 | 扩展 E1 |
| E8 | 过/欠采样 | B4 | MS | Muon, SGD | $\gamma = m/d^2 \in \{0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0\}$ | 15 | 采样率–优势曲线 | 扩展 E1 |
| E9 | 权重衰减消融 | A3, D2 | MS, MF | Muon, SGD | $\lambda \in \{0, 10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}\}$ | 15 | $\lambda$ 交互效应 | 全新维度 |
| E10 | 矩形矩阵 | B2 | MS, MF-rect | Muon, SGD | $(m,n)$ 六组形状 | 15 | 形状泛化性 | 扩展 E1/E2 |
| E11 | 多基线对比 | A6 | MS, MF | Muon, SGD, Adam, RMSprop, Momentum, L-BFGS | 6 种算法 | 15 | Muon 在谱系中的位置 | 扩展基线 |
| E12 | Hessian 谱动态 | G1, G4 | MS, MF-L2 | Muon, SGD | 时间序列追踪（每 50 步） | 10 | 方向–Hessian 对齐 | 全新维度 |
| E13 | Wall-clock 时间 | E1, E2 | MS | Muon-Exact, Muon-RandSVD, Muon-Trunc, SGD | 实现变体 × 硬件 | 10 | 理论–实际差距 | 验证 E5 |
| E14 | 随机 SVD 权衡 | E3 | MS | Muon-Exact, Muon-RandomSVD | $(r_{\text{approx}}, q, p)$ 参数 | 10 | 精度–速度帕累托前沿 | 扩展 E13 |
| E15 | 大规模可扩展性 | E4 | MS | Muon-Exact, Muon-RandSVD, SGD | $d \in \{500, 1000, 2000\}$ | 5 | 规模硬墙 | 扩展 E1 |
| E16 | 初始化尺度敏感性 | C1, C3 | MS, MF | Muon, SGD | $\sigma_{\text{init}}$ 五水平 | 15 | 尺度不变性验证 | 扩展 E2 |
| E17 | 正交/谱初始化 | C2 | MS, MF | Muon, SGD | 四种初始化策略 | 15 | 初始化–算法交互 | 扩展 E2 |
| E18 | 条件数控制 | B5 | MS | Muon, SGD | $\kappa_{\text{target}} \in \{10, 10^2, \dots, 10^6\}$ | 15 | 病态放大优势 | 全新维度 |
| E19 | 矩阵分布泛化 | H1, H3 | MS | Muon, SGD | 五种测量分布 × 三种目标谱型 | 15 | 分布鲁棒性 | 扩展 E1 |
| E20 | 功效与样本量 | F1, F2, F3 | MS, MF | Muon, SGD | 样本量扫描 + Bootstrap | — | 统计严格性 | 元分析 |

### 3.3 整合后的超级析因设计

将所有实验统一表示为一个高维因子空间 $\mathcal{E} = (\mathcal{P}, \mathcal{A}, \mathcal{D}, \mathcal{M}, \mathcal{R})$：

**问题空间 $\mathcal{P}$**：
$$
\mathcal{P} = \left\{ \begin{array}{l}
\text{Problem-Type} \in \{\text{MS}, \text{MF-L2}, \text{MF-L3}, \text{MF-L4}, \text{MF-rect}\} \\
\text{Shape} \in \{(d,d), (m,n)\} \\
\text{Dimension } d \in \{50, 100, 200, 500, 1000, 2000\} \\
\text{Rank-Ratio } r/d \in \{0.01, 0.05, 0.1, 0.2, 0.5, 1.0\} \\
\text{Sampling-Rate } \gamma \in \{0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0\} \\
\text{Noise-Level } \sigma_\epsilon \in \{0, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\} \\
\text{Condition-Number } \kappa \in \{10, 10^2, 10^3, 10^4, 10^5, 10^6\} \\
\text{Measurement-Dist} \in \{\mathcal{N}, \text{Rad}, \text{Sparse}, \text{Sphere}, \text{FJL}\} \\
\text{Target-Spec} \in \{\text{hard-cutoff}, \text{poly-}\alpha, \text{exp-}\beta\}
\end{array} \right\}
$$

**算法空间 $\mathcal{A}$**：
$$
\mathcal{A} = \left\{ \begin{array}{l}
\text{Algorithm} \in \{\text{Muon-Exact}, \text{Muon-RandSVD}, \text{Muon-Trunc}, \\
\quad\quad\quad\quad\quad\quad \text{SGD}, \text{Momentum-SGD}, \text{Adam}, \text{RMSprop}, \text{L-BFGS}\} \\
\text{Learning-Rate } \eta \in [10^{-4}, 10^{-1}] \text{（连续或对数网格）} \\
\text{Weight-Decay } \lambda \in \{0, 10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}\} \\
\text{Momentum } \beta \in \{0.0, 0.5, 0.9, 0.99\} \text{（若适用）}
\end{array} \right\}
$$

**数据与初始化空间 $\mathcal{D}$**：
$$
\mathcal{D} = \left\{ \begin{array}{l}
\text{Seed } s \in \{1, 2, \dots, n_{\max}\} \\
\text{Init-Strategy} \in \{\text{Gaussian}, \text{Orthogonal}, \text{Spectral}, \text{Zero}\} \\
\text{Init-Scale } \sigma_{\text{init}} \in \{10^{-3}, 10^{-2}, 10^{-1}, 1, 10\}/\sqrt{d}
\end{array} \right\}
$$

**度量空间 $\mathcal{M}$**：
$$
\mathcal{M} = \left\{ \begin{array}{l}
K_\epsilon \text{（收敛迭代）}, \quad F_\epsilon \text{（总 FLOPs）}, \quad T_\epsilon \text{（Wall-clock Time）} \\
f_{\text{final}} \text{（最终 loss）}, \quad \delta_{\text{conv}} \in \{0, 1\} \\
\bar{\sigma}_{\log} \text{（平均对数谱比）}, \quad \kappa_{\text{sp}} \text{（谱条件数）} \\
\text{rank}_\epsilon \text{（数值秩）}, \quad \kappa_{\text{cond}} \text{（条件数）} \\
\tau_{\text{iter}} \text{（每迭代时间）}, \quad M_{\text{peak}} \text{（峰值内存）} \\
\theta^{(k)} \text{（梯度-方向夹角）}, \quad \mathcal{A}^{(k)} \text{（Hessian 对齐度）} \\
\kappa_2^{(k)}(H) \text{（Hessian 条件数动态）}
\end{array} \right\}
$$

**响应与推断空间 $\mathcal{R}$**：
$$
\mathcal{R} = \left\{ \begin{array}{l}
\text{假设检验结果（} p \text{ 值，效应量，置信区间）} \\
\text{ANOVA 分解（主效应 + 交互效应）} \\
\text{功效分析（} n_{\min} \text{，实际功效）} \\
\text{响应面模型参数 } \{\beta_j\} \\
\text{排序与推荐（最优算法配置）}
\end{array} \right\}
$$

### 3.4 实验配置总数估算

| 实验组 | 配置数（问题 × 算法 × 参数 × 重复） | 预计运行时间 |
|--------|-------------------------------------|-------------|
| E1–E5 现有 | $\approx 2 \times 2 \times 10 \times 10 = 400$ | 已完成 |
| E6 噪声 | $2 \times 2 \times 3 \times 5 \times 20 = 1,200$ | 12 h |
| E7 秩扫描 | $1 \times 2 \times 3 \times 6 \times 15 = 540$ | 6 h |
| E8 采样率 | $1 \times 2 \times 3 \times 7 \times 15 = 630$ | 8 h |
| E9 权重衰减 | $2 \times 2 \times 2 \times 5 \times 15 = 600$ | 10 h |
| E10 矩形 | $2 \times 2 \times 6 \times 15 = 360$ | 5 h |
| E11 多基线 | $2 \times 6 \times 2 \times 15 = 360$ | 8 h |
| E12 Hessian 动态 | $2 \times 2 \times 2 \times 10 = 80$（时间序列） | 6 h |
| E13 Wall-clock | $1 \times 4 \times 4 \times 10 = 160$ | 4 h |
| E14 随机 SVD | $1 \times 2 \times 12 \times 10 = 240$ | 4 h |
| E15 大规模 | $1 \times 3 \times 3 \times 5 = 45$ | 12 h |
| E16 初始化尺度 | $2 \times 2 \times 2 \times 5 \times 15 = 600$ | 8 h |
| E17 正交初始化 | $2 \times 2 \times 2 \times 4 \times 15 = 480$ | 6 h |
| E18 条件数 | $1 \times 2 \times 1 \times 6 \times 2 \times 15 = 360$ | 8 h |
| E19 分布泛化 | $1 \times 2 \times 2 \times 5 \times 3 \times 15 = 900$ | 10 h |
| E20 功效分析 | 元分析（使用已有数据） | 2 h |
| **总计** | **约 6,055 次运行** | **约 109 h（4.5 天）** |

注：大规模实验（E15）和动态追踪（E12）占据主要时间；其余实验可并行化。

---

## 21. 优先级与资源分配建议

### 4.1 优先级排序（科学价值 vs 资源成本）

使用 **必须做（Must-Have）**、**应该做（Should-Have）**、**探索性（Explore）** 三级分类：

| 优先级 | 实验 | 科学价值 | 资源成本 | 分类理由 |
|--------|------|----------|----------|----------|
| **P0（必须做）** | E11 多基线对比 | 极高 | 中 | 没有 Adam 等基线，无法声称 Muon 优于 "标准" 方法 |
| **P0** | E13 Wall-clock 时间 | 极高 | 低 | 理论 FLOPs 优势必须经实际时间验证 |
| **P0** | E6 噪声敏感性 | 高 | 中 | 实际问题几乎总有噪声；无噪声结论可能严重偏乐观 |
| **P0** | E18 条件数控制 | 高 | 中 | 验证 Muon 的核心价值主张——对谱结构的利用 |
| **P1（应该做）** | E8 过/欠采样 | 高 | 中 | 决定结论是否适用于采样受限的实际场景 |
| **P1** | E7 秩比例扫描 | 高 | 中 | 确定优势的适用范围边界 |
| **P1** | E16 初始化尺度 | 中 | 低 | 成本低，对实际调参指导价值高 |
| **P1** | E20 功效分析 | 高 | 低 | 元分析，成本低，显著提升统计可信度 |
| **P1** | E15 大规模可扩展性 | 高 | 高 | 决定工业部署可行性 |
| **P2（探索性）** | E9 权重衰减 | 中 | 中 | 工业重要，但属于超参数调优层面 |
| **P2** | E10 矩形矩阵 | 中 | 中 | 扩展形状空间，但核心机制可能不变 |
| **P2** | E12 Hessian 动态 | 高 | 高 | 机制理解深刻，但成本高、分析复杂 |
| **P2** | E14 随机 SVD 权衡 | 中 | 中 | 若 E15 显示精确 SVD 不可行，则升级至 P1 |
| **P2** | E17 正交初始化 | 低 | 低 | 成本低，但预期差异可能不大 |
| **P2** | E19 分布泛化 | 中 | 高 | 若 E18 已验证谱结构核心性，分布可能次要 |

### 4.2 推荐执行顺序

**第一阶段（立即执行，约 2 天）**：
1. E20 功效分析（使用现有数据，立即确定是否需要增加重复）
2. E11 多基线对比（建立优化器谱系基准）
3. E13 Wall-clock 时间（验证理论效率）
4. E16 初始化尺度（低成本，高实用价值）

**第二阶段（约 2 天）**：
5. E6 噪声敏感性（核心鲁棒性）
6. E18 条件数控制（核心机制验证）
7. E7 秩比例扫描（范围边界）
8. E8 过/欠采样（采样鲁棒性）

**第三阶段（约 2 天）**：
9. E15 大规模可扩展性（工业可行性）
10. E12 Hessian 动态（机制深化）
11. E14 随机 SVD 权衡（若大规模实验需要）

**第四阶段（可选，约 1 天）**：
12. E9 权重衰减
13. E10 矩形矩阵
14. E17 正交初始化
15. E19 分布泛化

### 4.3 资源预算估算

| 资源类型 | 第一阶段 | 第二阶段 | 第三阶段 | 第四阶段 | 总计 |
|----------|----------|----------|----------|----------|------|
| CPU 核心时 | 24 h | 40 h | 30 h | 15 h | 109 h |
| 峰值内存 | 8 GB | 16 GB | 32 GB | 16 GB | — |
| 存储（原始数据） | 2 GB | 4 GB | 3 GB | 2 GB | 11 GB |
| 人力分析时间 | 8 h | 12 h | 16 h | 8 h | 44 h |

**加速策略**：
- 所有中小规模实验（$d \leq 200$）可完全并行化（约 200 个配置可同时在 20 核上运行）。
- E15（大规模）可限制为 $n=5$ 重复，使用随机 SVD 变体作为默认 Muon 实现。
- E12（Hessian 动态）可降采样至每 100 步而非 50 步记录。
- 若 E20 发现现有 $n=10$ 不足，优先增加 E1–E5 的重复数至 $n=20$，而非增加新实验。

### 4.4 风险与缓解

| 风险 | 可能性 | 影响 | 缓解策略 |
|------|--------|------|----------|
| E11 显示 Adam 优于 Muon | 中 | 极高 | 在论文中明确定位 Muon 的适用场景（低秩、病态、高精度） |
| E13 显示 Muon Wall-clock 慢于 SGD | 高 | 极高 | 强调随机 SVD 变体（E14），重新定位为 "理论启发 + 近似实现" |
| E6 显示噪声消除 Muon 优势 | 中 | 高 | 建立 SNR 阈值理论，限定适用条件 |
| E15 显示 $d=1000$ 不可行 | 低 | 中 | 接受 $d \leq 500$ 为有效范围，工业部署需近似实现 |
| E20 显示 $n=10$ 功效极低 | 中 | 中 | 增加现有实验重复数，重新运行统计检验 |

---

---

# 附录

> 以下附录汇总全文关键符号、实验编码规则和假设检验框架，作为快速参考工具。


## A. 符号速查表

### A.1 基础符号

| 符号 | 含义 | 首次出现 |
|:---:|:---|:---:|
| $d$ | 矩阵维度 | §1.1 |
| $r$ | 矩阵秩（或目标秩） | §1.1 |
| $m$ | 测量样本数 | §1.1 |
| $L$ | 矩阵分解深度 | §2.6.1 |
| $R$ | 实验重复次数 | §3.1.1 |
| $k$ | 迭代索引 | §1.1 |
| $K_\epsilon$ | 达到精度 $\epsilon$ 的迭代复杂度 | §1.2.1 |
| $F_\epsilon$ | 达到精度 $\epsilon$ 的总 FLOPs | §4.1.4 |
| $X^\star$ | 真实/目标矩阵 | §1.1 |
| $X^{(k)}$ | 第 $k$ 次迭代的参数矩阵 | §1.1 |
| $G^{(k)}$ | 第 $k$ 步的梯度矩阵 | §1.3.1 |
| $D^{(k)}$ | 第 $k$ 步的搜索方向 | §1.3.1 |
| $f(X)$ | 目标函数 | §1.1 |
| $f^\star$ | 最优函数值 | §1.1 |
| $\delta_k$ | 函数值差距 $f(X^{(k)}) - f^\star$ | §1.1 |
| $\eta$ | 学习率 / 步长 | §1.3.1 |
| $\lambda$ | 权重衰减系数 | §2.2.3 |

### A.2 范数与谱分析

| 符号 | 含义 | 首次出现 |
|:---:|:---|:---:|
| $\|X\|_2$ | 谱范数（最大奇异值 $\sigma_1$） | §2.1.1 |
| $\|X\|_F$ | Frobenius 范数 | §2.1.2 |
| $\|X\|_*$ | 核范数（奇异值之和） | §2.1.3 |
| $\|X\|_{S_p}$ | Schatten $p$-范数 | §2.1.4 |
| $\sigma_i(X)$ | $X$ 的第 $i$ 个奇异值 | §2.1.1 |
| $\kappa_{\text{cond}}(X)$ | 矩阵谱条件数 $\sigma_1/\sigma_r$ | §2.3.1 |
| $\kappa_{\text{sp}}(X)$ | 谱比 $\|X\|_2/\|X\|_F$ | §2.3.1 |
| $\kappa(f)$ | 函数条件数 $L/\mu$ | §1.3.2 |
| $r_{\text{eff}}(X)$ | 有效秩 $\|X\|_F^2/\|X\|_2^2$ | §2.4.1 |
| $\text{rank}_\epsilon(X)$ | 数值 $\epsilon$-秩 | §2.4.1 |

### A.3 测量与算子

| 符号 | 含义 | 首次出现 |
|:---:|:---|:---:|
| $\mathcal{A}$ | 线性测量算子 | §2.5.1 |
| $\mathcal{A}^*$ | 伴随算子 | §2.5.1 |
| $\delta_r(\mathcal{A})$ | RIP 常数 | §2.5.2 |
| $\mathcal{N}_{\text{spec}}(G)$ | 谱归一化算子 $UV^\top$ | §2.2.2 |
| $\mathcal{S}(G)$ | SVD 归一化算子（同 $\mathcal{N}_{\text{spec}}$） | §7.3.1 |
| $\Pi(W_1, \ldots, W_L)$ | 乘积映射 $W_L \cdots W_1$ | §2.6.1 |

### A.4 统计与检验

| 符号 | 含义 | 首次出现 |
|:---:|:---|:---:|
| $\alpha$ | 显著性水平 | §3.2.3 |
| $p$ | p 值 | §3.2.3 |
| $\beta$ | 第二类错误概率 | §3.4.1 |
| $\text{Power}$ | 统计功效 $1-\beta$ | §3.4.1 |
| $\Delta_\epsilon^{(r)}$ | 性能差异 $K_{\epsilon,r}^{(A)} - K_{\epsilon,r}^{(B)}$ | §3.2.1 |
| $\text{ES}$ | 标准化效应量 | §3.4.1 |
| $d_{\text{Cohen}}$ | Cohen's d | §5.3 |
| $\text{CI}_{1-\alpha}$ | $(1-\alpha)$ 置信区间 | §3.3.1 |
| $\text{FWER}$ | 族错误率 | §3.5.1 |
| $\text{FDR}$ | 错误发现率 | §3.5.3 |

### A.5 实验配置符号

| 符号 | 含义 | 首次出现 |
|:---:|:---|:---:|
| $\mathcal{E}$ | 实验五元组 $(\mathcal{P}, \mathcal{A}, \mathcal{D}, \mathcal{M}, \mathcal{R})$ | §5.1 |
| $\mathcal{P}$ | 问题实例空间 | §5.1 |
| $\mathcal{P}_{MS}$ | 矩阵感知问题空间 | §5.1 |
| $\mathcal{P}_{MF}$ | 矩阵分解问题空间 | §5.1 |
| $\mathcal{A}$ | 算法空间 | §5.1 |
| $\mathcal{D}$ | 数据/超参数空间 | §5.1 |
| $\mathcal{M}$ | 度量空间 | §5.1 |
| $\mathcal{R}$ | 随机性空间 | §5.1 |
| $\rho_K$ | 迭代效率比 $K_\epsilon^{\text{Muon}}/K_\epsilon^{\text{SGD}}$ | §7.3.4 |
| $\rho_F$ | FLOPs 效率比 $F_\epsilon^{\text{Muon}}/F_\epsilon^{\text{SGD}}$ | §7.3.4 |
| $\bar{\sigma}_{\log}$ | 平均对数标准差（稳定性度量） | §7.3.4 |
| $\mathbb{I}_{\text{conv}}$ | 收敛标志 | §12.2 |

### A.6 补充实验专用符号

| 符号 | 含义 | 首次出现 |
|:---:|:---|:---:|
| $\sigma_\epsilon$ | 噪声标准差 | §18. B3 |
| $\gamma = m/d^2$ | 采样率 | §18. B4 |
| $\kappa_{\text{target}}$ | 目标条件数 | §18. B5 |
| $\sigma_{\text{init}}$ | 初始化尺度 | §18. C1 |
| $\delta_{\text{ortho}}$ | 正交性偏差 | §13.3 |
| $\theta^{(k)}$ | 梯度-方向夹角 | §19. E12 |
| $\mathcal{A}^{(k)}$ | Hessian 对齐度 | §19. E12 |
| $\tau_{\text{iter}}$ | 每迭代 Wall-clock 时间 | §19. E13 |
| $r_{\text{approx}}$ | 随机 SVD 近似秩 | §19. E14 |
| $S_L$ | 可扩展性比 | §15.1 |


## B. 实验配置编码方案

### B.1 实验 ID 编码规则

所有实验运行使用统一的标识符编码，便于数据库存储、检索与分析。

**编码格式**：

```
{Problem}_{Shape}_d{dim}_r{rank}_L{depth}_{init}_s{seed}_a{algo}_lr{eta}_wd{lambda}_n{noise}
```

**字段说明**：

| 字段 | 取值示例 | 说明 |
|:-----|:---------|:-----|
| `Problem` | `MS`, `MF` | 问题类型：矩阵感知 / 矩阵分解 |
| `Shape` | `sq`, `rect` | 形状：方阵 / 矩形 |
| `dim` | `100`, `200` | 矩阵维度（方阵时 $d$，矩形时 $m \times n$） |
| `rank` | `10`, `full` | 目标秩（数值或 `full`） |
| `depth` | `2`, `3`, `4` | 分解深度（MF 时有效，MS 时省略） |
| `init` | `gauss`, `orth`, `spect`, `zero`, `bal` | 初始化策略 |
| `seed` | `0`–`49` | 随机种子编号 |
| `algo` | `muon`, `sgd`, `adam`, `rmsprop`, `lbfgs` | 算法标识 |
| `eta` | `1e-3`, `1e-2`, `1e-1` | 学习率 |
| `lambda` | `0`, `1e-5`–`1e-2` | 权重衰减系数 |
| `noise` | `0`, `1e-4`–`1e-1` | 噪声水平 |

**编码示例**：

| 实验描述 | 实验 ID |
|:---------|:--------|
| 矩阵感知，方阵，$d=100$，低秩 $r=10$，高斯初始化，种子 7，Muon，$\eta=0.01$ | `MS_sq_d100_r10_gauss_s7_a_muon_lr1e-2_wd0_n0` |
| 矩阵分解，$L=3$，$d=200$，满秩，平衡初始化，种子 12，SGD，$\eta=0.1$ | `MF_sq_d200_full_L3_bal_s12_a_sgd_lr1e-1_wd0_n0` |
| 补充实验 E6：噪声敏感性，$\sigma_\epsilon=10^{-2}$ | `MS_sq_d100_r10_gauss_s0_a_muon_lr1e-2_wd0_n1e-2` |
| 补充实验 E10：矩形矩阵，$100 \times 200$ | `MS_rect_d100x200_r10_gauss_s0_a_muon_lr1e-2_wd0_n0` |

### B.2 配置字典与响应字典模板

**配置字典（Config）**：

```python
config = {
    # 问题配置
    "problem": "MS",              # MS 或 MF
    "shape": "square",          # square 或 rectangular
    "d": 100,                   # 矩阵维度
    "m": 300,                   # 测量数（MS 时）
    "r": 10,                    # 目标秩
    "L": None,                  # 分解深度（MF 时）

    # 初始化配置
    "init_strategy": "gaussian", # gaussian / orthogonal / spectral / zero / balanced
    "init_scale": 1.0,          # 初始化尺度因子

    # 算法配置
    "algorithm": "Muon",        # Muon / SGD / Adam / RMSprop / L-BFGS
    "eta": 1e-2,                # 学习率
    "lambda_wd": 0.0,           # 权重衰减
    "momentum": 0.0,            # 动量系数（若适用）
    "svd_variant": "exact",     # exact / random / truncated（Muon 时）

    # 实验控制
    "seed": 42,                 # 随机种子
    "T_max": 100000,            # 最大迭代数
    "epsilon": 1e-6,            # 收敛容差
    "batch_size": "full",       # full 或整数

    # 噪声与测量
    "sigma_eps": 0.0,           # 噪声标准差
    "meas_dist": "gaussian",    # gaussian / rademacher / sparse / spherical / fjl
}
```

**响应字典（Response）**：

```python
response = {
    # 核心响应
    "K_eps": 15234,             # 收敛迭代数（未收敛则为 T_max）
    "F_eps": 3.2e9,             # 收敛总 FLOPs
    "I_conv": 1,                # 收敛标志（0/1）
    "f_final": 1e-7,            # 最终目标函数值

    # 资源度量
    "T_wall": 45.2,             # Wall-clock 时间（秒）
    "M_peak": 512.0,            # 峰值内存（MB）
    "tau_iter": 0.003,          # 每迭代平均时间

    # 精度度量
    "grad_norm_final": 1e-5,    # 最终梯度范数
    "f_dist_to_opt": 1e-7,      # 到最优值的距离

    # 协变量
    "kappa_sp_0": 0.32,         # 初始谱比
    "kappa_cond_0": 15.7,       # 初始条件数
    "kappa_cond_star": 10.0,     # 真实条件数
    "r_eps": 18,                # 数值 epsilon-秩
    "mu_SR": 0.85,              # 强凸性参数（MS）

    # 动态追踪（可选，E12）
    "theta_k": [...],           # 梯度-方向夹角序列
    "A_k": [...],               # Hessian 对齐度序列
    "kappa_2_k": [...],         # Hessian 条件数动态
    "sigma_i_k": [...],         # 奇异值轨迹
}
```

### B.3 目录组织规范

```
experiment_data/
├── configs/                    # 所有配置文件（JSON）
│   ├── MS_sq_d100_r10_...
│   └── MF_sq_d200_full_...
├── raw/                        # 原始迭代轨迹
│   ├── MS_sq_d100_r10_.../loss_curve.npy
│   └── MS_sq_d100_r10_.../param_trajectory.npy
├── aggregated/                 # 聚合统计结果
│   ├── summary_by_config.csv
│   └── summary_by_problem.csv
├── statistical_tests/          # 假设检验结果
│   ├── H1_results.json
│   └── H5_results.json
└── figures/                    # 自动生成的图表
    ├── convergence_curves/
    ├── efficiency_heatmaps/
    └── hypothesis_summary/
```


## C. 假设检验汇总表

### C.1 核心假设 H1–H5（现有实验）

| 假设 | 科学问题 | 零假设 $H_0$ | 备择假设 $H_1$ | 检验统计量 | 拒绝域 | 分布 | 效应量 |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **H1** | Muon 无条件收敛更快 | $\mathbb{E}[\Delta_K(p)] \geq 0$ 对至少一个 $p$ | $\mathbb{E}[\Delta_K(p)] < 0$ 对所有 $p$ | $T^{(1)}(p) = \bar{\Delta}_K / (\hat{\sigma}_\Delta / \sqrt{R})$ | $T^{(1)} < -t_{0.05, 9} = -1.833$ | 配对 t（$df=9$） | Cohen's $d$ |
| **H2** | 高谱比时 Muon 优势更显著 | $\bar{\rho}_K^{\text{high}} \geq \bar{\rho}_K^{\text{low}}$ | $\bar{\rho}_K^{\text{high}} < \bar{\rho}_K^{\text{low}}$ | $T^{(2)} = \hat{\gamma} / \sqrt{s_{\text{low}}^2/n_{\text{low}} + s_{\text{high}}^2/n_{\text{high}}}$ | $T^{(2)} > t_{\alpha/2, df^\star}$ | Welch's t | $\hat{\gamma}$ |
| **H3a** | 深度 $L=2 \to 3$ 优势增大 | $\bar{\rho}_K(3) \geq \bar{\rho}_K(2)$ | $\bar{\rho}_K(3) < \bar{\rho}_K(2)$ | $T^{(3a)} = (\bar{\rho}_K(3) - \bar{\rho}_K(2)) / (\hat{\sigma}_{2,3} / \sqrt{n_{\text{pair}}})$ | $T^{(3a)} < -t_{0.05, 39}$ | 配对 t（$df=39$） | $d_{\text{Cohen}}$ |
| **H3b** | 深度 $L=3 \to 4$ 优势增大 | $\bar{\rho}_K(4) \geq \bar{\rho}_K(3)$ | $\bar{\rho}_K(4) < \bar{\rho}_K(3)$ | $T^{(3b)} = (\bar{\rho}_K(4) - \bar{\rho}_K(3)) / (\hat{\sigma}_{3,4} / \sqrt{n_{\text{pair}}})$ | $T^{(3b)} < -t_{0.05, 39}$ | 配对 t（$df=39$） | $d_{\text{Cohen}}$ |
| **H4** | Muon 稳定性更高 | $\mathbb{E}[S(\text{Muon})] \leq \mathbb{E}[S(\text{SGD})]$ | $\mathbb{E}[S(\text{Muon})] > \mathbb{E}[S(\text{SGD})]$ | $T^{(4)} = (\bar{r}_\sigma - 1) / (\hat{\sigma}_{r_\sigma} / \sqrt{R})$ | $T^{(4)} > t_{0.05, 9}$ | 配对 t（$df=9$） | $d_{\text{Cohen}}$ |
| **H5** | Muon 总 FLOPs 效率更高 | $\mathbb{E}[\rho_F] \geq 1$ | $\mathbb{E}[\rho_F] < 1$ | $T^{(5)} = (\bar{\rho}_F - 1) / (\hat{\sigma}_{\rho_F} / \sqrt{R})$ | $T^{(5)} < -t_{0.05, 9} = -1.833$ | 配对 t（$df=9$） | $d_{\text{Cohen}}$ |

### C.2 补充实验 E6–E20

| 编号 | 实验名称 | 核心假设 | 零假设 $H_0$ | 备择假设 $H_1$ | 检验统计量 | 拒绝域 | 检验类型 |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **E6** | 噪声敏感性 | 噪声下 Muon 仍更快 | $\mathbb{E}[\log K_\epsilon^{\text{Muon}}] = \mathbb{E}[\log K_\epsilon^{\text{SGD}}]$ | $\mathbb{E}[\log K_\epsilon^{\text{Muon}}] < \mathbb{E}[\log K_\epsilon^{\text{SGD}}]$ | $T^{(d,\sigma)} = \bar{D} / (S_D / \sqrt{n})$ | $T < -t_{0.95, n-1}$ | 配对对数 t |
| **E7** | 秩比例扫描 | 存在 $r/d$ 甜点区 | 算法$\times(r/d)$ 交互不显著 | 存在 $(r/d)^*$ 使差异最大 | $F_{\text{int}} = \text{MS}_{\text{Alg} \times r/d} / \text{MS}_{\text{Error}}$ | $F > F_{\alpha, df_1, df_2}$ | 双因子 ANOVA |
| **E8** | 过/欠采样 | 采样率调节优势 | $P(\delta_{\text{conv}}^{\text{Muon}}=1) = P(\delta_{\text{conv}}^{\text{SGD}}=1)$ | $P(\delta_{\text{conv}}^{\text{Muon}}=1) > P(\delta_{\text{conv}}^{\text{SGD}}=1)$ | $Z = (\hat{p}_M - \hat{p}_S) / \sqrt{\hat{p}(1-\hat{p})(2/n)}$ | $Z > z_{0.95}$ | 比例 z-检验 |
| **E9** | 权重衰减消融 | $\lambda$ 改变算法相对动态 | $\mathbb{E}[\log K_\epsilon^{\text{Muon}}] = \mathbb{E}[\log K_\epsilon^{\text{SGD}}]$ 对所有 $\lambda$ | 存在 $\lambda^*$ 使差异最大 | $F_{\text{Alg} \times \lambda} = \text{MS}_{\text{Alg} \times \lambda} / \text{MS}_{\text{Error}}$ | $F > F_{\alpha, df_1, df_2}$ | 重复测量 ANOVA |
| **E10** | 矩形矩阵 | 矩形性不消除优势 | $\mathbb{E}[K_\epsilon^{\text{Muon}} - K_\epsilon^{\text{SGD}}] = 0$ | 矩形性不消除（或增强）优势 | $T_\alpha = \bar{D}_\alpha / (S_{D_\alpha} / \sqrt{n})$ | $|T| > t_{0.975, n-1}$ | 配对 t |
| **E11** | 多基线对比 | Muon 优于所有基线 | $\mathbb{E}[\log K_\epsilon^{\text{Muon}}] = \mathbb{E}[\log K_\epsilon^{\text{Algo}_i}]$ | $\mathbb{E}[\log K_\epsilon^{\text{Muon}}] < \mathbb{E}[\log K_\epsilon^{\text{Algo}_i}]$ | $T_i = \bar{D}_i / (S_{D_i} / \sqrt{n})$ | $T_i < -t_{0.05/M, n-1}$ | Holm 校正多重 t |
| **E12** | Hessian 动态 | Muon 方向与 Hessian 更好对齐 | $\mathbb{E}[\theta_{\text{Muon}}^{(k)}] = \mathbb{E}[\theta_{\text{SGD}}^{(k)}]$ | 存在 $k^*$ 使 Muon 对齐更好 | 功能型 ANOVA F-检验 | $F > F_{\alpha, df}$ | 功能型 ANOVA |
| **E13** | Wall-clock 时间 | SVD 开销抵消理论优势 | $T_\epsilon^{\text{Muon-Exact}} = T_\epsilon^{\text{SGD}}$ | $T_\epsilon^{\text{Muon-Exact}} > T_\epsilon^{\text{SGD}}$ | Wilcoxon 符号秩 $W$ | $W < W_{\alpha, n}$ | Wilcoxon |
| **E14** | 随机 SVD 权衡 | 近似 SVD 可保持精度 | $K_\epsilon^{\text{RandomSVD}} = K_\epsilon^{\text{Exact}}$ | 存在最优 $(r^*, q^*)$ 使效率更高 | 配对 t（每参数组合） | $T < -t_{0.95, n-1}$ | 响应面 + t |
| **E15** | 大规模可扩展性 | 存在规模交叉点 $d_{\text{cross}}$ | $T_\epsilon^{\text{Muon}}(d) \leq T_\epsilon^{\text{SGD}}(d)$ 对所有 $d$ | 存在 $d_{\text{cross}}$ 使 Muon 退化 | 对数比率 t-检验 | $\log R_j > t_{0.95, n-1} \cdot \text{SE}$ | 单样本 t |
| **E16** | 初始化尺度 | Muon 具有尺度不变性 | Muon $K_\epsilon$ 不依赖于 $\sigma_{\text{init}}$ | Muon $K_\epsilon$ 依赖于 $\sigma_{\text{init}}$ | 单因子 ANOVA $F$ | $F > F_{\alpha, df}$ | ANOVA |
| **E17** | 正交/谱初始化 | Muon 对初始化敏感度更低 | 算法$\times$初始化 无交互 | Muon 敏感度低于 SGD | $F_{\text{int}} = \text{MS}_{\text{Alg} \times \text{Init}} / \text{MS}_{\text{Error}}$ | $F > F_{\alpha, df}$ | 双因子 ANOVA |
| **E18** | 条件数控制 | 病态放大 Muon 优势 | $\Delta(\kappa) = \text{const}$ 对所有 $\kappa$ | $\Delta(\kappa)$ 随 $\kappa$ 单调增 | Spearman $\rho_S = \text{corr}_{\text{rank}}(\Delta(\kappa), \log \kappa)$ | $\rho_S > \rho_{\text{crit}}$ | Spearman |
| **E19** | 矩阵分布泛化 | 结论不依赖高斯假设 | 测量分布不影响 $\Delta K$ | 存在分布使 $\Delta K$ 显著不同 | Kruskal-Wallis $H$ | $H > \chi^2_{0.95, 4}$ | Kruskal-Wallis |
| **E20** | 功效与样本量 | $n=10$ 不足 | $n=10$ 足以检测 $\delta_{\min}=0.5$ | $n_{\min} > 10$ | 直接计算 $\hat{\delta}$ 与 $\hat{n}_{\min}$ | $\hat{n}_{\min} > 10$ | Bootstrap + 功效 |

### C.3 多重检验校正策略

| 假设组 | 检验数 $M$ | 推荐校正 | 调整后 $\alpha^*$ | 说明 |
|:---|:---:|:---|:---|:---|
| H1–H5（核心假设） | 5 | Holm-Bonferroni | $\alpha / (5 - j + 1)$ | 控制 FWER，功效优于 Bonferroni |
| H3a + H3b（深度子检验） | 2 | Bonferroni | $\alpha / 2 = 0.025$ | 子检验数少，保守校正 |
| E6–E20（探索性补充） | 15 | Benjamini-Hochberg | $p_{(i)} \leq \frac{i}{M} \cdot 0.10$ | 控制 FDR = 0.10，适合探索性分析 |
| E11（多基线） | 5（算法$\times$问题） | Holm 逐步 | $\alpha / (5 - j + 1)$ | 多重比较校正 |
| 全部配置成对比较 | $\gg 40$ | Benjamini-Hochberg | FDR = 0.10 | 大规模探索性扫描 |

### C.4 检验分布速查

| 检验类型 | 统计量 | 零分布 | 自由度 | 适用条件 |
|:---|:---|:---|:---|:---|
| 配对 t-检验 | $T = \bar{D} / (S_D / \sqrt{n})$ | $t_{n-1}$ | $df = n - 1$ | 配对差异近似正态 |
| Welch's t | $T = (\bar{X}_1 - \bar{X}_2) / \sqrt{s_1^2/n_1 + s_2^2/n_2}$ | 近似 t | Welch-Satterthwaite | 两组方差不等 |
| 单因子 ANOVA | $F = \text{MS}_{\text{between}} / \text{MS}_{\text{within}}$ | $F_{k-1, N-k}$ | 组间 $k-1$，组内 $N-k$ | 多组均值比较 |
| 双因子 ANOVA | $F_{\text{int}} = \text{MS}_{AB} / \text{MS}_E$ | $F_{(a-1)(b-1), ab(n-1)}$ | 见交互项 | 交互效应检验 |
| 比例 z-检验 | $Z = (\hat{p}_1 - \hat{p}_2) / \sqrt{\hat{p}(1-\hat{p})(1/n_1 + 1/n_2)}$ | $\mathcal{N}(0,1)$ | $\infty$ | 大样本二项比例 |
| Wilcoxon 符号秩 | $W = \sum \text{rank}(|D_i|) \cdot \mathbb{I}(D_i > 0)$ | 查表 | $n$ | 非正态配对差异 |
| Kruskal-Wallis | $H = \frac{12}{N(N+1)}\sum \frac{R_i^2}{n_i} - 3(N+1)$ | $\chi^2_{k-1}$ | $df = k - 1$ | 多组非参数 |
| Spearman 相关 | $\rho_S = 1 - \frac{6\sum d_i^2}{n(n^2-1)}$ | 查表或近似 t | $df = n - 2$ | 单调关系检验 |
| 功能型 ANOVA | $F = \text{MS}_{\text{Alg} \times \text{Time}} / \text{MS}_E$ | $F$ | 见设计 | 时间序列曲线比较 |

> **附录完**。本文档符号速查表、配置编码方案和假设检验汇总表应与正文配合使用，确保实验设计、执行和报告的一致性与可复现性。


