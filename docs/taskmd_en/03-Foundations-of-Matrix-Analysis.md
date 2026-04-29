<!--
Original Document: Mathematical Foundations and Statistical Formalization for Muon (μ) Optimizer Experimental Design
This File: 2. Foundations of Matrix Analysis
Split Index: 03
-->

[TOC]

---

## 2. Foundations of Matrix Analysis

### 2.1 Matrix Norm System

Let $X \in \mathbb{R}^{d_1 \times d_2}$, $r = \text{rank}(X)$, with singular value decomposition $X = U \cdot \text{diag}(\sigma_1, \ldots, \sigma_r) \cdot V^\top$, where $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$ are the singular values.

#### 2.1.1 Spectral Norm (Operator Norm)

**Definition 2.1** (Spectral Norm). The spectral norm of matrix $X$ is defined as
$$
\|X\|_2 := \sigma_1(X) = \max_{\|v\|_2 = 1} \|Xv\|_2 = \max_{u, v \neq 0} \frac{u^\top X v}{\|u\|_2 \|v\|_2}.
$$
The spectral norm is also the maximum stretching factor when mapping the $\ell_2$ norm unit ball.

**Property**: The spectral norm is a **unitarily invariant norm**, i.e., for any orthogonal matrices $Q_1, Q_2$, we have $\|Q_1 X Q_2^\top\|_2 = \|X\|_2$.

**Experimental Significance**: The spectral norm $\|X^{(k)} - X^\star\|_2$ measures the maximum directional error of the matrix approximate solution. In matrix sensing, the spectral norm is commonly used as an error metric because it corresponds to the worst-case directional deviation.

#### 2.1.2 Frobenius Norm

**Definition 2.2** (Frobenius Norm). The Frobenius norm of matrix $X$ is defined as
$$
\|X\|_F := \sqrt{\sum_{i,j} X_{ij}^2} = \sqrt{\text{tr}(X^\top X)} = \sqrt{\sum_{i=1}^r \sigma_i^2(X)}.
$$

**Property**: The Frobenius norm is equivalent to the $\ell_2$ norm of the vectorized matrix, i.e., $\|X\|_F = \|\text{vec}(X)\|_2$. It is also a unitarily invariant norm, and satisfies the following relationship with the spectral norm:
$$
\|X\|_2 \leq \|X\|_F \leq \sqrt{r} \cdot \|X\|_2 \leq \sqrt{\min(d_1, d_2)} \cdot \|X\|_2.
$$

**Experimental Significance**: The Frobenius norm is the most commonly used error metric in experiments, as it can be interpreted as the sum of squared errors over all elements and directly corresponds to the objective function of matrix factorization.

#### 2.1.3 Nuclear Norm (Trace Norm)

**Definition 2.3** (Nuclear Norm). The nuclear norm of matrix $X$ is defined as the sum of singular values:
$$
\|X\|_* := \sum_{i=1}^r \sigma_i(X).
$$

**Property**: The nuclear norm is the convex envelope of $\|X\|_2$, and is widely used as a convex relaxation of the rank function in low-rank matrix recovery. It satisfies
$$
\|X\|_2 \leq \|X\|_F \leq \|X\|_* \leq \sqrt{r} \cdot \|X\|_F.
$$

**Experimental Significance**: In matrix sensing, nuclear norm minimization is the standard tool for convex relaxation methods. The spectral normalization update direction $D^{(k)} = UV^\top$ of Muon is precisely the extreme point of the nuclear norm unit ball $\{X : \|X\|_* \leq 1\}$, revealing the deep connection between Muon and low-rank induction.

#### 2.1.4 Schatten $p$-Norm

**Definition 2.4** (Schatten $p$-Norm). For $p \in [1, \infty]$, the Schatten $p$-norm of matrix $X$ is defined as
$$
\|X\|_{S_p} := \left(\sum_{i=1}^r \sigma_i^p(X)\right)^{1/p},
$$
with the convention that $\|X\|_{S_\infty} = \sigma_1(X) = \|X\|_2$ and $\|X\|_{S_1} = \|X\|_*$.

**Note**: The Schatten $p$-norms constitute a continuous interpolation family from the spectral norm ($p=\infty$) to the Frobenius norm ($p=2$) to the nuclear norm ($p=1$).

#### 2.1.5 Matrix Inner Product and Orthogonal Projection

**Definition 2.5** (Matrix Inner Product). The standard inner product on the matrix space is
$$
\langle X, Y \rangle := \text{tr}(X^\top Y) = \sum_{i,j} X_{ij} Y_{ij}.
$$
Under this inner product, the Frobenius norm satisfies $\|X\|_F = \sqrt{\langle X, X \rangle}$.

### 2.2 Singular Value Decomposition and Spectral Normalization

#### 2.2.1 Singular Value Decomposition

**Theorem 2.1** (SVD). Any matrix $X \in \mathbb{R}^{d_1 \times d_2}$ (assuming $d_1 \leq d_2$) can be decomposed as
$$
X = U \cdot \Sigma \cdot V^\top = \sum_{i=1}^{d_1} \sigma_i u_i v_i^\top,
$$
where $U \in \mathbb{R}^{d_1 \times d_1}$ and $V \in \mathbb{R}^{d_2 \times d_2}$ are orthogonal matrices, $\Sigma = \text{diag}(\sigma_1, \ldots, \sigma_{d_1}) \in \mathbb{R}^{d_1 \times d_2}$, and $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_{d_1} \geq 0$.

#### 2.2.2 Spectral Normalization

**Definition 2.6** (Spectral Normalization Operator). Given a gradient matrix $G \in \mathbb{R}^{d \times d}$ with SVD $G = U \cdot \text{diag}(\sigma_1, \ldots, \sigma_d) \cdot V^\top$. The spectral normalization direction is defined as
$$
\mathcal{N}_{\text{spec}}(G) := U \cdot V^\top.
$$
If $G$ is not full rank, take the first $r = \text{rank}(G)$ columns of $V$ and the corresponding columns of $U$.

**Properties**:
1. $\|\mathcal{N}_{\text{spec}}(G)\|_2 = 1$ (spectral norm unitization)
2. $\|\mathcal{N}_{\text{spec}}(G)\|_F = \sqrt{r}$, where $r = \text{rank}(G)$
3. $\langle G, \mathcal{N}_{\text{spec}}(G) \rangle = \sum_{i=1}^r \sigma_i = \|G\|_*$

**Theorem 2.2** (Optimality of Spectral Normalization). For any matrix $G$ of rank $r$, we have
$$
\mathcal{N}_{\text{spec}}(G) \in \arg\max_{\|D\|_2 \leq 1} \langle G, D \rangle.
$$
That is, the spectral normalization direction is the direction within the spectral norm unit ball that maximizes the inner product with the gradient direction.

**Proof Sketch**: By von Neumann's trace inequality, $\langle G, D \rangle \leq \sum_i \sigma_i(G) \sigma_i(D) \leq \sum_i \sigma_i(G) \cdot 1 = \|G\|_*$, with equality when $D = UV^\top$.

**Physical Interpretation**: Spectral normalization "flattens" the gradient matrix into an isotropic direction (all singular values become 1), eliminating scale disparities across different spectral directions of the gradient. For gradients with low-rank structure, spectral normalization is particularly effective because it treats all non-zero singular directions equally. This stands in sharp contrast to SGD, which directly uses the raw gradient (weighted by singular values in each direction).

#### 2.2.3 Mathematical Interpretation of Muon Update

The complete update rule of Muon is
$$
X^{(k+1)} = X^{(k)} - \eta \cdot U^{(k)} V^{(k)\top} - \lambda \cdot X^{(k)} = (1 - \lambda) X^{(k)} - \eta \cdot \mathcal{N}_{\text{spec}}(G^{(k)}).
$$
This can be rewritten as
$$
X^{(k+1)} = (1 - \lambda) X^{(k)} - \eta \cdot \frac{G^{(k)}}{\text{``spectral operator weighting''}},
$$
where "spectral operator weighting" is replaced by SVD truncation. Equivalently, Muon performs updates after normalizing the gradient in the spectral domain.

### 2.3 Condition Number and Problem Ill-Conditioning

#### 2.3.1 Multiple Definitions of Condition Number

**Definition 2.7** (Function Condition Number). For an $L$-smooth and $\mu$-strongly convex function $f$, its **condition number** is defined as
$$
\kappa(f) := \frac{L}{\mu} \geq 1.
$$

**Definition 2.8** (Matrix Spectral Condition Number). For a positive definite matrix $H$, its **spectral condition number** is
$$
\kappa_{\text{cond}}(H) := \frac{\lambda_{\max}(H)}{\lambda_{\min}(H)} = \frac{\sigma_1(H)}{\sigma_{\min}(H)}.
$$
For a general matrix $X$, the condition number is defined as the ratio of the largest to the smallest non-zero singular value:
$$
\kappa_{\text{cond}}(X) := \frac{\sigma_1(X)}{\sigma_r(X)}, \quad r = \text{rank}(X).
$$

**Definition 2.9** (Spectral Ratio Used in Experiments). In this experiment, the **spectral ratio** is defined as
$$
\kappa_{\text{sp}}(X) := \frac{\|X\|_2}{\|X\|_F} = \frac{\sigma_1(X)}{\sqrt{\sum_i \sigma_i^2(X)}} \in \left[\frac{1}{\sqrt{r}}, 1\right].
$$

**Physical Interpretation**: The spectral ratio measures the degree of energy concentration in the spectral direction of a matrix. When $\kappa_{\text{sp}}(X) \approx 1$, energy is concentrated in a single direction; when $\kappa_{\text{sp}}(X) \approx 1/\sqrt{r}$, energy is uniformly distributed. Spectral normalization has different effects on gradient matrices with different spectral ratios---for energy-concentrated gradients (high spectral ratio), spectral normalization changes more; for uniformly distributed energy gradients, spectral normalization differs less from SGD.

#### 2.3.2 Impact of Problem Ill-Conditioning on Convergence

**Theorem 2.3** (Condition Number Effect). For the quadratic objective $f(X) = \frac{1}{2}\langle X - X^\star, H (X - X^\star) \rangle$, where $H$ is the Hessian matrix, the convergence ratio of GD is
$$
\rho = \frac{\kappa_{\text{cond}}(H) - 1}{\kappa_{\text{cond}}(H) + 1}.
$$
The larger the condition number, the closer $\rho$ is to 1, and the slower the convergence.

**Experimental Significance**: Properties of the measured matrix in experiments (such as RIP constants) directly affect the effective condition number of the problem. Muon's spectral normalization may reduce the effective condition number by eliminating anisotropy of the gradient in spectral directions, thereby accelerating convergence.

### 2.4 Mathematical Characterization of Low-Rank Matrices

#### 2.4.1 Rank and Numerical Rank

**Definition 2.10** (Matrix Rank). $\text{rank}(X) := \dim(\text{range}(X)) = r$, i.e., the number of non-zero singular values.

**Definition 2.11** (Numerical $\epsilon$-Rank). Given a threshold $\epsilon > 0$, the **numerical $\epsilon$-rank** of matrix $X$ is defined as
$$
\text{rank}_\epsilon(X) := \#\{i : \sigma_i(X) > \epsilon\}.
$$

**Definition 2.12** (Effective Rank). The effective rank of a matrix is defined as
$$
r_{\text{eff}}(X) := \frac{\|X\|_F^2}{\|X\|_2^2} = \frac{\sum_i \sigma_i^2}{\sigma_1^2} = \sum_i \left(\frac{\sigma_i}{\sigma_1}\right)^2 \in [1, r].
$$

**Experimental Significance**: The target $X^\star$ in matrix sensing typically has exact low rank ($r \ll d$), and the product $W_L \cdots W_1$ in matrix factorization also evolves on the low-rank manifold. The effective rank $r_{\text{eff}}$ provides a smoother measure of low-rankness than hard thresholding, suitable for tracking the implicit low-rankness of optimization trajectories.

#### 2.4.2 Low-Rank Matrix Manifold

**Definition 2.13** (Rank-$r$ Matrix Manifold). The set of matrices with rank at most $r$
$$
\mathcal{M}_r := \{X \in \mathbb{R}^{d \times d} : \text{rank}(X) \leq r\}
$$
constitutes an algebraic variety. At points where the rank is exactly $r$, $\mathcal{M}_r$ is a smooth manifold with dimension $r(2d - r)$.

**Definition 2.14** (Tangent Space). At the point $X = U \Sigma V^\top \in \mathcal{M}_r$ ($\Sigma \in \mathbb{R}^{r \times r}$), the tangent space is
$$
\mathcal{T}_X \mathcal{M}_r = \{U A^\top + B V^\top : A \in \mathbb{R}^{d \times r}, B \in \mathbb{R}^{d \times r}\}.
$$

**Experimental Significance**: If the optimization trajectory $X^{(k)}$ always lies on or near $\mathcal{M}_r$, it implicitly exploits the low-rank structure. Muon's spectral normalization may more naturally maintain the trajectory on this manifold, while SGD may deviate from the low-rank structure.

### 2.5 Restricted Isometry Property (RIP) for Matrix Sensing

#### 2.5.1 Linear Measurement Operator

**Definition 2.15** (Measurement Operator). The linear measurement operator $\mathcal{A}: \mathbb{R}^{d \times d} \to \mathbb{R}^m$ in matrix sensing is defined as
$$
[\mathcal{A}(X)]_i := \langle A_i, X \rangle = \text{tr}(A_i^\top X), \quad i = 1, \ldots, m,
$$
where $\{A_i\}_{i=1}^m$ are the measurement matrices. Its adjoint operator $\mathcal{A}^*: \mathbb{R}^m \to \mathbb{R}^{d \times d}$ is
$$
\mathcal{A}^*(y) = \sum_{i=1}^m y_i A_i.
$$

The observation model is $y = \mathcal{A}(X^\star) + \epsilon$, where $\epsilon \in \mathbb{R}^m$ is a noise vector.

#### 2.5.2 RIP Definition

**Definition 2.16** (Restricted Isometry Property, RIP). An operator $\mathcal{A}$ is said to satisfy the **rank-$r$ restricted isometry property** (Rank-$r$ RIP) with constant $\delta_r \in [0, 1)$ if for all matrices $X$ with rank at most $r$:
$$
(1 - \delta_r) \|X\|_F^2 \leq \frac{1}{m}\|\mathcal{A}(X)\|_2^2 \leq (1 + \delta_r) \|X\|_F^2.
$$
Equivalently, the **restricted isometry constant** (RIC) is defined as
$$
\delta_r(\mathcal{A}) := \sup_{\text{rank}(X) \leq r} \left| \frac{1}{m}\|\mathcal{A}(X)\|_2^2 - \|X\|_F^2 \right| / \|X\|_F^2.
$$

**Theorem 2.4** (RIP for Gaussian Measurement Matrices). If each element of $\{A_i\}$ is independently drawn from $\mathcal{N}(0, 1)$ and $m \geq C \cdot r d$ ($C$ is an absolute constant), then with high probability (at least $1 - \exp(-cm)$), $\mathcal{A}$ satisfies rank-$r$ RIP with $\delta_r \leq \delta$ for some pre-specified small constant $\delta$.

**Experimental Significance**: RIP guarantees that the measurement operator approximately preserves the geometric structure of low-rank matrices, making recovery of low-rank matrices from linear measurements possible. In experiments, by controlling the ratio of $m$ (sample size) to $rd$ (low-rank degrees of freedom), the RIP strength of the problem can be adjusted.

#### 2.5.3 RIP and Optimization Landscape

**Theorem 2.5** (Spurious Local Minima-Free Under RIP). If $\mathcal{A}$ satisfies rank-$2r$ RIP and $\delta_{2r} < 1/5$, then for the matrix sensing problem
$$
\min_{X} \frac{1}{2m}\|\mathcal{A}(X) - y\|_2^2 \quad \text{s.t.} \quad \text{rank}(X) \leq r
$$
all local minimum points are global minimum points.

**Experimental Significance**: Under favorable RIP conditions, matrix sensing has no spurious local minima, and any algorithm converging to a stationary point can find the global optimal solution. This provides a fair basis for comparing the convergence rates of Muon and SGD---both will eventually converge to the global optimum, with the difference lying only in the convergence path and speed.

### 2.6 Non-Convexity Analysis of Matrix Factorization

#### 2.6.1 Deep Matrix Factorization Problem

**Definition 2.17** (Deep Matrix Factorization, MF-$L$). Given a target matrix $X^\star \in \mathbb{R}^{d \times d}$ and depth $L \geq 2$, the deep matrix factorization problem is
$$
\min_{W_1, \ldots, W_L} f_{MF}(W_1, \ldots, W_L) := \frac{1}{2}\|W_L W_{L-1} \cdots W_1 - X^\star\|_F^2,
$$
where $W_\ell \in \mathbb{R}^{d \times d}$, $\ell = 1, \ldots, L$.

**Definition 2.18** (Reconstruction Mapping). Define the **product mapping** $\Pi: \mathbb{R}^{d \times d} \times \cdots \times \mathbb{R}^{d \times d} \to \mathbb{R}^{d \times d}$ as
$$
\Pi(W_1, \ldots, W_L) := W_L W_{L-1} \cdots W_1.
$$
The objective function can be written as $f_{MF} = \frac{1}{2}\|\Pi(\mathbf{W}) - X^\star\|_F^2$.

#### 2.6.2 Scale Symmetry and Equivalent Parameterization

**Property** (Scale Symmetry). For any invertible diagonal matrices $\{D_\ell\}_{\ell=1}^{L-1}$, let
$$
\tilde{W}_\ell := W_\ell D_\ell, \quad \tilde{W}_{\ell+1} := D_\ell^{-1} W_{\ell+1},
$$
then $\Pi(\tilde{W}_1, \ldots, \tilde{W}_L) = \Pi(W_1, \ldots, W_L)$. This means that the loss function has a large number of equivalent solutions in the parameter space.

**Definition 2.19** (Balancedness). A parameterization $\mathbf{W} = (W_1, \ldots, W_L)$ is called **balanced** if for all $\ell = 1, \ldots, L-1$:
$$
W_{\ell+1}^\top W_{\ell+1} = W_\ell W_\ell^\top.
$$
In particular, when $L=2$, the balancedness condition is $W_2^\top W_2 = W_1 W_1^\top$.

#### 2.6.3 Optimization Landscape: Saddle Points and Local Minima

**Theorem 2.6** (First-Order Stationary Point Structure for MF-$L$). For $L = 2$, i.e., $f(W_1, W_2) = \frac{1}{2}\|W_2 W_1 - X^\star\|_F^2$:
- All global minima satisfy $W_2 W_1 = X^\star$
- Saddle points exist: if $W_1 = 0$ and $W_2 = 0$, then this is a saddle point (the Hessian has both positive and negative eigenvalues)
- If $X^\star$ is full rank, then all first-order stationary points satisfying the balancedness condition are either global minima or strict saddle points

**Theorem 2.7** (Strict Saddle Property for Deep MF). For $L \geq 2$, under appropriate initialization, gradient descent can escape strict saddle points. If all saddle points are strict saddle points (the Hessian has negative curvature directions), then randomly initialized gradient descent converges to a local minimum with probability 1.

**Definition 2.20** (PL* Condition). In a neighborhood of point $X$, a function $f$ satisfies the $\mu$-PL* condition if
$$
\|\nabla f(X)\|_F^2 \geq \mu \cdot (f(X) - f^\star).
$$
PL* is a localized PL condition, guaranteeing local linear convergence.

**Experimental Significance**: The optimization landscape of MF-$L$ (especially $L \geq 3$) is extremely complex, with a large number of saddle points and scale equivalence classes. Muon's spectral normalization mechanism may improve optimization behavior in the following ways: (1) providing more stable update directions in regions with large gradients; (2) potentially escaping flat regions through spectral truncation effects near saddle points; (3) spectral normalization of layer parameters may implicitly promote balancedness.

---

