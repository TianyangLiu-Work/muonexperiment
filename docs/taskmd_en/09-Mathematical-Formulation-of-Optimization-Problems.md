<!--
Original Document: Mathematical Foundations and Statistical Formalization of Muon (μ) Optimization Algorithm Experimental Design
This File: 7. Mathematical Formulation of Optimization Problems
Split Index: 09
-->

[TOC]

---

## 7. Mathematical Formulation of Optimization Problems

### 3.1 Matrix Sensing Problem (MS)

**Definition 3.1 (Matrix Sensing Objective Function).** Given data $(\{A_i\}_{i=1}^m, \{y_i\}_{i=1}^m)$, the objective function of the matrix sensing problem is:

$$
f_{MS}: \mathbb{R}^{d \times d} \to \mathbb{R}_{\geq 0}, \quad f_{MS}(X) = \frac{1}{2m} \sum_{i=1}^m \left( \langle A_i, X \rangle - y_i \right)^2
$$

**Property Analysis:**

- **Smoothness**: $f_{MS} \in \mathcal{C}^\infty(\mathbb{R}^{d \times d})$, infinitely differentiable;
- **Convexity**: $f_{MS}$ is a convex function (as it is composed of quadratic functions of linear functions);
- **Strong Convexity**: $f_{MS}$ is strongly convex when $m \geq d^2$ and $\{A_i\}$ span the full space, with strong convexity modulus:

  ```math
  \mu_{MS} = \frac{1}{m} \lambda_{\min}\left( \sum_{i=1}^m \mathrm{vec}(A_i) \mathrm{vec}(A_i)^T \right)
  ```

- **$L$-smoothness**:

  ```math
  L_{MS} = \frac{1}{m} \lambda_{\max}\left( \sum_{i=1}^m \mathrm{vec}(A_i) \mathrm{vec}(A_i)^T \right)
  ```

- **Condition Number**:

  ```math
  \kappa_{MS} = \frac{L_{MS}}{\mu_{MS}} = \frac{\lambda_{\max}(\mathcal{A}^T \mathcal{A})}{\lambda_{\min}(\mathcal{A}^T \mathcal{A})}
  ```

  where $\mathcal{A} \in \mathbb{R}^{m \times d^2}$ is the matrix representation of the measurement operator, with the $i$-th row being $\mathrm{vec}(A_i)^T$.

**Definition 3.2 (Matrix Sensing Gradient).** The gradient of the objective function $f_{MS}$ is:

$$
\nabla f_{MS}(X) = \frac{1}{m} \sum_{i=1}^m \left( \langle A_i, X \rangle - y_i \right) A_i \in \mathbb{R}^{d \times d}
$$

Or equivalently in vectorized form:

$$
\mathrm{vec}(\nabla f_{MS}(X)) = \frac{1}{m} \mathcal{A}^T (\mathcal{A} \, \mathrm{vec}(X) - y)
$$

where $y = (y_1, \ldots, y_m)^T \in \mathbb{R}^m$.

**Definition 3.3 (Matrix Sensing Hessian Operator).** Since $f_{MS}$ is a quadratic function, its Hessian is a constant matrix:

$$
\nabla^2 f_{MS} = \frac{1}{m} \mathcal{A}^T \mathcal{A} \in \mathbb{R}^{d^2 \times d^2}
$$

Or as a linear operator $H_{MS}: \mathbb{R}^{d \times d} \to \mathbb{R}^{d \times d}$:

$$
H_{MS}[V] = \frac{1}{m} \sum_{i=1}^m \langle A_i, V \rangle A_i
$$

**Definition 3.4 (Optimal Value of Matrix Sensing).** In the noiseless case ($\sigma_\epsilon = 0$), the optimal solution is $X^\star$ and the optimal value is:

$$
f_{MS}^\star = f_{MS}(X^\star) = 0
$$

In the noisy case ($\sigma_\epsilon > 0$), the optimal value is given by the closed-form least-squares solution:

$$
\hat{X}_{LS} = \arg\min_X f_{MS}(X) = \left( \mathcal{A}^T \mathcal{A} \right)^{-1} \mathcal{A}^T y
$$

$$
f_{MS}^\star = f_{MS}(\hat{X}_{LS}) = \frac{1}{2m} y^T \left( I_m - \mathcal{A}(\mathcal{A}^T \mathcal{A})^{-1}\mathcal{A}^T \right) y
$$

---

### 3.2 Matrix Factorization Problem (MF- $`L`$)

**Definition 3.5 (Matrix Factorization Objective Function).** Given the target matrix $X^\star \in \mathbb{R}^{d \times d}$ and factorization depth $L \in \{2, 3, 4\}$, the objective function of the matrix factorization problem is:

$$
f_{MF}: \underbrace{\mathbb{R}^{d \times d} \times \cdots \times \mathbb{R}^{d \times d}}_{L \text{ copies}} \to \mathbb{R}_{\geq 0}
$$

$$
f_{MF}(W_1, \ldots, W_L) = \frac{1}{2} \left\| W_L W_{L-1} \cdots W_1 - X^\star \right\|_F^2
$$

where the Frobenius norm of the matrix product is defined as:

$$
\|M\|_F = \sqrt{\sum_{i,j=1}^d M_{ij}^2} = \sqrt{\mathrm{tr}(M^T M)}
$$

**Property Analysis:**

- **Smoothness**: $f_{MF} \in \mathcal{C}^\infty(\mathbb{R}^{d^2L})$;
- **Convexity**: $f_{MF}$ is **non-convex** (due to the nonlinearity of matrix multiplication);
- **Local Strong Convexity**: Locally strongly convex in a neighborhood of the global optimum (guaranteed by the full-rank assumption of $X^\star$);
- **Saddle Points**: A large number of saddle points exist (especially when $L \geq 3$).

**Definition 3.6 (Partial Gradients of Matrix Factorization).** The partial gradient with respect to each parameter matrix $W_i$ is:

Let the product residual be:

$$
R = W_L \cdots W_1 - X^\star \in \mathbb{R}^{d \times d}
$$

Define the left cumulative product and right cumulative product:

$$
\Pi_{>i} = W_L W_{L-1} \cdots W_{i+1}, \quad \Pi_{<i} = W_{i-1} \cdots W_1
$$

(When $i=L$, $\Pi_{>L} = I_d$; when $i=1$, $\Pi_{<1} = I_d$.)

Then the partial gradient of the $i$-th parameter is:

$$
\frac{\partial f_{MF}}{\partial W_i} = \Pi_{>i}^T \cdot R \cdot \Pi_{<i}^T \in \mathbb{R}^{d \times d}
$$

In particular, when $L=2$:

$$
\frac{\partial f_{MF}}{\partial W_1} = W_2^T (W_2 W_1 - X^\star), \quad \frac{\partial f_{MF}}{\partial W_2} = (W_2 W_1 - X^\star) W_1^T
$$

**Definition 3.7 (Optimal Value of Matrix Factorization).** The global optimal value is:

$$
f_{MF}^\star = 0
$$

Under the assumption that $X^\star$ is full-rank, the optimal solution set is:

$$
\mathcal{X}^\star_{MF} = \left\{ (W_1, \ldots, W_L) \; \middle| \; W_L \cdots W_1 = X^\star \right\}
$$

This is a $d^2(L-1)$-dimensional manifold (with infinitely many optimal solutions).

**Definition 3.8 (Hessian of Matrix Factorization).** Due to the high dimensionality of the parameter space and the non-convexity of the problem, the Hessian is a block matrix of size $Ld^2 \times Ld^2$. At the global optimum, its structure is given by the chain rule. In particular, at $L=2$ and at the equilibrium point, the spectral structure of the Hessian is closely related to the singular values of $X^\star$.

---

### 3.3 Iterative Mapping of Algorithms (Operator Form)

**Definition 3.9 (Unified Algorithm Iteration Operator).** Let $\theta \in \Theta$ be the parameter vector in the parameter space ($\Theta = \mathbb{R}^{d \times d}$ for MS, $\Theta = \mathbb{R}^{d \times dL}$ for MF). The iteration operator for algorithm $\mathcal{A} \in \{\text{Muon}, \text{SGD}\}$ is:

$$
\mathcal{T}_{\eta}^{\mathcal{A}}: \Theta \times \mathcal{G} \to \Theta
$$

where $\mathcal{G}$ is the gradient space. For the deterministic full-gradient setting:

- **Muon:**

  ```math
  \mathcal{T}_{\eta}^{\text{Muon}}(\theta, G) = \theta - \eta \cdot \mathcal{S}(G) - \lambda \cdot \theta
  ```

  where $\mathcal{S}: \mathbb{R}^{d \times d} \to \mathbb{R}^{d \times d}$ is the SVD normalization operator:

  ```math
  \mathcal{S}(G) = U_G V_G^T, \quad G = U_G \Sigma_G V_G^T \text{ (SVD)}
  ```

- **SGD:**

  ```math
  \mathcal{T}_{\eta}^{\text{SGD}}(\theta, G) = \theta - \eta \cdot G - \lambda \cdot \theta
  ```

**Complete Iteration Sequence:** Given initialization $\theta^{(0)}$ and number of iterations $K$, the sequence is defined as:

$$
\theta^{(k+1)} = \mathcal{T}_{\eta}^{\mathcal{A}}(\theta^{(k)}, G^{(k)}), \quad k = 0, 1, \ldots, K-1
$$

where $G^{(k)} = \nabla f(\theta^{(k)})$ is the full gradient (this experiment uses full-batch gradient rather than mini-batch, hence no additional randomness).

---

### 3.4 Mathematical Form of Evaluation Metrics

**Definition 3.10 (Number of Iterations to Reach Target Precision).** Given target precision $\epsilon > 0$ and maximum iterations $K_{\max}$, the number of iterations for algorithm $\mathcal{A}$ to reach precision $\epsilon$ under the $s$-th random seed is:

$$
K_\epsilon^{(\mathcal{A})}(s) = \min \left\{ k \in \{1, \ldots, K_{\max}\} \; \middle| \; f(\theta^{(k)}) - f^\star \leq \epsilon \right\}
$$

If the algorithm does not converge within $K_{\max}$ steps, define:

$$
K_\epsilon^{(\mathcal{A})}(s) = K_{\max} + 1 \quad \text{(truncated value)}
$$

**Definition 3.11 (Per-Step FLOPs Computation).**

- **Per-Step FLOPs for SGD:**

  For MS: gradient computation requires $O(md^2)$, update requires $O(d^2)$, totaling:

  ```math
  C_{\text{SGD}}^{MS} = 2md^2 + d^2 \approx 6d^4 + d^2 \text{ FLOPs}
  ```

  For MF- $`L`$: gradient computation requires $O(Ld^3)$, update requires $O(Ld^2)$, totaling:

  ```math
  C_{\text{SGD}}^{MF} = O(Ld^3) \text{ FLOPs}
  ```

- **Per-Step FLOPs for Muon:**

  In addition to the SGD cost, the SVD cost is added. For the SVD of a $d \times d$ matrix:

  ```math
  C_{\text{SVD}}(d) = O(d^3) \text{ FLOPs}
  ```

  For MS:

  ```math
  C_{\text{Muon}}^{MS} = C_{\text{SGD}}^{MS} + C_{\text{SVD}}(d)
  ```

  For MF- $`L`$ (SVD is performed on each layer matrix):


  ```math
  C_{\text{Muon}}^{MF} = C_{\text{SGD}}^{MF} + L \cdot C_{\text{SVD}}(d)
  ```

**Definition 3.12 (Total FLOPs Consumption).**

$$
F_\epsilon^{(\mathcal{A})}(s) = K_\epsilon^{(\mathcal{A})}(s) \cdot C_{\mathcal{A}}^{\text{problem}} \cdot \mathbb{1}\left[ K_\epsilon^{(\mathcal{A})}(s) \leq K_{\max} \right] + \infty \cdot \mathbb{1}\left[ K_\epsilon^{(\mathcal{A})}(s) > K_{\max} \right]
$$

**Definition 3.13 (Stability Measure of Log-Convergence Trajectory).** For $R$ independent random runs, define the log-error sequence:

$$
\ell_s^{(k)} = \log_{10}\left( f(\theta^{(k)}(s)) - f^\star + 10^{-16} \right), \quad s = 1, \ldots, R
$$

The log-standard deviation at step $k$ is:

$$
\sigma_{\log}^{(k)} = \sqrt{\frac{1}{R-1} \sum_{s=1}^R \left( \ell_s^{(k)} - \bar{\ell}^{(k)} \right)^2}
$$

where $\bar{\ell}^{(k)} = \frac{1}{R} \sum_{s=1}^R \ell_s^{(k)}$. The global stability measure is:

$$
\bar{\sigma}_{\log} = \frac{1}{K_\epsilon} \sum_{k=1}^{K_\epsilon} \sigma_{\log}^{(k)}
$$

**Definition 3.14 (Efficiency Ratio).** For the same problem instance and the same random seed $s$, the efficiency ratio between the two algorithms is:

- **Iteration Efficiency Ratio:**

  ```math
  \rho_K(s) = \frac{K_\epsilon^{\text{Muon}}(s)}{K_\epsilon^{\text{SGD}}(s)}
  ```

- **FLOPs Efficiency Ratio:**

  ```math
  \rho_F(s) = \frac{F_\epsilon^{\text{Muon}}(s)}{F_\epsilon^{\text{SGD}}(s)} = \rho_K(s) \cdot \frac{C_{\text{Muon}}}{C_{\text{SGD}}}
  ```

**Definition 3.15 (Early Stopping Condition).** Define the relative decrease between adjacent iterations:

$$
\delta_k = \frac{f(\theta^{(k)}) - f(\theta^{(k-1)})}{f(\theta^{(k-1)}) - f^\star + 10^{-16}}
$$

The early stopping trigger condition is that $w = 100$ consecutive steps satisfy:

$$
\delta_k > -0.001 \quad \text{(i.e., decrease is less than 0.1\%)}
$$

That is, if for some $k_0$, for all $k \in \{k_0, k_0+1, \ldots, k_0+w-1\}$ we have $\delta_k > -0.001$, then early stopping is triggered at step $k_0 + w - 1$, and $K_\epsilon = k_0 + w - 1$ is recorded (marked as non-convergent).

---

