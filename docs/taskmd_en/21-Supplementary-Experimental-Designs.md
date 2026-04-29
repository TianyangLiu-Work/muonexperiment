<!--
Original Document: Mathematical Foundations and Statistical Formalization of Muon ($\mu$) Optimization Algorithm Experimental Design
This File: 19. Supplementary Experimental Designs
Split Number: 21
-->

[TOC]

---

## 19. Supplementary Experimental Designs

The following 15 supplementary experiments (numbered E6--E20) systematically cover key gaps across the aforementioned 8 dimensions. Each experiment includes:
- **Experiment Name and Number**
- **Research Question**
- **Mathematical Formulation**: data generation, optimization objective, algorithm variants, parameter settings
- **Statistical Model**: factorial design, response variables, null hypothesis $H_0$ and alternative hypothesis $H_1$, test statistics
- **Relationship to Existing Experiments**
- **Expected Results and Interpretation**

---

### E6: Noise Sensitivity Experiment (Problem Variant B3)

**Research Question**: When the matrix sensing/factorization problem is subject to additive observation noise, how do the convergence rate, stability, and final accuracy of Muon and SGD degrade? Does Muon's spectral normalization possess intrinsic robustness or fragility against noise?

**Mathematical Formulation**:

Data generation process (matrix sensing):
$$
X^\star = U^\star \Sigma^\star (V^\star)^\top, \quad U^\star, V^\star \sim \text{Haar}, \quad \Sigma^\star = \text{diag}(\lambda_1, \dots, \lambda_r)
$$
$$
y_i = \langle A_i, X^\star \rangle + \epsilon_i, \quad A_{ij} \overset{iid}{\sim} \mathcal{N}(0,1), \quad \epsilon_i \overset{iid}{\sim} \mathcal{N}(0, \sigma_\epsilon^2)
$$
where the noise level $\sigma_\epsilon \in \{0, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\}$ (with fixed signal norm $\|X^\star\|_F = 1$).

Matrix factorization (noise injected into the target matrix):
$$
X^\star_{\text{noisy}} = X^\star + \sigma_\epsilon E, \quad E_{ij} \overset{iid}{\sim} \mathcal{N}(0,1)
$$
Optimization objectives:
$$
f_{\text{MS}}^{(\sigma)}(X) = \frac{1}{2m}\sum_{i=1}^m (\langle A_i, X \rangle - y_i)^2, \quad f_{\text{MF}}^{(\sigma)} = \frac{1}{2}\|W_L \cdots W_1 - X^\star_{\text{noisy}}\|_F^2
$$

Parameter settings:
- $d \in \{50, 100, 200\}$ (focusing on medium scale to avoid coupling between noise and dimension)
- $r = d/10$ (low-rank), $m = 3d^2$
- Muon / SGD, $\lambda = 0$, optimal $\eta$ searched independently
- Maximum iterations $K_{\max} = 10^5$, tolerance $\epsilon = 10^{-6}$
- $n = 20$ repetitions per configuration (noise increases variance, requiring more samples)

**Statistical Model**:

Factorial design (full factorial):
$$
\text{Algorithm} \times \text{Problem} \times d \times \sigma_\epsilon \times \text{Seed}
$$
Response variables:
- $K_\epsilon^{(j)}$: number of iterations to reach convergence (right-censored: if not converged, $K_\epsilon = K_{\max}$)
- $F_\epsilon^{(j)} = K_\epsilon^{(j)} \times \text{FLOPs-per-iter}$
- $f_{\text{final}}^{(j)} = f(X^{(K_{\max})})$ (final loss)

Null and alternative hypotheses (for each $(d, \sigma_\epsilon)$ combination):
$$
H_0^{(d,\sigma)}: \mathbb{E}[\log K_\epsilon^{\text{Muon}}] = \mathbb{E}[\log K_\epsilon^{\text{SGD}}] \quad \text{(no difference under noise)}
$$
$$
H_1^{(d,\sigma)}: \mathbb{E}[\log K_\epsilon^{\text{Muon}}] < \mathbb{E}[\log K_\epsilon^{\text{SGD}}] \quad \text{(Muon is faster)}
$$
Test statistic: paired log t-statistic
$$
T^{(d,\sigma)} = \frac{\bar{D} - 0}{S_D / \sqrt{n}}, \quad D_j = \log K_{\epsilon,j}^{\text{Muon}} - \log K_{\epsilon,j}^{\text{SGD}}
$$
where $\bar{D}$ and $S_D$ are the mean and standard deviation of the difference samples. If $T < -t_{0.95, n-1}$, reject $H_0$.

**Relationship to Existing Experiments**: Directly extends E1 (matrix sensing) and E2 (matrix factorization), generalizing noise from $\sigma_\epsilon=0$ to $>0$.

**Expected Results and Interpretation**:
- **Expectation 1**: As $\sigma_\epsilon$ increases, $K_\epsilon$ increases for both algorithms, but Muon's spectral normalization may amplify high-frequency noise components (if noise flattens the gradient spectrum), causing the relative advantage to shrink.
- **Expectation 2**: At $\sigma_\epsilon = 10^{-1}$ (high noise), the difference in $K_\epsilon$ may no longer be significant, suggesting that Muon's advantage is limited to problems with signal-to-noise ratio (SNR) above a certain threshold.
- **Interpretation Framework**: Establish the "$\sigma_\epsilon$--advantage boundary" hypothesis---there exists a critical $\sigma_\epsilon^*(d)$ such that Muon is significantly better than SGD when $\sigma_\epsilon < \sigma_\epsilon^*$, and no significant difference otherwise.

---

### E7: Rank Ratio Sweep Experiment (Problem Variant B1)

**Research Question**: Does Muon's advantage over SGD depend on a specific rank ratio $r/d$? Is there an "$r/d$ sweet spot" where spectral normalization's information compression is most effective?

**Mathematical Formulation**:

Matrix sensing data generation:
$$
r \in \{0.01d, 0.05d, 0.1d, 0.2d, 0.5d, 1.0d\} \quad \Rightarrow \quad r/d \in \{0.01, 0.05, 0.1, 0.2, 0.5, 1.0\}
$$
Construction of $X^\star$:
$$
X^\star = \sum_{i=1}^r \lambda_i u_i v_i^\top, \quad \lambda_i = \lambda_1 \cdot \rho^{i-1}, \quad \rho \in \{0.5, 0.9, 0.99\}
$$
(geometric decay, controlling spectral concentration)

Parameter settings:
- $d \in \{100, 200, 500\}$
- Three spectral decay rates $\rho$
- $m = 3d^2$, $A_{ij} \sim \mathcal{N}(0,1)$
- Muon / SGD, $\lambda=0$, $n=15$ repetitions
- Tolerance $\epsilon = 10^{-8}$ (stricter, to detect the effect of rank on convergence accuracy)

**Statistical Model**:

Factorial design:
$$
\text{Algorithm} \times r/d \times \rho \times d \times \text{Seed}
$$
Response variables: $K_\epsilon$, $\bar{\sigma}_{\log}$, $\kappa_{\text{sp}}$, $\text{rank}_\epsilon(X^{(\infty)})$.

Null and alternative hypotheses:
$$
H_0: \text{Algorithm} \times (r/d) \text{ interaction is not significant on } K_\epsilon
$$
$$
H_1: \exists \; (r/d)^* \text{ such that } \Delta K_\epsilon^{\text{Muon-SGD}} \text{ is maximized at this point}
$$
Test statistic: two-way ANOVA interaction F-statistic
$$
F_{\text{int}} = \frac{\text{MS}_{\text{Alg} \times r/d}}{\text{MS}_{\text{Error}}}, \quad \text{where } \text{MS} \text{ is the mean square}
$$
If $F_{\text{int}} > F_{\alpha, df_1, df_2}$, reject $H_0$ (significant interaction effect exists).

Additional analysis: fit response surfaces for each algorithm separately
$$
\log K_\epsilon^{(\text{Alg})}(r/d, \rho) = \beta_0 + \beta_1(r/d) + \beta_2\rho + \beta_3(r/d)^2 + \beta_4(r/d)\rho + \varepsilon
$$
Compare the response surface differences between the two algorithms.

**Relationship to Existing Experiments**: Extends the existing low-rank ($r=d/10$) vs. full-rank dichotomy into a continuous rank ratio sweep.

**Expected Results and Interpretation**:
- **Expectation 1**: At extremely low rank ($r/d = 0.01$), Muon's advantage is greatest because the gradient spectrum is highly concentrated and $UV^\top$ captures nearly all effective directions.
- **Expectation 2**: At full rank ($r/d = 1.0$), the advantage disappears or reverses, because spectral normalization loses the basis for information compression.
- **Expectation 3**: The spectral decay rate $\rho$ modulates the magnitude of advantage---fast decay (small $\rho$) enhances Muon's advantage, while slow decay (large $\rho$) weakens it.

---

### E8: Oversampling/Undersampling Experiment (Problem Variant B4)

**Research Question**: In matrix sensing, how does the ratio of the number of measurements $m$ to the degrees of freedom $d^2$ affect the convergence behavior of Muon vs. SGD? Under undersampling ($m < d^2$) and critical sampling ($m \approx d^2$), does Muon's spectral information still contribute to convergence acceleration?

**Mathematical Formulation**:

Sampling rate definition:
$$
\gamma = m / d^2 \in \{0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0\}
$$
Sensing operator:
$$
\mathcal{A}: \mathbb{R}^{d \times d} \to \mathbb{R}^m, \quad (\mathcal{A}(X))_i = \langle A_i, X \rangle, \quad A_{ij} \overset{iid}{\sim} \mathcal{N}(0, 1/d^2)
$$
(note the normalization: $\mathbb{E}[\|\mathcal{A}(X)\|^2] = \|X\|_F^2$)

Optimization objective:
$$
f_{\text{MS}}^{(\gamma)}(X) = \frac{1}{2m}\|\mathcal{A}(X) - y\|_2^2
$$

Parameter settings:
- $d \in \{50, 100, 200\}$
- $r = d/10$
- $\gamma \in \{0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0\}$
- Muon / SGD, $\lambda=0$, $n=15$ repetitions
- Tolerance $\epsilon = 10^{-6}$, maximum iterations $K_{\max} = 2 \times 10^5$

**Statistical Model**:

Factorial design:
$$
\text{Algorithm} \times \gamma \times d \times \text{Seed}
$$
Response variables:
- $K_\epsilon$ (convergence iterations)
- $\hat{\kappa}_{\text{cond}}(\mathcal{A}^*\mathcal{A})$: estimated condition number of the sensing operator (approximation of the Hessian)
- Convergence indicator $\delta_{\text{conv}} \in \{0, 1\}$ (binary response; may not converge under undersampling)

Null and alternative hypotheses:
$$
H_0^{(\gamma)}: P(\delta_{\text{conv}}^{\text{Muon}} = 1) = P(\delta_{\text{conv}}^{\text{SGD}} = 1) \quad \text{(equal convergence probability given } \gamma \text{)}
$$
$$
H_1^{(\gamma)}: P(\delta_{\text{conv}}^{\text{Muon}} = 1) > P(\delta_{\text{conv}}^{\text{SGD}} = 1)
$$
Test statistic: two-sample proportion z-test
$$
Z = \frac{\hat{p}_{\text{Muon}} - \hat{p}_{\text{SGD}}}{\sqrt{\hat{p}(1-\hat{p})(2/n)}}, \quad \hat{p} = \frac{\hat{p}_{\text{Muon}} + \hat{p}_{\text{SGD}}}{2}
$$

For $K_\epsilon$ (computed only on the converged subsample):
$$
T^{(\gamma)} = \frac{\bar{D}}{S_D / \sqrt{n_{\text{conv}}}}, \quad D_j = \log K_{\epsilon,j}^{\text{Muon}} - \log K_{\epsilon,j}^{\text{SGD}}
$$

**Relationship to Existing Experiments**: Extends the fixed $m=3d^2$ ($\gamma=3$) in E1 to a full-range $\gamma$ sweep.

**Expected Results and Interpretation**:
- **Expectation 1**: Under undersampling ($\gamma < 1$), both algorithms may fail to converge or converge extremely slowly, but Muon's spectral directions may provide more stable gradient estimates (prior on low-rank structure).
- **Expectation 2**: Near critical sampling ($\gamma \approx 1$), Muon's advantage is greatest because the problem is just barely solvable and the utilization of spectral information is most critical.
- **Expectation 3**: Under oversampling ($\gamma \geq 3$), both algorithms converge, but Muon's advantage is stable (consistent with existing experiments).
- **Interpretation Framework**: Establish the "sampling rate--advantage" curve $\Delta(\gamma) = \mathbb{E}[\log K_\epsilon^{\text{SGD}} - \log K_\epsilon^{\text{Muon}} \mid \text{converged}]$.

---

### E9: Weight Decay Ablation Experiment (Hyperparameter D2 / Algorithm A3)

**Research Question**: How does weight decay ($L_2$ regularization) alter the convergence dynamics of Muon and SGD? Is there a synergistic or antagonistic interaction between Muon's spectral normalization direction and the shrinkage effect of weight decay?

**Mathematical Formulation**:

Algorithm update rules (with weight decay):
$$
\text{Muon:} \quad X^{(k+1)} = X^{(k)} - \eta D^{(k)} - \lambda X^{(k)} = (1-\lambda)X^{(k)} - \eta \cdot UV^\top
$$
$$
\text{SGD:} \quad X^{(k+1)} = X^{(k)} - \eta G^{(k)} - \lambda X^{(k)} = (1-\lambda)X^{(k)} - \eta G^{(k)}
$$

Parameter settings:
- $\lambda \in \{0, 10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}\}$
- $\eta$ searched independently via grid search on each $(\lambda, \text{Algorithm})$ combination ($\eta \in \{10^{-3}, 10^{-2}, 10^{-1}\}$)
- Problems: MS ($d=100, 200$) and MF-L2 ($d=100, 200$)
- $r = d/10$, $m = 3d^2$
- $n = 15$ repetitions
- Tolerance $\epsilon = 10^{-6}$

**Statistical Model**:

Factorial design:
$$
\text{Algorithm} \times \lambda \times d \times \text{Problem} \times \text{Seed}
$$
Response variables: $K_\epsilon$, $F_\epsilon$ (total FLOPs), $\|X^{(K)}\|_F$ (final parameter norm).

Null and alternative hypotheses:
$$
H_0^{(\lambda)}: \text{Given } \lambda, \; \mathbb{E}[\log K_\epsilon^{\text{Muon}}] = \mathbb{E}[\log K_\epsilon^{\text{SGD}}]
$$
$$
H_1^{(\lambda)}: \exists \; \lambda^* \text{ such that } \Delta(\lambda^*) = \max_{\lambda} \left|\mathbb{E}[\log K_\epsilon^{\text{Muon}} - \log K_\epsilon^{\text{SGD}}]\right|
$$
Test statistic: paired t-test for each $\lambda$ separately; overall repeated-measures ANOVA:
$$
F_{\text{Alg} \times \lambda} = \frac{\text{MS}_{\text{Alg} \times \lambda}}{\text{MS}_{\text{Error}}}
$$

**Relationship to Existing Experiments**: Fills the gap of $\lambda = 0$ in existing experiments, directly controlling weight decay---a key hyperparameter in industrial-scale optimization.

**Expected Results and Interpretation**:
- **Expectation 1**: When $\lambda > 0$, the effective convergence rate of both algorithms is affected by $\lambda$, but Muon's spectral direction may respond differently to the shrinkage term (because $UV^\top$ and $X^{(k)}$ have related spectral structure).
- **Expectation 2**: There exist optimal $\lambda_{\text{Muon}}^*$ and $\lambda_{\text{SGD}}^*$, which may differ---if Muon implicitly shrinks (spectral normalization itself suppresses large singular values), then it is less sensitive to external $\lambda$.
- **Expectation 3**: On MF problems, weight decay may alter the stability of balanced initialization, and Muon's depth advantage may therefore change.

---

### E10: Rectangular Matrix Experiment (Problem Variant B2)

**Research Question**: In non-square (rectangular) matrix sensing and rectangular factorization, does Muon's spectral normalization (based on SVD of $G \in \mathbb{R}^{m \times n}$) maintain its advantage over SGD? Does rectangularity (aspect ratio $\alpha = m/n$) modulate the magnitude of advantage?

**Mathematical Formulation**:

Rectangular matrix sensing:
$$
X^\star \in \mathbb{R}^{m \times n}, \quad m \neq n, \quad \text{rank}(X^\star) = r
$$
$$
y_i = \langle A_i, X^\star \rangle, \quad A_i \in \mathbb{R}^{m \times n}, \quad A_{ij} \overset{iid}{\sim} \mathcal{N}(0, 1)
$$
Number of measurements $m_{\text{meas}} = 3mn$ (maintaining a similar oversampling rate to the square case).

Rectangular matrix factorization:
$$
W_1 \in \mathbb{R}^{d_1 \times d_0}, \; W_2 \in \mathbb{R}^{d_2 \times d_1}, \; \dots, \; W_L \in \mathbb{R}^{d_L \times d_{L-1}}, \quad d_L = m, \; d_0 = n
$$
$$
f_{\text{MF-rect}} = \frac{1}{2}\|W_L W_{L-1} \cdots W_1 - X^\star\|_F^2
$$

Parameter settings:
- Shape parameters: $(m, n) \in \{(50, 100), (100, 50), (100, 200), (200, 100), (200, 500), (500, 200)\}$
- Aspect ratio $\alpha = m/n \in \{0.5, 2.0\}$
- Rank $r = \min(m, n) / 10$
- Muon / SGD, $\lambda=0$, $n=15$ repetitions
- Tolerance $\epsilon = 10^{-6}$

**Statistical Model**:

Factorial design:
$$
\text{Algorithm} \times \alpha \times \min(m,n) \times \text{Problem} \times \text{Seed}
$$
Response variables: $K_\epsilon$, $F_\epsilon$, $\bar{\sigma}_{\log}(G)$ (singular value distribution of the rectangular gradient matrix).

Null and alternative hypotheses:
$$
H_0: \text{For rectangular matrices, } \mathbb{E}[K_\epsilon^{\text{Muon}} - K_\epsilon^{\text{SGD}}] = 0
$$
$$
H_1: \text{Rectangularity does not eliminate Muon's advantage (or enhances it)}
$$
Test statistic: paired t-test, grouped by $\alpha$; testing the moderating effect of aspect ratio:
$$
T_{\alpha} = \frac{\bar{D}_{\alpha} - 0}{S_{D_\alpha} / \sqrt{n}}
$$

**Relationship to Existing Experiments**: Extends square matrix experiments to rectangular ones, verifying the generalizability of conclusions across matrix shapes.

**Expected Results and Interpretation**:
- **Expectation 1**: Muon's advantage persists in rectangular settings, because SVD is well-defined for arbitrary $m \times n$ matrices.
- **Expectation 2**: When $\alpha \ll 1$ (extremely flat) or $\alpha \gg 1$ (extremely tall), the computational cost of SVD $O(mn \cdot \min(m,n))$ may alter the FLOPs-efficiency comparison.
- **Expectation 3**: The depth effect (H3) of rectangular factorization may differ from the square case---dimension mismatch may introduce additional rank collapse patterns.

---

### E11: Comparison with Adam / RMSprop / AdaGrad Experiment (Algorithm Baseline A6)

**Research Question**: Does Muon outperform only SGD, or also industrial-standard adaptive methods (Adam, RMSprop, AdaGrad)? Are the element-wise gains of adaptive methods competitive with or complementary to Muon's full-spectrum normalization?

**Mathematical Formulation**:

Baseline algorithms:
- **Adam**: $m^{(k)} = \beta_1 m^{(k-1)} + (1-\beta_1) G^{(k)}$, $v^{(k)} = \beta_2 v^{(k-1)} + (1-\beta_2) (G^{(k)})^2$, update $X^{(k+1)} = X^{(k)} - \eta \frac{m^{(k)}/(1-\beta_1^k)}{\sqrt{v^{(k)}/(1-\beta_2^k)} + \epsilon}$
- **RMSprop**: $v^{(k)} = \beta v^{(k-1)} + (1-\beta) (G^{(k)})^2$, $X^{(k+1)} = X^{(k)} - \eta \frac{G^{(k)}}{\sqrt{v^{(k)}} + \epsilon}$
- **AdaGrad**: $V^{(k)} = V^{(k-1)} + (G^{(k)})^2$, $X^{(k+1)} = X^{(k)} - \eta \frac{G^{(k)}}{\sqrt{V^{(k)}} + \epsilon}$
- **Momentum SGD**: $M^{(k)} = \beta M^{(k-1)} + G^{(k)}$, $X^{(k+1)} = X^{(k)} - \eta M^{(k)}$
- **L-BFGS** (second-order, as reference): implemented via scipy.optimize, memory length $m=10$

Learning rates for all adaptive methods searched independently ($\eta \in \{10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\}$).

Parameter settings:
- Problems: MS ($d=100, 200$) and MF-L2 ($d=100, 200$)
- $r = d/10$, $m = 3d^2$
- Algorithm set: Muon, SGD, Adam ($\beta_1=0.9, \beta_2=0.999$), RMSprop ($\beta=0.99$), Momentum SGD ($\beta=0.9$)
- $n = 15$ repetitions
- Tolerance $\epsilon = 10^{-6}$, maximum iterations $K_{\max} = 10^5$

**Statistical Model**:

Factorial design:
$$
\text{Algorithm} \times \text{Problem} \times d \times \text{Seed}
$$
Response variables: $K_\epsilon$, $F_\epsilon$ (FLOPs count for Adam and other adaptive methods requires additional accounting for first/second moment accumulation).

Null and alternative hypotheses (multiple comparison framework):
$$
H_0^{(i)}: \mathbb{E}[\log K_\epsilon^{\text{Muon}}] = \mathbb{E}[\log K_\epsilon^{\text{Algo}_i}], \quad i \in \{\text{Adam, RMSprop, Momentum, SGD}\}
$$
$$
H_1^{(i)}: \mathbb{E}[\log K_\epsilon^{\text{Muon}}] < \mathbb{E}[\log K_\epsilon^{\text{Algo}_i}]
$$
Multiple testing correction: use the Holm method to control FWER at $\alpha=0.05$.

Test statistic:
$$
T_i = \frac{\bar{D}_i}{S_{D_i} / \sqrt{n}}, \quad D_{i,j} = \log K_{\epsilon,j}^{\text{Muon}} - \log K_{\epsilon,j}^{\text{Algo}_i}
$$

**Relationship to Existing Experiments**: Extends the baseline from a single SGD to a complete optimizer benchmark suite, establishing Muon's true position in the optimizer landscape.

**Expected Results and Interpretation**:
- **Expectation 1**: Adam may approach Muon at moderate precision ($\epsilon \sim 10^{-3}$) but be surpassed by Muon at high precision ($\epsilon \sim 10^{-6}$), due to numerical noise accumulation from adaptive gains.
- **Expectation 2**: Momentum SGD may narrow the gap with Muon (because momentum provides partial directional smoothing), but cannot fully close the gap (because spectral normalization provides additional scale invariance).
- **Expectation 3**: L-BFGS may be optimal at small-to-medium scale ($d \leq 200$), validating Muon's positioning as a "quasi-second-order but first-order cost" method.

---

### E12: Hessian Spectrum Dynamics Tracking Experiment (Dynamic Behavior G1/G4)

**Research Question**: How does the spectral structure of the Hessian matrix evolve during optimization? Does Muon's spectral normalization direction align with Hessian eigenvectors? How does this alignment change across iterations?

**Mathematical Formulation**:

For the MS problem, the Hessian is:
$$
\nabla^2 f_{\text{MS}}(X) = \frac{1}{m}\sum_{i=1}^m \text{vec}(A_i) \text{vec}(A_i)^\top \in \mathbb{R}^{d^2 \times d^2}
$$
Eigenvalues $\lambda_1(H) \geq \lambda_2(H) \geq \dots \geq \lambda_{d^2}(H)$.

For the MF problem (taking $L=2$ as an example), $W = W_2 W_1$, Hessian block structure near equilibrium points (computed via automatic differentiation).

Tracking metrics:
1. **Condition number dynamics**: $\kappa_2^{(k)} = \lambda_{\max}(H^{(k)}) / \lambda_{\min}(H^{(k)})$
2. **Spectral normalization direction-Hessian alignment**:
   $$
   \mathcal{A}_{\text{Muon}}^{(k)} = \frac{\|H^{(k)} \text{vec}(D^{(k)})\|_2}{\lambda_{\max}(H^{(k)}) \|\text{vec}(D^{(k)})\|_2}, \quad \mathcal{A}_{\text{SGD}}^{(k)} = \frac{\|H^{(k)} \text{vec}(G^{(k)})\|_2}{\lambda_{\max}(H^{(k)}) \|\text{vec}(G^{(k)})\|_2}
   $$
   This quantity measures the alignment of the update direction with the maximum curvature direction of the Hessian.
3. **Gradient-direction angle**:
   $$
   \theta^{(k)} = \arccos\left(\frac{\langle G^{(k)}, D^{(k)}\rangle}{\|G^{(k)}\|_F \|D^{(k)}\|_F}\right)
   $$

Parameter settings:
- Problems: MS ($d=50, 100$) and MF-L2 ($d=50, 100$)
- $r = d/10$, $m = 3d^2$
- Record Hessian eigenvalues every $T = 50$ steps (approximate the top 20 eigenvalues via Lanczos iteration to avoid $O(d^4)$ exact computation)
- Muon / SGD, $\lambda=0$
- $n = 10$ repetitions (dynamic tracking generates large data volumes, so reduce repetitions)
- Maximum iterations $K_{\max} = 5000$

**Statistical Model**:

Time series analysis framework: treat $\kappa_2^{(k)}$, $\mathcal{A}^{(k)}$, $\theta^{(k)}$ as discrete-time stochastic processes.

Null and alternative hypotheses:
$$
H_0: \mathbb{E}[\theta^{(k)}_{\text{Muon}}] = \mathbb{E}[\theta^{(k)}_{\text{SGD}}] \quad \forall k \in \{1, \dots, K/T\}
$$
$$
H_1: \exists \; k^* \text{ such that } \mathbb{E}[\theta^{(k^*)}_{\text{Muon}}] < \mathbb{E}[\theta^{(k^*)}_{\text{SGD}}] \text{ and } \mathcal{A}^{(k^*)}_{\text{Muon}} > \mathcal{A}^{(k^*)}_{\text{SGD}}
$$
(Muon's direction is closer to the gradient and better aligned with the Hessian)

Test statistics:
- Functional ANOVA on the angle sequence or a repeated-measures mixed-effects model:
  $$
  \theta_{ij}^{(k)} = \mu + \alpha_i^{\text{Alg}} + \beta_j^{\text{Seed}} + \gamma_k^{\text{Time}} + (\alpha\gamma)_{ik}^{\text{Alg}\times\text{Time}} + \varepsilon_{ijk}
  $$
  where $(\alpha\gamma)_{ik}$ is the key interaction term---the different evolutionary patterns of algorithms over time.
- F-test: $H_0: (\alpha\gamma)_{ik} = 0 \; \forall i, k$.

**Relationship to Existing Experiments**: Entirely new dynamic analysis dimension, providing a temporal evolution version of existing static spectral metrics ($\bar{\sigma}_{\log}$).

**Expected Results and Interpretation**:
- **Expectation 1**: SGD's $\theta^{(k)} = 0$ always holds (because $D^{(k)} = G^{(k)}$), while Muon's $\theta^{(k)} > 0$ but gradually decreases (the spectral normalization direction aligns with the gradient direction in high-dimensional space).
- **Expectation 2**: In the early stages of convergence, Muon's $\mathcal{A}^{(k)}$ may be significantly higher than SGD's, indicating that the spectral normalization direction better exploits Hessian structure; the difference narrows as convergence is approached.
- **Expectation 3**: The Hessian condition number $\kappa_2^{(k)}$ may first increase and then decrease during iterations (the "saddle point inflation" effect), and Muon may traverse high-condition-number regions more quickly.

---

### E13: Wall-Clock Time Measurement Experiment (Computational Efficiency E1/E2)

**Research Question**: Does Muon's theoretical per-iteration FLOPs advantage translate into wall-clock time advantage on actual hardware? How does SVD overhead affect practical efficiency across different hardware (CPU vs. GPU) and different implementations (dense SVD vs. randomized SVD)?

**Mathematical Formulation**:

Timing metric definitions:
- **Per-iteration time**: $\tau_{\text{iter}}^{(\text{Alg})}$ (seconds/iteration)
- **Total time**: $T_\epsilon^{(\text{Alg})} = K_\epsilon^{(\text{Alg})} \times \tau_{\text{iter}}^{(\text{Alg})}$
- **Time-accuracy curve to reach precision $\epsilon$**: $\{(\tau_{\text{cum}}^{(k)}, f(X^{(k)}))\}_{k=0}^{K_\epsilon}$

Implementation variants:
- **Muon-Exact**: `numpy.linalg.svd()` (Golub-Reinsch, $O(d^3)$)
- **Muon-RandomSVD**: Halko algorithm ($O(d^2 r_{\text{approx}})$, $r_{\text{approx}} = 2r$)
- **Muon-Truncated**: `scipy.sparse.linalg.svds()` (computes only the top $r_{\max}$ singular values)
- **SGD**: standard matrix operations ($O(d^2)$ per iteration)

Parameter settings:
- Problem: MS ($d=50, 100, 200, 500$)
- $r = d/10$, $m = 3d^2$
- Algorithms: Muon-Exact, Muon-RandomSVD, Muon-Truncated, SGD
- $n=10$ repetitions per configuration (take median to eliminate outliers)
- Hardware: standard CPU (single/multi-core) and optional GPU
- Tolerance $\epsilon = 10^{-6}$

**Statistical Model**:

Factorial design:
$$
\text{Algorithm-Variant} \times d \times \text{Hardware} \times \text{Seed}
$$
Response variables: $\tau_{\text{iter}}$ (seconds), $T_\epsilon$ (total time), peak memory $M_{\text{peak}}$ (MB).

Null and alternative hypotheses:
$$
H_0^{(d)}: T_\epsilon^{\text{Muon-Exact}} = T_\epsilon^{\text{SGD}} \quad \text{(no difference in actual time)}
$$
$$
H_1^{(d)}: T_\epsilon^{\text{Muon-Exact}} > T_\epsilon^{\text{SGD}} \quad \text{(SVD overhead cancels iteration advantage)}
$$
Additional hypotheses:
$$
H_0^{\text{Rand}}: T_\epsilon^{\text{Muon-RandomSVD}} = T_\epsilon^{\text{Muon-Exact}}
$$
$$
H_1^{\text{Rand}}: T_\epsilon^{\text{Muon-RandomSVD}} < T_\epsilon^{\text{Muon-Exact}} \quad \text{(randomized SVD acceleration is effective)}
$$

Test statistic: paired Wilcoxon signed-rank test (timing data may be skewed):
$$
W = \sum_{j=1}^n \text{rank}(|D_j|) \cdot \mathbb{I}(D_j > 0), \quad D_j = T_{\epsilon,j}^{\text{Alg}_1} - T_{\epsilon,j}^{\text{Alg}_2}
$$

**Relationship to Existing Experiments**: Replaces the theoretical FLOPs count in existing experiments with actual wall-clock time measurements; a critical validation of existing computational efficiency analysis.

**Expected Results and Interpretation**:
- **Expectation 1**: For $d \leq 100$, Muon-Exact's $T_\epsilon$ may exceed SGD's (high SVD constant factor); for $d \geq 500$, Muon's iteration advantage may dominate SVD overhead.
- **Expectation 2**: Muon-RandomSVD is significantly faster than Muon-Exact at large $d$, but may sacrifice accuracy (requires quantification of the accuracy-speed trade-off).
- **Expectation 3**: In terms of memory footprint, Muon needs to store the gradient and perform SVD (temporary $O(d^3)$), while SGD only needs $O(d^2)$. There exists some $d_{\text{mem}}$ beyond which Muon degrades due to memory pressure.

---

### E14: Randomized SVD Approximation Accuracy-Efficiency Trade-off Experiment (Computational Efficiency E3)

**Research Question**: When using randomized SVD approximation as a substitute for exact SVD, how do Muon's convergence accuracy and speed trade off? What is the relationship between the approximation error threshold $\delta_{\text{svd}}$ and the convergence tolerance $\epsilon$?

**Mathematical Formulation**:

Randomized SVD (Halko et al.) parameterization:
$$
\text{RandomSVD}(G, r_{\text{approx}}, q, p): \quad \tilde{U}\tilde{\Sigma}\tilde{V}^\top \approx G
$$
where $r_{\text{approx}}$ is the approximation rank, $q$ is the number of power iterations, and $p$ is the oversampling parameter.

Approximate direction:
$$
\tilde{D}^{(k)} = \tilde{U}\tilde{V}^\top \quad \text{or} \quad \tilde{D}^{(k)} = \tilde{U}\tilde{\Sigma}^{-1}\tilde{V}^\top
$$

Approximation error metric:
$$
\delta_{\text{svd}} = \|G - \tilde{U}\tilde{\Sigma}\tilde{V}^\top\|_F / \|G\|_F
$$

Parameter settings:
- Problem: MS ($d=200, 500$)
- $r = d/10$
- Randomized SVD parameter combinations:
  - $r_{\text{approx}} \in \{r, 2r, 5r, 10r\}$
  - $q \in \{0, 1, 2\}$
  - $p \in \{5, 10, 20\}$
- Muon-Exact as baseline
- $n = 10$ repetitions
- Tolerance $\epsilon = 10^{-6}$

**Statistical Model**:

Factorial design:
$$
\text{SVD-Variant} \times r_{\text{approx}}/r \times q \times d \times \text{Seed}
$$
Response variables:
- $K_\epsilon$ (convergence iterations; may not converge if approximation is too coarse)
- $\tau_{\text{iter}}$ (per-iteration time)
- $\delta_{\text{svd}}$ (per-step approximation error)
- Composite efficiency: $E = K_\epsilon \times \tau_{\text{iter}}$

Null and alternative hypotheses:
$$
H_0^{(r_{\text{approx}}, q)}: K_\epsilon^{\text{RandomSVD}(r_{\text{approx}}, q)} = K_\epsilon^{\text{Exact}}
$$
$$
H_1^{(r_{\text{approx}}, q)}: \exists \; (r_{\text{approx}}^*, q^*) \text{ such that } E^{\text{RandomSVD}} < E^{\text{Exact}} \text{ and } K_\epsilon^{\text{RandomSVD}} \approx K_\epsilon^{\text{Exact}}
$$
Test statistic: paired t-test for each parameter combination; use response surface methodology to find the optimal $(r_{\text{approx}}^*, q^*)$.

**Relationship to Existing Experiments**: Extends E13's wall-clock analysis, specifically focusing on SVD approximation strategies.

**Expected Results and Interpretation**:
- **Expectation 1**: There exists a "good enough approximation" threshold---when $r_{\text{approx}} \geq 2r$ and $q \geq 1$, $K_\epsilon$ is almost identical to exact SVD, but $\tau_{\text{iter}}$ is reduced by 2--5x.
- **Expectation 2**: Excessively low $r_{\text{approx}}$ (e.g., $r_{\text{approx}} = r$) may lead to degraded direction quality and increased $K_\epsilon$, offsetting the time gains.
- **Expectation 3**: Diminishing marginal returns for power iterations $q$---$q=1$ is usually sufficient; $q=2$ offers limited improvement but at significantly increased cost.

---

### E15: Larger-Scale Scalability Experiment (Computational Efficiency E4)

**Research Question**: As dimension $d$ scales from 500 to 1000 and 2000, does Muon's $O(d^3)$ SVD overhead constitute a hard wall? At what scale is any iteration advantage of Muon completely consumed by SVD cost?

**Mathematical Formulation**:

Scale sequence:
$$
d \in \{500, 1000, 2000\}
$$
Corresponding problem sizes:
- MS: $m = 3d^2 \in \{7.5 \times 10^5, 3 \times 10^6, 1.2 \times 10^7\}$ measurements
- MF: $W_i \in \mathbb{R}^{d \times d}$

SVD complexity boundary:
$$
T_{\text{SVD}}(d) = C_{\text{svd}} \cdot d^3, \quad C_{\text{svd}} \approx 10^{-8} \text{ s (typical CPU)}
$$
$$
T_{\text{SGD-iter}}(d) = C_{\text{sgd}} \cdot d^2, \quad C_{\text{sgd}} \approx 10^{-9} \text{ s}
$$
Intersection scale $d^*$ satisfies:
$$
K_{\text{Muon}} \cdot C_{\text{svd}} d^3 = K_{\text{SGD}} \cdot C_{\text{sgd}} d^2 \quad \Rightarrow \quad d^* = \frac{K_{\text{SGD}}}{K_{\text{Muon}}} \cdot \frac{C_{\text{sgd}}}{C_{\text{svd}}}
$$
If $K_{\text{SGD}} / K_{\text{Muon}} \approx 10$, then $d^* \approx 10 \times 0.1 = 1$ (i.e., SGD is always faster), but this contradicts the theoretical advantage---indicating that constant factor estimates require experimental calibration.

Parameter settings:
- Problem: MS ($r = d/10$, $m = 3d^2$)
- Algorithms: Muon-Exact, Muon-RandomSVD($2r$), SGD
- $d \in \{500, 1000, 2000\}$
- $n = 5$ repetitions (large-scale experiments are costly)
- Tolerance $\epsilon = 10^{-4}$ (moderate precision, ensuring completion within reasonable time)
- Maximum wall-clock time: 4 hours per configuration

**Statistical Model**:

Factorial design:
$$
\text{Algorithm} \times d \times \text{Seed}
$$
Response variables:
- $K_\epsilon$ (if converged)
- $T_\epsilon$ (total wall-clock time)
- $\delta_{\text{conv}} \in \{0, 1\}$ (whether convergence is achieved within the time limit)
- Scaling efficiency: $S(d) = T_\epsilon(d/2) / T_\epsilon(d)$ (ideal value is 8 for Muon exact SVD, 4 for SGD)

Null and alternative hypotheses:
$$
H_0: T_\epsilon^{\text{Muon}}(d) / T_\epsilon^{\text{SGD}}(d) \leq 1 \quad \forall d \in \{500, 1000, 2000\}
$$
$$
H_1: \exists \; d_{\text{cross}} \text{ such that } d > d_{\text{cross}} \Rightarrow T_\epsilon^{\text{Muon}}(d) > T_\epsilon^{\text{SGD}}(d)
$$
Test statistic: paired test for time ratio at each $d$:
$$
R_j(d) = T_{\epsilon,j}^{\text{Muon}}(d) / T_{\epsilon,j}^{\text{SGD}}(d)
$$
Using log transformation: one-sample t-test of $\log R_j(d)$ vs. 0.

**Relationship to Existing Experiments**: Extends the existing dimension upper bound from $d=500$ to $d=2000$, verifying scalability.

**Expected Results and Interpretation**:
- **Expectation 1**: For $d=1000$, Muon-RandomSVD may still maintain a time advantage; for $d=2000$, exact SVD may be impractical due to memory/time constraints.
- **Expectation 2**: There exists a clear $d_{\text{cross}}$ (estimated $1500 < d_{\text{cross}} < 3000$ for exact SVD), beyond which randomized SVD becomes the only feasible implementation for Muon.
- **Expectation 3**: Scaling efficiency $S(d)$ will reveal implementation quality---if $S(d) \gg 8$, it indicates serious memory bottlenecks or cache thrashing.

---

### E16: Initialization Scale Sensitivity Experiment (Initialization C1/C3)

**Research Question**: Is Muon's spectral normalization intrinsically invariant to initialization scale? That is, if the scale of the initial gradient $G^{(0)}$ varies by a factor of $c$, is Muon's direction $D^{(0)} = UV^\top$ completely unaffected? Does the effect of different initialization variances on convergence rate differ between Muon and SGD?

**Mathematical Formulation**:

Initialization distribution:
$$
X^{(0)}_{ij} \overset{iid}{\sim} \mathcal{N}(0, \sigma_{\text{init}}^2), \quad \sigma_{\text{init}} \in \left\{\frac{10^{-3}}{\sqrt{d}}, \frac{10^{-2}}{\sqrt{d}}, \frac{10^{-1}}{\sqrt{d}}, \frac{1}{\sqrt{d}}, \frac{10}{\sqrt{d}}\right\}
$$
($1/\sqrt{d}$ is the existing "standard" initialization)

Key theoretical relationships:
- SGD update $X^{(1)} = X^{(0)} - \eta G^{(0)}$ is directly affected by the scale of $G^{(0)}$ ($\|G^{(0)}\|_F \propto \sigma_{\text{init}}$ for the MS problem).
- Muon update $X^{(1)} = X^{(0)} - \eta UV^\top$, where $G^{(0)} = U\Sigma V^\top$, so $UV^\top$ is theoretically independent of $\Sigma$.

But in practice:
- If $G^{(0)} \approx 0$ (extremely small initialization), numerical SVD is unstable.
- If $G^{(0)}$ is extremely large, floating-point overflow may be triggered or the learning rate $\eta$ needs to be recalibrated.

Parameter settings:
- Problems: MS ($d=100, 200$) and MF-L2 ($d=100, 200$)
- $r = d/10$, $m = 3d^2$
- Five $\sigma_{\text{init}}$ values
- Muon / SGD, $\lambda=0$
- $\eta$ searched independently on each $(\sigma_{\text{init}}, \text{Algorithm})$ combination
- $n = 15$ repetitions
- Tolerance $\epsilon = 10^{-6}$

**Statistical Model**:

Factorial design:
$$
\text{Algorithm} \times \sigma_{\text{init}} \times d \times \text{Problem} \times \text{Seed}
$$
Response variables: $K_\epsilon$, $\eta^*$ (optimal learning rate), $f_{\text{final}}$, $\|\nabla f(X^{(0)})\|_F$.

Null and alternative hypotheses:
$$
H_0^{\text{scale-inv}}: \text{For Muon, } K_\epsilon \text{ does not depend on } \sigma_{\text{init}} \quad \text{(scale invariance hypothesis)}
$$
$$
H_1^{\text{scale-inv}}: \exists \; \sigma_{\text{init}}^1, \sigma_{\text{init}}^2 \text{ such that } K_\epsilon(\sigma_{\text{init}}^1) \neq K_\epsilon(\sigma_{\text{init}}^2) \text{ for Muon}
$$
$$
H_0^{\text{SGD-dep}}: \text{For SGD, } K_\epsilon \text{ does not depend on } \sigma_{\text{init}}
$$
$$
H_1^{\text{SGD-dep}}: K_\epsilon^{\text{SGD}} \text{ significantly depends on } \sigma_{\text{init}}
$$

Test statistics:
- For each algorithm, one-way ANOVA (factor: $\sigma_{\text{init}}$):
  $$
  F = \frac{\text{MS}_{\sigma_{\text{init}}}}{\text{MS}_{\text{Error}}}
  $$
- If Muon's ANOVA is not significant while SGD's is, this supports the scale invariance hypothesis.
- Simultaneously test learning rate calibration requirements: the magnitude of change in $\eta^*$ with $\sigma_{\text{init}}$.

**Relationship to Existing Experiments**: Extends the fixed initialization $\mathcal{N}(0, 1/d)$ in existing experiments to a full scale sweep, verifying Muon's theoretical scale invariance.

**Expected Results and Interpretation**:
- **Expectation 1**: Muon's $K_\epsilon$ dependence on $\sigma_{\text{init}}$ is significantly weaker than SGD's, verifying the scale invariance of spectral normalization.
- **Expectation 2**: At extreme scales ($\sigma_{\text{init}} = 10^{-3}/\sqrt{d}$ or $10/\sqrt{d}$), Muon may also degrade---too small leads to numerical SVD instability, too large causes the initial step to overshoot the basin.
- **Expectation 3**: The optimal learning rate $\eta^*_{\text{Muon}}$ is less sensitive to $\sigma_{\text{init}}$ than $\eta^*_{\text{SGD}}$, which is a practical advantage of Muon in real-world hyperparameter tuning.

---

### E17: Orthogonal vs. Spectral Initialization Comparison Experiment (Initialization C2)

**Research Question**: If the initialization already has a specific spectral structure (e.g., orthogonal initialization, spectral initialization), does Muon's spectral normalization still provide additional value? Or, when initialization is already "good," does the gap between Muon and SGD narrow?

**Mathematical Formulation**:

Initialization strategies:
1. **Gaussian initialization** (baseline): $X^{(0)}_{ij} \sim \mathcal{N}(0, 1/d)$
2. **Orthogonal initialization**: $X^{(0)} = Q R$ (QR decomposition, $Q$ orthogonal/unitary), or partial QR for rectangular matrices.
3. **Spectral initialization** (Spectral Init): $X^{(0)} = U^{(0)} \text{diag}(\lambda^{(0)}) (V^{(0)})^\top$, where
   $$
   \lambda_i^{(0)} = \lambda_1 \cdot \rho^{i-1}, \quad \rho \in \{0.5, 0.9\}
   $$
   i.e., the initialization already has the same spectral decay pattern as the true matrix.
4. **Zero initialization**: $X^{(0)} = 0$ (test Muon's degradation under zero gradient)

Parameter settings:
- Problems: MS ($d=100, 200$) and MF-L2 ($d=100, 200$)
- $r = d/10$
- Four initialization strategies
- Muon / SGD, $\lambda=0$, $\eta$ searched independently
- $n = 15$ repetitions
- Tolerance $\epsilon = 10^{-6}$

**Statistical Model**:

Factorial design:
$$
\text{Algorithm} \times \text{Init-Strategy} \times d \times \text{Problem} \times \text{Seed}
$$
Response variables: $K_\epsilon$, $K_{\epsilon,\text{normalized}} = K_\epsilon \cdot \|X^{(0)} - X^\star\|_F$ (normalized to initial distance).

Null and alternative hypotheses:
$$
H_0^{\text{init}}: \text{Algorithm} \times \text{Init-Strategy} \text{ has no interaction effect on } K_\epsilon
$$
$$
H_1^{\text{init}}: \text{Muon is less sensitive to initialization strategy than SGD} \quad \text{(or vice versa)}
$$
Test statistic: two-way ANOVA interaction test
$$
F_{\text{int}} = \frac{\text{MS}_{\text{Alg} \times \text{Init}}}{\text{MS}_{\text{Error}}}
$$

Post-hoc analysis: Tukey HSD test to compare algorithm differences under each initialization strategy.

**Relationship to Existing Experiments**: Extends the three special initializations (for MF-L2) in existing experiments to a systematic comparison of initialization strategies.

**Expected Results and Interpretation**:
- **Expectation 1**: Spectral initialization may accelerate both Muon and SGD, but Muon's gain is smaller (because spectral normalization already provides similar "structured" information).
- **Expectation 2**: Zero initialization is an extreme test for Muon---if $G^{(0)} = 0$, SVD degenerates; need to analyze how the implementation handles zero gradients.
- **Expectation 3**: Orthogonal initialization may provide a more favorable initial Hessian condition for Muon, thereby amplifying the advantage.

---

### E18: Condition Number Control Experiment (Problem Variant B5)

**Research Question**: Is the ill-conditioning degree of the problem (Hessian condition number) a key moderating variable for Muon's advantage? Between well-conditioned problems ($\kappa \approx 1$) and ill-conditioned problems ($\kappa \approx 10^6$), how does Muon's convergence rate change relative to SGD?

**Mathematical Formulation**:

Condition number control method (sensing operator construction):

Goal: Construct a family of sensing operators $\{\mathcal{A}_\kappa\}$ such that the Hessian $H = \mathcal{A}^*\mathcal{A}/m$ has a specified condition number $\kappa_{\text{target}}$.

Method 1 (parameterized matrix family):
$$
H = Q^\top \text{diag}(\lambda_1, \dots, \lambda_{d^2}) Q, \quad \lambda_i = 1 + (\kappa_{\text{target}} - 1) \cdot \frac{i-1}{d^2-1}
$$
(linear spectral distribution, from 1 to $\kappa_{\text{target}}$)

Construct sensing vectors via Cholesky decomposition $H = L L^\top$:
$$
\text{vec}(A_i) = \text{row}_i(L) + \mathcal{N}(0, \sigma^2 I_{d^2})
$$
(add small noise to prevent degeneracy)

Method 2 (SVD water-filling):
Directly construct $X^\star = U^\star \text{diag}(\lambda_1, \dots, \lambda_r) (V^\star)^\top$, where $\lambda_i$ are distributed according to a specified decay rate.

Condition number levels:
$$
\kappa_{\text{target}} \in \{10, 10^2, 10^3, 10^4, 10^5, 10^6\}
$$

Parameter settings:
- Problem: MS ($d=100$)
- $r = 10$ (fixed), $m = 3d^2 = 30000$
- Six condition number levels
- Two spectral distributions: linear decay vs. geometric decay
- Muon / SGD, $\lambda=0$, $\eta$ searched independently
- $n = 15$ repetitions
- Tolerance $\epsilon = 10^{-6}$, maximum iterations $K_{\max} = 5 \times 10^5$ (ill-conditioned problems require more iterations)

**Statistical Model**:

Factorial design:
$$
\text{Algorithm} \times \kappa_{\text{target}} \times \text{Spectrum-Type} \times \text{Seed}
$$
Response variables:
- $K_\epsilon$ (convergence iterations)
- $\kappa_{\text{eff}} = \lambda_{\max}(H)/\lambda_{\min}(H)$ (actually measured condition number, verifying construction accuracy)
- Convergence rate estimate (local linear fit $\log(f(X^{(k)})) \sim -\rho k$)

Null and alternative hypotheses:
$$
H_0^{(\kappa)}: \mathbb{E}[\log K_\epsilon^{\text{Muon}} \mid \kappa] - \mathbb{E}[\log K_\epsilon^{\text{SGD}} \mid \kappa] = \text{const} \quad \forall \kappa
$$
$$
H_1^{(\kappa)}: \text{Advantage } \Delta(\kappa) = \mathbb{E}[\log K_\epsilon^{\text{SGD}}] - \mathbb{E}[\log K_\epsilon^{\text{Muon}}] \text{ increases monotonically with } \kappa
$$
(i.e., ill-conditioned problems amplify Muon's advantage)

Test statistics:
- Spearman rank correlation test: correlation between $\Delta(\kappa)$ and $\log \kappa$
  $$
  \rho_S = \text{corr}_{\text{rank}}(\Delta(\kappa), \log \kappa)
  $$
  $H_0: \rho_S = 0$, $H_1: \rho_S > 0$.
- Or linear model:
  $$
  \log K_\epsilon^{(i,j)} = \beta_0 + \beta_1 \log \kappa_j + \beta_2 \mathbb{I}_{\text{Muon}} + \beta_3 \log \kappa_j \cdot \mathbb{I}_{\text{Muon}} + \varepsilon_{ij}
  $$
  Test $\beta_3 < 0$ (Muon is less sensitive to condition number).

**Relationship to Existing Experiments**: Entirely new dimension---existing experiments use the natural condition number of random matrices without systematic control.

**Expected Results and Interpretation**:
- **Expectation 1**: On well-conditioned problems ($\kappa \approx 10$), the difference between Muon and SGD is small.
- **Expectation 2**: On ill-conditioned problems ($\kappa \approx 10^6$), Muon's advantage is significantly amplified, because the spectral normalization direction avoids gradient oscillations in the small-eigenvalue directions of the Hessian.
- **Expectation 3**: Establish the quantitative relationship $\Delta(\kappa) \approx c \cdot \log \kappa$, providing empirical coefficients for theoretical analysis.

---

### E19: Generalization across Different Matrix Distributions Experiment (Generalization H1/H3)

**Research Question**: Do existing experimental conclusions depend on specific properties of Gaussian measurement matrices (e.g., sub-Gaussianity, rotational invariance)? When measurement matrices are drawn from other distributions (Rademacher, structured, spherical Gaussian), does Muon's relative performance persist?

**Mathematical Formulation**:

Measurement matrix distribution family:
1. **Gaussian** (baseline): $A_{ij} \overset{iid}{\sim} \mathcal{N}(0, 1/d^2)$
2. **Rademacher**: $A_{ij} \overset{iid}{\sim} \text{Unif}\{-1/d, +1/d\}$
3. **Bernoulli-sparse**: $A_{ij} \overset{iid}{\sim} \text{Bernoulli}(p) \cdot \mathcal{N}(0, 1/(pd^2))$, $p=0.1$
4. **Spherical Gaussian**: $\text{vec}(A_i) \sim \text{Unif}(\mathbb{S}^{d^2-1})$ (uniform on the unit sphere)
5. **Fast JL transform**: $A_i = P H D \text{vec}(e_i)$, where $H$ is the Hadamard matrix, $D$ is a random diagonal $\pm 1$, and $P$ is random subsampling

Target matrix distributions:
- Standard: $X^\star = U \Sigma V^\top$, $U, V \sim \text{Haar}$
- Polynomial decay spectrum: $\lambda_i = i^{-\alpha}$, $\alpha \in \{0.5, 1.0, 2.0\}$
- Exponential decay spectrum: $\lambda_i = e^{-\beta i}$, $\beta \in \{0.1, 0.5\}$

Parameter settings:
- Problem: MS ($d=100, 200$)
- $r = d/10$, $m = 3d^2$
- Five measurement distributions $\times$ three target spectral types
- Muon / SGD, $\lambda=0$
- $n = 15$ repetitions
- Tolerance $\epsilon = 10^{-6}$

**Statistical Model**:

Factorial design:
$$
\text{Algorithm} \times \text{Meas-Dist} \times \text{Target-Spec} \times d \times \text{Seed}
$$
Response variables: $K_\epsilon$, $\bar{\sigma}_{\log}(G^{(0)})$, $\kappa_{\text{sp}}$.

Null and alternative hypotheses:
$$
H_0: \text{Measurement distribution does not affect the distribution of } \Delta K = K_\epsilon^{\text{SGD}} - K_\epsilon^{\text{Muon}}
$$
$$
H_1: \exists \; \text{Meas-Dist} \text{ such that } \Delta K \text{ is significantly different from the Gaussian baseline}
$$
Test statistics:
- Kruskal-Wallis H-test (multi-group non-parametric test):
  $$
  H = \frac{12}{N(N+1)}\sum_{i=1}^5 \frac{R_i^2}{n_i} - 3(N+1)
  $$
  where $R_i$ is the mean rank of the $i$-th distribution group, $N=5n$, $n_i=n$. If $H > \chi^2_{0.95, 4}$, reject $H_0$.
- Post-hoc: Dunn test for pairwise comparisons.

**Relationship to Existing Experiments**: Extends the fixed Gaussian measurement in existing experiments to a complete distribution family, verifying distributional robustness of conclusions.

**Expected Results and Interpretation**:
- **Expectation 1**: Sub-Gaussian distributions (Rademacher, spherical Gaussian) yield similar results to Gaussian, because random sensing theory guarantees that sub-Gaussian matrices satisfy the RIP.
- **Expectation 2**: Structured matrices (fast JL transform) may introduce different dynamics due to correlation structure, but the limiting behavior of low-rank recovery may be consistent.
- **Expectation 3**: Target spectral type strongly influences results---polynomial/exponential decay spectra may be more easily exploited by Muon than "hard truncation" low-rank spectra.

---

### E20: Statistical Power and Sample Size Determination Experiment (Statistical Reliability F1/F2)

**Research Question**: Existing experiments use $n=10$ repetitions; is this sample size sufficient to detect the true convergence rate difference between Muon and SGD? If not, how many repetitions are needed to achieve 80% power? Also, how can strict confidence intervals and multiple testing correction be provided for all hypothesis tests?

**Mathematical Formulation**:

Power analysis framework:

Define effect size:
$$
\delta = \frac{\mathbb{E}[\log K_\epsilon^{\text{SGD}}] - \mathbb{E}[\log K_\epsilon^{\text{Muon}}]}{\sigma_{\text{within}}}
$$
where $\sigma_{\text{within}}$ is the standard deviation of paired differences.

Estimate from existing data (or pilot experiments E6--E19):
- Baseline means: $\mu_{\text{SGD}} = \mathbb{E}[\log K_\epsilon^{\text{SGD}}]$, $\mu_{\text{Muon}} = \mathbb{E}[\log K_\epsilon^{\text{Muon}}]$
- Difference standard deviation: $\sigma_D = \text{SD}(\log K_\epsilon^{\text{SGD}} - \log K_\epsilon^{\text{Muon}})$

Power calculation:
For a paired t-test, given $\alpha = 0.05$ (one-sided), sample size $n$, and effect size $\delta$:
$$
\text{Power} = 1 - \beta = \Phi\left(\sqrt{n} \cdot \delta - z_{1-\alpha}\right)
$$
where $\Phi$ is the standard normal CDF, $z_{1-\alpha} = 1.645$.

Solve for $n_{\min}$ such that $\text{Power} \geq 0.8$:
$$
n_{\min} = \left\lceil \frac{(z_{1-\alpha} + z_{1-\beta})^2}{\delta^2} \right\rceil = \left\lceil \frac{(1.645 + 0.84)^2}{\delta^2} \right\rceil \approx \left\lceil \frac{6.2}{\delta^2} \right\rceil
$$

Parameter settings:
- Use data from E1 (MS) and E2 (MF-L2) (or run an extended experiment with $n=30$ if not yet available)
- Problems: MS ($d=100, 200$), MF-L2 ($d=100, 200$)
- Compute $\delta(d)$ and $n_{\min}(d)$ for each $(d, \text{Problem})$ combination
- Perform multiple testing correction analysis for all existing hypotheses H1--H5

**Statistical Model**:

Bootstrap confidence interval construction:
For response variable $Y = \log K_\epsilon$, use BCa Bootstrap:
$$
Y^{*(b)}_j = \text{resample}(Y_1, \dots, Y_n), \quad b = 1, \dots, B=10000
$$
$$
CI_{95\%} = [\hat{Y}^{*(\alpha/2)}, \hat{Y}^{*(1-\alpha/2)}]
$$

Multiple testing correction:
Existing experiments involve hypotheses H1--H5, a total of 5 main hypotheses. If testing at each dimension level, the total number of hypotheses $M$ may reach $5 \times 4 \times 2 = 40$ (dimensions $\times$ problems $\times$ ranks).

Use Holm stepwise correction:
$$
p_{(1)} \leq p_{(2)} \leq \dots \leq p_{(M)} \quad \text{(sorted p-values)}
$$
Reject all hypotheses satisfying $p_{(i)} \leq \alpha / (M - i + 1)$.

Or use Benjamini-Hochberg FDR control:
$$
\text{Reject all } p_{(i)} \leq \frac{i}{M} \cdot \alpha
$$

Null and alternative hypotheses:
$$
H_0^{\text{power}}: n=10 \text{ is sufficient to detect } \delta_{\min} = 0.5 \text{ (medium effect size)}
$$
$$
H_1^{\text{power}}: n_{\min} > 10 \text{ is needed to achieve } 80\% \text{ power}
$$
Test: directly compute $\hat{\delta}$ and $\hat{n}_{\min}$.

**Relationship to Existing Experiments**: Meta-analysis of the statistical rigor of existing experimental designs, determining whether additional repetitions are needed.

**Expected Results and Interpretation**:
- **Expectation 1**: For $d=50$, the effect size may be large ($\delta \approx 1.0$), and $n=10$ is sufficient; for $d=500$, the effect size may be small ($\delta \approx 0.3$), $n=10$ is underpowered, and $n_{\min} \approx 30$--$50$ is recommended.
- **Expectation 2**: Under $n=10$, some "marginally significant" results ($p \approx 0.04$) may no longer be significant after Holm correction, requiring a reassessment of the robustness of conclusions.
- **Expectation 3**: Provide a standardized effect size reporting template (Cohen's $d$, log ratio, speedup ratio and their confidence intervals) to improve the reproducibility and comparability of experiments.

---

