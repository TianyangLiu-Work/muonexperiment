<!--
Original Document: Mathematical Foundations and Statistical Formalization of Experimental Design for the Muon (μ) Optimization Algorithm
This File: 5. Axiomatic Definition of Experimental Space
Split Number: 07
-->

[TOC]

---

# Part II: Mathematical Formalization of Experiments

> This section transforms existing experimental protocols into rigorous mathematical and statistical language, ensuring that all concepts, variables, processes, and assumptions have precise formal definitions. Building upon the mathematical foundations established in Part I, it lays an unambiguous groundwork for subsequent experimental execution, statistical testing, and conclusion inference.

> **Purpose of This Chapter**: To transform existing experimental protocols into rigorous mathematical and statistical language, ensuring that all concepts, variables, processes, and assumptions have precise formal definitions, thereby laying an unambiguous foundation for subsequent experimental execution, statistical testing, and conclusion inference.

---

## 5. Axiomatic Definition of Experimental Space

### 1.1 Experimental Quintuple

An experiment $\mathcal{E}$ is defined as a quintuple:

$$
\mathcal{E} = (\mathcal{P}, \mathcal{A}, \mathcal{D}, \mathcal{M}, \mathcal{R})
$$

The definitions of each space are given below.

---

### 1.2 Problem Instance Space $\mathcal{P}$

The problem instance space is the set of all optimization problems to be investigated. This experiment includes two types of instances:

**Definition 1.1 (Matrix Sensing Instance).** A matrix sensing instance is determined by the following parameters:

$$
\mathcal{P}_{MS} = \left\{ p_{MS}(d, r, m, \sigma_\epsilon) \; \middle| \; d \in \{50, 100, 200, 500\}, \; r \in \{5, \lfloor d/2 \rfloor, d\}, \; m = 3d^2, \; \sigma_\epsilon \in \{0, 0.01\} \right\}
$$

where:
- $d \in \mathbb{N}^+$: matrix dimension;
- $r \in \{1, \ldots, d\}$: rank of the true matrix $X^\star$ ($r = d$ indicates full rank);
- $m \in \mathbb{N}^+$: number of measurements;
- $\sigma_\epsilon \geq 0$: noise standard deviation.

The specific construction of each instance is described in the data generation process in Section 2.

**Definition 1.2 (Matrix Factorization Instance).** A matrix factorization instance is determined by the following parameters:

$$
\mathcal{P}_{MF} = \left\{ p_{MF}(d, r, L) \; \middle| \; d \in \{50, 100, 200, 500\}, \; r = d, \; L \in \{2, 3, 4\} \right\}
$$

where:
- $d \in \mathbb{N}^+$: dimension of all intermediate matrices (square matrices);
- $L \in \{2, 3, 4\}$: factorization depth (number of layers);
- $r = d$: rank of the target matrix (always full rank in this experiment).

**Total Problem Instance Space:**

$$
\mathcal{P} = \mathcal{P}_{MS} \cup \mathcal{P}_{MF}
$$

---

### 1.3 Algorithm Space $\mathcal{A}$

The algorithm space contains the two first-order optimization algorithms compared in this experiment:

$$
\mathcal{A} = \{ \text{Muon}, \; \text{SGD} \}
$$

**Definition 1.3 (Muon Algorithm).** Given parameters $(\eta, \lambda, \mu_{mom}, \mu_{Nesterov}) \in \mathcal{D}$, the iterative mapping of the Muon algorithm is denoted as $\mathcal{T}_\eta^{\text{Muon}}$. Its update rule is:

$$
X^{(k+1)} = \mathcal{T}_\eta^{\text{Muon}}(X^{(k)}) := X^{(k)} - \eta \cdot D^{(k)} - \lambda \cdot X^{(k)}
$$

where the construction of $D^{(k)}$ is as follows: let the current stochastic gradient be $G^{(k)} \in \mathbb{R}^{d \times d}$, and perform SVD on it:

$$
G^{(k)} = U \cdot \Sigma \cdot V^T
$$

where $U, V \in \mathbb{R}^{d \times d}$ are orthogonal matrices, $\Sigma = \mathrm{diag}(\sigma_1, \ldots, \sigma_d)$ and $\sigma_1 \geq \cdots \geq \sigma_d \geq 0$. Then:

$$
D^{(k)} := U V^T
$$

That is, take the outer product of the left and right singular vectors from the SVD, discarding the singular value information.

**Definition 1.4 (SGD Algorithm).** Given parameters $(\eta, \lambda) \in \mathcal{D}$, the iterative mapping of the SGD algorithm is denoted as $\mathcal{T}_\eta^{\text{SGD}}$. Its update rule is:

$$
X^{(k+1)} = \mathcal{T}_\eta^{\text{SGD}}(X^{(k)}) := X^{(k)} - \eta \cdot G^{(k)} - \lambda \cdot X^{(k)}
$$

where $G^{(k)}$ is the current stochastic gradient, used directly without spectral decomposition.

---

### 1.4 Data / Initialization / Hyperparameter Space $\mathcal{D}$

The space $\mathcal{D}$ contains all data and hyperparameter configurations controlling the experimental conditions:

$$
\mathcal{D} = \mathcal{D}_{\text{data}} \times \mathcal{D}_{\text{init}} \times \mathcal{D}_{\text{hyper}}
$$

**Definition 1.5 (Data Space).**

$$
\mathcal{D}_{\text{data}} = \{ (\{A_i\}_{i=1}^m, y, X^\star) \; | \; \text{generated according to the DGP in Section 2} \}
$$

**Definition 1.6 (Initialization Space).**

For matrix sensing problems:

$$
\mathcal{D}_{\text{init}}^{MS} = \left\{ X^{(0)} \in \mathbb{R}^{d \times d} \; \middle| \; X^{(0)}_{ij} \overset{iid}{\sim} \mathcal{N}(0, \alpha^2), \; \alpha \in \{0.01, 0.1, 1.0\} \right\}
$$

For matrix factorization problems ($L=2$):

$$
\mathcal{D}_{\text{init}}^{MF, L=2} = \left\{ (W_1^{(0)}, W_2^{(0)}) \in \mathbb{R}^{d \times d} \times \mathbb{R}^{d \times d} \; \middle| \; \text{three initialization schemes, see Section 2} \right\}
$$

For matrix factorization problems ($L \geq 3$):

$$
\mathcal{D}_{\text{init}}^{MF, L\geq 3} = \left\{ (W_1^{(0)}, \ldots, W_L^{(0)}) \; \middle| \; W_i^{(0)} \in \mathbb{R}^{d \times d}, \; W_{i,jk}^{(0)} \overset{iid}{\sim} \mathcal{N}(0, 1/d) \right\}
$$

**Definition 1.7 (Hyperparameter Space).**

$$
\mathcal{D}_{\text{hyper}} = \left\{ (\eta, \lambda, K_{\max}, \epsilon) \; \middle| \; \eta \in \{10^{-3}, 10^{-2}, 10^{-1}\}, \; \lambda = 0, \; K_{\max} = 10^5, \; \epsilon = 10^{-6} \right\}
$$

---

### 1.5 Metric Space $\mathcal{M}$

The metric space contains all random variables used to evaluate algorithm performance:

$$
\mathcal{M} = \{ K_\epsilon, F_\epsilon, \bar{\sigma}_{\log}, \rho_K, \rho_F, \delta_k \}
$$

The precise definitions of each metric are given in Section 3.4.

---

### 1.6 Randomness Space $\mathcal{R}$

The randomness space captures all sources of non-determinism in the experiment:

$$
\mathcal{R} = \mathcal{R}_{\text{seed}} \times \mathcal{R}_{\text{problem}} \times \mathcal{R}_{\text{noise}} \times \mathcal{R}_{\text{init}} \times \mathcal{R}_{\text{stoch}}
$$

where:
- $\mathcal{R}_{\text{seed}} = \{1, 2, \ldots, R\}$, with $R = 10$ independent random seeds;
- $\mathcal{R}_{\text{problem}}$: random generation of problem data (measurement matrices $\{A_i\}$, true matrix $X^\star$);
- $\mathcal{R}_{\text{noise}}$: random generation of observation noise $\{\epsilon_i\}$;
- $\mathcal{R}_{\text{init}}$: random generation of parameter initialization $X^{(0)}$ or $\{W_i^{(0)}\}$;
- $\mathcal{R}_{\text{stoch}}$: randomness in stochastic gradient sampling (when using mini-batch SGD).

To control experimental variables, given a random seed $s \in \mathcal{R}_{\text{seed}}$, all other random sources are determined by $s$ through a deterministic pseudorandom number generator:

$$
(\{A_i\}, X^\star, \{\epsilon_i\}, X^{(0)}) = \Phi(s)
$$

where $\Phi$ is a fixed random number generation mapping, ensuring the **fair randomization principle** (identical randomness given identical seed) for cross-algorithm comparison.

---

