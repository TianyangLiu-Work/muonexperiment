<!--
Original Document: Mathematical Foundations and Statistical Formalization of Experimental Design for the Muon (μ) Optimization Algorithm
This File: 6. Formalization of the Data Generation Process
Split Index: 08
-->

[TOC]

---

## 6. Formalization of the Data Generation Process

### 2.1 DGP for the Matrix Sensing Problem

**Definition 2.1 (Matrix Sensing Data Generation Process).** Given parameters $(d, r, m, \sigma_\epsilon, s)$, the data are generated according to the following procedure:

**Step 1: Generate measurement matrices.**

$$
A_i \overset{iid}{\sim} \mathcal{N}_{d \times d}(0, 1), \quad i = 1, \ldots, m
$$

That is, each $A_i$ is a $d \times d$ matrix whose elements are independent and identically distributed (i.i.d.) standard normal random variables:

$$
(A_i)_{jk} \overset{iid}{\sim} \mathcal{N}(0, 1), \quad \forall j, k \in \{1, \ldots, d\}
$$

**Step 2: Generate the ground-truth matrix $X^\star$.**

**Low-rank case** ($r < d$):


$$
U, V \in \mathbb{R}^{d \times r}, \quad U_{jk} \overset{iid}{\sim} \mathcal{N}(0, 1), \quad V_{jk} \overset{iid}{\sim} \mathcal{N}(0, 1)
$$



$$
X^\star = \frac{1}{\sqrt{r}} U V^T
$$


The scaling factor $1/\sqrt{r}$ ensures that $\mathbb{E}[\|X^\star\|_F^2] = d^2$, which is of the same order of magnitude as the full-rank case.


**Full-rank case** ($r = d$):


$$
X^\star_{ij} \overset{iid}{\sim} \mathcal{N}(0, 1)
$$



**Step 3: Generate measurements and noise.**

$$
y_i = \langle A_i, X^\star \rangle + \epsilon_i, \quad \epsilon_i \overset{iid}{\sim} \mathcal{N}(0, \sigma_\epsilon^2)
$$

where the matrix inner product is defined as:

$$
\langle A, B \rangle = \sum_{j=1}^d \sum_{k=1}^d A_{jk} B_{jk} = \mathrm{tr}(A^T B)
$$

**Joint distribution:**

Given a random seed $s$, the data generation mapping $\Phi_{MS}$ outputs the joint sample:

$$
(\{A_i\}_{i=1}^m, X^\star, \{\epsilon_i\}_{i=1}^m) = \Phi_{MS}(s; d, r, m, \sigma_\epsilon)
$$

Its joint probability density (in the full-rank, noiseless case) is:

$$
p_{MS}(\{A_i\}, X^\star) = \prod_{i=1}^m \prod_{j,k=1}^d \phi(A_{i,jk}) \cdot \prod_{j,k=1}^d \phi(X^\star_{jk})
$$

where $\phi(x) = (2\pi)^{-1/2} e^{-x^2/2}$ is the standard normal density. In the presence of noise, multiply by $\prod_{i=1}^m \phi_{\sigma_\epsilon}(\epsilon_i)$.

---

### 2.2 DGP for the Matrix Factorization Problem

**Definition 2.2 (Matrix Factorization Data Generation Process).** Given parameters $(d, L, s)$, the data are generated according to the following procedure:

**Step 1: Generate the target matrix.**

$$
X^\star_{ij} \overset{iid}{\sim} \mathcal{N}(0, 1)
$$

**Step 2: Generate initializations (three schemes for $L=2$).**

Define the noise matrix $E$, where $E_{ij} \overset{iid}{\sim} \mathcal{N}(0, 0.01^2)$:

**Scheme (a)** — Imbalanced Initialization I:


$$
W_1^{(0)} = X^\star + E, \quad W_2^{(0)} = I_d
$$



**Scheme (b)** — Imbalanced Initialization II:


$$
W_1^{(0)} = I_d, \quad W_2^{(0)} = X^\star + E
$$



**Scheme (c)** — Symmetric Initialization:


$$
W_1^{(0)} = c \cdot I_d, \quad W_2^{(0)} = \frac{1}{c} \cdot (X^\star + E)
$$


where $c > 0$ is an arbitrary constant (typically $c = 1$).


**Step 3: Generate initializations ($L \geq 3$).**

$$
W_i^{(0)} \in \mathbb{R}^{d \times d}, \quad (W_i^{(0)})_{jk} \overset{iid}{\sim} \mathcal{N}\left(0, \frac{1}{d}\right), \quad i = 1, \ldots, L
$$

The scaling factor $1/d$ ensures that $\mathbb{E}[\|W_i^{(0)}\|_F^2] = d$, keeping the spectral norm at the order of $O(1)$.

**Joint distribution:**

$$
(X^\star, W_1^{(0)}, \ldots, W_L^{(0)}) = \Phi_{MF}(s; d, L)
$$

Its joint density (for $L \geq 3$ with independent initializations) is:

$$
p_{MF}(X^\star, \{W_i^{(0)}\}) = \prod_{j,k=1}^d \phi(X^\star_{jk}) \cdot \prod_{i=1}^L \prod_{j,k=1}^d \sqrt{d} \cdot \phi\left(\sqrt{d} \cdot (W_i^{(0)})_{jk}\right)
$$

---

### 2.3 Unified Data Generation Mapping

Synthesizing the two problems above, the unified data generation mapping is:

$$
\Phi: \mathcal{R}_{\text{seed}} \times \mathcal{P} \longrightarrow \mathcal{D}_{\text{data}} \times \mathcal{D}_{\text{init}}
$$

That is, given a random seed and the problem instance parameters, it deterministically outputs all data and initializations.

---
