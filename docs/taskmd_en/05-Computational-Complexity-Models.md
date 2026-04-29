<!--
Original Document: Mathematical Foundations and Statistical Formalization of Experimental Design for the Muon ($\mu$) Optimization Algorithm
This File: 4. Computational Complexity Models
Split Index: 05
-->

[TOC]

---

## 4. Computational Complexity Models

### 4.1 FLOPs Counting Model

FLOPs (Floating-Point Operations) is the basic unit for measuring computational cost of an algorithm. In matrix optimization, precise FLOPs counting is essential for both theoretical analysis and experimental validation.

#### 4.1.1 FLOPs for Basic Matrix Operations

**Definition 4.1** (Matrix Multiplication FLOPs). Let $A \in \mathbb{R}^{m \times n}$, $B \in \mathbb{R}^{n \times p}$, then the FLOPs count for the product $C = AB$ is
$$\text{FLOPs}(A \cdot B) = 2mnp - mp \approx 2mnp \quad (\text{when } n \gg 1).$$
In particular, for $d \times d$ square matrix multiplication: $\text{FLOPs}(A \cdot B) \approx 2d^3$.

**Definition 4.2** (Matrix-Vector Multiplication FLOPs). $A \in \mathbb{R}^{m \times n}$, $x \in \mathbb{R}^n$:
$$\text{FLOPs}(Ax) = 2mn - m \approx 2mn.$$

**Definition 4.3** (Matrix Inner Product FLOPs). $\langle A, B \rangle = \text{tr}(A^\top B)$ requires $2d_1 d_2$ FLOPs.

#### 4.1.2 FLOPs for SVD Decomposition

**Definition 4.4** (FLOPs for Full SVD). For a dense matrix $X \in \mathbb{R}^{d_1 \times d_2}$ (assuming $d_1 \leq d_2$), the computational complexity of full SVD is
$$\text{FLOPs}(\text{SVD}(X)) \approx 4d_1^2 d_2 + 8d_1^3 \approx O(d_1^2 d_2).$$
For a $d \times d$ square matrix: $\text{FLOPs}(\text{SVD}) \approx 12d^3$.

More precise counting (Golub-Reinsch SVD algorithm):
- First, bidiagonalization via Householder transformations: $\frac{8}{3}d^3$ FLOPs
- QR iteration for singular values: $O(d^2)$ to $O(d^3)$, depending on convergence rate
- Total complexity is approximately $12d^3$ to $20d^3$ FLOPs

**Definition 4.5** (FLOPs for Truncated SVD / Partial SVD). If only the first $k \ll d$ singular values and singular vectors are needed (e.g., using Lanczos or randomized SVD methods), the complexity can be reduced to
$$\text{FLOPs}(\text{truncated-SVD}_k) = O(k \cdot d^2 \cdot \text{iterations}) \ll d^3.$$
In Muon, if the gradient matrix is approximately low-rank, truncated SVD can significantly reduce the computational cost.

#### 4.1.3 Per-Step FLOPs Counting for Each Algorithm

**Definition 4.6** (Per-Step FLOPs for SGD). For the matrix sensing problem, the gradient of SGD (full-batch GD) is
$$G^{(k)} = \frac{1}{m}\mathcal{A}^*(\mathcal{A}(X^{(k)}) - y) = \frac{1}{m}\sum_{i=1}^m \left(\langle A_i, X^{(k)}\rangle - y_i\right) A_i.$$
The gradient computation involves $m$ matrix inner products and $m$ scalar-matrix multiplications:
$$\text{FLOPs}_k^{\text{SGD}} = m \cdot (2d^2 + d^2) + d^2 = 3md^2 + d^2 \approx 3md^2.$$
The parameter update $X^{(k+1)} = (1-\lambda)X^{(k)} - \eta G^{(k)}$ requires $3d^2$ FLOPs.
Total:
$$\text{FLOPs}_k^{\text{SGD}} = 3md^2 + 3d^2 \approx 3(m+1)d^2.$$

**Definition 4.7** (Per-Step FLOPs for Muon). The Muon update is
$$X^{(k+1)} = (1-\lambda) X^{(k)} - \eta \cdot \mathcal{N}_{\text{spec}}(G^{(k)}).$$
The spectral normalization requires SVD decomposition of the gradient matrix $G^{(k)}$. Precise counting:
- Gradient computation: $3md^2$ FLOPs (same as SGD)
- SVD decomposition: $\approx 12d^3$ FLOPs (dense full SVD)
- Spectral normalization $UV^\top$: $2d^3$ FLOPs ($d \times d$ matrix multiplied by $d \times d$ matrix)
- Parameter update: $3d^2$ FLOPs
Total:
$$\text{FLOPs}_k^{\text{Muon}} = 3md^2 + 12d^3 + 2d^3 + 3d^2 \approx 3md^2 + 14d^3.$$

**Definition 4.8** (FLOPs Ratio). The per-step FLOPs ratio of Muon relative to SGD is
$$\gamma_{\text{FLOPs}} := \frac{\text{FLOPs}_k^{\text{Muon}}}{\text{FLOPs}_k^{\text{SGD}}} = \frac{3md^2 + 14d^3}{3md^2} = 1 + \frac{14d}{3m} = 1 + O\left(\frac{d}{m}\right).$$
When $m \gg d$, $\gamma_{\text{FLOPs}} \approx 1$, and the additional overhead of SVD is relatively small; when $m \approx d$ or $m < d$, $\gamma_{\text{FLOPs}} \gg 1$.

#### 4.1.4 Cumulative Computational Complexity

**Definition 4.9** (Total Computational Complexity). The **total FLOPs** for algorithm $A$ to achieve accuracy $\epsilon$ is
$$F_\epsilon^{(A)} := \sum_{k=0}^{K_\epsilon^{(A)}-1} \text{FLOPs}_k^{(A)}.$$
If the per-step FLOPs is constant, then $F_\epsilon^{(A)} = K_\epsilon^{(A)} \cdot \text{FLOPs}_{\text{per-step}}^{(A)}$.

**Definition 4.10** (Computational Efficiency Comparison). The **computational efficiency ratio** of algorithm $A$ relative to algorithm $B$ is
$$\rho_F^{(A,B)} := \frac{F_\epsilon^{(A)}}{F_\epsilon^{(B)}} = \frac{K_\epsilon^{(A)} \cdot \text{FLOPs}^{(A)}}{K_\epsilon^{(B)} \cdot \text{FLOPs}^{(B)}}.$$
When $\rho_F^{(A,B)} < 1$, algorithm $A$ is more efficient in terms of total computational cost, even if each individual step is more expensive.

**Experimental Implication**: Muon's per-step cost is higher than SGD's (mainly due to SVD), but if Muon's iteration complexity $K_\epsilon^{\text{Muon}}$ is significantly smaller than $K_\epsilon^{\text{SGD}}$, the total computational cost may still be more favorable. In experiments, both $K_\epsilon$ (iteration efficiency) and $F_\epsilon$ (computational efficiency) must be compared simultaneously for a comprehensive evaluation of the algorithm.

### 4.2 Asymptotic Complexity Analysis

#### 4.2.1 Asymptotic Notation System

**Definition 4.11** (Asymptotic Notation). Let $f(n), g(n)$ be positive functions:
- **Big-O notation**: $f(n) = O(g(n))$ means there exist $C > 0$ and $n_0$ such that $f(n) \leq C g(n)$ for all $n \geq n_0$.
- **Big-Omega notation**: $f(n) = \Omega(g(n))$ means there exist $c > 0$ and $n_0$ such that $f(n) \geq c g(n)$ for all $n \geq n_0$.
- **Big-Theta notation**: $f(n) = \Theta(g(n))$ means $f(n) = O(g(n))$ and $f(n) = \Omega(g(n))$.
- **Little-o notation**: $f(n) = o(g(n))$ means $\lim_{n \to \infty} f(n)/g(n) = 0$.

#### 4.2.2 Asymptotic Analysis of Problem Dimensions

**Definition 4.12** (Problem Scale Parameters). The scale parameters of the matrix sensing problem are $(d, r, m)$, where:
- $d$: matrix dimension (dominant parameter)
- $r$: target rank ($r \ll d$)
- $m$: number of measurements (typically $m = O(rd)$ or $m = O(d^2)$)

**Asymptotic Complexity Summary**:
| Operation | Complexity |
|:----------|:-----------|
| Gradient computation (full-batch) | $O(md^2)$ |
| Full SVD (dense) | $O(d^3)$ |
| Truncated SVD (first $r$) | $O(rd^2)$ |
| Matrix multiplication ($d \times d$) | $O(d^3)$ |
| Matrix-vector multiplication | $O(d^2)$ |

#### 4.2.3 Asymptotic Comparison of Total Algorithm Complexity

**Definition 4.13** (Total Asymptotic Complexity of Muon). Let the iteration complexity of Muon be $K_\epsilon^{\text{Muon}}$, then
$$F_\epsilon^{\text{Muon}} = K_\epsilon^{\text{Muon}} \cdot O(md^2 + d^3).$$
If $m \gg d$, the dominant term is $md^2$; if $m \approx d$, then the $d^3$ term (SVD) may dominate.

**Definition 4.14** (Total Asymptotic Complexity of SGD).
$$F_\epsilon^{\text{SGD}} = K_\epsilon^{\text{SGD}} \cdot O(md^2).$$

**Comparison Condition**: Muon is computationally more efficient than SGD if and only if
$$K_\epsilon^{\text{Muon}} \cdot (md^2 + d^3) < K_\epsilon^{\text{SGD}} \cdot md^2,$$
i.e.,
$$\frac{K_\epsilon^{\text{Muon}}}{K_\epsilon^{\text{SGD}}} < \frac{md^2}{md^2 + d^3} = \frac{1}{1 + d/m}.$$
When $m \gg d$, the right-hand side $\approx 1$, requiring $K_\epsilon^{\text{Muon}} < K_\epsilon^{\text{SGD}}$ (any advantage may lead to a more favorable total computational cost); when $m \ll d$, the right-hand side $\approx m/d \ll 1$, requiring Muon's iteration advantage to be sufficiently large to compensate for the SVD overhead.

### 4.3 Relationship Between Wall-Clock Time and Theoretical FLOPs

#### 4.3.1 Theoretical FLOPs and Measured Time

**Definition 4.15** (Wall-Clock Time). The **wall-clock time** $T_{\text{wall}}$ of an algorithm is the physical elapsed time from the start to the end of the algorithm, including:
- Actual floating-point operation time
- Memory access time (loading/storing data)
- Cache miss penalties
- Parallelization overhead (thread synchronization, communication)
- System overhead (OS scheduling, interrupts)

**Definition 4.16** (Relationship Between Theoretical Time and FLOPs). Let the peak floating-point performance of the processor be $P$ FLOPs/second (e.g., modern CPUs at $10^{10} \sim 10^{11}$, GPUs at $10^{13} \sim 10^{14}$), then the **theoretical minimum time** is
$$T_{\text{theory}} = \frac{\text{Total FLOPs}}{P}.$$
The actual wall-clock time satisfies $T_{\text{wall}} \geq T_{\text{theory}}$, and typically $T_{\text{wall}} = c \cdot T_{\text{theory}}$, where $c \geq 1$ is the **implementation efficiency factor**.

**Definition 4.17** (Arithmetic Intensity and Memory Wall). **Arithmetic intensity** is defined as
$$I := \frac{\text{FLOPs}}{\text{Bytes moved}}.$$
When $I$ is high, the algorithm is compute-bound, approaching peak performance; when $I$ is low, the algorithm is memory-bound, and the actual performance is far below the peak.

The arithmetic intensity of matrix multiplication is $O(d)$ (for $d \times d$ matrix multiplication); that of SVD is similar. Both are compute-intensive operations and can achieve relatively high efficiency on modern hardware.

#### 4.3.2 The Importance of Constant Factors

**Definition 4.18** (Effective Constant). In practical comparisons, the constant factors hidden by big-O notation are crucial. The **effective constant** is defined as
$$C_{\text{eff}} := \frac{T_{\text{wall}} \cdot P}{\text{Theoretical FLOPs}} = \frac{\text{Actual Time} \times \text{Peak Performance}}{\text{Theoretical FLOPs}}.$$
The $C_{\text{eff}}$ of different algorithms (with different operation mixes) may differ substantially. For example:
- The $C_{\text{eff}}$ of SVD is typically higher than that of matrix multiplication, because SVD involves more serial dependencies and branch decisions
- Highly optimized matrix multiplication libraries (e.g., BLAS, cuBLAS) have $C_{\text{eff}}$ close to 1

**Experimental Implication**: Experimental reports should present both theoretical FLOPs and actual wall-clock time. Theoretical FLOPs serve as a hardware-agnostic baseline for fair comparison; wall-clock time reflects the actual performance on the software implementation and hardware. The difference between the two reveals the implementation efficiency of the algorithm.

### 4.4 Memory Complexity

#### 4.4.1 Space Complexity

**Definition 4.19** (Space Complexity). The **space complexity** (memory footprint) of an algorithm is the amount of data that must reside in memory simultaneously during execution.

For SGD in the matrix sensing problem:
- Current parameters $X^{(k)}$: $d^2$ floating-point numbers
- Gradient $G^{(k)}$: $d^2$ floating-point numbers
- Measurement matrices (if stored): $m \cdot d^2$ floating-point numbers (usually $m \cdot d^2$ is very large, can be replaced with on-the-fly generation or a compressed representation)
Total space: $S_{\text{SGD}} = 2d^2 + m \cdot d^2 \approx O(md^2)$ (if all measurement matrices are stored) or $O(d^2)$ (if generated online).

For Muon:
- Current parameters $X^{(k)}$: $d^2$ floating-point numbers
- Gradient $G^{(k)}$: $d^2$ floating-point numbers
- Temporary matrices $U, \Sigma, V$ during SVD decomposition: $3d^2$ floating-point numbers
- Measurement matrices: same as SGD
Total space: $S_{\text{Muon}} = 5d^2 + m \cdot d^2 \approx O(md^2)$ (when measurement matrices are stored) or $O(d^2)$ (when generated online).

#### 4.4.2 Cache Efficiency and Data Locality

**Definition 4.20** (Cache Complexity). Let the cache line size be $L_c$ bytes and the cache capacity be $C$ bytes. The **number of cache misses** is an important metric for measuring memory access efficiency.

Matrix multiplication can optimize cache efficiency through blocking, and although the time complexity remains $O(d^3)$, the constant factor is significantly reduced. The bidiagonalization process of SVD naturally has good data locality, but the random access patterns during the QR iteration phase may lead to a large number of cache misses.

**Experimental Implication**: When $d$ is large, memory footprint may become a bottleneck. If $d^2$ exceeds the L3 cache capacity (typically $10 \sim 30$ MB), memory bandwidth becomes the limiting factor. In experiments, peak memory usage should be recorded to evaluate the scalability of the algorithm.

---

