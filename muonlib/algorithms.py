"""Muon optimization algorithms and baselines for matrix sensing/factorization.

Per task.md §2.2.3: Muon update = X - lr * U@V^T - lr*λ*X
where G = U Σ V^T is the SVD of the gradient.
"""
import numpy as np
from scipy.sparse.linalg import svds


# ═══════════════════════════════════════════════
# Muon Optimizer
# ═══════════════════════════════════════════════
class MuonOptimizer:
    """Muon optimizer with spectral normalization and decoupled weight decay.

    Update rule (task.md §2.2.3):
        D = U @ V^T          (all singular values → 1)
        momentum = μ * momentum + D
        X_new = X - lr * momentum - lr * λ * X

    Variants:
        'exact'     — full SVD
        'randsvd'   — randomized SVD (Halko algorithm)
        'truncated' — top-r truncated SVD: D = U[:,:r] @ Vt[:r,:]

    """
    def __init__(self, variant='exact', mu=0.9,
                 weight_decay=0.0, r=None, p=10, q=2,
                 nesterov=False, random_state=0):
        import warnings
        if nesterov:
            warnings.warn(
                "nesterov=True is deprecated and not implemented in this version. "
                "Use lookahead() manually if needed.",
                DeprecationWarning,
                stacklevel=2
            )
        self.variant = variant
        self.mu = mu
        self.weight_decay = weight_decay
        self.r = r       # target rank for truncated
        self.p = p       # oversampling for randomized SVD
        self.q = q       # power iterations for randomized SVD
        self.momentum = None
        self.rng = np.random.RandomState(random_state)

    def lookahead(self, X):
        """Return look-ahead position for Nesterov momentum."""
        if self.momentum is not None and np.any(self.momentum):
            return X + self.mu * self.momentum
        return X

    def step(self, X, G, lr):
        """Single Muon step."""
        if self.momentum is None:
            self.momentum = np.zeros_like(X)

        # Spectral normalization: D = U @ V^T (task.md §2.2.3)
        if self.variant == 'truncated':
            if self.r is None:
                self.r = max(1, min(G.shape) // 10)
            min_dim = min(G.shape)
            if min_dim <= 1:
                G_eff = G / max(np.linalg.norm(G), 1e-16)
            else:
                r_eff = max(1, min(self.r, min_dim - 1))
                U, s, Vt = svds(G, k=r_eff, which='LM')
                order = np.argsort(s)[::-1]
                U, s, Vt = U[:, order], s[order], Vt[order, :]
                G_eff = U @ Vt
        elif self.variant == 'randsvd':
            U, s, Vt = randomized_svd(G, self.p, self.q, rng=self.rng)
            G_eff = U @ Vt
        else:
            U, s, Vt = np.linalg.svd(G, full_matrices=False)
            G_eff = U @ Vt

        self.momentum = self.mu * self.momentum + G_eff

        # Update with decoupled weight decay
        X_new = X - lr * self.momentum
        if self.weight_decay > 0:
            X_new = X_new - lr * self.weight_decay * X
        return X_new

    def reset(self):
        self.momentum = None


# ═══════════════════════════════════════════════
# Baseline Optimizers
# ═══════════════════════════════════════════════
class SGDOptimizer:
    """SGD with optional momentum and weight decay."""
    def __init__(self, momentum=0.0, weight_decay=0.0):
        self.momentum_beta = momentum
        self.weight_decay = weight_decay
        self.velocity = None

    def step(self, X, G, lr):
        if self.velocity is None:
            self.velocity = np.zeros_like(X)
        self.velocity = self.momentum_beta * self.velocity + G
        X_new = X - lr * self.velocity
        if self.weight_decay > 0:
            X_new = X_new - lr * self.weight_decay * X
        return X_new

    def reset(self):
        self.velocity = None


class AdamOptimizer:
    """Adam with bias correction."""
    def __init__(self, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = None
        self.v = None
        self.t = 0

    def step(self, X, G, lr):
        if self.m is None:
            self.m = np.zeros_like(X)
            self.v = np.zeros_like(X)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * G
        self.v = self.beta2 * self.v + (1 - self.beta2) * G**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        X_new = X - lr * m_hat / (np.sqrt(v_hat) + self.eps)
        if self.weight_decay > 0:
            X_new = X_new - lr * self.weight_decay * X
        return X_new

    def reset(self):
        self.m = None; self.v = None; self.t = 0


class RMSpropOptimizer:
    """RMSprop."""
    def __init__(self, decay=0.99, eps=1e-8, weight_decay=0.0):
        self.decay = decay
        self.eps = eps
        self.weight_decay = weight_decay
        self.v = None

    def step(self, X, G, lr):
        if self.v is None:
            self.v = np.zeros_like(X)
        self.v = self.decay * self.v + (1 - self.decay) * G**2
        X_new = X - lr * G / (np.sqrt(self.v) + self.eps)
        if self.weight_decay > 0:
            X_new = X_new - lr * self.weight_decay * X
        return X_new

    def reset(self):
        self.v = None


class MomentumSGDOptimizer:
    """Standard momentum SGD."""
    def __init__(self, mu=0.9, weight_decay=0.0):
        self.mu = mu
        self.weight_decay = weight_decay
        self.buf = None

    def step(self, X, G, lr):
        if self.buf is None:
            self.buf = np.zeros_like(X)
        self.buf = self.mu * self.buf + G
        X_new = X - lr * self.buf
        if self.weight_decay > 0:
            X_new = X_new - lr * self.weight_decay * X
        return X_new

    def reset(self):
        self.buf = None


# ═══════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════
def randomized_svd(A, p=10, q=2, rng=None):
    """Randomized SVD returning U, s, Vt (Halko algorithm)."""
    m, n = A.shape
    if rng is None:
        rng = np.random.RandomState(0)
    Omega = rng.randn(n, p)
    Y = A @ Omega
    for _ in range(q):
        Y = A.T @ Y
        Y = A @ Y
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ A
    Ub, s, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ Ub
    return U, s, Vt


# Algorithm registry
ALGORITHMS = {
    'Muon-Exact':    lambda **kw: MuonOptimizer(variant='exact', **kw),
    'Muon-RandSVD':  lambda **kw: MuonOptimizer(variant='randsvd', **kw),
    'Muon-Trunc':    lambda **kw: MuonOptimizer(variant='truncated', **kw),
    'SGD':           lambda **kw: SGDOptimizer(momentum=kw.pop('momentum', 0), **kw),
    'Momentum-SGD':  lambda **kw: MomentumSGDOptimizer(**kw),
    'Adam':          lambda **kw: AdamOptimizer(**kw),
    'RMSprop':       lambda **kw: RMSpropOptimizer(**kw),
}
