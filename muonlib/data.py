"""Data generation for matrix sensing and matrix factorization experiments."""
import numpy as np

def generate_target_matrix(d, r=None, spectrum='hard-cutoff', kappa=1.0, seed=None):
    """Generate a rank-r target matrix X* with specified spectral profile.
    
    spectrum types:
    - 'hard-cutoff': top r singular values = 1, rest = 0
    - 'poly-alpha': sv_i = i^{-alpha}
    - 'exp-beta': sv_i = exp(-beta * i)
    """
    rng = np.random.RandomState(seed)
    if r is None:
        r = d
    
    U, _ = np.linalg.qr(rng.randn(d, d))
    V, _ = np.linalg.qr(rng.randn(d, d))
    
    s = np.zeros(d)
    if spectrum == 'hard-cutoff':
        s[:r] = 1.0
    elif spectrum == 'poly-alpha':
        alpha = 1.0
        for i in range(r):
            s[i] = (i + 1) ** (-alpha)
        s /= s[0]
    elif spectrum == 'exp-beta':
        beta = 0.5
        for i in range(r):
            s[i] = np.exp(-beta * i)
    else:
        s[:r] = 1.0
    
    # Apply condition number scaling
    if kappa > 1.0 and r > 1:
        s[:r] = np.linspace(1.0, 1.0 / kappa, r)
    
    X = U @ np.diag(s) @ V.T
    return X

def generate_rectangular_target(m, n, r=None, spectrum='hard-cutoff', seed=None):
    """Generate rectangular target matrix (m x n)."""
    rng = np.random.RandomState(seed)
    if r is None:
        r = min(m, n) // 10
    
    U, _ = np.linalg.qr(rng.randn(m, m))
    V, _ = np.linalg.qr(rng.randn(n, n))
    
    s = np.zeros(min(m, n))
    s[:r] = 1.0
    S = np.zeros((m, n))
    np.fill_diagonal(S, s)
    
    return U @ S @ V.T

def generate_measurement_matrices(d, m_meas, dist='normal', sparsity=0.1, seed=None):
    """Generate measurement matrices A_i for matrix sensing.
    
    dist types: 'normal', 'rademacher', 'sparse', 'sphere', 'fjl'
    """
    rng = np.random.RandomState(seed)
    matrices = []
    
    for _ in range(m_meas):
        if dist == 'normal':
            A = rng.randn(d, d)
        elif dist == 'rademacher':
            A = rng.choice([-1, 1], size=(d, d)).astype(float)
        elif dist == 'sparse':
            A = rng.randn(d, d) * (rng.rand(d, d) < sparsity) / np.sqrt(sparsity * d * d)
        elif dist == 'sphere':
            A = rng.randn(d, d)
            A /= np.linalg.norm(A, 'fro')
        elif dist == 'fjl':
            A = rng.choice([-1, 1], size=(d, d)).astype(float) / np.sqrt(d)
        elif dist == 'uniform':
            A = rng.uniform(-1, 1, size=(d, d))
        elif dist == 'sparse_normal':
            A = rng.randn(d, d) * (rng.rand(d, d) < sparsity) / np.sqrt(sparsity * d * d)
        else:
            A = rng.randn(d, d)
        matrices.append(A)
    
    return matrices

def generate_rectangular_measurement_matrices(m, n, m_meas, dist='normal', seed=None):
    """Generate rectangular measurement matrices (m x n)."""
    rng = np.random.RandomState(seed)
    matrices = []
    
    for _ in range(m_meas):
        if dist == 'normal':
            A = rng.randn(m, n)
        elif dist == 'rademacher':
            A = rng.choice([-1, 1], size=(m, n)).astype(float)
        else:
            A = rng.randn(m, n)
        matrices.append(A)
    
    return matrices

def generate_matrix_sensing(d, r=None, m_meas=None, noise_std=0.0, 
                             dist='normal', kappa=1.0, seed=None):
    """Generate full matrix sensing problem instance.
    
    Returns dict with: X_star, A_matrices, y, noise_std, d, r, m_meas
    """
    if m_meas is None:
        m_meas = 3 * d * d  # 3x oversampling
    if r is None:
        r = d // 10
    
    rng = np.random.RandomState(seed)
    X_star = generate_target_matrix(d, r, kappa=kappa, seed=rng.randint(0, 2**31))
    A_matrices = generate_measurement_matrices(d, m_meas, dist=dist, 
                                                seed=rng.randint(0, 2**31))
    
    y = np.array([np.trace(A.T @ X_star) for A in A_matrices])
    
    if noise_std > 0:
        noise = rng.randn(m_meas) * noise_std
        y += noise
    
    return {
        'X_star': X_star, 'A_matrices': A_matrices, 'y': y,
        'noise_std': noise_std, 'd': d, 'r': r, 'm_meas': m_meas
    }

def compute_gradient_matrix_sensing(X, A_matrices, y):
    """Compute gradient for matrix sensing problem.
    
    f(X) = 1/(2m) * sum_i (tr(A_i^T X) - y_i)^2
    grad = 1/m * sum_i (tr(A_i^T X) - y_i) * A_i
    """
    m = len(A_matrices)
    grad = np.zeros_like(X)
    for A, yi in zip(A_matrices, y):
        residual = np.trace(A.T @ X) - yi
        grad += residual * A
    return grad / m

def compute_loss_matrix_sensing(X, A_matrices, y):
    """Compute MSE loss for matrix sensing."""
    m = len(A_matrices)
    loss = 0.0
    for A, yi in zip(A_matrices, y):
        residual = np.trace(A.T @ X) - yi
        loss += residual**2
    return loss / (2 * m)

def generate_matrix_factorization(d, L, r=None, seed=None):
    """Generate matrix factorization problem (deep matrix completion).
    
    f(W_1,...,W_L) = 1/2 ||W_L...W_1 - X*||_F^2
    
    Returns: X_star, L, d, r
    """
    rng = np.random.RandomState(seed)
    if r is None:
        r = d // 10
    
    X_star = generate_target_matrix(d, r, seed=rng.randint(0, 2**31))
    return {'X_star': X_star, 'L': L, 'd': d, 'r': r}

def compute_loss_mf(W_list, X_star):
    """Compute loss for matrix factorization."""
    W = W_list[0]
    for Wi in W_list[1:]:
        W = Wi @ W
    return 0.5 * np.linalg.norm(W - X_star, 'fro')**2

def compute_gradient_mf(W_list, X_star):
    """Compute gradient for MF with respect to each W_i.
    
    grad_i = W_{i+1:}^T (W_{L:1} - X*) W_{:i-1}^T
    """
    L = len(W_list)
    # Forward: W_{L:1}
    W_forward = W_list[0]
    for Wi in W_list[1:]:
        W_forward = Wi @ W_forward
    
    residual = W_forward - X_star
    grads = []
    
    for i in range(L):
        # Left factor: W_{L:i+2}^T * residual
        if i < L - 1:
            W_left = W_list[-1]
            for j in range(L-2, i, -1):
                W_left = W_left @ W_list[j]
            grad_i = W_left.T @ residual
        else:
            grad_i = residual
        
        # Right factor: W_{i-1:1}^T
        if i > 0:
            W_right = W_list[i-1]
            for j in range(i-2, -1, -1):
                W_right = W_right @ W_list[j]
            grad_i = grad_i @ W_right.T
        
        grads.append(grad_i)
    
    return grads
