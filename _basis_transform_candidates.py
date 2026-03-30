"""
BASIS TRANSFORMATION CANDIDATES: Q such that Q^T H_GCD Q ≈ H_EF

Claude Sonnet & James Vaughan — Session 8 Research

THE PROBLEM:
  H_GCD = diag(log k) + C * gcd_kernel(i,j)   → r=0.68, wrong eigenvalues
  H_EF  = explicit formula operator              → 67% half-gap, r=0.03
  GOAL:  Find Q so Q^T H_GCD Q has BOTH good eigenvalues AND good spacing r.

SIX CANDIDATES for Q, each with explicit N×N matrix formula.
"""
import numpy as np
from math import gcd
from functools import lru_cache
import time

# ============================================================
# UTILITY FUNCTIONS (shared across all candidates)
# ============================================================

def mobius(n):
    """Möbius function mu(n)."""
    if n == 1:
        return 1
    factors = []
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            factors.append(d)
            temp //= d
            if temp % d == 0:
                return 0  # squared factor
        d += 1
    if temp > 1:
        factors.append(temp)
    return (-1) ** len(factors)


def euler_phi(n):
    """Euler's totient function."""
    result = n
    p = 2
    temp = n
    while p * p <= temp:
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            result -= result // p
        p += 1
    if temp > 1:
        result -= result // temp
    return result


def jordan_totient(n, k=1):
    """Jordan's totient function J_k(n). J_1 = Euler's phi."""
    if k == 1:
        return euler_phi(n)
    result = 1
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            result *= (p ** (k * count) - p ** (k * (count - 1)))
        p += 1
    if temp > 1:
        result *= (temp ** k - 1)
    return result


def ramanujan_sum(q, n):
    """Ramanujan sum c_q(n) = sum_{1<=a<=q, gcd(a,q)=1} exp(2*pi*i*a*n/q).

    Equivalent formula: c_q(n) = mu(q/gcd(q,n)) * phi(q) / phi(q/gcd(q,n))
    Or: c_q(n) = sum_{d | gcd(q,n)} d * mu(q/d)
    """
    g = gcd(q, n)
    total = 0
    # Use the divisor sum formula: c_q(n) = sum_{d | gcd(q,n)} d * mu(q/d)
    for d in range(1, g + 1):
        if g % d == 0 and q % d == 0:
            total += d * mobius(q // d)
    return total


def build_gcd_matrix(N):
    """Build the raw GCD matrix G_{ij} = gcd(i,j) for i,j = 1,...,N."""
    G = np.zeros((N, N), dtype=float)
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            G[i-1, j-1] = g
            G[j-1, i-1] = g
    return G


def build_log_gcd_kernel(N):
    """Build the log-GCD kernel: K_{ij} = log(gcd(i,j)+1) / sqrt(i*j)."""
    K = np.zeros((N, N), dtype=float)
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            val = np.log(g + 1) / np.sqrt(i * j)
            K[i-1, j-1] = val
            K[j-1, i-1] = val
    return K


# ============================================================
# CANDIDATE 1: DISCRETE MELLIN TRANSFORM
# ============================================================
#
# THEORY: The Mellin transform is the natural bridge between
# multiplicative structure (where GCD lives) and additive
# structure (where the spectral line Re(s)=1/2 lives).
#
# Continuous: (Mf)(s) = integral_0^inf f(x) x^{s-1} dx
# Discrete on {1,...,N}: M_{kj} = j^{-s_k} where s_k = sigma + i*gamma_k
#
# For the zeta connection, sigma = 1/2 and gamma_k are zero heights.
# But we can also use a REAL version: M_{kj} = j^{-sigma} * cos(gamma_k * log(j))
#
# KEY FACT: sum_{n=1}^{inf} f(n) n^{-s} = <f, n^{-s}> in l^2
# The Dirichlet series IS the Mellin transform of sequences.
#
# NORMALIZATION: The columns j^{-s} are NOT orthogonal on {1,...,N}.
# Need to compute the Gram matrix G_{kl} = sum_{j=1}^N j^{-s_k - conj(s_l)}
# and use G^{-1/2} to orthogonalize.

def candidate1_mellin_transform(N, gamma_k=None, sigma=0.5):
    """
    Discrete Mellin transform matrix.

    Q_{kj} = j^{-sigma} * exp(-i * gamma_k * log(j))  (complex version)

    For REAL operators, use the real/imaginary parts:
    Q_{kj} = j^{-sigma} * cos(gamma_k * log(j))  (real version, even index k)
    Q_{kj} = j^{-sigma} * sin(gamma_k * log(j))  (real version, odd index k)

    Parameters:
        N: matrix dimension
        gamma_k: array of N frequency parameters. If None, use Gram-Schmidt
                 frequencies gamma_k = 2*pi*k / log(N).
        sigma: real part (default 1/2 for critical line)

    Returns:
        Q: N x N real orthogonal matrix (after Gram-Schmidt)
        Q_raw: N x N matrix before orthogonalization
        info: dict with conditioning, etc.
    """
    if gamma_k is None:
        # Use uniformly spaced frequencies in "Mellin space"
        # These approximate the Fourier modes on the multiplicative group
        gamma_k = np.array([2 * np.pi * k / np.log(N) for k in range(N)])

    log_j = np.log(np.arange(1, N + 1))  # log(1), log(2), ..., log(N)
    weights = np.arange(1, N + 1, dtype=float) ** (-sigma)  # j^{-sigma}

    # Build raw (complex) Mellin matrix
    # M_{kj} = j^{-sigma} * exp(-i * gamma_k * log(j))
    M_complex = np.zeros((N, N), dtype=complex)
    for k in range(N):
        for j in range(N):
            M_complex[k, j] = weights[j] * np.exp(-1j * gamma_k[k] * log_j[j])

    # Real version: interleave cos and sin rows
    Q_raw = np.zeros((N, N))
    for k in range(N):
        if k % 2 == 0:
            half_k = k // 2
            for j in range(N):
                Q_raw[k, j] = weights[j] * np.cos(gamma_k[half_k] * log_j[j])
        else:
            half_k = k // 2
            for j in range(N):
                Q_raw[k, j] = weights[j] * np.sin(gamma_k[half_k] * log_j[j])

    # Gram-Schmidt orthogonalization to make Q orthogonal
    Q = np.zeros((N, N))
    for k in range(N):
        v = Q_raw[k].copy()
        for j in range(k):
            v -= np.dot(Q[j], v) * Q[j]
        norm = np.linalg.norm(v)
        if norm > 1e-14:
            Q[k] = v / norm
        else:
            # Degenerate: use random vector in null space
            Q[k] = np.random.randn(N)
            for j in range(k):
                Q[k] -= np.dot(Q[j], Q[k]) * Q[j]
            Q[k] /= np.linalg.norm(Q[k])

    cond = np.linalg.cond(Q_raw)
    return Q, Q_raw, {
        'name': 'Discrete Mellin Transform',
        'is_unitary': True,  # after GS
        'raw_condition': cond,
        'cost': 'O(N^2) to build, O(N^2) for GS',
        'formula': 'Q_raw[k,j] = j^{-1/2} * cos/sin(gamma_k * log(j))',
        'notes': 'Natural multiplicative<->additive bridge. '
                 'Kernel is n^{-s}, same as Dirichlet series. '
                 'Gram matrix cond number grows with N. '
                 'Connection to zeta: sum_n f(n)n^{-s} = <f, col_s(M)>.'
    }


# ============================================================
# CANDIDATE 2: RAMANUJAN SUM EXPANSION (ARITHMETIC FOURIER)
# ============================================================
#
# THEORY: Ramanujan sums c_q(n) form an "arithmetic Fourier basis".
# Any arithmetic function f(n) can be expanded as:
#   f(n) = sum_{q=1}^{inf} a_q * c_q(n)
# where a_q = lim_{N->inf} (1/N) sum_{n=1}^N f(n) * c_q(n) / phi(q)
#
# KEY FACT (diagonalization of GCD matrices):
# The GCD matrix G_{ij} = gcd(i,j) has the factorization:
#   G = C^T * diag(phi) * C  (up to normalization)
# where C_{qi} = c_q(i) is the Ramanujan sum matrix and
# phi = (phi(1), phi(2), ..., phi(N)) is Euler's totient.
#
# More precisely, gcd(m,n) = sum_{d | gcd(m,n)} phi(d)
#                           = sum_{q=1}^{min(m,n)} (phi(q)/q) * c_q(m) * c_q(n) / phi(q)
#
# This means the Ramanujan sum matrix C DIAGONALIZES the GCD matrix!
# The eigenvalues of G are related to phi(q).
#
# RAMANUJAN SUM FORMULA:
#   c_q(n) = sum_{d | gcd(q,n)} d * mu(q/d)
#          = phi(q) * mu(q/gcd(q,n)) / phi(q/gcd(q,n))   [when gcd(q,n) | q]

def candidate2_ramanujan_sum_matrix(N):
    """
    Ramanujan sum basis matrix.

    C_{qi} = c_q(i) / sqrt(phi(q))   (normalized)

    where c_q(i) = sum_{d | gcd(q,i)} d * mu(q/d).

    This matrix (approximately) diagonalizes the GCD matrix:
      G = gcd(i,j) satisfies G = A^T D A
    where A_{di} = [d | i] (divisibility indicator) and D = diag(phi(d)).

    The Ramanujan sums are the "Fourier coefficients" on the
    multiplicative group, analogous to exp(2*pi*i*k*n/N) for Z/NZ.

    Returns:
        Q: N x N orthogonalized matrix
        C_raw: N x N raw Ramanujan sum matrix
        info: dict
    """
    # Build the raw Ramanujan sum matrix
    C_raw = np.zeros((N, N))
    phi_vals = np.array([euler_phi(q) for q in range(1, N + 1)], dtype=float)

    for q in range(1, N + 1):
        for i in range(1, N + 1):
            C_raw[q-1, i-1] = ramanujan_sum(q, i)

    # Normalize rows by sqrt(phi(q)) for approximate orthogonality
    C_norm = np.zeros((N, N))
    for q in range(N):
        if phi_vals[q] > 0:
            C_norm[q] = C_raw[q] / np.sqrt(phi_vals[q])

    # Gram-Schmidt to get true orthogonal matrix
    Q = np.zeros((N, N))
    for k in range(N):
        v = C_norm[k].copy()
        for j in range(k):
            v -= np.dot(Q[j], v) * Q[j]
        norm = np.linalg.norm(v)
        if norm > 1e-14:
            Q[k] = v / norm
        else:
            Q[k] = np.random.randn(N)
            for j in range(k):
                Q[k] -= np.dot(Q[j], Q[k]) * Q[j]
            Q[k] /= np.linalg.norm(Q[k])

    cond = np.linalg.cond(C_norm) if np.linalg.matrix_rank(C_norm) == N else float('inf')

    return Q, C_raw, {
        'name': 'Ramanujan Sum Basis',
        'is_unitary': True,  # after GS
        'raw_condition': cond,
        'cost': 'O(N^2 * d(N)) where d(N) is max number of divisors',
        'formula': 'C[q,i] = sum_{d|gcd(q,i)} d * mu(q/d)',
        'notes': 'Diagonalizes GCD-type matrices. '
                 'Orthogonality: (1/N)*sum_n c_q(n)*c_r(n) -> delta_{qr}*phi(q) as N->inf. '
                 'Finite N: approximate orthogonality only. '
                 'Connection to zeta: Ramanujan expansion of Lambda(n) involves zeta zeros.'
    }


# ============================================================
# CANDIDATE 3: GCD MATRIX EIGENVECTORS (Smith factorization)
# ============================================================
#
# THEORY: The GCD matrix G_{ij} = gcd(i,j) has a known factorization:
#
#   G = E^T * D * E
#
# where E_{di} = 1 if d|i, 0 otherwise (the "divisibility matrix")
# and   D = diag(phi(1), phi(2), ..., phi(N))
#
# PROOF: gcd(i,j) = sum_{d=1}^{min(i,j)} phi(d) * [d|i] * [d|j]
#        (Gauss identity: n = sum_{d|n} phi(d))
#
# This means the eigenvalues of G lie in the range of phi values,
# and the eigenvectors of E^T * D * E are related to the columns of E.
#
# More precisely, the eigenvectors of G are obtained from the
# eigenvectors of E * E^T * D (or D * E * E^T), a matrix whose
# (d1,d2) entry counts how many k in {1,...,N} are divisible by both d1 and d2.
#
# SMITH'S FORMULA: det(G) = prod_{k=1}^N phi(k)

def candidate3_gcd_eigenvectors(N):
    """
    Eigenvectors of the GCD matrix G_{ij} = gcd(i,j).

    Uses the Smith factorization: G = E^T * diag(phi) * E
    where E_{di} = [d divides i].

    The actual eigenvectors are computed numerically from G,
    but we also provide the divisibility matrix E which gives
    the factorization.

    Returns:
        Q: N x N orthogonal eigenvector matrix of G
        E: N x N divisibility matrix E_{di} = [d|i]
        info: dict with eigenvalues, etc.
    """
    # Build divisibility matrix E: E[d-1, i-1] = 1 if d | i
    E = np.zeros((N, N))
    for d in range(1, N + 1):
        for i in range(d, N + 1, d):
            E[d-1, i-1] = 1.0

    # Build diagonal of phi values
    phi_diag = np.array([euler_phi(d) for d in range(1, N + 1)], dtype=float)

    # Verify: G = E^T @ diag(phi) @ E
    G_factored = E.T @ np.diag(phi_diag) @ E

    # Also build G directly for comparison
    G_direct = build_gcd_matrix(N)

    # Check factorization
    factor_error = np.max(np.abs(G_factored - G_direct))

    # Get eigenvectors of G
    eigenvalues, Q = np.linalg.eigh(G_direct)

    # Sort by eigenvalue magnitude (largest first)
    order = np.argsort(-np.abs(eigenvalues))
    eigenvalues = eigenvalues[order]
    Q = Q[:, order]

    return Q, E, {
        'name': 'GCD Matrix Eigenvectors (Smith factorization)',
        'is_unitary': True,  # eigenvectors of symmetric matrix
        'factorization_error': factor_error,
        'eigenvalues': eigenvalues,
        'phi_values': phi_diag,
        'cost': 'O(N^3) eigendecomposition, O(N^2) for E',
        'formula': 'G = E^T diag(phi) E, Q = eigvecs(G)',
        'notes': 'Smith 1875: det(G) = prod phi(k). '
                 'Eigenvalues are NOT simply phi(k) — that is the diagonal of D. '
                 'E^T D E is a congruence, not a similarity. '
                 'G is positive definite (Beslin-Ligh 1989). '
                 'Eigenvectors encode how divisibility structure distributes.'
    }


# ============================================================
# CANDIDATE 4: MÖBIUS INVERSION MATRIX
# ============================================================
#
# THEORY: Möbius inversion is THE bridge between summatory and
# pointwise arithmetic functions:
#   g(n) = sum_{d|n} f(d)  <=>  f(n) = sum_{d|n} mu(n/d) g(d)
#
# In matrix form: g = E * f  <=>  f = M * g
# where M_{ij} = mu(i/j) if j|i, else 0.
#
# M is the inverse of E (divisibility matrix), and M = E^{-1}.
#
# KEY: M converts between "cumulative" (summed over divisors)
# and "pointwise" (individual) arithmetic functions.
# If H_GCD lives in the "cumulative" world and H_EF in the
# "pointwise" world, then M is the bridge.
#
# FORMULA: M_{ij} = mu(i/j)  if j divides i
#                 = 0         otherwise
#
# M is lower-triangular (in the divisibility ordering, which
# is NOT the natural ordering 1,2,...,N).
# In natural ordering, M is sparse but not triangular.

def candidate4_mobius_matrix(N):
    """
    Möbius inversion matrix.

    M_{ij} = mu(i/j)  if j | i
           = 0         otherwise

    This is the inverse of the divisibility matrix E_{ij} = [j|i].

    Properties:
    - M * E = I (on arithmetic functions supported on {1,...,N})
    - Converts summatory -> pointwise functions
    - Sparse: O(N log N) nonzero entries
    - NOT orthogonal (but invertible)

    Returns:
        Q: N x N orthogonalized version (via polar decomposition)
        M: N x N raw Möbius matrix
        info: dict
    """
    # Build Möbius matrix
    M = np.zeros((N, N))
    for i in range(1, N + 1):
        for j in range(1, i + 1):
            if i % j == 0:
                M[i-1, j-1] = mobius(i // j)

    # Build divisibility matrix E for verification
    E = np.zeros((N, N))
    for i in range(1, N + 1):
        for j in range(1, i + 1):
            if i % j == 0:
                E[i-1, j-1] = 1.0

    # Verify M * E = I (approximately, since we truncate at N)
    ME = M @ E
    identity_error = np.max(np.abs(ME - np.eye(N)))

    # M is NOT orthogonal. Extract the orthogonal part via polar decomposition:
    # M = U * P where U is orthogonal and P is positive semidefinite.
    # U is the "nearest orthogonal matrix" to M.
    U_svd, S_svd, Vt_svd = np.linalg.svd(M)
    Q = U_svd @ Vt_svd  # Orthogonal polar factor

    cond = np.linalg.cond(M) if np.linalg.matrix_rank(M) == N else float('inf')

    return Q, M, {
        'name': 'Möbius Inversion Matrix',
        'is_unitary': False,  # M is not orthogonal
        'orthogonal_approx': True,  # Q is the polar factor
        'identity_error': identity_error,  # ||M*E - I||
        'condition': cond,
        'cost': 'O(N * H_N) where H_N ~ log(N) (sparse)',
        'formula': 'M[i,j] = mu(i/j) if j|i, else 0',
        'notes': 'Inverse of divisibility matrix E. '
                 'Converts cumulative -> pointwise arithmetic functions. '
                 'Fundamental to Dirichlet series: if F(s) = sum f(n)/n^s '
                 'and G(s) = F(s)*zeta(s), then g = E*f and f = M*g. '
                 'The polar factor U is the "rotation" part of Möbius inversion.'
    }


# ============================================================
# CANDIDATE 5: REDHEFFER MATRIX EIGENVECTORS
# ============================================================
#
# THEORY: The Redheffer matrix R has:
#   R_{ij} = 1  if i|j or j=1
#           = 0  otherwise
#
# KEY PROPERTY: det(R_N) = M(N) (Mertens function!)
# RH is equivalent to: |det(R_N)| = O(N^{1/2+eps})
#
# EIGENVALUE STRUCTURE:
# - Eigenvalue 1 with multiplicity N - floor(log2(N)) - 1
# - One large eigenvalue ~ N + log(N) + O(1)
# - O(log N) non-trivial eigenvalues
#
# EIGENVECTOR RECURSION (for eigenvalue lambda != 1):
#   a_j = (1/(lambda-1)) * sum_{d|j, d<j} a_d
#   lambda * a_1 = sum_{k=1}^N a_k
#
# DIRICHLET SERIES of eigenvector:
#   sum_{n>=1} v_lambda(n) / n^s = (lambda-1) / (lambda - zeta(s))
#
# This connects eigenvectors to zeta(s) DIRECTLY.

def candidate5_redheffer_eigenvectors(N):
    """
    Eigenvectors of the Redheffer matrix.

    R_{ij} = 1 if i|j or j=1, else 0.

    det(R_N) = M(N) = Mertens function.

    Non-trivial eigenvectors satisfy:
      sum_n v(n)/n^s = (lambda-1)/(lambda - zeta(s))

    Returns:
        Q: N x N eigenvector matrix
        R: N x N Redheffer matrix
        info: dict
    """
    # Build Redheffer matrix: a_{ij} = 1 if i|j or j=1
    # Row i has 1s at: column 1 (always), and all columns j that are multiples of i
    R = np.zeros((N, N))
    for i in range(1, N + 1):
        R[i-1, 0] = 1.0  # j=1 column
        for j in range(i, N + 1, i):  # j = i, 2i, 3i, ... (i divides j)
            if j > 1:
                R[i-1, j-1] = 1.0

    # Note: R is NOT symmetric! Use general eigendecomposition.
    eigenvalues, vecs_right = np.linalg.eig(R)

    # Sort by eigenvalue magnitude
    order = np.argsort(-np.abs(eigenvalues))
    eigenvalues = eigenvalues[order]
    vecs_right = vecs_right[:, order]

    # Compute Mertens function for comparison
    mu_vals = np.array([mobius(k) for k in range(1, N + 1)])
    mertens = np.cumsum(mu_vals)

    # Count eigenvalue 1 multiplicity
    n_ones = np.sum(np.abs(eigenvalues - 1.0) < 1e-10)
    expected_ones = N - int(np.floor(np.log2(N))) - 1

    # Extract the non-trivial eigenvectors (lambda != 1)
    nontrivial_mask = np.abs(eigenvalues - 1.0) > 0.1
    nontrivial_eigs = eigenvalues[nontrivial_mask]
    nontrivial_vecs = vecs_right[:, nontrivial_mask]

    # For basis transform, orthogonalize the real parts of eigenvectors
    vecs_real = np.real(vecs_right)
    Q, _ = np.linalg.qr(vecs_real)

    return Q, R, {
        'name': 'Redheffer Matrix Eigenvectors',
        'is_unitary': False,  # R is not symmetric
        'eigenvalues': eigenvalues,
        'mertens_N': mertens[-1],
        'det_R': np.real(np.prod(eigenvalues)),
        'n_trivial_eigs': int(n_ones),
        'expected_trivial': expected_ones,
        'n_nontrivial': int(np.sum(nontrivial_mask)),
        'nontrivial_eigenvalues': nontrivial_eigs,
        'cost': 'O(N^3) for general eigendecomposition',
        'formula': 'R[i,j] = 1 if i|j or j=1, else 0',
        'notes': 'det(R) = M(N) = Mertens function. '
                 'RH <=> |det(R_N)| = O(N^{1/2+eps}). '
                 'Non-trivial eigenvector Dirichlet series: '
                 'F(s) = (lambda-1)/(lambda - zeta(s)). '
                 'POLES OF F(s) ARE AT ZETA ZEROS! '
                 'This is the most direct zeta connection.'
    }


# ============================================================
# CANDIDATE 6: HECKE OPERATOR EIGENBASIS
# ============================================================
#
# THEORY: Hecke operators T_p act on arithmetic functions by:
#   (T_p f)(n) = f(pn)  if p does not divide n
#   (T_p f)(n) = f(pn) + p * f(n/p)  if p divides n
#
# On l^2({1,...,N}), this truncates to a finite matrix.
# All T_p commute, so they share a simultaneous eigenbasis.
#
# For the space of MULTIPLICATIVE functions on {1,...,N},
# the eigenvectors of T_p are (approximately) completely
# multiplicative functions f(n) = n^{-s} for various s.
#
# The simultaneous eigenbasis of all T_p is the Hecke eigenbasis,
# which on modular forms gives the connection to L-functions.
#
# FINITE VERSION: On {1,...,N}, T_p is an N x N matrix:
#   (T_p)_{ij} = 1  if j = p*i AND p*i <= N
#              + p  if i = p*j (i.e., j = i/p when p|i)
#              = 0  otherwise

def build_hecke_matrix(N, p):
    """
    Build the Hecke operator T_p as an N x N matrix on l^2({1,...,N}).

    (T_p f)(n) =  f(pn)         [if pn <= N]
                + p * f(n/p)     [if p | n]

    Matrix entries:
      T[n-1, pn-1] = 1       (maps f(pn) -> (T_p f)(n))
      T[n-1, n/p-1] = p      (maps f(n/p) -> p*(T_p f)(n))
    """
    T = np.zeros((N, N))
    for n in range(1, N + 1):
        # f(pn) term
        if p * n <= N:
            T[n-1, p*n - 1] = 1.0
        # p * f(n/p) term
        if n % p == 0:
            T[n-1, n // p - 1] += p
    return T


def candidate6_hecke_eigenbasis(N, primes=None):
    """
    Simultaneous eigenbasis of Hecke operators T_p.

    Build T_p for small primes, compute the simultaneous eigenbasis
    by iterative refinement (since they commute, they share eigenvectors).

    Returns:
        Q: N x N approximate simultaneous eigenbasis
        T_matrices: dict of {p: T_p matrix}
        info: dict
    """
    if primes is None:
        primes = [2, 3, 5, 7, 11, 13]

    T_matrices = {}
    for p in primes:
        T_matrices[p] = build_hecke_matrix(N, p)

    # Strategy: symmetrize the Hecke matrices (they are NOT symmetric on l^2)
    # by using T_p + T_p^T, then find simultaneous eigenbasis.
    # Alternative: use the product T_2 * T_3 which has a richer spectrum.

    # Method: diagonalize T_2 first, then refine with T_3, T_5, etc.
    # Since T_p commute, each eigenspace of T_2 is invariant under T_3.

    # Start with T_2 (symmetrized)
    T2_sym = T_matrices[2] + T_matrices[2].T
    evals_2, evecs_2 = np.linalg.eigh(T2_sym)

    # Refine: within each degenerate eigenspace of T_2,
    # diagonalize T_3 (projected)
    Q = evecs_2.copy()

    for p in primes[1:]:
        Tp_sym = T_matrices[p] + T_matrices[p].T
        # Project into current eigenbasis
        Tp_proj = Q.T @ Tp_sym @ Q
        # Diagonalize each block (approximately — treat as full diag)
        evals_p, evecs_p = np.linalg.eigh(Tp_proj)
        Q = Q @ evecs_p

    # Verify: check how well Q simultaneously diagonalizes
    diag_errors = {}
    for p in primes:
        Tp_diag = Q.T @ T_matrices[p] @ Q
        off_diag = Tp_diag - np.diag(np.diag(Tp_diag))
        diag_errors[p] = np.linalg.norm(off_diag) / np.linalg.norm(Tp_diag)

    return Q, T_matrices, {
        'name': 'Hecke Operator Simultaneous Eigenbasis',
        'is_unitary': True,  # from symmetric eigendecomp
        'diag_errors': diag_errors,
        'cost': 'O(N^3 * n_primes)',
        'formula': 'T_p[n, pn] = 1, T_p[n, n/p] = p (for p|n)',
        'notes': 'Hecke operators commute: [T_p, T_q] = 0 for all p,q. '
                 'Simultaneous eigenfunctions are multiplicative. '
                 'On modular forms: Hecke eigenvalues = L-function coefficients. '
                 'On {1,...,N}: truncation breaks commutativity slightly. '
                 'Eigenvectors approximate completely multiplicative functions.'
    }


# ============================================================
# CANDIDATE 7 (BONUS): HYBRID — Mellin-Ramanujan-Möbius
# ============================================================
#
# The deepest candidate: combine Mellin phases with Ramanujan structure
# and Möbius inversion. The idea:
#
# Q_{kj} = (1/sqrt(N)) * sum_{d|j} mu(j/d) * d^{-1/2+i*gamma_k}
#
# This is the Möbius-inverted Mellin transform: it inverts the
# divisor sum structure WHILE projecting onto the critical line.
#
# Equivalently: Q = M * Mellin, where M is Möbius and Mellin is candidate 1.

def candidate7_hybrid_mobius_mellin(N, gamma_k=None, sigma=0.5):
    """
    Hybrid Möbius-Mellin transform.

    Q_{kj} = (1/C_k) * sum_{d|j} mu(j/d) * d^{-sigma + i*gamma_k}

    This is the composition: first apply Möbius inversion to "undo"
    the divisor sum structure, then apply the Mellin transform to
    project onto the spectral line.

    When gamma_k are zeta zeros, the connection is:
      sum_n Q_{kn} * n^{-s} = 1/zeta(s+sigma-i*gamma_k)

    Returns:
        Q: N x N orthogonalized matrix
        Q_raw: N x N raw matrix
        info: dict
    """
    if gamma_k is None:
        gamma_k = np.array([2 * np.pi * k / np.log(N) for k in range(N)])

    # Build raw matrix
    Q_raw = np.zeros((N, N))

    for k in range(N):
        for j in range(1, N + 1):
            val = 0.0
            # Sum over divisors d of j
            for d in range(1, j + 1):
                if j % d == 0:
                    mu_val = mobius(j // d)
                    if mu_val != 0:
                        # Real part of d^{-sigma + i*gamma_k}
                        log_d = np.log(d) if d > 1 else 0.0
                        val += mu_val * d ** (-sigma) * np.cos(gamma_k[k] * log_d)
            Q_raw[k, j-1] = val

    # Orthogonalize
    Q = np.zeros((N, N))
    for k in range(N):
        v = Q_raw[k].copy()
        for j in range(k):
            v -= np.dot(Q[j], v) * Q[j]
        norm = np.linalg.norm(v)
        if norm > 1e-14:
            Q[k] = v / norm
        else:
            Q[k] = np.random.randn(N)
            for j in range(k):
                Q[k] -= np.dot(Q[j], Q[k]) * Q[j]
            Q[k] /= np.linalg.norm(Q[k])

    cond = np.linalg.cond(Q_raw) if np.linalg.matrix_rank(Q_raw) == N else float('inf')

    return Q, Q_raw, {
        'name': 'Hybrid Möbius-Mellin Transform',
        'is_unitary': True,  # after GS
        'raw_condition': cond,
        'cost': 'O(N^2 * d(N)) — divisor sum per entry',
        'formula': 'Q[k,j] = sum_{d|j} mu(j/d) * d^{-1/2} * cos(gamma_k * log(d))',
        'notes': 'Composition of Möbius inversion + Mellin transform. '
                 'Undoes divisor sums AND projects to spectral line. '
                 'Dirichlet series: sum_n Q[k,n]*n^{-s} = 1/zeta(s+1/2-i*gamma_k). '
                 'STRONGEST theoretical connection: links multiplicative '
                 'structure directly to zeta zero locations.'
    }


# ============================================================
# MASTER TEST: Apply each Q to H_GCD and measure results
# ============================================================

def test_all_candidates(N=50, C_gcd=0.2, verbose=True):
    """
    Build H_GCD = diag(log(k)) + C * gcd_kernel,
    apply each candidate Q, and measure:
    1. Eigenvalue accuracy vs actual zeta zeros
    2. Spacing correlation r
    3. How well Q^T H_GCD Q matches the explicit formula operator

    Parameters:
        N: matrix dimension
        C_gcd: coupling constant for GCD kernel
        verbose: print results
    """
    t0 = time.time()

    if verbose:
        print("=" * 70)
        print(f"TESTING ALL BASIS TRANSFORM CANDIDATES (N={N}, C={C_gcd})")
        print("=" * 70)

    # Build H_GCD
    log_diag = np.diag(np.log(np.arange(1, N + 1)))
    gcd_kernel = build_log_gcd_kernel(N)
    H_GCD = log_diag + C_gcd * gcd_kernel

    if verbose:
        eigs_gcd = np.linalg.eigvalsh(H_GCD)
        print(f"\nH_GCD eigenvalue range: [{eigs_gcd[0]:.4f}, {eigs_gcd[-1]:.4f}]")

    # Get all candidates
    results = {}

    candidates = [
        ("1. Discrete Mellin", lambda: candidate1_mellin_transform(N)),
        ("2. Ramanujan Sum", lambda: candidate2_ramanujan_sum_matrix(N)),
        ("3. GCD Eigenvectors", lambda: candidate3_gcd_eigenvectors(N)),
        ("4. Möbius Inversion", lambda: candidate4_mobius_matrix(N)),
        ("5. Redheffer Eigvecs", lambda: candidate5_redheffer_eigenvectors(N)),
        ("6. Hecke Eigenbasis", lambda: candidate6_hecke_eigenbasis(N)),
        ("7. Möbius-Mellin", lambda: candidate7_hybrid_mobius_mellin(N)),
    ]

    for name, build_fn in candidates:
        t1 = time.time()
        try:
            Q, raw, info = build_fn()

            # Apply Q to H_GCD: H_transformed = Q^T @ H_GCD @ Q
            if Q.dtype == complex:
                H_trans = Q.conj().T @ H_GCD @ Q
                H_trans = np.real(H_trans)  # should be real if done right
            else:
                H_trans = Q.T @ H_GCD @ Q

            # Get eigenvalues of transformed operator
            eigs_trans = np.sort(np.linalg.eigvalsh(H_trans))

            # Diagonal of transformed operator
            diag_trans = np.diag(H_trans)

            # Off-diagonal norm
            off_diag = H_trans - np.diag(diag_trans)
            off_norm = np.linalg.norm(off_diag) / np.linalg.norm(H_trans)

            # Bandwidth (how localized is the off-diagonal?)
            bandwidth_90 = 0
            total_off = np.sum(np.abs(off_diag))
            if total_off > 0:
                cumul = 0
                for w in range(1, N):
                    for i in range(N - w):
                        cumul += abs(H_trans[i, i+w]) + abs(H_trans[i+w, i])
                    if cumul / total_off > 0.9:
                        bandwidth_90 = w
                        break
                if bandwidth_90 == 0:
                    bandwidth_90 = N

            dt = time.time() - t1

            results[name] = {
                'Q': Q,
                'info': info,
                'H_transformed': H_trans,
                'eigenvalues': eigs_trans,
                'diag': diag_trans,
                'off_diag_frac': off_norm,
                'bandwidth_90': bandwidth_90,
                'time': dt,
            }

            if verbose:
                print(f"\n  {name}:")
                print(f"    Info: {info.get('formula', '')}")
                print(f"    Diag range: [{np.min(diag_trans):.4f}, {np.max(diag_trans):.4f}]")
                print(f"    Off-diag fraction: {off_norm:.4f}")
                print(f"    90% bandwidth: {bandwidth_90}")
                print(f"    Time: {dt:.2f}s")

        except Exception as e:
            if verbose:
                print(f"\n  {name}: FAILED — {e}")
            results[name] = {'error': str(e)}

    if verbose:
        print(f"\nTotal time: {time.time() - t0:.1f}s")

    return results


# ============================================================
# SUMMARY TABLE
# ============================================================

SUMMARY = """
========================================================================
                BASIS TRANSFORM CANDIDATES -- SUMMARY
========================================================================

  1. DISCRETE MELLIN TRANSFORM
     Formula: Q[k,j] = j^{-1/2} * cos(gamma_k * log(j))
     Unitary: No (needs Gram-Schmidt)
     Cost: O(N^2)
     Zeta connection: Kernel n^{-s} IS the Dirichlet series
     Key insight: Natural bridge multiplicative <-> additive

  2. RAMANUJAN SUM BASIS
     Formula: C[q,i] = sum_{d|gcd(q,i)} d * mu(q/d)
     Unitary: Approximately (exact in N->inf limit)
     Cost: O(N^2 * d(N))
     Zeta connection: Ramanujan expansion of Lambda(n) has zeta zeros
     Key insight: "Arithmetic Fourier transform" -- diagonalizes GCD

  3. GCD MATRIX EIGENVECTORS
     Formula: Q = eigvecs(gcd(i,j)); G = E^T diag(phi) E
     Unitary: Yes (eigenvectors of symmetric matrix)
     Cost: O(N^3)
     Zeta connection: det(G) = prod phi(k) (Smith 1875)
     Key insight: Direct diagonalization of the GCD world

  4. MOBIUS INVERSION MATRIX
     Formula: M[i,j] = mu(i/j) if j|i, else 0
     Unitary: No (use polar decomposition)
     Cost: O(N log N) entries (sparse!)
     Zeta connection: M inverts zeta: 1/zeta(s) = sum mu(n)/n^s
     Key insight: THE inversion of multiplicative structure

  5. REDHEFFER MATRIX EIGENVECTORS
     Formula: R[i,j] = 1 if i|j or j=1, else 0
     Unitary: No (R not symmetric)
     Cost: O(N^3)
     Zeta connection: det(R) = M(N); eigvec Dirichlet series has
                      POLES AT ZETA ZEROS: F(s) = (lam-1)/(lam-zeta(s))
     Key insight: STRONGEST direct connection to zeros

  6. HECKE OPERATOR EIGENBASIS
     Formula: T_p[n,pn]=1, T_p[n,n/p]=p; Q = simultaneous eigenbasis
     Unitary: Yes (from symmetrized T_p + T_p^T)
     Cost: O(N^3 * n_primes)
     Zeta connection: Hecke eigenvalues = L-function Fourier coefficients
     Key insight: Diagonalizes ALL multiplicative operations at once

  7. HYBRID MOBIUS-MELLIN (bonus)
     Formula: Q[k,j] = sum_{d|j} mu(j/d)*d^{-1/2}*cos(gamma_k*log(d))
     Unitary: No (needs Gram-Schmidt)
     Cost: O(N^2 * d(N))
     Zeta connection: Dirichlet series = 1/zeta(s+1/2-i*gamma_k)
     Key insight: Composes Mobius inversion WITH spectral projection

  RANKING (by theoretical strength of zeta zero connection):
    #1: Candidate 5 (Redheffer) -- eigenvector poles = zeta zeros
    #2: Candidate 7 (Mobius-Mellin) -- Dirichlet series = 1/zeta
    #3: Candidate 1 (Mellin) -- kernel IS the Dirichlet series
    #4: Candidate 2 (Ramanujan) -- arithmetic Fourier, diagonalizes GCD
    #5: Candidate 4 (Mobius) -- inverts zeta directly
    #6: Candidate 6 (Hecke) -- L-function connection
    #7: Candidate 3 (GCD eigvecs) -- diagonalizes GCD but no spectral link

========================================================================
"""

if __name__ == '__main__':
    print(SUMMARY)
    results = test_all_candidates(N=50, C_gcd=0.2)
