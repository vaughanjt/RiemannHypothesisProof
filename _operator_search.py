"""Search for operators with eigenvector-eigenvalue coupling beta > 1.

The zeta function shows |Z(mid)| ~ gap^beta with beta ~ 2, while GUE gives
beta ~ 0. What CLASS of operators produces this growing coupling?

Candidates:
1. Arithmetic operators (Hecke, Bost-Connes): eigenvectors built from primes
2. Multiplicative operators: diagonal in some arithmetic basis
3. Schrodinger operators with arithmetic potentials
4. Transfer operators of arithmetic dynamical systems
5. Combinations of the above

For each candidate, we:
- Construct the operator at various sizes N
- Compute eigenvalues and eigenvectors
- Evaluate the "characteristic polynomial" at eigenvalue midpoints
- Measure the peak-gap correlation r and power law beta
- Compare to zeta (r ~ 0.8, beta ~ 2) and GUE (r ~ 0.04, beta ~ 0)
"""
import sys, time
sys.path.insert(0, 'src')
import numpy as np
from scipy.stats import pearsonr
from riemann.analysis.bost_connes_operator import polynomial_unfold

t0 = time.time()


def measure_peak_gap(eigenvalues, eigenvectors=None, H=None):
    """Measure peak-gap correlation for an operator.

    Computes |det(z - H)| at midpoints between consecutive eigenvalues,
    or equivalently prod_j |z - lambda_j|, then correlates with gaps.

    Returns: r (Pearson), beta (power law exponent), n_points.
    """
    eigs = np.sort(eigenvalues)
    sp = polynomial_unfold(eigs, trim_fraction=0.1)
    if len(sp) < 20:
        return 0, 0, 0
    sp = sp / np.mean(sp)

    # Trim eigenvalues to match
    n_trim = int(0.1 * len(eigs))
    eigs_trim = eigs[n_trim:-n_trim]

    # |det(z_mid - H)| = prod |z_mid - lambda_j|
    log_peaks = []
    gaps = []
    for k in range(len(sp)):
        if k + 1 >= len(eigs_trim):
            break
        z_mid = (eigs_trim[k] + eigs_trim[k + 1]) / 2
        log_det = np.sum(np.log(np.abs(z_mid - eigs) + 1e-30))
        log_peaks.append(log_det)
        gaps.append(sp[k])

    if len(gaps) < 20:
        return 0, 0, 0

    gaps = np.array(gaps)
    log_peaks = np.array(log_peaks)

    r, p = pearsonr(gaps, log_peaks)
    # Power law: log|peak| = beta * log(gap) + const
    mask = gaps > 0.1  # avoid log(0)
    if np.sum(mask) > 10:
        beta = np.polyfit(np.log(gaps[mask]), log_peaks[mask], 1)[0]
    else:
        beta = 0
    return r, beta, len(gaps)


rng = np.random.default_rng(42)

# ============================================================
# BASELINE: GUE
# ============================================================
print('=' * 70)
print('OPERATOR ZOO: peak-gap correlation r and power law beta')
print('=' * 70)

print(f'\n{"Operator":<45} {"N":>5} {"r":>8} {"beta":>8} {"pts":>5}')
print('-' * 75)

# GUE
gue_rs, gue_betas = [], []
for _ in range(50):
    A = rng.standard_normal((200, 200)) + 1j * rng.standard_normal((200, 200))
    H = (A + A.conj().T) / (2 * np.sqrt(400))
    eigs = np.linalg.eigvalsh(H)
    r, beta, n = measure_peak_gap(eigs)
    gue_rs.append(r)
    gue_betas.append(beta)
print(f'{"GUE (50 trials, N=200)":<45} {200:>5} {np.mean(gue_rs):>+8.4f} {np.mean(gue_betas):>8.3f} {"~160":>5}')

# ============================================================
# CANDIDATE 1: Diagonal + arithmetic off-diagonal
# ============================================================
# H = diag(log(n)) + epsilon * A_prime
# where A_prime has entries related to primes

for N in [200, 400]:
    # Pure diagonal: log(n) eigenvalues
    H_diag = np.diag(np.log(np.arange(1, N + 1, dtype=float)))
    eigs = np.diag(H_diag)
    r, beta, n = measure_peak_gap(eigs)
    print(f'{"diag(log n), N=" + str(N):<45} {N:>5} {r:>+8.4f} {beta:>8.3f} {n:>5}')

# ============================================================
# CANDIDATE 2: Hecke Prime Adjacency (from bost_connes_operator.py)
# ============================================================
from riemann.analysis.bost_connes_operator import (
    construct_hecke_prime_adjacency,
    construct_bc_hamiltonian,
    construct_divisor_operator,
)

for N in [200, 400]:
    H = construct_hecke_prime_adjacency(N)
    eigs = np.linalg.eigvalsh(H)
    r, beta, n = measure_peak_gap(eigs)
    print(f'{"Hecke Prime Adjacency, N=" + str(N):<45} {N:>5} {r:>+8.4f} {beta:>8.3f} {n:>5}')

# ============================================================
# CANDIDATE 3: Bost-Connes Hamiltonian
# ============================================================
for N, alpha in [(200, 0.5), (200, 1.0), (400, 1.0)]:
    H = construct_bc_hamiltonian(N, alpha=alpha)
    eigs = np.linalg.eigvalsh(H)
    r, beta, n = measure_peak_gap(eigs)
    print(f'{"BC Hamiltonian a=" + str(alpha) + " N=" + str(N):<45} {N:>5} {r:>+8.4f} {beta:>8.3f} {n:>5}')

# ============================================================
# CANDIDATE 4: Divisor operator
# ============================================================
for N in [200, 400]:
    H = construct_divisor_operator(N)
    eigs = np.linalg.eigvalsh(H)
    r, beta, n = measure_peak_gap(eigs)
    print(f'{"Divisor operator, N=" + str(N):<45} {N:>5} {r:>+8.4f} {beta:>8.3f} {n:>5}')

# ============================================================
# CANDIDATE 5: Redheffer matrix (Mertens function connection)
# ============================================================
for N in [200, 400]:
    R = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if (j + 1) % (i + 1) == 0 or i == 0:
                R[i, j] = 1
    # Symmetrize
    H = (R + R.T) / 2
    eigs = np.linalg.eigvalsh(H)
    r, beta, n = measure_peak_gap(eigs)
    print(f'{"Redheffer (sym), N=" + str(N):<45} {N:>5} {r:>+8.4f} {beta:>8.3f} {n:>5}')

# ============================================================
# CANDIDATE 6: GUE + diagonal arithmetic perturbation
# ============================================================
for eps in [0.01, 0.1, 0.5, 1.0, 5.0]:
    N = 200
    A = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    H_gue = (A + A.conj().T) / (2 * np.sqrt(2 * N))
    V = np.diag(np.log(np.arange(1, N + 1, dtype=float)))
    V = V / np.max(np.abs(np.diag(V)))  # normalize
    H = H_gue + eps * V
    eigs = np.linalg.eigvalsh(H)
    r, beta, n = measure_peak_gap(eigs)
    print(f'{"GUE + eps*diag(log n), eps=" + str(eps):<45} {N:>5} {r:>+8.4f} {beta:>8.3f} {n:>5}')

# ============================================================
# CANDIDATE 7: Schrodinger with prime potential
# ============================================================
for N in [200, 400]:
    # -d^2/dx^2 + V(x) where V(n) = sum_{p|n} log(p)
    H = np.zeros((N, N))
    # Kinetic: tridiagonal -1, 2, -1
    for i in range(N):
        H[i, i] = 2
        if i > 0: H[i, i-1] = -1
        if i < N-1: H[i, i+1] = -1
    # Potential: von Mangoldt function
    from sympy import factorint
    for n in range(1, N + 1):
        factors = factorint(n)
        if len(factors) == 1:
            p, k = list(factors.items())[0]
            H[n-1, n-1] += np.log(p)  # Lambda(n) = log(p) if n = p^k
    eigs = np.linalg.eigvalsh(H)
    r, beta, n_pts = measure_peak_gap(eigs)
    print(f'{"Schrodinger + Lambda(n), N=" + str(N):<45} {N:>5} {r:>+8.4f} {beta:>8.3f} {n_pts:>5}')

# ============================================================
# CANDIDATE 8: Multiplicative arithmetic in Fourier space
# ============================================================
# The DFT of a multiplicative function creates an operator whose
# eigenvectors are arithmetically structured.
for N in [200, 400]:
    # Build: H_{jk} = f(gcd(j,k)) for some arithmetic f
    H = np.zeros((N, N))
    for j in range(1, N + 1):
        for k in range(1, N + 1):
            g = np.gcd(j, k)
            H[j-1, k-1] = np.log(g + 1) / np.sqrt(j * k)
    H = (H + H.T) / 2  # ensure symmetric
    eigs = np.linalg.eigvalsh(H)
    r, beta, n_pts = measure_peak_gap(eigs)
    print(f'{"GCD kernel log(gcd)/sqrt(jk), N=" + str(N):<45} {N:>5} {r:>+8.4f} {beta:>8.3f} {n_pts:>5}')

# ============================================================
# CANDIDATE 9: Ramanujan sum operator
# ============================================================
for N in [200]:
    # H_{jk} = c_k(j) / sqrt(j*k) where c_k(j) is the Ramanujan sum
    H = np.zeros((N, N))
    for j in range(1, N + 1):
        for k in range(1, N + 1):
            # c_k(j) = sum_{d|gcd(j,k)} mu(k/d) * d
            g = np.gcd(j, k)
            # Simplified: use Euler phi approximation
            # c_k(j) = mu(k/gcd(j,k)) * phi(gcd(j,k)) / phi(k/gcd(j,k))
            # For simplicity, use gcd-based weight
            from sympy import totient
            H[j-1, k-1] = float(totient(g)) / np.sqrt(j * k)
    H = (H + H.T) / 2
    eigs = np.linalg.eigvalsh(H)
    r, beta, n_pts = measure_peak_gap(eigs)
    print(f'{"Ramanujan sum phi(gcd)/sqrt(jk), N=200":<45} {200:>5} {r:>+8.4f} {beta:>8.3f} {n_pts:>5}')

# ============================================================
# VERDICT
# ============================================================
print('\n' + '=' * 70)
print('VERDICT: WHICH OPERATORS HAVE beta > 1?')
print('=' * 70)
print(f'\n  Target: zeta zeros have r ~ +0.80, beta ~ 2')
print(f'  GUE baseline: r ~ +0.04, beta ~ 0')
print(f'  Any operator with r > 0.3 and beta > 0.5 is worth investigating.')

print(f'\nTotal time: {time.time() - t0:.1f}s')
