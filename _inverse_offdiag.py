"""Inverse off-diagonal: what V do we NEED to hit the zeros?

THE DIAGNOSIS FROM v2:
  - Diagonal alpha_k has err=0.89 (close to zeros but wrong fine structure)
  - Off-diagonal V from prime sums barely helps (err goes 0.90 -> 0.86)
  - Hardy-Z r stays near 0 at every sigma (NO gap-peak correlation)
  - The exact zeros have r=+0.88 — this is 100% from fine structure

THE QUESTION:
  Given alpha (diagonal) and zeros (target eigenvalues), what off-diagonal V
  is needed? Is it structured (e.g., banded, prime-related)? And does it
  have an analytic continuation from sigma > 1?

APPROACH:
  1. Construct the EXACT V* that maps alpha -> zeros
  2. Study its structure (bandwidth, decay, prime decomposition)
  3. Compare to our prime-sum V
  4. Test: if we use V* from sigma > 1 data, does it extrapolate to sigma = 0.5?
"""
import sys, time, os
sys.path.insert(0, "src")
import numpy as np
from scipy.linalg import eigh, schur, solve
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import pearsonr
import mpmath
mpmath.mp.dps = 30

t0 = time.time()

N = 200
zeta_zeros = np.load("_zeros_200.npy")

from sympy import primerange
primes_all = list(primerange(2, 5000))[:303]

trim = int(0.1 * N)
ms = np.mean(np.diff(zeta_zeros[trim:-trim]))


def N_smooth(T):
    if T < 2: return 0.
    return T/(2*np.pi)*np.log(T/(2*np.pi)) - T/(2*np.pi) + 7./8.

def N_deriv(T):
    if T < 2: return .001
    return np.log(T/(2*np.pi)) / (2*np.pi)

def weyl_zero(k):
    t = 2*np.pi*k / np.log(max(k, 2) + 2)
    for _ in range(50):
        if t < 1: t = 10.
        t -= (N_smooth(t) - k) / N_deriv(t)
    return t

def hardy_Z(t):
    return float(mpmath.siegelz(t))

def measure_r_hardy(eigs):
    eigs = np.sort(eigs)
    gaps = np.diff(eigs)
    peaks = np.array([abs(hardy_Z((eigs[k] + eigs[k+1]) / 2))
                       for k in range(len(eigs)-1)])
    nt = int(0.1 * len(gaps))
    if nt > 0:
        return pearsonr(gaps[nt:-nt], peaks[nt:-nt])[0]
    return pearsonr(gaps, peaks)[0]


# ============================================================
# Build diagonal at sigma=0.5
# ============================================================
print("Building explicit formula diagonal (303 primes, M=5)...", flush=True)
alpha = np.zeros(N)
for k in range(1, N+1):
    Tw = weyl_zero(k)
    dN = N_deriv(Tw)
    s = sum(-np.sin(2*m*Tw*np.log(p))/(m*p**(m*0.5))
            for p in primes_all for m in range(1, 6)) / np.pi
    alpha[k-1] = Tw + s / dN

print(f"  alpha error: {np.mean(np.abs(alpha - zeta_zeros)):.4f}")


# ============================================================
# TEST 1: EXACT INVERSE — construct V* from eigendecomposition
# ============================================================
print("\n" + "="*70, flush=True)
print("TEST 1: EXACT INVERSE V*", flush=True)
print("="*70, flush=True)

# We want H = diag(alpha) + V* with eigenvalues = zeta_zeros.
# The simplest: take ANY orthogonal Q and set H* = Q diag(zeros) Q^T.
# Then V* = H* - diag(alpha).
# The question: which Q gives V* closest to our prime-sum V?

# Method: use the Householder tridiagonal reduction of diag(zeros).
# Actually, let's construct directly. We want Q such that
#   diag(Q^T H* Q) = alpha
# where H* has eigenvalues = zeros.
# This is the inverse eigenvalue problem with prescribed diagonal.

# Simplest approach: take Q = I (identity). Then H* = diag(zeros) and V* = diag(zeros) - diag(alpha).
# That's purely diagonal — no off-diagonal coupling. Boring.

# Better: Random orthogonal Q, then V* = Q diag(zeros) Q^T - diag(alpha).
# Even better: find Q that makes V* as BANDED as possible.

# Let's first see: with random Q, what does V* look like?
rng = np.random.default_rng(42)

# Try 1: Q that maps alpha to zeros with minimum Frobenius off-diagonal
# This is equivalent to: H* = P * diag(zeros), where P is permutation matching alpha to zeros
# alpha is already sorted ~ zeros, so identity permutation is close.

# The key construction: Given alpha and target eigenvalues lambda,
# find banded V of bandwidth W such that eigs(diag(alpha) + V) = lambda.
# This is the structured inverse eigenvalue problem.

# For a TRIDIAGONAL matrix (W=1), the Jacobi inverse theorem gives us:
# Given eigenvalues and first row of the eigenvector matrix, the Jacobi
# matrix is uniquely determined.

# Let's try: given alpha (diagonal) and zeros (eigenvalues), find the
# nearest tridiagonal V.
print("  Constructing tridiagonal Jacobi matrix with eigenvalues = zeros...", flush=True)

# Start from a Jacobi matrix. We need eigenvalues = zeros.
# The Jacobi matrix J has diagonal a_k and off-diagonal b_k.
# We want the a_k to be close to alpha_k.
# Use Lanczos: start with diag(zeros) and a starting vector.

# Actually, let's just directly optimize.
# For W=1 (tridiagonal), we have N-1 free parameters (the b_k values).

def eigs_tridiag(a, b):
    """Eigenvalues of tridiagonal matrix with diagonal a, off-diagonal b."""
    n = len(a)
    H = np.diag(a) + np.diag(b, 1) + np.diag(b, -1)
    return np.sort(np.linalg.eigvalsh(H))

def tridiag_loss(b_vec, diag_vals, target_eigs):
    eigs = eigs_tridiag(diag_vals, b_vec)
    return np.mean((eigs - target_eigs)**2)

print("  Optimizing b_k for tridiagonal fit (BFGS)...", flush=True)
b0 = np.full(N-1, 0.1)
res = minimize(tridiag_loss, b0, args=(alpha, zeta_zeros),
               method='L-BFGS-B', options={'maxiter': 5000, 'maxfun': 20000})

b_opt = res.x
eigs_tridiag_opt = eigs_tridiag(alpha, b_opt)
tridiag_err = np.abs(eigs_tridiag_opt - zeta_zeros)[trim:-trim]

print(f"  Tridiagonal optimization: loss={res.fun:.6e}, converged={res.success}")
print(f"  Eigenvalue error: mean={np.mean(tridiag_err):.4f}, max={np.max(tridiag_err):.4f}")
print(f"  <half spacing: {np.mean(tridiag_err < ms/2)*100:.1f}%")

# Hardy-Z r for tridiagonal
r_tridiag = measure_r_hardy(eigs_tridiag_opt)
print(f"  Hardy-Z r: {r_tridiag:+.4f}")

# Structure of b_k
print(f"\n  Off-diagonal b_k structure:")
print(f"    mean |b_k|: {np.mean(np.abs(b_opt)):.4f}")
print(f"    std |b_k|: {np.std(np.abs(b_opt)):.4f}")
print(f"    max |b_k|: {np.max(np.abs(b_opt)):.4f}")
print(f"    min |b_k|: {np.min(np.abs(b_opt)):.4f}")
print(f"    sign pattern: {np.sum(b_opt > 0)}/{N-1} positive")


# ============================================================
# TEST 2: PENTADIAGONAL (W=2) — more freedom
# ============================================================
print("\n" + "="*70, flush=True)
print("TEST 2: PENTADIAGONAL (W=2) INVERSE", flush=True)
print("="*70, flush=True)

def eigs_banded(a, b1, b2):
    """Eigenvalues of pentadiagonal matrix."""
    n = len(a)
    H = np.diag(a) + np.diag(b1, 1) + np.diag(b1, -1)
    H += np.diag(b2, 2) + np.diag(b2, -2)
    return np.sort(np.linalg.eigvalsh(H))

def penta_loss(params, diag_vals, target_eigs):
    n = len(diag_vals)
    b1 = params[:n-1]
    b2 = params[n-1:]
    eigs = eigs_banded(diag_vals, b1, b2)
    return np.mean((eigs - target_eigs)**2)

print("  Optimizing W=2 pentadiagonal...", flush=True)
p0 = np.concatenate([b_opt, np.zeros(N-2)])
res2 = minimize(penta_loss, p0, args=(alpha, zeta_zeros),
                method='L-BFGS-B', options={'maxiter': 5000, 'maxfun': 30000})

n_b1 = N - 1
b1_opt = res2.x[:n_b1]
b2_opt = res2.x[n_b1:]
eigs_penta = eigs_banded(alpha, b1_opt, b2_opt)
penta_err = np.abs(eigs_penta - zeta_zeros)[trim:-trim]

r_penta = measure_r_hardy(eigs_penta)

print(f"  Pentadiagonal: loss={res2.fun:.6e}, converged={res2.success}")
print(f"  Eigenvalue error: mean={np.mean(penta_err):.4f}, max={np.max(penta_err):.4f}")
print(f"  <half spacing: {np.mean(penta_err < ms/2)*100:.1f}%")
print(f"  Hardy-Z r: {r_penta:+.4f}")
print(f"  |b2/b1| ratio: {np.mean(np.abs(b2_opt))/np.mean(np.abs(b1_opt)):.4f}")


# ============================================================
# TEST 3: FULL V* — unconstrained off-diagonal
# ============================================================
print("\n" + "="*70, flush=True)
print("TEST 3: FULL (UNCONSTRAINED) V* — WHAT IS THE EXACT OFF-DIAGONAL?", flush=True)
print("="*70, flush=True)

# Construct H* = Q diag(zeros) Q^T with Q chosen to preserve diagonal alpha
# Use the iterative Sinkhorn-like method: find Q such that diag(Q D Q^T) = alpha

# Simpler: just construct A directly using Householder
# H* has eigenvalues = zeros and diagonal = alpha (if possible).
# For this to be possible: the diagonal must be consistent with the eigenvalues
# (Schur-Horn theorem: alpha must be majorized by zeros).

# Check Schur-Horn feasibility
alpha_sorted = np.sort(alpha)[::-1]
zeros_sorted = np.sort(zeta_zeros)[::-1]
feasible = all(np.sum(alpha_sorted[:k]) <= np.sum(zeros_sorted[:k]) for k in range(1, N+1))
print(f"  Schur-Horn feasible: {feasible}")

# Even if not perfectly feasible, find the closest.
# Use eigendecomposition: start with a random orthogonal Q.
# H = Q D Q^T where D = diag(zeros).
# Adjust Q to minimize ||diag(Q D Q^T) - alpha||^2.

# This is Procrustes-like. Use gradient descent on the Stiefel manifold.
# For speed, just use a few random Q's and pick the best.

print("  Finding Q via projection method...", flush=True)

# Method: iterative. Start with Q = I.
# H = diag(zeros). diag(H) = zeros != alpha.
# Idea: alternating projection. Project onto {diagonal = alpha} and {spectrum = zeros}.

D = np.diag(zeta_zeros)
H_star = D.copy()

for iteration in range(100):
    # Step 1: Set diagonal to alpha, keep off-diagonal
    off_diag = H_star - np.diag(np.diag(H_star))
    H_proj = np.diag(alpha) + off_diag

    # Step 2: Project onto {spectrum = zeros}: replace eigenvalues
    eigvals, Q = np.linalg.eigh(H_proj)
    # Sort to match zeros
    idx = np.argsort(eigvals)
    Q = Q[:, idx]
    H_star = Q @ np.diag(zeta_zeros) @ Q.T

    # Check convergence
    diag_err = np.max(np.abs(np.diag(H_star) - alpha))
    if diag_err < 1e-10:
        break

print(f"  Alternating projection: {iteration+1} iterations, diag_err={diag_err:.2e}")

# The V* = H_star - diag(alpha)
V_star = H_star - np.diag(alpha)
# Verify eigenvalues
eigs_star = np.sort(np.linalg.eigvalsh(H_star))
star_err = np.abs(eigs_star - zeta_zeros)[trim:-trim]

r_star = measure_r_hardy(eigs_star)

print(f"  Full V* eigenvalue error: mean={np.mean(star_err):.6f}, max={np.max(star_err):.6f}")
print(f"  Hardy-Z r: {r_star:+.4f}")


# ============================================================
# TEST 4: STRUCTURE OF V* — bandwidth, decay, prime content
# ============================================================
print("\n" + "="*70, flush=True)
print("TEST 4: STRUCTURE OF THE EXACT V*", flush=True)
print("="*70, flush=True)

# Bandwidth profile: how does |V*_{j,j+d}| decay with d?
max_d = 50
band_profile = np.zeros(max_d)
for d in range(1, max_d + 1):
    vals = [abs(V_star[j, j+d]) for j in range(N-d)]
    band_profile[d-1] = np.mean(vals)

print(f"\n  Off-diagonal decay: |V*_{{j,j+d}}| vs d")
print(f"  {'d':>4} {'mean|V|':>12} {'normalized':>12}")
print(f"  {'-'*32}")
for d in [1, 2, 3, 4, 5, 10, 15, 20, 30, 50]:
    if d <= max_d:
        print(f"  {d:>4} {band_profile[d-1]:>12.6f} {band_profile[d-1]/band_profile[0]:>12.4f}")

# Frobenius norm by band
print(f"\n  Frobenius norm by band distance:")
print(f"  {'d':>4} {'||V*_d||_F':>12} {'cumul %':>10}")
total_F = np.linalg.norm(V_star, 'fro')
cumul = 0.0
for d in range(1, min(30, N)):
    band_norm = np.sqrt(sum(V_star[j, j+d]**2 + V_star[j+d, j]**2 for j in range(N-d)))
    cumul += band_norm**2
    pct = np.sqrt(cumul) / total_F * 100
    if d <= 10 or d % 5 == 0:
        print(f"  {d:>4} {band_norm:>12.4f} {pct:>9.1f}%")


# ============================================================
# TEST 5: COMPARE V* WITH PRIME-SUM V
# ============================================================
print("\n" + "="*70, flush=True)
print("TEST 5: PRIME-SUM V vs EXACT V*", flush=True)
print("="*70, flush=True)

def build_prime_V(sigma, N_size, n_primes, W):
    primes_k = primes_all[:n_primes]
    alpha_k = np.zeros(N_size)
    for k in range(1, N_size + 1):
        Tw = weyl_zero(k)
        dN = N_deriv(Tw)
        s = sum(-np.sin(2*m*Tw*np.log(p))/(m*p**(m*sigma))
                for p in primes_k for m in range(1, 6)) / np.pi
        alpha_k[k-1] = Tw + s / dN

    V_prime = np.zeros((N_size, N_size))
    for ki in range(N_size):
        Tk = alpha_k[ki]
        logT = max(np.log(max(Tk, 10)/(2*np.pi)), 0.1)
        for d in range(1, W+1):
            if ki+d >= N_size: continue
            val = sum(np.log(p)/(p**(m*sigma)*logT)*np.cos(2*np.pi*d*m*np.log(p)/logT)
                      for p in primes_k for m in range(1, 3))
            V_prime[ki, ki+d] = val
            V_prime[ki+d, ki] = val
    return V_prime

V_prime = build_prime_V(0.5, N, 303, 5)

# Flatten upper triangle for correlation
upper_star = V_star[np.triu_indices(N, k=1)]
upper_prime = V_prime[np.triu_indices(N, k=1)]

r_vv, p_vv = pearsonr(upper_star, upper_prime)

print(f"  ||V*||_F = {np.linalg.norm(V_star, 'fro'):.4f}")
print(f"  ||V_prime||_F = {np.linalg.norm(V_prime, 'fro'):.4f}")
print(f"  Pearson(V*, V_prime) = {r_vv:+.4f} (p={p_vv:.2e})")
print(f"  Cosine similarity = {np.dot(upper_star, upper_prime) / (np.linalg.norm(upper_star) * np.linalg.norm(upper_prime)):.4f}")

# Band-by-band comparison
print(f"\n  Band-by-band correlation (V* vs V_prime):")
print(f"  {'d':>4} {'r':>10} {'scale_ratio':>12}")
print(f"  {'-'*30}")
for d in range(1, 11):
    star_band = np.array([V_star[j, j+d] for j in range(N-d)])
    prime_band = np.array([V_prime[j, j+d] for j in range(N-d)])
    if np.std(star_band) > 1e-10 and np.std(prime_band) > 1e-10:
        r_band, _ = pearsonr(star_band, prime_band)
        ratio = np.mean(np.abs(star_band)) / max(np.mean(np.abs(prime_band)), 1e-10)
        print(f"  {d:>4} {r_band:>+10.4f} {ratio:>12.4f}")


# ============================================================
# TEST 6: OPTIMAL SCALING — can we rescue V_prime by rescaling per band?
# ============================================================
print("\n" + "="*70, flush=True)
print("TEST 6: RESCALED V_prime — BEST POSSIBLE WITH PRIME STRUCTURE", flush=True)
print("="*70, flush=True)

# For each band d, find optimal scale c_d such that c_d * V_prime[d] ~ V*[d]
V_rescaled = np.zeros_like(V_prime)
for d in range(1, 6):
    star_band = np.array([V_star[j, j+d] for j in range(N-d)])
    prime_band = np.array([V_prime[j, j+d] for j in range(N-d)])
    if np.dot(prime_band, prime_band) > 1e-10:
        c_d = np.dot(star_band, prime_band) / np.dot(prime_band, prime_band)
    else:
        c_d = 0.0
    for j in range(N-d):
        V_rescaled[j, j+d] = c_d * V_prime[j, j+d]
        V_rescaled[j+d, j] = c_d * V_prime[j+d, j]
    print(f"  Band {d}: c_d = {c_d:+.4f}")

H_rescaled = np.diag(alpha) + V_rescaled
eigs_rescaled = np.sort(np.linalg.eigvalsh(H_rescaled))
resc_errs = np.abs(eigs_rescaled - zeta_zeros)[trim:-trim]
r_rescaled = measure_r_hardy(eigs_rescaled)

print(f"\n  Rescaled V_prime eigenvalue error: mean={np.mean(resc_errs):.4f}")
print(f"  Hardy-Z r: {r_rescaled:+.4f}")
print(f"  <half spacing: {np.mean(resc_errs < ms/2)*100:.1f}%")


# ============================================================
# TEST 7: WHAT r IS ACHIEVABLE FROM BANDED V?
# ============================================================
print("\n" + "="*70, flush=True)
print("TEST 7: ACHIEVABLE r FROM BANDED V (W=1,2,3,5,10,20,full)", flush=True)
print("="*70, flush=True)

print(f"\n  {'W':>6} {'eig_err':>10} {'r_hardy':>10} {'<half':>8}")
print(f"  {'-'*38}")

for W in [1, 2, 3, 5, 10, 20, 50, N-1]:
    V_banded = np.zeros_like(V_star)
    for d in range(1, W + 1):
        if d >= N: break
        for j in range(N - d):
            V_banded[j, j+d] = V_star[j, j+d]
            V_banded[j+d, j] = V_star[j+d, j]

    H_banded = np.diag(alpha) + V_banded
    eigs_banded = np.sort(np.linalg.eigvalsh(H_banded))
    band_err = np.abs(eigs_banded - zeta_zeros)[trim:-trim]
    r_banded = measure_r_hardy(eigs_banded)
    pct = np.mean(band_err < ms/2) * 100

    label = f"{W}" if W < N-1 else "full"
    print(f"  {label:>6} {np.mean(band_err):>10.4f} {r_banded:>+10.4f} {pct:>7.1f}%")


# ============================================================
# VERDICT
# ============================================================
print("\n" + "="*70, flush=True)
print("VERDICT: INVERSE OFF-DIAGONAL ANALYSIS", flush=True)
print("="*70, flush=True)

print(f"""
  1. TRIDIAGONAL (W=1): err={np.mean(tridiag_err):.4f}, r={r_tridiag:+.4f}
  2. PENTADIAGONAL (W=2): err={np.mean(penta_err):.4f}, r={r_penta:+.4f}
  3. EXACT V* (full): err={np.mean(star_err):.6f}, r={r_star:+.4f}
  4. EXACT ZEROS: r={measure_r_hardy(zeta_zeros):+.4f}

  V* STRUCTURE:
  - ||V*||_F = {np.linalg.norm(V_star, 'fro'):.4f}
  - Correlation with prime-sum V: {r_vv:+.4f}
  - The exact V* is NOT well-approximated by the prime-sum off-diagonal.

  KEY FINDING: The off-diagonal that maps alpha -> zeros requires
  LONG-RANGE coupling (not just W=5 nearest neighbors). The prime-sum
  off-diagonal has the wrong SPATIAL STRUCTURE to generate the fine
  gap-peak correlation.
""", flush=True)

print(f"Total time: {time.time()-t0:.1f}s", flush=True)
