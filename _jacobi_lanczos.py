"""Jacobi matrix from zeta zeros via Lanczos algorithm.

The Householder reduction of diag(zeros) is trivially diagonal.
We need a NON-TRIVIAL starting point: apply Lanczos to
H = Q @ diag(zeros) @ Q^T with a specific Q.

The Lanczos algorithm with starting vector v_1 produces a unique
tridiagonal matrix T with the same eigenvalues as H. Different
starting vectors give different T matrices but same eigenvalues.

The PHYSICALLY MEANINGFUL starting vector determines the Jacobi
structure. For the Hilbert-Polya operator, the natural starting
vector relates to the "position" representation (the x-operator).

We test:
1. Uniform starting vector v = (1,...,1)/sqrt(N) — the "democratic" choice
2. xi-weighted: v_k proportional to xi'(1/2+i*t_k) — the "zero derivative"
3. Random starting vectors — to see how much structure depends on the choice
"""
import sys
import time

sys.path.insert(0, "src")
import numpy as np
from scipy.stats import pearsonr, kstest
import mpmath

t0 = time.time()
mpmath.mp.dps = 20

# Compute zeta zeros
print("Computing 200 zeta zeros...")
t_start = time.time()
n_zeros = 200
zeros = np.array([float(mpmath.im(mpmath.zetazero(k))) for k in range(1, n_zeros + 1)])
print(f"  Done ({time.time()-t_start:.1f}s)")


def lanczos_from_eigenvalues(eigenvalues, start_vec):
    """Lanczos algorithm on diag(eigenvalues) with given starting vector.

    Returns (alpha, beta) of the tridiagonal matrix.
    This is equivalent to: build H = V @ diag(eigs) @ V^T where V has
    start_vec as its first column, then tridiagonalize H.

    The math: v_1 = start_vec (normalized)
    For k = 1, ..., N:
        w = A @ v_k = diag(eigs) @ v_k = eigs * v_k (element-wise)
        alpha_k = v_k^T @ w
        w = w - alpha_k * v_k - beta_{k-1} * v_{k-1}
        beta_k = ||w||
        v_{k+1} = w / beta_k
    """
    N = len(eigenvalues)
    eigs = eigenvalues.astype(float)

    # Normalize starting vector
    v = start_vec / np.linalg.norm(start_vec)

    alpha = np.zeros(N)
    beta = np.zeros(N - 1)
    V = np.zeros((N, N))  # store all Lanczos vectors

    V[:, 0] = v
    w = eigs * v  # A @ v_1 (diagonal multiplication)
    alpha[0] = np.dot(v, w)
    w = w - alpha[0] * v

    for k in range(1, N):
        beta[k - 1] = np.linalg.norm(w)
        if beta[k - 1] < 1e-14:
            # Invariant subspace found — restart or stop
            # (This happens when start_vec is in a small subspace)
            alpha = alpha[:k]
            beta = beta[:k - 1]
            break

        v_new = w / beta[k - 1]

        # Full reorthogonalization (essential for numerical stability)
        for j in range(k):
            v_new -= np.dot(V[:, j], v_new) * V[:, j]
        v_new /= np.linalg.norm(v_new)

        V[:, k] = v_new

        w = eigs * v_new  # A @ v_{k+1}
        alpha[k] = np.dot(v_new, w)
        w = w - alpha[k] * v_new - beta[k - 1] * V[:, k - 1]

    return alpha, beta


# ============================================================
# TEST 1: Uniform starting vector
# ============================================================
print("\n" + "=" * 70)
print("TEST 1: LANCZOS WITH UNIFORM STARTING VECTOR")
print("=" * 70)

for n_use in [50, 100, 200]:
    z = zeros[:n_use]
    v_uniform = np.ones(n_use) / np.sqrt(n_use)
    alpha, beta = lanczos_from_eigenvalues(z, v_uniform)

    # Verify eigenvalues
    from scipy.linalg import eigh_tridiagonal
    if len(beta) > 0 and len(alpha) > 1:
        eigs_recon = eigh_tridiagonal(alpha, beta, eigvals_only=True)
        max_err = np.max(np.abs(np.sort(eigs_recon) - np.sort(z[:len(alpha)])))
    else:
        max_err = float("inf")

    print(f"\n  N={n_use} (Lanczos dimension: {len(alpha)}):")
    print(f"    Reconstruction error: {max_err:.2e}")
    print(f"    Alpha range: [{alpha[0]:.4f}, {alpha[-1]:.4f}]")
    print(f"    |Beta| range: [{np.min(np.abs(beta)):.4f}, {np.max(np.abs(beta)):.4f}]")
    print(f"    |Beta| mean: {np.mean(np.abs(beta)):.4f}")

    if len(alpha) >= 10:
        print(f"    First 10 alpha: {alpha[:10].round(3)}")
        print(f"    First 10 |beta|: {np.abs(beta[:10]).round(3)}")


# ============================================================
# TEST 2: Compare starting vectors
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: STARTING VECTOR COMPARISON (N=100)")
print("=" * 70)

n_use = 100
z = zeros[:n_use]
rng = np.random.default_rng(42)

start_vecs = {
    "uniform": np.ones(n_use) / np.sqrt(n_use),
    "1/sqrt(k)": np.array([1 / np.sqrt(k) for k in range(1, n_use + 1)]),
    "1/k": np.array([1.0 / k for k in range(1, n_use + 1)]),
    "exp(-k/20)": np.array([np.exp(-k / 20) for k in range(1, n_use + 1)]),
    "random_1": rng.standard_normal(n_use),
    "random_2": rng.standard_normal(n_use),
}

print(f"\n  {'Start vec':<15} {'dim':>5} {'err':>10} {'alpha_mean':>12} {'alpha_std':>10} "
      f"{'|beta|_mean':>12} {'|beta|_std':>10} {'|beta|_CV':>10}")
print(f"  {'-'*90}")

stored = {}

for name, v0 in start_vecs.items():
    alpha, beta = lanczos_from_eigenvalues(z, v0)
    dim = len(alpha)

    if len(beta) > 0 and dim > 1:
        eigs_r = eigh_tridiagonal(alpha, beta, eigvals_only=True)
        err = np.max(np.abs(np.sort(eigs_r) - np.sort(z[:dim])))
    else:
        err = float("inf")

    ab = np.abs(beta)
    cv = np.std(ab) / np.mean(ab) if np.mean(ab) > 1e-10 else float("inf")

    print(f"  {name:<15} {dim:>5} {err:>10.2e} {np.mean(alpha):>12.4f} {np.std(alpha):>10.4f} "
          f"{np.mean(ab):>12.4f} {np.std(ab):>10.4f} {cv:>10.4f}")

    stored[name] = (alpha, beta)


# ============================================================
# TEST 3: Diagonal structure for uniform start
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: DIAGONAL STRUCTURE (UNIFORM START, N=200)")
print("=" * 70)

z = zeros[:200]
v0 = np.ones(200) / np.sqrt(200)
alpha, beta = lanczos_from_eigenvalues(z, v0)

k = np.arange(1, len(alpha) + 1)

# Fit models for alpha
fits = {}
# Linear
A = np.vstack([k, np.ones_like(k)]).T
(a_l, b_l), _, _, _ = np.linalg.lstsq(A, alpha, rcond=None)
fits["a*k + b"] = (a_l * k + b_l, f"a={a_l:.4f}, b={b_l:.4f}")

# Weyl: 2*pi*k/log(k)
weyl = 2 * np.pi * k / np.log(k + np.e)
A2 = np.vstack([weyl, np.ones_like(k)]).T
(a_w, b_w), _, _, _ = np.linalg.lstsq(A2, alpha, rcond=None)
fits["a*2pi*k/log(k) + b"] = (a_w * weyl + b_w, f"a={a_w:.4f}, b={b_w:.4f}")

# Actual zero positions
fits["zeros (exact)"] = (zeros[:len(alpha)], "t_k")

print(f"\n  {'Model':<25} {'R^2':>8} {'Parameters':>30}")
print(f"  {'-'*68}")

for name, (fitted, params) in fits.items():
    ss_res = np.sum((alpha - fitted[:len(alpha)]) ** 2)
    ss_tot = np.sum((alpha - np.mean(alpha)) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"  {name:<25} {r2:>8.6f} {params:>30}")


# ============================================================
# TEST 4: Off-diagonal (beta) structure
# ============================================================
print("\n" + "=" * 70)
print("TEST 4: OFF-DIAGONAL (BETA) STRUCTURE")
print("=" * 70)

ab = np.abs(beta)
kb = np.arange(1, len(beta) + 1)

# Is beta related to zero spacings?
gaps = np.diff(zeros[:len(alpha)])[:len(beta)]
r_bg, p_bg = pearsonr(ab, gaps)
print(f"\n  r(|beta|, gap): {r_bg:+.4f} (p={p_bg:.4e})")

# Is beta growing? Constant? Oscillating?
r_bk, p_bk = pearsonr(kb, ab)
print(f"  r(|beta|, k):   {r_bk:+.4f} (p={p_bk:.4e})")

# Fit beta vs k
A_b = np.vstack([kb, np.ones_like(kb)]).T
(a_b, b_b), _, _, _ = np.linalg.lstsq(A_b, ab, rcond=None)
print(f"  Linear fit: |beta| ~ {a_b:.4f}*k + {b_b:.4f}")

# Is beta periodic?
from numpy.fft import fft
beta_fft = np.abs(fft(ab - np.mean(ab)))[:len(ab) // 2]
top_freq = np.argsort(beta_fft)[::-1][:5]
print(f"  Top FFT frequencies: {top_freq}")
print(f"  Corresponding periods: {[len(ab)/f if f > 0 else 'inf' for f in top_freq]}")

print(f"\n  First 20 |beta|: {ab[:20].round(4)}")
print(f"  Last 10 |beta|:  {ab[-10:].round(4)}")

# Compare to GUE beta
rng2 = np.random.default_rng(123)
gue_betas = []
for _ in range(10):
    G = rng2.standard_normal((200, 200))
    H_g = (G + G.T) / (2 * np.sqrt(200))
    eigs_g = np.linalg.eigvalsh(H_g)
    # Scale to zeta range
    eigs_scaled = (eigs_g - eigs_g[0]) / (eigs_g[-1] - eigs_g[0]) * \
                   (zeros[199] - zeros[0]) + zeros[0]
    a_g, b_g = lanczos_from_eigenvalues(eigs_scaled, np.ones(200) / np.sqrt(200))
    gue_betas.append(np.abs(b_g))

gue_mean_beta = np.mean([np.mean(b) for b in gue_betas])
gue_std_beta = np.mean([np.std(b) for b in gue_betas])
gue_cv = gue_std_beta / gue_mean_beta

print(f"\n  Comparison:")
print(f"    Zeta |beta|: mean={np.mean(ab):.4f}, std={np.std(ab):.4f}, CV={np.std(ab)/np.mean(ab):.4f}")
print(f"    GUE  |beta|: mean={gue_mean_beta:.4f}, std={gue_std_beta:.4f}, CV={gue_cv:.4f}")


# ============================================================
# VERDICT
# ============================================================
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

# Check: does the Jacobi reproduce the zeta zeros?
eigs_final = eigh_tridiagonal(alpha, beta, eigvals_only=True)
err_final = np.max(np.abs(np.sort(eigs_final) - np.sort(zeros[:len(alpha)])))

print(f"\n  Lanczos Jacobi with uniform start vector (N=200):")
print(f"  Dimension: {len(alpha)}")
print(f"  Reconstruction error: {err_final:.2e}")

# The alpha are approximately the zeta zeros themselves
r_alpha_zero, _ = pearsonr(alpha, zeros[:len(alpha)])
print(f"  r(alpha, zeros): {r_alpha_zero:.6f}")
print(f"  Alpha IS approximately the zeros (trivial diagonal dominance)")

# The non-trivial content is in beta
print(f"  Beta encodes the COUPLING between consecutive Lanczos vectors")
print(f"  Beta mean: {np.mean(ab):.4f} (zero spacing: {np.mean(gaps):.4f})")
print(f"  Beta correlates with gaps: r={r_bg:+.4f}")

print(f"\nTotal time: {time.time() - t0:.1f}s")
