"""S'(T) coupling: the off-diagonal should encode the DERIVATIVE of arg(zeta).

THE INSIGHT:
  Diagonal:     alpha_k = T_k + S(T_k) / N'(T_k)     [S shifts positions]
  Off-diagonal: b_k ~ g_k ~ 1/N' + S'(T_k)/N'^2       [S' sets coupling]

  where S(T) = (1/pi) arg(zeta(1/2+iT))
  and   S'(T) = (1/pi) Im[zeta'(1/2+iT)/zeta(1/2+iT)]

  The S' term is what creates the fine gap structure that gives r=+0.88.
  Without S': b_k = 1/N'(T_k) -> r ≈ 0 (confirmed).
  With S': b_k has the right oscillatory structure -> r should be high.

  This is the MISSING PIECE: the prime sum enters the off-diagonal
  through the logarithmic derivative of zeta, not just through arg(zeta).
"""
import sys, time
sys.path.insert(0, "src")
import numpy as np
from scipy.stats import pearsonr
from scipy.optimize import minimize_scalar
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

def measure_r(eigs):
    eigs = np.sort(eigs)
    gaps = np.diff(eigs)
    mids = (eigs[:-1] + eigs[1:]) / 2
    peaks = np.array([abs(hardy_Z(m)) for m in mids])
    nt = int(0.1 * len(gaps))
    return pearsonr(gaps[nt:-nt], peaks[nt:-nt])[0]


# ============================================================
# Build diagonal alpha_k at sigma=0.5
# ============================================================
print("Building diagonal alpha_k...", flush=True)
alpha = np.zeros(N)
for k in range(1, N+1):
    Tw = weyl_zero(k); dN = N_deriv(Tw)
    s = sum(-np.sin(2*m*Tw*np.log(p))/(m*p**(m*0.5))
            for p in primes_all for m in range(1, 6)) / np.pi
    alpha[k-1] = Tw + s / dN


# ============================================================
# Compute S'(T) via mpmath — the logarithmic derivative of zeta
# ============================================================
print("\nComputing S'(T) at alpha positions via mpmath...", flush=True)

def S_prime_mpmath(T):
    """S'(T) = (1/pi) * Im[zeta'(s)/zeta(s)] at s = 1/2 + iT."""
    s = mpmath.mpc(0.5, T)
    z = mpmath.zeta(s)
    zp = mpmath.diff(mpmath.zeta, s)
    if abs(z) < 1e-20:
        return 0.0
    return float(mpmath.im(zp / z)) / np.pi

S_prime_vals = np.zeros(N)
for k in range(N):
    T = alpha[k]
    S_prime_vals[k] = S_prime_mpmath(T)
    if k % 50 == 0:
        print(f"  k={k}: T={T:.2f}, S'(T)={S_prime_vals[k]:+.6f}", flush=True)


# ============================================================
# Also compute S'(T) from prime sum (for analytic continuation)
# ============================================================
print("\nComputing S'(T) from prime sum...", flush=True)

def S_prime_primes(T, sigma=0.5, primes=None, M=5):
    """S'(T) from the explicit formula:
    S'(T) = -(1/pi) sum_p sum_m log(p) cos(2mT log p) / (m p^{m*sigma})
    (derivative of S(T) with respect to T)
    """
    if primes is None:
        primes = primes_all
    s = 0.0
    for p in primes:
        lp = np.log(p)
        for m in range(1, M+1):
            s -= 2 * lp * np.cos(2 * m * T * lp) / (m * p**(m * sigma))
    return s / np.pi

S_prime_prime = np.zeros(N)
for k in range(N):
    S_prime_prime[k] = S_prime_primes(alpha[k])


# ============================================================
# TEST: b_k formulas using S'(T)
# ============================================================
print("\n" + "="*70, flush=True)
print("TEST: OFF-DIAGONAL b_k FROM S'(T)", flush=True)
print("="*70, flush=True)

# Formula: b_k = 1/N'(T_k) * (1 + c * S'(T_k) / N'(T_k))
# where c is a scaling parameter to optimize

# Reference: exact zero gaps
exact_gaps = np.diff(zeta_zeros)
smooth_gaps = np.array([1.0 / N_deriv(alpha[k]) for k in range(N-1)])

# S' correction at midpoints
S_prime_mid = (S_prime_vals[:-1] + S_prime_vals[1:]) / 2
S_prime_prime_mid = (S_prime_prime[:-1] + S_prime_prime[1:]) / 2
N_prime_mid = np.array([N_deriv((alpha[k] + alpha[k+1])/2) for k in range(N-1)])

print(f"\n  Correlation between S'(T) versions:")
r_sp, _ = pearsonr(S_prime_vals, S_prime_prime)
print(f"    mpmath S' vs prime-sum S': r = {r_sp:+.4f}")

print(f"\n  S' statistics:")
print(f"    mpmath:    mean={np.mean(S_prime_vals):+.4f}, std={np.std(S_prime_vals):.4f}")
print(f"    prime-sum: mean={np.mean(S_prime_prime):+.4f}, std={np.std(S_prime_prime):.4f}")

# Compare S' with zero gap fluctuations
gap_fluct = exact_gaps - smooth_gaps  # deviation from smooth
r_sf, _ = pearsonr(S_prime_mid, gap_fluct)
r_sfp, _ = pearsonr(S_prime_prime_mid, gap_fluct)
print(f"\n  Correlation of S'(T) with gap fluctuations:")
print(f"    mpmath S' vs gap_fluct: r = {r_sf:+.4f}")
print(f"    prime-sum S' vs gap_fluct: r = {r_sfp:+.4f}")


# ============================================================
# Sweep: b_k = smooth + c * S'/N'^2
# ============================================================
print(f"\n  {'formula':>40} {'c':>6} {'eig_err':>10} {'r_hardy':>10} {'<half':>8}")
print(f"  {'-'*78}")

# b_k = exact zero gap (target)
H = np.diag(alpha) + np.diag(exact_gaps, 1) + np.diag(exact_gaps, -1)
eigs = np.sort(np.linalg.eigvalsh(H))
err = np.mean(np.abs(eigs - zeta_zeros)[trim:-trim])
r = measure_r(eigs)
print(f"  {'b_k = exact_zero_gap':>40} {'---':>6} {err:>10.4f} {r:>+10.4f} "
      f"{np.mean(np.abs(eigs-zeta_zeros)[trim:-trim]<ms/2)*100:>7.1f}%")

# b_k = 1/N' (smooth only)
H = np.diag(alpha) + np.diag(smooth_gaps, 1) + np.diag(smooth_gaps, -1)
eigs = np.sort(np.linalg.eigvalsh(H))
err = np.mean(np.abs(eigs - zeta_zeros)[trim:-trim])
r = measure_r(eigs)
print(f"  {'b_k = 1/N_prime (smooth)':>40} {'---':>6} {err:>10.4f} {r:>+10.4f} "
      f"{np.mean(np.abs(eigs-zeta_zeros)[trim:-trim]<ms/2)*100:>7.1f}%")

# Sweep c for mpmath S'
for c in [-5, -2, -1, -0.5, -0.2, 0, 0.2, 0.5, 1, 2, 5]:
    b_k = smooth_gaps * (1 + c * S_prime_mid / N_prime_mid)
    b_k = np.maximum(b_k, 0.01)  # keep positive for stability
    H = np.diag(alpha) + np.diag(b_k, 1) + np.diag(b_k, -1)
    eigs = np.sort(np.linalg.eigvalsh(H))
    err = np.mean(np.abs(eigs - zeta_zeros)[trim:-trim])
    r = measure_r(eigs)
    print(f"  {'b_k = smooth*(1+c*S_mpmath/N)':>40} {c:>+6.1f} {err:>10.4f} {r:>+10.4f} "
          f"{np.mean(np.abs(eigs-zeta_zeros)[trim:-trim]<ms/2)*100:>7.1f}%")

# Sweep c for prime-sum S'
print()
for c in [-5, -2, -1, -0.5, -0.2, 0, 0.2, 0.5, 1, 2, 5]:
    b_k = smooth_gaps * (1 + c * S_prime_prime_mid / N_prime_mid)
    b_k = np.maximum(b_k, 0.01)
    H = np.diag(alpha) + np.diag(b_k, 1) + np.diag(b_k, -1)
    eigs = np.sort(np.linalg.eigvalsh(H))
    err = np.mean(np.abs(eigs - zeta_zeros)[trim:-trim])
    r = measure_r(eigs)
    print(f"  {'b_k = smooth*(1+c*S_prime/N)':>40} {c:>+6.1f} {err:>10.4f} {r:>+10.4f} "
          f"{np.mean(np.abs(eigs-zeta_zeros)[trim:-trim]<ms/2)*100:>7.1f}%")


# ============================================================
# Fine sweep around best c
# ============================================================
print("\n" + "="*70, flush=True)
print("FINE SWEEP: optimal c for S'(T) coupling", flush=True)
print("="*70, flush=True)

best_r = -1
best_c = 0
best_source = ""

for source, S_mid in [("mpmath", S_prime_mid), ("primes", S_prime_prime_mid)]:
    for c in np.linspace(-3, 3, 61):
        b_k = smooth_gaps * (1 + c * S_mid / N_prime_mid)
        b_k = np.maximum(b_k, 0.01)
        H = np.diag(alpha) + np.diag(b_k, 1) + np.diag(b_k, -1)
        eigs = np.sort(np.linalg.eigvalsh(H))
        gaps = np.diff(eigs)
        # Use fast proxy (logdet) to find optimal c, then compute hardy-Z for best
        mids = (eigs[:-1] + eigs[1:]) / 2
        logdet = np.array([np.sum(np.log(np.abs(m - eigs) + 1e-30)) for m in mids])
        nt = int(0.1 * len(gaps))
        r_proxy, _ = pearsonr(gaps[nt:-nt], logdet[nt:-nt])

        if r_proxy > best_r:
            best_r = r_proxy
            best_c = c
            best_source = source

print(f"  Best proxy r: {best_r:+.4f} at c={best_c:+.2f} ({best_source})")

# Compute exact Hardy-Z r for the best
S_best = S_prime_mid if best_source == "mpmath" else S_prime_prime_mid
b_best = smooth_gaps * (1 + best_c * S_best / N_prime_mid)
b_best = np.maximum(b_best, 0.01)
H_best = np.diag(alpha) + np.diag(b_best, 1) + np.diag(b_best, -1)
eigs_best = np.sort(np.linalg.eigvalsh(H_best))
err_best = np.mean(np.abs(eigs_best - zeta_zeros)[trim:-trim])
r_best = measure_r(eigs_best)
print(f"  Hardy-Z r at best c: {r_best:+.4f}, err={err_best:.4f}")

# Also try multiplicative scaling: b_k = c0 * smooth_gaps + c1 * |S'|
print("\n  Multiplicative: b_k = c0*smooth + c1*|S'|")
for c0, c1 in [(1.0, 0), (1.0, 0.1), (1.0, 0.5), (1.0, 1.0),
               (0.5, 0.5), (0, 1.0), (0.8, 0.2)]:
    b_k = c0 * smooth_gaps + c1 * np.abs(S_prime_mid)
    b_k = np.maximum(b_k, 0.01)
    H = np.diag(alpha) + np.diag(b_k, 1) + np.diag(b_k, -1)
    eigs = np.sort(np.linalg.eigvalsh(H))
    err = np.mean(np.abs(eigs - zeta_zeros)[trim:-trim])
    r = measure_r(eigs)
    print(f"    c0={c0:.1f}, c1={c1:.1f}: err={err:.4f}, r={r:+.4f}")


# ============================================================
# KEY TEST: Does the S' formula at sigma > 1 extrapolate?
# ============================================================
print("\n" + "="*70, flush=True)
print("KEY TEST: S'(T) COUPLING AT sigma > 1 — ANALYTIC CONTINUATION", flush=True)
print("="*70, flush=True)

for sigma in [2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.5]:
    # Build alpha at this sigma
    alpha_s = np.zeros(N)
    for k in range(1, N+1):
        Tw = weyl_zero(k); dN = N_deriv(Tw)
        s = sum(-np.sin(2*m*Tw*np.log(p))/(m*p**(m*sigma))
                for p in primes_all for m in range(1, 6)) / np.pi
        alpha_s[k-1] = Tw + s / dN

    # S'(T) at sigma (from prime sum)
    S_prime_s = np.zeros(N)
    for k in range(N):
        T = alpha_s[k]
        s_p = 0.0
        for p in primes_all:
            lp = np.log(p)
            for m in range(1, 6):
                s_p -= 2 * lp * np.cos(2 * m * T * lp) / (m * p**(m * sigma))
        S_prime_s[k] = s_p / np.pi

    S_prime_s_mid = (S_prime_s[:-1] + S_prime_s[1:]) / 2
    N_prime_s_mid = np.array([N_deriv((alpha_s[k] + alpha_s[k+1])/2) for k in range(N-1)])

    # Use best c from above
    b_k = smooth_gaps * (1 + best_c * S_prime_s_mid / N_prime_s_mid)
    b_k = np.maximum(b_k, 0.01)
    H = np.diag(alpha_s) + np.diag(b_k, 1) + np.diag(b_k, -1)
    eigs = np.sort(np.linalg.eigvalsh(H))
    err = np.mean(np.abs(eigs - zeta_zeros)[trim:-trim])
    r = measure_r(eigs)

    print(f"  sigma={sigma:.2f}: err={err:.4f}, r={r:+.4f}, "
          f"mean|S'|={np.mean(np.abs(S_prime_s)):.4f}, "
          f"||V||={np.linalg.norm(np.diag(b_k,1)+np.diag(b_k,-1), ord=2):.4f}")


# ============================================================
# VERDICT
# ============================================================
print("\n" + "="*70, flush=True)
print("VERDICT: S'(T) COUPLING", flush=True)
print("="*70, flush=True)

print(f"""
  THE STRUCTURE:
    diagonal:     alpha_k = T_k + S(T_k)/N'(T_k)     [from arg(zeta)]
    off-diagonal: b_k = (1/N') * (1 + c*S'/N')        [from d/dT arg(zeta)]

  S'(T) from mpmath vs prime-sum: r = {r_sp:+.4f}
  S' correlation with gap fluctuations: {r_sf:+.4f} (mpmath), {r_sfp:+.4f} (primes)

  BEST RESULT:
    c = {best_c:+.2f} ({best_source})
    Hardy-Z r = {r_best:+.4f}
    eig_err = {err_best:.4f}

  COMPARISON:
    Exact zeros:    r = +0.88
    b_k = zero_gap: r = +0.50
    Optimized W=1:  r = +0.58
    S'(T) formula:  r = {r_best:+.4f}
""", flush=True)

print(f"Total time: {time.time()-t0:.1f}s", flush=True)
