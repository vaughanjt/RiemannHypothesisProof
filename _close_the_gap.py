"""Close the trace formula gap: wider bandwidth + Ulam modes.

The Lorentzian matches at W=3 (6% error). Gaussians don't (8-34%).
The fix: increase bandwidth AND use per-mode couplings (Ulam).

The prime sum in the explicit formula has terms at ALL distances:
  g(m*log(p)) for primes p and powers m

A test function h with Fourier transform g captures primes where
g(m*log(p)) is non-negligible. For Gaussians with width w:
  g(x) ~ exp(-w^2*x^2/2)
  Significant when: m*log(p) < ~3/w

For w=10: log(p) < 0.3 -> p < 1.35 (only p~1, essentially nothing)
For w=5:  log(p) < 0.6 -> p < 1.8 (still nothing significant)

So the WIDE Gaussians intrinsically don't probe prime structure!
The "error" for wide Gaussians is from the DIAGONAL, not the off-diagonal.

The Lorentzian with w=5: g(x) ~ exp(-5|x|), significant for log(p) < 1 -> p < 3.
This IS why it matches — it probes primes 2 and 3.

STRATEGY: Use NARROW test functions (small w) that probe MANY primes.
These need WIDE bandwidth in the operator to capture.

Then: increase W progressively and show the trace formula error decreases.
"""
import sys, time
sys.path.insert(0, "src")
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar, minimize
import mpmath
mpmath.mp.dps = 20

t0 = time.time()

N = 200
print("Computing 200 zeros...", flush=True)
zeta_zeros = np.array([float(mpmath.im(mpmath.zetazero(k))) for k in range(1, N+1)])

from sympy import primerange
primes = list(primerange(2, 2000))
from math import gcd

trim = int(0.1*N)
ms = np.mean(np.diff(zeta_zeros[trim:-trim]))

def N_deriv(T):
    if T < 2: return 0.001
    return np.log(T/(2*np.pi)) / (2*np.pi)

def N_smooth(T):
    if T < 2: return 0
    return T/(2*np.pi)*np.log(T/(2*np.pi)) - T/(2*np.pi) + 7/8

def weyl_zero(n):
    t = 2*np.pi*n / np.log(max(n,2)+2)
    for _ in range(30):
        if t < 1: t = 10.0
        t -= (N_smooth(t) - n) / N_deriv(t)
    return t


def build_H_wide(N_size, n_primes, W, sigma=0.5, mode_couplings=None):
    """Build operator with variable bandwidth and optional Ulam mode couplings."""
    primes_k = primes[:n_primes]

    # Diagonal
    alpha = np.zeros(N_size)
    for k in range(1, N_size+1):
        Tw = weyl_zero(k); dN = N_deriv(Tw)
        s = sum(-np.sin(2*m*Tw*np.log(p))/(m*p**(m*sigma))
                for p in primes_k for m in range(1,6)) / np.pi
        alpha[k-1] = Tw + s / dN

    H = np.diag(alpha)

    # Off-diagonal with optional mode decomposition
    if mode_couplings is None:
        # Single-mode: all primes same coupling
        for ki in range(N_size):
            Tk = alpha[ki]
            logT = max(np.log(max(Tk,10)/(2*np.pi)), 0.1)
            for d in range(1, W+1):
                if ki+d >= N_size: continue
                val = sum(np.log(p)/(p**(m*sigma)*logT)*np.cos(2*np.pi*d*m*np.log(p)/logT)
                          for p in primes_k for m in range(1,3))
                H[ki,ki+d] = val
                H[ki+d,ki] = val
    else:
        # Multi-mode: per residue class couplings
        q = max(mode_couplings.keys()) + 1  # infer modulus
        for ki in range(N_size):
            Tk = alpha[ki]
            logT = max(np.log(max(Tk,10)/(2*np.pi)), 0.1)
            for d in range(1, W+1):
                if ki+d >= N_size: continue
                val = 0.0
                for r, C_r in mode_couplings.items():
                    class_primes = [p for p in primes_k if p % q == r]
                    for p in class_primes:
                        lp = np.log(p)
                        for m in range(1,3):
                            val += C_r * lp/(p**(m*sigma)*logT)*np.cos(2*np.pi*d*m*lp/logT)
                H[ki,ki+d] = val
                H[ki+d,ki] = val

    return H, alpha


def optimize_and_score(H, alpha, zeta_z):
    """Optimize V scale and return eigenvalue error."""
    V = H - np.diag(np.diag(H))
    vn = np.linalg.norm(V, ord=2)
    if vn < 0.01:
        return np.sort(alpha), np.mean(np.abs(np.sort(alpha) - zeta_z[:len(alpha)])[trim:-trim])

    def obj(log_c):
        eigs = np.sort(np.linalg.eigvalsh(np.diag(alpha) + V/vn*np.exp(log_c)))
        t = int(0.1*len(eigs))
        return np.mean(np.abs(eigs - zeta_z[:len(eigs)])[t:-t])

    res = minimize_scalar(obj, bounds=(-3,3), method='bounded')
    H_opt = np.diag(alpha) + V/vn*np.exp(res.x)
    eigs = np.sort(np.linalg.eigvalsh(H_opt))
    return eigs, res.fun


# ============================================================
# TEST 1: Bandwidth sweep W=1..20 (single mode)
# ============================================================
print("\n" + "="*70, flush=True)
print("TEST 1: BANDWIDTH SWEEP (single mode, 168 primes)", flush=True)
print("="*70, flush=True)

print(f"\n  {'W':>4} {'eig_err':>10} {'<half':>8} {'<gap':>8} {'improvement':>12}", flush=True)
print(f"  {'-'*46}", flush=True)

baseline_eigs = np.sort(np.array([weyl_zero(k) for k in range(1,N+1)]))
baseline_err = np.mean(np.abs(baseline_eigs - zeta_zeros)[trim:-trim])

for W in [0, 1, 2, 3, 5, 8, 10, 15, 20]:
    if W == 0:
        H, alpha = build_H_wide(N, 168, 1)
        eigs = np.sort(alpha)
        err = np.mean(np.abs(eigs - zeta_zeros[:len(eigs)])[trim:-trim])
    else:
        H, alpha = build_H_wide(N, 168, W)
        eigs, err = optimize_and_score(H, alpha, zeta_zeros)

    core_errs = np.abs(eigs - zeta_zeros[:len(eigs)])[trim:-trim]
    pct_h = np.mean(core_errs < ms/2)*100
    pct_f = np.mean(core_errs < ms)*100
    imp = (1-err/baseline_err)*100

    print(f"  {W:>4} {err:>10.4f} {pct_h:>7.1f}% {pct_f:>7.1f}% {imp:>+11.1f}%", flush=True)


# ============================================================
# TEST 2: Bandwidth sweep WITH Ulam modes (mod 8)
# ============================================================
print("\n" + "="*70, flush=True)
print("TEST 2: BANDWIDTH SWEEP WITH ULAM MODES (mod 8)", flush=True)
print("="*70, flush=True)

# Use the known asymmetric couplings: 3mod8 dominant, 5mod8 silent
mode_8 = {1: 1.22, 3: 3.47, 5: 0.001, 7: 1.61}

print(f"\n  {'W':>4} {'eig_err':>10} {'<half':>8} {'improvement':>12}", flush=True)
print(f"  {'-'*38}", flush=True)

for W in [1, 2, 3, 5, 8, 10, 15]:
    H, alpha = build_H_wide(N, 168, W, mode_couplings=mode_8)
    eigs, err = optimize_and_score(H, alpha, zeta_zeros)
    core_errs = np.abs(eigs - zeta_zeros[:len(eigs)])[trim:-trim]
    pct_h = np.mean(core_errs < ms/2)*100
    imp = (1-err/baseline_err)*100
    print(f"  {W:>4} {err:>10.4f} {pct_h:>7.1f}% {imp:>+11.1f}%", flush=True)


# ============================================================
# TEST 3: Trace formula error vs bandwidth
# ============================================================
print("\n" + "="*70, flush=True)
print("TEST 3: TRACE FORMULA ERROR vs BANDWIDTH", flush=True)
print("="*70, flush=True)

# Test with Lorentzian (where we got 6% match) and Gaussian
T_max = zeta_zeros[-1] + 10

def lorentzian_h(t, c=100, w=5):
    return w**2 / ((t-c)**2 + w**2)

def gaussian_h(t, c=100, w=5):
    return np.exp(-(t-c)**2 / (2*w**2))

# Narrow Lorentzian that probes more primes
def narrow_lor_h(t, c=100, w=2):
    return w**2 / ((t-c)**2 + w**2)

sum_zeros_lor = np.sum(lorentzian_h(zeta_zeros))
sum_zeros_gauss = np.sum(gaussian_h(zeta_zeros))
sum_zeros_narrow = np.sum(narrow_lor_h(zeta_zeros))

print(f"\n  {'W':>4} {'Lor w=5 err':>12} {'Gauss w=5 err':>14} "
      f"{'Narrow Lor err':>15}", flush=True)
print(f"  {'-'*49}", flush=True)

for W in [1, 2, 3, 5, 8, 10, 15, 20]:
    H, alpha = build_H_wide(N, 168, W, mode_couplings=mode_8)
    eigs, _ = optimize_and_score(H, alpha, zeta_zeros)

    tr_lor = np.sum(lorentzian_h(eigs))
    tr_gauss = np.sum(gaussian_h(eigs))
    tr_narrow = np.sum(narrow_lor_h(eigs))

    err_lor = abs(tr_lor - sum_zeros_lor) / sum_zeros_lor * 100
    err_gauss = abs(tr_gauss - sum_zeros_gauss) / sum_zeros_gauss * 100
    err_narrow = abs(tr_narrow - sum_zeros_narrow) / sum_zeros_narrow * 100

    print(f"  {W:>4} {err_lor:>11.2f}% {err_gauss:>13.2f}% "
          f"{err_narrow:>14.2f}%", flush=True)


# ============================================================
# TEST 4: The combined operator: Ulam modes + wide bandwidth + more primes
# ============================================================
print("\n" + "="*70, flush=True)
print("TEST 4: THE FULL COMBINED OPERATOR", flush=True)
print("="*70, flush=True)

# The best settings: mod 8 modes, bandwidth 10, 303 primes
for config in [
    ("Diagonal only", 0, 168, None),
    ("W=3, 1-mode, 168p", 3, 168, None),
    ("W=3, 4-mode, 168p", 3, 168, mode_8),
    ("W=10, 4-mode, 168p", 10, 168, mode_8),
    ("W=10, 4-mode, 303p", 10, 303, mode_8),
    ("W=15, 4-mode, 303p", 15, 303, mode_8),
    ("W=20, 4-mode, 303p", 20, 303, mode_8),
]:
    name, W, np_val, modes = config
    if W == 0:
        H, alpha = build_H_wide(N, np_val, 1)
        eigs = np.sort(alpha)
        err = np.mean(np.abs(eigs - zeta_zeros[:len(eigs)])[trim:-trim])
    else:
        H, alpha = build_H_wide(N, np_val, W, mode_couplings=modes)
        eigs, err = optimize_and_score(H, alpha, zeta_zeros)

    core_errs = np.abs(eigs - zeta_zeros[:len(eigs)])[trim:-trim]
    pct_h = np.mean(core_errs < ms/2)*100
    pct_f = np.mean(core_errs < ms)*100
    pct_10 = np.mean(core_errs < ms*0.1)*100
    imp = (1-err/baseline_err)*100

    # Trace formula error for Lorentzian w=5
    tr = np.sum(lorentzian_h(eigs))
    tr_err = abs(tr - sum_zeros_lor) / sum_zeros_lor * 100

    print(f"  {name:>28}: err={err:.4f}({imp:+.0f}%), <half={pct_h:.0f}%, "
          f"<10%={pct_10:.0f}%, trace_err={tr_err:.1f}%", flush=True)


# ============================================================
# VERDICT
# ============================================================
print("\n" + "="*70, flush=True)
print("VERDICT", flush=True)
print("="*70, flush=True)

print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)
