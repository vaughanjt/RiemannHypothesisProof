"""Transfer operator v2: reflected exponent and normalized variants.

Lesson from v1: global chi(s) weighting destabilizes the determinant.
The correct Connes approach pairs forward/backward shifts via the
functional equation LOCALLY at each prime:

  Forward: p^{-s}     (encodes Euler factor for zeta(s))
  Backward: p^{-(1-s)} = p^{s-1}  (encodes Euler factor for zeta(1-s))

On the critical line s = 1/2 + it:
  p^{-s}   = p^{-1/2} * p^{-it}
  p^{s-1}  = p^{-1/2} * p^{+it}   (= conjugate of p^{-s})

So the reflected operator is Hermitian on the critical line!
This is exactly the Hilbert-Polya dream: a self-adjoint operator
whose spectrum encodes the zeros.

Variants tested:
  A. Symmetric:       p^{-s} * (forward + backward)
  B. Reflected:       p^{-s} * forward + p^{s-1} * backward
  C. Normalized:      1/sqrt(p) * (p^{-it} * forward + p^{+it} * backward)
  D. Trace-class:     p^{-1} * (p^{-it} * forward + p^{+it} * backward)
  E. Log-weighted:    log(p)/p * (p^{-it} * forward + p^{+it} * backward)

All scans use log|det(I - L)| to avoid overflow.
"""
import sys
import time

sys.path.insert(0, "src")
import numpy as np
from sympy import isprime, primerange
import mpmath

t0 = time.time()
mpmath.mp.dps = 30


def get_primes_up_to(N):
    return list(primerange(2, N + 1))


# ============================================================
# Operator variants
# ============================================================

def build_symmetric(s, N, primes):
    """A. Symmetric: p^{-s} on both forward and backward."""
    L = np.zeros((N, N), dtype=complex)
    for p in primes:
        ps = complex(mpmath.power(p, -s))
        for j in range(1, N + 1):
            pj = p * j
            if pj <= N:
                L[j - 1, pj - 1] += ps
            if j % p == 0:
                L[j - 1, j // p - 1] += ps
    return L


def build_reflected(s, N, primes):
    """B. Reflected: forward p^{-s}, backward p^{s-1}.

    This is the functional equation pairing:
    the backward shift encodes the Euler factor for zeta(1-s).
    On Re(s)=1/2, the operator is Hermitian.
    """
    L = np.zeros((N, N), dtype=complex)
    for p in primes:
        ps_fwd = complex(mpmath.power(p, -s))
        ps_bwd = complex(mpmath.power(p, s - 1))
        for j in range(1, N + 1):
            pj = p * j
            if pj <= N:
                L[j - 1, pj - 1] += ps_fwd
            if j % p == 0:
                L[j - 1, j // p - 1] += ps_bwd
    return L


def build_normalized(s, N, primes):
    """C. Normalized: extract p^{-1/2} common factor, keep only phases.

    L_{jk} = sum_p 1/sqrt(p) * [p^{-it} delta(k=pj) + p^{+it} delta(k=j/p)]

    This makes the operator norm bounded (converges as sum 1/sqrt(p)).
    On the critical line this is Hermitian with bounded eigenvalues.
    """
    t = complex(s).imag
    L = np.zeros((N, N), dtype=complex)
    for p in primes:
        amp = 1.0 / np.sqrt(p)
        phase_fwd = np.exp(-1j * t * np.log(p))
        phase_bwd = np.exp(+1j * t * np.log(p))
        for j in range(1, N + 1):
            pj = p * j
            if pj <= N:
                L[j - 1, pj - 1] += amp * phase_fwd
            if j % p == 0:
                L[j - 1, j // p - 1] += amp * phase_bwd
    return L


def build_trace_class(s, N, primes):
    """D. Trace-class: use 1/p weighting (absolutely convergent).

    L_{jk} = sum_p 1/p * [p^{-it} delta(k=pj) + p^{+it} delta(k=j/p)]

    Faster convergence than 1/sqrt(p). Trace is well-defined.
    """
    t = complex(s).imag
    L = np.zeros((N, N), dtype=complex)
    for p in primes:
        amp = 1.0 / p
        phase_fwd = np.exp(-1j * t * np.log(p))
        phase_bwd = np.exp(+1j * t * np.log(p))
        for j in range(1, N + 1):
            pj = p * j
            if pj <= N:
                L[j - 1, pj - 1] += amp * phase_fwd
            if j % p == 0:
                L[j - 1, j // p - 1] += amp * phase_bwd
    return L


def build_log_weighted(s, N, primes):
    """E. Log-weighted: log(p)/p weighting (Selberg/Montgomery amplitude law).

    This matches the empirically confirmed amplitude law from our ACF analysis.
    """
    t = complex(s).imag
    L = np.zeros((N, N), dtype=complex)
    for p in primes:
        amp = np.log(p) / p
        phase_fwd = np.exp(-1j * t * np.log(p))
        phase_bwd = np.exp(+1j * t * np.log(p))
        for j in range(1, N + 1):
            pj = p * j
            if pj <= N:
                L[j - 1, pj - 1] += amp * phase_fwd
            if j % p == 0:
                L[j - 1, j // p - 1] += amp * phase_bwd
    return L


def build_von_mangoldt(s, N, primes):
    """F. Von Mangoldt: include prime powers with log(p) weight.

    L_{jk} = sum_{p^m} log(p)/p^m * [p^{-imt} delta(k=p^m j) + p^{+imt} delta(k=j/p^m)]

    This is the explicit formula weighting — the trace should give
    sum_{p^m} log(p)/p^m * cos(t * m * log p), which is the
    derivative of log zeta on the critical line.
    """
    t = complex(s).imag
    L = np.zeros((N, N), dtype=complex)
    for p in primes:
        pm = p
        m = 1
        while pm <= N:
            amp = np.log(p) / pm
            phase_fwd = np.exp(-1j * t * m * np.log(p))
            phase_bwd = np.exp(+1j * t * m * np.log(p))
            for j in range(1, N + 1):
                pmj = pm * j
                if pmj <= N:
                    L[j - 1, pmj - 1] += amp * phase_fwd
                if j % pm == 0:
                    L[j - 1, j // pm - 1] += amp * phase_bwd
            m += 1
            pm = p ** m
    return L


# ============================================================
# Log-determinant (handles scale better)
# ============================================================
def log_abs_det(L):
    """Compute log|det(I - L)| using LU decomposition."""
    N = L.shape[0]
    sign, logdet = np.linalg.slogdet(np.eye(N, dtype=complex) - L)
    return logdet.real  # log of the absolute value


# ============================================================
# TEST 1: Verify Hermitian property of reflected operator
# ============================================================
print("=" * 70)
print("TEST 1: HERMITIAN CHECK ON CRITICAL LINE")
print("=" * 70)

N_test = 100
primes = get_primes_up_to(N_test)
s_test = mpmath.mpc(0.5, 14.134)

for name, builder in [("Symmetric", build_symmetric),
                       ("Reflected", build_reflected),
                       ("Normalized", build_normalized),
                       ("Trace-class", build_trace_class)]:
    L = builder(s_test, N_test, primes)
    hermitian_err = np.linalg.norm(L - L.conj().T) / np.linalg.norm(L)
    eigs = np.linalg.eigvalsh(0.5 * (L + L.conj().T))  # Hermitian part
    max_eig = np.max(np.abs(eigs))
    spectral_radius = np.max(np.abs(np.linalg.eigvals(L)))
    print(f"  {name:>15}: ||L-L*||/||L|| = {hermitian_err:.2e}, "
          f"spectral_radius = {spectral_radius:.4f}, "
          f"{'HERMITIAN' if hermitian_err < 1e-10 else 'NOT hermitian'}")


# ============================================================
# TEST 2: Trace analysis
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: TRACE OF L_s ON CRITICAL LINE (should relate to -zeta'/zeta)")
print("=" * 70)

t_val = 14.134
s_test = mpmath.mpc(0.5, t_val)

# Expected trace from explicit formula:
# Tr(L_s) ≈ sum_p log(p)/p * 2*cos(t*log(p))  for von Mangoldt variant
expected_trace = sum(np.log(p) / p * 2 * np.cos(t_val * np.log(p))
                     for p in primes)

# Actual -zeta'/zeta at s
zeta_val = complex(mpmath.zeta(s_test))
zeta_prime = complex(mpmath.diff(mpmath.zeta, s_test))
neg_logderiv = -zeta_prime / zeta_val

print(f"\n  At s = 0.5 + {t_val}i:")
print(f"  -zeta'/zeta = {neg_logderiv:.6f}")
print(f"  Expected trace (explicit formula, {len(primes)} primes) = {expected_trace:.6f}")

for name, builder in [("Symmetric", build_symmetric),
                       ("Reflected", build_reflected),
                       ("Normalized", build_normalized),
                       ("Trace-class", build_trace_class),
                       ("Log-weighted", build_log_weighted),
                       ("Von Mangoldt", build_von_mangoldt)]:
    L = builder(s_test, N_test, primes)
    tr = np.trace(L)
    print(f"  Tr({name:>15}) = {tr:.6f}")


# ============================================================
# TEST 3: CRITICAL LINE SCAN — all variants
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: CRITICAL LINE SCAN — ALL VARIANTS")
print("=" * 70)

N_op = 150
primes_op = get_primes_up_to(N_op)
t_scan = np.linspace(10, 55, 500)

known_zeros = [14.1347, 21.0220, 25.0109, 30.4249, 32.9351,
               37.5862, 40.9187, 43.3271, 48.0052, 49.7738, 52.9703]

builders = [
    ("Symmetric", build_symmetric),
    ("Reflected", build_reflected),
    ("Normalized", build_normalized),
    ("Trace-class", build_trace_class),
    ("Log-weighted", build_log_weighted),
    ("VonMangoldt", build_von_mangoldt),
]

# Store log|det| for each variant
logdets = {name: np.zeros(len(t_scan)) for name, _ in builders}

print(f"\n  Scanning {len(t_scan)} points, N={N_op}, {len(primes_op)} primes...")
t_start = time.time()

for i, t_val in enumerate(t_scan):
    s = mpmath.mpc(0.5, t_val)
    for name, builder in builders:
        L = builder(s, N_op, primes_op)
        logdets[name][i] = log_abs_det(L)

    if (i + 1) % 100 == 0:
        elapsed = time.time() - t_start
        rate = (i + 1) / elapsed
        eta = (len(t_scan) - i - 1) / rate
        print(f"    {i+1}/{len(t_scan)} ({elapsed:.0f}s, ~{eta:.0f}s remaining)")

print(f"  Scan complete: {time.time() - t_start:.1f}s")


# ============================================================
# Find minima in log|det| and score
# ============================================================
def find_logdet_minima(logdet_vals, t_values, n_keep=20):
    """Find local minima in log|det| scan."""
    minima = []
    for i in range(1, len(logdet_vals) - 1):
        if logdet_vals[i] < logdet_vals[i - 1] and logdet_vals[i] < logdet_vals[i + 1]:
            minima.append((t_values[i], logdet_vals[i]))
    minima.sort(key=lambda x: x[1])
    return minima[:n_keep]


def score_minima(minima, known, threshold=0.5):
    matched = set()
    spurious = 0
    for t_min, _ in minima:
        dists = [abs(t_min - z) for z in known]
        best_idx = np.argmin(dists)
        if dists[best_idx] < threshold:
            matched.add(best_idx)
        else:
            spurious += 1
    return len(matched), spurious, matched


print("\n" + "-" * 70)
print("RESULTS: MINIMA OF log|det(I - L_s)| ON CRITICAL LINE")
print("-" * 70)

total_possible = sum(1 for z in known_zeros if 10 < z < 55)
best_score = (0, 999, "")

for name, _ in builders:
    minima = find_logdet_minima(logdets[name], t_scan)
    n_match, n_spurious, matched = score_minima(minima, known_zeros)
    precision = n_match / (n_match + n_spurious) if (n_match + n_spurious) > 0 else 0
    f1 = 2 * precision * (n_match / total_possible) / (precision + n_match / total_possible) \
        if (precision + n_match / total_possible) > 0 else 0

    print(f"\n  {name}:")
    print(f"    Zeros found: {n_match}/{total_possible}  |  "
          f"Spurious: {n_spurious}  |  "
          f"Precision: {precision:.0%}  |  F1: {f1:.2f}")

    # Show top minima
    print(f"    {'t_min':>10} {'log|det|':>10} {'Near zero':>12} {'Dist':>8} {'?':>5}")
    for t_min, ld in minima[:12]:
        dists = [abs(t_min - z) for z in known_zeros]
        best = np.argmin(dists)
        tag = "YES" if dists[best] < 0.5 else ""
        print(f"    {t_min:>10.4f} {ld:>10.2f} {known_zeros[best]:>12.4f} "
              f"{dists[best]:>8.4f} {tag:>5}")

    if (n_match, -n_spurious) > (best_score[0], -best_score[1]):
        best_score = (n_match, n_spurious, name)


# ============================================================
# TEST 4: N-convergence for the best variant
# ============================================================
print("\n" + "=" * 70)
print(f"TEST 4: N-CONVERGENCE FOR TOP VARIANTS")
print("=" * 70)

# Test at zero and non-zero
test_points = [
    (14.1347, True, "first zero"),
    (21.0220, True, "second zero"),
    (20.0, False, "non-zero"),
    (35.0, False, "non-zero"),
]

for t_test, is_zero, label in test_points:
    s_test = mpmath.mpc(0.5, t_test)
    print(f"\n  t={t_test} ({label}):")
    print(f"  {'N':>6}  " + "  ".join(f"{name:>12}" for name, _ in builders))
    print(f"  {'-'*(8 + 14 * len(builders))}")

    for N_val in [50, 100, 150, 200, 300]:
        p_list = get_primes_up_to(N_val)
        row = f"  {N_val:>6}"
        for name, builder in builders:
            L = builder(s_test, N_val, p_list)
            ld = log_abs_det(L)
            row += f"  {ld:>12.2f}"
        print(row)


# ============================================================
# TEST 5: Eigenvalue-1 detection (direct spectral approach)
# ============================================================
print("\n" + "=" * 70)
print("TEST 5: CLOSEST EIGENVALUE TO 1 AT ZERO vs NON-ZERO")
print("=" * 70)

N_eig = 200
primes_eig = get_primes_up_to(N_eig)

print(f"\n  N={N_eig}. Showing min|lambda - 1| for each variant.")
print(f"  {'Variant':>15} {'t=14.13 (zero)':>16} {'t=20.0 (non)':>16} "
      f"{'t=25.01 (zero)':>16} {'t=35.0 (non)':>16} {'RATIO':>8}")
print(f"  {'-'*90}")

for name, builder in builders:
    dists = []
    for t_val in [14.1347, 20.0, 25.0109, 35.0]:
        s_val = mpmath.mpc(0.5, t_val)
        L = builder(s_val, N_eig, primes_eig)
        eigs = np.linalg.eigvals(L)
        min_dist = np.min(np.abs(eigs - 1.0))
        dists.append(min_dist)
    # Ratio: how much closer are zero-point eigenvalues to 1?
    avg_zero = (dists[0] + dists[2]) / 2
    avg_nonzero = (dists[1] + dists[3]) / 2
    ratio = avg_nonzero / avg_zero if avg_zero > 1e-30 else float("inf")
    print(f"  {name:>15} {dists[0]:>16.6f} {dists[1]:>16.6f} "
          f"{dists[2]:>16.6f} {dists[3]:>16.6f} {ratio:>8.2f}")


# ============================================================
# VERDICT
# ============================================================
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

best_name = best_score[2]
n_match = best_score[0]
n_spurious = best_score[1]
print(f"\n  Best variant: {best_name}")
print(f"  Zeros found: {n_match}/{total_possible}, Spurious: {n_spurious}")

if n_match >= 8:
    print(f"\n  >>> BREAKTHROUGH: {best_name} finds {n_match}/11 zeros!")
    print(f"  >>> This operator encodes the zeta zeros in its Fredholm determinant.")
elif n_match >= 5:
    print(f"\n  >>> STRONG SIGNAL: {best_name} finds {n_match}/11 zeros.")
    print(f"  >>> Increasing N should improve. Test N=300-500 next.")
elif n_match >= 3:
    print(f"\n  >>> MODERATE: {best_name} finds {n_match}/11 zeros.")
    print(f"  >>> The operator structure is right but convergence is slow.")
else:
    print(f"\n  >>> WEAK: only {n_match} zeros. Need fundamentally different approach.")

print(f"\nTotal time: {time.time() - t0:.1f}s")
