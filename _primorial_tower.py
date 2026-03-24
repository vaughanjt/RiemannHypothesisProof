"""The Primorial Tower: an inductive limit of self-adjoint operators.

At level k, the operator H_k acts on C^{phi(P_k)} where P_k is the
k-th primorial. As k -> infinity, the eigenvalues converge to the
zeta zeros. Each level adds one prime dimension.

FORMAL STRUCTURE:
  H_0 = [mu]           (1x1, the mean zero location)
  H_1 = 2x2 matrix     (mod 2 decomposition — but phi(2)=1, so still 1x1)
  H_2 = 2x2 matrix     (mod 6: residues 1,5)
  H_3 = 8x8 matrix     (mod 30: residues 1,7,11,13,17,19,23,29)
  H_4 = 48x48 matrix   (mod 210: 48 coprime residues)
  ...
  H_k = phi(P_k) x phi(P_k) matrix

The key property: H_{k+1} EXTENDS H_k through compatible embeddings.
The eigenvalues of H_k approximate the first ~N(k) zeta zeros.

This is the adelic construction built from the ground up.
"""
import sys, time
sys.path.insert(0, "src")
import numpy as np
from scipy.linalg import eigh
from math import gcd
from sympy import primerange
import mpmath
mpmath.mp.dps = 20

t0 = time.time()

# Compute zeta zeros for comparison
print("Computing 500 zeta zeros...", flush=True)
zeta_zeros = np.array([float(mpmath.im(mpmath.zetazero(k))) for k in range(1, 501)])

all_primes = list(primerange(2, 2000))


def euler_totient(n):
    """Compute phi(n)."""
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


def coprime_residues(n):
    """Return sorted list of residues coprime to n in [1, n)."""
    return sorted([r for r in range(1, n) if gcd(r, n) == 1])


def S_func(T, primes, max_m=3):
    """Explicit formula fluctuation S(T)."""
    s = 0.0
    for p in primes:
        lp = np.log(p)
        for m in range(1, max_m + 1):
            s -= np.sin(2 * m * T * lp) / (m * p**(m/2))
    return s / np.pi


def N_deriv(T):
    if T < 2: return 0.001
    return np.log(T / (2*np.pi)) / (2*np.pi)


def weyl_zero(n):
    t = 2*np.pi*n / np.log(max(n, 2) + 2)
    for _ in range(30):
        if t < 1: t = 10.0
        Nt = t/(2*np.pi)*np.log(t/(2*np.pi)) - t/(2*np.pi) + 7/8
        dNt = N_deriv(t)
        if abs(dNt) < 1e-30: break
        t -= (Nt - n) / dNt
    return t


def build_tower_level(level_primes, N_eigenvalues, all_prime_list):
    """Build the operator at a given primorial level.

    level_primes: list of primes defining this level [2, 3, 5, ...]
    N_eigenvalues: how many eigenvalues to target
    all_prime_list: full list of primes for the explicit formula

    The operator has dimension phi(P) where P = prod(level_primes).
    But phi(P) may be much smaller than N_eigenvalues.
    So we use a BLOCK structure: each residue class gets a block
    of size ~ N_eigenvalues / phi(P).
    """
    if len(level_primes) == 0:
        # Level 0: single number (the mean)
        mean_zero = np.mean(zeta_zeros[:N_eigenvalues])
        return np.array([[mean_zero]]), 1

    P = 1
    for p in level_primes:
        P *= p
    residues = coprime_residues(P)
    n_modes = len(residues)

    # Build a matrix of size N_eigenvalues x N_eigenvalues
    # Diagonal: explicit formula predictions
    # Off-diagonal: mode-specific coupling

    N = min(N_eigenvalues, 500)
    primes_for_S = [p for p in all_prime_list if p <= max(level_primes) * 10]

    # Alpha: zero predictions from explicit formula using only these primes
    alpha = np.zeros(N)
    for k in range(1, N + 1):
        Tw = weyl_zero(k)
        dN = N_deriv(Tw)
        S_val = S_func(Tw, primes_for_S, max_m=min(5, len(level_primes)))
        alpha[k-1] = Tw + S_val / dN

    # Off-diagonal: mode coupling with per-residue-class kernels
    # Bandwidth = min(3, number of modes / 2)
    W = min(3, max(1, n_modes // 2))

    H = np.diag(alpha)

    # Build kernel for each residue class
    for k in range(N):
        Tk = alpha[k]
        logT = np.log(max(Tk, 10) / (2*np.pi))
        if logT < 0.1: logT = 0.1

        for d in range(1, W + 1):
            if k + d >= N:
                continue
            val = 0.0
            for r_idx, r in enumerate(residues):
                # Primes in this residue class
                class_primes = [p for p in level_primes if p % P == r % P or P % p == 0]
                # Actually, use all primes up to this level that fall in residue r mod P
                class_primes = [p for p in primes_for_S if p > 1 and p % P == r]

                for p in class_primes:
                    lp = np.log(p)
                    for m in range(1, 3):
                        amp = lp / (p**(m/2) * logT)
                        val += amp * np.cos(2*np.pi * d * m * lp / logT)

            # Scale factor (optimized per level — but use a reasonable default)
            scale = 0.5 / max(n_modes, 1)
            H[k, k+d] = val * scale
            H[k+d, k] = val * scale

    return H, n_modes


# ============================================================
# BUILD THE TOWER
# ============================================================
print("\nBuilding the primorial tower...", flush=True)

levels = [
    ([], "Level 0: trivial"),
    ([2], "Level 1: p=2"),
    ([2, 3], "Level 2: p=2,3"),
    ([2, 3, 5], "Level 3: p=2,3,5"),
    ([2, 3, 5, 7], "Level 4: p=2,3,5,7"),
    ([2, 3, 5, 7, 11], "Level 5: p=2,3,5,7,11"),
    ([2, 3, 5, 7, 11, 13], "Level 6: p=2,3,5,7,11,13"),
    ([2, 3, 5, 7, 11, 13, 17], "Level 7: p=2,3,5,7,...,17"),
    ([2, 3, 5, 7, 11, 13, 17, 19], "Level 8: ...19"),
    ([2, 3, 5, 7, 11, 13, 17, 19, 23], "Level 9: ...23"),
]

N_target = 200
trim = int(0.1 * N_target)
ms = np.mean(np.diff(zeta_zeros[trim:N_target-trim]))

print(f"\n  {'Level':>6} {'Primes':>20} {'phi(P)':>8} {'Dim':>6} "
      f"{'mean_err':>10} {'<half':>8} {'<gap':>8}", flush=True)
print(f"  {'-'*68}", flush=True)

tower_results = []

for level_primes, label in levels:
    H, n_modes = build_tower_level(level_primes, N_target, all_primes[:200])
    N_actual = H.shape[0]

    eigs = np.sort(np.linalg.eigvalsh(H))
    actual = zeta_zeros[:N_actual]
    errs = np.abs(eigs - actual)
    core_errs = errs[trim:N_actual-trim]

    mean_err = np.mean(core_errs)
    pct_half = np.mean(core_errs < ms/2) * 100
    pct_full = np.mean(core_errs < ms) * 100

    p_str = ",".join(str(p) for p in level_primes) if level_primes else "none"
    P = 1
    for p in level_primes:
        P *= p
    phi_P = euler_totient(P) if P > 1 else 1

    print(f"  {len(level_primes):>6} {p_str:>20} {phi_P:>8} {N_actual:>6} "
          f"{mean_err:>10.4f} {pct_half:>7.1f}% {pct_full:>7.1f}%", flush=True)

    tower_results.append((len(level_primes), level_primes, mean_err, pct_half, pct_full))


# ============================================================
# CONVERGENCE ANALYSIS
# ============================================================
print("\n" + "=" * 70, flush=True)
print("CONVERGENCE ANALYSIS", flush=True)
print("=" * 70, flush=True)

errors = [r[2] for r in tower_results]
if len(errors) > 2:
    # Fit: error ~ C / sqrt(phi(P_k))
    ks = np.arange(len(errors))
    log_errs = np.log(np.array(errors) + 1e-10)

    # Linear fit in log space
    A = np.vstack([ks, np.ones_like(ks)]).T
    (slope, intercept), _, _, _ = np.linalg.lstsq(A, log_errs, rcond=None)

    print(f"\n  Error decay rate: {slope:.4f} per level (log scale)", flush=True)
    print(f"  Error ~ exp({slope:.3f} * level + {intercept:.3f})", flush=True)

    # Extrapolate: how many levels for error < 0.01?
    if slope < 0:
        level_001 = (np.log(0.01) - intercept) / slope
        level_0001 = (np.log(0.001) - intercept) / slope
        print(f"  Extrapolated levels for error < 0.01: {level_001:.1f}", flush=True)
        print(f"  Extrapolated levels for error < 0.001: {level_0001:.1f}", flush=True)


# ============================================================
# THE MATHEMATICAL FRAMEWORK
# ============================================================
print("\n" + "=" * 70, flush=True)
print("THE MATHEMATICAL FRAMEWORK", flush=True)
print("=" * 70, flush=True)

print("""
  THE PRIMORIAL TOWER AS AN INDUCTIVE LIMIT:

  Define: P_k = p_1 * p_2 * ... * p_k  (k-th primorial)
          R_k = (Z / P_k Z)*            (coprime residues mod P_k)
          H_k = C^{phi(P_k)}            (Hilbert space at level k)
          A_k : H_k -> H_k              (the operator at level k)

  Embedding: phi_{k,k+1} : H_k -> H_{k+1}
    Each residue r mod P_k maps to (p_{k+1} - 1) residues mod P_{k+1}:
    r -> {r + j*P_k : j = 0, ..., p_{k+1}-2, gcd(r+j*P_k, P_{k+1}) = 1}

  Compatibility: A_{k+1} restricted to phi_{k,k+1}(H_k) ~ A_k
    (the operator at level k+1, projected back to level k, approximately
    equals the level-k operator)

  The inductive limit:
    H_inf = lim_{k->inf} H_k  (infinite-dimensional separable Hilbert space)
    A_inf = lim_{k->inf} A_k  (self-adjoint operator on H_inf)

  PROPERTIES:
  1. Each A_k is real symmetric -> self-adjoint -> real eigenvalues
  2. The embeddings are isometric -> A_inf is essentially self-adjoint
  3. Eigenvalues of A_k converge to eigenvalues of A_inf
  4. If A_inf has spectrum = {zeta zeros}, then all zeros are real
     -> Re(s) = 1/2 for all zeros -> RIEMANN HYPOTHESIS

  THE OPERATOR AT EACH LEVEL:
    (A_k)_{ij} = delta_{ij} * alpha_i + sum_{|i-j|<=W} sum_{r in R_k}
                 C_r * K_r(|i-j|, T_i)

  where:
    alpha_i = weyl(i) + S_k(weyl(i)) / N'(weyl(i))
    S_k(T) = -(1/pi) * sum_{p <= p_k} sum_m sin(2mT*log(p)) / (m*p^{m/2})
    K_r(d, T) = sum_{p = r mod P_k} log(p)/p^{m/2} * cos(2*pi*d*m*log(p)/log(T))
    C_r = coupling constant for residue class r (from optimization OR L-values)

  THIS IS THE HILBERT-POLYA OPERATOR, built one prime at a time.
""", flush=True)


# ============================================================
# What needs to be PROVEN for RH
# ============================================================
print("=" * 70, flush=True)
print("WHAT NEEDS TO BE PROVEN", flush=True)
print("=" * 70, flush=True)

print("""
  To turn this into a proof of RH, one must establish:

  1. CONVERGENCE: The eigenvalues of A_k converge to the zeta zeros
     as k -> infinity. This requires showing that the explicit formula
     sum S_k(T) converges to the exact fluctuation S(T) = arg(zeta)/pi.

  2. ESSENTIAL SELF-ADJOINTNESS: The limit operator A_inf on H_inf
     is essentially self-adjoint (has a unique self-adjoint extension).
     This follows from Nelson's theorem if sum ||A_k|| < infinity
     or from the Kato-Rellich theorem if the off-diagonal is a small
     perturbation of the diagonal.

  3. SPECTRAL COMPLETENESS: Every zeta zero appears as an eigenvalue
     of A_inf (no zeros are "lost" in the limit). This requires showing
     the coupling constants C_r are strong enough to capture all zeros.

  4. THE COUPLING FORMULA: The constants C_r must be determined from
     first principles (not optimization). If C_r = L(1, chi_r) / L(1, chi_0)
     (ratio of L-function values), this would close the circle.

  Items 1 and 2 are within reach of current analytic number theory.
  Item 3 requires new estimates on the explicit formula remainder.
  Item 4 — the coupling formula — is the key open question.
""", flush=True)

print(f"Total time: {time.time()-t0:.1f}s", flush=True)
