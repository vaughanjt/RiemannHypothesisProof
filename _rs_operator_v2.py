"""Riemann-Siegel operator v2: Corrected phase structure.

SESSION 5 RESULT:
  - Original RS (multiplicative coefficient): r = +0.16  (WEAK)
  - Random phases (additive):                 r = +0.75  (STRONG)
  - Zeta target:                              r = +0.80

THE BUG: The original RS operator used:
  H_{jk} = sum_n [cos(theta-t*log(n))/sqrt(n)] * cos(2*pi*n*(j-k)/N)

This is: coefficient(n) * spatial_oscillation(n,j-k).
The RS phase is a MULTIPLICATIVE factor, making the matrix a smooth circulant
with constrained eigenvalues.

The random-phase version used:
  H_{jk} = sum_n cos(2*pi*n*(j-k)/N + phi_n) / sqrt(n)

This has the phase INSIDE the cosine — an ADDITIVE phase. This changes the
matrix structure fundamentally: the eigenvalues are no longer smooth functions
of the DFT frequencies.

THE FIX: Use the RS phase as an ADDITIVE phase:
  H_{jk}(T) = sum_n cos(theta(T) - T*log(n) + 2*pi*n*(j-k)/N) / sqrt(n)

This is the "correct" RS operator. For each T, the phases phi_n = theta(T) - T*log(n)
are deterministic. As T varies, they sweep continuously, acting quasi-randomly.

ADDITIONAL VARIANTS:
  A. Additive-phase circulant (the fix above)
  B. Z-Toeplitz: H_{jk} = Z(T + (j-k)*delta) / sqrt(N)
  C. Non-circulant: phase depends on j*k, not j-k
  D. Log-frequency: spatial frequency at log(n), not n
  E. Outer product with multiplicative characters
"""
import sys
import time

sys.path.insert(0, "src")
import numpy as np
from scipy.stats import pearsonr, kstest
import mpmath
from riemann.analysis.bost_connes_operator import polynomial_unfold

t0 = time.time()
mpmath.mp.dps = 15


# ============================================================
# Peak-gap measurement
# ============================================================
def measure_peak_gap(eigs_raw):
    """Measure peak-gap correlation r."""
    eigs = np.sort(eigs_raw)
    sp = polynomial_unfold(eigs, trim_fraction=0.1)
    if len(sp) < 20:
        return 0.0, 0
    sp = sp / np.mean(sp)
    n_trim = int(0.1 * len(eigs))
    eigs_trim = eigs[n_trim:-n_trim]
    log_peaks, gaps = [], []
    for k in range(min(len(sp), len(eigs_trim) - 1)):
        z_mid = (eigs_trim[k] + eigs_trim[k + 1]) / 2
        log_det = np.sum(np.log(np.abs(z_mid - eigs) + 1e-30))
        log_peaks.append(log_det)
        gaps.append(sp[k])
    if len(gaps) < 10:
        return 0.0, 0
    r, _ = pearsonr(np.array(gaps), np.array(log_peaks))
    return r, len(gaps)


def wigner_cdf(s):
    return 1 - np.exp(-np.pi * s ** 2 / 4)


# ============================================================
# Operator builders
# ============================================================
def build_original_rs(T, N, N_sum=None):
    """Original (buggy) RS: multiplicative coefficient.

    H_{jk} = sum_n a_n * cos(2*pi*n*(j-k)/N) / C
    where a_n = cos(theta(T) - T*log(n)) / sqrt(n)
    """
    if N_sum is None:
        N_sum = max(int(np.sqrt(T / (2 * np.pi))), 5)
        N_sum = min(N_sum, N)

    theta_T = float(mpmath.siegeltheta(T))
    coeffs = np.array([np.cos(theta_T - T * np.log(n)) / np.sqrt(n)
                        for n in range(1, N_sum + 1)])

    # Build circulant-like matrix
    H = np.zeros((N, N))
    jk = np.arange(N)
    for n_idx in range(N_sum):
        n = n_idx + 1
        cos_vals = np.cos(2 * np.pi * n * jk / N)
        H += coeffs[n_idx] * np.outer(np.ones(N), cos_vals)
        # Actually: H_{jk} depends on j-k, so this should be:
        # For each row j: H_{j,:} = sum_n coeff_n * cos(2*pi*n*(j - np.arange(N))/N)

    # Rebuild correctly
    H = np.zeros((N, N))
    for j in range(N):
        for k in range(j, N):
            val = 0.0
            for n_idx in range(N_sum):
                val += coeffs[n_idx] * np.cos(2 * np.pi * (n_idx + 1) * (j - k) / N)
            H[j, k] = val
            H[k, j] = val

    scale = np.sqrt(np.mean(H ** 2) * N)
    if scale > 1e-10:
        H /= scale
    return H


def build_additive_phase_rs(T, N, N_sum=None):
    """Corrected RS: additive phase inside cosine.

    H_{jk} = (1/C) * sum_n cos(phi_n(T) + 2*pi*n*(j-k)/N) / sqrt(n)
    where phi_n(T) = theta(T) - T*log(n)
    """
    if N_sum is None:
        N_sum = max(int(np.sqrt(T / (2 * np.pi))), 5)
        N_sum = min(N_sum, N)

    theta_T = float(mpmath.siegeltheta(T))
    phases = np.array([theta_T - T * np.log(n) for n in range(1, N_sum + 1)])
    amps = np.array([1.0 / np.sqrt(n) for n in range(1, N_sum + 1)])

    H = np.zeros((N, N))
    for j in range(N):
        for k in range(j, N):
            val = 0.0
            for n_idx in range(N_sum):
                n = n_idx + 1
                val += amps[n_idx] * np.cos(phases[n_idx] + 2 * np.pi * n * (j - k) / N)
            H[j, k] = val
            H[k, j] = val

    scale = np.sqrt(np.mean(H ** 2) * N)
    if scale > 1e-10:
        H /= scale
    return H


def build_random_phase(N, N_sum, rng):
    """Random phase version (the r=+0.75 baseline).

    H_{jk} = (1/C) * sum_n cos(2*pi*n*(j-k)/N + phi_n) / sqrt(n)
    where phi_n ~ Uniform(0, 2*pi)
    """
    phases = rng.uniform(0, 2 * np.pi, N_sum)
    amps = np.array([1.0 / np.sqrt(n) for n in range(1, N_sum + 1)])

    H = np.zeros((N, N))
    for j in range(N):
        for k in range(j, N):
            val = 0.0
            for n_idx in range(N_sum):
                n = n_idx + 1
                val += amps[n_idx] * np.cos(2 * np.pi * n * (j - k) / N + phases[n_idx])
            H[j, k] = val
            H[k, j] = val

    scale = np.sqrt(np.mean(H ** 2) * N)
    if scale > 1e-10:
        H /= scale
    return H


def build_z_toeplitz(T, N, delta=None):
    """Z-Toeplitz: entries are Z(T + (j-k)*delta).

    H_{jk} = Z(T + (j-k)*delta) / C
    where delta is chosen so N*delta ~ mean gap.
    """
    if delta is None:
        # Mean gap at height T is 2*pi/log(T/(2*pi))
        mean_gap = 2 * np.pi / np.log(T / (2 * np.pi))
        delta = mean_gap / N  # one gap spread across the matrix

    H = np.zeros((N, N))
    # Cache Z values
    z_vals = {}
    for d in range(-(N - 1), N):
        t_val = T + d * delta
        if t_val > 10:
            z_vals[d] = float(mpmath.siegelz(t_val))
        else:
            z_vals[d] = 0.0

    for j in range(N):
        for k in range(j, N):
            d = j - k
            H[j, k] = z_vals[d]
            H[k, j] = z_vals[d]

    scale = np.sqrt(np.mean(H ** 2) * N)
    if scale > 1e-10:
        H /= scale
    return H


def build_log_freq_rs(T, N, N_sum=None):
    """Log-frequency RS: spatial oscillation at log(n)/log(N).

    H_{jk} = (1/C) * sum_n cos(phi_n(T) + 2*pi*(j-k)*log(n)/log(N)) / sqrt(n)

    The log-frequency puts primes at rationally related positions,
    introducing number-theoretic structure in the eigenvalues.
    """
    if N_sum is None:
        N_sum = max(int(np.sqrt(T / (2 * np.pi))), 5)
        N_sum = min(N_sum, N)

    theta_T = float(mpmath.siegeltheta(T))
    log_N = np.log(N)

    H = np.zeros((N, N))
    for j in range(N):
        for k in range(j, N):
            val = 0.0
            for n in range(1, N_sum + 1):
                phi_n = theta_T - T * np.log(n)
                val += np.cos(phi_n + 2 * np.pi * (j - k) * np.log(n) / log_N) / np.sqrt(n)
            H[j, k] = val
            H[k, j] = val

    scale = np.sqrt(np.mean(H ** 2) * N)
    if scale > 1e-10:
        H /= scale
    return H


def build_noncirculant_rs(T, N, N_sum=None):
    """Non-circulant: phase depends on j+k (not j-k), breaking shift symmetry.

    H_{jk} = (1/C) * sum_n cos(phi_n(T) + pi*n*(j+k)/N) / sqrt(n) * cos(pi*n*(j-k)/N)

    This is NOT circulant because it depends on j+k separately.
    The cos(pi*n*(j+k)/N) factor creates a position-dependent modulation.
    """
    if N_sum is None:
        N_sum = max(int(np.sqrt(T / (2 * np.pi))), 5)
        N_sum = min(N_sum, N)

    theta_T = float(mpmath.siegeltheta(T))

    H = np.zeros((N, N))
    for j in range(N):
        for k in range(j, N):
            val = 0.0
            for n in range(1, N_sum + 1):
                phi_n = theta_T - T * np.log(n)
                # Product of two cosines: one depends on j-k, one on j+k
                val += np.cos(phi_n + np.pi * n * (j - k) / N) * \
                       np.cos(np.pi * n * (j + k) / N) / np.sqrt(n)
            H[j, k] = val
            H[k, j] = val

    scale = np.sqrt(np.mean(H ** 2) * N)
    if scale > 1e-10:
        H /= scale
    return H


# ============================================================
# TEST 1: Single-T comparison of all variants
# ============================================================
print("=" * 70)
print("TEST 1: ALL VARIANTS AT INDIVIDUAL T VALUES")
print("=" * 70)

N = 100
rng = np.random.default_rng(42)

variants = [
    ("Original (mult coeff)", lambda T: build_original_rs(T, N)),
    ("Additive phase RS", lambda T: build_additive_phase_rs(T, N)),
    ("Random phase", lambda T: build_random_phase(N, max(int(np.sqrt(T/(2*np.pi))),5), rng)),
    ("Z-Toeplitz", lambda T: build_z_toeplitz(T, N)),
    ("Log-frequency RS", lambda T: build_log_freq_rs(T, N)),
    ("Non-circulant RS", lambda T: build_noncirculant_rs(T, N)),
]

T_values = [200, 500, 1000, 2000]

print(f"\n  N={N}")
print(f"  {'Variant':<25} " + " ".join(f"{'T='+str(T):>10}" for T in T_values) + f" {'mean':>8}")
print(f"  {'-'*(25 + 11*len(T_values) + 9)}")

for name, builder in variants:
    rs = []
    for T in T_values:
        H = builder(T)
        eigs = np.linalg.eigvalsh(H)
        r, _ = measure_peak_gap(eigs)
        rs.append(r)
    mean_r = np.mean(rs)
    row = f"  {name:<25}"
    for r in rs:
        row += f" {r:>+10.4f}"
    row += f" {mean_r:>+8.4f}"
    print(row)


# ============================================================
# TEST 2: Averaged eigenvalue statistics (GUE test)
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: AVERAGED EIGENVALUE STATISTICS (GUE TEST)")
print("=" * 70)

N_mat = 80
T_sweep = np.linspace(200, 2000, 60)

print(f"\n  N={N_mat}, {len(T_sweep)} T-values in [{T_sweep[0]:.0f}, {T_sweep[-1]:.0f}]")
print(f"  {'Variant':<25} {'mean_r':>8} {'p(GUE)':>8} {'p(Poi)':>8} {'n_sp':>8} {'verdict':>10}")
print(f"  {'-'*75}")

for name, builder_func in [
    ("Original", lambda T: build_original_rs(T, N_mat)),
    ("Additive phase", lambda T: build_additive_phase_rs(T, N_mat)),
    ("Random phase", lambda T: build_random_phase(N_mat, max(int(np.sqrt(T/(2*np.pi))),5), np.random.default_rng(int(T*100)))),
    ("Z-Toeplitz", lambda T: build_z_toeplitz(T, N_mat)),
    ("Log-frequency", lambda T: build_log_freq_rs(T, N_mat)),
    ("Non-circulant", lambda T: build_noncirculant_rs(T, N_mat)),
]:
    all_spacings = []
    all_rs = []

    for T in T_sweep:
        H = builder_func(T)
        eigs = np.linalg.eigvalsh(H)
        sp = polynomial_unfold(eigs, trim_fraction=0.1)
        if len(sp) > 10:
            sp = sp / np.mean(sp)
            all_spacings.extend(sp.tolist())
        r, _ = measure_peak_gap(eigs)
        all_rs.append(r)

    all_spacings = np.array(all_spacings)
    mean_r = np.mean(all_rs)
    _, p_gue = kstest(all_spacings, wigner_cdf) if len(all_spacings) > 20 else (0, 0)
    _, p_poi = kstest(all_spacings, "expon", args=(0, 1)) if len(all_spacings) > 20 else (0, 0)

    if p_gue > 0.05 and mean_r > 0.3:
        verdict = "BOTH!"
    elif p_gue > 0.05:
        verdict = "GUE only"
    elif mean_r > 0.3:
        verdict = "r only"
    else:
        verdict = "neither"

    print(f"  {name:<25} {mean_r:>+8.4f} {p_gue:>8.4f} {p_poi:>8.4f} "
          f"{len(all_spacings):>8} {verdict:>10}")


# ============================================================
# TEST 3: N-scaling for the best variant(s)
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: N-SCALING FOR TOP VARIANTS")
print("=" * 70)

T_fixed = 1000
T_range = np.linspace(500, 2000, 30)

for name, builder_gen in [
    ("Additive phase", lambda T, N: build_additive_phase_rs(T, N)),
    ("Z-Toeplitz", lambda T, N: build_z_toeplitz(T, N)),
    ("Log-frequency", lambda T, N: build_log_freq_rs(T, N)),
    ("Non-circulant", lambda T, N: build_noncirculant_rs(T, N)),
]:
    print(f"\n  {name}:")
    print(f"  {'N':>6} {'r(T=1000)':>12} {'mean_r':>10} {'p(GUE)':>10}")
    print(f"  {'-'*42}")

    for N_test in [50, 80, 120, 160]:
        # Single T
        H = builder_gen(T_fixed, N_test)
        eigs = np.linalg.eigvalsh(H)
        r_single, _ = measure_peak_gap(eigs)

        # Averaged over T
        rs_avg = []
        spacings_avg = []
        for T in T_range:
            H = builder_gen(T, N_test)
            eigs = np.linalg.eigvalsh(H)
            r, _ = measure_peak_gap(eigs)
            rs_avg.append(r)
            sp = polynomial_unfold(eigs, trim_fraction=0.1)
            if len(sp) > 10:
                spacings_avg.extend((sp / np.mean(sp)).tolist())

        mean_r = np.mean(rs_avg)
        _, p_g = kstest(np.array(spacings_avg), wigner_cdf) if len(spacings_avg) > 20 else (0, 0)
        print(f"  {N_test:>6} {r_single:>+12.4f} {mean_r:>+10.4f} {p_g:>10.4f}")


# ============================================================
# TEST 4: Z-Toeplitz at various delta scales
# ============================================================
print("\n" + "=" * 70)
print("TEST 4: Z-TOEPLITZ DELTA SENSITIVITY")
print("=" * 70)

N_zt = 80
T_zt = 1000
mean_gap = 2 * np.pi / np.log(T_zt / (2 * np.pi))

print(f"\n  T={T_zt}, N={N_zt}, mean_gap={mean_gap:.4f}")
print(f"  {'delta/gap':>12} {'delta':>10} {'r':>8} {'p(GUE)':>8}")
print(f"  {'-'*42}")

for delta_frac in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
    delta = mean_gap * delta_frac / N_zt
    H = build_z_toeplitz(T_zt, N_zt, delta=delta)
    eigs = np.linalg.eigvalsh(H)
    r, _ = measure_peak_gap(eigs)
    sp = polynomial_unfold(eigs, trim_fraction=0.1)
    if len(sp) > 10:
        sp = sp / np.mean(sp)
        _, p_g = kstest(sp, wigner_cdf)
    else:
        p_g = 0
    print(f"  {delta_frac:>12.1f} {delta:>10.6f} {r:>+8.4f} {p_g:>8.4f}")


# ============================================================
# TEST 5: Head-to-head at zeta's own scale
# ============================================================
print("\n" + "=" * 70)
print("TEST 5: HEAD-TO-HEAD AT ZETA SCALE (T ~ zeros)")
print("=" * 70)

# Use actual zero heights for T
zeros = np.load("_zeros_200.npy")
N_h2h = 80

print(f"\n  Using {len(zeros)} zero heights, N={N_h2h}")
print(f"  {'Variant':<25} {'mean_r':>8} {'std_r':>8} {'min_r':>8} {'max_r':>8}")
print(f"  {'-'*60}")

for name, builder_func in [
    ("Original", lambda T: build_original_rs(T, N_h2h)),
    ("Additive phase", lambda T: build_additive_phase_rs(T, N_h2h)),
    ("Z-Toeplitz", lambda T: build_z_toeplitz(T, N_h2h)),
    ("Log-frequency", lambda T: build_log_freq_rs(T, N_h2h)),
    ("Non-circulant", lambda T: build_noncirculant_rs(T, N_h2h)),
]:
    rs = []
    for T in zeros[::5]:  # Every 5th zero to save time
        if T > 20:
            H = builder_func(T)
            eigs = np.linalg.eigvalsh(H)
            r, pts = measure_peak_gap(eigs)
            if pts > 10:
                rs.append(r)

    rs = np.array(rs)
    if len(rs) > 0:
        print(f"  {name:<25} {np.mean(rs):>+8.4f} {np.std(rs):>8.4f} "
              f"{np.min(rs):>+8.4f} {np.max(rs):>+8.4f}")
    else:
        print(f"  {name:<25} {'N/A':>8}")


# ============================================================
# VERDICT
# ============================================================
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

print(f"\nTotal time: {time.time() - t0:.1f}s")
