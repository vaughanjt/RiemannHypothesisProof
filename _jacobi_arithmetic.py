"""Test: does the Jacobi matrix contain arithmetic structure beyond Weyl?

STRATEGY:
1. Build Jacobi from actual zeta zeros (the "signal")
2. Build Jacobi from WEYL zeros (smooth approximation — no primes)
3. Build Jacobi from GUE eigenvalues (random, scaled to match density)
4. The DIFFERENCE (zeta - Weyl) is the arithmetic content
5. Look for PRIME FREQUENCIES in the residual beta values
6. Test: can you reconstruct the zeros from Weyl + prime corrections?

If the beta residuals show prime-frequency structure matching the
Selberg/Montgomery amplitude law, the Jacobi matrix IS encoding
the arithmetic modulation — and in a SELF-ADJOINT form.
"""
import sys
import time

sys.path.insert(0, "src")
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.stats import pearsonr, kstest
import mpmath

t0 = time.time()
mpmath.mp.dps = 20


def lanczos_from_eigenvalues(eigenvalues, start_vec):
    """Lanczos on diag(eigenvalues) with given starting vector."""
    N = len(eigenvalues)
    eigs = eigenvalues.astype(float)
    v = start_vec / np.linalg.norm(start_vec)

    alpha = np.zeros(N)
    beta = np.zeros(N - 1)
    V = np.zeros((N, N))
    V[:, 0] = v

    w = eigs * v
    alpha[0] = np.dot(v, w)
    w = w - alpha[0] * v

    for k in range(1, N):
        beta[k - 1] = np.linalg.norm(w)
        if beta[k - 1] < 1e-14:
            return alpha[:k], beta[:k - 1]
        v_new = w / beta[k - 1]
        for j in range(k):
            v_new -= np.dot(V[:, j], v_new) * V[:, j]
        v_new /= np.linalg.norm(v_new)
        V[:, k] = v_new
        w = eigs * v_new
        alpha[k] = np.dot(v_new, w)
        w = w - alpha[k] * v_new - beta[k - 1] * V[:, k - 1]

    return alpha, beta


# ============================================================
# Compute zeros
# ============================================================
print("Computing 300 zeta zeros...")
t_start = time.time()
N_zeros = 300
zeta_zeros = np.array([float(mpmath.im(mpmath.zetazero(k))) for k in range(1, N_zeros + 1)])
print(f"  Done ({time.time()-t_start:.1f}s)")

# Weyl approximation: t_n ~ 2*pi*n / W(n/e) where W is Lambert W
# Simpler: t_n such that N(t_n) = n, where N(t) = t/(2pi) * log(t/(2pi)) - t/(2pi)
# Use Newton's method to invert
def weyl_zero(n):
    """Compute the n-th Weyl-law zero (smooth approximation)."""
    # N(t) = (t/2pi)*log(t/2pi) - t/2pi + 7/8
    # Solve N(t) = n
    t = 2 * np.pi * n / np.log(n + 2)  # Initial guess
    for _ in range(20):
        if t < 1:
            t = 10.0
        Nt = t / (2 * np.pi) * np.log(t / (2 * np.pi)) - t / (2 * np.pi) + 7 / 8
        dNt = np.log(t / (2 * np.pi)) / (2 * np.pi)
        if abs(dNt) < 1e-30:
            break
        t = t - (Nt - n) / dNt
    return t

print("Computing Weyl zeros...")
weyl_zeros = np.array([weyl_zero(n) for n in range(1, N_zeros + 1)])
print(f"  Weyl vs actual: mean |diff| = {np.mean(np.abs(weyl_zeros - zeta_zeros)):.4f}")

# GUE eigenvalues (10 realizations, scaled to zeta density)
print("Computing GUE eigenvalues...")
rng = np.random.default_rng(42)
gue_sets = []
for trial in range(10):
    G = rng.standard_normal((N_zeros, N_zeros))
    H = (G + G.T) / (2 * np.sqrt(N_zeros))
    eigs = np.linalg.eigvalsh(H)
    # Scale to match zeta zero range and density
    eigs_scaled = np.interp(
        np.linspace(0, 1, N_zeros),
        np.linspace(0, 1, N_zeros),
        np.sort(eigs)
    )
    # Map to zeta range using Weyl law
    eigs_mapped = np.interp(eigs_scaled, np.linspace(eigs[0], eigs[-1], N_zeros), weyl_zeros)
    gue_sets.append(eigs_mapped)


# ============================================================
# BUILD JACOBI MATRICES
# ============================================================
print("\nBuilding Jacobi matrices...")
v0 = np.ones(N_zeros) / np.sqrt(N_zeros)

alpha_z, beta_z = lanczos_from_eigenvalues(zeta_zeros, v0)
alpha_w, beta_w = lanczos_from_eigenvalues(weyl_zeros, v0)

gue_alphas, gue_betas = [], []
for gue_eigs in gue_sets:
    a, b = lanczos_from_eigenvalues(gue_eigs, v0)
    gue_alphas.append(a)
    gue_betas.append(b)

print(f"  Zeta Jacobi: {len(alpha_z)} diagonal, {len(beta_z)} off-diagonal")
print(f"  Weyl Jacobi: {len(alpha_w)} diagonal, {len(beta_w)} off-diagonal")


# ============================================================
# TEST 1: Beta residuals (zeta - Weyl)
# ============================================================
print("\n" + "=" * 70)
print("TEST 1: BETA RESIDUALS (ZETA - WEYL)")
print("=" * 70)

n_common = min(len(beta_z), len(beta_w))
beta_residual = np.abs(beta_z[:n_common]) - np.abs(beta_w[:n_common])

print(f"\n  Residual |beta_zeta| - |beta_weyl|:")
print(f"    Mean:  {np.mean(beta_residual):+.6f}")
print(f"    Std:   {np.std(beta_residual):.6f}")
print(f"    Max:   {np.max(np.abs(beta_residual)):.6f}")
print(f"    Relative std: {np.std(beta_residual)/np.mean(np.abs(beta_z[:n_common])):.4f}")

# Compare to GUE residuals
gue_residuals = []
for b_g in gue_betas:
    n_c = min(len(b_g), len(beta_w))
    res = np.abs(b_g[:n_c]) - np.abs(beta_w[:n_c])
    gue_residuals.append(np.std(res))

print(f"\n  Residual std comparison:")
print(f"    Zeta:     {np.std(beta_residual):.6f}")
print(f"    GUE mean: {np.mean(gue_residuals):.6f} +/- {np.std(gue_residuals):.6f}")
print(f"    Zeta/GUE: {np.std(beta_residual)/np.mean(gue_residuals):.4f}")

if np.std(beta_residual) < np.mean(gue_residuals):
    print(f"    >>> Zeta betas are CLOSER to Weyl than GUE (more regular)")
else:
    print(f"    >>> Zeta betas deviate MORE than GUE from Weyl")


# ============================================================
# TEST 2: Fourier analysis of beta residuals — prime frequencies?
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: FOURIER ANALYSIS OF BETA RESIDUALS")
print("=" * 70)

# Remove linear trend from beta residual
k = np.arange(n_common)
trend = np.polyfit(k, beta_residual, 1)
beta_detrended = beta_residual - np.polyval(trend, k)

# FFT
fft_beta = np.abs(np.fft.rfft(beta_detrended))
freqs = np.fft.rfftfreq(n_common)

# The "frequency" in the Jacobi index space maps to a frequency
# in the zero-counting function. If the Jacobi encodes primes,
# the FFT peaks should be at frequencies related to log(p).
#
# In the ACF analysis, primes appeared at theta_p = log(p)/(2*pi*log(T))
# For the Jacobi index, the natural variable is n (zero count),
# so the prime frequencies should be at f_p ~ log(p) / (2*pi * average_spacing)

mean_spacing = np.mean(np.diff(zeta_zeros))
prime_freqs = {p: np.log(p) / (2 * np.pi * mean_spacing) for p in [2, 3, 5, 7, 11, 13]}

print(f"\n  Mean zero spacing: {mean_spacing:.4f}")
print(f"  Expected prime frequencies in Jacobi space:")
for p, f_p in prime_freqs.items():
    print(f"    p={p:>2}: f = {f_p:.6f} (period = {1/f_p:.1f})")

# Find top FFT peaks
n_peaks = 10
peak_indices = np.argsort(fft_beta[1:])[::-1][:n_peaks] + 1  # skip DC
print(f"\n  Top {n_peaks} FFT peaks:")
print(f"  {'Rank':>4} {'Freq':>10} {'Period':>10} {'Amplitude':>12} {'Near prime?':>15}")
print(f"  {'-'*55}")

for rank, idx in enumerate(peak_indices):
    f = freqs[idx]
    period = 1 / f if f > 0 else float("inf")
    amp = fft_beta[idx]
    # Check if near any prime frequency
    nearest_p = min(prime_freqs.items(), key=lambda x: abs(x[1] - f))
    dist = abs(nearest_p[1] - f)
    tag = f"p={nearest_p[0]} (d={dist:.4f})" if dist < 0.01 else ""
    print(f"  {rank+1:>4} {f:>10.6f} {period:>10.1f} {amp:>12.4f} {tag:>15}")


# ============================================================
# TEST 3: Same analysis on GUE beta residuals (null hypothesis)
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: GUE BETA RESIDUALS (NULL HYPOTHESIS)")
print("=" * 70)

# For each GUE realization, compute FFT of beta residual
gue_fft_amps = np.zeros((len(gue_betas), len(freqs)))
for i, b_g in enumerate(gue_betas):
    n_c = min(len(b_g), len(beta_w))
    res = np.abs(b_g[:n_c]) - np.abs(beta_w[:n_c])
    trend_g = np.polyfit(np.arange(n_c), res, 1)
    res_det = res - np.polyval(trend_g, np.arange(n_c))
    fft_g = np.abs(np.fft.rfft(res_det, n=n_common))
    gue_fft_amps[i, :len(fft_g)] = fft_g[:len(freqs)]

gue_mean_fft = np.mean(gue_fft_amps, axis=0)
gue_std_fft = np.std(gue_fft_amps, axis=0)

# Z-score of zeta FFT peaks relative to GUE distribution
z_scores = (fft_beta - gue_mean_fft) / (gue_std_fft + 1e-10)

print(f"\n  Z-scores of zeta beta FFT peaks (relative to GUE):")
print(f"  {'Freq':>10} {'Zeta amp':>10} {'GUE mean':>10} {'GUE std':>10} {'z-score':>10} {'Near prime':>12}")
print(f"  {'-'*65}")

# Show top z-scores
top_z = np.argsort(np.abs(z_scores[1:]))[::-1][:15] + 1
for idx in top_z:
    f = freqs[idx]
    nearest_p = min(prime_freqs.items(), key=lambda x: abs(x[1] - f))
    dist_p = abs(nearest_p[1] - f)
    tag = f"p={nearest_p[0]}" if dist_p < 0.01 else ""
    print(f"  {f:>10.6f} {fft_beta[idx]:>10.4f} {gue_mean_fft[idx]:>10.4f} "
          f"{gue_std_fft[idx]:>10.4f} {z_scores[idx]:>+10.2f} {tag:>12}")

# How many z-scores exceed 2 sigma?
n_sig = np.sum(np.abs(z_scores[1:]) > 2)
n_expected = int(0.05 * len(z_scores[1:]))
print(f"\n  Significant (|z|>2): {n_sig} (expected by chance: ~{n_expected})")


# ============================================================
# TEST 4: Eigenvalue sensitivity to prime-frequency beta perturbation
# ============================================================
print("\n" + "=" * 70)
print("TEST 4: EIGENVALUE SENSITIVITY TO BETA PERTURBATION")
print("=" * 70)

# Test: if we REMOVE the prime-frequency content from beta,
# how much do the eigenvalues shift?

# Original eigenvalues
eigs_orig = eigh_tridiagonal(alpha_z, beta_z, eigvals_only=True)

# Remove all Fourier content above a frequency cutoff
for cutoff_frac in [0.01, 0.05, 0.1, 0.5]:
    beta_filtered = beta_z.copy()
    fft_b = np.fft.rfft(np.abs(beta_filtered))
    n_keep = max(int(cutoff_frac * len(fft_b)), 1)
    fft_b[n_keep:] = 0
    beta_smooth = np.fft.irfft(fft_b, n=len(beta_z))
    # Preserve sign
    beta_mod = np.sign(beta_z) * np.abs(beta_smooth)

    try:
        eigs_mod = eigh_tridiagonal(alpha_z, beta_mod, eigvals_only=True)
        shift = np.mean(np.abs(np.sort(eigs_mod) - np.sort(eigs_orig)))
        max_shift = np.max(np.abs(np.sort(eigs_mod) - np.sort(eigs_orig)))
        print(f"  Keep {cutoff_frac*100:>5.1f}% of beta spectrum: "
              f"mean shift = {shift:.4f}, max shift = {max_shift:.4f}")
    except Exception as e:
        print(f"  Keep {cutoff_frac*100:>5.1f}%: failed ({e})")

# Now specifically remove prime frequencies
print(f"\n  Remove specific prime frequencies:")
for p in [2, 3, 5, 7]:
    beta_no_prime = beta_z.copy()
    fft_b = np.fft.rfft(np.abs(beta_no_prime))
    # Zero out the bin nearest to the prime frequency
    f_p = prime_freqs[p]
    idx_p = np.argmin(np.abs(freqs - f_p))
    fft_b[max(0, idx_p - 1):idx_p + 2] = 0  # Remove peak +/- 1 bin
    beta_nop = np.fft.irfft(fft_b, n=len(beta_z))
    beta_nop = np.sign(beta_z) * np.abs(beta_nop)

    try:
        eigs_nop = eigh_tridiagonal(alpha_z, beta_nop, eigvals_only=True)
        shift = np.mean(np.abs(np.sort(eigs_nop) - np.sort(eigs_orig)))
        print(f"  Remove p={p}: mean eigenvalue shift = {shift:.6f}")
    except Exception as e:
        print(f"  Remove p={p}: failed ({e})")


# ============================================================
# TEST 5: Alpha residuals (zeta - Weyl) — same analysis
# ============================================================
print("\n" + "=" * 70)
print("TEST 5: ALPHA RESIDUALS")
print("=" * 70)

n_a = min(len(alpha_z), len(alpha_w))
alpha_residual = alpha_z[:n_a] - alpha_w[:n_a]

# Detrend
trend_a = np.polyfit(np.arange(n_a), alpha_residual, 1)
alpha_det = alpha_residual - np.polyval(trend_a, np.arange(n_a))

fft_alpha = np.abs(np.fft.rfft(alpha_det))
freqs_a = np.fft.rfftfreq(n_a)

print(f"\n  Alpha residual (zeta - Weyl):")
print(f"    Mean: {np.mean(alpha_residual):+.4f}")
print(f"    Std:  {np.std(alpha_residual):.4f}")

# Top FFT peaks
peak_a = np.argsort(fft_alpha[1:])[::-1][:10] + 1
print(f"\n  Top alpha FFT peaks:")
print(f"  {'Freq':>10} {'Period':>10} {'Amplitude':>12} {'Near prime':>15}")
for idx in peak_a[:8]:
    f = freqs_a[idx]
    period = 1 / f if f > 0 else float("inf")
    nearest_p = min(prime_freqs.items(), key=lambda x: abs(x[1] - f))
    dist_p = abs(nearest_p[1] - f)
    tag = f"p={nearest_p[0]} (d={dist_p:.4f})" if dist_p < 0.015 else ""
    print(f"  {f:>10.6f} {period:>10.1f} {fft_alpha[idx]:>12.4f} {tag:>15}")


# ============================================================
# VERDICT
# ============================================================
print("\n" + "=" * 70)
print("VERDICT: ARITHMETIC CONTENT IN THE JACOBI MATRIX")
print("=" * 70)

# Summarize
has_prime_structure = n_sig > 2 * n_expected  # More significant peaks than expected
print(f"\n  Significant FFT peaks in beta residual: {n_sig} (expected ~{n_expected})")
print(f"  Prime structure detected: {'YES' if has_prime_structure else 'NO'}")

print(f"\nTotal time: {time.time() - t0:.1f}s")
