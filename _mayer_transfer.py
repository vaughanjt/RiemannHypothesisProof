"""The Mayer transfer operator for the Gauss map.

The Gauss map T(x) = {1/x} on (0,1] generates continued fraction dynamics.
Its transfer operator L_s acts on functions by:
  (L_s f)(z) = sum_{n=1}^inf (z + n)^{-2s} f(1/(z+n))

Key properties:
  - Trace = sum over periodic orbits (integers!)
  - Fredholm determinant relates to the Selberg zeta function for SL(2,Z)
  - The Selberg zeta function factors through the Riemann zeta function
  - Eigenvectors are arithmetically structured (continued fraction expansions)

This is the operator whose resolvent trace IS a sum over integers with
arithmetic weights — exactly the Riemann-Siegel sum structure we identified.

Test: compute the Mayer operator at s = 1/2 + it for various t,
find its eigenvalues, and measure:
  1. Peak-gap correlation (target: r ~ 0.80)
  2. Eigenvalue statistics (target: GUE)
  3. Connection to zeta zeros
"""
import sys, time
sys.path.insert(0, 'src')
import numpy as np
from scipy.stats import pearsonr, kstest
from riemann.analysis.bost_connes_operator import polynomial_unfold

t0 = time.time()

# ============================================================
# BUILD THE MAYER TRANSFER OPERATOR
# ============================================================
# We discretize L_s in the monomial basis {z^k}_{k=0}^{N-1} on the
# interval [0, 1]. The matrix element is:
#
#   L_{jk} = sum_{n=1}^{M} integral_0^1 z^j * (z+n)^{-2s} * (1/(z+n))^k dz
#
# For s = 1/2 + it (on the critical line):
#   (z+n)^{-2s} = (z+n)^{-1-2it} = (z+n)^{-1} * (z+n)^{-2it}
#
# The integral can be computed numerically for each (j, k, n).

def build_mayer_matrix(s, N_basis, N_terms, n_quad=100):
    """Build the Mayer transfer operator L_s truncated to N_basis x N_basis.

    s: complex parameter (use s = 0.5 + i*t for critical line)
    N_basis: truncation order (matrix size)
    N_terms: number of terms in the sum over n
    n_quad: quadrature points for the integral
    """
    # Gauss-Legendre quadrature on [0, 1]
    x_gl, w_gl = np.polynomial.legendre.leggauss(n_quad)
    # Map from [-1,1] to [0,1]
    x = (x_gl + 1) / 2
    w = w_gl / 2

    L = np.zeros((N_basis, N_basis), dtype=complex)

    for n in range(1, N_terms + 1):
        for j in range(N_basis):
            for k in range(N_basis):
                # Integrand: z^j * (z+n)^{-2s} * (1/(z+n))^k
                # = z^j * (z+n)^{-2s-k}
                integrand = x ** j * (x + n) ** (-2 * s - k)
                L[j, k] += np.sum(w * integrand)

    return L


# ============================================================
# STEP 1: Eigenvalues of L_s at s = 1/2
# ============================================================
print('=' * 70)
print('STEP 1: MAYER OPERATOR AT s = 1/2 (real axis)')
print('=' * 70)

N_basis = 30
N_terms = 50

print(f'  Basis size: {N_basis}, Sum terms: {N_terms}')
L_half = build_mayer_matrix(0.5, N_basis, N_terms)
eigs_half = np.linalg.eigvals(L_half)
eigs_half_sorted = np.sort(np.real(eigs_half))[::-1]

print(f'  Top 10 eigenvalues of L_{{1/2}}:')
for i in range(min(10, len(eigs_half_sorted))):
    print(f'    lambda_{i+1} = {eigs_half_sorted[i]:+.6f}')

# The leading eigenvalue should be 1 (the Perron-Frobenius eigenvalue
# for the Gauss map, related to the density of the Gauss measure)
print(f'\n  Leading eigenvalue: {eigs_half_sorted[0]:.6f} (should be ~1.0)')

# ============================================================
# STEP 2: L_s on the critical line: s = 1/2 + it
# ============================================================
print('\n' + '=' * 70)
print('STEP 2: MAYER OPERATOR ON THE CRITICAL LINE s = 1/2 + it')
print('=' * 70)

# The Fredholm determinant det(I - L_s) has zeros related to the
# Selberg zeta function. For SL(2,Z), these connect to zeta zeros.
#
# Compute det(I - L_s) at many t values and look for sign changes
# (zeros of the real part when the determinant is real-valued on
# the critical line... actually det is complex for complex s).
#
# Instead, compute |det(I - L_s)| and look for minima.

print(f'  Scanning |det(I - L_s)| along critical line...')

t_values = np.linspace(5, 100, 500)
det_vals = np.zeros(len(t_values), dtype=complex)

for i, t_val in enumerate(t_values):
    s = 0.5 + 1j * t_val
    L = build_mayer_matrix(s, N_basis, N_terms, n_quad=50)
    det_vals[i] = np.linalg.det(np.eye(N_basis) - L)

abs_det = np.abs(det_vals)

# Find local minima (candidate zeros)
minima = []
for i in range(1, len(abs_det) - 1):
    if abs_det[i] < abs_det[i-1] and abs_det[i] < abs_det[i+1]:
        if abs_det[i] < 0.1 * np.median(abs_det):
            minima.append((t_values[i], abs_det[i]))

print(f'  Found {len(minima)} deep minima of |det(I - L_s)|:')
for t_min, d_min in minima[:15]:
    print(f'    t = {t_min:.4f}, |det| = {d_min:.6f}')

# Compare to known zeta zeros
import mpmath
mpmath.mp.dps = 15
known_zeros = []
for i in range(1, 30):
    z = float(mpmath.zetazero(i).imag)
    known_zeros.append(z)
    if z > 100:
        break

print(f'\n  Known zeta zeros (imaginary parts):')
for z in known_zeros[:15]:
    print(f'    gamma = {z:.4f}')

# Match minima to known zeros
if minima:
    print(f'\n  Matching Mayer minima to zeta zeros:')
    print(f'  {"Mayer min":>12} {"Nearest zero":>14} {"Distance":>10}')
    for t_min, d_min in minima[:10]:
        dists = [abs(t_min - z) for z in known_zeros]
        best = np.argmin(dists)
        print(f'  {t_min:>12.4f} {known_zeros[best]:>14.4f} {dists[best]:>10.4f}')

# ============================================================
# STEP 3: Eigenvalue statistics of L_s
# ============================================================
print('\n' + '=' * 70)
print('STEP 3: EIGENVALUE STATISTICS')
print('=' * 70)

# Collect eigenvalues at multiple t values
all_eig_magnitudes = []
N_basis_stat = 25
N_terms_stat = 40

for t_val in np.linspace(20, 80, 30):
    s = 0.5 + 1j * t_val
    L = build_mayer_matrix(s, N_basis_stat, N_terms_stat, n_quad=40)
    eigs = np.linalg.eigvals(L)
    # The eigenvalues are complex; look at their magnitudes
    mags = np.sort(np.abs(eigs))[::-1]
    all_eig_magnitudes.append(mags)

# Analyze the spectrum
mags_flat = np.concatenate(all_eig_magnitudes)
print(f'  Collected {len(mags_flat)} eigenvalue magnitudes from 30 t-values')
print(f'  Magnitude range: {np.min(mags_flat):.6f} to {np.max(mags_flat):.6f}')
print(f'  Mean: {np.mean(mags_flat):.6f}, Median: {np.median(mags_flat):.6f}')

# The eigenvalues of L_s decay rapidly (it's a trace-class operator)
# Look at the PHASES of the eigenvalues instead
all_phases = []
for t_val in np.linspace(20, 80, 30):
    s = 0.5 + 1j * t_val
    L = build_mayer_matrix(s, N_basis_stat, N_terms_stat, n_quad=40)
    eigs = np.linalg.eigvals(L)
    phases = np.angle(eigs)
    all_phases.extend(phases.tolist())

all_phases = np.array(all_phases)
# Phase distribution — uniform would be GUE-like
print(f'\n  Phase distribution of eigenvalues:')
print(f'    Mean phase: {np.mean(all_phases):.4f}')
print(f'    Std phase: {np.std(all_phases):.4f} (uniform on [-pi,pi] gives std = {np.pi/np.sqrt(3):.4f})')

# ============================================================
# STEP 4: Trace of L_s — the periodic orbit sum
# ============================================================
print('\n' + '=' * 70)
print('STEP 4: TRACE OF L_s — THE PERIODIC ORBIT SUM')
print('=' * 70)

# Tr(L_s) = sum_{n=1}^inf integral_0^1 (z+n)^{-2s} * delta(z - 1/(z+n)) dz
# The fixed points of z -> 1/(z+n) satisfy z(z+n) = 1, i.e., z^2 + nz - 1 = 0
# z_n = (-n + sqrt(n^2 + 4)) / 2 (the positive root)
#
# The trace contribution from the n-th fixed point is:
# Tr_n = |z_n + n|^{-2s} / |1 - (z_n + n)^{-2}|
#       = (z_n + n)^{-2s} / (1 - (z_n + n)^{-2})
#
# This IS a sum over integers with specific weights — the Riemann-Siegel structure!

def mayer_trace(s, N_terms=100):
    """Compute Tr(L_s) as a sum over fixed points."""
    tr = 0
    for n in range(1, N_terms + 1):
        z_n = (-n + np.sqrt(n ** 2 + 4)) / 2
        lambda_n = z_n + n  # = 1/z_n
        # Fixed point contribution
        tr += lambda_n ** (-2 * s) / (1 - lambda_n ** (-2))
    return tr

# Compute trace along the critical line
print(f'  Tr(L_s) along critical line:')
print(f'  {"t":>8} {"Re(Tr)":>12} {"Im(Tr)":>12} {"|Tr|":>12}')
for t_val in [10, 14.13, 21.02, 25.01, 30.42, 50, 100]:
    s = 0.5 + 1j * t_val
    tr = mayer_trace(s, 200)
    print(f'  {t_val:>8.2f} {np.real(tr):>+12.6f} {np.imag(tr):>+12.6f} {np.abs(tr):>12.6f}')

# Check: does the trace oscillate with prime-like frequencies?
t_dense = np.linspace(10, 100, 1000)
tr_dense = np.array([mayer_trace(0.5 + 1j * t, 200) for t in t_dense])

# Fourier analysis of the trace
tr_real = np.real(tr_dense)
fft_tr = np.fft.rfft(tr_real - np.mean(tr_real))
power_tr = np.abs(fft_tr) ** 2
freq_tr = np.fft.rfftfreq(len(t_dense), d=(t_dense[1] - t_dense[0]))

# Find peaks
top_peaks = np.argsort(power_tr[1:])[-10:][::-1] + 1
print(f'\n  Fourier peaks of Re(Tr(L_s)):')
print(f'  {"Rank":>5} {"Freq":>10} {"Period":>10} {"log(p)?":>15}')
from sympy import primerange
primes_check = list(primerange(2, 50))
for rank, idx in enumerate(top_peaks):
    freq = freq_tr[idx]
    period = 1 / freq if freq > 0 else float('inf')
    match = ''
    for p in primes_check:
        if abs(freq - np.log(p) / (2 * np.pi)) < 0.02:
            match = f'log({p})/2pi'
            break
    print(f'  {rank+1:>5} {freq:>10.4f} {period:>10.2f} {match:>15}')

# ============================================================
# VERDICT
# ============================================================
print('\n' + '=' * 70)
print('VERDICT: MAYER TRANSFER OPERATOR')
print('=' * 70)

if minima:
    # Check how many minima match zeta zeros
    n_match = 0
    for t_min, _ in minima:
        if min(abs(t_min - z) for z in known_zeros) < 0.5:
            n_match += 1
    print(f'\n  Mayer minima matching zeta zeros (within 0.5): {n_match}/{len(minima)}')

print(f'\n  The Mayer transfer operator:')
print(f'    - Has trace = sum over integers (periodic orbits)')
print(f'    - Fredholm determinant relates to Selberg zeta for SL(2,Z)')
print(f'    - Eigenvalues decay rapidly (trace-class)')
print(f'    - The connection to Riemann zeta goes through the Selberg zeta')

print(f'\nTotal time: {time.time() - t0:.1f}s')
