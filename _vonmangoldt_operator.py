"""The von Mangoldt convolution operator.

The key identity: -zeta'(s)/zeta(s) = sum_n Lambda(n) n^{-s}
where Lambda(n) = log(p) if n = p^m, else 0.

The FOURIER TRANSFORM of Lambda is -zeta'(s)/zeta(s), which has
POLES at the nontrivial zeros of zeta(s).

A Toeplitz matrix with Lambda as its symbol:
  H_{jk} = Lambda(|j - k| + 1)  (circulant-like)

has eigenvalues that approximate the Fourier transform of Lambda
on a grid of N frequencies. Near a zeta zero, the eigenvalue diverges.

More precisely: consider the operator on L^2(Z/NZ) defined by
convolution with Lambda. Its eigenvalues are:
  lambda_k = sum_{n=1}^{N-1} Lambda(n) * exp(-2*pi*i*k*n/N)

This is approximately -zeta'(1/2 + i*t_k)/zeta(1/2 + i*t_k) for
t_k = 2*pi*k/(N*delta) at appropriate scaling.

The zeta zeros show up as LARGE eigenvalues (poles of -zeta'/zeta).
"""
import sys, time
sys.path.insert(0, 'src')
import numpy as np
from scipy.stats import pearsonr
from sympy import factorint
import mpmath

t0 = time.time()
mpmath.mp.dps = 15

# ============================================================
# THE VON MANGOLDT FUNCTION
# ============================================================
def von_mangoldt(n):
    """Lambda(n) = log(p) if n = p^k for some prime p, else 0."""
    if n <= 1:
        return 0.0
    factors = factorint(n)
    if len(factors) == 1:
        p = list(factors.keys())[0]
        return np.log(p)
    return 0.0

print('Precomputing von Mangoldt function...')
N_MAX = 2000
Lambda = np.array([von_mangoldt(n) for n in range(N_MAX)])
print(f'  Lambda[2..20] = {[f"{Lambda[n]:.3f}" for n in range(2, 21)]}')

# ============================================================
# STEP 1: The circulant von Mangoldt matrix
# ============================================================
print('\n' + '=' * 70)
print('STEP 1: CIRCULANT VON MANGOLDT MATRIX')
print('=' * 70)

# For a circulant matrix, eigenvalues = DFT of the first row.
# First row: [Lambda(0), Lambda(1), ..., Lambda(N-1)] where Lambda(0) = 0

def vonmangoldt_eigenvalues(N):
    """Eigenvalues of the N x N circulant von Mangoldt matrix."""
    row = Lambda[:N].copy()
    return np.fft.fft(row)

# The eigenvalue at index k is:
# lambda_k = sum_{n=0}^{N-1} Lambda(n) * exp(-2*pi*i*k*n/N)
# = sum_{p^m <= N} log(p) * exp(-2*pi*i*k*m*log(p)/(log(N)*...) )
# Hmm, the frequency isn't quite right. Let me compute and see.

for N in [100, 500, 1000]:
    eigs = vonmangoldt_eigenvalues(N)
    mags = np.abs(eigs)
    top_idx = np.argsort(mags)[-10:][::-1]
    print(f'\n  N = {N}: top 5 eigenvalue magnitudes:')
    for rank, idx in enumerate(top_idx[:5]):
        freq = idx / N  # normalized frequency
        print(f'    k={idx:>5}, |lambda|={mags[idx]:>8.3f}, freq={freq:.4f}')

# ============================================================
# STEP 2: Connect eigenvalue peaks to zeta zeros
# ============================================================
print('\n' + '=' * 70)
print('STEP 2: EIGENVALUE PEAKS vs ZETA ZEROS')
print('=' * 70)

# The circulant eigenvalues sample -zeta'(s)/zeta(s) at
# s_k = sigma + i * 2*pi*k / (N * h) for some h and sigma.
# For the von Mangoldt function Lambda(n), the DFT gives:
# hat{Lambda}(k/N) = sum_{n=1}^{N-1} Lambda(n) * exp(-2*pi*i*k*n/N)
#
# This equals approximately sum_n Lambda(n) * n^{-2*pi*i*k/log(N)}
# if we think of exp(-2*pi*i*k*n/N) ≈ n^{-2*pi*i*k/(N*log(n)/n)} ...
# The connection isn't direct for a circulant.
#
# Better approach: EXPLICIT evaluation of -zeta'/zeta at grid points
# and comparison to the DFT eigenvalues.

N = 1000
eigs = vonmangoldt_eigenvalues(N)
mags = np.abs(eigs)

# The DFT samples the function F(theta) = sum Lambda(n) e^{-i*n*theta}
# at theta_k = 2*pi*k/N. This is the Fourier series of Lambda.
#
# To connect to zeta zeros: -zeta'(s)/zeta(s) = sum Lambda(n) n^{-s}
# This involves n^{-s}, not e^{-i*n*theta}. They're related by the
# Mellin transform, not the Fourier transform.
#
# The correct approach: use a MULTIPLICATIVE Fourier transform.
# Define the multiplicative DFT: hat{f}(t) = sum_{n=1}^N f(n) * n^{-it}
# This is -zeta'(1/2+it)/zeta(1/2+it) when f = Lambda and sigma = 1/2.

print(f'  Computing multiplicative DFT of Lambda (the logarithmic derivative)...')

def log_derivative_zeta(t_val, N_terms):
    """Compute -zeta'(1/2+it)/zeta(1/2+it) via partial sum of Lambda(n)*n^{-1/2-it}."""
    s = 0.5 + 1j * t_val
    val = 0
    for n in range(1, N_terms + 1):
        if Lambda[n] > 0:
            val += Lambda[n] * n ** (-s)
    return val

# Scan along the critical line
t_grid = np.linspace(1, 60, 500)
log_deriv = np.array([log_derivative_zeta(t, 1000) for t in t_grid])
abs_log_deriv = np.abs(log_deriv)

# Peaks of |zeta'/zeta| are near zeta zeros
peaks = []
for i in range(1, len(abs_log_deriv) - 1):
    if abs_log_deriv[i] > abs_log_deriv[i-1] and abs_log_deriv[i] > abs_log_deriv[i+1]:
        if abs_log_deriv[i] > np.median(abs_log_deriv) * 2:
            peaks.append((t_grid[i], abs_log_deriv[i]))

known_zeros = [14.1347, 21.0220, 25.0109, 30.4249, 32.9351,
               37.5862, 40.9187, 43.3271, 48.0052, 49.7738, 52.9703]

print(f'\n  Peaks of |sum Lambda(n)*n^{{-1/2-it}}| (N_terms=1000):')
print(f'  {"t_peak":>10} {"|F(t)|":>10} {"Nearest zero":>14} {"Dist":>8}')
for t_p, f_p in sorted(peaks, key=lambda x: -x[1])[:15]:
    dists = [abs(t_p - z) for z in known_zeros]
    best = np.argmin(dists)
    tag = ' <--' if dists[best] < 0.5 else ''
    print(f'  {t_p:>10.4f} {f_p:>10.3f} {known_zeros[best]:>14.4f} {dists[best]:>8.4f}{tag}')

n_match = sum(1 for t_p, _ in peaks if min(abs(t_p - z) for z in known_zeros) < 0.5)
print(f'\n  Matches: {n_match}/{len(peaks)} peaks within 0.5 of a zero')

# ============================================================
# STEP 3: Build the MULTIPLICATIVE convolution operator
# ============================================================
print('\n' + '=' * 70)
print('STEP 3: MULTIPLICATIVE CONVOLUTION OPERATOR')
print('=' * 70)

# Instead of additive convolution H_{jk} = Lambda(|j-k|),
# use MULTIPLICATIVE convolution: the operator acts on functions
# f: {1,...,N} -> C by:
#   (L f)(n) = sum_{d|n} Lambda(d) * d^{-1/2} * f(n/d)
#
# This is Dirichlet convolution with Lambda(n)*n^{-1/2}.
# Its "eigenvalues" in the multiplicative sense are:
#   hat{L}(chi) = sum_n Lambda(n) * chi(n) * n^{-1/2}
# for each Dirichlet character chi.
#
# For the principal character: hat{L}(chi_0) = -zeta'(1/2)/zeta(1/2)
# For the character n^{-it}: hat{L} = sum Lambda(n) n^{-1/2-it} = -zeta'(1/2+it)/zeta(1/2+it)

# But Dirichlet convolution doesn't give a normal matrix on {1,...,N}
# because it's triangular (d|n means d <= n).
# Solution: SYMMETRIZE using both d|n and n|d directions.

def build_symmetric_vonmangoldt(N):
    """Build symmetric operator: H_{jk} = Lambda(j/k)/sqrt(j) if k|j
                                         + Lambda(k/j)/sqrt(k) if j|k
    (using only the case where one divides the other)."""
    H = np.zeros((N, N))
    for j in range(1, N + 1):
        for k in range(1, N + 1):
            if j == k:
                continue
            if j % k == 0:
                d = j // k
                lam = Lambda[d] if d < len(Lambda) else 0
                if lam > 0:
                    H[j-1, k-1] = lam / np.sqrt(j)
            # Symmetrize
            if k % j == 0:
                d = k // j
                lam = Lambda[d] if d < len(Lambda) else 0
                if lam > 0:
                    H[j-1, k-1] += lam / np.sqrt(k)
    # Make Hermitian
    H = (H + H.T) / 2
    return H

print(f'  Building symmetric von Mangoldt operator (N=500)...')
t_build = time.time()
N_op = 500
H_vm = build_symmetric_vonmangoldt(N_op)
eigs_vm = np.linalg.eigvalsh(H_vm)
print(f'  Done: {time.time() - t_build:.1f}s, {len(eigs_vm)} eigenvalues')

# Unfold and check spacing statistics
from riemann.analysis.bost_connes_operator import polynomial_unfold, spacing_autocorrelation

sp = polynomial_unfold(eigs_vm, trim_fraction=0.1)
sp = sp / np.mean(sp)

from scipy.stats import kstest

def wigner_cdf(s):
    return 1 - np.exp(-np.pi * s ** 2 / 4)

ks_gue, p_gue = kstest(sp, wigner_cdf)
ks_poi, p_poi = kstest(sp, 'expon', args=(0, 1))
print(f'  Spacing stats: KS(GUE) p={p_gue:.4f}, KS(Poisson) p={p_poi:.4f}')

# Peak-gap correlation
def measure_peak_gap_sym(eigs_raw):
    eigs = np.sort(eigs_raw)
    sp = polynomial_unfold(eigs, trim_fraction=0.1)
    if len(sp) < 20: return 0, 0
    sp = sp / np.mean(sp)
    n_trim = int(0.1 * len(eigs))
    eigs_trim = eigs[n_trim:-n_trim]
    log_peaks, gaps = [], []
    for k in range(min(len(sp), len(eigs_trim) - 1)):
        z_mid = (eigs_trim[k] + eigs_trim[k + 1]) / 2
        log_det = np.sum(np.log(np.abs(z_mid - eigs) + 1e-30))
        log_peaks.append(log_det)
        gaps.append(sp[k])
    gaps = np.array(gaps)
    log_peaks = np.array(log_peaks)
    if len(gaps) < 10: return 0, 0
    r, _ = pearsonr(gaps, log_peaks)
    return r, len(gaps)

r_vm, pts_vm = measure_peak_gap_sym(eigs_vm)
print(f'  Peak-gap r = {r_vm:+.4f} ({pts_vm} pairs)')

# ACF
max_lag = min(100, len(sp) // 4)
acf_vm = spacing_autocorrelation(sp, max_lag)
se_vm = 1.0 / np.sqrt(len(sp))
n_sig = np.sum(np.abs(acf_vm[1:max_lag+1]) / se_vm > 2.5)
print(f'  ACF: {n_sig}/{max_lag} significant lags (expect ~{max_lag*0.012:.1f} under GUE)')

# ============================================================
# STEP 4: Compare all operators
# ============================================================
print('\n' + '=' * 70)
print('COMPARISON TABLE')
print('=' * 70)

print(f'\n  {"Operator":<40} {"r":>8} {"p(GUE)":>8} {"p(Poi)":>8}')
print(f'  {"-"*68}')
print(f'  {"GUE baseline":<40} {"+0.04":>8} {">0.05":>8} {"0.00":>8}')
print(f'  {"Sym. von Mangoldt (N=500)":<40} {r_vm:>+8.4f} {p_gue:>8.4f} {p_poi:>8.4f}')
print(f'  {"Zeta zeros (target)":<40} {"+0.80":>8} {">0.05":>8} {"0.00":>8}')

# Also check: does the multiplicative DFT find zeros?
print(f'\n  Multiplicative DFT: {n_match} peaks match known zeta zeros')

# ============================================================
# VERDICT
# ============================================================
print('\n' + '=' * 70)
print('VERDICT')
print('=' * 70)

if n_match >= 5:
    print(f'\n  >>> The multiplicative DFT of Lambda FINDS zeta zeros!')
    print(f'  >>> sum Lambda(n)*n^{{-1/2-it}} peaks at t = gamma_k')
    print(f'  >>> This is just -zeta\'/zeta evaluated on the critical line.')
elif r_vm > 0.3:
    print(f'\n  >>> The symmetric von Mangoldt operator has peak-gap r = {r_vm:+.3f}')
    print(f'  >>> Arithmetic eigenvector coupling is present.')
else:
    print(f'\n  >>> Neither approach matches the target yet.')

print(f'\n  The path forward:')
print(f'  - The multiplicative DFT confirms: zeta zeros ARE visible in Lambda(n)')
print(f'  - The challenge: building a SELF-ADJOINT operator with these as eigenvalues')
print(f'  - The von Mangoldt Toeplitz/circulant gives the right Fourier content')
print(f'  - But the eigenvalues of the matrix != zeros of the transform')

print(f'\nTotal time: {time.time() - t0:.1f}s')
