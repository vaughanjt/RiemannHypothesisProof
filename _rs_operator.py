"""The Riemann-Siegel operator: deterministic phases from theta(t).

The RS + random phase model achieves r=+0.75 with random phi_n.
The actual zeta function uses phi_n(t) = theta(t) - t*log(n),
which is deterministic but varies with t.

Key test: construct H(t) with the REAL theta function phases,
compute eigenvalues at many t values, and check whether:
1. Individual H(t) has high peak-gap r (the wave function structure)
2. Averaged eigenvalue statistics are GUE (the Montgomery property)

If both hold, we have found the zeta operator.
"""
import sys, time
sys.path.insert(0, 'src')
import numpy as np
from scipy.stats import pearsonr, kstest
import mpmath
from riemann.analysis.bost_connes_operator import polynomial_unfold

t0 = time.time()
mpmath.mp.dps = 15

# ============================================================
# THE RIEMANN-SIEGEL OPERATOR
# ============================================================
print('=' * 70)
print('THE RIEMANN-SIEGEL OPERATOR H(t)')
print('=' * 70)

# H(t)_{jk} = (1/sqrt(N_sum)) * sum_{n=1}^{N_sum} (1/sqrt(n))
#              * cos(theta(t) - t*log(n)) * basis_function(j, k, n)
#
# The simplest basis: delta_{j-k, n mod N} (cyclic convolution)
# More natural: cos(2*pi*n*(j-k)/N) (Fourier basis on [1..N])
#
# The matrix H(t) at a given t has entries determined by the
# Riemann-Siegel sum. Its eigenvalues should approximate the
# local zero structure near height t.

def build_rs_matrix(t_val, N, N_sum=None):
    """Build the Riemann-Siegel matrix at height t.

    H_{jk} = sum_n a_n(t) * cos(2*pi*n*(j-k)/N)
    where a_n(t) = cos(theta(t) - t*log(n)) / sqrt(n)
    """
    if N_sum is None:
        N_sum = max(int(np.sqrt(t_val / (2 * np.pi))), 5)
        N_sum = min(N_sum, N)

    # Compute the Riemann-Siegel coefficients
    theta_t = float(mpmath.siegeltheta(t_val))
    coeffs = np.zeros(N_sum)
    for n in range(1, N_sum + 1):
        coeffs[n - 1] = np.cos(theta_t - t_val * np.log(n)) / np.sqrt(n)

    # Build the circulant-like matrix
    H = np.zeros((N, N))
    for j in range(N):
        for k in range(j, N):
            val = 0
            for n in range(N_sum):
                val += coeffs[n] * np.cos(2 * np.pi * (n + 1) * (j - k) / N)
            H[j, k] = val
            H[k, j] = val

    # Normalize
    scale = np.sqrt(np.mean(H ** 2) * N)
    if scale > 1e-10:
        H /= scale
    return H


def measure_peak_gap(eigs_raw):
    eigs = np.sort(eigs_raw)
    sp = polynomial_unfold(eigs, trim_fraction=0.1)
    if len(sp) < 20:
        return 0, 0
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
    if len(gaps) < 10:
        return 0, 0
    r, _ = pearsonr(gaps, log_peaks)
    return r, len(gaps)


# ============================================================
# STEP 1: Single-t behavior
# ============================================================
print('\nStep 1: H(t) at individual t values')

N = 150  # matrix size (keep small for speed with theta)
t_values = [100, 200, 300, 500, 1000]

print(f'\n  {"t":>8} {"N_sum":>6} {"r":>8} {"pts":>5}')
print(f'  {"-"*32}')

for t_val in t_values:
    H = build_rs_matrix(t_val, N)
    eigs = np.linalg.eigvalsh(H)
    r, pts = measure_peak_gap(eigs)
    N_sum = max(int(np.sqrt(t_val / (2 * np.pi))), 5)
    print(f'  {t_val:>8} {min(N_sum, N):>6} {r:>+8.4f} {pts:>5}')

# ============================================================
# STEP 2: Eigenvalue statistics averaged over t
# ============================================================
print('\nStep 2: Eigenvalue statistics averaged over t-window')

# Collect spacings from many H(t) at different t values
N_mat = 100
all_spacings = []
all_rs = []
t_sweep = np.linspace(200, 800, 50)

print(f'  Sweeping t from {t_sweep[0]:.0f} to {t_sweep[-1]:.0f} ({len(t_sweep)} values)...')

for t_val in t_sweep:
    H = build_rs_matrix(t_val, N_mat)
    eigs = np.linalg.eigvalsh(H)
    sp = polynomial_unfold(eigs, trim_fraction=0.1)
    if len(sp) > 10:
        sp = sp / np.mean(sp)
        all_spacings.extend(sp.tolist())
    r, _ = measure_peak_gap(eigs)
    all_rs.append(r)

all_spacings = np.array(all_spacings)
print(f'  Collected {len(all_spacings)} spacings, mean r = {np.mean(all_rs):+.4f}')

# Test GUE/Poisson
def wigner_cdf(s):
    return 1 - np.exp(-np.pi * s ** 2 / 4)

ks_gue, p_gue = kstest(all_spacings, wigner_cdf)
ks_poi, p_poi = kstest(all_spacings, 'expon', args=(0, 1))

print(f'\n  Averaged spacing statistics:')
print(f'    Mean: {np.mean(all_spacings):.4f}, Std: {np.std(all_spacings):.4f}')
print(f'    KS vs GUE (Wigner): D = {ks_gue:.4f}, p = {p_gue:.4f}')
print(f'    KS vs Poisson: D = {ks_poi:.4f}, p = {p_poi:.4f}')

if p_gue > 0.05:
    print('    -> GUE ACCEPTED!')
elif p_gue > p_poi:
    print('    -> More GUE-like than Poisson')
else:
    print('    -> More Poisson-like than GUE')

# ============================================================
# STEP 3: Vary matrix size
# ============================================================
print('\nStep 3: Size dependence')

print(f'\n  {"N":>5} {"mean r":>8} {"p(GUE)":>8} {"n_spacings":>10}')
print(f'  {"-"*35}')

for N_test in [50, 100, 150, 200]:
    spacings_n = []
    rs_n = []
    for t_val in np.linspace(200, 800, 30):
        H = build_rs_matrix(t_val, N_test)
        eigs = np.linalg.eigvalsh(H)
        sp = polynomial_unfold(eigs, trim_fraction=0.1)
        if len(sp) > 10:
            sp = sp / np.mean(sp)
            spacings_n.extend(sp.tolist())
        r, _ = measure_peak_gap(eigs)
        rs_n.append(r)

    spacings_n = np.array(spacings_n)
    _, p_g = kstest(spacings_n, wigner_cdf) if len(spacings_n) > 20 else (0, 0)
    print(f'  {N_test:>5} {np.mean(rs_n):>+8.4f} {p_g:>8.4f} {len(spacings_n):>10}')

# ============================================================
# STEP 4: Compare to random phase version
# ============================================================
print('\nStep 4: Deterministic theta(t) phases vs random phases')

rng = np.random.default_rng(123)
N_comp = 100

# Deterministic RS at t=500
H_det = build_rs_matrix(500, N_comp)
eigs_det = np.linalg.eigvalsh(H_det)
r_det, _ = measure_peak_gap(eigs_det)

# Random phase version
N_sum = max(int(np.sqrt(500 / (2 * np.pi))), 5)
phases = rng.uniform(0, 2 * np.pi, N_sum)
H_rand = np.zeros((N_comp, N_comp))
for j in range(N_comp):
    for k in range(j, N_comp):
        val = 0
        for n in range(N_sum):
            val += np.cos(2 * np.pi * (n + 1) * (j - k) / N_comp + phases[n]) / np.sqrt(n + 1)
        H_rand[j, k] = val
        H_rand[k, j] = val
scale = np.sqrt(np.mean(H_rand ** 2) * N_comp)
if scale > 1e-10:
    H_rand /= scale

eigs_rand = np.linalg.eigvalsh(H_rand)
r_rand, _ = measure_peak_gap(eigs_rand)

print(f'  Deterministic theta(t=500): r = {r_det:+.4f}')
print(f'  Random phases:              r = {r_rand:+.4f}')

# ============================================================
# VERDICT
# ============================================================
print('\n' + '=' * 70)
print('VERDICT: THE RIEMANN-SIEGEL OPERATOR')
print('=' * 70)

mean_r = np.mean(all_rs)
print(f'\n  Peak-gap r (averaged over t): {mean_r:+.4f}')
print(f'  Eigenvalue statistics (averaged): p(GUE) = {p_gue:.4f}')
print(f'\n  Zeta target: r ~ +0.80, p(GUE) > 0.05')

if mean_r > 0.5 and p_gue > 0.05:
    print(f'\n  >>> MATCH: The RS operator has BOTH high r AND GUE eigenvalues!')
    print(f'  >>> This is the zeta operator candidate.')
elif mean_r > 0.3:
    print(f'\n  >>> PARTIAL: The RS operator has the right peak-gap structure')
    print(f'  >>> but eigenvalue statistics need work (p(GUE) = {p_gue:.4f}).')
    print(f'  >>> The basis function choice or normalization may need refinement.')
else:
    print(f'\n  >>> The RS operator in this form does not match the target.')

print(f'\nTotal time: {time.time() - t0:.1f}s')
