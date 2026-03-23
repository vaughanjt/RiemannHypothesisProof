"""Arithmetically modulated random matrices.

The zeta operator lives at the intersection of chaos and arithmetic:
  - Eigenvalues: GUE (chaotic)
  - Eigenvectors: arithmetic (prime-structured)

Key idea: make the RANDOMNESS arithmetic. Instead of
  H = GUE + epsilon * Arithmetic  (additive, fails)
use
  H_{jk} = Z_{jk} * w(j,k)  (multiplicative: random THROUGH arithmetic)

where Z is random and w encodes the multiplicative structure of integers.
The variance profile E[|H_{jk}|^2] = w(j,k)^2 / N determines eigenvalue
statistics; the arithmetic structure of w determines eigenvector coupling.

If w(j,k) = f(gcd(j,k))/sqrt(j*k), we get:
  - Eigenvalues: potentially GUE (if w is smooth enough)
  - Eigenvectors: arithmetically structured (from the gcd weight)
  - Peak-gap coupling: from the arithmetic modulation of variance
"""
import sys, time
sys.path.insert(0, 'src')
import numpy as np
from scipy.stats import pearsonr, kstest
from riemann.analysis.bost_connes_operator import polynomial_unfold

t0 = time.time()
rng = np.random.default_rng(42)


def measure_operator(eigs_raw, label=""):
    """Measure peak-gap r, spacing KS vs GUE, and return metrics."""
    eigs = np.sort(eigs_raw)
    sp = polynomial_unfold(eigs, trim_fraction=0.1)
    if len(sp) < 20:
        return {'r': 0, 'ks_gue': 1, 'ks_poisson': 1, 'label': label}
    sp = sp / np.mean(sp)
    n_trim = int(0.1 * len(eigs))
    eigs_trim = eigs[n_trim:-n_trim]

    # Peak-gap
    log_peaks, gaps = [], []
    for k in range(min(len(sp), len(eigs_trim) - 1)):
        z_mid = (eigs_trim[k] + eigs_trim[k + 1]) / 2
        log_det = np.sum(np.log(np.abs(z_mid - eigs) + 1e-30))
        log_peaks.append(log_det)
        gaps.append(sp[k])
    gaps = np.array(gaps)
    log_peaks = np.array(log_peaks)
    r, _ = pearsonr(gaps, log_peaks) if len(gaps) > 10 else (0, 1)

    # Spacing statistics
    def wigner_cdf(s):
        return 1 - np.exp(-np.pi * s ** 2 / 4)
    ks_gue, p_gue = kstest(sp, wigner_cdf)
    ks_poi, p_poi = kstest(sp, 'expon', args=(0, 1))

    return {'r': r, 'p_gue': p_gue, 'p_poi': p_poi,
            'ks_gue': ks_gue, 'ks_poi': ks_poi, 'label': label,
            'mean_sp': np.mean(sp), 'std_sp': np.std(sp)}


# ============================================================
# THE ZOO OF ARITHMETICALLY MODULATED RANDOM MATRICES
# ============================================================
print('=' * 70)
print('ARITHMETICALLY MODULATED RANDOM MATRICES')
print('=' * 70)

N = 300
n_trials = 10  # average over trials for stability

def run_trials(build_func, label, N=N, n_trials=n_trials):
    """Run multiple trials and average metrics."""
    rs, p_gues, p_pois = [], [], []
    for trial in range(n_trials):
        H = build_func(N, rng)
        eigs = np.linalg.eigvalsh(H)
        m = measure_operator(eigs)
        rs.append(m['r'])
        p_gues.append(m['p_gue'])
        p_pois.append(m['p_poi'])
    return np.mean(rs), np.mean(p_gues), np.mean(p_pois)


print(f'\n{"Operator":<50} {"r":>7} {"p(GUE)":>8} {"p(Poi)":>8}')
print('-' * 78)

# --- BASELINE: Pure GUE ---
def pure_gue(N, rng):
    A = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    return np.real((A + A.conj().T) / (2 * np.sqrt(2 * N)))

r, pg, pp = run_trials(pure_gue, 'GUE')
print(f'{"Pure GUE":<50} {r:>+7.4f} {pg:>8.4f} {pp:>8.4f}')

# --- TYPE 1: Random * gcd weight ---
# H_{jk} = Z_{jk} * f(gcd(j,k)) / sqrt(j*k)
# Z_{jk} ~ N(0,1), symmetrized

def random_gcd(N, rng, f=None):
    if f is None:
        f = lambda g: np.log(g + 1)
    Z = rng.standard_normal((N, N))
    H = np.zeros((N, N))
    for j in range(N):
        for k in range(j, N):
            g = np.gcd(j + 1, k + 1)
            w = f(g) / np.sqrt((j + 1) * (k + 1))
            val = Z[j, k] * w
            H[j, k] = val
            H[k, j] = val
    # Normalize
    H /= np.sqrt(np.mean(H ** 2) * N) + 1e-10
    return H

r, pg, pp = run_trials(lambda N, rng: random_gcd(N, rng), 'Z * log(gcd+1)/sqrt(jk)')
print(f'{"Z * log(gcd+1)/sqrt(jk)":<50} {r:>+7.4f} {pg:>8.4f} {pp:>8.4f}')

r, pg, pp = run_trials(lambda N, rng: random_gcd(N, rng, f=lambda g: np.sqrt(g)),
                        'Z * sqrt(gcd)/sqrt(jk)')
print(f'{"Z * sqrt(gcd)/sqrt(jk)":<50} {r:>+7.4f} {pg:>8.4f} {pp:>8.4f}')

r, pg, pp = run_trials(lambda N, rng: random_gcd(N, rng, f=lambda g: float(g)),
                        'Z * gcd/sqrt(jk)')
print(f'{"Z * gcd/sqrt(jk)":<50} {r:>+7.4f} {pg:>8.4f} {pp:>8.4f}')

# --- TYPE 2: GUE with arithmetic variance profile ---
# H = A * W where A is GUE-like and W_{jk} = w(j,k)
# Element-wise (Hadamard) product

def gue_hadamard_gcd(N, rng, f=None):
    if f is None:
        f = lambda g: np.log(g + 1)
    A = rng.standard_normal((N, N))
    A = (A + A.T) / 2
    W = np.zeros((N, N))
    for j in range(N):
        for k in range(j, N):
            g = np.gcd(j + 1, k + 1)
            w = f(g) / np.sqrt((j + 1) * (k + 1))
            W[j, k] = w
            W[k, j] = w
    H = A * W  # Hadamard product
    H /= np.sqrt(np.mean(H ** 2) * N) + 1e-10
    return H

r, pg, pp = run_trials(lambda N, rng: gue_hadamard_gcd(N, rng), 'GUE (*) log(gcd+1)/sqrt(jk)')
print(f'{"GUE (*) log(gcd+1)/sqrt(jk)":<50} {r:>+7.4f} {pg:>8.4f} {pp:>8.4f}')

# --- TYPE 3: Variance-modulated Wigner ---
# H_{jk} ~ N(0, sigma_{jk}^2) where sigma_{jk} = w(j,k)/sqrt(N)

def variance_modulated(N, rng, f=None):
    if f is None:
        f = lambda g: np.log(g + 1)
    H = np.zeros((N, N))
    for j in range(N):
        for k in range(j, N):
            g = np.gcd(j + 1, k + 1)
            sigma = f(g) / np.sqrt((j + 1) * (k + 1))
            val = rng.standard_normal() * sigma
            H[j, k] = val
            H[k, j] = val
    H /= np.sqrt(np.mean(H ** 2) * N) + 1e-10
    return H

r, pg, pp = run_trials(lambda N, rng: variance_modulated(N, rng), 'N(0, log(gcd)^2/jk)')
print(f'{"N(0, log(gcd)^2/jk)":<50} {r:>+7.4f} {pg:>8.4f} {pp:>8.4f}')

r, pg, pp = run_trials(lambda N, rng: variance_modulated(N, rng, f=lambda g: float(g)),
                        'N(0, gcd^2/jk)')
print(f'{"N(0, gcd^2/jk)":<50} {r:>+7.4f} {pg:>8.4f} {pp:>8.4f}')

# --- TYPE 4: Deterministic arithmetic + random phase ---
# H_{jk} = w(j,k) * exp(i * theta_{jk}) where theta is random

def arithmetic_random_phase(N, rng, f=None):
    if f is None:
        f = lambda g: np.log(g + 1)
    theta = rng.uniform(0, 2 * np.pi, (N, N))
    theta = (theta + theta.T) / 2  # symmetrize phases
    H = np.zeros((N, N))
    for j in range(N):
        for k in range(j, N):
            g = np.gcd(j + 1, k + 1)
            w = f(g) / np.sqrt((j + 1) * (k + 1))
            H[j, k] = w * np.cos(theta[j, k])
            H[k, j] = H[j, k]
    H /= np.sqrt(np.mean(H ** 2) * N) + 1e-10
    return H

r, pg, pp = run_trials(lambda N, rng: arithmetic_random_phase(N, rng),
                        'log(gcd)/sqrt(jk) * cos(random phase)')
print(f'{"log(gcd)/sqrt(jk) * cos(rand phase)":<50} {r:>+7.4f} {pg:>8.4f} {pp:>8.4f}')

# --- TYPE 5: The Riemann-Siegel inspired operator ---
# H_{jk} = sum_n (1/n) * cos(log(n) * (j-k) * 2*pi/N)
# This mimics the Riemann-Siegel sum structure

def riemann_siegel_operator(N, rng):
    H = np.zeros((N, N))
    N_sum = int(np.sqrt(N))  # analogous to sqrt(T/2pi)
    for j in range(N):
        for k in range(j, N):
            val = 0
            for n in range(1, N_sum + 1):
                val += np.cos(np.log(n) * (j - k) * 2 * np.pi / N) / np.sqrt(n)
            H[j, k] = val
            H[k, j] = val
    H /= np.sqrt(np.mean(H ** 2) * N) + 1e-10
    return H

r, pg, pp = run_trials(lambda N, rng: riemann_siegel_operator(N, rng),
                        'RS: sum cos(logn*(j-k))/sqrt(n)', n_trials=3)
print(f'{"RS: sum cos(logn*(j-k))/sqrt(n)":<50} {r:>+7.4f} {pg:>8.4f} {pp:>8.4f}')

# --- TYPE 6: RS + random phase (per n) ---
def rs_random_phase(N, rng):
    H = np.zeros((N, N))
    N_sum = int(np.sqrt(N))
    phases = rng.uniform(0, 2 * np.pi, N_sum + 1)  # random phase per n
    for j in range(N):
        for k in range(j, N):
            val = 0
            for n in range(1, N_sum + 1):
                val += np.cos(np.log(n) * (j - k) * 2 * np.pi / N + phases[n]) / np.sqrt(n)
            H[j, k] = val
            H[k, j] = val
    H /= np.sqrt(np.mean(H ** 2) * N) + 1e-10
    return H

r, pg, pp = run_trials(lambda N, rng: rs_random_phase(N, rng),
                        'RS + random phase per n', n_trials=3)
print(f'{"RS + random phase per n":<50} {r:>+7.4f} {pg:>8.4f} {pp:>8.4f}')

# --- TYPE 7: Divisibility indicator with random signs ---
def divisibility_random(N, rng):
    signs = rng.choice([-1, 1], size=(N, N))
    signs = signs * signs.T  # make symmetric: sign_{jk} = sign_j * sign_k
    # Actually, use independent symmetric signs
    signs = rng.choice([-1, 1], size=(N, N))
    signs = (signs + signs.T) / 2
    signs = np.sign(signs)
    signs[signs == 0] = 1

    H = np.zeros((N, N))
    for j in range(1, N + 1):
        for k in range(j, N + 1):
            if k % j == 0 or j % k == 0:
                H[j-1, k-1] = signs[j-1, k-1] * np.log(max(j, k)) / np.sqrt(j * k)
                H[k-1, j-1] = H[j-1, k-1]
    H /= np.sqrt(np.mean(H ** 2) * N) + 1e-10
    return H

r, pg, pp = run_trials(lambda N, rng: divisibility_random(N, rng),
                        'Divisibility * random signs * log/sqrt')
print(f'{"Divisibility * random signs * log/sqrt":<50} {r:>+7.4f} {pg:>8.4f} {pp:>8.4f}')

# --- TYPE 8: The "zeta kernel" ---
# H_{jk} = Re[sum_p log(p)/sqrt(p) * exp(i*2*pi*(j-k)*log(p)/log(N))] / N
# Direct encoding of prime oscillations into the matrix

def zeta_kernel(N, rng):
    from sympy import primerange
    primes = list(primerange(2, N))
    log_N = np.log(N)
    H = np.zeros((N, N))
    for j in range(N):
        for k in range(j, N):
            val = 0
            for p in primes:
                val += np.log(p) / np.sqrt(p) * np.cos(2 * np.pi * (j - k) * np.log(p) / log_N)
            H[j, k] = val
            H[k, j] = val
    H /= np.sqrt(np.mean(H ** 2) * N) + 1e-10
    return H

r, pg, pp = run_trials(lambda N, rng: zeta_kernel(N, rng),
                        'Zeta kernel: prime cosines', n_trials=1)
print(f'{"Zeta kernel: prime cosines":<50} {r:>+7.4f} {pg:>8.4f} {pp:>8.4f}')

# --- TYPE 8b: Zeta kernel + random phase per prime ---
def zeta_kernel_random(N, rng):
    from sympy import primerange
    primes = list(primerange(2, N))
    log_N = np.log(N)
    phases = rng.uniform(0, 2 * np.pi, len(primes))
    H = np.zeros((N, N))
    for j in range(N):
        for k in range(j, N):
            val = 0
            for i_p, p in enumerate(primes):
                val += np.log(p) / np.sqrt(p) * np.cos(
                    2 * np.pi * (j - k) * np.log(p) / log_N + phases[i_p])
            H[j, k] = val
            H[k, j] = val
    H /= np.sqrt(np.mean(H ** 2) * N) + 1e-10
    return H

r, pg, pp = run_trials(lambda N, rng: zeta_kernel_random(N, rng),
                        'Zeta kernel + random phase/prime', n_trials=3)
print(f'{"Zeta kernel + random phase/prime":<50} {r:>+7.4f} {pg:>8.4f} {pp:>8.4f}')

# ============================================================
# VERDICT
# ============================================================
print('\n' + '=' * 70)
print('VERDICT: ARITHMETICALLY MODULATED CHAOS')
print('=' * 70)
print(f'\n  Target: r ~ +0.80, p(GUE) > 0.05')
print(f'  Best combination: high r AND GUE-like eigenvalues')
print(f'\nTotal time: {time.time() - t0:.1f}s')
