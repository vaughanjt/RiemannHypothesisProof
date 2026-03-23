"""Bost-Connes system as a transfer operator whose Fredholm determinant IS zeta(s).

The Euler product: zeta(s) = prod_p (1 - p^{-s})^{-1}

Rewrite as: 1/zeta(s) = prod_p (1 - p^{-s}) = det(I - L_s)

where L_s is a transfer operator whose "periodic orbits" are the primes.
Specifically, if L_s has eigenvalues {p^{-s} : p prime}, then:
  det(I - L_s) = prod_p (1 - p^{-s}) = 1/zeta(s)

So zeta(s) = 1/det(I - L_s), and the ZEROS of zeta are where
det(I - L_s) has POLES — i.e., where L_s has eigenvalue 1.

The operator: L_s acts on a space indexed by primes, with
  L_s |p> = p^{-s} |p>

This is diagonal — too simple. The interesting structure comes from
the MULTIPLICATIVE relations between primes (the Hecke algebra).

The Bost-Connes C*-algebra approach:
  - Hilbert space: l^2(N) with basis |n> for n = 1, 2, 3, ...
  - Shift operators: mu_n |m> = |nm> (multiplication by n)
  - Number operator: H |n> = log(n) |n>
  - The partition function Tr(e^{-beta*H}) = sum_n n^{-beta} = zeta(beta)

So the HEAT KERNEL of H = diag(log(n)) gives zeta!
And the zeros of zeta correspond to resonances of H.

But H = diag(log(n)) has eigenvalues {log(n)}, which are NOT the zeta zeros.
The zeta zeros appear in the SPECTRAL ZETA FUNCTION of H:
  zeta_H(s) = Tr(H^{-s}) = sum_n (log n)^{-s}  — no, that's not right either.

The correct connection: the zeros of zeta(s) are the values of s where
  det(I - L_s) = 0, i.e., where L_s has eigenvalue 1.

L_s is NOT diagonal — it must encode the MULTIPLICATIVE structure.
The right construction: L_s acts on l^2(N*) by
  (L_s f)(n) = sum_{d|n, d>1} d^{-s} f(n/d)   [convolution with n^{-s}]

or equivalently via the HECKE operators:
  T_p f(n) = f(pn) + p^{-1} f(n/p)  [if p|n, else f(pn)]

Let's build several candidates and test which one has:
1. Fredholm determinant = 1/zeta(s)  (zeros at zeta zeros)
2. Eigenvector rigidity (peak-gap r >> 0.04)
3. Prime-frequency trace oscillations
"""
import sys, time
sys.path.insert(0, 'src')
import numpy as np
from scipy.stats import pearsonr
import mpmath

t0 = time.time()
mpmath.mp.dps = 15

# ============================================================
# CANDIDATE 1: The Dirichlet convolution operator
# ============================================================
print('=' * 70)
print('CANDIDATE 1: DIRICHLET CONVOLUTION OPERATOR')
print('=' * 70)

# (L_s f)(n) = sum_{d|n, d>1} d^{-s} f(n/d)
# Matrix form: L_{mn} = m^{-s} if m|n and m > 1, else 0
# Then: det(I - L_s) should relate to 1/zeta(s)

def build_dirichlet_operator(s, N):
    """Build the Dirichlet convolution operator L_s on {1,...,N}."""
    L = np.zeros((N, N), dtype=complex)
    for n in range(1, N + 1):
        for d in range(2, n + 1):
            if n % d == 0:
                m = n // d  # f(n/d) = f(m), so column m, row n
                L[n - 1, m - 1] += d ** (-s)
    return L

def check_fredholm(s, N, known_zeta=None):
    """Check if det(I - L_s) ≈ 1/zeta(s)."""
    L = build_dirichlet_operator(s, N)
    det_val = np.linalg.det(np.eye(N, dtype=complex) - L)
    if known_zeta is not None:
        ratio = det_val * known_zeta
        return det_val, ratio
    return det_val, None

print('\nChecking det(I - L_s) vs 1/zeta(s):')
print(f'  {"s":>12} {"det(I-L)":>20} {"1/zeta(s)":>20} {"ratio":>12}')
print(f'  {"-"*68}')

for s_val in [2.0, 3.0, 4.0, 1.5, 0.5 + 14.13j, 0.5 + 21.02j, 0.5 + 25.01j]:
    zeta_val = complex(mpmath.zeta(s_val))
    inv_zeta = 1.0 / zeta_val if abs(zeta_val) > 1e-10 else float('inf')
    det_val, ratio = check_fredholm(s_val, 100, zeta_val)
    print(f'  {str(s_val):>12} {det_val.real:>+12.6f}{det_val.imag:>+8.4f}i '
          f'{inv_zeta.real:>+12.6f}{inv_zeta.imag:>+8.4f}i {abs(ratio):>12.6f}')

# ============================================================
# CANDIDATE 2: The von Mangoldt / prime operator
# ============================================================
print('\n' + '=' * 70)
print('CANDIDATE 2: PRIME-ONLY OPERATOR')
print('=' * 70)

# Instead of all divisors, use only PRIME divisors:
# (L_s f)(n) = sum_{p|n, p prime} p^{-s} f(n/p)
# Then det(I - L_s) = prod_p (1 - p^{-s}) * (correction) ≈ 1/zeta(s)

from sympy import isprime

def build_prime_operator(s, N):
    """L_s with only prime divisor contributions."""
    L = np.zeros((N, N), dtype=complex)
    for n in range(1, N + 1):
        for p in range(2, n + 1):
            if isprime(p) and n % p == 0:
                m = n // p
                L[n - 1, m - 1] += p ** (-s)
    return L

print('\nChecking prime-only det(I - L_s) vs 1/zeta(s):')
print(f'  {"s":>12} {"det(I-L)":>20} {"1/zeta(s)":>20} {"ratio":>12}')
print(f'  {"-"*68}')

for s_val in [2.0, 3.0, 4.0, 1.5]:
    zeta_val = complex(mpmath.zeta(s_val))
    inv_zeta = 1.0 / zeta_val
    L = build_prime_operator(s_val, 100)
    det_val = np.linalg.det(np.eye(100, dtype=complex) - L)
    ratio = det_val * zeta_val
    print(f'  {str(s_val):>12} {det_val.real:>+12.6f}{det_val.imag:>+8.4f}i '
          f'{inv_zeta.real:>+12.6f}{inv_zeta.imag:>+8.4f}i {abs(ratio):>12.6f}')

# ============================================================
# CANDIDATE 3: Direct Euler product operator
# ============================================================
print('\n' + '=' * 70)
print('CANDIDATE 3: DIRECT EULER PRODUCT (diagonal in prime basis)')
print('=' * 70)

# The simplest operator: diagonal with entries p^{-s} for each prime
# det(I - L_s) = prod_{p<=N} (1 - p^{-s})
# This converges to 1/zeta(s) as N -> infinity.

from sympy import primerange

def euler_product_det(s, P_max):
    """Compute prod_{p<=P_max} (1 - p^{-s})."""
    prod = 1.0 + 0j
    for p in primerange(2, P_max + 1):
        prod *= (1 - p ** (-s))
    return prod

print('\nEuler product convergence to 1/zeta(s):')
print(f'  {"s":>8} {"P_max":>6} {"prod":>20} {"1/zeta":>20} {"|ratio|":>10}')
print(f'  {"-"*68}')

for s_val in [2.0, 3.0, 0.5 + 14.134j]:
    zeta_val = complex(mpmath.zeta(s_val))
    inv_zeta = 1.0 / zeta_val if abs(zeta_val) > 1e-10 else 0
    for P_max in [10, 50, 100, 500]:
        ep = euler_product_det(s_val, P_max)
        ratio = abs(ep / inv_zeta) if abs(inv_zeta) > 1e-10 else 0
        if P_max == 500 or P_max == 10:
            print(f'  {str(s_val):>8} {P_max:>6} {ep.real:>+12.6f}{ep.imag:>+8.4f}i '
                  f'{inv_zeta.real:>+12.6f}{inv_zeta.imag:>+8.4f}i {ratio:>10.6f}')

# ============================================================
# CANDIDATE 4: Scan the critical line for zeros
# ============================================================
print('\n' + '=' * 70)
print('CANDIDATE 4: SCANNING CRITICAL LINE WITH DIRICHLET OPERATOR')
print('=' * 70)

# Use the Dirichlet convolution operator and scan for where
# det(I - L_s) is near zero (corresponding to zeta zeros)

N_op = 200  # matrix size
t_scan = np.linspace(10, 55, 500)
det_scan = np.zeros(len(t_scan), dtype=complex)

print(f'  Scanning det(I - L_s) on critical line, N={N_op}...')
t_build = time.time()
for i, t_val in enumerate(t_scan):
    s = 0.5 + 1j * t_val
    L = build_dirichlet_operator(s, N_op)
    det_scan[i] = np.linalg.det(np.eye(N_op, dtype=complex) - L)

print(f'  Done: {time.time() - t_build:.1f}s')

abs_det = np.abs(det_scan)

# Find minima
minima = []
for i in range(1, len(abs_det) - 1):
    if abs_det[i] < abs_det[i-1] and abs_det[i] < abs_det[i+1]:
        minima.append((t_scan[i], abs_det[i]))

# Sort by depth
minima.sort(key=lambda x: x[1])

# Known zeta zeros
known = [14.1347, 21.0220, 25.0109, 30.4249, 32.9351,
         37.5862, 40.9187, 43.3271, 48.0052, 49.7738, 52.9703]

print(f'\n  Deepest minima of |det(I - L_s)|:')
print(f'  {"t":>10} {"|det|":>12} {"Nearest zero":>14} {"Distance":>10}')
for t_min, d_min in minima[:15]:
    dists = [abs(t_min - z) for z in known]
    best = np.argmin(dists)
    tag = ' <-- MATCH' if dists[best] < 0.5 else ''
    print(f'  {t_min:>10.4f} {d_min:>12.4f} {known[best]:>14.4f} {dists[best]:>10.4f}{tag}')

n_match = sum(1 for t_min, _ in minima[:20] if min(abs(t_min - z) for z in known) < 0.5)
print(f'\n  Matches (within 0.5): {n_match} out of top {min(20, len(minima))} minima')
print(f'  Known zeros in range: {sum(1 for z in known if 10 < z < 55)}')

# ============================================================
# VERDICT
# ============================================================
print('\n' + '=' * 70)
print('VERDICT')
print('=' * 70)

if n_match >= 3:
    print(f'\n  >>> THE DIRICHLET OPERATOR FINDS ZETA ZEROS!')
    print(f'  >>> det(I - L_s) has minima at the known zero locations.')
    print(f'  >>> This IS the zeta operator (in truncated form).')
    print(f'  >>> The zeros of zeta(s) are eigenvalue-1 resonances of L_s.')
else:
    print(f'\n  >>> Only {n_match} matches found.')
    print(f'  >>> The truncation N={N_op} may be too small,')
    print(f'  >>> or the operator needs modification.')

print(f'\nTotal time: {time.time() - t0:.1f}s')
