"""
SESSION 48b -- SELBERG TRACE FORMULA WITH BARRIER'S OWN TEST FUNCTION

The heat kernel failed because B(L) != K(t) on SL(2,Z)\H.
But the Selberg trace formula works for ANY admissible test function h(r).

Strategy:
  1. Define h(r) = |H_w(1/2 + ir)|^2  (the barrier's spectral weight)
  2. This gives spectral side = Sum h(gamma_k) = Sum |H_w(rho_k)|^2 = B(L) + C(L)
  3. Compute the Weil explicit formula (GL(1) trace formula) arithmetic side
  4. The arithmetic side = prime sum + analytic terms
  5. Check: is the arithmetic side positive? Can we PROVE it's positive?

The Weil explicit formula for even test function h:
  Sum_rho h(gamma) = h(i/2) + (1/2pi) integral h(r) [psi(1/4+ir/2)/Gamma...] dr
                     - 2 Sum_p Sum_m log(p)/(p^m) * g(m*log(p))

where g is the Fourier transform of h: g(x) = (1/2pi) integral h(r) e^{-irx} dr

Key question: for h(r) = |H_w(1/2+ir)|^2, is the arithmetic side > 0?

The spectral weight |H_w|^2 is a SQUARED magnitude -- this is special because:
  - h(r) >= 0 for all r (non-negative test function)
  - h is even (|H_w(1/2+ir)|^2 = |H_w(1/2-ir)|^2 by conjugate symmetry)
  - h decays as r -> infinity (the Mellin transform decays)

A non-negative even test function is VERY special in the trace formula world.
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, log, pi, exp, sin, cos, quad,
                    zetazero, power, sqrt, fabs, im, re, conj, nstr,
                    loggamma, gamma as mpgamma, digamma, zeta)
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

mp.dps = 25


# =================================================================
# TEST FUNCTION h(r) = |H_w(1/2 + ir)|^2
# =================================================================

def build_w_hat_mp(lam_sq, N=None):
    """Build normalized Lorentzian test vector."""
    L = log(mpf(lam_sq))
    L_f = float(L)
    if N is None:
        N = max(15, round(6 * L_f))

    coeffs = []
    norm_sq = mpf(0)
    for n in range(-N, N + 1):
        val = mpf(n) / (L**2 + 16 * pi**2 * mpf(n)**2)
        coeffs.append(val)
        norm_sq += val**2

    norm = sqrt(norm_sq)
    w_hat = [c / norm for c in coeffs]
    return w_hat, L, N


def compute_Hw_at_r(r_val, w_hat_positive, L):
    """
    Compute H_w(1/2 + ir) = integral_0^L g_w(x) * x^{-1/2+ir} dx

    where g_w(x) = 2 * sum_n w_hat[n] * sin(2*pi*n*x/L)
    """
    s = mpf(1)/2 + mpc(0, mpf(r_val))
    L_s = power(L, s)

    G = mpc(0, 0)
    N = len(w_hat_positive)

    for n_idx in range(N):
        n = n_idx + 1
        wn = w_hat_positive[n_idx]
        if fabs(wn) < mpf(10)**(-20):
            continue

        freq = 2 * pi * n

        def integrand_real(u, _freq=freq):
            return sin(_freq * u) * power(u, re(s) - 1) * cos(im(s) * log(u))

        def integrand_imag(u, _freq=freq):
            return sin(_freq * u) * power(u, re(s) - 1) * sin(im(s) * log(u))

        I_real = quad(integrand_real, [mpf(0), mpf(1)], maxdegree=6)
        I_imag = quad(integrand_imag, [mpf(0), mpf(1)], maxdegree=6)

        I_n = L_s * mpc(I_real, I_imag)
        G += 2 * wn * I_n

    return G


def h_test_function(r_val, w_hat_positive, L):
    """h(r) = |H_w(1/2 + ir)|^2 -- the barrier's spectral weight."""
    Hw = compute_Hw_at_r(r_val, w_hat_positive, L)
    return float(fabs(Hw)**2)


# =================================================================
# FOURIER TRANSFORM g(x) of h(r)
# =================================================================

def g_fourier_transform(x_val, w_hat_positive, L, n_sample=200, r_max=100.0):
    """
    Compute g(x) = (1/2pi) integral_{-inf}^{inf} h(r) e^{-irx} dr

    Since h(r) = h(-r) (even), this simplifies to:
    g(x) = (1/pi) integral_0^{inf} h(r) cos(rx) dr

    Numerically approximate with truncation at r_max.
    """
    x = mpf(x_val)

    def integrand(r):
        if r < mpf('0.01'):
            return mpf(0)  # avoid singularity at r=0
        hr = h_test_function(float(r), w_hat_positive, L)
        return mpf(hr) * cos(r * x)

    # This is expensive -- use coarse quadrature
    result = quad(integrand, [mpf(0), mpf(r_max)], maxdegree=4)
    return float(result / pi)


# =================================================================
# WEIL EXPLICIT FORMULA (GL(1) trace formula)
# =================================================================

def weil_explicit_spectral(zeros_imag, w_hat_positive, L, n_zeros=30):
    """
    Spectral side of Weil explicit formula:
    Sum_rho h(gamma_rho)
    = Sum_{k=1}^{K} |H_w(rho_k)|^2  (for h = |H_w|^2)

    This should equal B(L) + C(L) from Session 43.
    """
    total = 0.0
    contributions = []
    for k in range(min(n_zeros, len(zeros_imag))):
        gamma = zeros_imag[k]
        Hw = compute_Hw_at_r(gamma, w_hat_positive, L)
        hw_sq = float(fabs(Hw)**2)
        total += hw_sq
        contributions.append(hw_sq)

    return total, contributions


def weil_explicit_arithmetic_fast(lam_sq, N_basis=None):
    """
    Compute the arithmetic side of the Weil explicit formula DIRECTLY
    from the Connes matrix, bypassing zero computation entirely.

    The Weil explicit formula says:
      Sum_rho h(gamma) = <w, W02 w> - <w, M_prime w> + analytic_terms
                       = B(L) + corrections

    But we know <w, Q_W w> = <w, W02 w> - <w, M w> = B(L).
    And M = M_diag + M_prime + M_alpha.

    So the arithmetic side = <w, W02 w> (spectral weight for all rho)
    minus the prime contribution <w, M_prime w>.

    The KEY insight: <w, W02 w> = Sum_rho |H_w(rho)|^2 + continuous spectrum
    is the FULL spectral side. The arithmetic side is the trace formula
    evaluated on the PRIME side.

    For the Connes barrier:
      Q_W = W02 - M   (where M encodes primes)
      B(L) = <w, Q_W w> = <w, W02 w> - <w, M w>

    The Selberg/Weil trace formula relates:
      <w, W02 w> = identity + hyperbolic + elliptic + parabolic contributions

    So B(L) = (trace formula geometric side for W02) - <w, M w>
    But <w, M w> IS the hyperbolic contribution (primes = geodesics in GL(1)).

    Therefore: B(L) = identity + elliptic + parabolic - (extra M terms beyond geodesics)
    """
    from connes_crossterm import build_all
    L_f = np.log(lam_sq)
    if N_basis is None:
        N_basis = max(15, round(6 * L_f))

    W02, M, QW = build_all(lam_sq, N_basis)
    ns = np.arange(-N_basis, N_basis + 1, dtype=float)
    w = ns / (L_f**2 + (4*np.pi)**2 * ns**2)
    w[N_basis] = 0.0
    w_hat = w / np.linalg.norm(w)

    # The three pieces
    w02_piece = float(w_hat @ W02 @ w_hat)  # Full spectral weight
    m_piece = float(w_hat @ M @ w_hat)       # Prime contribution
    barrier = float(w_hat @ QW @ w_hat)      # B(L) = W02 - M

    return {
        'W02': w02_piece,
        'M': m_piece,
        'barrier': barrier,
        'lam_sq': lam_sq,
        'L': L_f,
        'N_basis': N_basis,
    }


# =================================================================
# GEOMETRIC SIDE DECOMPOSITION
# =================================================================

def decompose_geometric_side(lam_sq, N_basis=None):
    """
    Decompose the barrier into trace formula contributions.

    In the GL(1) trace formula (Weil explicit formula):
    B(L) = <w, Q_W w> = spectral_side - prime_side

    The spectral side Sum |H_w(rho)|^2 comes from W02.
    The prime side comes from M = M_diag + M_prime + M_alpha.

    We decompose M to understand the geometric side:
    - M_diag: diagonal (analytic, no primes)
    - M_prime: off-diagonal prime terms
    - M_alpha: cross terms (mixed)

    The question: is there a natural decomposition where each piece
    can be shown positive or bounded?
    """
    from connes_crossterm import build_all
    L_f = np.log(lam_sq)
    if N_basis is None:
        N_basis = max(15, round(6 * L_f))

    W02, M, QW = build_all(lam_sq, N_basis)
    ns = np.arange(-N_basis, N_basis + 1, dtype=float)
    w = ns / (L_f**2 + (4*np.pi)**2 * ns**2)
    w[N_basis] = 0.0
    w_hat = w / np.linalg.norm(w)

    dim = 2 * N_basis + 1

    # Decompose M into components
    M_diag_matrix = np.diag(np.diag(M))
    M_offdiag = M - M_diag_matrix

    # The prime matrix M_prime has entries involving log(p)/p^{|m-n|/2}
    # The diagonal is the "identity" contribution in trace formula language
    # The off-diagonal is the "hyperbolic" contribution (primes = geodesics)

    w02 = float(w_hat @ W02 @ w_hat)
    m_total = float(w_hat @ M @ w_hat)
    m_diag = float(w_hat @ M_diag_matrix @ w_hat)
    m_offdiag = float(w_hat @ M_offdiag @ w_hat)
    barrier = float(w_hat @ QW @ w_hat)

    # Eigenvalue analysis of Q_W
    qw_eigs = np.linalg.eigvalsh(QW)

    return {
        'W02': w02,
        'M_total': m_total,
        'M_diag': m_diag,
        'M_offdiag': m_offdiag,
        'barrier': barrier,
        'Q_W_min_eig': float(qw_eigs[0]),
        'Q_W_max_eig': float(qw_eigs[-1]),
        'Q_W_n_positive': int(np.sum(qw_eigs > 0)),
        'Q_W_n_total': len(qw_eigs),
        'dim': dim,
    }


# =================================================================
# TEST FUNCTION ADMISSIBILITY FOR SELBERG
# =================================================================

def check_selberg_admissibility(w_hat_positive, L, r_values=None):
    """
    Check if h(r) = |H_w(1/2+ir)|^2 satisfies Selberg trace formula conditions:

    1. h(r) is even: h(r) = h(-r) [automatic since |...|^2]
    2. h(r) is holomorphic in strip |Im(r)| <= 1/2 + epsilon
    3. h(r) = O((1+|r|)^{-2-delta}) as r -> infinity
    4. h(r) >= 0 for all r [automatic since |...|^2]

    Condition 3 is the key one -- check decay rate numerically.
    """
    if r_values is None:
        r_values = [1, 5, 10, 20, 50, 100, 200]

    print('\n  -- TEST FUNCTION ADMISSIBILITY --')
    print(f'\n  h(r) = |H_w(1/2 + ir)|^2')
    print(f'\n  {"r":>8}  {"h(r)":>14}  {"r^2*h(r)":>14}  {"r^3*h(r)":>14}')
    print('  ' + '-' * 56)

    h_values = []
    for r in r_values:
        h = h_test_function(r, w_hat_positive, L)
        h_values.append(h)
        print(f'  {r:8.1f}  {h:14.6e}  {r**2*h:14.6e}  {r**3*h:14.6e}')

    # Check decay: fit log(h) vs log(r) for large r
    large_r = [(r, h) for r, h in zip(r_values, h_values) if r >= 10 and h > 0]
    if len(large_r) >= 2:
        log_r = np.log([x[0] for x in large_r])
        log_h = np.log([x[1] for x in large_r])
        slope, intercept = np.polyfit(log_r, log_h, 1)
        print(f'\n  Decay rate: h(r) ~ r^{slope:.2f}')
        print(f'  Selberg needs: exponent < -2 (got {slope:.2f})')
        print(f'  Admissible: {slope < -2}')
        return slope, h_values
    else:
        print('\n  Not enough data for decay fit')
        return None, h_values


# =================================================================
# THE KEY QUESTION: GEOMETRIC POSITIVITY
# =================================================================

def analyze_positivity_structure(lam_sq_values):
    """
    For each lambda^2, decompose B(L) and check:
    1. Is W02 >> M? (spectral dominance)
    2. Is M_diag the dominant correction? (Session 34 mechanism)
    3. What fraction is M_offdiag? (prime contribution)
    4. Is there a natural positive decomposition?
    """
    print('\n  -- BARRIER POSITIVITY STRUCTURE --')
    print(f'\n  {"lam^2":>8} {"B(L)":>10} {"W02":>10} {"M_tot":>10} '
          f'{"M_diag":>10} {"M_off":>10} {"off/tot":>8} {"min_eig":>10}')
    print('  ' + '-' * 88)

    for lam_sq in lam_sq_values:
        d = decompose_geometric_side(lam_sq)
        ratio = d['M_offdiag'] / d['M_total'] if d['M_total'] != 0 else 0
        print(f'  {lam_sq:8.0f} {d["barrier"]:10.6f} {d["W02"]:10.4f} '
              f'{d["M_total"]:10.4f} {d["M_diag"]:10.4f} {d["M_offdiag"]:10.4f} '
              f'{ratio:8.4f} {d["Q_W_min_eig"]:10.6f}')


# =================================================================
# MAIN
# =================================================================

if __name__ == '__main__':
    print()
    print('#' * 72)
    print('  SESSION 48b -- SELBERG WITH BARRIER TEST FUNCTION')
    print('#' * 72)

    # Load zeros
    print('\n  Loading zeta zeros...', flush=True)
    n_zeros = 30
    zeros_imag = []
    for k in range(1, n_zeros + 1):
        z = zetazero(k)
        zeros_imag.append(float(z.imag))
    zeros_imag = np.array(zeros_imag)
    print(f'  Loaded {n_zeros} zeros')

    # =============================================================
    # PART A: Check test function admissibility
    # =============================================================
    print('\n\n' + '=' * 70)
    print('  A. TEST FUNCTION ADMISSIBILITY')
    print('=' * 70)

    lam_sq = 200
    w_hat_mp, L_mp, N = build_w_hat_mp(lam_sq)
    w_hat_pos = [w_hat_mp[N + n] for n in range(1, N + 1)]

    slope, h_values = check_selberg_admissibility(w_hat_pos, L_mp)

    # =============================================================
    # PART B: Barrier decomposition at multiple scales
    # =============================================================
    print('\n\n' + '=' * 70)
    print('  B. BARRIER DECOMPOSITION (TRACE FORMULA VIEW)')
    print('=' * 70)

    lam_sq_values = [20, 50, 100, 200, 500, 1000, 5000, 10000, 50000]

    analyze_positivity_structure(lam_sq_values)

    # =============================================================
    # PART C: Spectral vs arithmetic comparison
    # =============================================================
    print('\n\n' + '=' * 70)
    print('  C. SPECTRAL vs ARITHMETIC SIDE')
    print('=' * 70)

    print(f'\n  Computing spectral side (Sum |H_w(rho)|^2) at lam^2=200...')
    t0 = time.time()
    spectral_total, contribs = weil_explicit_spectral(
        zeros_imag, w_hat_pos, L_mp, n_zeros=n_zeros
    )
    print(f'  Spectral sum ({n_zeros} zeros): {spectral_total:.8f}  ({time.time()-t0:.1f}s)')

    arith = weil_explicit_arithmetic_fast(200)
    print(f'  W02 (full spectral):  {arith["W02"]:.8f}')
    print(f'  M (prime correction): {arith["M"]:.8f}')
    print(f'  Barrier B(L):         {arith["barrier"]:.8f}')
    print(f'  Gap (W02 - spectral): {arith["W02"] - spectral_total:.8f}'
          f'  (= continuous spectrum + higher zeros)')

    # =============================================================
    # PART D: The positivity argument
    # =============================================================
    print('\n\n' + '=' * 70)
    print('  D. POSITIVITY ARGUMENT')
    print('=' * 70)

    print("""
  The Weil explicit formula (GL(1) trace formula) says:

    Sum_rho h(gamma) = [identity term] + [prime sum] + [analytic corrections]

  For h(r) = |H_w(1/2+ir)|^2:
    - Left side = Sum |H_w(rho)|^2 >= 0  (sum of non-negative terms)
    - Identity term = integral involving h and digamma
    - Prime sum = -2 Sum_p Sum_m log(p)/p^{m/2} * g(m*log(p))
    - Analytic = Gamma-function terms

  B(L) = W02 - M = (full spectral) - (prime matrix)

  The question: does the Weil explicit formula give B(L) > 0
  from the ARITHMETIC side without needing RH?

  Key observation: h(r) >= 0 is a very special condition.
  Most trace formula results use ARBITRARY h. The positivity of h
  may force the arithmetic side to be positive.
  """)

    # Check: is the prime sum negative? (which would help B > 0)
    for lam_sq in [50, 200, 1000, 10000]:
        d = decompose_geometric_side(lam_sq)
        print(f'  lam^2={lam_sq:>6}: M_diag={d["M_diag"]:.4f}  '
              f'M_offdiag={d["M_offdiag"]:.4f}  '
              f'B={d["barrier"]:.6f}  '
              f'W02-M_diag={d["W02"]-d["M_diag"]:.6f}')

    print(f'\n  If W02 - M_diag > 0 and M_offdiag < W02 - M_diag,')
    print(f'  then B = (W02 - M_diag) - M_offdiag > 0.')
    print(f'  The question: can W02 - M_diag be shown positive from')
    print(f'  the trace formula identity term alone?')

    print()
