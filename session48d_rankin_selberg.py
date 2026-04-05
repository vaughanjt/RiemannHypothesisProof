"""
SESSION 48d -- RANKIN-SELBERG L-VALUE IDENTIFICATION

If B(L) = L(1, f x f_bar) for some automorphic form f,
then B(L) > 0 automatically because Petersson norms are positive.

The Rankin-Selberg L-function:
  L(s, f x f_bar) = Sum_{n=1}^{inf} |a_n(f)|^2 / n^s

At s=1: L(1, f x f_bar) = Sum |a_n|^2 / n

For the Ramanujan Delta function:
  L(s, Delta x Delta_bar) = Sum |tau(n)|^2 / n^s

B(L) depends on L through the test vector w_L. So we need either:
  (a) A family of forms f_L parametrized by L
  (b) A fixed form f with B(L) = c(L) * L(s(L), f x f_bar)
  (c) B(L) related to a Rankin-Selberg integral (not just an L-value)

Approach: compute B(L) and several Rankin-Selberg L-values,
look for numerical matches or functional relationships.
"""

import numpy as np
import sys
import time

sys.path.insert(0, '.')
from connes_crossterm import build_all


def compute_barrier(lam_sq):
    """Compute barrier B(L) at given lambda^2."""
    L_f = np.log(lam_sq)
    N = max(15, round(6 * L_f))
    W02, M, QW = build_all(lam_sq, N)
    ns = np.arange(-N, N + 1, dtype=float)
    w = ns / (L_f**2 + (4*np.pi)**2 * ns**2)
    w[N] = 0.0
    w_hat = w / np.linalg.norm(w)
    return float(w_hat @ QW @ w_hat), L_f


def ramanujan_tau(n_max):
    """Compute Ramanujan tau function tau(n) for n=1..n_max.
    tau(n) = n-th coefficient of Delta(z) = q * prod_{n>=1} (1-q^n)^24
    """
    # Use the product formula
    coeffs = np.zeros(n_max + 1)
    coeffs[0] = 1.0  # start with 1

    # Multiply by (1-q^k)^24 for k=1,2,...
    for k in range(1, n_max + 1):
        # (1-q^k)^24: expand using binomial
        # More efficient: multiply iteratively by (1-q^k) 24 times
        for _ in range(24):
            new_coeffs = coeffs.copy()
            for j in range(k, n_max + 1):
                new_coeffs[j] -= coeffs[j - k]
            coeffs = new_coeffs

    # Delta = q * product, so shift by 1
    tau = np.zeros(n_max + 1)
    for n in range(1, n_max + 1):
        tau[n] = coeffs[n - 1]  # coefficient of q^n

    return tau


def rankin_selberg_partial(tau, s_val, n_max):
    """Compute partial sum of L(s, Delta x Delta) = Sum |tau(n)|^2/n^s."""
    total = 0.0
    for n in range(1, min(n_max + 1, len(tau))):
        if tau[n] != 0:
            total += tau[n]**2 / n**s_val
    return total


def eisenstein_rankin_selberg(s_val, n_max=1000):
    """L(s, E_k x E_k) for Eisenstein series.
    For E_k, a_n = sigma_{k-1}(n), so
    L(s, E_k x E_k) = Sum sigma_{k-1}(n)^2 / n^s
    = zeta(s) * zeta(s-k+1) * zeta(s-k+1) * ... (Euler product)

    Actually for weight k Eisenstein series normalized:
    L(s, E_k x E_k) = zeta(s) * zeta(s-2k+2) * L(s-k+1, trivial)

    Simpler: just compute the Dirichlet series directly.
    """
    from sympy import divisor_sigma
    total = 0.0
    for n in range(1, n_max + 1):
        # sigma_{k-1}(n) for k=12 (weight 12 Eisenstein)
        sig = float(divisor_sigma(n, 11))  # k-1 = 11
        total += sig**2 / n**s_val
    return total


def find_rankin_selberg_match(barriers, L_values):
    """
    Try to match B(L) to Rankin-Selberg L-values.

    Strategy 1: B(L) = c * L(s, Delta x Delta) for fixed s, varying c
    Strategy 2: B(L) = L(s(L), Delta x Delta) for varying s
    Strategy 3: B(L) related to the INTEGRAL representation
      L(s, f x f_bar) = integral_0^inf <f, E_s> y^s dy/y
    """
    print('\n  Computing Ramanujan tau coefficients...')
    n_max = 500
    tau = ramanujan_tau(n_max)
    print(f'  tau(1)={tau[1]:.0f}, tau(2)={tau[2]:.0f}, tau(3)={tau[3]:.0f}, '
          f'tau(4)={tau[4]:.0f}, tau(5)={tau[5]:.0f}')

    # Strategy 1: Fixed s, check ratio B(L) / L(s, Delta x Delta)
    print('\n  -- Strategy 1: B(L) = c * L(s, Delta x Delta) --')
    for s_val in [1.0, 1.5, 2.0]:
        ls = rankin_selberg_partial(tau, s_val, n_max)
        print(f'\n  L({s_val}, Delta x Delta) partial sum ({n_max} terms) = {ls:.6e}')
        print(f'  {"L":>8} {"B(L)":>12} {"B/L-val":>14} {"ratio stable?":>14}')
        ratios = []
        for i, (b, L) in enumerate(zip(barriers, L_values)):
            ratio = b / ls if ls != 0 else float('inf')
            ratios.append(ratio)
            print(f'  {L:8.3f} {b:12.6f} {ratio:14.6e}')
        if len(ratios) > 1:
            cv = np.std(ratios) / np.mean(ratios) if np.mean(ratios) != 0 else float('inf')
            print(f'  Coefficient of variation: {cv:.4f} (0 = perfect match)')

    # Strategy 2: For each L, find s such that B(L) = L(s, Delta x Delta)
    print('\n\n  -- Strategy 2: B(L) = L(s(L), Delta x Delta) --')
    print(f'  Finding s(L) such that L(s, Delta x Delta) = B(L)...')

    # Precompute L(s) for a range of s
    s_range = np.linspace(10, 30, 100)
    ls_values = [rankin_selberg_partial(tau, s, n_max) for s in s_range]

    print(f'  {"L":>8} {"B(L)":>12} {"s(L)":>10} {"L(s,DxD)":>14} {"error":>12}')
    s_found = []
    for b, L in zip(barriers, L_values):
        # Find s where L(s) is closest to B
        diffs = [abs(lv - b) for lv in ls_values]
        best_idx = np.argmin(diffs)
        best_s = s_range[best_idx]
        best_ls = ls_values[best_idx]
        s_found.append(best_s)
        print(f'  {L:8.3f} {b:12.6f} {best_s:10.3f} {best_ls:14.6f} {diffs[best_idx]:12.6e}')

    if len(s_found) > 1:
        # Check if s(L) has a simple functional form
        from numpy.polynomial import polynomial as P
        coeffs = np.polyfit(L_values, s_found, 1)
        print(f'\n  Linear fit: s(L) ~ {coeffs[0]:.4f} * L + {coeffs[1]:.4f}')
        residuals = [s - (coeffs[0]*L + coeffs[1]) for s, L in zip(s_found, L_values)]
        print(f'  Max residual: {max(abs(r) for r in residuals):.4f}')

    # Strategy 3: Direct Rankin-Selberg integral connection
    print('\n\n  -- Strategy 3: Rankin-Selberg integral representation --')
    print("""
  The Rankin-Selberg method expresses:
    L(s, f x f_bar) = integral_{SL(2,Z) \\ H} |f(z)|^2 E(z,s) dmu(z)

  where E(z,s) is the Eisenstein series and dmu is hyperbolic measure.

  The barrier B(L) = <w_L, Q_W w_L> where Q_W = W02 - M.

  Q_W is the Connes-Weil quadratic form. It acts on test vectors w_L
  in a (2N+1)-dimensional space indexed by Fourier modes.

  CONNECTION: if w_L can be identified with the Fourier expansion of
  some automorphic form f_L at height y = e^L in the upper half-plane,
  then <w_L, Q_W w_L> might equal a Rankin-Selberg integral.

  The test vector w_L has components:
    w[n] = n / (L^2 + 16*pi^2*n^2)  (Lorentzian)

  This looks like a Poisson kernel or a Cauchy kernel evaluated at
  height y ~ L in the upper half-plane.
  """)


def check_poisson_connection(lam_sq_values):
    """
    Check if the test vector w_L is related to the Poisson kernel
    for the upper half-plane at height y.

    Poisson kernel: P_y(n) = y / (y^2 + n^2)
    Our w_L:        w(n)   = n / (L^2 + 16*pi^2*n^2)

    These are DIFFERENT: Poisson has y in numerator, ours has n.
    But w(n) = (1/16pi^2) * n / ((L/4pi)^2 + n^2)

    This is the CONJUGATE Poisson kernel (Hilbert transform of Poisson)!
    Q_y(n) = n / (y^2 + n^2) is the conjugate Poisson kernel at height y = L/(4pi).

    The conjugate Poisson kernel is the imaginary part of 1/(n + iy),
    while the Poisson kernel is the real part.
    """
    print('\n\n  -- POISSON / CONJUGATE POISSON CONNECTION --')
    print("""
  Test vector: w(n) = n / (L^2 + 16*pi^2*n^2)
             = (1/16pi^2) * n / ((L/4pi)^2 + n^2)

  This is the CONJUGATE Poisson kernel Q_y(n) = n/(y^2+n^2)
  at height y = L/(4pi)!

  The conjugate Poisson kernel is Im[1/(n+iy)].
  The regular Poisson kernel is Re[1/(n+iy)] = y/(y^2+n^2).

  In automorphic forms language:
  - Poisson kernel at height y convolves with a function on R/Z
    to give its value at height y in the upper half-plane
  - Conjugate Poisson gives the HARMONIC CONJUGATE

  So <w_L, Q_W w_L> is the quadratic form Q_W evaluated on a
  conjugate Poisson kernel. This IS related to automorphic forms --
  it's computing something at height y = L/(4pi) in H.
  """)

    print(f'  {"lam^2":>8} {"L":>8} {"y=L/4pi":>10} {"B(L)":>10}')
    print('  ' + '-' * 42)
    for lam_sq in lam_sq_values:
        L_f = np.log(lam_sq)
        y = L_f / (4 * np.pi)
        b, _ = compute_barrier(lam_sq)
        print(f'  {lam_sq:8.0f} {L_f:8.3f} {y:10.4f} {b:10.6f}')

    # Check: does B(L) look like a function of y = L/4pi?
    print('\n  If B is a Rankin-Selberg integral at height y:')
    print('  B(L) = integral |f(z)|^2 * K(z, y) dmu')
    print('  where K is a kernel depending on y = L/(4pi)')
    print('  Then B > 0 iff the integrand is non-negative.')
    print('  |f(z)|^2 >= 0 always, so need K(z,y) >= 0.')


if __name__ == '__main__':
    print()
    print('#' * 72)
    print('  SESSION 48d -- RANKIN-SELBERG L-VALUE IDENTIFICATION')
    print('#' * 72)

    # Compute barriers
    print('\n  Computing barriers...')
    lam_sq_values = [20, 50, 100, 200, 500]
    barriers = []
    L_values = []
    for lam_sq in lam_sq_values:
        t0 = time.time()
        b, L = compute_barrier(lam_sq)
        barriers.append(b)
        L_values.append(L)
        print(f'    lam^2={lam_sq:6.0f}  L={L:.3f}  B={b:.6f}  ({time.time()-t0:.0f}s)')

    # Try Rankin-Selberg matching
    print('\n\n' + '=' * 72)
    print('  RANKIN-SELBERG L-VALUE MATCHING')
    print('=' * 72)

    find_rankin_selberg_match(barriers, L_values)

    # Check Poisson kernel connection
    print('\n\n' + '=' * 72)
    print('  AUTOMORPHIC INTERPRETATION')
    print('=' * 72)

    check_poisson_connection(lam_sq_values)

    print()
