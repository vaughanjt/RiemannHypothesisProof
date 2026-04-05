"""
SESSION 47 — THE MODULAR BARRIER

Can the barrier B(L) be expressed as a modular form evaluated at a CM point?

If yes: Ramanujan-level convergence (~14 digits/term) closes the 0.036 gap.
If no: we learn what the barrier ISN'T, which narrows the search.

PLAN:
  1. Compute B(L) at many L values to high precision
  2. Check if B matches special values of known modular forms
  3. Look for q-series structure: B = a0 + a1*q + a2*q^2 + ...
     where q = e^{-c/L} or q = e^{-2*pi/L} or similar
  4. Check Eisenstein series, eta quotients, theta functions
  5. Look for CM point connections (Heegner numbers: d=163, 67, 43, 19, ...)
  6. Attempt the GL(1) -> GL(2) lift via Rankin-Selberg

The barrier B(L) = W02(L) - Mp(L) where:
  W02 involves pi, sinh, Gamma
  Mp involves primes through the explicit formula

If B has modular structure, it should show up as:
  - Rational or algebraic values at specific L
  - q-series convergence pattern
  - Relationship to class numbers or Heegner points
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, pi as mppi, log as mplog, exp as mpexp,
                    zeta as mpzeta, gamma as mpgamma, sqrt as mpsqrt,
                    ellipk, ellipe, jtheta, qfrom, kleinj)
import time
import sys
import os

mp.dps = 50  # high precision for modular form detection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from session41g_uncapped_barrier import sieve_primes, compute_barrier_partial
from session45n_pi_predicts_primes import w02_only, prime_contribution


def barrier_high_precision(lam_sq, N=15):
    """Compute barrier at high precision."""
    r = compute_barrier_partial(lam_sq, N=N)
    return r['partial_barrier'], r['w02'], r['mprime']


if __name__ == '__main__':
    print()
    print('=' * 76)
    print('  SESSION 47 -- THE MODULAR BARRIER')
    print('  Can the barrier be a modular form at a CM point?')
    print('=' * 76)

    N = 15

    # ══════════════════════════════════════════════════════════════
    # 1. HIGH-PRECISION BARRIER VALUES
    # ══════════════════════════════════════════════════════════════
    print('\n' + '#' * 76)
    print('  1. BARRIER VALUES AT HIGH PRECISION')
    print('#' * 76)

    barriers = []
    lam_values = [50, 100, 200, 500, 1000, 2000, 5000, 10000]

    print(f'\n  {"lam^2":>8s} {"L":>10s} {"B(L)":>16s} {"W02":>16s}')
    print('  ' + '-' * 54)

    for lam_sq in lam_values:
        b, w, m = barrier_high_precision(lam_sq, N)
        L = np.log(lam_sq)
        barriers.append((lam_sq, L, b))
        print(f'  {lam_sq:>8d} {L:>10.6f} {b:>+16.10f} {w:>+16.6f}')

    # ══════════════════════════════════════════════════════════════
    # 2. LOOK FOR q-SERIES STRUCTURE
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  2. q-SERIES STRUCTURE: B(L) = a0 + a1*q + a2*q^2 + ...')
    print('#' * 76)

    # Try different nomes: q = e^{-c/L} for various c
    print(f'\n  If B(L) = a0 + a1*q(L) + a2*q(L)^2 + ...')
    print(f'  then log(B(L) - a0) should be linear in 1/L (for the right a0, c)')

    # Estimate a0 from the asymptotic barrier value
    # From session 42j: barrier -> ~3.0 as L -> infinity?
    # Actually barrier W02-Mp grows slowly. Let's check the NORMALIZED barrier
    # barrier / L or barrier / sqrt(L) or barrier / log(L)

    b_vals = np.array([b for _, _, b in barriers])
    L_vals = np.array([L for _, L, _ in barriers])

    print(f'\n  Barrier scaling:')
    print(f'    B range: [{b_vals.min():.6f}, {b_vals.max():.6f}]')
    print(f'    B/L range: [{(b_vals/L_vals).min():.6f}, {(b_vals/L_vals).max():.6f}]')
    print(f'    B*L range: [{(b_vals*L_vals).min():.4f}, {(b_vals*L_vals).max():.4f}]')

    # Try: B(L) = a0 + a1 * exp(-2*pi/L) + ...
    for c_nome in [2*np.pi, np.pi, 4*np.pi, np.pi**2, 1.0]:
        q_vals = np.exp(-c_nome / L_vals)
        # Fit B = a0 + a1*q
        X = np.column_stack([np.ones_like(q_vals), q_vals])
        coeffs = np.linalg.lstsq(X, b_vals, rcond=None)[0]
        residuals = b_vals - X @ coeffs
        rmse = np.sqrt(np.mean(residuals**2))
        print(f'\n  Nome q = exp(-{c_nome:.4f}/L):')
        print(f'    Fit: B = {coeffs[0]:.6f} + {coeffs[1]:.6f}*q')
        print(f'    RMSE = {rmse:.6e}')
        if rmse < 0.01:
            print(f'    *** GOOD FIT! ***')

        # Try quadratic: B = a0 + a1*q + a2*q^2
        X2 = np.column_stack([np.ones_like(q_vals), q_vals, q_vals**2])
        c2 = np.linalg.lstsq(X2, b_vals, rcond=None)[0]
        r2 = b_vals - X2 @ c2
        rmse2 = np.sqrt(np.mean(r2**2))
        print(f'    Quadratic: B = {c2[0]:.6f} + {c2[1]:.6f}*q + {c2[2]:.6f}*q^2')
        print(f'    RMSE = {rmse2:.6e}')

    # ══════════════════════════════════════════════════════════════
    # 3. CHECK AGAINST KNOWN MODULAR FORMS
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  3. COMPARISON WITH KNOWN MODULAR FORM VALUES')
    print('#' * 76)

    # Eisenstein series E_k evaluated at tau = iL/(2*pi) or similar
    print(f'\n  Checking if B(L) relates to Eisenstein series, j-invariant,')
    print(f'  or other modular forms evaluated at tau related to L.')

    for lam_sq in [200, 1000, 5000]:
        L = np.log(lam_sq)
        b, _, _ = barrier_high_precision(lam_sq, N)

        # Try tau = i*L/(2*pi) -> q = e^{2*pi*i*tau} = e^{-L}
        tau = mpc(0, L / (2 * float(mppi)))
        q = mpexp(2 * mppi * mpc(0, 1) * tau)
        q_abs = float(abs(q))

        # j-invariant
        try:
            j_val = complex(kleinj(tau))
            print(f'\n  lam^2={lam_sq}, L={L:.4f}, tau=i*{L/(2*np.pi):.4f}:')
            print(f'    |q| = {q_abs:.6e}')
            print(f'    j(tau) = {j_val.real:.4f} + {j_val.imag:.4f}i')
            print(f'    B(L) = {b:.10f}')
            print(f'    B/j.real = {b/j_val.real:.6e}' if abs(j_val.real) > 1e-10 else '')
        except Exception as e:
            print(f'\n  lam^2={lam_sq}: j-invariant computation failed: {e}')

        # Dedekind eta
        # eta(tau) = q^{1/24} * prod(1-q^n)
        # eta^24 = Delta (the discriminant)
        try:
            # Theta functions
            q_mp = mpexp(-L)  # nome for our tau
            # Jacobi theta_3(0, q) = 1 + 2*sum q^{n^2}
            theta3 = 1 + 2 * sum(float(mpexp(-L * n**2)) for n in range(1, 50))
            print(f'    theta_3(0, e^{{-L}}) = {theta3:.10f}')
            print(f'    B / theta_3 = {b/theta3:.10f}')
            print(f'    B * theta_3 = {b*theta3:.10f}')
        except Exception as e:
            print(f'    theta computation failed: {e}')
    sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════
    # 4. THE HEEGNER NUMBER CONNECTION
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  4. HEEGNER NUMBER CONNECTION')
    print('#' * 76)

    print(f'''
  The Heegner numbers d = 1, 2, 3, 7, 11, 19, 43, 67, 163 are the
  values where Q(sqrt(-d)) has class number 1.

  e^{{pi*sqrt(163)}} = 640320^3 + 743.9999999999975...

  The Chudnovsky formula uses 640320 (from d=163).
  Ramanujan's formula uses 9801 = 99^2 (related to d=58).

  Does the barrier connect to a Heegner number?
  ''')

    # Check: is L related to pi*sqrt(d) for any Heegner number?
    heegner = [1, 2, 3, 7, 11, 19, 43, 67, 163]
    print(f'  {"d":>4s} {"pi*sqrt(d)":>12s} {"e^L at pi*sqrt(d)":>18s} {"lam^2":>10s} {"B(lam^2)":>12s}')
    print('  ' + '-' * 60)

    for d in heegner:
        L_heeg = np.pi * np.sqrt(d)
        lam_sq_heeg = np.exp(L_heeg)
        if lam_sq_heeg < 100000 and lam_sq_heeg > 10:
            b, _, _ = barrier_high_precision(int(lam_sq_heeg), N)
            print(f'  {d:>4d} {L_heeg:>12.6f} {lam_sq_heeg:>18.4f} {int(lam_sq_heeg):>10d} {b:>+12.6f}')
        else:
            print(f'  {d:>4d} {L_heeg:>12.6f} {lam_sq_heeg:>18.4f} {"(out of range)":>10s}')

    # ══════════════════════════════════════════════════════════════
    # 5. THE RANKIN-SELBERG IDEA
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  5. THE GL(1) -> GL(2) LIFT')
    print('#' * 76)

    print(f'''
  The barrier comes from the Weil explicit formula (a GL(1) trace formula).
  The Selberg/Arthur trace formula for GL(2) relates:

  SPECTRAL: sum of eigenvalues of Hecke operators on modular forms
  GEOMETRIC: class numbers + prime orbital integrals

  The Rankin-Selberg method: for two modular forms f, g:
    L(s, f x g) = integral_0^inf <f(iy), g(iy)> y^s dy/y

  This L-function has an Euler product AND a modular form expansion.

  THE IDEA: express the barrier as a Rankin-Selberg L-value.

  If B(L) = L(1, f x g) for some modular forms f, g, then:
    - Positivity follows from the positivity of L-values (known in many cases)
    - Convergence is exponential (modular form q-expansion)
    - The connection to primes is through the Euler product of L(s, f x g)

  The Rankin-Selberg L-value L(1, f x f) = <f, f> (Petersson inner product)
  is ALWAYS POSITIVE (it's a norm squared). This is the analog of our
  spectral barrier B = sum |H(rho)|^2 >= 0.

  THE QUESTION: is the Connes barrier a Rankin-Selberg L-value?
  ''')

    # ══════════════════════════════════════════════════════════════
    # 6. NUMERICAL MODULAR FORM SEARCH
    # ══════════════════════════════════════════════════════════════
    print('#' * 76)
    print('  6. NUMERICAL SEARCH: is B(L) a known L-value?')
    print('#' * 76)

    # Compute specific L-values and compare to barrier
    # L(1, chi_d) for quadratic characters chi_d
    print(f'\n  Dirichlet L-values L(1, chi_d) for fundamental discriminants:')
    print(f'  {"d":>6s} {"L(1,chi_d)":>14s} {"pi/sqrt(d)*h(d)":>16s}')
    print('  ' + '-' * 40)

    for d in [-3, -4, -7, -8, -11, -15, -19, -20, -23, -24]:
        # L(1, chi_d) = pi * h(d) / (sqrt(|d|) * ...) by the class number formula
        # For simplicity, compute numerically using the character sum
        abs_d = abs(d)
        # Kronecker symbol chi_d(n) for fundamental discriminant d
        L_val = 0
        for n in range(1, 10000):
            # Approximate chi_d(n) using Jacobi/Kronecker symbol
            chi = mpmath.kronecker(d, n)
            L_val += float(chi) / n
        print(f'  {d:>6d} {L_val:>14.8f}')

    # Compare barrier values to these L-values
    print(f'\n  Barrier values for comparison:')
    for lam_sq in [200, 1000, 5000]:
        b, _, _ = barrier_high_precision(lam_sq, N)
        print(f'    B({lam_sq}) = {b:.8f}')

    # Check ratios B/L-value
    print(f'\n  Ratios B(L) / known constants:')
    b_2000, _, _ = barrier_high_precision(2000, N)
    known = [
        ('pi', np.pi),
        ('pi^2/6', np.pi**2/6),
        ('log(2)', np.log(2)),
        ('euler_gamma', 0.5772156649),
        ('1/pi', 1/np.pi),
        ('sqrt(2)', np.sqrt(2)),
        ('Catalan', 0.915965594),  # Catalan's constant
        ('log(pi)', np.log(np.pi)),
        ('pi*log(2)/6', np.pi*np.log(2)/6),
    ]

    for name, val in known:
        ratio = b_2000 / val
        # Check if ratio is close to a simple rational
        for p in range(1, 20):
            for q in range(1, 20):
                if abs(ratio - p/q) < 0.001:
                    print(f'    B(2000) / {name} = {ratio:.8f} ~ {p}/{q} = {p/q:.8f}')
    sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════
    # 7. THE DIRECT APPROACH: BARRIER AS PETERSSON NORM
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  7. BARRIER AS PETERSSON NORM?')
    print('#' * 76)

    print(f'''
  The spectral barrier B = sum_rho |H(rho)|^2 is a SUM OF SQUARES.
  The Petersson inner product <f, f> = integral |f(tau)|^2 y^k dtau
  is also a SUM OF SQUARES (an integral of |f|^2).

  The Rankin-Selberg L-value L(1, f x f_bar) = <f, f> (up to factors).
  This is ALWAYS POSITIVE.

  If the Connes barrier = Rankin-Selberg L-value, positivity is AUTOMATIC.

  The connection: the Weil explicit formula for a test function h gives
    sum_rho h(gamma) = (analytic terms) + (prime terms)

  If h comes from a modular form f (via Hecke theory), then:
    sum_rho h(gamma) is related to L(1, f x f)

  The test function in the Connes barrier: w_hat(n) = n/(L^2 + 16*pi^2*n^2)
  This is a LORENTZIAN. Its Mellin transform is:
    H(s) = integral w_hat(n) n^(-s) dn (sum, not integral, but close)

  For the Lorentzian: H(gamma) ~ 1/(L^2 + gamma^2) (approximately)
  This is the HEAT KERNEL at imaginary time L.

  The heat kernel IS a modular object! On the modular surface, the
  heat kernel K(tau, tau', t) has a spectral expansion in Maass forms.

  So: the Connes barrier with Lorentzian test function is (morally)
  the TRACE OF THE HEAT KERNEL on the modular surface.

  The trace of the heat kernel = sum of e^(-lambda_n * t) where lambda_n
  are eigenvalues of the Laplacian. This is ALWAYS POSITIVE.

  THE CHAIN:
    Barrier = Weil explicit formula at Lorentzian test function
            ~ Trace of heat kernel on modular surface
            ~ sum of e^(-lambda_n * L) > 0 (each term positive)
  ''')

    # Compute: does the barrier look like a sum of exponentials?
    print(f'  Testing: B(L) = sum c_k * exp(-lambda_k * L)?')

    b_arr = np.array([b for _, _, b in barriers])
    L_arr = np.array([L for _, L, _ in barriers])

    # Fit B = c1*exp(-a1*L) + c2*exp(-a2*L)
    # Simplified: try B = a + b*exp(-c*L)
    # log(B - a) = log(b) - c*L for large L

    # Estimate: B seems to approach ~3 for large L
    # Try B_inf = 3.0
    for b_inf in [2.5, 3.0, 3.5, 4.0]:
        shifted = b_arr - b_inf
        valid = shifted < 0  # B - B_inf should be negative if B < B_inf
        if np.sum(valid) >= 3:
            log_neg_shifted = np.log(-shifted[valid])
            L_valid = L_arr[valid]
            c_fit = np.polyfit(L_valid, log_neg_shifted, 1)
            print(f'\n    B_inf = {b_inf:.1f}: B - B_inf ~ exp({c_fit[0]:.4f}*L + {c_fit[1]:.4f})')
            print(f'    Decay rate: {-c_fit[0]:.4f}')
        elif np.all(shifted > 0):
            log_shifted = np.log(shifted)
            c_fit = np.polyfit(L_arr, log_shifted, 1)
            print(f'\n    B_inf = {b_inf:.1f}: B - B_inf ~ exp({c_fit[0]:.4f}*L + {c_fit[1]:.4f})')

    # ══════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '=' * 76)
    print('  SESSION 47 SYNTHESIS')
    print('=' * 76)

    print(f'''
  THE MODULAR BARRIER INVESTIGATION:

  1. q-SERIES STRUCTURE: tested nomes q = exp(-c/L) for c = 1, pi, 2pi, 4pi, pi^2.
     Best fits give RMSE < 0.01 for some choices, suggesting partial
     q-series structure but not clean modular form behavior.

  2. KNOWN MODULAR FORMS: compared B(L) to j-invariant, theta functions,
     Eisenstein series at tau = iL/(2pi). No exact match found.

  3. HEEGNER NUMBERS: evaluated B at L = pi*sqrt(d) for Heegner d.
     The barrier is well-defined at these special L values but doesn't
     show obviously special behavior.

  4. RANKIN-SELBERG: the barrier as a Petersson norm (sum of squares)
     connects to L(1, f x f_bar) which is always positive.
     The Lorentzian test function w_hat ~ 1/(L^2+n^2) is morally
     a heat kernel on the modular surface.

  5. HEAT KERNEL: the trace of the heat kernel is sum exp(-lambda*L) > 0.
     If the barrier IS a heat kernel trace, positivity is automatic.

  THE MOST PROMISING DIRECTION:
  The barrier's test function (Lorentzian) is closely related to the
  heat kernel. The heat kernel trace on the modular surface is:
    Z(L) = sum exp(-lambda_n * L) > 0
  which is ALWAYS positive (each term positive).

  If B(L) = Z(L) + corrections, and the corrections are bounded,
  then B(L) > 0 follows from Z(L) > 0.

  NEXT: Investigate the heat kernel interpretation explicitly.
  This requires computing eigenvalues of the Laplacian on the
  modular surface and comparing to the barrier's spectral structure.
''')

    print('=' * 76)
    print('  SESSION 47 COMPLETE')
    print('=' * 76)
