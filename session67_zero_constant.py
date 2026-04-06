"""
SESSION 67 -- THE ZERO-ENCODING CONSTANT

Pi encodes primes: pi^2/6 = Product_p 1/(1-1/p^2).
What constant encodes zeros?

Known: Sum_rho 1/(rho(1-rho)) = 2 + gamma_Euler - log(4*pi)
For rho = 1/2 + i*g: rho(1-rho) = 1/4 + g^2
So: Sum_gamma 2/(1/4 + gamma^2) = 2 + gamma_E - log(4*pi)

The sign lemma sensitivity: Sum_gamma 4/gamma^2 (related but not identical).

Question: does Z_SL = 4 * Sum 1/gamma^2 have a closed form in terms of
known constants? And does this close the proof?
"""

import sys
import numpy as np
import mpmath
from mpmath import mp, mpf, zetazero, euler as gamma_euler

mp.dps = 30


def run():
    print()
    print('#' * 76)
    print('  SESSION 67 -- THE ZERO-ENCODING CONSTANT')
    print('#' * 76)

    # ==================================================================
    # PART 1: COMPUTE THE KNOWN ZERO CONSTANTS
    # ==================================================================
    print('\n  === PART 1: KNOWN ZERO CONSTANTS ===\n')

    # The Hadamard constant
    ge = float(gamma_euler)
    hadamard_sum = 2 + ge - np.log(4 * np.pi)
    print(f'  gamma_Euler = {ge:.10f}')
    print(f'  log(4*pi) = {np.log(4*np.pi):.10f}')
    print(f'  2 + gamma_E - log(4*pi) = {hadamard_sum:.10f}')
    print(f'  This equals Sum_rho 1/(rho(1-rho)) = Sum_gamma 2/(1/4+g^2)')
    print()

    # Compute Sum 1/(1/4 + g^2) from zeros
    n_zeros = 500
    print(f'  Computing {n_zeros} zeros...', end='', flush=True)
    gammas = []
    for k in range(1, n_zeros + 1):
        gammas.append(float(mpmath.im(zetazero(k))))
    print(' done.')
    gammas = np.array(gammas)

    # Sum 2/(1/4 + g^2)
    sum_hadamard = np.sum(2.0 / (0.25 + gammas**2))
    # Tail correction: integral from gamma_K to infinity
    gK = gammas[-1]
    tail_had = 2 * np.log(gK) / (2*np.pi*gK)  # approximate tail
    sum_hadamard_corr = sum_hadamard + tail_had

    print(f'  Sum_{{k=1}}^{{{n_zeros}}} 2/(1/4+g_k^2) = {sum_hadamard:.10f}')
    print(f'  + tail correction:                    {sum_hadamard_corr:.10f}')
    print(f'  Known value:                          {hadamard_sum:.10f}')
    print(f'  Agreement: {abs(sum_hadamard_corr - hadamard_sum):.2e}')
    sys.stdout.flush()

    # ==================================================================
    # PART 2: THE SIGN LEMMA CONSTANT
    # ==================================================================
    print('\n  === PART 2: THE SIGN LEMMA CONSTANT Z_SL ===\n')

    # Sum 4/gamma^2
    sum_inv_g2 = np.sum(1.0 / gammas**2)
    Z_SL = 4 * sum_inv_g2

    # Tail: integral_gK^inf 1/g^2 * log(g)/(2*pi) dg = (log(gK)+1)/(2*pi*gK)
    tail_g2 = (np.log(gK) + 1) / (2*np.pi*gK)
    sum_inv_g2_corr = sum_inv_g2 + tail_g2
    Z_SL_corr = 4 * sum_inv_g2_corr

    print(f'  Sum 1/gamma^2 (first {n_zeros}): {sum_inv_g2:.10f}')
    print(f'  + tail correction:          {sum_inv_g2_corr:.10f}')
    print(f'  Z_SL = 4 * Sum 1/gamma^2:  {Z_SL_corr:.10f}')
    print()

    # Relationship to Hadamard constant
    # Sum 1/gamma^2 = Sum 1/(1/4+gamma^2) + Sum 1/(4*gamma^2*(1/4+gamma^2))
    # The second sum = 4*[Sum 1/gamma^2 - Sum 1/(1/4+gamma^2)] ... circular
    # Instead: 1/gamma^2 - 1/(1/4+gamma^2) = 1/(4*gamma^2*(1/4+gamma^2))
    correction = np.sum(1.0 / (4 * gammas**2 * (0.25 + gammas**2)))
    tail_corr = (np.log(gK) + 1) / (2*np.pi * 4 * gK**3)  # approximate
    correction_total = correction + tail_corr

    print(f'  Sum 1/gamma^2 = Sum 1/(1/4+g^2) + correction')
    print(f'  Sum 1/(1/4+g^2) = {hadamard_sum/2:.10f} (= half the Hadamard constant)')
    print(f'  Correction:       {correction_total:.10f}')
    print(f'  Total:            {hadamard_sum/2 + correction_total:.10f}')
    print(f'  Direct:           {sum_inv_g2_corr:.10f}')
    print()

    # Express Z_SL in terms of known constants
    Z_from_known = 4 * (hadamard_sum/2 + correction_total)
    print(f'  Z_SL = 4*(1 + gamma_E/2 - log(4*pi)/2)/2 + 4*correction')
    print(f'       = 2*(1 + gamma_E/2 - log(4*pi)/2) + 4*correction')
    print(f'       = 2 + gamma_E - log(4*pi) + 4*correction')
    print(f'       = {2 + ge - np.log(4*np.pi):.10f} + {4*correction_total:.10f}')
    print(f'       = {2 + ge - np.log(4*np.pi) + 4*correction_total:.10f}')
    sys.stdout.flush()

    # ==================================================================
    # PART 3: xi''(1/2) AND THE ZERO CONSTANT
    # ==================================================================
    print('\n  === PART 3: xi\'\'(1/2) -- THE MASTER CONSTANT ===\n')

    # Sum 1/gamma^2 = -(1/2) * xi''(1/2)/xi(1/2)
    # (from Hadamard product, differentiated twice at s=1/2)

    # Compute xi(1/2) and xi''(1/2) using high-precision mpmath
    mp.dps = 50  # need high precision for second derivative
    s_half = mpf('0.5')

    def xi_func(s):
        return mpmath.mpf('0.5') * s * (s - 1) * \
               mpmath.power(mpmath.pi, -s/2) * \
               mpmath.gamma(s/2) * mpmath.zeta(s)

    xi_half = float(xi_func(s_half))

    # Use mpmath's diff for numerical differentiation (high precision)
    xi_prime_half = float(mpmath.diff(xi_func, s_half, n=1))
    xi_pp_half = float(mpmath.diff(xi_func, s_half, n=2))
    mp.dps = 30

    print(f'  xi(1/2) = {xi_half:.10f}')
    print(f'  xi\'(1/2) = {xi_prime_half:.6e} (should be ~0 by symmetry)')
    print(f'  xi\'\'(1/2) = {xi_pp_half:.10f}')
    print()

    # Sum 1/gamma^2 from xi (corrected sign):
    # xi''/xi(1/2) = -Sum_rho 1/(rho-1/2)^2 = Sum_gamma 2/gamma^2
    # So: Sum 1/gamma^2 = (1/2) * xi''(1/2)/xi(1/2)
    sum_from_xi = (1.0/2) * xi_pp_half / xi_half
    print(f'  Sum 1/gamma^2 = (1/2)*xi\'\'(1/2)/xi(1/2)')
    print(f'               = (1/2)*{xi_pp_half:.6f}/{xi_half:.6f}')
    print(f'               = {sum_from_xi:.10f}')
    print(f'  Direct sum:    {sum_inv_g2_corr:.10f}')
    print(f'  Agreement:     {abs(sum_from_xi - sum_inv_g2_corr):.2e}')
    print()

    Z_master = 2 * xi_pp_half / xi_half
    print(f'  Z_SL = 4*Sum 1/gamma^2 = 2*xi\'\'(1/2)/xi(1/2)')
    print(f'       = {Z_master:.10f}')
    sys.stdout.flush()

    # ==================================================================
    # PART 4: THE PROOF CONNECTION
    # ==================================================================
    print('\n  === PART 4: THE PROOF CONNECTION ===\n')

    print(f'  The sign lemma tail: S = Sum delta_k * 4/gamma_k^2')
    print(f'  Upper bound: S <= delta_max * Z_SL = delta_max * {Z_master:.6f}')
    print(f'  Margin at L: mu(L) ~ 3e-6 * L^(-0.97)')
    print()
    print(f'  For S < mu(L): delta_max < mu(L) / Z_SL = {3e-6:.0e} / ({Z_master:.4f} * L)')
    print()

    print(f'  At L=12.6: delta_max < {3e-6 / (12.6**0.97 * Z_master):.6e}')
    print(f'  At L=100:  delta_max < {3e-6 / (100**0.97 * Z_master):.6e}')
    print(f'  At L=1000: delta_max < {3e-6 / (1000**0.97 * Z_master):.6e}')
    print(f'  As L -> inf: delta_max -> 0')
    print()

    print(f'  This proves: if M_odd < 0 for all L (i.e., if we could verify')
    print(f'  computationally at all L), then all delta_k = 0 (RH).')
    print()
    print(f'  The gap: we can only verify computationally for finite L.')
    print(f'  The margin goes to zero, so the bound delta_max -> 0, but')
    print(f'  we can\'t take L -> inf without infinite computation.')
    print()

    # ==================================================================
    # PART 5: WHAT IF Z_SL HAS A CLEAN CLOSED FORM?
    # ==================================================================
    print(f'  === PART 5: SEARCHING FOR THE CLEAN FORM ===\n')

    # Z_SL = -2*xi''(1/2)/xi(1/2)
    # Is this expressible purely in terms of pi, gamma_E, log(2), etc.?

    # Test various combinations
    pi_val = np.pi
    ge_val = ge

    candidates = {
        'Z_SL': Z_master,
        '2 + gamma_E - log(4*pi)': 2 + ge_val - np.log(4*pi_val),
        '2*(2+gamma_E-log(4*pi))': 2*(2 + ge_val - np.log(4*pi_val)),
        'gamma_E^2': ge_val**2,
        '4/pi^2': 4/pi_val**2,
        '1/(2*pi)': 1/(2*pi_val),
        'gamma_E/pi': ge_val/pi_val,
        'log(2)/pi': np.log(2)/pi_val,
        '(gamma_E-log(2))^2': (ge_val - np.log(2))**2,
        '1 - log(pi)/2': 1 - np.log(pi_val)/2,
        '2*gamma_E': 2*ge_val,
    }

    print(f'  Z_SL = {Z_master:.10f}\n')
    print(f'  {"candidate":>30} {"value":>14} {"ratio Z/val":>12}')
    print('  ' + '-' * 58)

    for name, val in candidates.items():
        ratio = Z_master / val if abs(val) > 1e-15 else float('inf')
        marker = ' <--' if abs(ratio - 1) < 0.01 else ''
        print(f'  {name:>30} {val:>14.10f} {ratio:>12.6f}{marker}')
    sys.stdout.flush()

    # ==================================================================
    # PART 6: THE DEEP QUESTION
    # ==================================================================
    print(f'\n  === PART 6: THE DEEP QUESTION ===\n')
    print(f'  Z_SL = -2*xi\'\'(1/2)/xi(1/2) = {Z_master:.10f}')
    print()
    print(f'  xi(1/2) involves: pi, Gamma(1/4), zeta(1/2)')
    print(f'  xi\'\'(1/2) involves: derivatives of all three')
    print()
    print(f'  zeta(1/2) = {float(mpmath.zeta(s_half)):.10f}')
    print(f'  Gamma(1/4) = {float(mpmath.gamma(mpf("0.25"))):.10f}')
    print()
    print(f'  The "zero master constant" Z_SL connects:')
    print(f'    - Pi (through pi^(-s/2) in xi)')
    print(f'    - Gamma function (through Gamma(s/2) in xi)')
    print(f'    - Zeta itself (through zeta(s) in xi)')
    print(f'    - The zeros (through the Hadamard sum)')
    print()
    print(f'  Z_SL IS the total sensitivity of M_odd to ALL zeros.')
    print(f'  It measures how much "room" the zeros have to move off-line.')
    print(f'  Its value ({Z_master:.6f}) is determined by pi, Gamma, and zeta.')
    sys.stdout.flush()

    # ==================================================================
    print()
    print('=' * 76)
    print('  SESSION 67 VERDICT')
    print('=' * 76)


if __name__ == '__main__':
    run()
