"""
SESSION 78 -- THE LINE CONSTRAINT

RH says zeros lie on Re(s) = 1/2, a LINE in the complex plane.
User insight: in whatever embedding, zeros trace a line, and we need
to show this line doesn't deviate.

Session 64 showed: moving a zero OFF the critical line (sigma != 1/2)
pushes M_odd eigenvalue toward positive. Coefficient P[n,n] > 0.

New question: what about moving a zero ALONG the critical line
(shifting gamma)? Is this a NEUTRAL perturbation?

If off-line = destabilizing and along-line = neutral, then the
critical line is a RIDGE: the unique 1-dimensional path where
the Lorentzian property survives.

PROBES:
  1. Build M zero-by-zero: add zeros one at a time, track M_odd spectrum
  2. Off-line perturbation: shift one zero to sigma != 1/2, measure damage
  3. Along-line perturbation: shift one zero's gamma, measure effect
  4. The constraint surface: what tube of zero configurations preserves
     M_odd < 0? How wide is it? Does it shrink to a line?
  5. All zeros on a DIFFERENT line (Re(s) = sigma, sigma != 1/2): what happens?
  6. The functional equation constraint: zeros come in pairs (rho, 1-rho_bar).
     What does this pairing buy us?
"""

import sys
import numpy as np
import mpmath

sys.path.insert(0, '.')
from session49c_weil_residual import (
    build_all_fast, _compute_alpha, _compute_wr_diag
)
from session41g_uncapped_barrier import sieve_primes

mpmath.mp.dps = 30


def odd_block(M, N):
    dim = 2 * N + 1
    P = np.zeros((dim, N))
    for n in range(1, N + 1):
        P[N + n, n - 1] = 1.0 / np.sqrt(2)
        P[N - n, n - 1] = -1.0 / np.sqrt(2)
    return P.T @ M @ P


def build_M_with_fake_zeros(lam_sq, fake_zeros, N=None):
    """Build M using fake zeros instead of actual zeta zeros.

    fake_zeros: list of (sigma, gamma) pairs.
    Each pair contributes as if there were a zero at sigma + i*gamma
    and its functional-equation partner at (1-sigma) + i*gamma.

    For sigma = 1/2, the pair is a conjugate pair on the critical line.
    For sigma != 1/2, it's an off-line pair.
    """
    L = float(np.log(lam_sq))
    if N is None:
        N = max(15, round(6 * L))
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)

    # Start with archimedean + prime parts (these don't change)
    _, M_real, _ = build_all_fast(lam_sq, N)

    # Now we need to understand what the zeros contribute.
    # In the Weil explicit formula, the zero contribution to M is
    # encoded THROUGH the primes via the explicit formula. The matrix M
    # is built from primes directly, not from zeros.
    #
    # But we CAN test the SIGN LEMMA approach: perturb one zero and
    # see the first-order effect on M_odd's eigenvalue.
    #
    # The sign lemma (Session 64) gives:
    #   d(eig_max(M_odd))/d(sigma) = v^T * (dM/d(sigma)) * v
    # where v is the critical eigenvector.
    #
    # For a zero at rho = sigma + i*gamma contributing to the explicit formula:
    #   The contribution to M involves h(rho) where h is the test function.
    #   Moving rho off the critical line changes h(rho).
    #
    # Actually, the cleanest approach: USE the explicit formula directly.
    # M encodes the Weil explicit formula. The zeros appear through
    # sum_rho h(rho) where h is the Lorentzian test function.
    #
    # For our Cauchy-Loewner matrix, the off-diagonal is:
    #   M_{nm} = (B_m - B_n)/(n-m) where B_n = alpha_n + sum_p (prime terms)
    # The zeros don't appear directly — they appear through the explicit
    # formula IDENTITY that connects sum_rho to sum_p.
    #
    # So we can't directly "add zeros" to M. Instead, we test:
    # if RH is true, B_n has specific values. If a zero moves off-line,
    # B_n changes. We compute this change.

    return M_real, N, L, dim, ns


def build_M_direct_from_zeros(lam_sq, zero_gammas, N=None, sigma=0.5):
    """Build the ZERO CONTRIBUTION to the Weil explicit formula.

    For a zero at rho = sigma + i*gamma, the explicit formula contributes:
    h(rho) = L^2 / (L^2 + (2*pi*(sigma - 1/2) + i*2*pi*gamma/L)^2 * something)

    Actually, the test function for our Lorentzian is:
    h(r) = L / (L^2/4 + r^2) where r = gamma for on-line zeros.

    The zero-sum side of the explicit formula gives:
    sum_rho h(rho - 1/2) = sum_gamma L / (L^2/4 + gamma^2)

    For M, the n-th Fourier component of the zero sum is:
    Z_n = sum_gamma [L / (L^2/4 + gamma^2)] * cos(gamma * 2*pi*n/L)  (even)
    or sin for odd.

    If a zero moves to sigma != 1/2, the contribution changes.
    The test function evaluated at rho = sigma + i*gamma:
    h(rho - 1/2) = L / (L^2/4 + (gamma + i*(sigma-1/2))^2)
    which has COMPLEX value if sigma != 1/2.

    But the explicit formula requires pairing with 1-rho_bar:
    rho = sigma + i*gamma, 1-rho_bar = (1-sigma) + i*gamma
    h(rho - 1/2) + h(1-rho_bar - 1/2) = h((sigma-1/2) + i*gamma) + h((1/2-sigma) + i*gamma)

    For sigma = 1/2: 2 * L / (L^2/4 + gamma^2) = 2h(gamma)  [real, positive]
    For sigma != 1/2: sum is still real (by symmetry) but different value.
    """
    L = float(np.log(lam_sq))
    if N is None:
        N = max(15, round(6 * L))
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)

    # Compute the zero-sum contribution for each n
    delta = sigma - 0.5
    Z_diag = np.zeros(dim)
    Z_B = np.zeros(dim)

    for gamma in zero_gammas:
        # h(rho - 1/2) for rho = sigma + i*gamma
        # rho - 1/2 = delta + i*gamma
        # h(z) = L / (L^2/4 + z^2) where z is complex
        # z = delta + i*gamma
        # z^2 = delta^2 - gamma^2 + 2i*delta*gamma
        # L^2/4 + z^2 = L^2/4 + delta^2 - gamma^2 + 2i*delta*gamma
        # h = L / (L^2/4 + delta^2 - gamma^2 + 2i*delta*gamma)

        # Pair with 1-rho_bar: 1-(sigma-i*gamma) = (1-sigma) + i*gamma
        # (1-rho_bar) - 1/2 = (1/2-sigma) + i*gamma = -delta + i*gamma
        # h(-delta + i*gamma) = L / (L^2/4 + delta^2 - gamma^2 - 2i*delta*gamma)
        # Sum: h_pair = h(delta+i*gamma) + h(-delta+i*gamma)
        # = L * [denom_conj + denom] / |denom|^2
        # where denom = L^2/4 + delta^2 - gamma^2 + 2i*delta*gamma

        re_denom = L**2 / 4 + delta**2 - gamma**2
        im_denom = 2 * delta * gamma
        abs_denom_sq = re_denom**2 + im_denom**2

        h_pair_real = L * 2 * re_denom / abs_denom_sq

        # On-line value (delta=0): h_pair = 2*L / (L^2/4 + gamma^2) [always positive since L^2/4 < gamma^2 for most zeros... wait, that could be negative if L^2/4 > gamma^2]
        # Actually for gamma > L/2, L^2/4 - gamma^2 < 0, so re_denom < 0, so h_pair could be negative
        # For gamma < L/2, h_pair > 0 at delta=0.

        # Fourier components: the contribution to M involves
        # h_pair * cos(gamma * 2*pi*n/L) for the diagonal-like terms
        # and h_pair * sin(gamma * 2*pi*n/L) / pi for the B_n terms

        phase = gamma * 2 * np.pi * ns / L
        Z_diag += h_pair_real * np.cos(phase)
        Z_B += h_pair_real * np.sin(phase) / np.pi

    return Z_diag, Z_B, N, L, dim, ns


def run():
    print()
    print('#' * 76)
    print('  SESSION 78 -- THE LINE CONSTRAINT')
    print('#' * 76)

    # Load zeros
    n_zeros = 30
    zeros = [float(mpmath.zetazero(k).imag) for k in range(1, n_zeros + 1)]

    # ======================================================================
    # PROBE 1: Zero-by-zero effect on M_odd eigenvalue
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 1: SIGN LEMMA — OFF-LINE vs ALONG-LINE PERTURBATION')
    print(f'{"="*76}\n')

    lam_sq = 1000
    L = np.log(lam_sq)
    N = max(15, round(6 * L))
    _, M, _ = build_all_fast(lam_sq, N)
    Mo = odd_block(M, N)
    dim_odd = N

    evals_o, evecs_o = np.linalg.eigh(Mo)
    eig_max = evals_o[-1]
    v_crit = evecs_o[:, -1]

    print(f'  Baseline: lam^2={lam_sq}, M_odd eig_max = {eig_max:+.6e}')
    print(f'  Critical eigenvector: v[1]={v_crit[0]:+.4f}, v[2]={v_crit[1]:+.4f}')
    print()

    # For each zero, compute the SENSITIVITY of eig_max to:
    # (a) moving sigma from 1/2 to 1/2 + epsilon  (OFF-LINE)
    # (b) moving gamma to gamma + epsilon  (ALONG-LINE)
    #
    # Use the Hellmann-Feynman theorem:
    #   d(eig)/d(param) = v^T * (dM/d(param)) * v
    #
    # For the explicit formula, M depends on zeros through the sum
    #   sum_rho h(rho - 1/2)
    # The zero contribution to the n-th Fourier component is:
    #   h_pair(gamma, delta) * cos(gamma * 2*pi*n/L)  (even)
    #   h_pair(gamma, delta) * sin(gamma * 2*pi*n/L) / pi  (odd B_n)

    # Compute h_pair and its derivatives
    print(f'  {"zero#":>6} {"gamma":>10} {"h_pair(on)":>12} '
          f'{"dh/d(sigma)":>14} {"dh/d(gamma)":>14} {"ratio sigma/gamma":>18}')
    print('  ' + '-' * 80)

    for k in range(min(15, len(zeros))):
        gamma = zeros[k]
        delta = 0  # on-line

        # h_pair at delta=0: 2*L / (L^2/4 - gamma^2 + delta^2)
        # Wait, need to be more careful with the test function
        # h(r) for the Lorentzian test function is the Poisson kernel type

        # Using the formula from build_all_fast:
        # The actual test function in the Weil formula at our truncation is
        # determined by the matrix definition. Let me use finite differences.

        # h_pair(delta, gamma) = L * 2 * (L^2/4 + delta^2 - gamma^2) /
        #                        ((L^2/4 + delta^2 - gamma^2)^2 + (2*delta*gamma)^2)

        eps = 1e-6

        def h_pair(delta, gamma):
            re = L**2/4 + delta**2 - gamma**2
            im = 2 * delta * gamma
            return L * 2 * re / (re**2 + im**2)

        h0 = h_pair(0, gamma)
        dh_dsigma = (h_pair(eps, gamma) - h_pair(-eps, gamma)) / (2 * eps)
        dh_dgamma = (h_pair(0, gamma + eps) - h_pair(0, gamma - eps)) / (2 * eps)

        ratio = abs(dh_dsigma / dh_dgamma) if abs(dh_dgamma) > 1e-20 else float('inf')

        print(f'  {k+1:>6d} {gamma:>10.4f} {h0:>12.6f} '
              f'{dh_dsigma:>+14.6e} {dh_dgamma:>+14.6e} {ratio:>18.4f}')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 2: Off-line perturbation — shift ALL zeros by delta
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 2: SHIFT ALL ZEROS OFF THE CRITICAL LINE')
    print(f'{"="*76}\n')

    # The zero contribution Z(delta) = sum_k h_pair(delta, gamma_k) * basis_k
    # At delta=0 (on-line), this gives the actual zero contribution.
    # At delta != 0, the contribution changes.
    #
    # We can compute Z(delta) - Z(0) and add it to M to simulate
    # what would happen if all zeros were at sigma = 1/2 + delta.

    print(f'  Shifting all zeros to sigma = 1/2 + delta:')
    print(f'  {"delta":>10} {"sum h_pair":>14} {"change from on-line":>18}')
    print('  ' + '-' * 46)

    for delta in [0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.49]:
        total = sum(L * 2 * (L**2/4 + delta**2 - g**2) /
                    ((L**2/4 + delta**2 - g**2)**2 + (2*delta*g)**2)
                    for g in zeros)
        total_online = sum(L * 2 * (L**2/4 - g**2) /
                          ((L**2/4 - g**2)**2)
                          for g in zeros)
        print(f'  {delta:>10.4f} {total:>+14.6f} {total - total_online:>+18.6f}')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 3: The LINE in eigenvalue space
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 3: ZERO-BY-ZERO CONTRIBUTION TO M_ODD EIGENVALUE')
    print(f'{"="*76}\n')

    # For each zero gamma_k, compute its contribution to the Rayleigh
    # quotient v_crit^T M v_crit.
    #
    # Each zero contributes through the explicit formula via the prime sum.
    # But we CAN decompose the h_pair sum into per-zero contributions
    # and see how they affect the critical eigenvalue.

    # The zero contribution to B_n is (schematically):
    #   B_n^{zero} = sum_k h(gamma_k) * sin(gamma_k * 2*pi*n/L) / pi
    #
    # This enters M through the Cauchy off-diagonal.
    # The contribution to the diagonal a_n is:
    #   a_n^{zero} = sum_k h(gamma_k) * cos(gamma_k * 2*pi*n/L)

    # Build the odd-block projection
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)
    P_odd = np.zeros((dim, N))
    for n in range(1, N + 1):
        P_odd[N + n, n - 1] = 1.0 / np.sqrt(2)
        P_odd[N - n, n - 1] = -1.0 / np.sqrt(2)

    print(f'  Rayleigh quotient decomposition on critical eigenvector:')
    print(f'  {"zero#":>6} {"gamma":>10} {"h(gamma)":>12} '
          f'{"diag contrib":>14} {"offdiag contrib":>14} {"total":>14}')
    print('  ' + '-' * 76)

    cum_total = 0
    for k in range(min(20, len(zeros))):
        gamma = zeros[k]

        # h_pair at delta=0
        h = 2 * L / (L**2/4 - gamma**2) if abs(L**2/4 - gamma**2) > 1e-10 else 0
        # Note: this can be negative when gamma > L/2

        # Zero k's contribution to the diagonal (even in n):
        cos_phase = np.cos(gamma * 2 * np.pi * ns / L)
        # Zero k's contribution to B_n (odd in n):
        sin_phase = np.sin(gamma * 2 * np.pi * ns / L) / np.pi

        # Build the delta_M from this zero
        delta_diag = h * cos_phase

        nm = ns[:, None] - ns[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            delta_offdiag = h * (sin_phase[None, :] - sin_phase[:, None]) / nm
        np.fill_diagonal(delta_offdiag, 0)
        delta_M = np.diag(delta_diag) + delta_offdiag
        delta_M = (delta_M + delta_M.T) / 2

        # Project to odd block
        delta_Mo = P_odd.T @ delta_M @ P_odd

        # Rayleigh quotient on critical eigenvector
        rq_diag = v_crit @ np.diag(np.diag(delta_Mo)) @ v_crit
        rq_offdiag = v_crit @ (delta_Mo - np.diag(np.diag(delta_Mo))) @ v_crit
        rq_total = v_crit @ delta_Mo @ v_crit
        cum_total += rq_total

        print(f'  {k+1:>6d} {gamma:>10.4f} {h:>+12.6f} '
              f'{rq_diag:>+14.6e} {rq_offdiag:>+14.6e} {rq_total:>+14.6e}')

    print(f'\n  Cumulative zero contribution to Rayleigh quotient: {cum_total:+.6e}')
    print(f'  Actual M_odd eig_max: {eig_max:+.6e}')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 4: The RIDGE — is sigma=1/2 a critical point?
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 4: IS SIGMA=1/2 A CRITICAL POINT OF h_pair?')
    print(f'{"="*76}\n')

    # For h_pair(delta, gamma) with delta = sigma - 1/2:
    # At delta=0: dh/d(delta) = ?
    #
    # h_pair = 2L * (L^2/4 + delta^2 - gamma^2) /
    #          ((L^2/4 + delta^2 - gamma^2)^2 + 4*delta^2*gamma^2)
    #
    # At delta=0:
    # h_pair = 2L * (L^2/4 - gamma^2) / (L^2/4 - gamma^2)^2
    #        = 2L / (L^2/4 - gamma^2)
    #
    # dh/d(delta)|_{delta=0}:
    # Numerator derivative: 2*delta = 0
    # Denominator derivative: 2*(L^2/4-gamma^2)*2*delta + 4*2*delta*gamma^2 = 0
    # So dh/d(delta)|_{delta=0} = 0 !!!
    #
    # The critical line IS a critical point of h_pair!
    # d^2h/d(delta^2) determines whether it's a max or min.

    print(f'  h_pair(delta, gamma) = 2L*(L^2/4 + delta^2 - gamma^2) / ((...)^2 + 4*delta^2*gamma^2)')
    print(f'  At delta=0: dh/d(delta) = 0 for ALL gamma.')
    print(f'  The critical line is a STATIONARY POINT of h_pair.')
    print()
    print(f'  Second derivative test (d^2h/d(delta^2) at delta=0):')
    print(f'  {"gamma":>10} {"h(0)":>12} {"d^2h/ddelta^2":>16} {"type":>8}')
    print('  ' + '-' * 50)

    for gamma in zeros[:15]:
        def h_pair(d):
            re = L**2/4 + d**2 - gamma**2
            im = 2 * d * gamma
            return 2 * L * re / (re**2 + im**2)

        eps = 1e-4
        h0 = h_pair(0)
        d2h = (h_pair(eps) - 2*h_pair(0) + h_pair(-eps)) / eps**2
        typ = 'MAX' if d2h < 0 else 'min' if d2h > 0 else 'inflect'
        print(f'  {gamma:>10.4f} {h0:>+12.6f} {d2h:>+16.6e} {typ:>8}')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 5: Sensitivity ratio — how much more sensitive is off-line
    #          vs along-line?
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 5: OFF-LINE vs ALONG-LINE SENSITIVITY')
    print(f'{"="*76}\n')

    # For each zero: |d^2h/d(delta^2)| vs |d^2h/d(gamma^2)|
    # If the off-line curvature is much larger, the line is a sharp ridge.
    print(f'  {"gamma":>10} {"|d2h/ddelta2|":>16} {"|d2h/dgamma2|":>16} {"ratio":>10}')
    print('  ' + '-' * 56)

    for gamma in zeros[:15]:
        eps = 1e-4
        def h_pair_d(d):
            re = L**2/4 + d**2 - gamma**2
            im = 2 * d * gamma
            return 2 * L * re / (re**2 + im**2)

        def h_pair_g(g):
            re = L**2/4 - g**2
            return 2 * L * re / (re**2) if abs(re) > 1e-15 else 0

        d2h_delta = (h_pair_d(eps) - 2*h_pair_d(0) + h_pair_d(-eps)) / eps**2
        d2h_gamma = (h_pair_g(gamma+eps) - 2*h_pair_g(gamma) + h_pair_g(gamma-eps)) / eps**2

        ratio = abs(d2h_delta / d2h_gamma) if abs(d2h_gamma) > 1e-20 else float('inf')
        print(f'  {gamma:>10.4f} {abs(d2h_delta):>16.6e} {abs(d2h_gamma):>16.6e} {ratio:>10.2f}')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 6: The functional equation constraint
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 6: THE FUNCTIONAL EQUATION AS A LINE CONSTRAINT')
    print(f'{"="*76}\n')

    # The functional equation xi(s) = xi(1-s) forces:
    # If rho is a zero, so is 1-rho_bar.
    # On the critical line: rho = 1/2 + i*gamma, 1-rho_bar = 1/2 + i*gamma (same!)
    # Off the line: rho = sigma + i*gamma, partner = (1-sigma) + i*gamma
    #
    # Key property: h_pair(delta, gamma) = h(delta+i*gamma) + h(-delta+i*gamma)
    # This is EVEN in delta! So dh/d(delta) = 0 at delta = 0 always.
    #
    # This means: the functional equation forces the critical line to be
    # a STATIONARY point of the test function. Moving off the line has
    # only SECOND-ORDER effect on h_pair.
    #
    # But the SIGN of the second-order effect matters:
    # If d^2h/d(delta^2) < 0 for all gamma (maximum), then
    # h_pair is MAXIMIZED on the critical line.
    # If > 0 (minimum), h_pair is minimized.

    print(f'  The functional equation forces h_pair to be EVEN in delta.')
    print(f'  Therefore dh/d(delta) = 0 at delta=0 for ALL gamma.')
    print(f'  The critical line is automatically a stationary point.')
    print()
    print(f'  This is a GEOMETRIC fact: the critical line is the')
    print(f'  fixed locus of s -> 1-s_bar, and any function respecting')
    print(f'  this symmetry has zero first derivative at the fixed locus.')
    print()

    # Check: is sigma=1/2 a MAXIMUM or MINIMUM of h_pair?
    n_max = 0
    n_min = 0
    for gamma in zeros:
        eps = 1e-4
        def h_d(d):
            re = L**2/4 + d**2 - gamma**2
            im = 2*d*gamma
            return 2*L*re / (re**2 + im**2)
        d2 = (h_d(eps) - 2*h_d(0) + h_d(-eps)) / eps**2
        if d2 < 0:
            n_max += 1
        else:
            n_min += 1

    print(f'  Of {len(zeros)} zeros: sigma=1/2 is a MAXIMUM for {n_max}, '
          f'MINIMUM for {n_min}')
    print(f'  (For MAX: moving off-line DECREASES h_pair)')
    print(f'  (For MIN: moving off-line INCREASES h_pair)')
    sys.stdout.flush()

    # ======================================================================
    # VERDICT
    # ======================================================================
    print()
    print('=' * 76)
    print('  SESSION 78 VERDICT')
    print('=' * 76)
    print()
    print('  The functional equation forces dh/d(sigma) = 0 at sigma = 1/2.')
    print('  The critical line is AUTOMATICALLY a stationary point of the')
    print('  zero contribution to the explicit formula.')
    print()
    print('  This is not a coincidence — it is the GEOMETRY of the functional')
    print('  equation. The involution s -> 1-s_bar has the critical line as')
    print('  its fixed locus, and any even function of delta = sigma - 1/2')
    print('  has zero derivative at the fixed locus.')
    print()
    print('  The question is: is it a MAX or MIN? And does the second-order')
    print('  effect on M_odd\'s eigenvalue have a definite sign?')
    print()


if __name__ == '__main__':
    run()
