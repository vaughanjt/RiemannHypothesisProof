"""
SESSION 42b — SPECTRAL REPRESENTATION OF THE BARRIER

Compute barrier = sum_rho |h_w(gamma_rho)|^2 using zeta zeros directly.

The Weil explicit formula for the test function associated with w_hat gives:
    <w, QW, w> = sum_rho |H_w(gamma_rho)|^2

where H_w(gamma) = sum_n w_hat[n] * phi_n(gamma)
and phi_n(gamma) is related to the Mellin transform of the test function basis.

For the Connes framework with test functions omega_n(x) = 2(1-x/L)cos(2*pi*n*x/L):
The Mellin transform at s = 1/2 + i*gamma is:
    hat_n(gamma) = int_0^L omega_n(x) * x^{-1/2 - i*gamma} dx

Then the spectral barrier is:
    barrier = sum_rho | sum_n w_hat[n] * hat_n(gamma_rho) |^2

This provides:
1. Independent check of the matrix computation
2. Shows whether the barrier converges to 0 or stays positive
3. Reveals which zeros contribute most to the barrier
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, zetazero, log, pi, cos, exp, quad, mpc
import time
import sys

mp.dps = 30


def get_zeros(n_zeros):
    """Get first n_zeros imaginary parts of zeta zeros."""
    zeros = []
    for k in range(1, n_zeros + 1):
        z = zetazero(k)
        zeros.append(float(z.imag))
    return np.array(zeros)


def mellin_transform_omega_n(n_val, gamma, L_mp):
    """
    Compute hat_n(gamma) = int_0^L 2(1-x/L)cos(2*pi*n*x/L) * x^{-1/2 - i*gamma} dx

    This is the Mellin transform of omega_n(x) at s = 1/2 + i*gamma.
    """
    n_val = int(n_val)
    s = mpf(1)/2 + mpc(0, mpf(gamma))

    def integrand(x):
        omega = 2 * (1 - x/L_mp) * cos(2 * pi * n_val * x / L_mp)
        return omega * x**(-s)

    # Use mpmath quadrature
    result = quad(integrand, [mpf(0), L_mp], error=True)
    return complex(result[0])


def spectral_barrier(lam_sq, n_zeros=100, N=None):
    """
    Compute barrier as sum over zeta zeros.

    barrier = sum_{rho} | H_w(gamma_rho) |^2

    where H_w(gamma) = sum_n w_hat[n] * hat_n(gamma)
    """
    L_f = np.log(lam_sq)
    L_mp = log(mpf(lam_sq))
    if N is None:
        N = max(15, round(6 * L_f))
    dim = 2 * N + 1

    # Build w_hat
    ns = np.arange(-N, N + 1, dtype=float)
    w_vec = ns / (L_f**2 + (4 * np.pi)**2 * ns**2)
    w_vec[N] = 0.0
    w_hat = w_vec / np.linalg.norm(w_vec)

    # Get zeros
    zeros = get_zeros(n_zeros)

    # For each zero, compute H_w(gamma) = sum_n w_hat[n] * hat_n(gamma)
    H_values = []
    contributions = []

    for idx, gamma in enumerate(zeros):
        # Compute hat_n(gamma) for each n, then dot with w_hat
        # Since w_hat is odd and omega_n involves cos, we use symmetry
        # hat_n(gamma) for n and -n are related

        H = 0.0 + 0.0j
        for i in range(dim):
            n_val = int(ns[i])
            if abs(w_hat[i]) < 1e-15:
                continue
            hn = mellin_transform_omega_n(n_val, gamma, L_mp)
            H += w_hat[i] * hn

        H_values.append(H)
        contributions.append(abs(H)**2)

        if (idx + 1) % 20 == 0:
            running_sum = sum(contributions)
            print(f'    {idx+1}/{n_zeros} zeros processed, '
                  f'running barrier = {running_sum:.8f}', flush=True)

    barrier = sum(contributions)

    return {
        'lam_sq': lam_sq,
        'L': L_f,
        'N': N,
        'n_zeros': n_zeros,
        'barrier': barrier,
        'H_values': np.array(H_values),
        'contributions': np.array(contributions),
        'zeros': zeros,
    }


def fast_spectral_barrier(lam_sq, n_zeros=200, N=None):
    """
    Faster version using numpy integration instead of mpmath quad.
    Less precise but much faster.
    """
    L_f = np.log(lam_sq)
    if N is None:
        N = max(15, round(6 * L_f))
    dim = 2 * N + 1

    ns = np.arange(-N, N + 1, dtype=float)
    w_vec = ns / (L_f**2 + (4 * np.pi)**2 * ns**2)
    w_vec[N] = 0.0
    w_hat = w_vec / np.linalg.norm(w_vec)

    zeros = get_zeros(n_zeros)

    # Precompute Mellin transforms using numpy quadrature
    n_quad = 2000
    x_pts = np.linspace(1e-10, L_f, n_quad)
    dx = x_pts[1] - x_pts[0]

    contributions = np.zeros(n_zeros)
    H_values = np.zeros(n_zeros, dtype=complex)

    for z_idx, gamma in enumerate(zeros):
        # x^{-1/2 - i*gamma} = x^{-1/2} * exp(-i*gamma*log(x))
        log_x = np.log(x_pts)
        x_factor = x_pts**(-0.5) * np.exp(-1j * gamma * log_x)

        H = 0.0 + 0.0j
        for i in range(dim):
            n_val = ns[i]
            if abs(w_hat[i]) < 1e-15:
                continue
            omega = 2 * (1 - x_pts / L_f) * np.cos(2 * np.pi * n_val * x_pts / L_f)
            integrand = omega * x_factor
            hn = np.sum(integrand) * dx
            H += w_hat[i] * hn

        H_values[z_idx] = H
        contributions[z_idx] = abs(H)**2

    barrier = np.sum(contributions)

    return {
        'lam_sq': lam_sq,
        'L': L_f,
        'n_zeros': n_zeros,
        'barrier': barrier,
        'contributions': contributions,
        'zeros': zeros,
    }


if __name__ == '__main__':
    print()
    print('#' * 70)
    print('  SESSION 42b -- SPECTRAL BARRIER (SUM OVER ZETA ZEROS)')
    print('#' * 70)

    # Load zeros once
    print('\n  Loading zeta zeros...', flush=True)
    t0 = time.time()
    all_zeros = get_zeros(200)
    print(f'  Loaded 200 zeros in {time.time()-t0:.1f}s')
    print(f'  Range: [{all_zeros[0]:.4f}, {all_zeros[-1]:.4f}]')

    # Part 1: Fast spectral barrier
    print('\n\n  PART 1: Fast spectral barrier (numpy quadrature)')
    print('  ' + '=' * 60)

    for lam_sq in [50, 200, 1000]:
        t0 = time.time()
        r = fast_spectral_barrier(lam_sq, n_zeros=200)
        dt = time.time() - t0

        # Top contributing zeros
        top_idx = np.argsort(r['contributions'])[::-1][:5]

        print(f'\n  lam^2={lam_sq}, L={r["L"]:.3f}')
        print(f'    Spectral barrier = {r["barrier"]:.8f}  ({dt:.1f}s)')
        print(f'    Top 5 contributing zeros:')
        for idx in top_idx:
            print(f'      gamma_{idx+1} = {r["zeros"][idx]:.4f}  '
                  f'|H|^2 = {r["contributions"][idx]:.6f}  '
                  f'({r["contributions"][idx]/r["barrier"]*100:.1f}%)')

        # Cumulative convergence
        cum = np.cumsum(r['contributions'])
        for n in [10, 50, 100, 200]:
            if n <= len(cum):
                print(f'    First {n:>3d} zeros: barrier = {cum[n-1]:.6f}  '
                      f'({cum[n-1]/r["barrier"]*100:.1f}%)')

    # Part 2: Comparison with matrix barrier
    print('\n\n  PART 2: Spectral vs matrix barrier comparison')
    print('  ' + '=' * 60)

    sys.path.insert(0, '.')
    from session41g_uncapped_barrier import compute_barrier_partial

    for lam_sq in [50, 200, 1000]:
        r_spec = fast_spectral_barrier(lam_sq, n_zeros=200)
        r_mat = compute_barrier_partial(lam_sq)

        print(f'  lam^2={lam_sq}:')
        print(f'    Matrix W02-Mp = {r_mat["partial_barrier"]:+.6f}')
        print(f'    Spectral (200 zeros) = {r_spec["barrier"]:.6f}')
        print(f'    Note: spectral = full barrier, matrix = partial (no M_diag/M_alpha)')

    # Part 3: Does H_w(gamma) -> 0 for large gamma?
    print('\n\n  PART 3: Decay of H_w(gamma) with zero height')
    print('  ' + '=' * 60)

    for lam_sq in [200, 1000]:
        r = fast_spectral_barrier(lam_sq, n_zeros=200)
        amps = np.sqrt(r['contributions'])  # |H_w(gamma)|

        # Fit decay: |H| ~ gamma^{-alpha}
        gammas = r['zeros']
        log_g = np.log(gammas[10:])  # skip first few
        log_a = np.log(amps[10:] + 1e-20)
        c = np.polyfit(log_g, log_a, 1)

        print(f'\n  lam^2={lam_sq}:')
        print(f'    |H_w(gamma)| ~ gamma^{{{c[0]:.3f}}}')
        print(f'    First 5: {amps[:5]}')
        print(f'    Last 5:  {amps[-5:]}')

    print('\n' + '#' * 70)
    print('  SESSION 42b COMPLETE')
    print('#' * 70)
