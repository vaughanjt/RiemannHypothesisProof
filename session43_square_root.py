"""
SESSION 43 — THE SQUARE ROOT OF THE WEIL FORM

Construct matrix A such that A^T * A = Q_W (or as close as possible).

A[n, rho] should be the "corrected Mellin transform" of basis function e_n
evaluated at zero rho, with Gamma/conductor corrections absorbed.

If A^T * A = Q_W exactly, then Q_W >= 0 is trivial (sum of squares).

The explicit formula gives:
  Q_W[n,m] = sum_rho G_n(rho) * conj(G_m(rho)) + CORRECTIONS[n,m]

The corrections come from:
  - Gamma function (the "archimedean" place)
  - Conductor (log(lambda^2 / 2pi))
  - Trivial zeros (at s = -2, -4, ...)

Strategy:
  1. Compute G_n(rho) at first K zeros -> matrix A_spectral
  2. Compute A_spectral^T * A_spectral -> Q_spectral
  3. Compute corrections = Q_W - Q_spectral
  4. Factor the corrections and absorb into A
  5. Verify A_corrected^T * A_corrected = Q_W
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, log, pi, exp, sin, cos, quad,
                    zetazero, loggamma, sqrt, re, im, conj, power, fabs)
import time
import sys
import os

mp.dps = 20

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from connes_crossterm import build_all


def mellin_basis(n_val, rho_imag, L):
    """
    Compute G_n(rho) = integral_0^L omega_n(x) * x^{rho-1} dx
    where omega_n(x) = 2(1-x/L)cos(2*pi*n*x/L)
    and rho = 1/2 + i*gamma.

    Uses mpmath quad for the oscillatory integral.
    """
    n_val = int(n_val)
    s = mpf(1)/2 + mpc(0, mpf(rho_imag))

    def integrand(x):
        omega = 2 * (1 - x/L) * cos(2 * pi * n_val * x / L)
        return omega * power(x, s - 1)

    result = quad(integrand, [mpf(0), L], maxdegree=6)
    return complex(result)


def build_spectral_matrix(lam_sq, K_zeros, N_basis=None):
    """
    Build A_spectral[n, k] = G_n(rho_k) for n = -N..N, k = 1..K.

    This matrix satisfies:
    A^H * A ~ Q_W + corrections (approximately)
    """
    L = log(mpf(lam_sq))
    L_f = float(L)
    if N_basis is None:
        N_basis = max(15, round(6 * L_f))
    dim = 2 * N_basis + 1

    # Get zeros
    zeros = [float(zetazero(k).imag) for k in range(1, K_zeros + 1)]

    # Build A: dim x K_zeros (complex matrix)
    A = np.zeros((dim, K_zeros), dtype=complex)

    ns = np.arange(-N_basis, N_basis + 1)

    for k_idx, gamma in enumerate(zeros):
        for n_idx, n_val in enumerate(ns):
            A[n_idx, k_idx] = mellin_basis(n_val, gamma, L)

        if (k_idx + 1) % 5 == 0:
            print(f'    {k_idx+1}/{K_zeros} zeros processed', flush=True)

    return A, zeros, ns


if __name__ == '__main__':
    print()
    print('#' * 72)
    print('  THE SQUARE ROOT OF THE WEIL FORM')
    print('#' * 72)

    lam_sq = 50  # small for speed
    L = np.log(lam_sq)
    N = max(15, round(6 * L))
    dim = 2 * N + 1
    K = 30  # number of zeros to use

    print(f'\n  lam^2 = {lam_sq}, N = {N}, dim = {dim}, K = {K} zeros')

    # Step 1: Build Q_W (the target)
    print('\n  Step 1: Computing Q_W...', flush=True)
    t0 = time.time()
    W02, M, QW = build_all(lam_sq, N, n_quad=4000)
    print(f'  Done in {time.time()-t0:.0f}s')

    evals_qw = np.linalg.eigvalsh(QW)
    print(f'  Q_W eigenvalues: [{evals_qw[0]:.2e}, {evals_qw[-1]:.4f}]')
    print(f'  All positive? {evals_qw[0] > 0}')

    # Step 2: Build A_spectral
    print(f'\n  Step 2: Building A_spectral ({dim} x {K})...', flush=True)
    t0 = time.time()
    A, zeros, ns = build_spectral_matrix(lam_sq, K, N)
    dt = time.time() - t0
    print(f'  Done in {dt:.0f}s')

    # Step 3: Compute Q_spectral = Re(A^H * A)
    print('\n  Step 3: Q_spectral = Re(A^H * A)')
    Q_spectral = np.real(A.conj().T @ A).T  # Wait, need A * A^H for row-space
    # Actually: Q_W ~ sum_rho |G_n(rho)|^2 stuff
    # (A * A^H)[n,m] = sum_k A[n,k] * conj(A[m,k]) = sum_k G_n(rho_k) * conj(G_m(rho_k))
    Q_spectral = np.real(A @ A.conj().T)

    print(f'  Q_spectral eigenvalues: [{np.linalg.eigvalsh(Q_spectral)[0]:.2e}, '
          f'{np.linalg.eigvalsh(Q_spectral)[-1]:.4f}]')

    # Step 4: The correction
    print('\n  Step 4: Corrections = Q_W - Q_spectral')
    corrections = QW - Q_spectral
    corr_norm = np.linalg.norm(corrections, 'fro')
    qw_norm = np.linalg.norm(QW, 'fro')
    print(f'  ||corrections|| / ||Q_W|| = {corr_norm/qw_norm:.6f}')
    print(f'  ||corrections|| = {corr_norm:.6f}')
    print(f'  ||Q_W|| = {qw_norm:.6f}')
    print(f'  ||Q_spectral|| = {np.linalg.norm(Q_spectral, "fro"):.6f}')

    evals_corr = np.linalg.eigvalsh(corrections)
    print(f'  Correction eigenvalues: [{evals_corr[0]:.4f}, {evals_corr[-1]:.4f}]')
    print(f'  Corrections positive? {evals_corr[0] > -1e-10}')
    print(f'  Corrections negative? {evals_corr[-1] < 1e-10}')

    # Step 5: Can we absorb the corrections?
    print('\n  Step 5: Absorbing corrections')

    # If corrections = -B^T * B (negative semidefinite), then:
    # Q_W = A*A^H - B^T*B -- can't help, indefinite
    # If corrections = +C^T * C (positive semidefinite), then:
    # Q_W = A*A^H + C^T*C >= 0 -- even better!
    # If corrections are indefinite: need a different approach.

    if evals_corr[0] > -1e-6:
        print(f'  Corrections are POSITIVE semidefinite!')
        print(f'  Q_W = Q_spectral + corrections')
        print(f'       = (A*A^H) + (positive stuff)')
        print(f'       >= A*A^H >= 0')
        print(f'  THIS WOULD PROVE Q_W >= 0!')
    elif evals_corr[-1] < 1e-6:
        print(f'  Corrections are NEGATIVE semidefinite.')
        print(f'  Q_W = Q_spectral + (negative stuff)')
        print(f'  Need Q_spectral to dominate the negative corrections.')
    else:
        print(f'  Corrections are INDEFINITE.')
        print(f'  Need to restructure the factorization.')

    # Step 6: Rescale A to minimize correction norm
    print('\n  Step 6: Optimal scaling')

    # Try: A_scaled = alpha * A, find alpha that minimizes ||QW - alpha^2 * Q_spectral||
    # d/d(alpha^2) ||QW - alpha^2 * Q_s|| = 0
    # => alpha^2 = tr(QW * Q_s) / tr(Q_s * Q_s)
    alpha_sq = np.trace(QW @ Q_spectral) / np.trace(Q_spectral @ Q_spectral)
    alpha = np.sqrt(max(alpha_sq, 0))

    Q_scaled = alpha**2 * Q_spectral
    corr_scaled = QW - Q_scaled
    evals_scaled = np.linalg.eigvalsh(corr_scaled)

    print(f'  Optimal alpha = {alpha:.6f}')
    print(f'  Scaled correction eigenvalues: [{evals_scaled[0]:.4f}, {evals_scaled[-1]:.4f}]')
    print(f'  ||scaled correction|| / ||Q_W|| = {np.linalg.norm(corr_scaled, "fro")/qw_norm:.6f}')

    # Step 7: The barrier on w direction
    print('\n  Step 7: Check on the w (odd) direction')
    w = ns.astype(float) / (L**2 + (4*np.pi)**2 * ns.astype(float)**2)
    w[N] = 0.0
    w_hat = w / np.linalg.norm(w)

    barrier_qw = float(w_hat @ QW @ w_hat)
    barrier_spectral = float(w_hat @ Q_spectral @ w_hat)
    barrier_correction = float(w_hat @ corrections @ w_hat)

    print(f'  <w, Q_W, w>        = {barrier_qw:.8f}')
    print(f'  <w, Q_spectral, w> = {barrier_spectral:.8f}')
    print(f'  <w, corrections, w> = {barrier_correction:.8f}')
    print(f'  Q_spectral piece = {barrier_spectral/barrier_qw*100:.1f}% of barrier')

    # Step 8: Is the spectral piece alone positive?
    print('\n  Step 8: Is A*A^H alone sufficient?')
    print(f'  Q_spectral >= 0? {np.linalg.eigvalsh(Q_spectral)[0] > -1e-10}  (ALWAYS TRUE - sum of squares!)')
    print(f'  Q_spectral barrier (w) = {barrier_spectral:.8f}  (positive!)')

    if barrier_correction > -1e-10:
        print(f'\n  *** THE CORRECTION IS POSITIVE ON THE w DIRECTION ***')
        print(f'  Q_W = (positive spectral) + (positive correction) > 0')
        print(f'  BARRIER POSITIVITY FOLLOWS WITHOUT RH!')
    elif barrier_spectral > abs(barrier_correction):
        print(f'\n  Spectral piece ({barrier_spectral:.6f}) > |correction| ({abs(barrier_correction):.6f})')
        print(f'  Q_W > 0 on w direction because spectral dominates')
    else:
        print(f'\n  Spectral piece alone insufficient.')
        print(f'  Need more zeros or better correction absorption.')

    print('\n' + '#' * 72)
    print('  SQUARE ROOT HUNT COMPLETE')
    print('#' * 72)
