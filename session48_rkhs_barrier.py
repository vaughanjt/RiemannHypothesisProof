"""
SESSION 48 -- REPRODUCING KERNEL HILBERT SPACE FOR THE BARRIER

The barrier B(L) = Sum_rho |H_w(rho)|? - C(L) from Session 43.

Key insight: if we define a kernel
    K(L_1, L_2) = Sum_rho H_w(rho, L_1) ? conj(H_w(rho, L_2))

then K(L,L) = Sum |H_w(rho)|? >= 0 (sum of squares = PD kernel on diagonal).
This is the spectral part. The question:

1. Is K(L_1, L_2) actually positive definite as a kernel? (Check Gram matrix)
2. What is the correction C(L) structure? Can it be absorbed?
3. Does B(L) = K(L,L) - C(L) have an RKHS interpretation?

Three approaches:
A. Direct: compute K(L_1, L_2) Gram matrix, check eigenvalues
B. Correction absorption: find a modified kernel K such that K(L,L) = B(L)
C. Feature map: write f_L(rho) = H_w(rho, L), check completeness

If the Gram matrix of K has eigenvalues that dominate C, we have a proof path.
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, log, pi, exp, sin, cos, quad,
                    zetazero, power, sqrt, fabs, im, re, conj, nstr)
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

mp.dps = 25


# ===============================================================
# BUILDING BLOCKS FROM SESSION 42h
# ===============================================================

def build_w_hat_mp(lam_sq, N=None):
    """Build normalized Lorentzian test vector w_hat in mpmath precision."""
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


def mellin_transform_gw(w_hat_positive, L, gamma_val):
    """
    Compute G_w(1/2 + i*gamma) = integral_0^L g_w(x) * x^{-1/2+i*gamma} dx

    where g_w(x) = 2 * sum_{n=1}^{N} w_hat[n] * sin(2*pi*n*x/L)
    """
    s = mpf(1)/2 + mpc(0, mpf(gamma_val))
    L_s = power(L, s)

    G = mpc(0, 0)
    N = len(w_hat_positive)

    for n_idx in range(N):
        n = n_idx + 1
        wn = w_hat_positive[n_idx]
        if fabs(wn) < mpf(10)**(-20):
            continue

        freq = 2 * pi * n

        def integrand_real(u, _n=n, _freq=freq):
            return sin(_freq * u) * power(u, re(s) - 1) * cos(im(s) * log(u))

        def integrand_imag(u, _n=n, _freq=freq):
            return sin(_freq * u) * power(u, re(s) - 1) * sin(im(s) * log(u))

        I_real = quad(integrand_real, [mpf(0), mpf(1)], maxdegree=6)
        I_imag = quad(integrand_imag, [mpf(0), mpf(1)], maxdegree=6)

        I_n = L_s * mpc(I_real, I_imag)
        G += 2 * wn * I_n

    return G


def matrix_barrier(lam_sq, N=None):
    """Compute barrier from Connes matrix for comparison."""
    from connes_crossterm import build_all
    L_f = np.log(lam_sq)
    if N is None:
        N = max(15, round(6 * L_f))

    W02, M, QW = build_all(lam_sq, N)
    ns = np.arange(-N, N + 1, dtype=float)
    w = ns / (L_f**2 + (4*np.pi)**2 * ns**2)
    w[N] = 0.0
    w_hat = w / np.linalg.norm(w)
    return float(w_hat @ QW @ w_hat)


# ===============================================================
# RKHS KERNEL CONSTRUCTION
# ===============================================================

def compute_feature_vector(lam_sq, zeros_imag, n_zeros=None):
    """
    Compute the feature vector f_L = (H_w(rho_1, L), H_w(rho_2, L), ..., H_w(rho_K, L))
    for a given lambda^2.

    In RKHS language, f_L : rho ? H_w(rho, L) is the feature map.
    K(L_1, L_2) = <f_{L_1}, f_{L_2}> = Sum_rho H_w(rho, L_1) ? conj(H_w(rho, L_2))
    """
    if n_zeros is None:
        n_zeros = len(zeros_imag)

    w_hat_mp, L_mp, N = build_w_hat_mp(lam_sq)
    w_hat_pos = [w_hat_mp[N + n] for n in range(1, N + 1)]

    features = []
    for k in range(n_zeros):
        Gw = mellin_transform_gw(w_hat_pos, L_mp, zeros_imag[k])
        features.append(complex(Gw))

    return np.array(features)


def compute_gram_matrix(lam_sq_values, zeros_imag, n_zeros=30):
    """
    Compute the Gram matrix of the kernel K(L_i, L_j) = <f_{L_i}, f_{L_j}>.

    If this matrix is positive definite (all eigenvalues > 0),
    then K is a valid RKHS kernel.
    """
    M = len(lam_sq_values)
    features = []

    print(f'  Computing feature vectors for {M} lambda^2 values, {n_zeros} zeros...')
    for i, lam_sq in enumerate(lam_sq_values):
        t0 = time.time()
        fv = compute_feature_vector(lam_sq, zeros_imag[:n_zeros])
        dt = time.time() - t0
        features.append(fv)
        print(f'    lam^2={lam_sq:8.1f}  ||f||^2={np.sum(np.abs(fv)**2):.6f}  ({dt:.1f}s)')

    # Gram matrix: G[i,j] = <f_i, f_j> = sum_k f_i[k] * conj(f_j[k])
    F = np.array(features)  # shape (M, n_zeros)
    G = F @ F.conj().T      # shape (M, M)

    return G, features


def analyze_kernel(G, lam_sq_values, barriers):
    """
    Analyze the kernel Gram matrix:
    1. Is it positive definite? (eigenvalues)
    2. How does K(L,L) compare to B(L)?
    3. What is the correction C(L) = K(L,L) - B(L)?
    4. Can C be expressed as a rank-deficient perturbation?
    """
    eigenvalues = np.linalg.eigvalsh(G)
    K_diag = np.real(np.diag(G))  # K(L_i, L_i)

    print('\n  -- KERNEL ANALYSIS --')
    print(f'\n  Gram matrix: {G.shape[0]}x{G.shape[0]}')
    print(f'  Eigenvalues: min={eigenvalues[0]:.6e}, max={eigenvalues[-1]:.6e}')
    print(f'  Positive definite: {eigenvalues[0] > 0}')
    print(f'  Condition number: {eigenvalues[-1]/max(eigenvalues[0], 1e-30):.2e}')

    corrections = K_diag - np.array(barriers)

    print(f'\n  {"lam^2":>10}  {"K(L,L)":>12}  {"B(L)":>12}  {"C(L)":>12}  {"C/K ratio":>10}')
    print('  ' + '-' * 62)
    for i in range(len(lam_sq_values)):
        ratio = corrections[i] / K_diag[i] if K_diag[i] != 0 else float('inf')
        print(f'  {lam_sq_values[i]:10.1f}  {K_diag[i]:12.6f}  {barriers[i]:12.6f}  '
              f'{corrections[i]:12.6f}  {ratio:10.4f}')

    # Check if correction has a pattern
    print(f'\n  Correction statistics:')
    print(f'    Mean:   {np.mean(corrections):.6e}')
    print(f'    Std:    {np.std(corrections):.6e}')
    print(f'    Min:    {np.min(corrections):.6e}')
    print(f'    Max:    {np.max(corrections):.6e}')

    return eigenvalues, K_diag, corrections


def check_off_diagonal_structure(G, lam_sq_values):
    """
    Check the off-diagonal structure of the Gram matrix.
    A purely diagonal kernel means features are orthogonal across L values.
    Off-diagonal structure reveals correlations in the zero contributions.
    """
    M = G.shape[0]
    G_abs = np.abs(G)
    diag = np.diag(G_abs)

    print('\n  -- OFF-DIAGONAL STRUCTURE --')
    print(f'\n  {"L_i":>10} {"L_j":>10}  {"K(i,j)":>14}  {"|K(i,j)|/sqrt(K_ii*K_jj)":>25}')
    print('  ' + '-' * 65)

    for i in range(M):
        for j in range(i + 1, M):
            kij = G[i, j]
            normalized = abs(kij) / np.sqrt(G_abs[i,i] * G_abs[j,j]) if diag[i] * diag[j] > 0 else 0
            print(f'  {lam_sq_values[i]:10.1f} {lam_sq_values[j]:10.1f}  '
                  f'{complex(kij):14.6f}  {normalized:25.6f}')


# ===============================================================
# CORRECTION ABSORPTION ANALYSIS
# ===============================================================

def analyze_correction_absorption(K_diag, barriers, eigenvalues, G):
    """
    Can the correction C(L) = K(L,L) - B(L) be absorbed?

    Strategy: if C(L) = <f_L, R f_L> for some bounded operator R with ||R|| < 1,
    then B(L) = <f_L, (I - R) f_L> and (I - R) is positive definite.

    Equivalently: if C can be written as a rank-r perturbation to the Gram matrix,
    and all eigenvalues of G - diag(C) are positive, then we have an RKHS for B.
    """
    print('\n  -- CORRECTION ABSORPTION --')

    corrections = K_diag - np.array(barriers)

    # Modified Gram matrix: subtract correction from diagonal
    G_modified = G.copy()
    for i in range(G.shape[0]):
        G_modified[i, i] -= corrections[i]

    eigs_modified = np.linalg.eigvalsh(G_modified)

    print(f'\n  Modified Gram matrix G_B where G_B[i,i] = B(L_i):')
    print(f'    Eigenvalues: min={eigs_modified[0]:.6e}, max={eigs_modified[-1]:.6e}')
    print(f'    Positive definite: {eigs_modified[0] > 0}')

    if eigs_modified[0] > 0:
        print(f'\n  *** B(L) IS a valid RKHS kernel! ***')
        print(f'  The modified Gram matrix with B(L) on the diagonal is PD.')
        print(f'  This means there exists a Hilbert space H where B(L) = ||phi_L||?.')
    else:
        print(f'\n  B(L) is NOT a valid RKHS kernel with this feature map.')
        print(f'  Need a different feature space or more zeros.')

        # How many eigenvalues are negative?
        n_neg = np.sum(eigs_modified < 0)
        print(f'  Negative eigenvalues: {n_neg}/{len(eigs_modified)}')
        print(f'  Smallest: {eigs_modified[0]:.6e}')

        # How many more zeros would we need?
        # Estimate: spectral sum grows ~ log(K), correction grows ~ ???
        print(f'\n  Spectral gap: {-eigs_modified[0]:.6e} (need this much more positivity)')

    return eigs_modified


# ===============================================================
# MAIN
# ===============================================================

if __name__ == '__main__':
    print()
    print('#' * 72)
    print('  SESSION 48 -- RKHS KERNEL FOR THE CONNES BARRIER')
    print('#' * 72)

    # Load zeros
    print('\n  Loading zeta zeros...', flush=True)
    t0 = time.time()
    n_zeros = 30
    zeros_imag = []
    for k in range(1, n_zeros + 1):
        z = zetazero(k)
        zeros_imag.append(float(z.imag))
    zeros_imag = np.array(zeros_imag)
    print(f'  Loaded {n_zeros} zeros in {time.time()-t0:.1f}s')

    # Test lambda^2 values (covering the key range)
    lam_sq_values = [20, 50, 100, 200, 500]

    # Compute barriers for comparison
    print('\n  Computing matrix barriers...')
    barriers = []
    for lam_sq in lam_sq_values:
        b = matrix_barrier(lam_sq)
        barriers.append(b)
        print(f'    lam^2={lam_sq:8.1f}  B(L)={b:.8f}')

    # -- A. Compute the RKHS Gram matrix --
    print('\n\n' + '=' * 70)
    print('  A. RKHS GRAM MATRIX K(L_i, L_j)')
    print('=' * 70)

    G, features = compute_gram_matrix(lam_sq_values, zeros_imag, n_zeros=n_zeros)

    # -- B. Analyze the kernel --
    eigenvalues, K_diag, corrections = analyze_kernel(G, lam_sq_values, barriers)

    # -- C. Off-diagonal structure --
    check_off_diagonal_structure(G, lam_sq_values)

    # -- D. Can the correction be absorbed? --
    eigs_modified = analyze_correction_absorption(K_diag, barriers, eigenvalues, G)

    # -- Summary --
    print('\n\n' + '=' * 70)
    print('  SUMMARY')
    print('=' * 70)

    print(f'\n  Spectral kernel K(L,L) = Sum|H_w(rho)|? :')
    print(f'    Always positive: {all(k > 0 for k in K_diag)}')
    print(f'    Gram matrix PD:  {eigenvalues[0] > 0}')

    print(f'\n  Barrier B(L) = K(L,L) - C(L) :')
    print(f'    Always positive: {all(b > 0 for b in barriers)}')
    print(f'    Modified Gram PD: {eigs_modified[0] > 0}')

    if eigs_modified[0] > 0:
        print(f'\n  *** RKHS INTERPRETATION WORKS ***')
        print(f'  B(L) = ||phi_L||? in a modified Hilbert space.')
        print(f'  Positivity is AUTOMATIC from the inner product structure.')
    else:
        print(f'\n  RKHS interpretation does not work with {n_zeros} zeros.')
        print(f'  Gap: {-eigs_modified[0]:.6e}')
        print(f'  May need more zeros or a different feature space.')

    print()
