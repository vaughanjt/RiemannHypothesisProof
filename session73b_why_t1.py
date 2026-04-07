"""
SESSION 73b -- WHY IS THE MINIMUM AT EXACTLY t=1.0?

Session 73 found: eig_2(M(t)) has a parabolic minimum at EXACTLY t=1.0,
universal across all lambda. The 2nd eigenvalue touches zero and bounces
back positive on both sides.

By eigenvalue perturbation theory (Hellmann-Feynman):
  d/dt eig_2(t) = v_2(t)^T * D * v_2(t)

At t=1: this derivative must be ZERO (since it's a minimum).
So: v_2(1)^T * D * v_2(1) = 0.

The 2nd eigenvector of M is ORTHOGONAL to D in the Rayleigh quotient sense.
WHY? Is this an identity or a coincidence?

Plan:
  A. Perturbation theory at t=1: verify v_2^T D v_2 = 0
  B. Decompose D to understand which part forces the orthogonality
  C. Track the curvature (2nd derivative) and its lambda scaling
  D. The avalanche mechanism: why do 66 eigenvalues cross together?
  E. Ultra-fine structure of the transition
  F. Test: is v_2^T D v_2 = 0 an IDENTITY or approximate?
"""

import sys
import numpy as np

sys.path.insert(0, '.')
from session41g_uncapped_barrier import sieve_primes
from session49c_weil_residual import (
    build_all_fast, _compute_alpha, _compute_wr_diag
)


def extract_cauchy_and_diagonal(lam_sq):
    """Extract L_pure (Cauchy) and D (diagonal perturbation) from M."""
    L = float(np.log(lam_sq))
    N = max(15, round(6 * L))
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)

    wr = _compute_wr_diag(L, N)
    alpha = _compute_alpha(L, N)

    primes = sieve_primes(int(lam_sq))
    a_prime = np.zeros(dim)
    B_prime = np.zeros(dim)

    for p in primes:
        pk = int(p)
        logp = np.log(p)
        while pk <= lam_sq:
            w = logp * pk ** (-0.5)
            y = np.log(pk)
            a_prime += w * 2 * np.cos(2 * np.pi * ns * y / L)
            B_prime += w * np.sin(2 * np.pi * ns * y / L) / np.pi
            pk *= int(p)

    a_n = np.array([wr[abs(int(n))] for n in ns]) + a_prime
    B_n = alpha + B_prime

    nm = ns[:, None] - ns[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        L_pure = (B_n[None, :] - B_n[:, None]) / nm

    for i in range(dim):
        if 0 < i < dim - 1:
            L_pure[i, i] = (B_n[i + 1] - B_n[i - 1]) / 2
        elif i == 0:
            L_pure[i, i] = B_n[1] - B_n[0]
        else:
            L_pure[i, i] = B_n[-1] - B_n[-2]
    L_pure = (L_pure + L_pure.T) / 2

    _, M, _ = build_all_fast(lam_sq, N)
    D_diag = np.diag(M) - np.diag(L_pure)

    # Also decompose D into archimedean and prime parts
    wr_diag_full = np.array([wr[abs(int(n))] for n in ns])
    L_pure_diag = np.diag(L_pure)

    # D = (a_n - L_pure_diag) = (wr + a_prime - L_pure_diag)
    D_arch = wr_diag_full - L_pure_diag  # archimedean part of D
    # Actually the alpha contribution is in L_pure_diag via B_n.
    # The cleaner split: D = M_diag - L_pure_diag
    # M_diag = wr + a_prime (all diagonal terms)
    # L_pure_diag = finite-diff derivative of B_n

    return L_pure, D_diag, M, N, L, dim, ns, wr_diag_full, a_prime, L_pure


def run():
    print()
    print('#' * 76)
    print('  SESSION 73b -- WHY IS THE MINIMUM AT t=1.0?')
    print('#' * 76)

    # ==================================================================
    # PART A: Perturbation theory verification
    # ==================================================================
    print(f'\n  === PART A: HELLMANN-FEYNMAN AT t=1 ===\n')

    print(f'  d/dt eig_k(t) = v_k^T * D * v_k  (Hellmann-Feynman)')
    print(f'  At a minimum: this must be ZERO for the 2nd eigenvalue.')
    print()

    print(f'  {"lam^2":>8} {"v2^T D v2":>14} {"v1^T D v1":>14} '
          f'{"eig_2":>12} {"eig_1":>12}')
    print('  ' + '-' * 66)

    for lam_sq in [50, 100, 200, 500, 1000, 2000, 5000, 10000]:
        try:
            Lp, Dd, Mt, Nt, Lt, dt, ns, wr, ap, Lpm = \
                extract_cauchy_and_diagonal(lam_sq)

            evals, evecs = np.linalg.eigh(Mt)
            v2 = evecs[:, -2]  # 2nd eigenvector
            v1 = evecs[:, -1]  # 1st eigenvector

            # Hellmann-Feynman: d/dt eig_k = v_k^T * diag(D) * v_k = Sum D_i * v_k[i]^2
            hf_2 = np.sum(Dd * v2**2)
            hf_1 = np.sum(Dd * v1**2)

            print(f'  {lam_sq:>8d} {hf_2:>+14.6e} {hf_1:>+14.6e} '
                  f'{evals[-2]:>+12.4e} {evals[-1]:>+12.4f}')
        except Exception as e:
            print(f'  {lam_sq:>8d} ERROR: {e}')
    sys.stdout.flush()

    # ==================================================================
    # PART B: Decomposing the Hellmann-Feynman derivative
    # ==================================================================
    print(f'\n  === PART B: WHY IS v2^T D v2 = 0? ===\n')

    lam_sq = 1000
    Lp, Dd, Mt, Nt, Lt, dt, ns, wr, ap, Lpm = \
        extract_cauchy_and_diagonal(lam_sq)

    evals, evecs = np.linalg.eigh(Mt)
    v2 = evecs[:, -2]

    # Decompose D into positive and negative parts
    D_pos = np.maximum(Dd, 0)
    D_neg = np.minimum(Dd, 0)

    hf_pos = np.sum(D_pos * v2**2)
    hf_neg = np.sum(D_neg * v2**2)
    hf_total = hf_pos + hf_neg

    print(f'  lam^2 = {lam_sq}:')
    print(f'  v2^T D_pos v2 = {hf_pos:+.10e}  (positive D entries)')
    print(f'  v2^T D_neg v2 = {hf_neg:+.10e}  (negative D entries)')
    print(f'  v2^T D     v2 = {hf_total:+.10e}  (total = 0?)')
    print(f'  Cancellation ratio: {abs(hf_pos)/abs(hf_total):.1f}:1')

    # Which indices contribute most?
    contributions = Dd * v2**2
    top_contrib = np.argsort(np.abs(contributions))[::-1]

    print(f'\n  Top 15 contributions to v2^T D v2:')
    print(f'  {"n":>5} {"D[n]":>12} {"v2[n]^2":>12} {"D*v2^2":>14} {"cumsum":>14}')
    print('  ' + '-' * 60)
    cumsum = 0
    for rank, idx in enumerate(top_contrib[:15]):
        n = int(ns[idx])
        cumsum += contributions[idx]
        print(f'  {n:>5d} {Dd[idx]:>+12.4f} {v2[idx]**2:>12.6e} '
              f'{contributions[idx]:>+14.6e} {cumsum:>+14.6e}')
    sys.stdout.flush()

    # ==================================================================
    # PART C: Curvature (2nd derivative) and lambda scaling
    # ==================================================================
    print(f'\n  === PART C: CURVATURE OF eig_2(t) AT t=1 ===\n')

    print(f'  2nd derivative via finite differences:')
    print(f'  f"(1) = [eig_2(1+h) + eig_2(1-h) - 2*eig_2(1)] / h^2')
    print()

    print(f'  {"lam^2":>8} {"f(1)":>14} {"f\'\'(1)":>14} {"width":>10}')
    print('  ' + '-' * 50)

    for lam_sq in [50, 100, 200, 500, 1000, 2000, 5000]:
        try:
            Lp, Dd, Mt, Nt, Lt, dt, ns, wr, ap, Lpm = \
                extract_cauchy_and_diagonal(lam_sq)

            h = 0.001
            M_plus = Lp + (1 + h) * np.diag(Dd)
            M_minus = Lp + (1 - h) * np.diag(Dd)

            e_plus = np.linalg.eigvalsh(M_plus)[-2]
            e_minus = np.linalg.eigvalsh(M_minus)[-2]
            e_center = np.linalg.eigvalsh(Mt)[-2]

            curvature = (e_plus + e_minus - 2 * e_center) / h**2

            # Width of the parabola: delta_t where eig_2 = 0
            # eig_2(t) ~ f''(1)/2 * (t-1)^2, so zero at t=1 (exact),
            # reaches some threshold at delta_t = sqrt(2*threshold/f'')
            # Width to reach eig_2 = 0.01:
            if curvature > 0:
                width = np.sqrt(2 * 0.01 / curvature)
            else:
                width = float('inf')

            print(f'  {lam_sq:>8d} {e_center:>+14.6e} {curvature:>+14.4f} {width:>10.6f}')
        except Exception as e:
            print(f'  {lam_sq:>8d} ERROR: {e}')
    sys.stdout.flush()

    # ==================================================================
    # PART D: The avalanche -- eigenvalue level repulsion
    # ==================================================================
    print(f'\n  === PART D: AVALANCHE MECHANISM ===\n')

    lam_sq = 1000
    Lp, Dd, Mt, Nt, Lt, dt, ns, wr, ap, Lpm = \
        extract_cauchy_and_diagonal(lam_sq)

    # At t=0.99: many positive eigenvalues all near zero
    # Are they clustered? Is there level repulsion?
    t_pre = 0.99
    M_pre = Lp + t_pre * np.diag(Dd)
    evals_pre = np.linalg.eigvalsh(M_pre)

    # The positive eigenvalues near the transition
    pos_evals = evals_pre[evals_pre > 1e-10]
    pos_evals_sorted = np.sort(pos_evals)

    print(f'  At t={t_pre}, {len(pos_evals)} positive eigenvalues:')
    print(f'  Top eigenvalue: {pos_evals_sorted[-1]:.6f}')
    print(f'  Remaining {len(pos_evals)-1} positive eigenvalues:')

    small_pos = pos_evals_sorted[:-1]  # exclude the big one
    if len(small_pos) > 0:
        print(f'    Range: [{small_pos[0]:.6e}, {small_pos[-1]:.6e}]')
        print(f'    Mean:  {small_pos.mean():.6e}')
        print(f'    Std:   {small_pos.std():.6e}')
        print(f'    Max/Min: {small_pos[-1]/small_pos[0]:.2f}')

        # Spacing statistics
        spacings = np.diff(small_pos)
        print(f'    Mean spacing: {spacings.mean():.6e}')
        print(f'    Min spacing: {spacings.min():.6e}')
        print(f'    Max spacing: {spacings.max():.6e}')

        # Do they scale with something?
        print(f'\n  Are the small positive eigenvalues at t=0.99 proportional to (1-t)?')
        for t_test in [0.99, 0.995, 0.999]:
            M_test = Lp + t_test * np.diag(Dd)
            evals_test = np.linalg.eigvalsh(M_test)
            small_test = evals_test[evals_test > 1e-10]
            small_test = np.sort(small_test)[:-1]  # exclude top
            if len(small_test) > 0:
                mean_small = small_test.mean()
                print(f'    t={t_test}: mean small pos eig = {mean_small:.6e}, '
                      f'(1-t) = {1-t_test:.3e}, ratio = {mean_small/(1-t_test):.4f}')
    sys.stdout.flush()

    # ==================================================================
    # PART E: The bounce-back -- what eigenvector comes back?
    # ==================================================================
    print(f'\n  === PART E: BOUNCE-BACK ANALYSIS ===\n')

    # At t<1: v_2 is the 2nd eigenvector (positive eigenvalue)
    # At t=1: v_2 barely crosses zero
    # At t>1: v_2 comes back positive
    # Is it the SAME eigenvector?

    print(f'  Tracking 2nd eigenvector through the bounce:')
    print(f'  {"t":>6} {"eig_2":>14} {"overlap w/ v2(0.99)":>22} {"overlap w/ v2(1.01)":>22}')
    print('  ' + '-' * 68)

    M_pre2 = Lp + 0.99 * np.diag(Dd)
    _, evecs_pre = np.linalg.eigh(M_pre2)
    v2_pre = evecs_pre[:, -2]

    M_post = Lp + 1.01 * np.diag(Dd)
    _, evecs_post = np.linalg.eigh(M_post)
    v2_post = evecs_post[:, -2]

    for t in [0.95, 0.98, 0.99, 0.995, 0.999, 1.0, 1.001, 1.005, 1.01, 1.02, 1.05]:
        M_t = Lp + t * np.diag(Dd)
        evals_t, evecs_t = np.linalg.eigh(M_t)
        v2_t = evecs_t[:, -2]

        overlap_pre = abs(np.dot(v2_t, v2_pre))
        overlap_post = abs(np.dot(v2_t, v2_post))

        print(f'  {t:>6.3f} {evals_t[-2]:>+14.6e} {overlap_pre:>22.6f} {overlap_post:>22.6f}')
    sys.stdout.flush()

    # ==================================================================
    # PART F: Is v2^T D v2 = 0 EXACT or approximate?
    # ==================================================================
    print(f'\n  === PART F: EXACTNESS TEST ===\n')

    print(f'  If v2^T D v2 = 0 is exact, it should hold to machine precision.')
    print(f'  If approximate, it should drift with lambda or precision.')
    print()

    # High-precision test at multiple lambda
    print(f'  {"lam^2":>8} {"v2^T D v2":>18} {"eig_2":>14} {"ratio HF/eig2":>16}')
    print('  ' + '-' * 60)

    for lam_sq in [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]:
        try:
            Lp, Dd, Mt, Nt, Lt, dt, ns, wr, ap, Lpm = \
                extract_cauchy_and_diagonal(lam_sq)

            evals, evecs = np.linalg.eigh(Mt)
            v2 = evecs[:, -2]

            hf2 = np.sum(Dd * v2**2)
            eig2 = evals[-2]

            # The ratio tells us if HF derivative and eigenvalue are correlated
            ratio = hf2 / eig2 if abs(eig2) > 1e-20 else float('inf')

            print(f'  {lam_sq:>8d} {hf2:>+18.10e} {eig2:>+14.6e} {ratio:>16.6f}')
        except Exception as e:
            print(f'  {lam_sq:>8d} ERROR: {e}')
    sys.stdout.flush()

    # ==================================================================
    # PART G: The sum rule -- what constrains v2^T D v2?
    # ==================================================================
    print(f'\n  === PART G: SUM RULE ===\n')

    lam_sq = 1000
    Lp, Dd, Mt, Nt, Lt, dt, ns, wr, ap, Lpm = \
        extract_cauchy_and_diagonal(lam_sq)

    evals, evecs = np.linalg.eigh(Mt)

    # Sum over ALL eigenvectors: Sum_k v_k^T D v_k = tr(D)
    hf_all = np.array([np.sum(Dd * evecs[:, k]**2) for k in range(len(evals))])
    print(f'  Sum_k v_k^T D v_k = {np.sum(hf_all):+.6e}')
    print(f'  tr(D) = {np.sum(Dd):+.6e}')
    print(f'  Match: {abs(np.sum(hf_all) - np.sum(Dd)) < 1e-8}')

    # The HF derivatives for each eigenvector
    print(f'\n  Hellmann-Feynman derivatives for top eigenvectors:')
    print(f'  {"k":>3} {"eig_k":>14} {"v_k^T D v_k":>16} {"cum sum":>14}')
    print('  ' + '-' * 50)
    cumsum = 0
    for k in range(len(evals) - 1, max(len(evals) - 20, -1), -1):
        cumsum += hf_all[k]
        print(f'  {len(evals)-1-k:>3d} {evals[k]:>+14.6e} {hf_all[k]:>+16.8e} {cumsum:>+14.6e}')
    sys.stdout.flush()

    # ==================================================================
    # PART H: Connection to the Weil explicit formula
    # ==================================================================
    print(f'\n  === PART H: WEIL FORMULA CONNECTION ===\n')

    # M = L_pure + D comes from the Weil explicit formula:
    # W_R + W_p = M
    # The split into L_pure and D is our Cauchy decomposition:
    # L_pure = off-diagonal part (Cauchy divided differences of B_n)
    # D = diagonal correction (a_n minus Cauchy diagonal limit)
    #
    # The Weil explicit formula with test function h_lambda says:
    # Sum_rho h(gamma) = integral + sum_p contributions
    #
    # At t=1: exact Weil formula. At t != 1: deformed formula where
    # the diagonal (archimedean regular distribution) is scaled.
    #
    # The condition v2^T D v2 = 0 at t=1 means:
    # the 2nd eigenvector of M sees zero net diagonal contribution.
    # This is a BALANCE condition between the archimedean regular
    # distribution and the Cauchy derivative limit.

    print(f'  At t=1, the 2nd eigenvector v_2 satisfies:')
    print(f'    v_2^T * D * v_2 = 0')
    print(f'  where D = diag(M) - diag(L_pure)')
    print(f'       = (archimedean regular + prime diagonal) - (Cauchy diagonal limit)')
    print()
    print(f'  This means: v_2 sees ZERO net contribution from the')
    print(f'  diagonal mismatch between M and its Cauchy approximation.')
    print()

    # The diagonal mismatch: D_n = a_n - B'(n)
    # where a_n is the actual diagonal of M
    # and B'(n) is the derivative of the Cauchy generating function B
    print(f'  Diagonal mismatch D_n = a_n - B\'(n):')
    N = Nt
    idx_center = N
    print(f'  {"n":>4} {"D_n":>14} {"a_n":>14} {"B\'(n)":>14}')
    print('  ' + '-' * 50)
    for k in range(min(15, N + 1)):
        idx = idx_center + k
        print(f'  {k:>4d} {Dd[idx]:>+14.6f} {np.diag(Mt)[idx]:>+14.6f} '
              f'{np.diag(Lpm)[idx]:>+14.6f}')
    sys.stdout.flush()

    # ==================================================================
    # VERDICT
    # ==================================================================
    print()
    print('=' * 76)
    print('  SESSION 73b VERDICT')
    print('=' * 76)


if __name__ == '__main__':
    run()
