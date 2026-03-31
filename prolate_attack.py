"""
PROLATE OPERATOR ATTACK — an operator defined WITHOUT zeta.

W_lambda(xi)(x) = -d/dx(lambda^2 - x^2) d/dx xi(x) + (2*pi*lambda)^2 x^2 xi(x)

Sturm-Liouville operator. Self-adjoint => eigenvalues REAL.

Connes-Moscovici (PNAS 2022): negative UV eigenvalues of W_lambda
restricted to complement of [-lambda, lambda] reproduce gamma_k^2.

Question: how EXACT is this match?
If exact: eigenvalues real => zeros real => RH. No circularity.
"""

import numpy as np
from scipy.linalg import eigh_tridiagonal
import time


def prolate_eigenvalues(lam, n_grid=2000):
    """Compute eigenvalues of the prolate operator W_lambda.

    W_lambda xi = -d/dx[(lam^2 - x^2) xi'] + (2*pi*lam)^2 x^2 xi

    Discretize on [-lam, lam] with finite differences.
    """
    # Grid on (-lam, lam)
    h = 2 * lam / (n_grid + 1)
    x = np.linspace(-lam + h, lam - h, n_grid)

    # Coefficient: a(x) = lam^2 - x^2
    a = lam**2 - x**2

    # Potential: V(x) = (2*pi*lam)^2 * x^2
    V = (2 * np.pi * lam)**2 * x**2

    # Finite difference: -d/dx[a(x) d/dx xi]
    # Using centered differences:
    # -[(a_{i+1/2}(xi_{i+1}-xi_i) - a_{i-1/2}(xi_i-xi_{i-1}))/h^2]
    a_half_plus = 0.5 * (a[1:] + a[:-1])    # a at x_{i+1/2}
    a_half_minus = 0.5 * (a[:-1] + a[1:])    # same (symmetric)

    # Tridiagonal: diagonal and off-diagonal
    # But for the full -d/dx[a d/dx], the tridiagonal form is:
    # main diagonal: (a_{i-1/2} + a_{i+1/2})/h^2 + V_i
    # off diagonal: -a_{i+1/2}/h^2

    main_diag = np.zeros(n_grid)
    main_diag[0] = (a[0] + a_half_plus[0]) / h**2 + V[0]
    main_diag[-1] = (a_half_minus[-1] + a[-1]) / h**2 + V[-1]
    for i in range(1, n_grid - 1):
        main_diag[i] = (a_half_plus[i-1] + a_half_plus[i]) / h**2 + V[i]

    off_diag = -a_half_plus / h**2

    # Solve tridiagonal eigenvalue problem
    evals = eigh_tridiagonal(main_diag, off_diag, eigvals_only=True)

    return evals, x, h


def prolate_exterior_eigenvalues(lam, L_ext=50, n_grid=4000):
    """Eigenvalues of W_lambda on the EXTERIOR (-inf, -lam) union (lam, inf).

    These are the ones that should match zeta zeros.
    Approximate by truncating to [-L_ext, -lam] union [lam, L_ext].
    """
    # Grid on [lam, L_ext] (use symmetry for the other half)
    n_half = n_grid // 2
    h = (L_ext - lam) / (n_half + 1)
    x = np.linspace(lam + h, L_ext - h, n_half)

    # On exterior: a(x) = lam^2 - x^2 is NEGATIVE (since |x| > lam)
    # So the operator becomes: -d/dx[(lam^2-x^2)xi'] + (2pi*lam)^2 x^2 xi
    # With lam^2 - x^2 < 0, this changes the sign of the kinetic term
    a = lam**2 - x**2  # negative!

    V = (2 * np.pi * lam)**2 * x**2

    # Tridiagonal (same structure, but a < 0 on exterior)
    a_half = 0.5 * (a[1:] + a[:-1])

    main_diag = np.zeros(n_half)
    for i in range(n_half):
        a_left = a_half[i-1] if i > 0 else a[0]
        a_right = a_half[i] if i < n_half - 1 else a[-1]
        main_diag[i] = (a_left + a_right) / h**2 + V[i]

    # Fix boundary
    main_diag[0] = (a[0] + a_half[0]) / h**2 + V[0]
    if n_half > 1:
        main_diag[-1] = (a_half[-1] + a[-1]) / h**2 + V[-1]

    off_diag = -a_half / h**2

    evals = eigh_tridiagonal(main_diag, off_diag, eigvals_only=True)

    return evals


if __name__ == "__main__":
    print("PROLATE OPERATOR ATTACK")
    print("=" * 70)

    gammas = np.load("_zeros_500.npy")
    gamma_sq = gammas**2  # gamma_k^2 are the target

    # ================================================================
    # PART 1: Interior eigenvalues of W_lambda on [-lambda, lambda]
    # ================================================================
    print("\nPART 1: INTERIOR EIGENVALUES (Sonin space)")
    print("-" * 70)

    for lam in [3, 5, 10, 20]:
        evals, x, h = prolate_eigenvalues(lam, n_grid=2000)

        n_neg = np.sum(evals < 0)
        n_pos = np.sum(evals > 0)
        print(f"  lam={lam:>3}: {n_neg} negative, {n_pos} positive, "
              f"min={evals[0]:.4e}, max={evals[-1]:.4e}")
        if n_neg > 0:
            print(f"    Negative eigenvalues: {', '.join(f'{e:.2e}' for e in evals[evals < 0][:5])}")

    # ================================================================
    # PART 2: Exterior eigenvalues — should match gamma_k^2
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 2: EXTERIOR EIGENVALUES vs gamma_k^2")
    print("-" * 70)

    for lam in [5, 10, 20, 50]:
        L_ext = max(100, 5*lam)
        n_grid = max(4000, int(200 * lam))

        t0 = time.time()
        evals_ext = prolate_exterior_eigenvalues(lam, L_ext, n_grid)
        dt = time.time() - t0

        # Find NEGATIVE eigenvalues (these should match gamma^2)
        neg_evals = np.sort(evals_ext[evals_ext < 0])

        print(f"\n  lam={lam:>3} (L_ext={L_ext}, n={n_grid}, {dt:.1f}s):")
        print(f"    {len(neg_evals)} negative eigenvalues")

        if len(neg_evals) > 0:
            print(f"    Negative eigenvalues |E_k| vs gamma_k^2:")
            print(f"    {'k':>4} {'|E_k|':>14} {'gamma_k^2':>14} {'ratio':>10} {'diff':>14}")
            for k in range(min(10, len(neg_evals), len(gamma_sq))):
                E_k = abs(neg_evals[k])
                g_sq = gamma_sq[k]
                ratio = E_k / g_sq if g_sq > 0 else 0
                diff = E_k - g_sq
                print(f"    {k+1:>4} {E_k:>14.6f} {g_sq:>14.6f} {ratio:>10.6f} {diff:>14.6f}")

    # ================================================================
    # PART 3: Does the match improve with lambda?
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 3: MATCH QUALITY vs LAMBDA")
    print("-" * 70)

    print(f"{'lam':>5} {'|E_1|':>14} {'gamma_1^2':>14} {'rel_err_1':>12} "
          f"{'|E_2|':>14} {'gamma_2^2':>14} {'rel_err_2':>12}")
    print("-" * 85)

    for lam in [3, 5, 8, 10, 15, 20, 30, 50, 75, 100]:
        L_ext = max(200, 5*lam)
        n_grid = max(6000, int(300 * lam))

        evals_ext = prolate_exterior_eigenvalues(lam, L_ext, n_grid)
        neg_evals = np.sort(evals_ext[evals_ext < 0])

        if len(neg_evals) >= 2:
            E1 = abs(neg_evals[0])
            E2 = abs(neg_evals[1])
            err1 = abs(E1 - gamma_sq[0]) / gamma_sq[0]
            err2 = abs(E2 - gamma_sq[1]) / gamma_sq[1]
            print(f"{lam:>5} {E1:>14.6f} {gamma_sq[0]:>14.6f} {err1:>12.6e} "
                  f"{E2:>14.6f} {gamma_sq[1]:>14.6f} {err2:>12.6e}")
        elif len(neg_evals) == 1:
            E1 = abs(neg_evals[0])
            err1 = abs(E1 - gamma_sq[0]) / gamma_sq[0]
            print(f"{lam:>5} {E1:>14.6f} {gamma_sq[0]:>14.6f} {err1:>12.6e}  (only 1 neg)")
        else:
            print(f"{lam:>5}  (no negative eigenvalues)")

    print(f"\n{'='*70}")
    print("VERDICT")
    print("=" * 70)
    print("""
If rel_err -> 0 as lambda -> inf: the match becomes EXACT in the limit.
Then: eigenvalues of self-adjoint W_lambda -> gamma_k^2
=> gamma_k^2 real => gamma_k real => RH

This would be a NON-CIRCULAR proof: W_lambda is defined WITHOUT zeta.
""")
