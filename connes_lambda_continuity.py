"""
Session 29p: THE FINAL GAP — lambda-continuity of the eigenvector.

For fixed N, does xi_{lambda,N} stabilize as lambda -> infinity?

The eigenvector equation: Q_W(lambda) * xi = eps_0(lambda) * xi

By first-order perturbation theory (Hellmann-Feynman):
  d(eps_0)/d(lambda) = xi^T * (dQ_W/dlambda) * xi

And the eigenvector derivative (for non-degenerate eigenvalue):
  d(xi)/d(lambda) = sum_{k != 0} [xi_k^T * (dQ_W/dlambda) * xi_0] / (eps_0 - eps_k) * xi_k

The KEY: if ||dQ_W/dlambda|| is bounded and the spectral gap
|eps_k - eps_0| >= delta > 0 for all k != 0, then:
  ||d(xi)/d(lambda)|| <= ||dQ_W/dlambda|| / delta

For UNIFORM H1: we need ||d(xi)/d(lambda)|| to be integrable as lambda -> inf,
so that xi(lambda) converges.

Since lambda appears through L = log(lam^2) = 2*log(lambda):
  dL/dlambda = 2/lambda
  dQ_W/dlambda = (dQ_W/dL) * (2/lambda)

So ||dQ_W/dlambda|| = ||dQ_W/dL|| * (2/lambda).

The factor 2/lambda -> 0 helps! If ||dQ_W/dL|| is bounded:
  ||d(xi)/dlambda|| <= 2 * ||dQ_W/dL|| / (lambda * delta)

The integral from lambda_0 to infinity:
  integral ||d(xi)/dlambda|| dlambda <= integral (C / (lambda * delta)) dlambda

This converges iff delta doesn't go to 0 too fast.
With delta = eps_1 - eps_0 ~ 3*eps_0 ~ 3*C*exp(-cL) = 3*C*lam^{-2c}:
  integral C / (lambda * lam^{-2c}) dlambda = integral C * lam^{2c-1} dlambda

This CONVERGES for 2c - 1 < -1, i.e., c > 1. But c = L/(8*pi*N) which is small!
For N=15, c = L/(8*pi*15) ~ L/377. At large lambda, c ~ log(lambda)/188.

So c -> infinity with lambda, which means 2c-1 -> infinity. The integral DIVERGES!

WAIT: eps_0 is the smallest eigenvalue, but the GAP is to eps_1.
The gap delta = eps_1 - eps_0 ~ 3*eps_0 ~ exp(-cL).
But the gap to the BULK of the spectrum is eps_2 - eps_0 ~ O(1) - eps_0 ~ O(1).

For the eigenvector derivative, the sum over k includes ALL eigenvectors.
The dominant contribution comes from the NEAREST eigenvalue (eps_1),
but if eps_1 is also tiny, this is problematic.

HOWEVER: eps_1 is the SECOND tiny eigenvalue. The third eigenvalue eps_2
is O(1) (from our data: eps_2 ~ 0.08 at lam^2=14).

So the eigenvector derivative has two parts:
  - Rotation within the 2D "small eigenvalue" subspace: ~ dQ_W/delta_12
    where delta_12 = eps_1 - eps_0 ~ 2*eps_0
  - Rotation toward the bulk: ~ dQ_W/delta_bulk where delta_bulk ~ O(1)

The second part is bounded. The first part (2D rotation) is just the
ANGLE between the even/odd W02 eigenvectors, which is determined by
the parity structure and doesn't really "rotate" — the parity is FIXED.

Let me compute this numerically.
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, log, pi, euler, exp, cos, sin,
                    hyp2f1, digamma, sinh, nstr)
import time

mp.dps = 30

# Import build_QW
import sys
sys.path.insert(0, '.')
from connes_h1h2_correct import build_QW


if __name__ == "__main__":
    print("THE FINAL GAP: LAMBDA-CONTINUITY")
    print("=" * 70)

    N = 15  # Fixed N (smallest that gives positive Q_W for all lambda)

    # ================================================================
    # PART 1: Eigenvector trajectory as lambda varies
    # ================================================================
    print("\nPART 1: EIGENVECTOR TRAJECTORY vs LAMBDA")
    print("-" * 70)

    lam_sq_values = [10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 50, 60, 75, 100]

    prev_xi = None
    prev_eps = None
    prev_lam = None
    results = []

    print(f"{'lam^2':>6} {'eps_0':>12} {'eps_1':>12} {'eps_2':>12} "
          f"{'||d_xi||':>10} {'|d_eps|':>10}")
    print("-" * 70)

    for lam_sq in lam_sq_values:
        t0 = time.time()
        QW = build_QW(lam_sq, N)
        evals, evecs = np.linalg.eigh(QW)
        xi_0 = evecs[:, 0]
        eps_0, eps_1, eps_2 = evals[0], evals[1], evals[2]

        if prev_xi is not None:
            # Align sign
            if np.dot(xi_0, prev_xi) < 0:
                xi_0_aligned = -xi_0
            else:
                xi_0_aligned = xi_0

            d_xi = np.linalg.norm(xi_0_aligned - prev_xi)
            d_eps = abs(eps_0 - prev_eps)
            d_lam = lam_sq - prev_lam

            print(f"{lam_sq:>6} {eps_0:>12.4e} {eps_1:>12.4e} {eps_2:>12.4e} "
                  f"{d_xi:>10.4e} {d_eps:>10.4e}")

            results.append({
                'lam_sq': lam_sq, 'eps_0': eps_0, 'eps_1': eps_1, 'eps_2': eps_2,
                'd_xi': d_xi, 'd_eps': d_eps, 'd_lam': d_lam
            })
        else:
            print(f"{lam_sq:>6} {eps_0:>12.4e} {eps_1:>12.4e} {eps_2:>12.4e} "
                  f"{'---':>10} {'---':>10}")

        prev_xi = xi_0.copy()
        prev_eps = eps_0
        prev_lam = lam_sq

    # ================================================================
    # PART 2: Does ||d_xi/d_lambda|| decrease?
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 2: RATE OF EIGENVECTOR CHANGE")
    print("-" * 70)

    if results:
        print(f"\n{'lam^2':>6} {'d_xi/d_lam':>12} {'d_eps/d_lam':>12} "
              f"{'gap_12':>12} {'gap_bulk':>12}")
        print("-" * 60)

        for r in results:
            d_xi_rate = r['d_xi'] / r['d_lam']
            d_eps_rate = r['d_eps'] / r['d_lam']
            gap_12 = r['eps_1'] - r['eps_0']
            gap_bulk = r['eps_2'] - r['eps_0']
            print(f"{r['lam_sq']:>6} {d_xi_rate:>12.4e} {d_eps_rate:>12.4e} "
                  f"{gap_12:>12.4e} {gap_bulk:>12.4e}")

    # ================================================================
    # PART 3: Numerical derivative dQ_W/dL
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 3: ||dQ_W/dL|| NUMERICAL ESTIMATE")
    print("-" * 70)

    for lam_sq in [14, 50, 100]:
        L = np.log(lam_sq)
        delta_L = 0.01

        lam_sq_plus = np.exp(L + delta_L)
        lam_sq_minus = np.exp(L - delta_L)

        QW_plus = build_QW(lam_sq_plus, N)
        QW_minus = build_QW(lam_sq_minus, N)

        dQW_dL = (QW_plus - QW_minus) / (2 * delta_L)
        norm_dQW = np.linalg.norm(dQW_dL, ord=2)  # spectral norm

        # Hellmann-Feynman: d(eps_0)/dL = xi^T * dQ_W/dL * xi
        QW = build_QW(lam_sq, N)
        evals, evecs = np.linalg.eigh(QW)
        xi_0 = evecs[:, 0]
        eps_0 = evals[0]
        eps_1 = evals[1]
        gap = eps_1 - eps_0

        d_eps_dL = xi_0 @ dQW_dL @ xi_0

        # Eigenvector derivative norm bound
        # ||d(xi)/dL|| <= ||dQ_W/dL|| / gap (worst case)
        # But this is the bound using ALL eigenvalues, not just the nearest
        xi_deriv_bound = norm_dQW / gap

        # Better bound: decompose into 2D subspace + bulk
        # The 2D part (near eps_0 and eps_1) contributes:
        xi_1 = evecs[:, 1]
        coupling_01 = xi_1 @ dQW_dL @ xi_0
        rotation_rate = abs(coupling_01) / gap
        # The bulk part contributes:
        bulk_contrib = sum((evecs[:, k] @ dQW_dL @ xi_0)**2 / (eps_0 - evals[k])**2
                          for k in range(2, len(evals)))
        bulk_rate = np.sqrt(bulk_contrib)

        print(f"\n  lam^2={lam_sq} (L={L:.3f}):")
        print(f"    ||dQ_W/dL|| = {norm_dQW:.6f}")
        print(f"    d(eps_0)/dL = {d_eps_dL:.6e}")
        print(f"    gap(eps_1-eps_0) = {gap:.6e}")
        print(f"    Naive bound ||d(xi)/dL|| <= {xi_deriv_bound:.4e}")
        print(f"    2D rotation rate = {rotation_rate:.6e}")
        print(f"    Bulk rotation rate = {bulk_rate:.6e}")
        print(f"    Total ||d(xi)/dL|| ~ {np.sqrt(rotation_rate**2 + bulk_rate**2):.6e}")

    # ================================================================
    # PART 4: The convergence integral
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 4: CONVERGENCE INTEGRAL")
    print("-" * 70)

    # xi(lambda) = xi(lambda_0) + integral_{lambda_0}^{lambda} (d xi/d lambda') d lambda'
    # ||xi(lambda) - xi(lambda_0)|| <= integral ||d xi/d lambda'|| d lambda'
    #
    # d xi/d lambda = (d xi/d L) * (d L/d lambda) = (d xi/d L) * (2/lambda)
    #
    # So ||d xi / d lambda|| = ||d xi / d L|| * (2/lambda)
    #
    # The integral: integral_lambda0^inf ||d xi/d lambda|| d lambda
    #             = integral_lambda0^inf ||d xi/d L|| * (2/lambda) d lambda
    #
    # If ||d xi/d L|| is bounded by C, this integral = 2C * integral 1/lambda = DIVERGENT!
    #
    # But if ||d xi/d L|| DECREASES with lambda, the integral might converge.

    print("  Testing: does ||d xi/d L|| decrease with lambda?")
    for r in results:
        lam = np.sqrt(r['lam_sq'])
        L = np.log(r['lam_sq'])
        dL = r['d_lam'] / r['lam_sq']  # approximate dL from d(lam^2)
        if dL > 0:
            d_xi_dL = r['d_xi'] / dL
            print(f"    lam^2={r['lam_sq']:>4}: ||d xi/d L|| ~ {d_xi_dL:.4e}")

    # ================================================================
    # PART 5: The 2D subspace evolution
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 5: 2D SUBSPACE EVOLUTION (EVEN/ODD)")
    print("-" * 70)

    # The eigenvector lives mostly in the 2D W02 subspace (overlap ~ 0.84).
    # Within this 2D subspace, the eigenvector is determined by the
    # even/odd mixing angle theta:
    #   xi ~ cos(theta) * v_even + sin(theta) * v_odd
    #
    # If theta is constant (or converges) as lambda -> inf, then
    # the eigenvector is stable.
    #
    # The angle theta comes from the 2x2 projected Q_W:
    #   [[QW_ee, QW_eo], [QW_oe, QW_oo]]
    # Since this is diagonal (by parity), theta = 0 or pi/2.
    # The EVEN or ODD vector is the eigenvector (whichever gives smaller eigenvalue).

    for lam_sq in [14, 30, 50, 75, 100]:
        QW = build_QW(lam_sq, N)
        evals, evecs = np.linalg.eigh(QW)
        xi_0 = evecs[:, 0]
        center = N

        # Check parity
        even_score = sum(abs(xi_0[center+k] - xi_0[center-k]) for k in range(1, N+1))
        odd_score = sum(abs(xi_0[center+k] + xi_0[center-k]) for k in range(1, N+1))
        parity = "EVEN" if even_score < odd_score else "ODD"

        # Overlap with W02 eigenvectors
        evals_w02, evecs_w02 = np.linalg.eigh(build_W02(lam_sq, N))
        idx_w02 = np.where(np.abs(evals_w02) > 1e-10)[0]
        overlaps = [abs(np.dot(xi_0, evecs_w02[:, idx])) for idx in idx_w02]

        print(f"  lam^2={lam_sq:>4}: eps_0={evals[0]:.4e}, parity={parity}, "
              f"W02 overlaps: {', '.join(f'{o:.4f}' for o in overlaps)}")


def build_W02(lam_sq, N_val):
    """Just the W02 component."""
    L = log(mpf(lam_sq))
    L_f = float(L)
    dim = 2*N_val + 1
    L2_f = L_f**2
    p2_f = (4*np.pi)**2
    pf_f = 32*L_f*float(sinh(L/4))**2
    W02 = np.zeros((dim, dim))
    for i in range(dim):
        n = i - N_val
        for j in range(dim):
            m = j - N_val
            W02[i,j] = pf_f*(L2_f - p2_f*m*n) / ((L2_f + p2_f*m**2)*(L2_f + p2_f*n**2))
    return W02


# Need to define this before using it in PART 5 — move function def above main
