"""
SESSION 37 -- PROLATE PLUNGE AND THE SLEPIAN TRANSITION

Grok asks: track the spectral flow of Q_W eigenvalues through the prolate
plunge region as lambda grows.

THE ANSWER WE ALREADY KNOW: Q_W eigenvalues on null(W02) are exactly zero
at every finite lambda. So the "flow" is trivially zero.

BUT: The real question is about the TRUNCATION. We use N ~ 6*log(lambda^2)
Fourier modes. The Slepian transition happens at k ~ 2*lambda/pi.
For large lambda, our N may be SMALLER than the Slepian transition,
meaning we miss modes in the plunge region.

EXPERIMENT:
1. Compare our N vs the Slepian transition 2*sqrt(lam_sq)/pi at each lambda
2. At selected lambda values, increase N into and beyond the Slepian transition
3. Check if eigenvalues remain zero for modes in the transition window
4. Compute actual Slepian eigenvalues for our kernel
"""

import numpy as np
from scipy.linalg import eigh_tridiagonal
import time
import sys
sys.path.insert(0, '.')
from connes_crossterm import build_all


def slepian_transition(lam_sq):
    """The Slepian transition index: k ~ 2*lambda/pi where lambda = sqrt(lam_sq)."""
    return 2 * np.sqrt(lam_sq) / np.pi


def our_N(lam_sq):
    """Our standard truncation parameter."""
    L = np.log(lam_sq)
    return max(15, round(6 * L))


def n_vs_slepian():
    """Compare our N with Slepian transition at each lambda."""
    print("N vs SLEPIAN TRANSITION", flush=True)
    print(f"  {'lam^2':>8} {'lambda':>8} {'N (ours)':>8} {'k_Slepian':>10} {'ratio N/k':>10} {'regime':>12}",
          flush=True)

    for lam_sq in [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]:
        N = our_N(lam_sq)
        k_s = slepian_transition(lam_sq)
        ratio = N / k_s
        if ratio > 3:
            regime = "well past"
        elif ratio > 1.5:
            regime = "past"
        elif ratio > 0.8:
            regime = "AT EDGE"
        else:
            regime = "BELOW"

        print(f"  {lam_sq:>8} {np.sqrt(lam_sq):>8.1f} {N:>8} {k_s:>10.1f} {ratio:>10.2f} {regime:>12}",
              flush=True)


def vary_N_at_lambda(lam_sq, N_values):
    """
    At a fixed lambda, compute eigenvalues with different N values.
    Check if eigenvalues remain zero for modes beyond our standard N,
    especially through the Slepian transition.
    """
    k_s = slepian_transition(lam_sq)
    N_std = our_N(lam_sq)

    print(f"\nVARY N at lam^2={lam_sq} (Slepian transition at k={k_s:.1f})", flush=True)
    print(f"  {'N':>4} {'dim':>5} {'null_dim':>8} {'max_eig(M|null)':>18} {'N/k_Slep':>10} {'time':>6}",
          flush=True)

    for N in N_values:
        t0 = time.time()
        try:
            W02, M, QW = build_all(lam_sq, N, n_quad=10000)
        except Exception as e:
            print(f"  {N:>4}   ERROR: {e}", flush=True)
            continue

        dim = 2 * N + 1
        ew, ev = np.linalg.eigh(W02)
        thresh = np.max(np.abs(ew)) * 1e-10
        P_null = ev[:, np.abs(ew) <= thresh]
        d_null = P_null.shape[1]

        if d_null == 0:
            elapsed = time.time() - t0
            print(f"  {N:>4} {dim:>5} {0:>8} {'N/A':>18} {N/k_s:>10.2f} {elapsed:>5.1f}s", flush=True)
            continue

        M_null = P_null.T @ M @ P_null
        evals = np.linalg.eigvalsh(M_null)
        max_ev = np.max(evals)
        elapsed = time.time() - t0

        flag = "" if max_ev < 1e-6 else " ***FAIL***"
        marker = " <-- std" if N == N_std else (" <-- Slepian" if abs(N - round(k_s)) < 2 else "")
        print(f"  {N:>4} {dim:>5} {d_null:>8} {max_ev:>+18.8e} {N/k_s:>10.2f} {elapsed:>5.1f}s{flag}{marker}",
              flush=True)


def eigenvalue_count_flow():
    """
    Track how many modes exist in null(W02) as lambda grows.
    The null block dimension = dim - 2 = 2*N - 1 (since W02 has rank 2).
    As lambda grows, N grows, so null(W02) gains modes.
    These new modes all have eigenvalue = 0 (the tautology).
    """
    print(f"\nEIGENVALUE COUNT FLOW", flush=True)
    print(f"  {'lam^2':>8} {'N':>4} {'dim':>5} {'null_dim':>8} {'k_Slep':>8} "
          f"{'modes past Slep':>16} {'max_eig':>16}", flush=True)

    for lam_sq in [4, 10, 20, 50, 100, 200, 500, 1000]:
        N = our_N(lam_sq)
        k_s = slepian_transition(lam_sq)
        dim = 2 * N + 1
        null_dim = dim - 2

        # Modes beyond Slepian transition
        modes_past = max(0, N - round(k_s))

        t0 = time.time()
        W02, M, QW = build_all(lam_sq, N, n_quad=10000)

        ew, ev = np.linalg.eigh(W02)
        thresh = np.max(np.abs(ew)) * 1e-10
        P_null = ev[:, np.abs(ew) <= thresh]
        M_null = P_null.T @ M @ P_null
        max_ev = np.max(np.linalg.eigvalsh(M_null))
        elapsed = time.time() - t0

        print(f"  {lam_sq:>8} {N:>4} {dim:>5} {null_dim:>8} {k_s:>8.1f} "
              f"{modes_past:>16} {max_ev:>+16.6e} ({elapsed:.0f}s)", flush=True)


def full_spectrum_at_lambda(lam_sq, N=None):
    """
    Show the FULL eigenvalue spectrum of M on null(W02).
    All should be <= 0 (exactly 0 within quadrature error).
    Show the distribution: how many near zero, how many deeply negative?
    """
    if N is None:
        N = our_N(lam_sq)
    dim = 2 * N + 1

    print(f"\nFULL SPECTRUM at lam^2={lam_sq}, N={N}, dim={dim}", flush=True)

    W02, M, QW = build_all(lam_sq, N, n_quad=10000)

    ew, ev = np.linalg.eigh(W02)
    thresh = np.max(np.abs(ew)) * 1e-10
    P_null = ev[:, np.abs(ew) <= thresh]
    M_null = P_null.T @ M @ P_null
    evals = np.sort(np.linalg.eigvalsh(M_null))
    d = len(evals)

    # Classify eigenvalues by magnitude
    deeply_neg = np.sum(evals < -0.1)
    moderately_neg = np.sum((evals >= -0.1) & (evals < -1e-4))
    near_zero = np.sum(np.abs(evals) < 1e-4)

    print(f"  Total eigenvalues: {d}", flush=True)
    print(f"  Deeply negative (<-0.1): {deeply_neg}", flush=True)
    print(f"  Moderately negative: {moderately_neg}", flush=True)
    print(f"  Near zero (|e|<1e-4): {near_zero}", flush=True)
    print(f"  Most negative: {evals[0]:+.6f}", flush=True)
    print(f"  Least negative: {evals[-1]:+.6e}", flush=True)
    print(f"  Eigenvalue distribution:", flush=True)
    print(f"  {'idx':>4} {'eigenvalue':>14}", flush=True)
    # Show first 10, last 10
    for i in range(min(10, d)):
        print(f"  {i:>4} {evals[i]:>+14.6f}", flush=True)
    if d > 20:
        print(f"  {'...':>4}", flush=True)
    for i in range(max(d-10, 10), d):
        print(f"  {i:>4} {evals[i]:>+14.6e}", flush=True)


if __name__ == "__main__":
    print("SESSION 37 -- PROLATE PLUNGE", flush=True)
    print("=" * 80, flush=True)

    # 1. Where does our N sit relative to Slepian?
    n_vs_slepian()

    # 2. Full spectrum at a couple of lambda values
    full_spectrum_at_lambda(50)
    full_spectrum_at_lambda(200)

    # 3. Vary N at lam_sq=100 through the Slepian transition
    k_s_100 = slepian_transition(100)
    N_vals_100 = [5, 7, 10, 15, 20, 25, 28]
    vary_N_at_lambda(100, N_vals_100)

    # 4. Vary N at lam_sq=1000 — approaching Slepian boundary
    k_s_1000 = slepian_transition(1000)
    N_vals_1000 = [15, 20, 25, 30, 35, 41]
    vary_N_at_lambda(1000, N_vals_1000)

    # 5. Eigenvalue count flow
    eigenvalue_count_flow()

    print(f"\nDone.", flush=True)
