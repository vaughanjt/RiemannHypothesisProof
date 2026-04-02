"""
SESSION 34 — ASYMPTOTIC PROOF: 2x2 Range Block PD for ALL lambda

GOAL: Prove the split-and-bound argument works for all lambda >= lambda_0.

The 2x2 range block has:
  margin_v = s_v - <u_v, M u_v>
  margin_w = s_w - <u_w, M u_w>

Both stay positive with margin ~ 0.03 (scaling as lam^{-0.024}).

The split-and-bound proof at specific lambda:
  1. Compute small primes exactly (p <= P0)
  2. Bound tail (p > P0) by max|F/sqrt(t)| * |theta(lam^2) - theta(P0) - (lam^2 - P0)|
  3. Show analytic_gap > exact_error + tail_bound

For asymptotic: show this works for ALL lam >= lam_0 by:
  a. The analytic gap grows as ~ C1 * sqrt(lam)
  b. The tail bound grows as ~ C2 * sqrt(lam) / log(lam)  [PNT error]
  c. Ratio ~ C1/C2 * log(lam) -> infinity
  d. Find explicit lam_0 where ratio first exceeds 1

ALSO: Fine scan over many lambda values to verify no failures.
"""

import numpy as np
import time, sys
sys.path.insert(0, '.')
from connes_crossterm import build_all
from session33_sieve_bypass import compute_M_decomposition


def get_primes(limit):
    sieve = [True]*(limit+1); sieve[0]=sieve[1]=False
    for i in range(2, int(limit**0.5)+2):
        if i<=limit and sieve[i]:
            for j in range(i*i,limit+1,i): sieve[j]=False
    return [p for p in range(2,limit+1) if sieve[p]]


def F_batch(u, N, L, y_arr):
    dim = 2*N+1
    ns = np.arange(-N, N+1, dtype=float)
    uu = np.outer(u, u)
    m_grid = ns[:, None]
    n_grid = ns[None, :]
    diff = m_grid - n_grid
    results = np.zeros(len(y_arr))
    for idx, y in enumerate(y_arr):
        sin_n = np.sin(2*np.pi*n_grid*y/L)
        sin_m = np.sin(2*np.pi*m_grid*y/L)
        Q = np.where(diff != 0, (sin_n - sin_m) / (np.pi * np.where(diff!=0, diff, 1)), 0)
        diag_vals = 2*(L-y)/L * np.cos(2*np.pi*ns*y/L)
        np.fill_diagonal(Q, diag_vals)
        results[idx] = np.sum(uu * Q)
    return results


def quick_2x2_test(lam_sq, N=None):
    """Fast test: is the 2x2 range block PD for this lambda?"""
    if N is None:
        L = np.log(lam_sq)
        N = max(15, round(6*L))
    dim = 2*N+1; L = np.log(lam_sq)
    pf = 32*L*np.sinh(L/4)**2
    ks = np.arange(-N, N+1, dtype=float)
    v = 1.0/(L**2 + 4*np.pi**2*ks**2)
    w = ks/(L**2 + 4*np.pi**2*ks**2)
    s_v = pf*L**2*np.dot(v,v)
    s_w = -pf*4*np.pi**2*np.dot(w,w)
    u_v = v/np.linalg.norm(v)
    u_w = w/np.linalg.norm(w)

    W02, M, QW = build_all(lam_sq, N)
    Mvv = u_v @ M @ u_v
    Mww = u_w @ M @ u_w
    margin_v = s_v - Mvv
    margin_w = s_w - Mww
    return margin_v, margin_w, s_v, s_w


def fine_scan():
    """Scan many lambda values to verify the 2x2 block stays PD."""
    print("FINE SCAN: 2x2 range block margins across lambda")
    print("=" * 75)

    # Dense scan of lambda^2 values
    lam_sq_values = list(range(10, 101, 5)) + list(range(100, 501, 25)) + \
                    list(range(500, 2001, 100))

    print(f"  {'lam^2':>6} {'margin_v':>10} {'margin_w':>10} {'min':>10} {'PD':>5}")

    all_pd = True
    results = []
    for lam_sq in lam_sq_values:
        mv, mw, sv, sw = quick_2x2_test(lam_sq)
        pd = mv > 0 and mw > 0
        if not pd:
            all_pd = False
        results.append((lam_sq, mv, mw))
        if lam_sq <= 100 or lam_sq % 100 == 0 or not pd:
            print(f"  {lam_sq:>6} {mv:>10.4f} {mw:>10.4f} {min(mv,abs(mw)):>10.4f} "
                  f"{'YES' if pd else '*** NO ***'}")

    print(f"\n  ALL PD: {'YES' if all_pd else 'NO'}")
    print(f"  Tested {len(lam_sq_values)} values from lam^2={lam_sq_values[0]} to {lam_sq_values[-1]}")

    # Fit the scaling
    lams = np.array([r[0] for r in results], dtype=float)
    mvs = np.array([r[1] for r in results])
    mws = np.array([abs(r[2]) for r in results])

    # margin_v scaling
    valid = mvs > 0
    if np.sum(valid) > 5:
        alpha, logC = np.polyfit(np.log(lams[valid]), np.log(mvs[valid]), 1)
        print(f"\n  margin_v ~ {np.exp(logC):.4f} * lam^({alpha:.4f})")

    # margin_w scaling
    valid_w = mws > 0
    if np.sum(valid_w) > 5:
        alpha_w, logC_w = np.polyfit(np.log(lams[valid_w]), np.log(mws[valid_w]), 1)
        print(f"  margin_w ~ {np.exp(logC_w):.4f} * lam^({alpha_w:.4f})")

    return results


def asymptotic_bound():
    """
    Derive the asymptotic behavior of the split-and-bound proof.

    For large lambda:
    - s_v ~ pf * L^2 * sum 1/(L^2+4pi^2k^2)^2
          ~ pf * L^2 * (L/(4pi)) * (1/L^4) (Euler-Maclaurin)
          ~ 32L*sinh^2(L/4) / (4pi*L)
          ~ 8*sinh^2(L/4) / pi
          ~ 2*exp(L/2) / pi  for large L

    - Mvv grows similarly but slightly less due to PNT error

    The RATIO margin_v / s_v should be computed for scaling analysis.
    """
    print("\n\nASYMPTOTIC SCALING ANALYSIS")
    print("=" * 75)

    lam_sq_values = [50, 100, 200, 500, 1000, 2000]
    print(f"  {'lam^2':>6} {'s_v':>12} {'margin_v':>12} {'margin/s_v':>12} "
          f"{'s_w':>12} {'margin_w':>12} {'margin/s_w':>12}")

    for lam_sq in lam_sq_values:
        mv, mw, sv, sw = quick_2x2_test(lam_sq)
        rv = mv / sv if sv != 0 else 0
        rw = mw / sw if sw != 0 else 0
        print(f"  {lam_sq:>6} {sv:>12.4f} {mv:>12.4f} {rv:>12.6f} "
              f"{sw:>12.4f} {mw:>12.4f} {rw:>12.6f}")

    print(f"\n  The RELATIVE margin (margin/eigenvalue) tells us")
    print(f"  what fraction of W02 survives after subtracting M.")
    print(f"  If this ratio stays bounded above 0, the proof works asymptotically.")


if __name__ == "__main__":
    print("SESSION 34 -- ASYMPTOTIC 2x2 PROOF")
    print("=" * 75)

    # Part 1: Fine scan
    results = fine_scan()

    # Part 2: Asymptotic scaling
    asymptotic_bound()

    import json
    with open('session34_asymptotic.json', 'w') as f:
        json.dump({'scan': [(r[0], float(r[1]), float(r[2])) for r in results]}, f, indent=2)
    print(f"\nSaved to session34_asymptotic.json")
