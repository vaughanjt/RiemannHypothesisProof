"""
SESSION 38d — PER-PRIME MARGIN ANALYSIS

The key discovery: Neg >= Ma + Pos on null(W02), where
  Neg = Gram matrix of negative eigenvectors of T(p^k)|null
  Pos = Gram matrix of positive eigenvectors of T(p^k)|null
  Ma = analytic part (digamma + alpha)

For this to hold at all bandwidths, we need:
  Per-prime margin: for each T(p^k)|null, the negative eigenvalue
  exceeds the positive eigenvalue by a consistent margin.

COMPUTE: For each prime power, the eigenvalue split of T(p^k)|null(W02).
Track the margin (|neg| - |pos|) as a function of p, k, and lambda.
"""

import numpy as np
import time
import sys
sys.path.insert(0, '.')
from connes_crossterm import build_all
from session33_sieve_bypass import compute_M_decomposition


def per_prime_eigenvalue_split(lam_sq, N=None):
    """
    For each T(p^k) restricted to null(W02), compute:
    - The most positive eigenvalue (lambda_+)
    - The most negative eigenvalue (lambda_-)
    - The trace (lambda_+ + lambda_- + rest)
    - The margin: |lambda_-| - |lambda_+|
    """
    if N is None:
        L_f = np.log(lam_sq)
        N = max(15, round(6 * L_f))
    dim = 2 * N + 1
    L_f = np.log(lam_sq)
    ns = np.arange(-N, N + 1, dtype=float)

    W02, M, QW = build_all(lam_sq, N, n_quad=10000)
    _, _, _, _, primes = compute_M_decomposition(lam_sq, N)

    ew, ev = np.linalg.eigh(W02)
    thresh = np.max(np.abs(ew)) * 1e-10
    P_null = ev[:, np.abs(ew) <= thresh]
    d_null = P_null.shape[1]

    print(f"PER-PRIME EIGENVALUE SPLIT: lam^2={lam_sq}, dim={dim}, null_dim={d_null}", flush=True)
    print(f"  {'p^k':>6} {'weight':>8} {'trace|null':>10} {'max_eig':>10} {'min_eig':>10} "
          f"{'|min|-|max|':>12} {'sum_pos':>10} {'sum_neg':>10}", flush=True)

    total_pos = 0
    total_neg = 0
    total_trace = 0
    margin_data = []

    for pk, logp, logpk in primes:
        Q = np.zeros((dim, dim))
        for i in range(dim):
            m = ns[i]
            for j in range(dim):
                n = ns[j]
                if m != n:
                    Q[i, j] = (np.sin(2*np.pi*n*logpk/L_f) -
                               np.sin(2*np.pi*m*logpk/L_f)) / (np.pi*(m-n))
                else:
                    Q[i, j] = 2*(L_f - logpk)/L_f * np.cos(2*np.pi*m*logpk/L_f)
        Q = (Q + Q.T) / 2
        w = logp * pk**(-0.5)
        T = w * Q

        T_null = P_null.T @ T @ P_null
        evals_T = np.linalg.eigvalsh(T_null)

        max_e = np.max(evals_T)
        min_e = np.min(evals_T)
        tr = np.sum(evals_T)
        s_pos = np.sum(evals_T[evals_T > 1e-12])
        s_neg = np.sum(evals_T[evals_T < -1e-12])
        margin = abs(min_e) - abs(max_e)

        total_pos += s_pos
        total_neg += s_neg
        total_trace += tr
        margin_data.append((pk, logp, w, margin, tr, max_e, min_e, s_pos, s_neg))

        print(f"  {pk:>6} {w:>8.4f} {tr:>+10.4f} {max_e:>+10.4f} {min_e:>+10.4f} "
              f"{margin:>+12.4f} {s_pos:>10.4f} {s_neg:>10.4f}", flush=True)

    print(f"\n  TOTALS:", flush=True)
    print(f"  Sum positive: {total_pos:.4f}", flush=True)
    print(f"  Sum negative: {total_neg:.4f}", flush=True)
    print(f"  Sum trace:    {total_trace:.4f}", flush=True)
    print(f"  Net margin:   {abs(total_neg) - total_pos:.4f}", flush=True)

    # Is every per-prime trace negative?
    n_neg_trace = sum(1 for _, _, _, _, tr, _, _, _, _ in margin_data if tr < -1e-10)
    n_pos_trace = sum(1 for _, _, _, _, tr, _, _, _, _ in margin_data if tr > 1e-10)
    print(f"\n  Primes with negative trace|null: {n_neg_trace}/{len(margin_data)}", flush=True)
    print(f"  Primes with positive trace|null: {n_pos_trace}/{len(margin_data)}", flush=True)

    # Is every per-prime margin positive (|min_eig| > |max_eig|)?
    n_pos_margin = sum(1 for _, _, _, m, _, _, _, _, _ in margin_data if m > 1e-10)
    print(f"  Primes with positive margin (|min|>|max|): {n_pos_margin}/{len(margin_data)}", flush=True)

    return margin_data


def margin_vs_bandwidth():
    """
    Track the per-prime margin and totals across multiple bandwidths.
    Does the structural dominance hold at all tested bandwidths?
    """
    print(f"\nMARGIN VS BANDWIDTH", flush=True)
    print(f"  {'lam^2':>8} {'#pk':>5} {'sum_pos':>10} {'sum_neg':>10} {'trace':>10} "
          f"{'neg-pos':>10} {'Ma_trace':>10} {'SURPLUS':>10} {'NSD?':>5}", flush=True)

    for lam_sq in [10, 20, 50, 100, 200, 500]:
        L_f = np.log(lam_sq)
        N = max(15, round(6 * L_f))
        dim = 2 * N + 1
        ns = np.arange(-N, N + 1, dtype=float)

        W02, M, QW = build_all(lam_sq, N, n_quad=10000)
        M_diag, M_alpha, M_prime, _, primes = compute_M_decomposition(lam_sq, N)

        ew, ev = np.linalg.eigh(W02)
        thresh = np.max(np.abs(ew)) * 1e-10
        P_null = ev[:, np.abs(ew) <= thresh]
        d_null = P_null.shape[1]

        Ma_null = P_null.T @ (M_diag + M_alpha) @ P_null
        Ma_trace = np.trace(Ma_null)

        total_pos = 0
        total_neg = 0
        for pk, logp, logpk in primes:
            Q = np.zeros((dim, dim))
            for i in range(dim):
                m = ns[i]
                for j in range(dim):
                    n = ns[j]
                    if m != n:
                        Q[i, j] = (np.sin(2*np.pi*n*logpk/L_f) -
                                   np.sin(2*np.pi*m*logpk/L_f)) / (np.pi*(m-n))
                    else:
                        Q[i, j] = 2*(L_f - logpk)/L_f * np.cos(2*np.pi*m*logpk/L_f)
            Q = (Q + Q.T) / 2
            w = logp * pk**(-0.5)
            T = w * Q
            T_null = P_null.T @ T @ P_null
            evals_T = np.linalg.eigvalsh(T_null)
            total_pos += np.sum(evals_T[evals_T > 1e-12])
            total_neg += np.sum(evals_T[evals_T < -1e-12])

        trace_Mp = total_pos + total_neg
        net_margin = abs(total_neg) - total_pos  # = -trace_Mp
        surplus = abs(total_neg) - total_pos - abs(Ma_trace)
        # surplus > 0 means Neg > Ma + Pos in trace

        M_null = P_null.T @ M @ P_null
        is_nsd = np.max(np.linalg.eigvalsh(M_null)) < 1e-6

        print(f"  {lam_sq:>8} {len(primes):>5} {total_pos:>10.2f} {total_neg:>10.2f} "
              f"{trace_Mp:>+10.2f} {net_margin:>10.2f} {Ma_trace:>+10.2f} "
              f"{surplus:>+10.2f} {'YES' if is_nsd else 'NO':>5}", flush=True)


if __name__ == "__main__":
    print("SESSION 38d — PER-PRIME MARGIN ANALYSIS", flush=True)
    print("=" * 80, flush=True)

    # Detailed per-prime split at lam_sq=50
    margin_data = per_prime_eigenvalue_split(50)

    # Margin vs bandwidth
    margin_vs_bandwidth()

    print(f"\nDone.", flush=True)
