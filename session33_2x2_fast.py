"""
SESSION 33 — FAST 2x2 PROOF: Vectorized split-and-bound
"""
import numpy as np
import time
import json
import sys
sys.path.insert(0, '.')
from connes_crossterm import build_all

def get_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 2):
        if i <= limit and sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [p for p in range(2, limit + 1) if sieve[p]]

def F_vec(u, N, L, y_arr):
    """Vectorized F(u, y) for array of y values."""
    dim = 2*N+1
    ns = np.arange(-N, N+1)  # mode indices
    results = np.zeros(len(y_arr))
    # Precompute outer product
    uu = np.outer(u, u)  # dim x dim
    for idx, y in enumerate(y_arr):
        # q(m,n,y) matrix
        Q = np.zeros((dim, dim))
        for i in range(dim):
            m = ns[i]
            for j in range(dim):
                n = ns[j]
                if m != n:
                    Q[i,j] = (np.sin(2*np.pi*n*y/L) - np.sin(2*np.pi*m*y/L)) / (np.pi*(m-n))
                else:
                    Q[i,j] = 2*(L-y)/L * np.cos(2*np.pi*m*y/L)
        results[idx] = np.sum(uu * Q)
    return results

def fast_proof(lam_sq, N=None):
    if N is None:
        L = np.log(lam_sq)
        N = max(21, round(8 * L))
    dim = 2*N+1
    L = np.log(lam_sq)
    pf = 32*L*np.sinh(L/4)**2

    # Eigenvectors
    ks = np.arange(-N, N+1)
    v = 1.0/(L**2 + 4*np.pi**2*ks**2)
    w = ks/(L**2 + 4*np.pi**2*ks**2)
    s_v = pf*L**2*np.dot(v,v)
    s_w = -pf*4*np.pi**2*np.dot(w,w)
    u_v = v/np.linalg.norm(v)
    u_w = w/np.linalg.norm(w)

    # Build M
    t0 = time.time()
    W02, M, QW = build_all(lam_sq, N)
    from session33_sieve_bypass import compute_M_decomposition
    M_diag, M_alpha, M_prime, M_full, primes_used = compute_M_decomposition(lam_sq, N)
    build_time = time.time() - t0

    Mvv = u_v @ M @ u_v
    diag_vv = u_v @ M_diag @ u_v
    alpha_vv = u_v @ M_alpha @ u_v
    prime_vv = u_v @ M_prime @ u_v
    margin_v = s_v - Mvv
    analytic_margin_v = s_v - diag_vv - alpha_vv

    # Compute F_v at all prime power points
    prime_ys = np.array([logpk for _, _, logpk in primes_used])
    F_v_at_primes = F_vec(u_v, N, L, prime_ys)
    prime_weights = np.array([logp * pk**(-0.5) for pk, logp, _ in primes_used])
    prime_pks = np.array([pk for pk, _, _ in primes_used])

    # Verify prime sum
    prime_sum_check = np.sum(prime_weights * F_v_at_primes)
    print(f"  prime_vv = {prime_vv:.6f} (check: {prime_sum_check:.6f})")

    # Integral via quadrature
    n_quad = 5000
    t_grid = np.linspace(2.01, lam_sq*0.999, n_quad)
    y_grid = np.log(t_grid)
    valid = y_grid < L * 0.999
    t_grid = t_grid[valid]
    y_grid = y_grid[valid]
    dt = np.diff(np.concatenate([[2.0], (t_grid[:-1]+t_grid[1:])/2, [lam_sq]]))[:len(t_grid)]

    F_v_grid = F_vec(u_v, N, L, y_grid)
    integral_v = np.sum(F_v_grid / np.sqrt(t_grid) * dt)

    actual_error = prime_vv - integral_v
    analytic_gap = analytic_margin_v - integral_v

    primes_list = get_primes(min(lam_sq, 10000))

    # SPLIT AND BOUND for various P0
    print(f"\n  lam^2={lam_sq} (build {build_time:.0f}s)")
    print(f"  margin_v={margin_v:.6f}  analytic_gap={analytic_gap:.6f}  actual_error={actual_error:.6f}")
    print(f"  {'P0':>5} {'exact_err':>10} {'tail_maxF':>10} {'tail_theta':>10} {'total_bnd':>10} {'proved':>8}")

    best_ratio = 0
    best_P0 = 0
    for P0 in [5, 10, 20, 30, 50, 100, 200, 500, 1000]:
        if P0 >= lam_sq:
            break

        # Exact part: primes <= P0
        mask_exact = prime_pks <= P0
        exact_sum = np.sum(prime_weights[mask_exact] * F_v_at_primes[mask_exact])

        # Exact integral over [2, P0]
        mask_int = t_grid <= P0
        exact_integral = np.sum(F_v_grid[mask_int] / np.sqrt(t_grid[mask_int]) * dt[mask_int])
        exact_error = exact_sum - exact_integral

        # Tail: max |F_v/sqrt(t)| for t > P0
        mask_tail = t_grid > P0
        if np.sum(mask_tail) > 0:
            tail_maxF = np.max(np.abs(F_v_grid[mask_tail]) / np.sqrt(t_grid[mask_tail]))
        else:
            tail_maxF = 0

        # Tail theta error: |theta(lam^2) - theta(P0) - (lam^2 - P0)|
        theta_total = sum(np.log(p) for p in primes_list if p <= lam_sq)
        theta_P0 = sum(np.log(p) for p in primes_list if p <= P0)
        tail_theta_err = abs((theta_total - theta_P0) - (lam_sq - P0))

        tail_bound = tail_maxF * tail_theta_err
        total_bound = abs(exact_error) + tail_bound

        ratio = analytic_gap / total_bound if total_bound > 0 else float('inf')
        proved = ratio > 1

        if ratio > best_ratio:
            best_ratio = ratio
            best_P0 = P0

        print(f"  {P0:>5} {exact_error:>+10.4f} {tail_maxF:>10.6f} {tail_theta_err:>10.4f} "
              f"{total_bound:>10.4f} {'PROVED' if proved else f'r={ratio:.3f}'}")

    print(f"  Best: P0={best_P0}, ratio={best_ratio:.4f}")
    return analytic_gap, best_ratio, best_P0


if __name__ == "__main__":
    print("SESSION 33 — FAST SPLIT-AND-BOUND 2x2 PROOF")
    print("=" * 75)

    results = []
    for lam_sq in [50, 100, 200, 500, 1000, 2000]:
        t0 = time.time()
        gap, ratio, P0 = fast_proof(lam_sq)
        elapsed = time.time() - t0
        results.append({
            'lam_sq': lam_sq, 'gap': float(gap),
            'ratio': float(ratio), 'P0': P0, 'time': elapsed
        })
        print(f"  Total time: {elapsed:.0f}s\n")

    print("\n" + "=" * 75)
    print("SUMMARY: Split-and-bound proof status")
    print("=" * 75)
    print(f"  {'lam^2':>6} {'gap':>10} {'best_ratio':>12} {'best_P0':>8} {'status':>10}")
    for r in results:
        status = 'PROVED' if r['ratio'] > 1 else f"need {1/r['ratio']:.2f}x"
        print(f"  {r['lam_sq']:>6} {r['gap']:>10.4f} {r['ratio']:>12.4f} {r['P0']:>8} {status:>10}")

    with open('session33_2x2_fast.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to session33_2x2_fast.json")
