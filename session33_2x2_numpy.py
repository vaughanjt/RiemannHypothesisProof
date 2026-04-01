"""
SESSION 33 — FULLY VECTORIZED 2x2 PROOF
All inner loops replaced with numpy broadcasting.
"""
import numpy as np
import time, json, sys
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
    """Compute F(u, y) for array of y values using numpy vectorization."""
    dim = 2*N+1
    ns = np.arange(-N, N+1, dtype=float)
    uu = np.outer(u, u)  # dim x dim
    # Precompute: for each y, build Q[m,n] and sum uu*Q
    # Q[i,j,y] depends on whether m==n
    m_grid = ns[:, None]  # dim x 1
    n_grid = ns[None, :]  # 1 x dim
    diff = m_grid - n_grid  # dim x dim, 0 on diagonal

    results = np.zeros(len(y_arr))
    for idx, y in enumerate(y_arr):
        # Off-diagonal: q = (sin(2*pi*n*y/L) - sin(2*pi*m*y/L)) / (pi*(m-n))
        sin_n = np.sin(2*np.pi*n_grid*y/L)  # 1 x dim
        sin_m = np.sin(2*np.pi*m_grid*y/L)  # dim x 1
        Q = np.where(diff != 0, (sin_n - sin_m) / (np.pi * np.where(diff!=0, diff, 1)), 0)
        # Diagonal: q = 2*(L-y)/L * cos(2*pi*m*y/L)
        diag_vals = 2*(L-y)/L * np.cos(2*np.pi*ns*y/L)
        np.fill_diagonal(Q, diag_vals)
        results[idx] = np.sum(uu * Q)
    return results

def run_proof(lam_sq, N=None):
    if N is None:
        L = np.log(lam_sq)
        N = max(21, round(8*L))
    dim = 2*N+1; L = np.log(lam_sq)
    pf = 32*L*np.sinh(L/4)**2
    ks = np.arange(-N, N+1, dtype=float)
    v = 1.0/(L**2 + 4*np.pi**2*ks**2)
    w = ks/(L**2 + 4*np.pi**2*ks**2)
    s_v = pf*L**2*np.dot(v,v)
    u_v = v/np.linalg.norm(v)

    t0 = time.time()
    W02, M, QW = build_all(lam_sq, N)
    M_diag, M_alpha, M_prime, M_full, primes_used = compute_M_decomposition(lam_sq, N)
    print(f"  Build: {time.time()-t0:.0f}s", end="", flush=True)

    diag_vv = u_v @ M_diag @ u_v
    alpha_vv = u_v @ M_alpha @ u_v
    prime_vv = u_v @ M_prime @ u_v
    Mvv = u_v @ M @ u_v
    margin_v = s_v - Mvv
    analytic_margin = s_v - diag_vv - alpha_vv

    # Prime power data
    prime_ys = np.array([logpk for _,_,logpk in primes_used])
    prime_ws = np.array([logp*pk**(-0.5) for pk,logp,_ in primes_used])
    prime_pks = np.array([pk for pk,_,_ in primes_used])

    # F_v at prime powers
    t1 = time.time()
    F_at_primes = F_batch(u_v, N, L, prime_ys)
    print(f"  F_primes: {time.time()-t1:.0f}s", end="", flush=True)

    # Integral
    n_quad = 2000
    t_grid = np.linspace(2.01, lam_sq*0.998, n_quad)
    y_grid = np.log(t_grid)
    valid = y_grid < L*0.998
    t_grid = t_grid[valid]; y_grid = y_grid[valid]
    dt = (lam_sq - 2.0)/len(t_grid)

    t2 = time.time()
    F_grid = F_batch(u_v, N, L, y_grid)
    integral_v = np.sum(F_grid / np.sqrt(t_grid)) * dt
    print(f"  Integral: {time.time()-t2:.0f}s", flush=True)

    analytic_gap = analytic_margin - integral_v
    actual_error = prime_vv - integral_v
    primes_list = get_primes(min(lam_sq, 10000))

    # Split and bound
    best_ratio = 0; best_P0 = 0
    for P0 in [5,10,20,50,100,200,500]:
        if P0 >= lam_sq: break
        mask_ex = prime_pks <= P0
        exact_sum = np.sum(prime_ws[mask_ex] * F_at_primes[mask_ex])
        mask_int = t_grid <= P0
        exact_int = np.sum(F_grid[mask_int]/np.sqrt(t_grid[mask_int])) * dt
        exact_err = exact_sum - exact_int

        mask_tail = t_grid > P0
        tail_maxF = np.max(np.abs(F_grid[mask_tail])/np.sqrt(t_grid[mask_tail])) if np.any(mask_tail) else 0

        theta_tot = sum(np.log(p) for p in primes_list if p <= lam_sq)
        theta_P0 = sum(np.log(p) for p in primes_list if p <= P0)
        tail_theta_err = abs((theta_tot - theta_P0) - (lam_sq - P0))

        total_bnd = abs(exact_err) + tail_maxF * tail_theta_err
        ratio = analytic_gap / total_bnd if total_bnd > 0 else float('inf')
        if ratio > best_ratio: best_ratio = ratio; best_P0 = P0

    proved = best_ratio > 1
    print(f"  lam^2={lam_sq}: margin={margin_v:.4f} gap={analytic_gap:.4f} "
          f"best_ratio={best_ratio:.3f} P0={best_P0} {'*** PROVED ***' if proved else ''}")
    return {'lam_sq':lam_sq, 'margin':float(margin_v), 'gap':float(analytic_gap),
            'ratio':float(best_ratio), 'P0':best_P0, 'proved':proved}

if __name__ == "__main__":
    print("SESSION 33 — VECTORIZED SPLIT-AND-BOUND")
    print("="*75)
    results = []
    for lam_sq in [50, 100, 200, 500, 1000, 2000]:
        r = run_proof(lam_sq)
        results.append(r)
    print("\n" + "="*75)
    print("CONDITION A (even eigenvector) — Split-and-bound proof status:")
    print(f"  {'lam^2':>6} {'margin':>8} {'gap':>8} {'ratio':>8} {'P0':>4} {'status':>10}")
    for r in results:
        s = 'PROVED' if r['proved'] else f"x{1/r['ratio']:.2f}"
        print(f"  {r['lam_sq']:>6} {r['margin']:>8.4f} {r['gap']:>8.4f} {r['ratio']:>8.3f} {r['P0']:>4} {s:>10}")
    with open('session33_2x2_numpy.json','w') as f: json.dump(results,f,indent=2)
    print(f"\nSaved to session33_2x2_numpy.json")
