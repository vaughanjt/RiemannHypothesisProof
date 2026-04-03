"""
SESSION 41g — UNCAPPED BARRIER COMPUTATION

Fix: remove the 10000 prime cap. Include ALL primes up to lam^2.
This allows reliable barrier computation beyond lam^2 = 10000.

Uses numpy vectorized matrix construction for speed.
Computes W02 - M_prime (the dominant balance).
"""

import numpy as np
import time


def sieve_primes(limit):
    """Sieve of Eratosthenes up to limit. Returns list of primes."""
    if limit < 2:
        return []
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[:2] = False
    for i in range(2, int(limit**0.5) + 2):
        if i <= limit and is_prime[i]:
            is_prime[i*i::i] = False
    return np.where(is_prime)[0]


def compute_barrier_partial(lam_sq, N=None):
    """
    Compute W02 - M_prime Rayleigh quotient on w direction.
    NO prime cap — includes ALL primes up to lam_sq.
    """
    L = np.log(lam_sq)
    if N is None:
        N = max(15, round(6 * L))
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)

    # w_hat (normalized odd eigenvector of W02)
    w = ns / (L**2 + (4 * np.pi)**2 * ns**2)
    w[N] = 0.0
    w_hat = w / np.linalg.norm(w)

    # W02 Rayleigh quotient (exact for truncated basis)
    pf = 32 * L * np.sinh(L / 4)**2
    denom = L**2 + (4 * np.pi)**2 * ns**2
    w_tilde = ns / denom
    wt_dot_wh = np.dot(w_tilde, w_hat)
    w02_rq = -pf * (4 * np.pi)**2 * wt_dot_wh**2

    # M_prime: include ALL primes up to lam_sq (NO CAP)
    primes = sieve_primes(int(lam_sq))
    n_primes = len(primes)

    # Collect prime powers
    pk_data = []  # (logp, weight, y=logpk)
    for p in primes:
        pk = int(p)
        k = 1
        logp = np.log(p)
        while pk <= lam_sq:
            pk_data.append((logp, logp * pk**(-0.5), k * logp))
            pk *= int(p)
            k += 1

    # Build M_prime matrix (vectorized per prime power)
    M_prime = np.zeros((dim, dim))
    nm_diff = ns[:, None] - ns[None, :]  # [i,j] = n_i - n_j

    for logp, weight, y in pk_data:
        sin_arr = np.sin(2 * np.pi * ns * y / L)
        cos_arr = np.cos(2 * np.pi * ns * y / L)

        # Diagonal
        diag = 2 * (L - y) / L * cos_arr
        np.fill_diagonal(M_prime, M_prime.diagonal() + weight * diag)

        # Off-diagonal
        sin_diff = sin_arr[None, :] - sin_arr[:, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            off_diag = sin_diff / (np.pi * nm_diff)
        np.fill_diagonal(off_diag, 0.0)
        M_prime += weight * off_diag

    M_prime = (M_prime + M_prime.T) / 2
    mp_rq = float(w_hat @ M_prime @ w_hat)

    return {
        'lam_sq': lam_sq,
        'L': L,
        'N': N,
        'dim': dim,
        'n_primes': n_primes,
        'n_prime_powers': len(pk_data),
        'w02': float(w02_rq),
        'mprime': float(mp_rq),
        'partial_barrier': float(w02_rq - mp_rq),
    }


if __name__ == '__main__':
    print()
    print('#' * 70)
    print('  SESSION 41g — UNCAPPED BARRIER (ALL PRIMES)')
    print('#' * 70)

    # ── Part 1: Verify against capped version ──
    print('\n  PART 1: Verify uncapped vs capped (should match for lam^2 <= 10000)')
    print('  ' + '=' * 60)

    for lam_sq in [200, 1000, 5000, 10000]:
        r = compute_barrier_partial(lam_sq)
        print(f'  lam^2={lam_sq:>6d}  W02-Mp={r["partial_barrier"]:+.6f}  '
              f'primes={r["n_primes"]}  powers={r["n_prime_powers"]}')

    # ── Part 2: Push beyond 10000 with ALL primes ──
    print('\n\n  PART 2: Extended barrier sweep (ALL primes included)')
    print('  ' + '=' * 60)

    lam_values = [500, 1000, 2000, 3000, 5000, 7000, 10000,
                  12000, 15000, 20000, 30000, 50000, 70000, 100000]

    print(f'\n  {"lam^2":>8s} {"L":>6s} {"N":>4s} {"primes":>7s} '
          f'{"W02":>12s} {"M_prime":>12s} {"W02-Mp":>12s} {"time":>6s}')
    print('  ' + '-' * 80)

    results = []
    for lam_sq in lam_values:
        t0 = time.time()
        r = compute_barrier_partial(lam_sq)
        dt = time.time() - t0
        results.append(r)
        print(f'  {lam_sq:>8d} {r["L"]:>6.2f} {r["N"]:>4d} {r["n_primes"]:>7d} '
              f'{r["w02"]:>+12.4f} {r["mprime"]:>+12.4f} '
              f'{r["partial_barrier"]:>+12.6f} {dt:>5.1f}s')

    # ── Part 3: Growth analysis ──
    print('\n\n  PART 3: Growth analysis of W02 - M_prime')
    print('  ' + '=' * 60)

    Ls = np.array([r['L'] for r in results])
    pbs = np.array([r['partial_barrier'] for r in results])
    w02s = np.array([r['w02'] for r in results])
    mps = np.array([r['mprime'] for r in results])

    # Log-slopes
    print(f'\n  Log-slopes (between consecutive points):')
    for i in range(1, len(results)):
        dL = Ls[i] - Ls[i-1]
        d_pb = pbs[i] - pbs[i-1]
        slope_w02 = (np.log(abs(w02s[i])) - np.log(abs(w02s[i-1]))) / dL if dL > 0 else 0
        slope_mp = (np.log(abs(mps[i])) - np.log(abs(mps[i-1]))) / dL if dL > 0 else 0
        print(f'  [{results[i-1]["lam_sq"]:>6d} -> {results[i]["lam_sq"]:>6d}] '
              f'W02-Mp={pbs[i]:+.6f}  delta={d_pb:+.6f}  '
              f'|W02| slope={slope_w02:.4f}  |Mp| slope={slope_mp:.4f}')

    # Fit: partial_barrier = a + b*L + c*L^2
    if len(Ls) >= 3:
        X = np.column_stack([np.ones_like(Ls), Ls, Ls**2])
        c = np.linalg.lstsq(X, pbs, rcond=None)[0]
        print(f'\n  Quadratic fit: W02-Mp = {c[0]:.4f} + {c[1]:.4f}*L + {c[2]:.6f}*L^2')

        # Linear fit
        c_lin = np.polyfit(Ls, pbs, 1)
        print(f'  Linear fit:    W02-Mp = {c_lin[0]:.4f}*L + {c_lin[1]:.4f}')

        # Does it converge?
        if c[2] > 0:
            print(f'  => Accelerating growth (quadratic term positive)')
        elif c[2] < 0 and c[1] > 0:
            print(f'  => Decelerating growth (approaches: {c[0] - c[1]**2/(4*c[2]):.4f})')
        else:
            print(f'  => Linear growth, no convergence to constant')

    # ── Part 4: N-convergence at large lambda ──
    print('\n\n  PART 4: N-convergence at large lambda^2')
    print('  ' + '=' * 60)

    for lam_sq in [10000, 50000, 100000]:
        L_f = np.log(lam_sq)
        N_base = max(15, round(6 * L_f))
        print(f'\n  lam^2={lam_sq}, L={L_f:.3f}:')
        print(f'  {"N":>5s} {"W02-Mp":>12s} {"primes":>7s}')
        print('  ' + '-' * 28)

        for mult in [1.0, 1.5, 2.0]:
            N = round(mult * N_base)
            t0 = time.time()
            r = compute_barrier_partial(lam_sq, N=N)
            dt = time.time() - t0
            print(f'  {N:>5d} {r["partial_barrier"]:>+12.6f} {r["n_primes"]:>7d}  ({dt:.1f}s)')

    print('\n' + '#' * 70)
    print('  SESSION 41g COMPLETE')
    print('#' * 70)
