"""
SESSION 41f — VECTORIZED BARRIER COMPUTATION

Fast numpy-only computation of the barrier = <w, QW, w>.
Avoids mpmath entirely for the M_prime and W02 terms.
Uses vectorized operations for speed.

For the FULL barrier, we still need M_diag and M_alpha (which require mpmath).
But W02 - M_prime is the dominant balance and can be computed fast.
"""

import numpy as np
import time


def build_w_hat(lam_sq, N):
    """Build normalized odd eigenvector of W02."""
    L = np.log(lam_sq)
    ns = np.arange(-N, N + 1, dtype=float)
    w = ns / (L**2 + (4 * np.pi)**2 * ns**2)
    w[N] = 0.0  # n=0
    return w / np.linalg.norm(w), L, ns


def w02_rayleigh(w_hat, L, ns):
    """Compute <w_hat, W02, w_hat> via rank-2 decomposition."""
    N = (len(ns) - 1) // 2
    pf = 32 * L * np.sinh(L / 4)**2
    denom = L**2 + (4 * np.pi)**2 * ns**2

    u_tilde = 1.0 / denom
    w_tilde = ns / denom

    # <w_hat, u_tilde> = 0 by parity (w_hat odd, u_tilde even)
    # <w_hat, w_tilde> = ||w_tilde|| by construction (w_hat = w_tilde/||w_tilde||)
    wt_dot_wh = np.dot(w_tilde, w_hat)

    # <w, W02, w> = pf * L^2 * <w,u>^2 - pf * (4pi)^2 * <w,w_tilde>^2
    result = -pf * (4 * np.pi)**2 * wt_dot_wh**2
    return result


def mprime_rayleigh_vectorized(w_hat, lam_sq, L, ns):
    """
    Compute <w_hat, M_prime, w_hat> using vectorized numpy.

    M_prime[i,j] = sum_{p^k <= lam^2} log(p) * p^{-k/2} * q(n_i, n_j, log(p^k))

    <w, M_prime, w> = sum_{pk} weight(pk) * <w, Q_{pk}, w>

    where <w, Q_y, w> = sum_{i,j} w[i] * q(n_i, n_j, y) * w[j]

    Vectorize: precompute sin/cos arrays, then dot.
    """
    dim = len(ns)

    # Sieve primes
    limit = min(int(lam_sq), 10000)
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(limit**0.5) + 2):
        if i <= limit and sieve[i]:
            sieve[i*i::i] = False

    # Collect prime powers
    pk_list = []  # (pk, logp, logpk)
    for p in range(2, limit + 1):
        if not sieve[p]:
            continue
        pk = p
        k = 1
        logp = np.log(p)
        while pk <= lam_sq:
            pk_list.append((pk, logp, k * logp))
            pk *= p
            k += 1

    total = 0.0

    for pk, logp, logpk in pk_list:
        weight = logp * pk**(-0.5)
        y = logpk

        # Diagonal part: sum_n |w[n]|^2 * 2(L-y)/L * cos(2*pi*n*y/L)
        cos_arr = np.cos(2 * np.pi * ns * y / L)
        diag_contrib = 2 * (L - y) / L * np.sum(w_hat**2 * cos_arr)

        # Off-diagonal part: sum_{n!=m} w[n]*q(n,m,y)*w[m]
        # q(n,m,y) = (sin(2*pi*m*y/L) - sin(2*pi*n*y/L)) / (pi*(n-m))
        #
        # Full quadratic form including off-diag:
        # sum_{n,m} w[n]*q(n,m,y)*w[m]
        # = sum_n |w[n]|^2 * q(n,n,y) + sum_{n!=m} w[n]*q(n,m,y)*w[m]
        #
        # For off-diag, use: sum_{n!=m} w[n]*(sin_m - sin_n)/(pi*(n-m))*w[m]
        # = 2*sum_{n!=m} w[n]*sin_m*w[m]/(pi*(n-m))  [by symmetry argument]
        #
        # Vectorized: sin_m = sin(2*pi*m*y/L)
        sin_arr = np.sin(2 * np.pi * ns * y / L)
        ws = w_hat * sin_arr  # w[m] * sin(2*pi*m*y/L)

        # We need: sum_{n!=m} w[n] * sin_arr[m] * w_hat[m] / (pi * (n-m))
        #        = (1/pi) * sum_{n!=m} w_hat[n] * ws[m] / (n - m)
        #
        # This is like a discrete convolution / Hilbert-like operation.
        # Using the result: Off-diag = 2 * C1 where
        # C1 = sum_m ws[m] * sum_{n!=m} w_hat[n] / (pi*(n-m))
        #    = sum_m ws[m] * (Hw)[m]  where (Hw)[m] = (1/pi) sum_{n!=m} w[n]/(n-m)

        # Compute Hilbert-like transform of w_hat
        # (Hw)[m] = (1/pi) * sum_{n!=m} w_hat[n] / (n - m)
        # Vectorized using Toeplitz structure
        Hw = np.zeros(dim)
        for m_idx in range(dim):
            mask = np.arange(dim) != m_idx
            nm = ns[mask] - ns[m_idx]  # n - m for n != m
            Hw[m_idx] = np.sum(w_hat[mask] / nm) / np.pi

        off_diag_contrib = 2 * np.dot(ws, Hw)

        total += weight * (diag_contrib + off_diag_contrib)

    return total


def mprime_rayleigh_matrix(w_hat, lam_sq, L, ns):
    """
    Compute <w_hat, M_prime, w_hat> by building the full M_prime matrix.
    Vectorized matrix construction.
    """
    dim = len(ns)

    # Sieve
    limit = min(int(lam_sq), 10000)
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(limit**0.5) + 2):
        if i <= limit and sieve[i]:
            sieve[i*i::i] = False

    M_prime = np.zeros((dim, dim))

    # Precompute n-m matrix
    nm_diff = ns[:, None] - ns[None, :]  # [i,j] = n_i - n_j

    for p in range(2, limit + 1):
        if not sieve[p]:
            continue
        pk = p
        k = 1
        logp = np.log(p)
        while pk <= lam_sq:
            logpk = k * logp
            weight = logp * pk**(-0.5)
            y = logpk

            sin_arr = np.sin(2 * np.pi * ns * y / L)  # sin(2*pi*n*y/L)
            cos_arr = np.cos(2 * np.pi * ns * y / L)

            # Diagonal: q(n,n,y) = 2(L-y)/L * cos(2*pi*n*y/L)
            diag = 2 * (L - y) / L * cos_arr
            np.fill_diagonal(M_prime, M_prime.diagonal() + weight * diag)

            # Off-diagonal: q(n,m,y) = (sin_m - sin_n) / (pi*(n-m))
            sin_diff = sin_arr[None, :] - sin_arr[:, None]  # sin_m - sin_n
            # Avoid division by zero on diagonal
            with np.errstate(divide='ignore', invalid='ignore'):
                off_diag = sin_diff / (np.pi * nm_diff)
            np.fill_diagonal(off_diag, 0.0)  # diagonal handled separately

            M_prime += weight * off_diag

            pk *= p
            k += 1

    M_prime = (M_prime + M_prime.T) / 2
    return w_hat @ M_prime @ w_hat


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print()
    print('#' * 70)
    print('  SESSION 41f — VECTORIZED BARRIER COMPUTATION')
    print('#' * 70)

    # ── Part 1: Verify vectorized vs loop ──
    print('\n  PART 1: Verification (vectorized M_prime matrix vs loop)')
    print('  ' + '=' * 60)

    for lam_sq in [50, 200]:
        w_hat, L, ns = build_w_hat(lam_sq, max(15, round(6 * np.log(lam_sq))))
        t0 = time.time()
        v1 = mprime_rayleigh_vectorized(w_hat, lam_sq, L, ns)
        dt1 = time.time() - t0
        t0 = time.time()
        v2 = mprime_rayleigh_matrix(w_hat, lam_sq, L, ns)
        dt2 = time.time() - t0
        print(f'  lam^2={lam_sq}: loop={v1:+.8f} ({dt1:.2f}s)  '
              f'matrix={v2:+.8f} ({dt2:.2f}s)  diff={abs(v1-v2):.2e}')

    # ── Part 2: N-convergence (fast) ──
    print('\n\n  PART 2: N-convergence of W02 - M_prime')
    print('  ' + '=' * 60)

    for lam_sq in [1000, 5000, 10000]:
        L_f = np.log(lam_sq)
        N_base = max(15, round(6 * L_f))
        print(f'\n  lam^2={lam_sq}:')
        print(f'  {"N":>5s} {"dim":>5s} {"W02":>12s} {"M_prime":>12s} {"W02-Mp":>12s}')
        print('  ' + '-' * 55)

        for mult in [1.0, 1.5, 2.0, 3.0, 4.0]:
            N = round(mult * N_base)
            w_hat, L, ns_ = build_w_hat(lam_sq, N)
            t0 = time.time()
            w02 = w02_rayleigh(w_hat, L, ns_)
            mp = mprime_rayleigh_matrix(w_hat, lam_sq, L, ns_)
            dt = time.time() - t0
            print(f'  {N:>5d} {2*N+1:>5d} {w02:>+12.6f} {mp:>+12.6f} '
                  f'{w02-mp:>+12.6f}  ({dt:.1f}s)')

    # ── Part 3: Dense sweep (W02 - M_prime) ──
    print('\n\n  PART 3: Dense barrier sweep (W02 - M_prime only)')
    print('  ' + '=' * 60)

    lam_values = (list(range(50, 501, 50)) +
                  list(range(600, 2001, 100)) +
                  list(range(2500, 5001, 500)) +
                  list(range(6000, 20001, 1000)))

    print(f'\n  {"lam^2":>7s} {"L":>6s} {"W02":>10s} {"M_prime":>10s} {"W02-Mp":>10s}')
    print('  ' + '-' * 50)

    sweep_results = []
    for lam_sq in lam_values:
        N = max(15, round(6 * np.log(lam_sq)))
        w_hat, L, ns_ = build_w_hat(lam_sq, N)
        w02 = w02_rayleigh(w_hat, L, ns_)
        mp = mprime_rayleigh_matrix(w_hat, lam_sq, L, ns_)
        diff = w02 - mp
        sweep_results.append({'lam_sq': lam_sq, 'L': L, 'w02': w02, 'mp': mp, 'diff': diff})
        print(f'  {lam_sq:>7d} {L:>6.2f} {w02:>+10.4f} {mp:>+10.4f} {diff:>+10.6f}')

    # ── Part 4: Analysis ──
    print('\n\n  PART 4: Trend analysis of W02 - M_prime')
    print('  ' + '=' * 60)

    diffs = np.array([r['diff'] for r in sweep_results])
    Ls = np.array([r['L'] for r in sweep_results])

    print(f'  Range: [{diffs.min():.6f}, {diffs.max():.6f}]')
    print(f'  Mean:  {diffs.mean():.6f}')
    print(f'  Std:   {diffs.std():.6f}')

    # Fit: diff = a + b/L + c/L^2
    X = np.column_stack([np.ones_like(Ls), 1/Ls, 1/Ls**2])
    coeffs = np.linalg.lstsq(X, diffs, rcond=None)[0]
    print(f'\n  Fit: W02-Mp = {coeffs[0]:.6f} + {coeffs[1]:.4f}/L + {coeffs[2]:.4f}/L^2')
    print(f'  Asymptotic (L->inf): {coeffs[0]:.6f}')

    # Check residuals
    resid = diffs - X @ coeffs
    print(f'  Residual std: {resid.std():.6f}')

    # Linear fit
    c_lin = np.polyfit(Ls, diffs, 1)
    print(f'  Linear fit: {c_lin[0]:+.6f} * L + {c_lin[1]:+.6f}')

    print('\n' + '#' * 70)
    print('  SESSION 41f COMPLETE')
    print('#' * 70)
