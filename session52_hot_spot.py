"""
SESSION 52 -- MARGIN-DRAIN HOT SPOT STRUCTURE

Session 51 found the minimum gap (margin - |drain|) at L ~ 4.68, lam^2 ~ 108.
Initial observation: the six local minima from Session 51 all occur at L
values just after a prime enters the sieve:

    L = 4.68 (gap 0.017) ~  log(107) + 0.007   <- tightest
    L = 4.30 (gap 0.033) ~  log(73)  + 0.010
    L = 6.46 (gap 0.024) ~  log(631) + 0.013
    L = 3.50 (gap 0.063) ~  log(31)  + 0.066
    L = 2.90 (gap 0.054) ~  log(17)  + 0.067
    L = 2.50 (gap 0.062) ~  log(11)  + 0.102

The tightest hot spots happen within 0.01-0.02 of a prime entry. Looser
dips happen further out. This suggests the drain has a discontinuous jump
when exp(L) crosses a prime, followed by decay as the PNT smooth integral
catches up.

This session verifies the hypothesis and uses it:

  1. Fine scan around L = log(107) = 4.6728 at dL = 0.001 to see the jump
  2. Per-prime decomposition at the peak to identify the dominant contributor
  3. Scan a range of primes (log(p) +- delta) to map the prime-entry bump profile
  4. Use the predicted hot-spot structure to search for the GLOBAL minimum
     gap (not just from the coarse Session 51 scan but from every L = log(p))
  5. Compare: is the pattern universal? Can we bound the drain at hot spots
     using the prime's entry weight (log p)/sqrt(p)?

If confirmed: the margin-drain proof problem reduces from "bound over
continuous L" to "bound at a discrete set of L = log(p_k) entry points".
"""

import sys
import time

import numpy as np

sys.path.insert(0, '.')
from session41g_uncapped_barrier import sieve_primes, compute_barrier_partial
from session42j_margin_vs_drain import mprime_pnt_integral
from session49c_weil_residual import build_all_fast


def drain_at_L(L_val, N=None):
    """Fast drain-only evaluator (skips mdiag_malpha for speed)."""
    lam_sq = int(round(np.exp(L_val)))
    if lam_sq < 2:
        lam_sq = 2
    if N is None:
        N = max(15, round(6 * L_val))
    r = compute_barrier_partial(lam_sq, N)
    actual_mp = r['mprime']
    pnt_mp = mprime_pnt_integral(lam_sq, N)
    return actual_mp - pnt_mp, lam_sq, N


def margin_and_drain_at_L(L_val, N=None):
    """
    Fast (margin, drain) evaluator using build_all_fast from Session 49c.
    Decomposition:
      margin = W02_rq - PNT_mp - M_other_rq
      drain  = M_prime_rq - PNT_mp
    where M_other = M - M_prime = wr_diag + alpha_offdiag (archimedean).
    """
    lam_sq = int(round(np.exp(L_val)))
    if lam_sq < 2:
        lam_sq = 2
    if N is None:
        N = max(15, round(6 * L_val))

    # W02_rq and M_prime_rq via session41g (fast vectorized)
    r = compute_barrier_partial(lam_sq, N)
    w02_rq = r['w02']
    mprime_rq = r['mprime']

    # Full M matrix via build_all_fast; extract M_other via quadratic form
    L_f = float(np.log(lam_sq))
    _, M_full, _ = build_all_fast(lam_sq, N)
    ns = np.arange(-N, N + 1, dtype=float)
    w = ns / (L_f**2 + (4 * np.pi)**2 * ns**2)
    w[N] = 0.0
    w_hat = w / np.linalg.norm(w)
    M_rq = float(w_hat @ M_full @ w_hat)
    M_other_rq = M_rq - mprime_rq  # wr_diag + alpha_offdiag contribution

    pnt_mp = mprime_pnt_integral(lam_sq, N)

    margin = w02_rq - pnt_mp - M_other_rq
    drain = mprime_rq - pnt_mp
    return margin, drain, lam_sq, N


def per_prime_contribution(p, lam_sq, N):
    """
    Compute one prime's contribution to M_prime Rayleigh quotient on the
    conjugate Poisson kernel. Returns a scalar.
    """
    L = np.log(lam_sq)
    ns = np.arange(-N, N + 1, dtype=float)
    w = ns / (L**2 + (4*np.pi)**2 * ns**2)
    w[N] = 0.0
    w_hat = w / np.linalg.norm(w)
    nm_diff = ns[:, None] - ns[None, :]

    M = np.zeros((2*N + 1, 2*N + 1))
    pk = int(p)
    logp = np.log(p)
    while pk <= lam_sq:
        weight = logp * pk**(-0.5)
        yk = np.log(pk)
        sin_arr = np.sin(2*np.pi*ns*yk/L)
        cos_arr = np.cos(2*np.pi*ns*yk/L)
        diag = 2*(L - yk)/L * cos_arr
        np.fill_diagonal(M, M.diagonal() + weight*diag)
        sin_diff = sin_arr[None, :] - sin_arr[:, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            off = sin_diff / (np.pi * nm_diff)
        np.fill_diagonal(off, 0.0)
        M += weight * off
        pk *= int(p)

    M = (M + M.T) / 2
    return float(w_hat @ M @ w_hat)


# ==========================================================================
#  Step 1: Fine scan around log(107)
# ==========================================================================

def step1_fine_scan_107():
    print()
    print('=' * 76)
    print('  STEP 1: Fine scan around L = log(107) = 4.6728')
    print('=' * 76)
    print('  Testing: does drain jump when exp(L) crosses a prime?')
    print()

    log_107 = np.log(107)
    log_109 = np.log(109)
    print(f'  log(103) = {np.log(103):.6f}')
    print(f'  log(107) = {log_107:.6f}')
    print(f'  log(109) = {log_109:.6f}')
    print()

    # Very fine scan right around log(107): before, at, after
    L_values = np.concatenate([
        np.linspace(4.625, 4.670, 10),      # before 107 enters
        np.linspace(4.672, 4.680, 9),       # right after 107
        np.linspace(4.685, 4.720, 8),       # decay phase
    ])
    L_values = np.sort(L_values)

    print(f'  {"L":>9} {"lam^2":>8} {"drain":>12} {"primes_in":>10} {"note":>20}')
    print('  ' + '-' * 62)
    sys.stdout.flush()

    prev_n_primes = None
    results = []
    for L in L_values:
        t0 = time.time()
        drain, lam_sq, N = drain_at_L(float(L))
        n_primes = len(sieve_primes(lam_sq))
        note = ''
        if prev_n_primes is not None and n_primes != prev_n_primes:
            note = f'+{n_primes - prev_n_primes} prime(s) added'
        prev_n_primes = n_primes
        print(f'  {L:9.5f} {lam_sq:8d} {drain:+12.6f} {n_primes:10d} {note:>20}',
              flush=True)
        results.append((float(L), lam_sq, drain, n_primes))

    return results


# ==========================================================================
#  Step 2: Per-prime decomposition at L = 4.68
# ==========================================================================

def step2_per_prime_decomposition():
    print()
    print('=' * 76)
    print('  STEP 2: Per-prime decomposition at L = 4.68 (peak drain)')
    print('=' * 76)
    print()

    L_val = 4.68
    lam_sq = int(round(np.exp(L_val)))
    N = max(15, round(6 * L_val))
    print(f'  L = {L_val}, lam_sq = {lam_sq}, N = {N}')

    primes = list(sieve_primes(lam_sq))
    print(f'  {len(primes)} primes in sieve')
    print()

    contribs = []
    for p in primes:
        c = per_prime_contribution(int(p), lam_sq, N)
        contribs.append((int(p), c))

    contribs.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f'  Top 15 contributors (by |contribution|):')
    print(f'  {"p":>5} {"logp/sqrt(p)":>14} {"contribution":>14} {"cumulative":>14}')
    print('  ' + '-' * 54)
    total = sum(c for _, c in contribs)
    cum = 0.0
    top_sum = 0.0
    for i, (p, c) in enumerate(contribs[:15]):
        cum += c
        top_sum += c
        weight = np.log(p) / np.sqrt(p)
        print(f'  {p:>5} {weight:>14.6f} {c:>+14.6f} {cum:>+14.6f}')

    print()
    print(f'  Total M_prime sum (direct)  = {total:+.6f}')
    print(f'  Top 15 alone                = {top_sum:+.6f}  ({100*top_sum/total:+.1f}%)')

    # The "newest" prime is 107 — check its contribution rank
    p107_rank = None
    for i, (p, _) in enumerate(contribs):
        if p == 107:
            p107_rank = i + 1
            break
    if p107_rank is not None:
        p_val, c_val = contribs[p107_rank - 1]
        print()
        print(f'  p = 107 (just entered sieve): rank {p107_rank}, '
              f'contribution {c_val:+.6f}')

    return contribs


# ==========================================================================
#  Step 3: Prime-entry bump profile for several primes
# ==========================================================================

def step3_prime_entry_profiles():
    print()
    print('=' * 76)
    print('  STEP 3: Drain profile across prime entries (structural confirmation)')
    print('=' * 76)
    print()

    # For each test prime p, scan L in [log(p) - 0.02, log(p) + 0.05]
    # and measure the drain. If the hypothesis is right, there should be
    # a discontinuous jump at L = log(p).
    test_primes = [31, 73, 107, 151, 251, 503]

    for p in test_primes:
        Lp = np.log(p)
        print(f'  Prime p = {p}, log(p) = {Lp:.4f}')
        print(f'  {"L":>9} {"lam_sq":>8} {"drain":>12} {"delta_drain":>14}')
        print('  ' + '-' * 50)

        L_window = np.array([Lp - 0.020, Lp - 0.005, Lp - 0.001,
                             Lp + 0.001, Lp + 0.005, Lp + 0.015, Lp + 0.030])
        prev_drain = None
        for L in L_window:
            drain, lam_sq, N = drain_at_L(float(L))
            delta = '' if prev_drain is None else f'{drain - prev_drain:+.6f}'
            print(f'  {L:9.5f} {lam_sq:8d} {drain:+12.6f} {delta:>14}',
                  flush=True)
            prev_drain = drain
        print()

    return test_primes


# ==========================================================================
#  Step 4: Search for the global minimum gap over primes p = 2..10000
# ==========================================================================

def step4_global_hot_spot_search():
    print()
    print('=' * 76)
    print('  STEP 4: Global hot-spot search over primes p <= 10000')
    print('=' * 76)
    print()
    print('  For each prime p in [11, 10000], compute (margin, drain) at')
    print('  L = log(p) + 0.008 (near the observed peak offset). Find the')
    print('  smallest gap and check whether it undercuts Session 51 scan.')
    print()

    # Use primes spaced enough apart to give meaningful hot-spot samples
    primes_all = list(sieve_primes(10000))
    # Subsample to avoid huge compute: every prime up to 500, then every 5th
    test_primes = [p for p in primes_all if p <= 500] + primes_all[primes_all.index(503)::5]
    test_primes = sorted(set(int(p) for p in test_primes))
    print(f'  Testing {len(test_primes)} primes')
    print()
    print(f'  {"p":>6} {"L":>9} {"margin":>12} {"|drain|":>10} {"gap":>10} '
          f'{"min so far":>12}')
    print('  ' + '-' * 68)
    sys.stdout.flush()

    min_gap = float('inf')
    min_gap_p = None
    min_gap_L = None
    offset = 0.008
    t_total = time.time()
    results = []
    for p in test_primes:
        L_val = float(np.log(p) + offset)
        margin, drain, lam_sq, N = margin_and_drain_at_L(L_val)
        gap = margin - abs(drain)
        if gap < min_gap:
            min_gap = gap
            min_gap_p = p
            min_gap_L = L_val
        results.append((p, L_val, margin, drain, gap))
        if p in (11, 31, 73, 107, 251, 503, 1009, 2003, 5003, 9973) or gap < 0.02:
            print(f'  {p:>6d} {L_val:>9.4f} {margin:>+12.6f} {abs(drain):>10.6f} '
                  f'{gap:>+10.6f} {min_gap:>+12.6f}', flush=True)

    print()
    print(f'  Scan done in {time.time() - t_total:.1f}s')
    print()
    print(f'  GLOBAL MINIMUM GAP: {min_gap:+.6f}')
    print(f'    at p = {min_gap_p}, L = {min_gap_L:.4f}, lam_sq = {int(round(np.exp(min_gap_L)))}')
    print(f'  Session 51 scan minimum: +0.01679 at L = 4.680')
    print(f'  Difference: {min_gap - 0.01679:+.6f} '
          f'({"new hot spot tighter" if min_gap < 0.01679 else "Session 51 minimum still holds"})')

    return results, min_gap, min_gap_p, min_gap_L


def run():
    print()
    print('#' * 76)
    print('  SESSION 52 -- MARGIN-DRAIN HOT SPOT STRUCTURE')
    print('#' * 76)

    step1_results = step1_fine_scan_107()
    step2_contribs = step2_per_prime_decomposition()
    step3_prime_entry_profiles()
    step4_results = step4_global_hot_spot_search()

    print()
    print('=' * 76)
    print('  SYNTHESIS')
    print('=' * 76)
    print()
    print('  Hypothesis: drain(L) has a discontinuous jump at each L = log(p)')
    print('  as a new prime enters the sieve, followed by decay. The tightest')
    print('  margin-drain gaps occur at L ~ log(p) + small_offset for each prime.')
    print()
    print('  If confirmed, the continuous-L proof problem reduces to a discrete')
    print('  bound over the countable set {log(p) : p prime}.')
    print()


if __name__ == '__main__':
    run()
