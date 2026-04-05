"""
SESSION 54 — CRAMÉR CONCENTRATION PROBE

Question: Under the Cramér random model (integers selected with
probability 1/log(n), count-matched to actual prime count), does
the drain concentrate tightly enough that |drain_Cramér(L)| < margin(L)
holds with probability -> 1 as L grows?

If yes: the proof path via equidistribution/large-sieve is alive —
real primes only need to satisfy a generic deviation bound from Cramér.

If no (fat tails even at large L): the margin-drain approach is
fundamentally dead, because even "generic" prime-like sequences
violate the bound with positive probability.

Step 1: Monte Carlo — sample drain_Cramér(L) at L ∈ {3, 4.68, 6.5, 8, 10}
Step 2: Variance scaling — compute Var(drain_Cramér) as function of L
        Rough prediction: Var ~ L (sum of ~exp(L)/L independent terms,
        each of variance ~L^2/exp(L)), so std ~ sqrtL.  Margin is ~constant.
        If margin/std ~ 1/sqrtL -> concentration WEAKENS with L -> KILL.

Protocol:
  - Precompute "response function" R(y) so M_prime Rayleigh quotient
    for a prime list is just Σ weight_k * R(y_k).  O(dim) per prime
    instead of O(dim^2). Enables 2000 trials/L in seconds.
  - Validate fast Rayleigh against matrix computation at start.
  - Report: drain distribution, P(|drain|>margin), margin/std vs L.
"""

import sys
import time

import numpy as np

sys.path.insert(0, '.')
from session41g_uncapped_barrier import sieve_primes, compute_barrier_partial
from session42j_margin_vs_drain import mprime_pnt_integral
from session49c_weil_residual import build_all_fast


# ==============================================================
#  FAST RAYLEIGH QUOTIENT VIA PRECOMPUTED RESPONSE FUNCTION
# ==============================================================

def precompute_response(L, N):
    """
    Precompute response coefficients so the M_prime Rayleigh quotient
    contribution of a prime power at log-height y is:

        R(y) = 2*(L-y)/L * (v_sq · cos(2pi ns y/L))
             + (q · sin(2pi ns y/L))

    where v_sq = v_hat^2, and q = -2 * v_hat * D with
    D[i] = Σ_{j≠i} v_hat[j] / (pi(i-j)).

    Returns (ns, v_hat, v_sq, q) — all shape (2N+1,).
    """
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)
    v = ns / (L ** 2 + (4 * np.pi) ** 2 * ns ** 2)
    v[N] = 0.0
    v_hat = v / np.linalg.norm(v)
    v_sq = v_hat ** 2

    idx = np.arange(dim, dtype=float)
    idx_diff = idx[:, None] - idx[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        kernel = 1.0 / (np.pi * idx_diff)
    np.fill_diagonal(kernel, 0.0)
    D = kernel @ v_hat
    q = -2.0 * v_hat * D

    return ns, v_hat, v_sq, q


def mp_rayleigh_fast(prime_list, lam_sq, L, N, ns, v_sq, q):
    """
    M_prime Rayleigh quotient from a prime list, O(n_primes * dim)
    instead of O(n_primes * dim^2).
    """
    prime_arr = np.asarray(prime_list, dtype=float)
    log_p = np.log(prime_arr)
    two_pi_over_L = 2.0 * np.pi / L

    # k=1 terms (vectorised over all primes)
    y_1 = log_p
    w_1 = log_p / np.sqrt(prime_arr)
    phase = two_pi_over_L * ns[None, :] * y_1[:, None]   # (n_p, dim)
    diag_c = 2.0 * (L - y_1) / L * (np.cos(phase) @ v_sq)
    off_c = np.sin(phase) @ q
    total = float(w_1 @ (diag_c + off_c))

    # k >= 2 corrections (scalar loop, few entries)
    sqrt_lam = np.sqrt(lam_sq)
    for p, lp in zip(prime_arr, log_p):
        if p > sqrt_lam:
            continue
        pk = p * p
        k = 2
        while pk <= lam_sq:
            y = k * lp
            w = lp * pk ** (-0.5)
            ph = two_pi_over_L * ns * y
            R = 2.0 * (L - y) / L * np.dot(v_sq, np.cos(ph)) + np.dot(q, np.sin(ph))
            total += w * R
            pk *= p
            k += 1

    return total


# ==============================================================
#  CRAMÉR SAMPLER
# ==============================================================

def make_cramer_weights(lam_sq):
    """Precompute weight array for Cramér sampling from [2, lam_sq]."""
    n_arr = np.arange(2, int(lam_sq) + 1)
    w = 1.0 / np.log(n_arr.astype(float))
    w /= w.sum()
    return n_arr, w


def cramer_sample(n_target, n_arr, weights, rng):
    """Draw n_target integers without replacement, 1/log(n) weighting."""
    idx = rng.choice(len(n_arr), size=n_target, replace=False, p=weights)
    return n_arr[idx]  # unsorted — fine for mp_rayleigh_fast


# ==============================================================
#  MARGIN COMPUTATION (barrier + drain, no mpmath needed)
# ==============================================================

def margin_at_L(L_val, N=None):
    """
    margin(L) = barrier(L) + drain(L)
              = (W02 - M_total)·v + (M_prime_actual - PNT_integral)
    Returns margin, drain, barrier, pnt_mp, lam_sq, N.
    """
    lam_sq = max(2, int(round(np.exp(L_val))))
    if N is None:
        N = max(15, round(6 * L_val))
    L_eff = np.log(lam_sq)

    # barrier from build_all_fast
    W02, M_full, QW = build_all_fast(lam_sq, N)
    ns = np.arange(-N, N + 1, dtype=float)
    v = ns / (L_eff ** 2 + (4 * np.pi) ** 2 * ns ** 2)
    v[N] = 0.0
    v_hat = v / np.linalg.norm(v)
    barrier = float(v_hat @ QW @ v_hat)

    # actual M_prime
    r = compute_barrier_partial(lam_sq, N)
    mp_actual = r['mprime']

    # PNT integral
    pnt_mp = mprime_pnt_integral(lam_sq, N)

    drain = mp_actual - pnt_mp
    margin = barrier + drain
    return margin, drain, barrier, pnt_mp, lam_sq, N


# ==============================================================
#  MAIN
# ==============================================================

def run():
    print()
    print('#' * 76)
    print('  SESSION 54 — CRAMÉR CONCENTRATION PROBE')
    print('#' * 76)

    rng = np.random.default_rng(seed=54)

    # -- Validation -----------------------------------------------
    print('\n  === VALIDATION: fast Rayleigh vs matrix computation ===')
    for test_lam in [200, 1000, 5000]:
        L_t = np.log(test_lam)
        N_t = max(15, round(6 * L_t))
        ns_t, vh_t, vsq_t, q_t = precompute_response(L_t, N_t)
        primes_t = sieve_primes(test_lam)
        mp_fast = mp_rayleigh_fast(primes_t, test_lam, L_t, N_t, ns_t, vsq_t, q_t)
        mp_matrix = compute_barrier_partial(test_lam, N_t)['mprime']
        print(f'  lam^2={test_lam:>5d}  fast={mp_fast:+.10f}  '
              f'matrix={mp_matrix:+.10f}  diff={abs(mp_fast-mp_matrix):.2e}')
    sys.stdout.flush()

    # -- Step 0: margins and actual drains -----------------------
    L_targets = [3.0, 4.68, 6.5, 8.0, 10.0]

    print('\n  === STEP 0: MARGINS AND ACTUAL DRAINS ===')
    print(f'  {"L":>6} {"lam^2":>8} {"margin":>12} {"drain":>12} '
          f'{"barrier":>12} {"gap":>12}')
    print('  ' + '-' * 70)

    L_data = {}
    for L_val in L_targets:
        t0 = time.time()
        mg, dr, br, pnt, lsq, N = margin_at_L(L_val)
        dt = time.time() - t0
        gap = mg - abs(dr)
        L_data[L_val] = dict(margin=mg, drain=dr, barrier=br,
                             pnt_mp=pnt, lam_sq=lsq, N=N)
        print(f'  {L_val:6.2f} {lsq:>8d} {mg:+12.6f} {dr:+12.6f} '
              f'{br:+12.6f} {gap:+12.6f}  ({dt:.1f}s)')
    sys.stdout.flush()

    # -- Step 1: Monte Carlo drain distribution ------------------
    n_trials = 2000

    print(f'\n  === STEP 1: MONTE CARLO — {n_trials} FULL-CRAMÉR TRIALS PER L ===')
    print(f'  (K = 0: ALL primes randomised, count-matched to pi(lambda^2))')

    mc_results = {}
    for L_val in L_targets:
        d = L_data[L_val]
        lam_sq = d['lam_sq']
        N = d['N']
        margin = d['margin']
        pnt_mp = d['pnt_mp']
        actual_drain = d['drain']
        L_eff = np.log(lam_sq)

        n_actual = len(sieve_primes(lam_sq))
        n_arr, w_arr = make_cramer_weights(lam_sq)
        ns_r, vh_r, vsq_r, q_r = precompute_response(L_eff, N)

        print(f'\n  L = {L_val:.2f}  (lambda^2 = {lam_sq}, pi(lambda^2) = {n_actual}, '
              f'margin = {margin:+.6f})')
        t0 = time.time()

        trial_drains = np.empty(n_trials)
        for i in range(n_trials):
            fakes = cramer_sample(n_actual, n_arr, w_arr, rng)
            mp_c = mp_rayleigh_fast(fakes, lam_sq, L_eff, N, ns_r, vsq_r, q_r)
            trial_drains[i] = mp_c - pnt_mp

        dt = time.time() - t0
        mean_d = trial_drains.mean()
        std_d = trial_drains.std()
        tail = np.mean(np.abs(trial_drains) > abs(margin))
        p5, p25, p50, p75, p95 = np.percentile(trial_drains, [5, 25, 50, 75, 95])
        max_abs = np.max(np.abs(trial_drains))

        mc_results[L_val] = dict(mean=mean_d, std=std_d, tail=tail,
                                 trial_drains=trial_drains)

        print(f'    Time:     {dt:.1f}s  ({dt/n_trials*1000:.1f} ms/trial)')
        print(f'    Mean:     {mean_d:+.6f}   (actual drain: {actual_drain:+.6f})')
        print(f'    Std:      {std_d:.6f}')
        print(f'    5/25/50/75/95 percentiles:  {p5:+.4f} / {p25:+.4f} / '
              f'{p50:+.4f} / {p75:+.4f} / {p95:+.4f}')
        print(f'    Max |drain|:  {max_abs:.6f}')
        print(f'    Margin:       {abs(margin):.6f}')
        print(f'    P(|drain| > margin):  {tail:.4f}  '
              f'({int(tail * n_trials)}/{n_trials})')
        print(f'    margin / std:  {abs(margin) / std_d:.3f} sigma')
        sys.stdout.flush()

    # -- Step 2: variance scaling --------------------------------
    print(f'\n  === STEP 2: VARIANCE vs L  (500 trials per point) ===')
    print(f'  Prediction if Var ~ L: std ~ sqrtL, margin ~ const -> margin/std -> 0')
    print()

    L_fine = np.arange(2.5, 12.5, 0.5)
    n_var = 500

    print(f'  {"L":>6} {"lam^2":>8} {"pi(lambda^2)":>7} {"std":>12} {"margin":>12} '
          f'{"m/sigma":>8} {"P(tail)":>8}')
    print('  ' + '-' * 74)

    scan = []
    for L_val in L_fine:
        lam_sq_v = max(2, int(round(np.exp(L_val))))
        N_v = max(15, round(6 * L_val))
        L_eff_v = np.log(lam_sq_v)

        mg_v, dr_v, br_v, pnt_v, _, _ = margin_at_L(L_val, N_v)
        n_actual_v = len(sieve_primes(lam_sq_v))
        n_arr_v, w_arr_v = make_cramer_weights(lam_sq_v)
        ns_v, vh_v, vsq_v, q_v = precompute_response(L_eff_v, N_v)

        t0 = time.time()
        td_v = np.empty(n_var)
        for i in range(n_var):
            fk = cramer_sample(n_actual_v, n_arr_v, w_arr_v, rng)
            mp_c = mp_rayleigh_fast(fk, lam_sq_v, L_eff_v, N_v, ns_v, vsq_v, q_v)
            td_v[i] = mp_c - pnt_v

        std_v = td_v.std()
        tail_v = np.mean(np.abs(td_v) > abs(mg_v))
        sig_v = abs(mg_v) / std_v if std_v > 1e-15 else float('inf')
        dt = time.time() - t0

        scan.append((float(L_val), lam_sq_v, n_actual_v,
                      std_v, mg_v, sig_v, tail_v))
        print(f'  {L_val:6.1f} {lam_sq_v:>8d} {n_actual_v:>7d} {std_v:12.6f} '
              f'{mg_v:+12.6f} {sig_v:8.3f} {tail_v:8.4f}  ({dt:.1f}s)',
              flush=True)

    scan = np.array(scan)  # columns: L, lam^2, pi, std, margin, m/sigma, P(tail)

    # -- Fits ----------------------------------------------------
    print('\n  === SCALING FITS ===')

    Ls = scan[:, 0]
    stds = scan[:, 3]
    sigs = scan[:, 5]

    # Fit std vs L: std = a * L^b
    mask = stds > 0
    log_L = np.log(Ls[mask])
    log_std = np.log(stds[mask])
    b_fit, log_a = np.polyfit(log_L, log_std, 1)
    a_fit = np.exp(log_a)
    print(f'  std(drain) ~ {a_fit:.4f} · L^{b_fit:.3f}')
    if abs(b_fit - 0.5) < 0.15:
        print(f'    (consistent with Var ~ L -> std ~ sqrtL)')
    else:
        print(f'    (exponent {b_fit:.3f} differs from 0.5; scaling is non-trivial)')

    # Fit margin/std vs L
    s_fit = np.polyfit(Ls, sigs, 1)
    print(f'  margin/std ~ {s_fit[0]:+.4f} · L + {s_fit[1]:.4f}')
    if s_fit[0] > 0.01:
        print(f'    -> INCREASING with L: concentration STRENGTHENS. Path alive.')
    elif s_fit[0] > -0.01:
        print(f'    -> FLAT: concentration neither strengthens nor weakens.')
    else:
        print(f'    -> DECREASING with L: concentration WEAKENS. Path in trouble.')

    # Check tail trend
    tails = scan[:, 6]
    t_fit = np.polyfit(Ls, tails, 1)
    print(f'  P(|drain|>margin) ~ {t_fit[0]:+.4f} · L + {t_fit[1]:.4f}')

    # -- Verdict -------------------------------------------------
    print()
    print('=' * 76)
    print('  VERDICT')
    print('=' * 76)

    last_sig = sigs[-1]
    first_sig = sigs[0]
    last_tail = tails[-1]

    if last_sig > 3.0:
        print(f'\n  margin/std at L={Ls[-1]:.0f}: {last_sig:.2f}sigma (> 3sigma)')
        print(f'  STRONG CONCENTRATION. Cramér drain lives well inside the margin.')
        print(f'  Proof path via equidistribution/large-sieve is ALIVE.')
        print(f'  -> Next: compute analytic Var(drain) and Chebyshev / sub-Gaussian')
        print(f'    bound to close the argument without Monte Carlo.')
    elif last_sig > 1.0:
        print(f'\n  margin/std at L={Ls[-1]:.0f}: {last_sig:.2f}sigma')
        print(f'  MODERATE concentration. Cramér drain is usually inside margin,')
        print(f'  but the tail probability P = {last_tail:.3f} is non-negligible.')
        print(f'  Proof requires more than Chebyshev — need sub-Gaussian or')
        print(f'  sharper tail bounds.')
        if last_sig > first_sig:
            print(f'  Trend: IMPROVING with L ({first_sig:.2f}sigma -> {last_sig:.2f}sigma).')
        else:
            print(f'  Trend: WORSENING ({first_sig:.2f}sigma -> {last_sig:.2f}sigma). Concern.')
    else:
        print(f'\n  margin/std at L={Ls[-1]:.0f}: {last_sig:.2f}sigma (< 1sigma)')
        print(f'  WEAK concentration. Cramér drain frequently exceeds margin.')
        print(f'  P(|drain|>margin) = {last_tail:.3f} at largest L tested.')
        if s_fit[0] < -0.005:
            print(f'  Trend: WORSENING with L. Slope = {s_fit[0]:+.4f} sigma/unit L.')
            print(f'  KILL: even generic Cramér sequences violate |drain| < margin.')
            print(f'  The margin-drain approach cannot work via concentration alone.')
            print(f'  Real primes would need ANTI-concentration (being BETTER than')
            print(f'  random), which is the opposite of what equidistribution gives.')
        else:
            print(f'  Trend: STABLE or improving.')
            print(f'  Not dead yet, but very tight. Need sharp tail bounds.')

    # Save
    np.savez('session54_cramer_concentration.npz',
             L_targets=np.array(L_targets),
             scan=scan)
    print('\n  Data saved to session54_cramer_concentration.npz')


if __name__ == '__main__':
    run()
