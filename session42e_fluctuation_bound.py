"""
SESSION 42e — FLUCTUATION BOUND FROM THE EXPLICIT FORMULA

The M_prime Rayleigh quotient fluctuation comes from the discrete prime sum.
We need: |fluctuation(L)| <= C / L^alpha for some alpha > 0.

Approach:
1. M_prime Rayleigh quotient = sum_{p^k <= lam^2} log(p)/sqrt(p^k) * F(log(p^k)/L)
   where F(t) = <w_hat, Q_t, w_hat> is the filter applied to each prime.

2. The smooth average = integral from 1 to lam^2 of (1/x) * F(log(x)/L) dx
   (by PNT: pi(x) ~ x/log(x), so sum log(p)/sqrt(p) ~ integral)

3. Fluctuation = prime_sum - smooth_integral
   = sum_{p} log(p)/sqrt(p) * F(t_p) - integral F(t) * dt/sqrt(e^{tL})

4. By the explicit formula: this difference involves sum over zeta zeros.
   The error in PNT is O(x * exp(-c*sqrt(log x))), giving fluctuation
   bound O(exp(-c*sqrt(L))).

Alternatively: direct computation of fluctuation amplitude at many L values,
fit the decay rate, and verify it's bounded by the smooth barrier.
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from session41g_uncapped_barrier import compute_barrier_partial


def smooth_mprime_integral(lam_sq, N=None):
    """
    Compute the smooth (PNT) approximation to M_prime Rayleigh quotient.

    Replace sum_{p<=x} log(p)/sqrt(p) * F(log(p)/L) with
    integral_2^x F(log(t)/L) / sqrt(t) dt  [by PNT: sum log(p)/sqrt(p) ~ integral 1/sqrt(t)]

    Actually, by Mertens/PNT:
    sum_{p<=x} log(p) * f(p) = integral_2^x f(t) dt + error

    So M_prime ~ integral_2^{lam^2} (1/sqrt(t)) * F(log(t)/L) dt
    where F is the per-prime filter function.
    """
    L = np.log(lam_sq)
    if N is None:
        N = max(15, round(6 * L))
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)

    w = ns / (L**2 + (4 * np.pi)**2 * ns**2)
    w[N] = 0.0
    w_hat = w / np.linalg.norm(w)

    nm_diff = ns[:, None] - ns[None, :]

    # Numerical integration: replace sum over primes with integral
    n_pts = 5000
    # Integrate over t from 2 to lam^2
    # Change variables: u = log(t)/L, so t = e^{uL}, dt = L*e^{uL} du
    # integral = integral_{log(2)/L}^{1} F(u) * 1/sqrt(e^{uL}) * L*e^{uL} du
    #          = L * integral F(u) * e^{uL/2} du
    # Hmm, this diverges. Let me think again.

    # Direct: integral_2^{lam^2} (1/sqrt(t)) * F(log(t)/L) dt
    # where F(t_val) = <w_hat, Q_{y=t_val*L}, w_hat> for all prime powers at y

    # For a single "prime" at value t (contributing at y = log(t)):
    # weight = log(t)/sqrt(t)  [approximating log(p) ~ log(t) for primes near t]
    # Actually PNT says: sum_{p<=x} log(p) g(p) ~ integral_2^x g(t) dt
    # So we replace: sum_p log(p)/sqrt(p) * F(log(p)/L)
    #            ~   integral_2^{lam^2} F(log(t)/L) / sqrt(t) dt  [WRONG: missing log(p)]

    # More precisely: sum_{p<=x} log(p) * h(p) = integral_2^x h(t) dt + error (PNT)
    # Here h(p) = p^{-1/2} * F(log(p)/L)
    # So M_prime ~ integral_2^{lam^2} t^{-1/2} * F(log(t)/L) dt + higher prime powers

    # Use substitution y = log(t), t = e^y, dt = e^y dy
    # = integral_{log 2}^{L} e^{-y/2} * F(y/L) * e^y dy
    # = integral_{log 2}^{L} e^{y/2} * F(y/L) dy

    y_pts = np.linspace(np.log(2), L, n_pts)
    dy = y_pts[1] - y_pts[0]

    integral = 0.0
    for y in y_pts:
        t_val = y / L  # t = log(p^k)/L in [0, 1]

        # F(t_val) = <w_hat, Q_y, w_hat> where Q_y has:
        # diagonal: 2(L-y)/L * cos(2*pi*n*y/L)
        # off-diagonal: (sin(2*pi*m*y/L) - sin(2*pi*n*y/L)) / (pi*(n-m))

        sin_arr = np.sin(2 * np.pi * ns * y / L)
        cos_arr = np.cos(2 * np.pi * ns * y / L)

        # Diagonal part
        diag_val = 2 * (L - y) / L * np.sum(w_hat**2 * cos_arr)

        # Off-diagonal part
        sin_diff = sin_arr[None, :] - sin_arr[:, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            off_diag = sin_diff / (np.pi * nm_diff)
        np.fill_diagonal(off_diag, 0.0)
        off_val = w_hat @ off_diag @ w_hat

        F_val = diag_val + off_val
        integral += np.exp(y / 2) * F_val * dy

    return integral


def compute_fluctuation_data(lam_values):
    """Compute W02-Mp at each lambda, plus the smooth integral approximation."""
    results = []
    for lam_sq in lam_values:
        r = compute_barrier_partial(lam_sq)
        results.append({
            'lam_sq': lam_sq,
            'L': r['L'],
            'w02_mp': r['partial_barrier'],
            'mprime': r['mprime'],
        })
    return results


if __name__ == '__main__':
    print()
    print('#' * 70)
    print('  SESSION 42e -- FLUCTUATION BOUND DERIVATION')
    print('#' * 70)

    # ── Part 1: Dense barrier data ──
    print('\n  PART 1: Dense W02-Mp computation')
    print('  ' + '=' * 60)

    # Very dense: every 50 from 100 to 10000
    lam_values = list(range(100, 10001, 50))
    t0 = time.time()
    results = compute_fluctuation_data(lam_values)
    print(f'  Computed {len(results)} points in {time.time()-t0:.0f}s')

    Ls = np.array([r['L'] for r in results])
    w02_mps = np.array([r['w02_mp'] for r in results])

    # ── Part 2: Moving averages at multiple scales ──
    print('\n\n  PART 2: Multi-scale smooth extraction')
    print('  ' + '=' * 60)

    for window_lam in [500, 1000, 2000, 4000]:
        window = window_lam // 50  # convert to index units
        if window < 3 or window >= len(w02_mps):
            continue
        smooth = np.convolve(w02_mps, np.ones(window)/window, mode='valid')
        smooth_L = Ls[window//2:window//2+len(smooth)]

        # Interpolate back
        smooth_full = np.interp(Ls, smooth_L, smooth)
        fluct = w02_mps - smooth_full

        # Measure amplitude in segments
        seg_size = max(window, 10)
        n_segs = max(1, len(fluct) // seg_size)

        amps = []
        L_mids = []
        for s in range(n_segs):
            i0 = s * seg_size
            i1 = min((s+1) * seg_size, len(fluct))
            if i1 <= i0:
                break
            amps.append(np.max(np.abs(fluct[i0:i1])))
            L_mids.append(Ls[i0:i1].mean())

        amps = np.array(amps)
        L_mids = np.array(L_mids)

        # Fit amplitude decay: amp ~ C * exp(-alpha * L)
        valid = amps > 1e-10
        if np.sum(valid) >= 3:
            log_amps = np.log(amps[valid])
            c_fit = np.polyfit(L_mids[valid], log_amps, 1)
            alpha = -c_fit[0]
            C = np.exp(c_fit[1])

            print(f'\n  Window = {window_lam} (in lam^2 units):')
            print(f'    Fluctuation amplitude ~ {C:.4f} * exp(-{alpha:.4f} * L)')
            print(f'    Half-life in L: {np.log(2)/alpha:.2f}')
            print(f'    At L=9.2: predicted amp = {C*np.exp(-alpha*9.2):.6f}')
            print(f'    At L=11.5: predicted amp = {C*np.exp(-alpha*11.5):.6f}')
            print(f'    At L=15: predicted amp = {C*np.exp(-alpha*15):.6f}')

    # ── Part 3: Direct fluctuation measurement ──
    print('\n\n  PART 3: Direct fluctuation measurement')
    print('  ' + '=' * 60)

    # Use window=2000 as baseline smooth
    window = 40  # 40 * 50 = 2000 in lam^2
    smooth = np.convolve(w02_mps, np.ones(window)/window, mode='valid')
    smooth_L = Ls[window//2:window//2+len(smooth)]
    smooth_full = np.interp(Ls, smooth_L, smooth)
    fluct = w02_mps - smooth_full

    print(f'  {"L_range":>12s} {"max_fluct":>10s} {"smooth_val":>10s} '
          f'{"ratio":>8s} {"safe":>6s}')
    print('  ' + '-' * 52)

    # 0.5 unit L segments
    L_min, L_max = Ls[window], Ls[-window]
    L_edges = np.arange(L_min, L_max, 0.5)

    all_safe = True
    for i in range(len(L_edges) - 1):
        mask = (Ls >= L_edges[i]) & (Ls < L_edges[i+1])
        if np.sum(mask) < 2:
            continue
        max_f = np.max(np.abs(fluct[mask]))
        avg_s = np.mean(smooth_full[mask])
        ratio = max_f / avg_s if avg_s > 0 else float('inf')
        safe = 'YES' if ratio < 1.0 else 'NO'
        if ratio >= 1.0:
            all_safe = False
        print(f'  [{L_edges[i]:.1f}, {L_edges[i+1]:.1f})  {max_f:>+10.6f} '
              f'{avg_s:>+10.6f} {ratio:>8.4f} {safe:>6s}')

    print(f'\n  All safe? {all_safe}')

    # ── Part 4: PNT integral vs prime sum ──
    print('\n\n  PART 4: Smooth integral approximation to M_prime')
    print('  ' + '=' * 60)

    for lam_sq in [500, 1000, 5000, 10000]:
        t0 = time.time()
        smooth_mp = smooth_mprime_integral(lam_sq)
        dt = time.time() - t0

        # Compare to actual M_prime
        r = compute_barrier_partial(lam_sq)
        actual_mp = r['mprime']

        diff = actual_mp - smooth_mp
        rel = abs(diff / actual_mp) * 100 if abs(actual_mp) > 1e-10 else 0

        print(f'  lam^2={lam_sq:>6d}: actual_Mp={actual_mp:+.4f}  '
              f'smooth_Mp={smooth_mp:+.4f}  diff={diff:+.4f}  ({rel:.1f}%)  ({dt:.1f}s)')

    # ── Part 5: Analytical fluctuation bound ──
    print('\n\n  PART 5: Analytical fluctuation bound')
    print('  ' + '=' * 60)

    # The PNT error term: sum_{p<=x} log(p) = x + O(x*exp(-c*sqrt(log x)))
    # For our sum: sum log(p)/sqrt(p) * F(t_p) - integral F(t)/sqrt(t) dt
    # The error is bounded by the PNT remainder applied to the smooth test function F.
    #
    # By partial summation:
    # |sum - integral| <= ||F||_inf * |pi(x) - li(x)| * max_weight / sqrt(x)
    #
    # Under RH: |pi(x) - li(x)| <= C * sqrt(x) * log(x) / (2*pi)
    # Without RH: |pi(x) - li(x)| <= C * x * exp(-c*sqrt(log x))
    #
    # But we can use the UNCONDITIONAL bound to get:
    # |fluctuation| <= C' * exp(-c*sqrt(L))  [since x = lam^2 = e^L]

    print('  Theoretical bound (unconditional PNT):')
    print('    |fluctuation| <= C * exp(-c * sqrt(L))')
    print()

    # Fit from data
    # Use window=2000 fluctuation amplitudes
    valid_mask = (Ls > 6.0) & (Ls < 9.3)
    valid_L = Ls[valid_mask]
    valid_fluct = np.abs(fluct[valid_mask])

    # Compute running max in segments
    seg_L = np.arange(6.0, 9.5, 0.3)
    seg_amps = []
    seg_mids = []
    for i in range(len(seg_L)-1):
        m = (valid_L >= seg_L[i]) & (valid_L < seg_L[i+1])
        if np.sum(m) > 0:
            seg_amps.append(np.max(valid_fluct[m]))
            seg_mids.append((seg_L[i] + seg_L[i+1]) / 2)

    seg_amps = np.array(seg_amps)
    seg_mids = np.array(seg_mids)

    if len(seg_amps) >= 3:
        # Fit: log(amp) = a + b*sqrt(L)  [exp(-c*sqrt(L)) form]
        sqrt_L = np.sqrt(seg_mids)
        log_amp = np.log(seg_amps + 1e-15)
        c_sqrt = np.polyfit(sqrt_L, log_amp, 1)

        # Also fit: log(amp) = a + b*L  [exp(-c*L) form]
        c_lin = np.polyfit(seg_mids, log_amp, 1)

        # Also fit: log(amp) = a + b*log(L)  [C/L^alpha form]
        c_pow = np.polyfit(np.log(seg_mids), log_amp, 1)

        print(f'  Empirical fits:')
        print(f'    exp(-c*sqrt(L)): |fluct| ~ exp({c_sqrt[0]:.3f}*sqrt(L) + {c_sqrt[1]:.3f})')
        print(f'    exp(-c*L):       |fluct| ~ exp({c_lin[0]:.3f}*L + {c_lin[1]:.3f})')
        print(f'    C/L^alpha:       |fluct| ~ L^{{{c_pow[0]:.3f}}} * exp({c_pow[1]:.3f})')

        print(f'\n  Predictions at large L:')
        for L_val in [10, 12, 15, 20]:
            pred_sqrt = np.exp(c_sqrt[0]*np.sqrt(L_val) + c_sqrt[1])
            pred_lin = np.exp(c_lin[0]*L_val + c_lin[1])
            pred_pow = np.exp(c_pow[0]*np.log(L_val) + c_pow[1])
            print(f'    L={L_val:>2d}: sqrt={pred_sqrt:.6f}  lin={pred_lin:.6f}  '
                  f'pow={pred_pow:.6f}')

    print('\n' + '#' * 70)
    print('  SESSION 42e COMPLETE')
    print('#' * 70)
