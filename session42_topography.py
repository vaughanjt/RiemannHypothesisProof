"""
SESSION 42 — TOPOGRAPHY OF THE BARRIER

The barrier B(lam^2) is a function we've computed at 499 integer points.
Study it as a time series:
1. Shape: what does it look like?
2. Statistics: distribution, mean, variance, autocorrelation
3. Jumps: each prime creates a discrete step — can we predict the pattern?
4. Fourier: what frequencies drive the oscillation?
5. Prediction: can we model B(lam^2) and forecast to infinity?

The barrier is:
B = margin(L) - drain(L)
  = smooth_background(L) - cumulative_prime_jumps(L)

Each prime p adds a jump delta_p to the drain when lam^2 crosses p.
The smooth background changes continuously with L = log(lam^2).
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from connes_crossterm import build_all


def compute_barriers_dense(lam_min, lam_max):
    """Compute odd and even barriers at every integer lam^2."""
    barriers_odd = []
    barriers_even = []

    for lam_sq in range(lam_min, lam_max + 1):
        L = np.log(lam_sq)
        N = max(15, round(6 * L))
        W02, M, QW = build_all(lam_sq, N, n_quad=4000)
        ns = np.arange(-N, N + 1, dtype=float)

        w = ns / (L**2 + (4*np.pi)**2 * ns**2)
        w[N] = 0.0
        w_hat = w / np.linalg.norm(w)

        u = 1.0 / (L**2 + (4*np.pi)**2 * ns**2)
        u_hat = u / np.linalg.norm(u)

        barriers_odd.append(float(w_hat @ QW @ w_hat))
        barriers_even.append(float(u_hat @ QW @ u_hat))

    return np.array(barriers_odd), np.array(barriers_even)


def is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i+2) == 0:
            return False
        i += 6
    return True


if __name__ == '__main__':
    print()
    print('#' * 72)
    print('  TOPOGRAPHY OF THE BARRIER')
    print('#' * 72)

    # Load the 499-point data (recompute a subset for speed)
    # Use range 2..500 from the Lipschitz computation
    print('\n  Computing barriers at lam^2 = 2..500...', flush=True)
    t0 = time.time()
    odd, even = compute_barriers_dense(2, 500)
    lam_range = np.arange(2, 501)
    L_range = np.log(lam_range)
    print(f'  Done in {time.time()-t0:.0f}s')

    # ═══════════════════════════════════════════════════════════
    # PART 1: BASIC STATISTICS
    # ═══════════════════════════════════════════════════════════
    print('\n\n  PART 1: Basic statistics')
    print('  ' + '=' * 60)

    for name, data in [('Odd', odd), ('Even', even)]:
        print(f'\n  {name} direction:')
        print(f'    Mean:   {data.mean():.6f}')
        print(f'    Std:    {data.std():.6f}')
        print(f'    Min:    {data.min():.6f} at lam^2={lam_range[data.argmin()]}')
        print(f'    Max:    {data.max():.6f} at lam^2={lam_range[data.argmax()]}')
        print(f'    Median: {np.median(data):.6f}')
        print(f'    Skew:   {((data - data.mean())**3).mean() / data.std()**3:.3f}')
        print(f'    CV:     {data.std()/data.mean():.3f}')

    # ═══════════════════════════════════════════════════════════
    # PART 2: JUMPS AT PRIMES
    # ═══════════════════════════════════════════════════════════
    print('\n\n  PART 2: Barrier jumps at primes')
    print('  ' + '=' * 60)

    # Compute jump = B(p) - B(p-1) at each prime p
    jumps_odd = np.diff(odd)   # jumps[i] = odd[i+1] - odd[i] = B(i+3) - B(i+2)
    jumps_even = np.diff(even)

    prime_mask = np.array([is_prime(k) for k in range(3, 501)])  # is k prime?
    composite_mask = ~prime_mask

    # Prime jumps vs composite jumps
    pj_odd = jumps_odd[prime_mask[:len(jumps_odd)]]
    cj_odd = jumps_odd[composite_mask[:len(jumps_odd)]]
    pj_even = jumps_even[prime_mask[:len(jumps_even)]]
    cj_even = jumps_even[composite_mask[:len(jumps_even)]]

    print(f'\n  Odd direction:')
    print(f'    At primes:     mean jump = {pj_odd.mean():+.6f}  std = {pj_odd.std():.6f}  '
          f'({len(pj_odd)} primes)')
    print(f'    At composites: mean jump = {cj_odd.mean():+.6f}  std = {cj_odd.std():.6f}  '
          f'({len(cj_odd)} composites)')
    print(f'    Ratio of stds: {pj_odd.std() / cj_odd.std():.2f}x')

    print(f'\n  Even direction:')
    print(f'    At primes:     mean jump = {pj_even.mean():+.6f}  std = {pj_even.std():.6f}')
    print(f'    At composites: mean jump = {cj_even.mean():+.6f}  std = {cj_even.std():.6f}')

    # Top 10 largest jumps
    print(f'\n  Top 10 largest |jumps| (odd):')
    top_idx = np.argsort(np.abs(jumps_odd))[::-1][:10]
    for idx in top_idx:
        k = idx + 3  # lam^2 value
        p = is_prime(k)
        print(f'    lam^2={k}: jump={jumps_odd[idx]:+.6f}  {"PRIME" if p else "composite"}')

    # ═══════════════════════════════════════════════════════════
    # PART 3: AUTOCORRELATION
    # ═══════════════════════════════════════════════════════════
    print('\n\n  PART 3: Autocorrelation')
    print('  ' + '=' * 60)

    # Autocorrelation of the barrier
    odd_centered = odd - odd.mean()
    acf = np.correlate(odd_centered, odd_centered, mode='full')
    acf = acf[len(acf)//2:]  # keep positive lags
    acf /= acf[0]  # normalize

    print(f'\n  Odd barrier autocorrelation:')
    print(f'    lag  1: {acf[1]:.4f}')
    print(f'    lag  2: {acf[2]:.4f}')
    print(f'    lag  5: {acf[5]:.4f}')
    print(f'    lag 10: {acf[10]:.4f}')
    print(f'    lag 30: {acf[30]:.4f}')
    print(f'    lag 50: {acf[50]:.4f}')

    # First zero crossing
    zero_cross = np.where(acf[1:] <= 0)[0]
    if len(zero_cross) > 0:
        print(f'    First zero crossing: lag {zero_cross[0]+1}')
    else:
        print(f'    No zero crossing in first {len(acf)} lags')

    # ═══════════════════════════════════════════════════════════
    # PART 4: FOURIER SPECTRUM
    # ═══════════════════════════════════════════════════════════
    print('\n\n  PART 4: Fourier spectrum')
    print('  ' + '=' * 60)

    fft_odd = np.fft.rfft(odd - odd.mean())
    freqs = np.fft.rfftfreq(len(odd), d=1)  # in cycles per lam^2
    power = np.abs(fft_odd)**2
    power[0] = 0  # remove DC

    top10 = np.argsort(power)[::-1][:10]
    print(f'\n  Top 10 frequencies (odd barrier):')
    print(f'  {"freq":>10s} {"period":>8s} {"power":>10s} {"pct":>6s}')
    print('  ' + '-' * 38)
    total_power = power.sum()
    for idx in top10:
        f = freqs[idx]
        period = 1/f if f > 0 else float('inf')
        print(f'  {f:>10.6f} {period:>8.1f} {power[idx]:>10.4f} {power[idx]/total_power*100:>5.1f}%')

    # Connection to log(prime)?
    print(f'\n  Notable periods and their prime connections:')
    for p in [2, 3, 5, 7, 11]:
        expected_period = p  # jumps every p integers when p divides lam^2
        expected_freq = 1.0 / expected_period
        # Find nearest frequency in spectrum
        nearest = np.argmin(np.abs(freqs - expected_freq))
        print(f'    p={p}: expected period={expected_period}, nearest freq={freqs[nearest]:.6f} '
              f'(period={1/freqs[nearest] if freqs[nearest]>0 else 0:.1f}), '
              f'power={power[nearest]:.4f}')

    # ═══════════════════════════════════════════════════════════
    # PART 5: RUNNING STATISTICS
    # ═══════════════════════════════════════════════════════════
    print('\n\n  PART 5: Running statistics (how do moments evolve?)')
    print('  ' + '=' * 60)

    window = 50
    print(f'\n  Window = {window} consecutive lam^2 values:')
    print(f'  {"lam^2 range":>15s} {"mean":>8s} {"std":>8s} {"min":>8s} {"max":>8s} '
          f'{"min/mean":>8s}')
    print('  ' + '-' * 60)

    for start in range(0, len(odd) - window, window):
        end = start + window
        segment = odd[start:end]
        lam_start = lam_range[start]
        lam_end = lam_range[end-1]
        print(f'  [{lam_start:>4d}, {lam_end:>4d}] {segment.mean():>8.4f} '
              f'{segment.std():>8.4f} {segment.min():>8.4f} {segment.max():>8.4f} '
              f'{segment.min()/segment.mean():>8.3f}')

    # ═══════════════════════════════════════════════════════════
    # PART 6: PREDICTION MODEL
    # ═══════════════════════════════════════════════════════════
    print('\n\n  PART 6: Can we predict barrier(lam^2)?')
    print('  ' + '=' * 60)

    # Model: B(k) = mu + sum_{p<=k} delta_p
    # where delta_p is the jump at prime p
    # and mu is the smooth background

    # Compute cumulative prime jumps
    cum_jumps = np.cumsum(jumps_odd * np.array([is_prime(k) for k in range(3, 501)][:len(jumps_odd)]))
    # Smooth background = barrier minus cumulative prime jumps
    smooth_bg = odd[1:] - cum_jumps[:len(odd)-1]

    print(f'  Smooth background (barrier - cumulative prime jumps):')
    print(f'    Mean: {smooth_bg.mean():.6f}')
    print(f'    Std:  {smooth_bg.std():.6f}')
    print(f'    Range: [{smooth_bg.min():.6f}, {smooth_bg.max():.6f}]')

    # What fraction of variance comes from primes?
    total_var = odd.var()
    prime_var = cum_jumps.var() if len(cum_jumps) > 0 else 0
    smooth_var = smooth_bg.var() if len(smooth_bg) > 0 else 0
    print(f'\n  Variance decomposition:')
    print(f'    Total variance:          {total_var:.6f}')
    print(f'    Prime jump variance:     {prime_var:.6f} ({prime_var/total_var*100:.1f}%)')
    print(f'    Smooth background var:   {smooth_var:.6f} ({smooth_var/total_var*100:.1f}%)')

    # Mean jump size at primes as function of p
    print(f'\n  Mean |jump| at primes by size:')
    for p_min, p_max in [(2, 10), (11, 50), (51, 100), (101, 200), (201, 500)]:
        mask = np.array([is_prime(k) and p_min <= k <= p_max
                         for k in range(3, 501)][:len(jumps_odd)])
        if np.sum(mask) > 0:
            mean_j = np.mean(np.abs(jumps_odd[mask]))
            print(f'    p in [{p_min:>3d}, {p_max:>3d}]: mean |jump| = {mean_j:.6f}  '
                  f'({np.sum(mask)} primes)')

    # ═══════════════════════════════════════════════════════════
    # PART 7: EXTRAPOLATION
    # ═══════════════════════════════════════════════════════════
    print('\n\n  PART 7: Extrapolation to infinity')
    print('  ' + '=' * 60)

    # The barrier's mean is stable. Does the standard deviation shrink?
    # Theory: if jumps are ~independent with size ~log(p)/sqrt(p),
    # the variance of the running sum grows as sum (log(p)/sqrt(p))^2 / n_primes
    # But the mean prime contribution also shrinks...

    # Model the barrier as: B(L) = mu(L) + sigma(L) * Z
    # where Z is a bounded random variable and sigma(L) shrinks

    # From the running statistics:
    means = []
    stds = []
    L_mids = []
    for start in range(0, len(odd) - window, window // 2):
        end = min(start + window, len(odd))
        if end - start < window // 2:
            break
        segment = odd[start:end]
        means.append(segment.mean())
        stds.append(segment.std())
        L_mids.append(L_range[start:end].mean())

    means = np.array(means)
    stds = np.array(stds)
    L_mids = np.array(L_mids)

    # Fit std decay
    if len(L_mids) >= 4:
        c_std = np.polyfit(L_mids, np.log(stds + 1e-10), 1)
        print(f'  Std decay: sigma ~ exp({c_std[0]:.3f} * L)')
        print(f'  Half-life of fluctuation amplitude: {np.log(2)/abs(c_std[0]):.2f} in L')
        print(f'  At L=15: predicted std = {np.exp(c_std[0]*15 + c_std[1]):.6f}')
        print(f'  At L=20: predicted std = {np.exp(c_std[0]*20 + c_std[1]):.6f}')

    # Mean trend
    c_mean = np.polyfit(L_mids, means, 1)
    print(f'\n  Mean trend: mu = {c_mean[0]:.6f}*L + {c_mean[1]:.6f}')
    print(f'  At L=15: predicted mean = {c_mean[0]*15 + c_mean[1]:.6f}')
    print(f'  At L=20: predicted mean = {c_mean[0]*20 + c_mean[1]:.6f}')

    # Probability of reaching zero
    if len(means) >= 4 and np.mean(stds) > 0:
        # Crude: if B ~ N(mu, sigma^2), P(B < 0) = Phi(-mu/sigma)
        mu = means[-1]
        sigma = stds[-1]
        z = mu / sigma
        print(f'\n  At lam^2~500: mu={mu:.4f}, sigma={sigma:.4f}, z=mu/sigma={z:.2f}')
        print(f'  If Gaussian: P(B<0) ~ exp(-z^2/2) = exp(-{z**2/2:.1f})')
        if z > 3:
            print(f'  => Barrier is {z:.1f} sigma above zero. Extremely unlikely to reach zero.')

    print('\n' + '#' * 72)
