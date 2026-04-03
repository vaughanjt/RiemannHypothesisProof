"""
SESSION 42d — ANATOMY OF BARRIER FLUCTUATIONS

Decompose the barrier into smooth + oscillatory parts.
Identify what drives the fluctuations:
1. Compute barrier at densely-spaced lambda^2 values
2. Extract smooth trend (moving average)
3. Analyze residual fluctuations
4. Correlate fluctuations with prime structure
5. Bound the oscillatory amplitude

Strategy: if smooth_barrier > |fluctuation|, then barrier > 0.
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from session41g_uncapped_barrier import compute_barrier_partial


def sieve_primes(limit):
    """Sieve of Eratosthenes."""
    if limit < 2:
        return np.array([], dtype=int)
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[:2] = False
    for i in range(2, int(limit**0.5) + 2):
        if i <= limit and is_prime[i]:
            is_prime[i*i::i] = False
    return np.where(is_prime)[0]


def per_prime_contribution(lam_sq, N=None):
    """
    Compute each prime's individual contribution to the M_prime Rayleigh quotient.
    Returns list of (p, contribution) sorted by |contribution|.
    """
    L = np.log(lam_sq)
    if N is None:
        N = max(15, round(6 * L))
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)

    w = ns / (L**2 + (4 * np.pi)**2 * ns**2)
    w[N] = 0.0
    w_hat = w / np.linalg.norm(w)

    primes = sieve_primes(int(lam_sq))
    nm_diff = ns[:, None] - ns[None, :]

    contributions = []

    for p in primes:
        pk = int(p)
        k = 1
        logp = np.log(p)
        prime_total = 0.0

        while pk <= lam_sq:
            logpk = k * logp
            weight = logp * pk**(-0.5)
            y = logpk

            sin_arr = np.sin(2 * np.pi * ns * y / L)
            cos_arr = np.cos(2 * np.pi * ns * y / L)

            # Diagonal
            diag = 2 * (L - y) / L * cos_arr
            diag_contrib = weight * np.sum(w_hat**2 * diag)

            # Off-diagonal
            sin_diff = sin_arr[None, :] - sin_arr[:, None]
            with np.errstate(divide='ignore', invalid='ignore'):
                off_diag = sin_diff / (np.pi * nm_diff)
            np.fill_diagonal(off_diag, 0.0)
            off_contrib = weight * (w_hat @ off_diag @ w_hat)

            prime_total += diag_contrib + off_contrib
            pk *= int(p)
            k += 1

        contributions.append((int(p), prime_total))

    return contributions


if __name__ == '__main__':
    print()
    print('#' * 70)
    print('  SESSION 42d -- FLUCTUATION ANATOMY')
    print('#' * 70)

    # ── Part 1: Dense barrier sweep ──
    print('\n  PART 1: Dense W02-Mp sweep (step=100 from 100 to 10000)')
    print('  ' + '=' * 60)

    # Dense grid
    lam_values = list(range(100, 10001, 100))
    results = []

    t0_total = time.time()
    for lam_sq in lam_values:
        r = compute_barrier_partial(lam_sq)
        results.append({
            'lam_sq': lam_sq,
            'L': r['L'],
            'w02_mp': r['partial_barrier'],
        })

    print(f'  Computed {len(results)} points in {time.time()-t0_total:.0f}s')

    Ls = np.array([r['L'] for r in results])
    w02_mps = np.array([r['w02_mp'] for r in results])

    # ── Part 2: Smooth envelope (moving average) ──
    print('\n\n  PART 2: Smooth envelope extraction')
    print('  ' + '=' * 60)

    # Moving average with window of 20 points (2000 in lambda^2)
    window = 20
    smooth = np.convolve(w02_mps, np.ones(window)/window, mode='valid')
    smooth_Ls = Ls[window//2:-window//2+1]

    # Fluctuation = raw - smooth (interpolated back to full grid)
    smooth_interp = np.interp(Ls, smooth_Ls, smooth)
    fluctuation = w02_mps - smooth_interp

    print(f'  Smooth range: [{smooth.min():.6f}, {smooth.max():.6f}]')
    print(f'  Fluctuation range: [{fluctuation.min():.6f}, {fluctuation.max():.6f}]')
    print(f'  Fluctuation RMS: {np.std(fluctuation):.6f}')
    print(f'  Max |fluctuation| / min smooth: {np.max(np.abs(fluctuation)) / smooth.min():.4f}')

    # Show smooth and fluctuation at key L values
    print(f'\n  {"lam^2":>7s} {"L":>6s} {"W02-Mp":>10s} {"smooth":>10s} {"fluct":>10s}')
    print('  ' + '-' * 48)
    for i in range(0, len(results), 10):
        r = results[i]
        print(f'  {r["lam_sq"]:>7d} {r["L"]:>6.2f} {r["w02_mp"]:>+10.6f} '
              f'{smooth_interp[i]:>+10.6f} {fluctuation[i]:>+10.6f}')

    # ── Part 3: Fourier analysis of fluctuations ──
    print('\n\n  PART 3: Fourier analysis of fluctuations')
    print('  ' + '=' * 60)

    # FFT of fluctuation signal
    fft = np.fft.rfft(fluctuation)
    freqs = np.fft.rfftfreq(len(fluctuation), d=(Ls[1]-Ls[0]))
    power = np.abs(fft)**2

    # Top 10 frequencies
    top_idx = np.argsort(power[1:])[::-1][:10] + 1  # skip DC
    print(f'  Top 10 fluctuation frequencies (in 1/L units):')
    for idx in top_idx:
        period = 1/freqs[idx] if freqs[idx] > 0 else np.inf
        print(f'    freq={freqs[idx]:.4f}  period={period:.2f}  '
              f'power={power[idx]:.6f}  ({power[idx]/power[1:].sum()*100:.1f}%)')

    # ── Part 4: Per-prime contribution analysis ──
    print('\n\n  PART 4: Per-prime contributions to M_prime')
    print('  ' + '=' * 60)

    for lam_sq in [1000, 5000, 10000]:
        print(f'\n  lam^2 = {lam_sq}:')
        t0 = time.time()
        contribs = per_prime_contribution(lam_sq)
        dt = time.time() - t0

        # Sort by absolute contribution
        contribs.sort(key=lambda x: abs(x[1]), reverse=True)

        total = sum(c for _, c in contribs)
        pos_total = sum(c for _, c in contribs if c > 0)
        neg_total = sum(c for _, c in contribs if c < 0)

        print(f'    Total M_prime = {total:+.6f}  ({dt:.1f}s)')
        print(f'    Positive contributions: {pos_total:+.6f} ({sum(1 for _,c in contribs if c>0)} primes)')
        print(f'    Negative contributions: {neg_total:+.6f} ({sum(1 for _,c in contribs if c<0)} primes)')
        print(f'    Cancellation: {abs(pos_total+neg_total)/max(abs(pos_total),abs(neg_total))*100:.1f}% residual')

        print(f'    Top 10 by magnitude:')
        for p, c in contribs[:10]:
            t = np.log(p) / np.log(lam_sq)
            sign = '+' if c > 0 else '-'
            print(f'      p={p:>5d}  t={t:.4f}  contrib={c:+.6f}  '
                  f'({abs(c)/abs(total)*100:.1f}%)')

    # ── Part 5: Fluctuation bound ──
    print('\n\n  PART 5: Can we bound the fluctuation?')
    print('  ' + '=' * 60)

    # The fluctuation amplitude vs L
    # Split into segments and compute local amplitude
    seg_size = 10  # 10 points per segment
    n_segs = len(fluctuation) // seg_size
    print(f'\n  {"L_mid":>8s} {"amp":>10s} {"smooth":>10s} {"ratio":>8s} {"safe":>6s}')
    print('  ' + '-' * 48)

    all_safe = True
    for seg in range(n_segs):
        idx_start = seg * seg_size
        idx_end = (seg + 1) * seg_size
        L_mid = Ls[idx_start:idx_end].mean()
        amp = np.max(np.abs(fluctuation[idx_start:idx_end]))
        sm = smooth_interp[idx_start:idx_end].mean()
        ratio = amp / sm if sm > 0 else float('inf')
        safe = 'YES' if amp < sm else 'NO'
        if amp >= sm:
            all_safe = False
        print(f'  {L_mid:>8.3f} {amp:>+10.6f} {sm:>+10.6f} {ratio:>8.4f} {safe:>6s}')

    print(f'\n  All segments safe? {all_safe}')

    # Overall bound
    max_fluct = np.max(np.abs(fluctuation))
    min_smooth = smooth.min()
    print(f'  Global max |fluctuation| = {max_fluct:.6f}')
    print(f'  Global min smooth        = {min_smooth:.6f}')
    print(f'  Ratio = {max_fluct/min_smooth:.4f}')
    if max_fluct < min_smooth:
        print(f'  => BARRIER PROVABLY POSITIVE on [{results[0]["lam_sq"]}, {results[-1]["lam_sq"]}]')
        print(f'     (smooth envelope always exceeds fluctuation amplitude)')
    else:
        print(f'  => Cannot prove positivity from this decomposition alone')
        print(f'     Need tighter fluctuation bound or larger smooth envelope')

    print('\n' + '#' * 70)
    print('  SESSION 42d COMPLETE')
    print('#' * 70)
