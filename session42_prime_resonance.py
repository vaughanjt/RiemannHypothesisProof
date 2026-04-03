"""
SESSION 42 — PRIME RESONANCE: DO THE BARRIER'S JUMPS ENCODE ZETA ZEROS?

The barrier has discrete jumps at each prime p.
The explicit formula says:
    sum_p log(p)/sqrt(p) * e^{-i*gamma*log(p)} ~ -sum_rho [zero contribution]

So the "Fourier transform" of the prime jumps in log-space should
have peaks at the imaginary parts of the zeta zeros.

Compute:
1. The sequence of prime jumps from the barrier data
2. Its Fourier transform F(gamma) = sum_p jump(p) * e^{-i*gamma*log(p)}
3. Check if |F(gamma)| peaks at gamma = 14.13, 21.02, 25.01, ...

If this works, the barrier's topography DIRECTLY encodes the zeros
without ever computing zeta(s).
"""

import numpy as np
import mpmath
from mpmath import zetazero
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from connes_crossterm import build_all
from session41g_uncapped_barrier import sieve_primes


def compute_prime_jumps(lam_max, N_basis=None):
    """
    Compute the barrier jump at each prime p <= lam_max.
    Jump at p = B(p) - B(p-1) where B is the odd barrier.

    Actually, more precisely: the jump comes from M_prime gaining the
    term for prime p. Compute per-prime contribution directly.
    """
    L = np.log(lam_max)
    if N_basis is None:
        N_basis = max(15, round(6 * L))
    dim = 2 * N_basis + 1
    ns = np.arange(-N_basis, N_basis + 1, dtype=float)

    w = ns / (L**2 + (4*np.pi)**2 * ns**2)
    w[N_basis] = 0.0
    w_hat = w / np.linalg.norm(w)
    nm_diff = ns[:, None] - ns[None, :]

    primes = sieve_primes(int(lam_max))
    jumps = []

    for p in primes:
        pk = int(p)
        k = 1
        logp = np.log(p)
        total = 0.0

        while pk <= lam_max:
            logpk = k * logp
            weight = logp * pk**(-0.5)
            y = logpk

            sin_arr = np.sin(2*np.pi*ns*y/L)
            cos_arr = np.cos(2*np.pi*ns*y/L)

            diag = 2*(L-y)/L * np.sum(w_hat**2 * cos_arr)
            sin_diff = sin_arr[None,:] - sin_arr[:,None]
            with np.errstate(divide='ignore', invalid='ignore'):
                off = sin_diff / (np.pi * nm_diff)
            np.fill_diagonal(off, 0.0)
            off_val = w_hat @ off @ w_hat

            total += weight * (diag + off_val)
            pk *= int(p)
            k += 1

        # Jump in barrier = -total (barrier = W02 - Mp, so adding to Mp reduces barrier)
        jumps.append((int(p), -total, logp))

    return jumps


def prime_fourier_transform(jumps, gamma_values):
    """
    Compute F(gamma) = sum_p jump(p) * exp(-i * gamma * log(p))

    This is the "spectral function" of the barrier's prime structure.
    Peaks should appear at the imaginary parts of zeta zeros.
    """
    result = np.zeros(len(gamma_values), dtype=complex)

    for p, jump, logp in jumps:
        phases = np.exp(-1j * gamma_values * logp)
        result += jump * phases

    return result


if __name__ == '__main__':
    print()
    print('#' * 72)
    print('  PRIME RESONANCE: ZETA ZEROS FROM BARRIER JUMPS')
    print('#' * 72)

    # Get zeta zeros for comparison
    print('\n  Loading zeta zeros...', flush=True)
    mpmath.mp.dps = 15
    true_zeros = [float(zetazero(k).imag) for k in range(1, 31)]
    print(f'  First 10 zeros: {[f"{z:.2f}" for z in true_zeros[:10]]}')

    # Compute prime jumps
    print('\n  Computing prime jumps up to lam^2 = 10000...')
    t0 = time.time()
    jumps = compute_prime_jumps(10000)
    print(f'  {len(jumps)} primes, computed in {time.time()-t0:.0f}s')

    # Show top jumps
    jumps_sorted = sorted(jumps, key=lambda x: abs(x[1]), reverse=True)
    print(f'\n  Top 10 largest jumps:')
    for p, j, lp in jumps_sorted[:10]:
        print(f'    p={p:>5d}: jump={j:+.6f}')

    # ── Fourier transform ──
    print('\n\n  Computing Fourier transform F(gamma)...')
    gamma_range = np.linspace(0, 60, 6000)  # fine grid up to gamma=60
    F = prime_fourier_transform(jumps, gamma_range)
    power = np.abs(F)**2

    # Find peaks
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(power, height=np.max(power)*0.05,
                                    distance=10)

    print(f'\n  Peaks in |F(gamma)|^2:')
    print(f'  {"gamma_peak":>10s} {"power":>10s} {"nearest_zero":>12s} {"error":>8s}')
    print('  ' + '-' * 48)

    detected_zeros = []
    for idx in peaks[:20]:
        g = gamma_range[idx]
        p = power[idx]
        # Find nearest true zero
        nearest = min(true_zeros, key=lambda z: abs(z - g))
        err = abs(g - nearest)
        detected_zeros.append((g, nearest, err))
        match = '<--' if err < 0.5 else ''
        print(f'  {g:>10.3f} {p:>10.2f} {nearest:>12.3f} {err:>8.3f}  {match}')

    # Success rate
    matched = sum(1 for _, _, e in detected_zeros if e < 1.0)
    print(f'\n  Peaks within 1.0 of a zero: {matched}/{len(detected_zeros)}')

    # ── Show the full spectrum near first few zeros ──
    print('\n\n  Detailed spectrum near first 5 zeros:')
    for k, z in enumerate(true_zeros[:5]):
        mask = (gamma_range > z - 2) & (gamma_range < z + 2)
        local = power[mask]
        local_g = gamma_range[mask]
        peak_idx = np.argmax(local)
        print(f'  zero_{k+1} = {z:.3f}: peak at {local_g[peak_idx]:.3f}, '
              f'power = {local[peak_idx]:.2f}')

    # ── Phase information ──
    print('\n\n  Phase of F at zeta zeros:')
    for k, z in enumerate(true_zeros[:10]):
        idx = np.argmin(np.abs(gamma_range - z))
        phase = np.angle(F[idx])
        amp = np.abs(F[idx])
        print(f'  gamma_{k+1} = {z:.3f}: |F| = {amp:.4f}, phase = {phase:.4f} rad')

    # ── Multiple lam_max comparison ──
    print('\n\n  Resolution vs lam_max:')
    for lam_max in [500, 1000, 5000, 10000]:
        jumps_test = compute_prime_jumps(lam_max)
        F_test = prime_fourier_transform(jumps_test, np.array(true_zeros[:5]))
        amps = np.abs(F_test)
        print(f'  lam_max={lam_max:>6d} ({len(jumps_test):>4d} primes): '
              f'|F| at first 5 zeros = {amps}')

    print('\n' + '#' * 72)
    print('  PRIME RESONANCE COMPLETE')
    print('#' * 72)
