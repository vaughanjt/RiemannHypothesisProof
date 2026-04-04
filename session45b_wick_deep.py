"""
SESSION 45b — THREE-PRONGED WICK INVESTIGATION

A. SPECTRAL WICK ROTATION
   Instead of rotating L, rotate the spectral parameter:
   s = 1/2 + i*gamma  -->  s = 1/2 - delta + i*gamma
   This shifts the Mellin contour off the critical line.
   The "Euclidean barrier" B_E(delta) reveals whether zeros being ON
   the critical line creates a special structure.

B. THE PERIOD 1.085
   This period appeared consistently in Session 45.
   Compute it at multiple L0 to find scaling law.
   What arithmetic object has this period?

C. WINDING NUMBER AND ZEROS
   Phase of B spans nearly [-pi, pi] -- close to full winding.
   Find where |B| is minimized (closest approach to zero).
   Locate actual zeros of B(L0 + i*tau) in the complex L-plane.
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, log, pi, euler, exp, cos, sin,
                    sinh, cosh, hyp2f1, digamma, quad, sqrt, arg, conj,
                    re, im, zetazero)
import time
import sys
import os

mp.dps = 25

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from session41g_uncapped_barrier import sieve_primes, compute_barrier_partial
from session45_wick_rotation import (complex_w02_rayleigh,
                                      complex_mprime_rayleigh,
                                      find_periodicity)


# ═══════════════════════════════════════════════════════════════
# A. SPECTRAL WICK ROTATION
# ═════════════════════════���═════════════════════════════════════

def spectral_wick_barrier(lam_sq, delta=0.0, n_zeros=200, N=None):
    """
    Compute barrier via spectral sum with shifted contour.

    Normal: barrier = sum_rho |H_w(gamma_rho)|^2  at s = 1/2 + i*gamma
    Shifted: barrier_E(delta) = sum_rho |H_w(gamma_rho; delta)|^2
             where H uses s = 1/2 - delta + i*gamma

    If delta=0: standard spectral barrier (on critical line)
    If delta>0: shifted LEFT of critical line (more damping)
    If delta<0: shifted RIGHT of critical line (less damping)

    KEY: if a zero is at s = 1/2 + sigma + i*gamma (off-line by sigma),
    its contribution becomes H_w(gamma; delta-sigma) instead of H_w(gamma; delta).
    This asymmetry in delta is detectable!
    """
    L_f = np.log(lam_sq)
    if N is None:
        N = max(12, round(4 * L_f))

    ns = np.arange(-N, N + 1, dtype=float)
    w_vec = ns / (L_f**2 + (4 * np.pi)**2 * ns**2)
    w_vec[N] = 0.0
    w_hat = w_vec / np.linalg.norm(w_vec)

    # Get zeros
    zeros = []
    for k in range(1, n_zeros + 1):
        z = zetazero(k)
        zeros.append(float(z.imag))
    zeros = np.array(zeros)

    # Integration grid for Mellin transform
    n_quad = 2000
    x_pts = np.linspace(1e-10, L_f, n_quad)
    dx = x_pts[1] - x_pts[0]
    log_x = np.log(x_pts)

    contributions = np.zeros(n_zeros)
    H_values = np.zeros(n_zeros, dtype=complex)

    for z_idx, gamma in enumerate(zeros):
        # x^{-(1/2 - delta) - i*gamma} = x^{-1/2+delta} * exp(-i*gamma*log(x))
        x_factor = x_pts**(-0.5 + delta) * np.exp(-1j * gamma * log_x)

        H = 0.0 + 0.0j
        for i in range(len(ns)):
            n_val = ns[i]
            if abs(w_hat[i]) < 1e-15:
                continue
            omega = 2 * (1 - x_pts / L_f) * np.cos(2 * np.pi * n_val * x_pts / L_f)
            hn = np.sum(omega * x_factor) * dx
            H += w_hat[i] * hn

        H_values[z_idx] = H
        contributions[z_idx] = abs(H)**2

    barrier = np.sum(contributions)

    return {
        'delta': delta,
        'barrier': barrier,
        'contributions': contributions,
        'zeros': zeros,
        'H_values': H_values,
    }


# ═══════════════════════════════════════════════════════════════
# B. PERIOD SCALING
# ═══════════════════════════════════════════════════════════════

def compute_period_at_L0(L0, lam_sq, N=12, n_tau=80, tau_max_frac=0.6):
    """
    Compute the dominant period of the barrier's imaginary variation.
    Uses ONLY the ratio Mp/W02 (bounded, clean periodicity).

    Returns the period and supplementary data.
    """
    four_pi = float(4 * np.pi)
    first_pole = np.sqrt(L0**2 + four_pi**2)
    tau_max = first_pole * tau_max_frac

    taus = np.linspace(-tau_max, tau_max, n_tau)
    ratio_re_vals = []
    ratio_im_vals = []

    for tau in taus:
        L_c = mpc(L0, tau)
        w02 = complex_w02_rayleigh(L_c, N)
        mp_rq = complex_mprime_rayleigh(L_c, lam_sq, N)

        if abs(w02) > 1e-30:
            ratio = mp_rq / w02
            ratio_re_vals.append(float(re(ratio)))
            ratio_im_vals.append(float(im(ratio)))
        else:
            ratio_re_vals.append(0.0)
            ratio_im_vals.append(0.0)

    ratio_re_vals = np.array(ratio_re_vals)
    ratio_im_vals = np.array(ratio_im_vals)

    # FFT for dominant period
    dtau = taus[1] - taus[0]
    centered = ratio_re_vals - ratio_re_vals.mean()
    power = np.abs(np.fft.fft(centered))**2
    freqs = np.fft.fftfreq(len(centered), d=dtau)
    pos = freqs > 0
    if np.any(pos):
        idx = np.argmax(power[pos])
        dom_freq = freqs[pos][idx]
        dom_period = 1.0 / dom_freq if dom_freq > 0 else float('inf')
    else:
        dom_period = float('inf')

    # Also get top 3 frequencies
    pos_freqs = freqs[pos]
    pos_power = power[pos]
    top3_idx = np.argsort(pos_power)[::-1][:3]
    top3 = [(pos_freqs[i], 1.0/pos_freqs[i] if pos_freqs[i] > 0 else float('inf'),
             pos_power[i]) for i in top3_idx]

    return {
        'L0': L0,
        'lam_sq': lam_sq,
        'period': dom_period,
        'top3': top3,
        'tau_range': (-tau_max, tau_max),
        'n_tau': n_tau,
        'ratio_re': ratio_re_vals,
        'ratio_im': ratio_im_vals,
        'taus': taus,
    }


# ═══════════════════════════════════════════════════════════════
# C. WINDING NUMBER AND ZEROS
# ═══════════════════════════════════════════════════════════════

def winding_analysis(L0, lam_sq, N=12, n_tau=200, tau_max_frac=0.7):
    """
    Detailed winding number analysis of B(L0 + i*tau).

    1. Compute arg(B) along the contour
    2. Find total winding = [arg(B(tau_max)) - arg(B(-tau_max))] / (2*pi)
    3. Find |B| minima (closest approach to zero)
    4. If winding ~ integer, there are zeros inside the strip
    """
    four_pi = float(4 * np.pi)
    first_pole = np.sqrt(L0**2 + four_pi**2)
    tau_max = first_pole * tau_max_frac

    taus = np.linspace(-tau_max, tau_max, n_tau)
    b_vals = []
    phases_unwrapped = []

    for j, tau in enumerate(taus):
        L_c = mpc(L0, tau)
        w02 = complex_w02_rayleigh(L_c, N)
        mp_rq = complex_mprime_rayleigh(L_c, lam_sq, N)
        b = w02 - mp_rq
        b_vals.append(complex(b))

    b_arr = np.array(b_vals)
    b_phases = np.angle(b_arr)
    b_phases_unwrap = np.unwrap(b_phases)
    b_abs = np.abs(b_arr)

    # Total winding
    total_phase_change = b_phases_unwrap[-1] - b_phases_unwrap[0]
    winding_number = total_phase_change / (2 * np.pi)

    # Find |B| minima
    min_idx = np.argmin(b_abs)
    min_tau = taus[min_idx]
    min_abs = b_abs[min_idx]

    # Local minima
    local_mins = []
    for i in range(1, len(b_abs) - 1):
        if b_abs[i] < b_abs[i-1] and b_abs[i] < b_abs[i+1]:
            local_mins.append((taus[i], b_abs[i], b_phases[i]))

    return {
        'L0': L0,
        'lam_sq': lam_sq,
        'taus': taus,
        'b_abs': b_abs,
        'b_phases': b_phases,
        'b_phases_unwrap': b_phases_unwrap,
        'winding_number': winding_number,
        'min_tau': min_tau,
        'min_abs': min_abs,
        'local_mins': local_mins,
        'tau_range': (-tau_max, tau_max),
    }


def find_complex_zeros(L0, lam_sq, N=12, tau_range=(-10, 10), n_grid=50):
    """
    Grid search for zeros of B(L0 + i*tau) in the complex plane.
    Uses |B|/|W02| (normalized) to find where the barrier vanishes.

    Also extends to a 2D grid: (L0 + dL) + i*tau for dL in some range,
    to find the nearest zero to the real axis.
    """
    dL_vals = np.linspace(-2, 2, n_grid // 2)
    tau_vals = np.linspace(float(tau_range[0]), float(tau_range[1]), n_grid)

    min_ratio = float('inf')
    min_point = (0, 0)
    results = []

    for dL in dL_vals:
        for tau in tau_vals:
            L_c = mpc(L0 + dL, tau)
            w02 = complex_w02_rayleigh(L_c, N)
            mp_rq = complex_mprime_rayleigh(L_c, lam_sq, N)
            b = w02 - mp_rq
            # Normalized: how close to zero relative to W02
            norm_ratio = float(abs(b) / abs(w02)) if float(abs(w02)) > 1e-30 else float('inf')

            if norm_ratio < min_ratio:
                min_ratio = norm_ratio
                min_point = (L0 + dL, tau)

            results.append((L0 + dL, tau, norm_ratio, float(abs(b))))

    return {
        'min_ratio': min_ratio,
        'min_point': min_point,
        'results': results,
    }


# ══��════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════��═══════════════════════════════════════════

if __name__ == '__main__':
    print()
    print('=' * 72)
    print('  SESSION 45b — THREE-PRONGED WICK INVESTIGATION')
    print('=' * 72)

    LAM_SQ = 2000
    L0 = np.log(LAM_SQ)
    N_BASIS = 12

    # ══════════════════════════════════════════════════════════
    # A. SPECTRAL WICK ROTATION
    # ══════════════════════════════════════════════════════════
    print('\n' + '#' * 72)
    print('  A. SPECTRAL WICK ROTATION — shifting off the critical line')
    print('#' * 72)

    print('\n  Loading zeta zeros...', flush=True)
    t0 = time.time()
    n_zeros = 150

    # Scan delta from -0.2 to +0.2 (shift off critical line)
    deltas = np.concatenate([
        np.linspace(-0.20, -0.01, 8),
        [0.0],
        np.linspace(0.01, 0.20, 8),
    ])

    print(f'\n  Spectral barrier B_E(delta) for delta in [{deltas[0]:.2f}, {deltas[-1]:.2f}]')
    print(f'  lam^2 = {LAM_SQ}, {n_zeros} zeros')
    print(f'\n  {"delta":>8s} {"B_E(delta)":>14s} {"B_E/B_E(0)":>12s} '
          f'{"top_zero":>10s} {"top_%":>8s}')
    print('  ' + '-' * 58)

    spectral_results = []
    b_at_0 = None
    for delta in deltas:
        t1 = time.time()
        r = spectral_wick_barrier(LAM_SQ, delta=delta, n_zeros=n_zeros, N=N_BASIS)
        dt = time.time() - t1
        spectral_results.append(r)

        if delta == 0.0:
            b_at_0 = r['barrier']

        top_idx = np.argmax(r['contributions'])
        top_pct = r['contributions'][top_idx] / r['barrier'] * 100 if r['barrier'] > 0 else 0

        ratio = r['barrier'] / b_at_0 if b_at_0 and b_at_0 > 0 else 0.0
        print(f'  {delta:>+8.3f} {r["barrier"]:>14.6f} {ratio:>12.4f} '
              f'  gamma_{top_idx+1:>3d} {top_pct:>7.1f}%  ({dt:.1f}s)')
        sys.stdout.flush()

    # Symmetry analysis: B_E(delta) vs B_E(-delta)
    print(f'\n  SYMMETRY TEST: B_E(delta) vs B_E(-delta)')
    print(f'  If zeros ON critical line: B_E should be symmetric in delta')
    print(f'  If zeros OFF critical line: asymmetry emerges')
    print(f'\n  {"delta":>8s} {"B_E(+d)":>14s} {"B_E(-d)":>14s} '
          f'{"ratio":>10s} {"asym":>12s}')
    print('  ' + '-' * 62)

    for i in range(len(deltas) // 2):
        d_pos = deltas[-(i+1)]
        d_neg = deltas[i]
        if abs(abs(d_pos) - abs(d_neg)) > 0.001:
            continue
        b_pos = spectral_results[-(i+1)]['barrier']
        b_neg = spectral_results[i]['barrier']
        ratio = b_pos / b_neg if b_neg > 0 else float('inf')
        asym = (b_pos - b_neg) / (b_pos + b_neg) * 2 if (b_pos + b_neg) > 0 else 0
        print(f'  {abs(d_pos):>+8.3f} {b_pos:>14.6f} {b_neg:>14.6f} '
              f'{ratio:>10.4f} {asym:>+12.6f}')

    # Derivative at delta=0 (spectral "temperature")
    if b_at_0 and b_at_0 > 0:
        # Central difference
        idx_0 = list(deltas).index(0.0)
        if idx_0 > 0 and idx_0 < len(deltas) - 1:
            d_minus = spectral_results[idx_0 - 1]['barrier']
            d_plus = spectral_results[idx_0 + 1]['barrier']
            dd = deltas[idx_0 + 1] - deltas[idx_0 - 1]
            deriv = (d_plus - d_minus) / dd
            print(f'\n  dB_E/d(delta) at delta=0: {deriv:+.6f}')
            print(f'  Normalized: (1/B)*dB/d(delta) = {deriv/b_at_0:+.6f}')
            if abs(deriv/b_at_0) < 0.01:
                print(f'  *** NEAR-ZERO DERIVATIVE: barrier is EXTREMAL at critical line ***')
                print(f'  This means Re(s)=1/2 is a stationary point of the spectral entropy!')
            elif deriv > 0:
                print(f'  Barrier increases when shifting RIGHT — critical line is a minimum')
            else:
                print(f'  Barrier increases when shifting LEFT — critical line is a maximum')

    # Convexity at delta=0
    if b_at_0 and idx_0 > 0 and idx_0 < len(deltas) - 1:
        d2 = (d_plus - 2 * b_at_0 + d_minus) / ((deltas[idx_0+1] - deltas[idx_0])**2)
        print(f'  d^2 B_E / d(delta)^2 at delta=0: {d2:+.6f}')
        if d2 > 0:
            print(f'  *** CONVEX: critical line is a LOCAL MINIMUM of B_E ***')
            print(f'  Boyle-Turok analog: flat universe minimizes a potential → maximum entropy')
            print(f'  Our analog: critical line minimizes spectral barrier → maximum "entropy"')
        elif d2 < 0:
            print(f'  CONCAVE: critical line is a local maximum of B_E')

    # H_w decay comparison at different delta
    print(f'\n  |H_w| DECAY RATE vs delta:')
    for r in [spectral_results[0], spectral_results[len(deltas)//2],
              spectral_results[-1]]:
        amps = np.sqrt(r['contributions'][10:])
        gammas = r['zeros'][10:]
        valid = amps > 1e-20
        if np.sum(valid) > 5:
            c = np.polyfit(np.log(gammas[valid]), np.log(amps[valid]), 1)
            print(f'    delta={r["delta"]:+.3f}: |H_w| ~ gamma^{{{c[0]:.3f}}}')

    # ══════════════════════════════════════════════════════════
    # B. THE PERIOD 1.085
    # ══════════════════════════════════════════════════════════
    print('\n\n' + '#' * 72)
    print('  B. THE PERIOD 1.085 — what is it?')
    print('#' * 72)

    # Compute period at multiple L0 values
    test_cases = [
        (np.log(200), 200),
        (np.log(500), 500),
        (np.log(1000), 1000),
        (np.log(2000), 2000),
        (np.log(5000), 5000),
        (np.log(10000), 10000),
    ]

    print(f'\n  {"L0":>7s} {"lam^2":>7s} {"period":>10s} '
          f'{"period*L":>10s} {"period/L":>10s} {"2pi/L":>10s} {"per*L/2pi":>10s}')
    print('  ' + '-' * 72)

    period_data = []
    for L0_val, lsq in test_cases:
        t1 = time.time()
        r = compute_period_at_L0(L0_val, lsq, N=10, n_tau=100)
        dt = time.time() - t1

        p = r['period']
        period_data.append((L0_val, p))
        two_pi_over_L = 2 * np.pi / L0_val
        print(f'  {L0_val:>7.3f} {lsq:>7d} {p:>10.4f} '
              f'{p*L0_val:>10.4f} {p/L0_val:>10.4f} {two_pi_over_L:>10.4f} '
              f'{p*L0_val/(2*np.pi):>10.4f}  ({dt:.0f}s)')

        # Report top 3 frequencies
        print(f'          Top freqs: ', end='')
        for freq, per, pw in r['top3']:
            print(f'T={per:.3f} (pow={pw:.1e})  ', end='')
        print()
        sys.stdout.flush()

    # Fit period vs L0
    L0s = np.array([x[0] for x in period_data])
    periods = np.array([x[1] for x in period_data])

    # Test: period ~ a/L? period ~ a? period ~ a*L?
    print(f'\n  SCALING ANALYSIS:')

    # period = a/L ?
    prods = periods * L0s
    print(f'  period * L range: [{prods.min():.4f}, {prods.max():.4f}] '
          f'(constant = {np.mean(prods):.4f} +/- {np.std(prods):.4f})')

    # period = const ?
    print(f'  period range: [{periods.min():.4f}, {periods.max():.4f}] '
          f'(mean = {np.mean(periods):.4f} +/- {np.std(periods):.4f})')

    # period = a*L ?
    ratios = periods / L0s
    print(f'  period / L range: [{ratios.min():.4f}, {ratios.max():.4f}] '
          f'(mean = {np.mean(ratios):.4f} +/- {np.std(ratios):.4f})')

    # period * L / (2*pi) = ???
    scaled = periods * L0s / (2 * np.pi)
    print(f'  period * L / (2*pi): [{scaled.min():.4f}, {scaled.max():.4f}] '
          f'(mean = {np.mean(scaled):.4f} +/- {np.std(scaled):.4f})')

    # Which is most constant?
    cvs = [np.std(prods)/np.mean(prods), np.std(periods)/np.mean(periods),
           np.std(ratios)/np.mean(ratios), np.std(scaled)/np.mean(scaled)]
    labels = ['period*L', 'period', 'period/L', 'period*L/(2pi)']
    best_idx = np.argmin(cvs)
    print(f'\n  Most constant: {labels[best_idx]} (CV = {cvs[best_idx]:.4f})')
    for l, cv in zip(labels, cvs):
        print(f'    {l:>15s}: CV = {cv:.4f}')

    # Check if period relates to first zeta zero
    gamma_1 = 14.1347  # first zeta zero
    print(f'\n  Comparison with zeta zero scales:')
    print(f'    gamma_1 = {gamma_1:.4f}')
    print(f'    2*pi/gamma_1 = {2*np.pi/gamma_1:.4f}')
    print(f'    gamma_1 / (4*pi) = {gamma_1/(4*np.pi):.4f}')
    for L0_val, p in period_data:
        print(f'    L0={L0_val:.3f}: period/gamma_1 = {p/gamma_1:.4f}, '
              f'period*gamma_1 = {p*gamma_1:.4f}')

    # ══════════════════════════════════════════════════════════
    # C. WINDING NUMBER AND ZEROS
    # ═══��══════════════════════════════════════════════════════
    print('\n\n' + '#' * 72)
    print('  C. WINDING NUMBER AND ZEROS')
    print('#' * 72)

    # Winding at multiple L0 values
    print(f'\n  Winding analysis:')
    print(f'  {"L0":>7s} {"lam^2":>7s} {"winding":>10s} {"min|B|":>12s} '
          f'{"min_tau":>10s} {"n_local_min":>12s}')
    print('  ' + '-' * 65)

    winding_data = []
    for L0_val, lsq in [(np.log(500), 500), (np.log(1000), 1000),
                         (np.log(2000), 2000), (np.log(5000), 5000)]:
        t1 = time.time()
        w = winding_analysis(L0_val, lsq, N=10, n_tau=200)
        dt = time.time() - t1
        winding_data.append(w)

        print(f'  {L0_val:>7.3f} {lsq:>7d} {w["winding_number"]:>+10.4f} '
              f'{w["min_abs"]:>12.4e} {w["min_tau"]:>+10.3f} '
              f'{len(w["local_mins"]):>12d}  ({dt:.0f}s)')

        # Local minima detail
        if w['local_mins']:
            print(f'          Local |B| minima:')
            for tau_m, abs_m, ph_m in sorted(w['local_mins'], key=lambda x: x[1])[:5]:
                print(f'            tau={tau_m:+8.3f}: |B|={abs_m:.4e}, phase={ph_m:+.4f}')
        sys.stdout.flush()

    # Detailed winding at main L0
    print(f'\n  Detailed winding at L0 = {L0:.3f}:')
    w_main = winding_analysis(L0, LAM_SQ, N=N_BASIS, n_tau=400)
    print(f'    Total winding: {w_main["winding_number"]:+.6f}')
    print(f'    Global min |B|: {w_main["min_abs"]:.6e} at tau = {w_main["min_tau"]:+.3f}')
    print(f'    Number of local minima: {len(w_main["local_mins"])}')

    # Phase profile at key points
    print(f'\n    Phase profile (unwrapped):')
    step = max(1, len(w_main['taus']) // 20)
    for i in range(0, len(w_main['taus']), step):
        t = w_main['taus'][i]
        ph = w_main['b_phases_unwrap'][i]
        ab = w_main['b_abs'][i]
        print(f'      tau={t:+8.3f}: phase={ph:+8.4f} rad ({ph/(2*np.pi):+.4f} turns), '
              f'log|B|={np.log10(ab+1e-300):+8.2f}')

    # Zero search in 2D grid
    print(f'\n  Searching for zeros of B in complex L-plane...')
    t1 = time.time()
    z_search = find_complex_zeros(L0, LAM_SQ, N=10, tau_range=(-8, 8), n_grid=30)
    dt = time.time() - t1
    print(f'    Nearest approach to zero: |B|/|W02| = {z_search["min_ratio"]:.6e}')
    print(f'    At L = {z_search["min_point"][0]:.3f} + i*{z_search["min_point"][1]:.3f}')
    print(f'    Search time: {dt:.1f}s')

    # ══════════════════════════════════════════════════════════
    # SYNTHESIS
    # ═══════════════���══════════════════════════════════════════
    print('\n\n' + '=' * 72)
    print('  SESSION 45b SYNTHESIS')
    print('=' * 72)

    print(f'''
  A. SPECTRAL WICK ROTATION:
     B_E(delta) = spectral barrier shifted delta off the critical line.
     B_E(0) = {b_at_0:.6f}
     Derivative at delta=0: measures whether critical line is special.
     Symmetry B_E(+d)/B_E(-d): tests if zeros are truly on Re(s)=1/2.

  B. THE PERIOD:
     Dominant period in the imaginary L-direction.
     Scaling with L0 determines arithmetic content.
     If period ~ constant: intrinsic scale (like 2*pi)
     If period ~ 1/L: comes from mode spacing
     If period*L/(2*pi) ~ constant: Fourier dual of mode structure

  C. WINDING NUMBER:
     B(L0+i*tau) traces a curve in the complex plane.
     Winding at L0={L0:.3f}: {w_main["winding_number"]:+.4f}
     Winding ~ 0: no zeros enclosed → barrier never vanishes
     Winding ~ ±1: one zero enclosed → barrier has a complex zero
     Nearest approach: |B|/|W02| = {z_search["min_ratio"]:.2e}

  BOYLE-TUROK CONNECTION:
     Their entropy S_g is maximized at flat (kappa=0) universe.
     Our spectral barrier B_E(delta) behavior at delta=0 tells us
     whether the critical line (Re(s)=1/2) is the analog of "flat":
     a stationary point of the spectral entropy.
''')

    print('=' * 72)
    print('  SESSION 45b COMPLETE')
    print('=' * 72)
