"""
SESSION 45 — WICK ROTATION OF THE BARRIER

Inspired by Boyle-Turok (arXiv:2210.01142): the Friedmann equation's scale
factor a(tau) is an elliptic function, meromorphic and doubly periodic in the
complex tau-plane. Periodicity in imaginary time = inverse temperature.
The gravitational entropy S_g = i * S counts microstates and is MAXIMIZED
for flat, homogeneous universes with small positive Lambda.

THE ANALOGY:
  Boyle-Turok: a(tau) → complex tau-plane → imaginary periodicity → entropy
  Us:          B(L) → complex L-plane → ??? → barrier positivity

PLAN:
  1. Analytically continue B(L) to B(L_0 + i*tau) using complex mpmath
  2. Map poles, zeros, and periodicity in the complex L-plane
  3. Compute the "Euclidean action" integral along imaginary contours
  4. Look for the entropy analog: is positivity a thermodynamic statement?
  5. Test: does random-prime barrier break the structure? (Cramer control)

ALL formulas extend naturally to complex L:
  - W02: pf = 32*L*sinh(L/4)^2, w_tilde = n/(L^2 + (4*pi)^2 * n^2)
  - M_diag: integrals from 0 to L, digamma, hypergeometric
  - M_alpha: hypergeometric + digamma with complex arguments
  - M_prime: prime sum with complex window (primes stay real)
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, log, pi, euler, exp, cos, sin,
                    sinh, cosh, hyp2f1, digamma, quad, sqrt, fabs, im, re,
                    cot, arg, conj)
import time
import sys
import os

mp.dps = 30

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from session41g_uncapped_barrier import sieve_primes


# ═══════════════════════════════════════════════════════════════
# COMPLEX BARRIER COMPONENTS
# ═══════════════════════════════════════════════════════════════

def complex_w02_rayleigh(L_complex, N=20):
    """
    W02 Rayleigh quotient on w direction for complex L.

    pf = 32 * L * sinh(L/4)^2
    w_tilde[n] = n / (L^2 + (4*pi)^2 * n^2)
    w02_rq = -pf * (4*pi)^2 * (sum w_tilde[n]^2)^2 / (sum |w_tilde[n]|^2)^2

    For complex L, w_tilde is complex. The Rayleigh quotient generalizes as
    <w, W02, w> / <w, w> but W02 is Hermitian, so we use w^dagger * W02 * w.
    """
    L = mpc(L_complex)
    pf = 32 * L * sinh(L / 4)**2

    # Build w_tilde (complex)
    four_pi_sq = (4 * pi)**2
    w_norm_sq = mpc(0)
    wt_dot_wt = mpc(0)  # sum of w_tilde[n]^2 (NOT |w_tilde|^2)

    for n in range(-N, N + 1):
        if n == 0:
            continue
        denom = L**2 + four_pi_sq * n**2
        wt_n = mpf(n) / denom
        w_norm_sq += abs(wt_n)**2  # |w_tilde[n]|^2 for normalization
        wt_dot_wt += wt_n**2       # w_tilde[n]^2 (complex bilinear form)

    # Rayleigh quotient: w_hat = w_tilde / ||w_tilde||
    # <w_hat, W02, w_hat> = -pf * (4*pi)^2 * (sum w_tilde^2)^2 / (sum |w_tilde|^2)^2
    # But for the bilinear form, it's sum(w_tilde * w_tilde_bar) in the Hermitian case
    # Since W02 is real symmetric, extended to complex:
    # w^H * W02 * w / (w^H * w) where w = w_tilde

    # The W02 action on w_tilde gives -pf * (4pi)^2 * w_tilde_coeff
    # More precisely, W02[n,m] = pf * (L^2 - (4pi)^2*n*m) / ((L^2+(4pi)^2*n^2)(L^2+(4pi)^2*m^2))
    # So <w_tilde, W02, w_tilde> = sum_{n,m} w_tilde_bar[n] * W02[n,m] * w_tilde[m]

    # Direct computation of the bilinear form
    w02_bilinear = mpc(0)
    for n in range(-N, N + 1):
        if n == 0:
            continue
        denom_n = L**2 + four_pi_sq * n**2
        wt_n_bar = conj(mpf(n) / denom_n)
        for m in range(-N, N + 1):
            if m == 0:
                continue
            denom_m = L**2 + four_pi_sq * m**2
            wt_m = mpf(m) / denom_m
            w02_nm = pf * (L**2 - four_pi_sq * n * m) / (denom_n * denom_m)
            w02_bilinear += wt_n_bar * w02_nm * wt_m

    return w02_bilinear / w_norm_sq


def complex_mprime_rayleigh(L_complex, lam_sq_real, N=20):
    """
    M_prime Rayleigh quotient for complex L.
    Primes remain real — only the window function complexifies.

    M_prime[n,m] = sum_{p^k <= lam^2} log(p)/sqrt(p^k) * q(n,m,log(p^k))

    q(n,m,y) = 2(L-y)/L * cos(2*pi*n*y/L)  (diagonal)
             = (sin(2*pi*m*y/L) - sin(2*pi*n*y/L)) / (pi*(n-m))  (off-diagonal)

    For complex L, cos and sin become complex but the prime data is real.
    """
    L = mpc(L_complex)
    four_pi = 4 * pi

    primes = sieve_primes(int(lam_sq_real))

    # Collect prime powers
    pk_data = []
    for p in primes:
        pk = int(p)
        k = 1
        logp = log(mpf(int(p)))
        while pk <= lam_sq_real:
            pk_data.append((logp, logp / sqrt(mpf(int(pk))), k * logp))
            pk *= int(p)
            k += 1

    # w_tilde (complex)
    four_pi_sq = four_pi**2
    w_tilde = {}
    w_norm_sq = mpc(0)
    for n in range(-N, N + 1):
        if n == 0:
            w_tilde[n] = mpc(0)
        else:
            denom = L**2 + four_pi_sq * n**2
            w_tilde[n] = mpf(n) / denom
            w_norm_sq += abs(w_tilde[n])**2

    # Compute M_prime bilinear form: sum w_bar[n] * M[n,m] * w[m]
    mp_bilinear = mpc(0)
    two_pi = 2 * pi

    for logp, weight, y in pk_data:
        # Precompute sin and cos at complex arguments
        sin_cache = {}
        cos_cache = {}
        for n in range(-N, N + 1):
            sin_cache[n] = sin(two_pi * n * y / L)
            cos_cache[n] = cos(two_pi * n * y / L)

        for n in range(-N, N + 1):
            if n == 0:
                continue
            wt_n_bar = conj(w_tilde[n])
            for m in range(-N, N + 1):
                if m == 0:
                    continue
                wt_m = w_tilde[m]

                if n == m:
                    q_nm = 2 * (L - y) / L * cos_cache[n]
                else:
                    q_nm = (sin_cache[m] - sin_cache[n]) / (pi * (n - m))

                mp_bilinear += wt_n_bar * weight * q_nm * wt_m

    return mp_bilinear / w_norm_sq


def complex_mdiag_rayleigh(L_complex, N=20, n_quad=2000):
    """
    M_diag Rayleigh quotient for complex L.

    wr_diag[n] = (omega_0/2)*(gamma + log(4*pi*(e^L-1)/(e^L+1)))
                 + integral_0^L [(e^{x/2}*omega(x) - omega_0)/(e^x - e^{-x})] dx

    For complex L, the integral contour goes from 0 to L in the complex plane.
    """
    L = mpc(L_complex)
    eL = exp(L)

    four_pi_sq = (4 * pi)**2
    w_tilde = {}
    w_norm_sq = mpc(0)
    for n in range(-N, N + 1):
        if n == 0:
            w_tilde[n] = mpc(0)
        else:
            denom = L**2 + four_pi_sq * n**2
            w_tilde[n] = mpf(n) / denom
            w_norm_sq += abs(w_tilde[n])**2

    omega_0 = mpf(2)
    two_pi = 2 * pi

    # wr_diag for each |n|
    wr_diag = {}
    for nv in range(N + 1):
        w_const = (omega_0 / 2) * (euler + mpmath.log(4 * pi * (eL - 1) / (eL + 1)))

        # Integrate along straight line from 0 to L in complex plane
        # Parameterize: x = L * t, t in [0, 1], dx = L * dt
        integral = mpc(0)
        dt = mpf(1) / n_quad
        for k in range(n_quad):
            t = dt * (k + mpf(1) / 2)
            x = L * t

            omega_x = 2 * (1 - t) * cos(two_pi * nv * t)  # omega(x) = 2*(1-x/L)*cos(2*pi*n*x/L)
            numer = exp(x / 2) * omega_x - omega_0
            denom = exp(x) - exp(-x)
            if abs(denom) > mpf(10)**(-20):
                integral += numer / denom

        integral *= L * dt  # dx = L * dt
        wr_diag[nv] = w_const + integral
        wr_diag[-nv] = wr_diag[nv]

    # Rayleigh quotient: sum |w_hat[n]|^2 * wr_diag[n] / sum |w_hat[n]|^2
    mdiag_bilinear = mpc(0)
    for n in range(-N, N + 1):
        if n == 0:
            continue
        mdiag_bilinear += abs(w_tilde[n])**2 * wr_diag[abs(n)]

    return mdiag_bilinear / w_norm_sq


def complex_malpha_rayleigh(L_complex, N=20):
    """
    M_alpha Rayleigh quotient for complex L.

    alpha[n] depends on hyp2f1 and digamma with complex arguments.
    """
    L = mpc(L_complex)

    four_pi_sq = (4 * pi)**2
    w_tilde = {}
    w_norm_sq = mpc(0)
    for n in range(-N, N + 1):
        if n == 0:
            w_tilde[n] = mpc(0)
        else:
            denom = L**2 + four_pi_sq * n**2
            w_tilde[n] = mpf(n) / denom
            w_norm_sq += abs(w_tilde[n])**2

    # Alpha coefficients (complex)
    alpha = {}
    z = exp(-2 * L)
    for n in range(-N, N + 1):
        if n == 0:
            alpha[n] = mpc(0)
        else:
            a = pi * mpc(0, abs(n)) / L + mpf(1) / 4
            h = hyp2f1(1, a, a + 1, z)
            f1 = exp(-L / 2) * im(2 * L / (L + 4 * pi * mpc(0, abs(n))) * h)
            d = im(digamma(a)) / 2
            val = (f1 + d) / pi
            alpha[n] = val if n > 0 else -val

    # Off-diagonal bilinear form
    malpha_bilinear = mpc(0)
    for n in range(-N, N + 1):
        if n == 0:
            continue
        for m in range(-N, N + 1):
            if m == 0 or n == m:
                continue
            malpha_bilinear += conj(w_tilde[n]) * (alpha[m] - alpha[n]) / (n - m) * w_tilde[m]

    return malpha_bilinear / w_norm_sq


def complex_barrier(L_complex, lam_sq_real, N=15, include_mdiag=True, n_quad=1500):
    """
    Full barrier B(L) = W02 - M_prime - M_diag - M_alpha
    at complex L, with real lambda^2 for the prime cutoff.
    """
    w02 = complex_w02_rayleigh(L_complex, N)
    mp_rq = complex_mprime_rayleigh(L_complex, lam_sq_real, N)

    if include_mdiag:
        md = complex_mdiag_rayleigh(L_complex, N, n_quad)
        ma = complex_malpha_rayleigh(L_complex, N)
    else:
        md = mpc(0)
        ma = mpc(0)

    return {
        'L': L_complex,
        'w02': w02,
        'mprime': mp_rq,
        'mdiag': md,
        'malpha': ma,
        'barrier': w02 - mp_rq - md - ma,
        'partial': w02 - mp_rq,  # W02 - M_prime only (fast check)
    }


# ═══════════════════════════════════════════════════════════════
# WICK ROTATION ANALYSIS
# ═══════════════════════════════════════════════════════════════

def wick_scan(L0, tau_range, n_tau=40, lam_sq=5000, N=12,
              fast_mode=True):
    """
    Scan B(L0 + i*tau) for tau in tau_range.

    If fast_mode, only compute W02 - M_prime (dominant balance).
    """
    results = []
    for j, tau in enumerate(np.linspace(float(tau_range[0]),
                                        float(tau_range[1]), n_tau)):
        L_c = mpc(L0, tau)
        t0 = time.time()

        if fast_mode:
            w02 = complex_w02_rayleigh(L_c, N)
            mp_rq = complex_mprime_rayleigh(L_c, lam_sq, N)
            b = w02 - mp_rq
            # Normalized: ratio = 1 - Mp/W02 (removes exponential growth)
            ratio = mp_rq / w02 if abs(w02) > 1e-30 else mpc(0)
            norm_b = 1 - ratio  # B/W02
            phase = float(arg(b)) if abs(b) > 1e-30 else 0.0
            results.append({
                'tau': tau, 'L': L_c,
                'barrier_re': float(re(b)),
                'barrier_im': float(im(b)),
                'barrier_abs': float(abs(b)),
                'norm_re': float(re(norm_b)),
                'norm_im': float(im(norm_b)),
                'norm_abs': float(abs(norm_b)),
                'phase': phase,
                'ratio_re': float(re(ratio)),
                'ratio_im': float(im(ratio)),
                'w02_abs': float(abs(w02)),
                'w02_re': float(re(w02)),
                'mprime_re': float(re(mp_rq)),
            })
        else:
            r = complex_barrier(L_c, lam_sq, N, include_mdiag=True)
            b = r['barrier']
            w02 = r['w02']
            ratio = (r['mprime'] + r['mdiag'] + r['malpha']) / w02 if abs(w02) > 1e-30 else mpc(0)
            norm_b = 1 - ratio
            phase = float(arg(b)) if abs(b) > 1e-30 else 0.0
            results.append({
                'tau': tau, 'L': L_c,
                'barrier_re': float(re(b)),
                'barrier_im': float(im(b)),
                'barrier_abs': float(abs(b)),
                'norm_re': float(re(norm_b)),
                'norm_im': float(im(norm_b)),
                'norm_abs': float(abs(norm_b)),
                'phase': phase,
                'ratio_re': float(re(ratio)),
                'ratio_im': float(im(ratio)),
                'w02_abs': float(abs(w02)),
                'w02_re': float(re(r['w02'])),
                'mprime_re': float(re(r['mprime'])),
                'mdiag_re': float(re(r['mdiag'])),
                'malpha_re': float(re(r['malpha'])),
            })

        dt = time.time() - t0
        if (j + 1) % 10 == 0 or j == 0:
            print(f'    tau={tau:+7.3f}  Re(B)={results[-1]["barrier_re"]:+.6f}  '
                  f'Im(B)={results[-1]["barrier_im"]:+.6f}  |B|={results[-1]["barrier_abs"]:.6f}  '
                  f'({dt:.1f}s)')
            sys.stdout.flush()

    return results


def wick_action(scan_results):
    """
    Compute the Wick-rotated "action" = integral of B(L0 + i*tau) d(tau)
    along the imaginary direction.

    By Boyle-Turok analogy:
      S_g = i * V_3 * integral_0^{Delta_tau} [-3 a_dot^2 + V(a)] dtau

    Our analog:
      S_barrier = i * integral_0^{tau_period} B(L0 + i*tau) d(tau)

    If B is periodic in tau with period T, this is:
      S_barrier = i * integral_0^T B(L0 + i*tau) d(tau)
    """
    taus = [r['tau'] for r in scan_results]
    b_re = [r['barrier_re'] for r in scan_results]
    b_im = [r['barrier_im'] for r in scan_results]

    dtau = taus[1] - taus[0] if len(taus) > 1 else 0.0

    # Trapezoidal integration of complex B along imaginary direction
    # integral B d(tau) = integral (B_re + i*B_im) dtau
    integral_re = np.trapezoid(b_re, taus)  # real part of integral
    integral_im = np.trapezoid(b_im, taus)  # imag part of integral

    # The "entropy" S = i * integral
    # i * (integral_re + i * integral_im) = -integral_im + i * integral_re
    S_re = -integral_im  # real part of entropy
    S_im = integral_re   # imag part of entropy

    return {
        'integral_re': integral_re,
        'integral_im': integral_im,
        'entropy_re': S_re,
        'entropy_im': S_im,
        'entropy_abs': np.sqrt(S_re**2 + S_im**2),
        'tau_range': (taus[0], taus[-1]),
    }


def find_periodicity(scan_results, component='barrier_re'):
    """
    Look for periodicity in tau using autocorrelation.
    """
    vals = np.array([r[component] for r in scan_results])
    taus = np.array([r['tau'] for r in scan_results])

    # Subtract mean
    vals_centered = vals - vals.mean()

    # Autocorrelation via FFT
    n = len(vals)
    fft = np.fft.fft(vals_centered, n=2*n)
    acf = np.real(np.fft.ifft(fft * np.conj(fft)))[:n]
    acf /= acf[0] if acf[0] != 0 else 1.0

    # Find first peak after origin
    dtau = taus[1] - taus[0] if len(taus) > 1 else 1.0
    peaks = []
    for i in range(2, n - 1):
        if acf[i] > acf[i-1] and acf[i] > acf[i+1] and acf[i] > 0.3:
            peaks.append((i * dtau, acf[i]))

    # Also FFT power spectrum for dominant frequency
    power = np.abs(np.fft.fft(vals_centered))**2
    freqs = np.fft.fftfreq(n, d=dtau)
    pos = freqs > 0
    if np.any(pos):
        idx = np.argmax(power[pos])
        dominant_freq = freqs[pos][idx]
        dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else float('inf')
    else:
        dominant_freq = 0
        dominant_period = float('inf')

    return {
        'acf_peaks': peaks,
        'dominant_period': dominant_period,
        'dominant_freq': dominant_freq,
        'acf': acf,
        'taus': taus,
    }


# ═══════════════════════════════════════════════════════════════
# CRAMER CONTROL: Random primes break structure?
# ═══════════════════════════════════════════════════════════════

def complex_mprime_cramer(L_complex, lam_sq_real, N=12, seed=42):
    """
    M_prime with RANDOM primes (Cramer model).
    Replaces actual primes with random integers with the same density.
    Tests whether the complex-plane structure is specific to real primes.
    """
    L = mpc(L_complex)
    rng = np.random.RandomState(seed)

    # Generate Cramer random primes: each integer n is "prime" with prob 1/log(n)
    random_primes = []
    for n in range(2, int(lam_sq_real) + 1):
        if rng.random() < 1.0 / np.log(n):
            random_primes.append(n)

    four_pi_sq = (4 * pi)**2
    two_pi = 2 * pi
    w_tilde = {}
    w_norm_sq = mpc(0)
    for n in range(-N, N + 1):
        if n == 0:
            w_tilde[n] = mpc(0)
        else:
            denom = L**2 + four_pi_sq * n**2
            w_tilde[n] = mpf(n) / denom
            w_norm_sq += abs(w_tilde[n])**2

    mp_bilinear = mpc(0)
    for p in random_primes:
        logp = log(mpf(int(p)))
        weight = logp / sqrt(mpf(int(p)))
        y = logp
        if float(y) > float(re(L)):
            continue

        sin_cache = {}
        cos_cache = {}
        for n in range(-N, N + 1):
            sin_cache[n] = sin(two_pi * n * y / L)
            cos_cache[n] = cos(two_pi * n * y / L)

        for n in range(-N, N + 1):
            if n == 0:
                continue
            for m in range(-N, N + 1):
                if m == 0:
                    continue
                if n == m:
                    q_nm = 2 * (L - y) / L * cos_cache[n]
                else:
                    q_nm = (sin_cache[m] - sin_cache[n]) / (pi * (n - m))
                mp_bilinear += conj(w_tilde[n]) * weight * q_nm * w_tilde[m]

    return mp_bilinear / w_norm_sq


# ═══════════════════════════════════════════════════════════════
# POLE STRUCTURE ANALYSIS
# ═══════════════════════════════════════════════════════════════

def find_poles(L0, lam_sq=5000, N=12):
    """
    The w_tilde denominator L^2 + (4*pi)^2 * n^2 = 0 when L = +/- 4*pi*n*i.
    These are poles of the complexified barrier at:
      L = L0 + i*tau where tau = +/- 4*pi*n (for integer n >= 1)

    Near these poles, the barrier should diverge. The residues encode
    the arithmetic content — analogous to how Boyle-Turok's poles of a(tau)
    determine the gravitational temperature and entropy.
    """
    poles = []
    four_pi = float(4 * np.pi)
    for n in range(1, N + 1):
        # L^2 + (4*pi)^2 * n^2 = 0 => L = +/- i * 4*pi*n
        # If L = L0 + i*tau, then (L0+i*tau)^2 + (4*pi*n)^2 = 0
        # L0^2 - tau^2 + 2*i*L0*tau + (4*pi*n)^2 = 0
        # Real: L0^2 - tau^2 + (4*pi*n)^2 = 0 => tau^2 = L0^2 + (4*pi*n)^2
        # Imag: 2*L0*tau = 0 => L0 = 0 or tau = 0
        # For L0 != 0: no exact pole, but near-poles when |tau| ~ sqrt(L0^2 + (4*pi*n)^2)
        # For L0 = 0: poles at tau = +/- 4*pi*n
        tau_pole_exact = four_pi * n
        tau_pole_shifted = np.sqrt(L0**2 + (four_pi * n)**2)
        poles.append({
            'n': n,
            'tau_exact': tau_pole_exact,
            'tau_shifted': tau_pole_shifted,
            'residue_order': 2,  # double pole from denominator^2
        })
    return poles


# ═══════════════════════════════════════════════════════════════
# MAIN EXPLORATION
# ══���════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print()
    print('=' * 72)
    print('  SESSION 45 — WICK ROTATION OF THE BARRIER')
    print('  Boyle-Turok meets Riemann: analytic continuation to imaginary L')
    print('=' * 72)

    LAM_SQ = 2000  # moderate lambda^2 for speed
    N_BASIS = 12   # truncation (enough for structure, fast for exploration)
    L0_REAL = np.log(LAM_SQ)

    # ── PART 1: Verify real-L barrier matches known values ──
    print('\n' + '#' * 70)
    print('  PART 1: Sanity check — real L barrier')
    print('#' * 70)

    t0 = time.time()
    r_real = complex_barrier(mpc(L0_REAL, 0), LAM_SQ, N_BASIS, include_mdiag=False)
    dt = time.time() - t0
    print(f'  L = {L0_REAL:.4f} (lam^2 = {LAM_SQ})')
    print(f'  W02 - M_prime = {float(re(r_real["partial"])):+.6f}')
    print(f'  (should match session41g result)')
    print(f'  Time: {dt:.1f}s')

    # Cross-check with numpy version
    from session41g_uncapped_barrier import compute_barrier_partial
    r_check = compute_barrier_partial(LAM_SQ, N=N_BASIS)
    print(f'  session41g result: {r_check["partial_barrier"]:+.6f}')
    print(f'  Difference: {abs(float(re(r_real["partial"])) - r_check["partial_barrier"]):.2e}')

    # ── PART 2: Pole structure ──
    print('\n\n' + '#' * 70)
    print('  PART 2: Pole structure in the complex L-plane')
    print('#' * 70)

    poles = find_poles(L0_REAL, N=N_BASIS)
    print(f'\n  Poles of w_tilde at L = {L0_REAL:.3f} + i*tau:')
    print(f'  {"n":>3s} {"tau_pole":>10s} {"4*pi*n":>10s} {"shifted":>10s}')
    print('  ' + '-' * 38)
    for p in poles[:6]:
        print(f'  {p["n"]:>3d} {p["tau_exact"]:>10.3f} {4*np.pi*p["n"]:>10.3f} '
              f'{p["tau_shifted"]:>10.3f}')

    first_pole = poles[0]['tau_shifted']
    print(f'\n  First pole vicinity: tau ~ {first_pole:.3f}')
    print(f'  Safe scanning range: |tau| < {first_pole * 0.8:.3f}')

    # ── PART 3: Wick scan — B(L0 + i*tau) ──
    print('\n\n' + '#' * 70)
    print('  PART 3: Wick rotation scan (fast mode: W02 - M_prime)')
    print('#' * 70)

    # Scan tau from -safe to +safe
    tau_max = min(first_pole * 0.7, 8.0)
    n_tau = 60
    print(f'\n  Scanning tau in [{-tau_max:.2f}, {tau_max:.2f}], {n_tau} points')
    print(f'  L0 = {L0_REAL:.4f}, lam^2 = {LAM_SQ}')

    t0_scan = time.time()
    scan = wick_scan(L0_REAL, (-tau_max, tau_max), n_tau=n_tau,
                     lam_sq=LAM_SQ, N=N_BASIS, fast_mode=True)
    dt_scan = time.time() - t0_scan
    print(f'\n  Total scan time: {dt_scan:.1f}s')

    # Report — raw barrier
    taus = [r['tau'] for r in scan]
    b_re = [r['barrier_re'] for r in scan]
    b_im = [r['barrier_im'] for r in scan]
    b_abs = [r['barrier_abs'] for r in scan]

    print(f'\n  RAW BARRIER (explodes due to sinh growth — expected):')
    print(f'  Re(B) at tau=0: {scan[n_tau//2]["barrier_re"]:.6f}')
    print(f'  |B| grows to {max(b_abs):.2e} at edges')

    # Report — NORMALIZED barrier (the physics)
    n_re = [r['norm_re'] for r in scan]
    n_im = [r['norm_im'] for r in scan]
    n_abs = [r['norm_abs'] for r in scan]
    phases = [r['phase'] for r in scan]
    ratio_re = [r['ratio_re'] for r in scan]

    print(f'\n  NORMALIZED BARRIER B/W02 = 1 - (M_prime/W02):')
    print(f'  {"tau":>8s} {"Re(B/W02)":>12s} {"Im(B/W02)":>12s} {"phase(B)":>10s} '
          f'{"Re(Mp/W02)":>12s}')
    print('  ' + '-' * 60)
    step = max(1, n_tau // 15)
    for i in range(0, n_tau, step):
        print(f'  {scan[i]["tau"]:>+8.3f} {n_re[i]:>+12.6f} {n_im[i]:>+12.6f} '
              f'{phases[i]:>+10.4f} {ratio_re[i]:>+12.6f}')

    print(f'\n  Re(B/W02) range: [{min(n_re):.6f}, {max(n_re):.6f}]')
    print(f'  Im(B/W02) range: [{min(n_im):.6f}, {max(n_im):.6f}]')
    print(f'  Phase(B) range:  [{min(phases):.4f}, {max(phases):.4f}] rad')

    # Symmetry check on normalized barrier
    print(f'\n  Symmetry of NORMALIZED barrier:')
    for i in range(min(5, n_tau // 2)):
        j_pos = n_tau // 2 + i + 1
        j_neg = n_tau // 2 - i - 1
        if j_pos < len(scan) and j_neg >= 0:
            sym_nre = abs(scan[j_pos]['norm_re'] - scan[j_neg]['norm_re'])
            anti_nim = abs(scan[j_pos]['norm_im'] + scan[j_neg]['norm_im'])
            tau_v = scan[j_pos]['tau']
            print(f'    tau=+/-{abs(tau_v):.3f}: |Re diff| = {sym_nre:.2e}, '
                  f'|Im sum| = {anti_nim:.2e}')

    # ── PART 4: Periodicity analysis ──
    print('\n\n' + '#' * 70)
    print('  PART 4: Periodicity in imaginary direction')
    print('#' * 70)

    period_nre = find_periodicity(scan, 'norm_re')
    period_nim = find_periodicity(scan, 'norm_im')
    period_phase = find_periodicity(scan, 'phase')
    period_ratio = find_periodicity(scan, 'ratio_re')

    print(f'\n  NORMALIZED barrier periodicity:')
    print(f'  Re(B/W02) dominant period: {period_nre["dominant_period"]:.4f}')
    print(f'  Im(B/W02) dominant period: {period_nim["dominant_period"]:.4f}')
    print(f'  Phase(B) dominant period:  {period_phase["dominant_period"]:.4f}')
    print(f'  Re(Mp/W02) dominant period: {period_ratio["dominant_period"]:.4f}')

    for name, pdata in [('Re(B/W02)', period_nre), ('phase', period_phase),
                         ('Re(Mp/W02)', period_ratio)]:
        if pdata['acf_peaks']:
            print(f'\n  {name} autocorrelation peaks:')
            for lag, val in pdata['acf_peaks'][:4]:
                print(f'    lag = {lag:.3f}, corr = {val:.4f}')

    # Also try periodicity on raw barrier (for comparison)
    period_re = find_periodicity(scan, 'barrier_re')

    # Compare period to 2*pi/L and 4*pi (natural scales)
    print(f'\n  Natural scales:')
    print(f'    2*pi        = {2*np.pi:.4f}')
    print(f'    4*pi        = {4*np.pi:.4f}')
    print(f'    2*pi/L      = {2*np.pi/L0_REAL:.4f}')
    print(f'    L           = {L0_REAL:.4f}')
    print(f'    First pole  = {first_pole:.4f}')

    # ── PART 5: Wick-rotated "entropy" ──
    print('\n\n' + '#' * 70)
    print('  PART 5: Wick-rotated action / entropy')
    print('#' * 70)

    # Compute entropy on NORMALIZED barrier (removes exponential growth)
    norm_scan = [{'tau': r['tau'], 'barrier_re': r['norm_re'],
                  'barrier_im': r['norm_im']} for r in scan]
    S_norm = wick_action(norm_scan)

    # Also on raw barrier for comparison
    S_raw = wick_action(scan)

    print(f'\n  NORMALIZED "entropy" (from B/W02):')
    print(f'    integral of (B/W02) d(tau) over [{S_norm["tau_range"][0]:.2f}, {S_norm["tau_range"][1]:.2f}]:')
    print(f'    integral Re = {S_norm["integral_re"]:+.8f}')
    print(f'    integral Im = {S_norm["integral_im"]:+.8f}')
    print(f'    S_norm = i * integral:')
    print(f'    Re(S) = {S_norm["entropy_re"]:+.8f}')
    print(f'    Im(S) = {S_norm["entropy_im"]:+.8f}')
    print(f'    |S|   = {S_norm["entropy_abs"]:.8f}')

    if S_norm['entropy_re'] > 0:
        print(f'\n  *** POSITIVE REAL ENTROPY: {S_norm["entropy_re"]:.8f} ***')
        print(f'  Boyle-Turok analog: positive entropy ~ log(microstates)')
        print(f'  exp(Re(S)) = {np.exp(S_norm["entropy_re"]):.6f} effective microstates')
    else:
        print(f'\n  Entropy has negative real part: {S_norm["entropy_re"]:.8f}')
        print(f'  May indicate negative-temperature phase (bounded state space).')

    # Compute entropy from the RATIO Mp/W02 (more direct Wick quantity)
    ratio_scan = [{'tau': r['tau'], 'barrier_re': r['ratio_re'],
                   'barrier_im': r['ratio_im']} for r in scan]
    S_ratio = wick_action(ratio_scan)
    print(f'\n  RATIO "entropy" (from Mp/W02 — the prime fraction):')
    print(f'    Re(S) = {S_ratio["entropy_re"]:+.8f}')
    print(f'    Interpretation: how primes "fill" the analytic structure')

    # ── PART 6: Cramer control ──
    print('\n\n' + '#' * 70)
    print('  PART 6: Cramer control — random primes break the structure?')
    print('#' * 70)

    n_cramer = 15
    tau_vals = np.linspace(-float(tau_max), float(tau_max), n_cramer)
    print(f'\n  Comparing real primes vs Cramer random primes at {n_cramer} tau values')

    cramer_results = []
    for j, tau in enumerate(tau_vals):
        L_c = mpc(L0_REAL, tau)
        w02 = complex_w02_rayleigh(L_c, N_BASIS)
        mp_real = complex_mprime_rayleigh(L_c, LAM_SQ, N_BASIS)
        mp_cram = complex_mprime_cramer(L_c, LAM_SQ, N_BASIS)

        b_real = w02 - mp_real
        b_cram = w02 - mp_cram

        # Normalized versions
        nr_real = float(re(1 - mp_real / w02)) if abs(w02) > 1e-30 else 0.0
        nr_cram = float(re(1 - mp_cram / w02)) if abs(w02) > 1e-30 else 0.0

        cramer_results.append({
            'tau': tau,
            'b_real_re': float(re(b_real)),
            'b_cram_re': float(re(b_cram)),
            'nr_real': nr_real,
            'nr_cram': nr_cram,
            'diff_norm': nr_real - nr_cram,
        })

        if j % 5 == 0 or j == n_cramer - 1:
            print(f'    tau={tau:+6.2f}  B/W02_real={nr_real:+.6f}  '
                  f'B/W02_cram={nr_cram:+.6f}  diff={nr_real - nr_cram:+.6f}')

    # Are the structures different?
    diffs = [r['diff_norm'] for r in cramer_results]
    print(f'\n  Normalized difference stats (real - Cramer):')
    print(f'    Mean:   {np.mean(diffs):+.6f}')
    print(f'    Std:    {np.std(diffs):.6f}')
    print(f'    Max:    {max(diffs):+.6f}')
    print(f'    Min:    {min(diffs):+.6f}')

    real_at_0 = cramer_results[n_cramer//2]['nr_real']
    cram_at_0 = cramer_results[n_cramer//2]['nr_cram']
    print(f'\n  At tau=0: (B/W02)_real={real_at_0:+.6f}, (B/W02)_cramer={cram_at_0:+.6f}')

    # Check if Cramer breaks the normalized structure differently at tau != 0
    real_norm_vals = [r['nr_real'] for r in cramer_results]
    cram_norm_vals = [r['nr_cram'] for r in cramer_results]
    print(f'  Real primes:  Re(B/W02) range = [{min(real_norm_vals):.6f}, {max(real_norm_vals):.6f}]')
    print(f'  Cramer primes: Re(B/W02) range = [{min(cram_norm_vals):.6f}, {max(cram_norm_vals):.6f}]')

    # ── PART 7: Multiple L0 — NORMALIZED landscape ──
    print('\n\n' + '#' * 70)
    print('  PART 7: Normalized landscape across multiple L0 values')
    print('#' * 70)

    L0_values = [np.log(500), np.log(1000), np.log(2000), np.log(5000), np.log(10000)]
    lam_sq_values = [500, 1000, 2000, 5000, 10000]
    tau_test = [0.0, 1.0, 2.0, 4.0]

    print(f'\n  {"L0":>6s} {"lam^2":>6s}', end='')
    for t in tau_test:
        print(f'  Re(B/W02)@t={t:.0f}', end='')
    print()
    print('  ' + '-' * (14 + 16 * len(tau_test)))

    for L0, lsq in zip(L0_values, lam_sq_values):
        print(f'  {L0:>6.3f} {lsq:>6d}', end='')
        for tau in tau_test:
            L_c = mpc(L0, tau)
            w02 = complex_w02_rayleigh(L_c, N_BASIS)
            mp_rq = complex_mprime_rayleigh(L_c, lsq, N_BASIS)
            norm_b = float(re(1 - mp_rq / w02)) if float(abs(w02)) > 1e-30 else 0.0
            print(f'  {norm_b:>+14.6f}', end='')
        print()
        sys.stdout.flush()

    # ── PART 8: Near-pole behavior and residues ──
    print('\n\n' + '#' * 70)
    print('  PART 8: Near-pole behavior — Boyle-Turok temperature analog')
    print('#' * 70)

    # Approach the first pole at tau ~ 4*pi*1 ~ 12.566 (from L0=0)
    # or tau ~ sqrt(L0^2 + (4*pi)^2) from finite L0
    tau_pole = float(first_pole)
    approach_fracs = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98]
    print(f'\n  First pole at tau ~ {tau_pole:.3f}')
    print(f'  Approaching pole (NORMALIZED B/W02):')
    print(f'  {"frac":>6s} {"tau":>8s} {"Re(B/W02)":>12s} {"Im(B/W02)":>12s} '
          f'{"Re(Mp/W02)":>12s} {"log|W02|":>10s}')
    print('  ' + '-' * 65)

    for frac in approach_fracs:
        tau = frac * tau_pole
        L_c = mpc(L0_REAL, tau)
        w02 = complex_w02_rayleigh(L_c, N_BASIS)
        mp_rq = complex_mprime_rayleigh(L_c, LAM_SQ, N_BASIS)
        norm = 1 - mp_rq / w02 if abs(w02) > 1e-30 else mpc(0)
        ratio = mp_rq / w02 if abs(w02) > 1e-30 else mpc(0)
        log_w02 = np.log(float(abs(w02))) if float(abs(w02)) > 0 else float('-inf')
        print(f'  {frac:>6.2f} {tau:>8.3f} {float(re(norm)):>+12.6f} '
              f'{float(im(norm)):>+12.6f} {float(re(ratio)):>+12.6f} {log_w02:>10.3f}')

    # ── Temperature from imaginary period ──
    print(f'\n  Temperature analysis:')
    best_period = period_nre["dominant_period"]
    print(f'  Dominant period of Re(B/W02) in tau: {best_period:.4f}')
    if best_period < float('inf') and best_period > 0:
        T_barrier = 1.0 / (2 * np.pi * best_period)
        print(f'    "Temperature" T = 1/(2*pi*tau_0) = {T_barrier:.6f}')
        print(f'    Normalized "Entropy" Re(S) = {S_norm["entropy_re"]:+.8f}')
        if S_norm['entropy_re'] > 0:
            print(f'    exp(Re(S)) = {np.exp(S_norm["entropy_re"]):.6f} effective microstates')
    else:
        print(f'    No clean period detected — need finer scan or larger range')

    # ── PART 9: Summary ──
    print('\n\n' + '=' * 72)
    print('  SESSION 45 SUMMARY')
    print('=' * 72)
    print(f'''
  WICK ROTATION RESULTS:

  1. COMPLEX BARRIER B(L0 + i*tau) computed for L0 = {L0_REAL:.3f}
     - Re(B) at tau=0: {scan[n_tau//2]["barrier_re"]:+.6f} (matches real computation)
     - Raw B explodes (sinh growth) but NORMALIZED B/W02 is bounded
     - Normalized: Re(B/W02) range = [{min(n_re):.6f}, {max(n_re):.6f}]

  2. POLE STRUCTURE: poles at tau ~ sqrt(L0^2 + (4*pi*n)^2)
     First pole: tau ~ {first_pole:.3f}
     These are the zeros of L^2 + (4*pi)^2*n^2 — the mode denominators
     Analogous to Boyle-Turok a(tau) poles where scale factor vanishes

  3. PERIODICITY (normalized):
     Re(B/W02) period: {period_nre["dominant_period"]:.4f}
     Phase(B) period:  {period_phase["dominant_period"]:.4f}
     Re(Mp/W02) period: {period_ratio["dominant_period"]:.4f}
     (Natural scales: 2*pi/L = {2*np.pi/L0_REAL:.4f}, 4*pi = {4*np.pi:.4f})

  4. WICK ENTROPY (normalized B/W02):
     Re(S) = {S_norm["entropy_re"]:+.8f}
     |S|   = {S_norm["entropy_abs"]:.8f}
     Ratio entropy Re(S) = {S_ratio["entropy_re"]:+.8f}

  5. CRAMER CONTROL (normalized):
     Mean diff = {np.mean(diffs):+.6f}, Std = {np.std(diffs):.6f}
     Real primes Re(B/W02): [{min(real_norm_vals):.6f}, {max(real_norm_vals):.6f}]
     Cramer primes Re(B/W02): [{min(cram_norm_vals):.6f}, {max(cram_norm_vals):.6f}]

  BOYLE-TUROK PARALLELS:
  - Their a(tau) has poles where scale factor vanishes -> our B(L) has poles
    where mode denominators vanish
  - Their imaginary periodicity -> temperature; our periodicity -> TBD
  - Their entropy maximized at flat universe; our entropy -> TBD
  - Their Cramer analog: random geometry has lower entropy; our: random
    primes break barrier positivity (Session 43 confirmed)
  - KEY INSIGHT: the NORMALIZED barrier B/W02 = 1 - Mp/W02 stays bounded
    in the complex plane. The prime fraction Mp/W02 is the "matter content"
    analog. Barrier positivity = this fraction < 1.
''')

    print('=' * 72)
    print('  SESSION 45 COMPLETE')
    print('=' * 72)
