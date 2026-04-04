"""
SESSION 45i — QUATERNIONIC FOURIER TRANSFORM OF THE BARRIER

The barrier B(L) = W02 - M_prime uses the Fourier basis:
  cos(2*pi*n*x/L),  sin(2*pi*n*x/L)

The Quaternionic Fourier Transform (QFT) replaces e^{i*theta} with
e^{I*theta} for an arbitrary imaginary unit I in S^2.

KEY IDEA: Use TWO independent imaginary units I, J for two parts:
  - W02 (analytic structure) expressed in the I-Fourier basis
  - M_prime (prime sum) expressed in the J-Fourier basis

If the analytic and arithmetic content project onto ORTHOGONAL
quaternionic directions, the barrier becomes:
  B = |analytic_I|^2 + |arithmetic_J|^2 >= 0

This would make positivity AUTOMATIC.

THE TEST:
  1. Compute the barrier's Fourier content along i vs j
  2. Measure the "quaternionic angle" between W02 and M_prime
     in the Fourier domain
  3. Check if they're closer to parallel (cancellation possible)
     or orthogonal (positivity automatic)

IMPLEMENTATION:
  The standard barrier uses real Fourier modes (cos, sin with complex i).
  The QFT version uses quaternionic Fourier modes:
    e_I(n,x) = cos(2*pi*n*x/L) + I*sin(2*pi*n*x/L)
    e_J(n,x) = cos(2*pi*n*x/L) + J*sin(2*pi*n*x/L)

  For a real function f(x), the QFT along direction I is:
    F_I(n) = integral_0^L f(x) * [cos(2*pi*n*x/L) - I*sin(2*pi*n*x/L)] dx

  The key: for the SAME function f(x), F_I and F_J differ only in their
  imaginary direction. But for DIFFERENT functions (W02 kernel vs prime kernel),
  the QFT reveals their angular separation in H.
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from session41g_uncapped_barrier import sieve_primes


class Q:
    """Minimal quaternion for speed."""
    __slots__ = ('a','b','c','d')
    def __init__(self, a=0., b=0., c=0., d=0.):
        self.a = a; self.b = b; self.c = c; self.d = d
    def __add__(self, o):
        if isinstance(o, (int,float)): return Q(self.a+o, self.b, self.c, self.d)
        return Q(self.a+o.a, self.b+o.b, self.c+o.c, self.d+o.d)
    def __sub__(self, o):
        if isinstance(o, (int,float)): return Q(self.a-o, self.b, self.c, self.d)
        return Q(self.a-o.a, self.b-o.b, self.c-o.c, self.d-o.d)
    def __mul__(self, o):
        if isinstance(o, (int,float)): return Q(self.a*o, self.b*o, self.c*o, self.d*o)
        a1,b1,c1,d1 = self.a,self.b,self.c,self.d
        a2,b2,c2,d2 = o.a,o.b,o.c,o.d
        return Q(a1*a2-b1*b2-c1*c2-d1*d2, a1*b2+b1*a2+c1*d2-d1*c2,
                 a1*c2-b1*d2+c1*a2+d1*b2, a1*d2+b1*c2-c1*b2+d1*a2)
    def __rmul__(self, o):
        if isinstance(o, (int,float)): return Q(self.a*o, self.b*o, self.c*o, self.d*o)
        return o.__mul__(self)
    def __truediv__(self, o):
        if isinstance(o, (int,float)): return Q(self.a/o, self.b/o, self.c/o, self.d/o)
        return self * o.inv()
    def conj(self): return Q(self.a, -self.b, -self.c, -self.d)
    def norm_sq(self): return self.a**2 + self.b**2 + self.c**2 + self.d**2
    def norm(self): return np.sqrt(self.norm_sq())
    def inv(self):
        ns = self.norm_sq()
        c = self.conj()
        return Q(c.a/ns, c.b/ns, c.c/ns, c.d/ns)
    def dot(self, o):
        """Real inner product in R^4."""
        return self.a*o.a + self.b*o.b + self.c*o.c + self.d*o.d
    def __repr__(self):
        return f'({self.a:.6f}, {self.b:.6f}i, {self.c:.6f}j, {self.d:.6f}k)'


# Quaternionic imaginary units
I_UNIT = Q(0, 1, 0, 0)  # standard complex i
J_UNIT = Q(0, 0, 1, 0)  # quaternionic j
K_UNIT = Q(0, 0, 0, 1)  # quaternionic k


def qft_coefficient(f_vals, x_pts, n, L, I_dir=I_UNIT):
    """
    Quaternionic Fourier coefficient:
    F_I(n) = integral f(x) * e_I^{-n}(x) dx
           = integral f(x) * [cos(2*pi*n*x/L) - I*sin(2*pi*n*x/L)] dx

    Returns a quaternion (real part from cos, I-part from -sin).
    """
    dx = x_pts[1] - x_pts[0] if len(x_pts) > 1 else 1.0
    theta = 2 * np.pi * n * x_pts / L

    cos_part = np.sum(f_vals * np.cos(theta)) * dx
    sin_part = np.sum(f_vals * np.sin(theta)) * dx

    # F = cos_part - I * sin_part = cos_part + I * (-sin_part)
    return Q(cos_part, -sin_part * I_dir.b, -sin_part * I_dir.c, -sin_part * I_dir.d)


def build_kernels(lam_sq, N_fourier=20, n_quad=2000):
    """
    Build the W02 and M_prime kernels as functions of x in [0, L].

    The barrier is B = <w, (W02 - M_prime), w> where the matrix elements
    involve integrals of kernel functions over [0, L].

    W02 kernel at (n,m): pf * (L^2 - (4*pi)^2*n*m) / (denom_n * denom_m)
    M_prime kernel at (n,m): sum over primes of weight * q(n,m,y)

    Instead of the full matrix, compute the DIAGONAL kernel:
    K_W02(x) = contribution to the barrier from point x (via the test function)
    K_Mp(x) = prime contribution from point x

    Specifically, the barrier can be written as:
    B = integral_0^L [K_W02(x) - K_Mp(x)] dx
    where K(x) = |h(x)|^2 * (analytic or prime weight at x)

    Actually, let's be more precise. The Rayleigh quotient on w_hat direction:
    <w, W02, w> = pf * (4*pi)^2 * (sum w_tilde)^2
    This is a GLOBAL quantity, not a pointwise one.

    For the QFT approach, we need to think differently.
    The M_prime matrix element involves:
      M_prime = sum_p weight_p * [kernel evaluated at y_p = log(p)]

    So M_prime is a sum of delta-function-like contributions at y = log(p).
    While W02 is an analytic function of L.

    The QFT idea: represent both as functions of a "frequency" variable n,
    then check their quaternionic alignment.
    """
    L = np.log(lam_sq)
    x_pts = np.linspace(0.001, L, n_quad)
    dx = x_pts[1] - x_pts[0]

    ns = np.arange(-N_fourier, N_fourier + 1, dtype=float)
    w_vec = ns / (L**2 + (4*np.pi)**2 * ns**2)
    w_vec[N_fourier] = 0.0
    w_hat = w_vec / np.linalg.norm(w_vec)

    # Build the W02 "spectral function": for each x, how much does W02 contribute?
    # W02[n,m] = pf * (L^2 - (4pi)^2*nm) / (denom_n * denom_m)
    # <w, W02, w> = pf * (4pi)^2 * (sum_n w_tilde[n])^2  (shortcut)

    # Instead: build the KERNEL FUNCTION of x that enters M_prime.
    # M_prime = sum_{p^k} weight * K(y_p) where K involves the w_hat weights.
    # K(y) = sum_{n,m} w_hat[n] * q(n,m,y) * w_hat[m]

    # For the QFT: represent K(y) in Fourier modes.
    # K(y) = sum_nu c_nu * e^{2*pi*i*nu*y/L}

    # Compute K(y) = the M_prime kernel at a set of y points
    def mp_kernel(y):
        """Rayleigh quotient contribution from a single prime power at y."""
        val = 0.0
        for i_idx in range(len(ns)):
            ni = ns[i_idx]
            if abs(w_hat[i_idx]) < 1e-15:
                continue
            for j_idx in range(len(ns)):
                nj = ns[j_idx]
                if abs(w_hat[j_idx]) < 1e-15:
                    continue
                if ni == nj:
                    q_nm = 2 * (L - y) / L * np.cos(2*np.pi*ni*y/L)
                else:
                    q_nm = (np.sin(2*np.pi*nj*y/L) - np.sin(2*np.pi*ni*y/L)) / (np.pi*(ni-nj))
                val += w_hat[i_idx] * q_nm * w_hat[j_idx]
        return val

    # Evaluate kernel at quadrature points
    kernel_vals = np.array([mp_kernel(x) for x in x_pts])

    # The prime sum: M_prime = sum weight(p) * kernel(log(p))
    primes = sieve_primes(int(lam_sq))
    prime_weights = {}  # y -> weight
    for p in primes:
        pk = int(p)
        k = 1
        logp = np.log(int(p))
        while pk <= lam_sq:
            y = k * logp
            w = logp * pk**(-0.5)
            if y not in prime_weights:
                prime_weights[y] = 0.0
            prime_weights[y] += w
            pk *= int(p)
            k += 1

    # Build the prime "density" function: delta functions at y=log(p^k)
    # Smoothed onto the grid for Fourier analysis
    prime_density = np.zeros_like(x_pts)
    for y, w in prime_weights.items():
        if y < L:
            idx = int(y / dx)
            if 0 <= idx < len(prime_density):
                prime_density[idx] += w / dx  # delta function -> density

    # The analytic "density": PNT says sum log(p)/sqrt(p) ~ integral e^{y/2} dy
    analytic_density = np.exp(x_pts / 2)  # PNT smooth approximation

    return {
        'L': L, 'x_pts': x_pts, 'dx': dx,
        'kernel_vals': kernel_vals,
        'prime_density': prime_density,
        'analytic_density': analytic_density,
        'prime_weights': prime_weights,
        'w_hat': w_hat, 'ns': ns,
        'n_primes': len(primes),
    }


def qft_decompose(data, N_modes=30):
    """
    Compute QFT of the prime density and the analytic density
    along DIFFERENT quaternionic directions (I, J, K).

    Then measure the quaternionic angle between the prime and analytic
    Fourier spectra.
    """
    L = data['L']
    x_pts = data['x_pts']

    # Function 1: prime_density * kernel  (what M_prime actually integrates)
    f_prime = data['prime_density'] * data['kernel_vals']

    # Function 2: analytic_density * kernel  (the PNT smooth approximation)
    f_analytic = data['analytic_density'] * data['kernel_vals']

    # Function 3: the kernel itself
    f_kernel = data['kernel_vals']

    results = {}

    for label, f_vals in [('prime', f_prime), ('analytic', f_analytic), ('kernel', f_kernel)]:
        # QFT along i (standard complex Fourier)
        coeffs_i = []
        # QFT along j (quaternionic Fourier)
        coeffs_j = []
        # QFT along k
        coeffs_k = []

        for nu in range(N_modes + 1):
            ci = qft_coefficient(f_vals, x_pts, nu, L, I_UNIT)
            cj = qft_coefficient(f_vals, x_pts, nu, L, J_UNIT)
            ck = qft_coefficient(f_vals, x_pts, nu, L, K_UNIT)
            coeffs_i.append(ci)
            coeffs_j.append(cj)
            coeffs_k.append(ck)

        results[label] = {
            'i': coeffs_i, 'j': coeffs_j, 'k': coeffs_k,
        }

    return results


def measure_angle(q1, q2):
    """Quaternionic angle between q1 and q2 (as R^4 vectors)."""
    d = q1.dot(q2)
    n1 = q1.norm()
    n2 = q2.norm()
    if n1 < 1e-15 or n2 < 1e-15:
        return 0.0
    cos_theta = max(-1, min(1, d / (n1 * n2)))
    return np.arccos(cos_theta)


if __name__ == '__main__':
    print()
    print('=' * 76)
    print('  SESSION 45i -- QUATERNIONIC FOURIER TRANSFORM OF THE BARRIER')
    print('=' * 76)

    LAM_SQ = 1000
    N_MODES = 25

    # ══════════════════════════════════════════════════════════════
    # 1. BUILD KERNELS
    # ══════════════════════════════════════════════════════════════
    print('\n' + '#' * 76)
    print('  1. BUILDING BARRIER KERNELS')
    print('#' * 76)

    t0 = time.time()
    data = build_kernels(LAM_SQ, N_fourier=15, n_quad=3000)
    dt = time.time() - t0
    print(f'\n  lam^2 = {LAM_SQ}, L = {data["L"]:.4f}, {data["n_primes"]} primes ({dt:.1f}s)')

    # Quick check: M_prime from kernel*density should match matrix computation
    mp_from_density = 0.0
    for y, w in data['prime_weights'].items():
        if y < data['L']:
            idx = min(int(y / data['dx']), len(data['kernel_vals'])-1)
            mp_from_density += w * data['kernel_vals'][idx]

    from session41g_uncapped_barrier import compute_barrier_partial
    r_check = compute_barrier_partial(LAM_SQ, N=15)
    print(f'  M_prime (matrix): {r_check["mprime"]:+.6f}')
    print(f'  M_prime (kernel*density): {mp_from_density:+.6f}')
    print(f'  Difference: {abs(mp_from_density - r_check["mprime"]):.4f}')

    # ══════════════════════════════════════════════════════════════
    # 2. QFT DECOMPOSITION
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  2. QUATERNIONIC FOURIER DECOMPOSITION')
    print('#' * 76)

    t0 = time.time()
    qft = qft_decompose(data, N_modes=N_MODES)
    dt = time.time() - t0
    print(f'  Computed QFT along i, j, k for {N_MODES} modes ({dt:.1f}s)')

    # Compare spectra
    print(f'\n  PRIME vs ANALYTIC Fourier spectra:')
    print(f'  {"mode":>5s} {"prime |F_i|":>12s} {"anal |F_i|":>12s} '
          f'{"prime |F_j|":>12s} {"anal |F_j|":>12s} {"angle(p,a)_i":>12s}')
    print('  ' + '-' * 68)

    angles_i = []
    angles_j = []
    for nu in range(min(20, N_MODES + 1)):
        pi_norm = qft['prime']['i'][nu].norm()
        ai_norm = qft['analytic']['i'][nu].norm()
        pj_norm = qft['prime']['j'][nu].norm()
        aj_norm = qft['analytic']['j'][nu].norm()

        angle_i = measure_angle(qft['prime']['i'][nu], qft['analytic']['i'][nu])
        angle_j = measure_angle(qft['prime']['j'][nu], qft['analytic']['j'][nu])
        angles_i.append(angle_i)
        angles_j.append(angle_j)

        print(f'  {nu:>5d} {pi_norm:>12.6f} {ai_norm:>12.6f} '
              f'{pj_norm:>12.6f} {aj_norm:>12.6f} {angle_i:>12.4f} rad')

    # ══════════════════════════════════════════════════════════════
    # 3. THE KEY TEST: QUATERNIONIC ANGLE BETWEEN PRIME AND ANALYTIC
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  3. QUATERNIONIC ANGLE: PRIME vs ANALYTIC')
    print('#' * 76)

    print(f'\n  For each QFT mode, the quaternionic angle between the prime')
    print(f'  and analytic Fourier coefficients measures their alignment.')
    print(f'  angle = 0: parallel (can cancel)')
    print(f'  angle = pi/2: orthogonal (barrier = sum of squares!)')
    print(f'  angle = pi: anti-parallel (constructive for barrier)')

    angles_i = np.array(angles_i)
    angles_j = np.array(angles_j)

    print(f'\n  Along i (standard Fourier):')
    print(f'    Mean angle: {np.mean(angles_i):.4f} rad ({np.mean(angles_i)/np.pi:.4f}*pi)')
    print(f'    Std:        {np.std(angles_i):.4f} rad')
    print(f'    Min angle:  {np.min(angles_i):.4f} rad (mode {np.argmin(angles_i)})')
    print(f'    Max angle:  {np.max(angles_i):.4f} rad (mode {np.argmax(angles_i)})')

    print(f'\n  Along j (quaternionic Fourier):')
    print(f'    Mean angle: {np.mean(angles_j):.4f} rad ({np.mean(angles_j)/np.pi:.4f}*pi)')
    print(f'    Std:        {np.std(angles_j):.4f} rad')

    if np.mean(angles_i) > np.pi * 0.4:
        print(f'\n  *** ANGLES NEAR pi/2 OR LARGER: PARTIAL ORTHOGONALITY ***')
        print(f'  Prime and analytic content have significant angular separation!')
    if np.mean(angles_i) > np.pi * 0.8:
        print(f'  *** NEAR ANTI-PARALLEL: barrier gets constructive contributions ***')

    # ══════════════════════════════════════════════════════════════
    # 4. THE MIXED QFT: PRIMES ALONG I, ANALYTIC ALONG J
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  4. MIXED QFT: primes in i-plane, analytic in j-plane')
    print('#' * 76)

    print(f'\n  If we represent:')
    print(f'    M_prime  in the I-Fourier basis (i-plane)')
    print(f'    W02      in the J-Fourier basis (j-plane)')
    print(f'  Then the barrier B = W02 - M_prime has I and J components.')
    print(f'  If they are orthogonal: |B|^2 = |W02_J|^2 + |Mp_I|^2 >= 0')

    # Compute total "energy" in each QFT direction
    E_prime_i = sum(c.norm_sq() for c in qft['prime']['i'])
    E_prime_j = sum(c.norm_sq() for c in qft['prime']['j'])
    E_anal_i = sum(c.norm_sq() for c in qft['analytic']['i'])
    E_anal_j = sum(c.norm_sq() for c in qft['analytic']['j'])

    print(f'\n  Energy distribution:')
    print(f'    Prime energy along i: {E_prime_i:.6f}')
    print(f'    Prime energy along j: {E_prime_j:.6f}')
    print(f'    Analytic energy along i: {E_anal_i:.6f}')
    print(f'    Analytic energy along j: {E_anal_j:.6f}')
    print(f'    Ratio E_prime_i / E_prime_j: {E_prime_i/E_prime_j:.6f}')
    print(f'    Ratio E_anal_i / E_anal_j:  {E_anal_i/E_anal_j:.6f}')

    if abs(E_prime_i/E_prime_j - E_anal_i/E_anal_j) > 0.1:
        print(f'\n  *** DIFFERENT ENERGY RATIOS: prime and analytic distribute')
        print(f'      differently across quaternionic Fourier directions! ***')
    else:
        print(f'\n  Same energy ratios: no quaternionic separation in Fourier domain.')

    # ══════════════════════════════════════════════════════════════
    # 5. CROSS-CORRELATION IN QUATERNIONIC FOURIER SPACE
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  5. CROSS-CORRELATION: prime-analytic overlap per mode')
    print('#' * 76)

    print(f'\n  C(nu) = Re(F_prime(nu)^* . F_analytic(nu)) / (|F_p||F_a|)')
    print(f'  This is the cosine of the angle per mode.')
    print(f'  If C ~ 0: orthogonal. If C ~ 1: parallel. If C ~ -1: anti-parallel.')

    cross_corr = []
    for nu in range(N_MODES + 1):
        fp = qft['prime']['i'][nu]
        fa = qft['analytic']['i'][nu]
        np_val = fp.norm()
        na_val = fa.norm()
        if np_val > 1e-15 and na_val > 1e-15:
            # Quaternionic inner product: Re(conj(fp) * fa)
            fp_conj = fp.conj()
            product = fp_conj * fa
            c = product.a / (np_val * na_val)
            cross_corr.append(c)
        else:
            cross_corr.append(0.0)

    cross_corr = np.array(cross_corr)
    print(f'\n  {"mode":>5s} {"C(nu)":>10s} {"interpretation":>15s}')
    print('  ' + '-' * 35)
    for nu in range(min(20, len(cross_corr))):
        c = cross_corr[nu]
        if c > 0.8:
            interp = 'parallel'
        elif c < -0.8:
            interp = 'anti-parallel'
        elif abs(c) < 0.2:
            interp = 'ORTHOGONAL'
        else:
            interp = 'mixed'
        print(f'  {nu:>5d} {c:>+10.4f} {interp:>15s}')

    print(f'\n  Mean cross-correlation: {np.mean(cross_corr):+.4f}')
    print(f'  Std:                    {np.std(cross_corr):.4f}')

    n_orth = np.sum(np.abs(cross_corr) < 0.3)
    n_par = np.sum(cross_corr > 0.7)
    n_anti = np.sum(cross_corr < -0.7)
    print(f'\n  Orthogonal modes (|C| < 0.3): {n_orth}/{len(cross_corr)}')
    print(f'  Parallel modes (C > 0.7):      {n_par}/{len(cross_corr)}')
    print(f'  Anti-parallel (C < -0.7):       {n_anti}/{len(cross_corr)}')

    # ══════════════════════════════════════════════════════════════
    # 6. UNIVERSALITY: SAME PATTERN AT DIFFERENT lam^2?
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  6. UNIVERSALITY: cross-correlation at different lam^2')
    print('#' * 76)

    for lam_sq in [200, 500, 1000, 2000, 5000]:
        d = build_kernels(lam_sq, N_fourier=12, n_quad=2000)
        q = qft_decompose(d, N_modes=15)

        cc = []
        for nu in range(16):
            fp = q['prime']['i'][nu]
            fa = q['analytic']['i'][nu]
            np_v = fp.norm()
            na_v = fa.norm()
            if np_v > 1e-15 and na_v > 1e-15:
                product = fp.conj() * fa
                cc.append(product.a / (np_v * na_v))
            else:
                cc.append(0.0)
        cc = np.array(cc)

        n_orth = np.sum(np.abs(cc) < 0.3)
        print(f'  lam^2={lam_sq:>5d}: mean C={np.mean(cc):+.4f}, '
              f'std={np.std(cc):.4f}, orthogonal={n_orth}/16')
    sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════
    # 7. THE BARRIER IN QFT REPRESENTATION
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  7. BARRIER RECONSTRUCTION FROM QFT')
    print('#' * 76)

    print(f'\n  The barrier B = integral [kernel * (analytic_density - prime_density)] dx')
    print(f'  In Fourier: B = sum_nu [F_analytic(nu) - F_prime(nu)] (mode 0 terms)')

    # Mode 0 = the total integral
    f0_p = qft['prime']['i'][0]
    f0_a = qft['analytic']['i'][0]
    f0_diff = f0_a - f0_p

    print(f'\n  Mode 0 (total integral):')
    print(f'    F_analytic(0) = {f0_a}')
    print(f'    F_prime(0)    = {f0_p}')
    print(f'    Difference    = {f0_diff}')
    print(f'    |F_a(0) - F_p(0)| = {f0_diff.norm():.6f}')

    # Build the barrier from difference of Fourier coefficients
    b_from_qft = Q(0, 0, 0, 0)
    for nu in range(N_MODES + 1):
        diff = qft['analytic']['i'][nu] - qft['prime']['i'][nu]
        b_from_qft = b_from_qft + diff

    print(f'\n  Barrier from QFT sum: {b_from_qft}')
    print(f'  |B_QFT| = {b_from_qft.norm():.6f}')
    print(f'  Re(B_QFT) = {b_from_qft.a:.6f}')

    # ══════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '=' * 76)
    print('  SESSION 45i SYNTHESIS')
    print('=' * 76)

    print(f'''
  THE QUATERNIONIC FOURIER TEST:

  We decomposed the prime density and the PNT analytic density into
  quaternionic Fourier modes and measured their angular alignment.

  RESULTS:
    Mean cross-correlation C = {np.mean(cross_corr):+.4f}
    If C ~ 0: prime and analytic are ORTHOGONAL in Fourier space
              -> barrier = sum of squares (automatic positivity!)
    If C ~ +1: parallel -> cancellation possible (hard inequality)
    If C ~ -1: anti-parallel -> constructive (barrier enhanced)

  Orthogonal modes: {n_orth}/{len(cross_corr)}
  Parallel modes:   {n_par}/{len(cross_corr)}
  Anti-parallel:    {n_anti}/{len(cross_corr)}

  The QFT along different quaternionic directions (I vs J) gives
  IDENTICAL spectra for a single real function (by construction).
  The angular separation must come from the INTERACTION of two
  different functions (prime vs analytic) in the Fourier domain.

  KEY INSIGHT: The QFT doesn't create orthogonality — it REVEALS
  whether the prime and analytic contributions are already orthogonal
  in their frequency content. If they have the same frequency structure
  (parallel), no quaternionic trick can separate them. If they have
  different frequency structure (orthogonal), the barrier is
  automatically a sum of squares.
''')

    print('=' * 76)
    print('  SESSION 45i COMPLETE')
    print('=' * 76)
