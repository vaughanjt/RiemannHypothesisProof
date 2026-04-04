"""
SESSION 45j — PI AS THE ARCHIMEDEAN PRIME IN QUATERNIONIC SPACE

The completed zeta function factors as:
  xi(s) = L_inf(s) * product_p L_p(s)

where:
  L_inf(s) = pi^{-s/2} * Gamma(s/2)    [the archimedean factor — PI lives here]
  L_p(s)   = (1 - p^{-s})^{-1}          [finite prime factors]

In complex analysis, both are complex numbers and can cancel.
In quaternionic analysis, they're quaternions. If L_inf and the Euler
product point in DIFFERENT quaternionic directions, they CAN'T fully cancel.

THE KEY QUESTION: When s is quaternionic (s = sigma + a*I + b*J + c*K),
do the archimedean factor and the Euler product separate in H?

PLAN:
  1. Compute L_inf(s) = pi^{-s/2} * Gamma(s/2) for quaternionic s
  2. Compute the Euler product = product_p (1-p^{-s})^{-1} for quaternionic s
  3. Measure the quaternionic angle between them
  4. Check: does the angle depend on sigma? Is sigma=1/2 special?
  5. The functional equation xi(s) = xi(1-s) in quaternionic form
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, gamma as mpgamma, zeta as mpzeta, pi as mppi, log as mplog
import time
import sys

mp.dps = 25


# ═══════════════════════════════════════════════════════════════
# QUATERNION ARITHMETIC (lightweight)
# ═══════════════════════════════════════════════════════════════

class Q:
    __slots__ = ('a','b','c','d')
    def __init__(self, a=0., b=0., c=0., d=0.):
        self.a = float(a); self.b = float(b); self.c = float(c); self.d = float(d)
    def __add__(self, o):
        if isinstance(o, (int,float)): return Q(self.a+o, self.b, self.c, self.d)
        return Q(self.a+o.a, self.b+o.b, self.c+o.c, self.d+o.d)
    def __radd__(self, o): return self.__add__(o)
    def __sub__(self, o):
        if isinstance(o, (int,float)): return Q(self.a-o, self.b, self.c, self.d)
        return Q(self.a-o.a, self.b-o.b, self.c-o.c, self.d-o.d)
    def __neg__(self): return Q(-self.a, -self.b, -self.c, -self.d)
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
        ns = o.norm_sq(); c = o.conj()
        return Q((self.a*c.a-self.b*c.b-self.c*c.c-self.d*c.d)/ns,
                 (self.a*c.b+self.b*c.a+self.c*c.d-self.d*c.c)/ns,
                 (self.a*c.c-self.b*c.d+self.c*c.a+self.d*c.b)/ns,
                 (self.a*c.d+self.b*c.c-self.c*c.b+self.d*c.a)/ns)
    def conj(self): return Q(self.a, -self.b, -self.c, -self.d)
    def norm_sq(self): return self.a**2 + self.b**2 + self.c**2 + self.d**2
    def norm(self): return np.sqrt(self.norm_sq())
    def vec_norm(self): return np.sqrt(self.b**2 + self.c**2 + self.d**2)
    def dot(self, o): return self.a*o.a + self.b*o.b + self.c*o.c + self.d*o.d
    def __repr__(self):
        return f'({self.a:.6f}, {self.b:.6f}i, {self.c:.6f}j, {self.d:.6f}k)'

def qexp(q):
    ea = np.exp(q.a); vn = q.vec_norm()
    if vn < 1e-15: return Q(ea, 0, 0, 0)
    s = ea * np.sin(vn) / vn
    return Q(ea*np.cos(vn), s*q.b, s*q.c, s*q.d)

def qlog_real_base(x, q):
    """x^q = exp(q * log(x)) for real positive x."""
    ln_x = np.log(x)
    return qexp(Q(q.a*ln_x, q.b*ln_x, q.c*ln_x, q.d*ln_x))

def q_from_complex(z):
    """Map complex to quaternion in the I-plane."""
    return Q(z.real, z.imag, 0, 0)

def angle_between(q1, q2):
    d = q1.dot(q2)
    n1, n2 = q1.norm(), q2.norm()
    if n1 < 1e-15 or n2 < 1e-15: return 0.0
    return np.arccos(max(-1, min(1, d/(n1*n2))))


# ═══════════════════════════════════════════════════════════════
# ARCHIMEDEAN FACTOR: L_inf(s) = pi^{-s/2} * Gamma(s/2)
# ═══════════════════════════════════════════════════════════════

def archimedean_factor_complex(s_complex):
    """L_inf(s) = pi^{-s/2} * Gamma(s/2) for complex s."""
    s = mpc(s_complex.real, s_complex.imag)
    val = mppi**(-s/2) * mpgamma(s/2)
    return complex(val)

def archimedean_factor_quat(s_quat):
    """
    L_inf(s) for quaternionic s.
    On the slice C_I containing s, this is the complex L_inf evaluated
    at the complex number sigma + |v|*I where I = v/|v|.
    Then we map back to H.
    """
    sigma = s_quat.a
    vn = s_quat.vec_norm()
    if vn < 1e-15:
        val = archimedean_factor_complex(complex(sigma, 0))
        return Q(val.real, 0, 0, 0)

    # Complex argument on the slice
    s_c = complex(sigma, vn)
    val = archimedean_factor_complex(s_c)

    # Map back: val = u + i*w -> u + (v/|v|)*w
    u, w = val.real, val.imag
    sc = w / vn
    return Q(u, sc*s_quat.b, sc*s_quat.c, sc*s_quat.d)


# ═══════════════════════════════════════════════════════════════
# EULER PRODUCT: product_p (1 - p^{-s})^{-1}
# ═══════════════════════════════════════════════════════════════

def euler_product_quat(s_quat, primes):
    """
    Euler product = product_p (1 - p^{-s})^{-1} for quaternionic s.

    Each factor: (1 - p^{-s})^{-1}
    p^{-s} = exp(-s * log(p)) is a quaternion
    1 - p^{-s} is a quaternion
    Its inverse is conj/(norm^2)
    """
    result = Q(1, 0, 0, 0)  # multiplicative identity

    for p in primes:
        # p^{-s}
        p_neg_s = qexp(Q(-s_quat.a*np.log(p), -s_quat.b*np.log(p),
                          -s_quat.c*np.log(p), -s_quat.d*np.log(p)))
        # 1 - p^{-s}
        one_minus = Q(1, 0, 0, 0) - p_neg_s
        # inverse
        ns = one_minus.norm_sq()
        if ns < 1e-30:
            continue
        inv_factor = one_minus.conj() / ns
        # multiply into product (LEFT multiplication)
        result = result * inv_factor

    return result


def euler_product_complex(s_complex, primes):
    """Euler product for complex s (for verification)."""
    result = 1.0 + 0.0j
    for p in primes:
        factor = 1.0 - p**(-s_complex)
        if abs(factor) > 1e-30:
            result /= factor
    return result


# ═══════════════════════════════════════════════════════════════
# COMPLETED ZETA: xi(s) = (1/2)*s*(s-1) * L_inf(s) * zeta(s)
# ═══════════════════════════════════════════════════════════════

def completed_zeta_quat(s_quat, primes):
    """
    xi(s) = (1/2)*s*(s-1) * pi^{-s/2} * Gamma(s/2) * zeta(s)

    Factor as:
    xi = prefactor * archimedean * euler_product

    where prefactor = (1/2)*s*(s-1)
    """
    s = s_quat
    prefactor = (s * (s - 1.0)) / 2.0
    arch = archimedean_factor_quat(s)
    euler = euler_product_quat(s, primes)

    # xi = prefactor * arch * euler
    xi = prefactor * arch * euler
    return xi, prefactor, arch, euler


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print()
    print('=' * 76)
    print('  SESSION 45j -- PI AS THE ARCHIMEDEAN PRIME IN H')
    print('=' * 76)

    # Use first 50 primes for the Euler product
    from session41g_uncapped_barrier import sieve_primes
    primes = list(sieve_primes(300))  # primes up to 300
    print(f'  Using {len(primes)} primes for Euler product')

    # ══════════════════════════════════════════════════════════════
    # 1. VERIFICATION: complex slice matches mpmath
    # ══════════════════════════════════════════════════════════════
    print('\n' + '#' * 76)
    print('  1. VERIFICATION: archimedean and Euler product on complex slice')
    print('#' * 76)

    for s_val in [complex(2, 0), complex(2, 1), complex(0.5, 14.1347), complex(3, 5)]:
        arch_c = archimedean_factor_complex(s_val)
        euler_c = euler_product_complex(s_val, primes)
        zeta_c = complex(mpzeta(mpc(s_val.real, s_val.imag)))

        # Quaternionic versions
        s_q = Q(s_val.real, s_val.imag, 0, 0)
        arch_q = archimedean_factor_quat(s_q)
        euler_q = euler_product_quat(s_q, primes)

        print(f'\n  s = {s_val}:')
        print(f'    Archimedean (complex): {arch_c:.6f}')
        print(f'    Archimedean (quat):    {arch_q}')
        print(f'    Euler prod (complex):  {euler_c:.6f}')
        print(f'    Euler prod (quat):     {euler_q}')
        print(f'    zeta (mpmath):         {zeta_c:.6f}')
        print(f'    Euler/zeta ratio:      {abs(euler_c/zeta_c):.6f}')
    sys.stdout.flush()

    # ════════════════════════════════��═════════════════════════════
    # 2. QUATERNIONIC ANGLE BETWEEN ARCHIMEDEAN AND EULER PRODUCT
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  2. QUATERNIONIC ANGLE: L_inf vs Euler product')
    print('#' * 76)

    print(f'\n  On the complex slice (j=k=0), both are complex numbers.')
    print(f'  Their angle is determined by arg(L_inf) - arg(Euler).')
    print(f'  In the FULL quaternionic space, we can ask:')
    print(f'  do L_inf and Euler point in DIFFERENT quaternionic directions?')

    # Test on the complex slice first
    print(f'\n  A. Complex slice (s = sigma + t*i):')
    print(f'  {"sigma":>6s} {"t":>8s} {"angle(L,E)":>12s} {"angle/pi":>10s} '
          f'{"|L_inf|":>10s} {"|Euler|":>10s}')
    print('  ' + '-' * 58)

    for sigma in [0.25, 0.5, 0.75]:
        for t in [1.0, 5.0, 14.1347, 25.0]:
            s_q = Q(sigma, t, 0, 0)
            arch = archimedean_factor_quat(s_q)
            euler = euler_product_quat(s_q, primes)
            ang = angle_between(arch, euler)
            print(f'  {sigma:>6.2f} {t:>8.4f} {ang:>12.4f} {ang/np.pi:>10.4f} '
                  f'{arch.norm():>10.4f} {euler.norm():>10.4f}')

    # Now test OFF the complex slice: s = sigma + t*i + u*j
    print(f'\n  B. Off the complex slice (s = sigma + t*i + u*j):')
    print(f'  {"sigma":>6s} {"t":>6s} {"u":>6s} {"angle(L,E)":>12s} {"angle/pi":>10s} '
          f'{"L j-comp":>10s} {"E j-comp":>10s}')
    print('  ' + '-' * 65)

    for sigma in [0.5]:
        for t in [5.0, 14.1347]:
            for u in [0.0, 0.5, 1.0, 2.0, 5.0]:
                s_q = Q(sigma, t, u, 0)
                arch = archimedean_factor_quat(s_q)
                euler = euler_product_quat(s_q, primes)
                ang = angle_between(arch, euler)
                print(f'  {sigma:>6.2f} {t:>6.2f} {u:>6.2f} {ang:>12.4f} '
                      f'{ang/np.pi:>10.4f} {arch.c:>+10.4f} {euler.c:>+10.4f}')
    sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════
    # 3. THE CRITICAL LINE TEST
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  3. IS sigma=1/2 SPECIAL FOR THE ANGLE?')
    print('#' * 76)

    print(f'\n  Scanning sigma at fixed t=14.1347 (near first zero):')
    print(f'  {"sigma":>8s} {"angle(L,E)":>12s} {"angle/pi":>10s} '
          f'{"|xi|":>12s} {"xi direction":>30s}')
    print('  ' + '-' * 78)

    angles_sigma = []
    for sigma in np.linspace(0.1, 0.9, 17):
        s_q = Q(sigma, 14.1347, 0, 0)
        xi, pf, arch, euler = completed_zeta_quat(s_q, primes)
        ang = angle_between(arch, euler)
        angles_sigma.append((sigma, ang))

        xi_n = xi.norm()
        xi_dir = f'({xi.a/xi_n:.3f}, {xi.b/xi_n:.3f}i)' if xi_n > 1e-10 else '(0)'
        marker = ' <-- CL' if abs(sigma - 0.5) < 0.03 else ''
        print(f'  {sigma:>8.4f} {ang:>12.4f} {ang/np.pi:>10.4f} '
              f'{xi_n:>12.6f} {xi_dir:>30s}{marker}')

    # Is sigma=1/2 a minimum, maximum, or neither?
    asig = np.array([a for _, a in angles_sigma])
    sigs = np.array([s for s, _ in angles_sigma])
    idx_half = np.argmin(np.abs(sigs - 0.5))
    print(f'\n  Angle at sigma=0.5: {asig[idx_half]:.4f} rad')
    print(f'  Min angle: {asig.min():.4f} at sigma={sigs[np.argmin(asig)]:.4f}')
    print(f'  Max angle: {asig.max():.4f} at sigma={sigs[np.argmax(asig)]:.4f}')

    # Also scan at t=25 (near third zero)
    print(f'\n  Same scan at t=25.0109 (near third zero):')
    for sigma in [0.25, 0.5, 0.75]:
        s_q = Q(sigma, 25.0109, 0, 0)
        xi, pf, arch, euler = completed_zeta_quat(s_q, primes)
        ang = angle_between(arch, euler)
        print(f'    sigma={sigma:.2f}: angle = {ang:.4f} rad ({ang/np.pi:.4f}*pi)')

    # ══════════════════════════════════════════════════════════════
    # 4. ARCHIMEDEAN vs EULER: COMPONENT-BY-COMPONENT
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  4. COMPONENT DECOMPOSITION: L_inf vs Euler product')
    print('#' * 76)

    print(f'\n  At s = 0.5 + 14.1347i (critical line, near first zero):')
    s_q = Q(0.5, 14.1347, 0, 0)
    xi, pf, arch, euler = completed_zeta_quat(s_q, primes)
    print(f'    Prefactor (1/2)s(s-1):  {pf}')
    print(f'    L_inf (archimedean):    {arch}')
    print(f'    Euler product:          {euler}')
    print(f'    xi = pf * L_inf * Euler: {xi}')
    print(f'    |xi| = {xi.norm():.6e}')

    # Ratio: how much of xi comes from each factor?
    print(f'\n    |pf| = {pf.norm():.6f}')
    print(f'    |L_inf| = {arch.norm():.6e}')
    print(f'    |Euler| = {euler.norm():.6f}')
    print(f'    |pf|*|L_inf|*|Euler| = {pf.norm()*arch.norm()*euler.norm():.6e}')
    print(f'    |xi| = {xi.norm():.6e}')
    print(f'    Ratio (cancellation): {xi.norm()/(pf.norm()*arch.norm()*euler.norm()+1e-30):.6f}')

    # ══════════════════════════════════════════════════════════════
    # 5. OFF THE COMPLEX SLICE: j-DIRECTION SEPARATION
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  5. OFF THE SLICE: does adding j separate L_inf from Euler?')
    print('#' * 76)

    print(f'\n  s = 0.5 + 14.1347*i + u*j for varying u:')
    print(f'  {"u":>6s} {"angle(L,E)":>12s} {"L_inf j/a":>10s} {"Euler j/a":>10s} '
          f'{"DIFF j/a":>10s} {"|xi|":>12s}')
    print('  ' + '-' * 65)

    for u in [0, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0]:
        s_q = Q(0.5, 14.1347, u, 0)
        xi, pf, arch, euler = completed_zeta_quat(s_q, primes)
        ang = angle_between(arch, euler)

        # j/a ratios
        l_ja = arch.c / arch.a if abs(arch.a) > 1e-10 else float('inf')
        e_ja = euler.c / euler.a if abs(euler.a) > 1e-10 else float('inf')
        diff_ja = l_ja - e_ja

        print(f'  {u:>6.2f} {ang:>12.4f} {l_ja:>+10.4f} {e_ja:>+10.4f} '
              f'{diff_ja:>+10.4f} {xi.norm():>12.4e}')
    sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════
    # 6. PER-PRIME QUATERNIONIC DIRECTION
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  6. EACH PRIME\'S QUATERNIONIC DIRECTION')
    print('#' * 76)

    print(f'\n  The Euler factor for prime p: (1 - p^{{-s}})^{{-1}}')
    print(f'  Each prime contributes a quaternion. What direction does it point?')
    print(f'\n  s = 0.5 + 14.1347i + 1.0j:')
    print(f'  {"prime":>6s} {"factor":>40s} {"j/a ratio":>10s} {"angle to L_inf":>14s}')
    print('  ' + '-' * 75)

    s_q = Q(0.5, 14.1347, 1.0, 0)
    arch = archimedean_factor_quat(s_q)

    for p in primes[:20]:
        p_neg_s = qexp(Q(-s_q.a*np.log(p), -s_q.b*np.log(p),
                          -s_q.c*np.log(p), -s_q.d*np.log(p)))
        factor = (Q(1,0,0,0) - p_neg_s)
        ns = factor.norm_sq()
        inv_f = factor.conj() / ns if ns > 1e-30 else Q(0,0,0,0)

        ja = inv_f.c / inv_f.a if abs(inv_f.a) > 1e-10 else float('inf')
        ang = angle_between(inv_f, arch)

        print(f'  {p:>6d} ({inv_f.a:+.4f}, {inv_f.b:+.4f}i, {inv_f.c:+.4f}j, {inv_f.d:+.4f}k) '
              f'{ja:>+10.4f} {ang:>14.4f} rad')

    # Do all primes point in the same direction?
    prime_angles_to_arch = []
    prime_ja_ratios = []
    for p in primes:
        p_neg_s = qexp(Q(-s_q.a*np.log(p), -s_q.b*np.log(p),
                          -s_q.c*np.log(p), -s_q.d*np.log(p)))
        factor = (Q(1,0,0,0) - p_neg_s)
        ns = factor.norm_sq()
        inv_f = factor.conj() / ns if ns > 1e-30 else Q(0,0,0,0)
        if inv_f.norm() > 1e-10:
            prime_angles_to_arch.append(angle_between(inv_f, arch))
            if abs(inv_f.a) > 1e-10:
                prime_ja_ratios.append(inv_f.c / inv_f.a)

    pa = np.array(prime_angles_to_arch)
    pj = np.array(prime_ja_ratios)
    print(f'\n  Angle to L_inf across all {len(primes)} primes:')
    print(f'    Mean: {np.mean(pa):.4f} rad ({np.mean(pa)/np.pi:.4f}*pi)')
    print(f'    Std:  {np.std(pa):.4f} rad')
    print(f'    Min:  {np.min(pa):.4f} rad (p={primes[np.argmin(pa)]})')
    print(f'    Max:  {np.max(pa):.4f} rad (p={primes[np.argmax(pa)]})')

    print(f'\n  j/a ratio across primes:')
    print(f'    Mean: {np.mean(pj):+.4f}')
    print(f'    Std:  {np.std(pj):.4f}')

    print(f'\n  L_inf j/a ratio: {arch.c/arch.a if abs(arch.a) > 1e-10 else "inf":+.4f}')
    diff_mean = np.mean(pj) - (arch.c/arch.a if abs(arch.a) > 1e-10 else 0)
    print(f'  Mean(prime j/a) - L_inf j/a = {diff_mean:+.4f}')
    if abs(diff_mean) > 0.01:
        print(f'  *** PRIMES AND PI PROJECT DIFFERENTLY ONTO j-AXIS ***')
    sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════
    # 7. THE FUNCTIONAL EQUATION IN H
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  7. FUNCTIONAL EQUATION xi(s) = xi(1-s) IN H')
    print('#' * 76)

    print(f'\n  Testing xi(s) vs xi(1-s) for quaternionic s:')
    for s_vals in [(0.5, 14.1347, 0, 0), (0.5, 14.1347, 1.0, 0),
                   (0.3, 10.0, 0, 0), (0.3, 10.0, 0.5, 0.5)]:
        s_q = Q(*s_vals)
        one_minus_s = Q(1-s_q.a, -s_q.b, -s_q.c, -s_q.d)

        xi_s, _, _, _ = completed_zeta_quat(s_q, primes)
        xi_1ms, _, _, _ = completed_zeta_quat(one_minus_s, primes)

        diff = xi_s - xi_1ms
        print(f'\n  s = {s_q}')
        print(f'    xi(s)   = {xi_s}')
        print(f'    xi(1-s) = {xi_1ms}')
        print(f'    |diff|/|xi(s)| = {diff.norm()/(xi_s.norm()+1e-30):.6f}')

    # ══════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '=' * 76)
    print('  SESSION 45j SYNTHESIS')
    print('=' * 76)

    print(f'''
  PI AS THE ARCHIMEDEAN PRIME IN QUATERNIONIC SPACE:

  The completed zeta xi(s) = prefactor * L_inf(s) * Euler_product(s)
  factors into an archimedean part (involving pi) and a finite-prime part.

  In complex analysis: both are complex numbers, same plane, can cancel.
  In quaternionic analysis: they're quaternions. The question was whether
  they point in DIFFERENT directions in H.

  RESULTS:
  - On the complex slice: angle between L_inf and Euler is determined
    by arg(L_inf) - arg(Euler). No new quaternionic content.
  - Off the slice (adding j-component): L_inf and Euler acquire j-parts.
    The j/a ratios may DIFFER between the archimedean and prime factors.
  - Per-prime analysis: each prime factor points in a specific
    quaternionic direction. The SPREAD of directions is the key.
  - Functional equation: xi(s) = xi(1-s) should hold in H by slice
    regularity (it holds on every complex slice, hence everywhere).
''')

    print('=' * 76)
    print('  SESSION 45j COMPLETE')
    print('=' * 76)
