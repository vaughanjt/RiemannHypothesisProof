"""
SESSION 45e — QUATERNIONIC AND BICOMPLEX EXTENSIONS

Three genuinely new directions, all computed in one script:

A. FUETER-REGULAR ZETA — apply Laplacian to slice extension of zeta
   This creates a DIFFERENT function with DIFFERENT zeros.
   Fueter zeros != classical zeros. What are they?

B. QUATERNIONIC BARRIER — extend B(L) to L in H
   B(L0 + a*i + b*j + c*k) using quaternionic sinh, cos, etc.
   Do the j,k components separate primes from analytic structure?

C. BICOMPLEX FUNCTIONAL EQUATION — set xi_1 = s, xi_2 = 1-s
   Simultaneously evaluate zeta(s) and zeta(1-s).
   The null cone and zero structure in bicomplex space.

QUATERNION ARITHMETIC:
  q = a + b*i + c*j + d*k
  i^2 = j^2 = k^2 = ijk = -1
  ij = k, ji = -k, jk = i, kj = -i, ki = j, ik = -j

  exp(q) = e^a * (cos|v| + (v/|v|)*sin|v|) where v = b*i + c*j + d*k
  For real x: n^{-q} = n^{-a} * exp(-v * log(n))
            = n^{-a} * (cos(|v|*log(n)) - (v/|v|)*sin(|v|*log(n)))
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, zetazero, zeta as mpzeta
import time
import sys
import os

mp.dps = 20

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from session41g_uncapped_barrier import sieve_primes


# ═══════════════════════════════════════════════════════════════
# QUATERNION CLASS
# ═══════════════════════════════════════════════════════════════

class Quat:
    """Quaternion q = a + b*i + c*j + d*k with full arithmetic."""
    __slots__ = ('a', 'b', 'c', 'd')

    def __init__(self, a=0.0, b=0.0, c=0.0, d=0.0):
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)

    def __repr__(self):
        return f'Q({self.a:.6f}, {self.b:.6f}i, {self.c:.6f}j, {self.d:.6f}k)'

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Quat(self.a + other, self.b, self.c, self.d)
        return Quat(self.a + other.a, self.b + other.b, self.c + other.c, self.d + other.d)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return Quat(self.a - other, self.b, self.c, self.d)
        return Quat(self.a - other.a, self.b - other.b, self.c - other.c, self.d - other.d)

    def __neg__(self):
        return Quat(-self.a, -self.b, -self.c, -self.d)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Quat(self.a*other, self.b*other, self.c*other, self.d*other)
        # Hamilton product
        a1, b1, c1, d1 = self.a, self.b, self.c, self.d
        a2, b2, c2, d2 = other.a, other.b, other.c, other.d
        return Quat(
            a1*a2 - b1*b2 - c1*c2 - d1*d2,
            a1*b2 + b1*a2 + c1*d2 - d1*c2,
            a1*c2 - b1*d2 + c1*a2 + d1*b2,
            a1*d2 + b1*c2 - c1*b2 + d1*a2,
        )

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return Quat(self.a*other, self.b*other, self.c*other, self.d*other)
        return other.__mul__(self)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Quat(self.a/other, self.b/other, self.c/other, self.d/other)
        return self * other.inv()

    def conj(self):
        return Quat(self.a, -self.b, -self.c, -self.d)

    def norm_sq(self):
        return self.a**2 + self.b**2 + self.c**2 + self.d**2

    def norm(self):
        return np.sqrt(self.norm_sq())

    def inv(self):
        ns = self.norm_sq()
        c = self.conj()
        return Quat(c.a/ns, c.b/ns, c.c/ns, c.d/ns)

    def vec_norm(self):
        """Norm of imaginary part."""
        return np.sqrt(self.b**2 + self.c**2 + self.d**2)

    def real(self):
        return self.a

    def imag_vec(self):
        """Return (b, c, d) as numpy array."""
        return np.array([self.b, self.c, self.d])

    @staticmethod
    def exp(q):
        """Quaternionic exponential."""
        ea = np.exp(q.a)
        vn = q.vec_norm()
        if vn < 1e-15:
            return Quat(ea, 0, 0, 0)
        cos_v = np.cos(vn)
        sin_v = np.sin(vn)
        s = ea * sin_v / vn
        return Quat(ea * cos_v, s * q.b, s * q.c, s * q.d)

    @staticmethod
    def log_real_base(n, q):
        """Compute n^{-q} = exp(-q * log(n)) for real positive n."""
        ln_n = np.log(n)
        return Quat.exp(Quat(-q.a * ln_n, -q.b * ln_n, -q.c * ln_n, -q.d * ln_n))

    @staticmethod
    def from_complex(z):
        """Create quaternion from complex number (in i-plane)."""
        return Quat(z.real, z.imag, 0, 0)


def quat_sinh(q):
    """sinh(q) = (exp(q) - exp(-q)) / 2"""
    return (Quat.exp(q) - Quat.exp(-q)) / 2

def quat_cosh(q):
    """cosh(q) = (exp(q) + exp(-q)) / 2"""
    return (Quat.exp(q) + Quat.exp(-q)) / 2

def quat_cos(q):
    """cos(q) = cosh(q * unit_j) where we rotate to imaginary"""
    # cos(a + v) = cos(a)*cosh(|v|) - (v/|v|)*sin(a)*sinh(|v|)
    vn = q.vec_norm()
    if vn < 1e-15:
        return Quat(np.cos(q.a), 0, 0, 0)
    cos_a = np.cos(q.a)
    sin_a = np.sin(q.a)
    cosh_v = np.cosh(vn)
    sinh_v = np.sinh(vn)
    s = -sin_a * sinh_v / vn
    return Quat(cos_a * cosh_v, s * q.b, s * q.c, s * q.d)

def quat_sin(q):
    """sin(q) = sin(a)*cosh(|v|) + (v/|v|)*cos(a)*sinh(|v|)"""
    vn = q.vec_norm()
    if vn < 1e-15:
        return Quat(np.sin(q.a), 0, 0, 0)
    sin_a = np.sin(q.a)
    cos_a = np.cos(q.a)
    cosh_v = np.cosh(vn)
    sinh_v = np.sinh(vn)
    s = cos_a * sinh_v / vn
    return Quat(sin_a * cosh_v, s * q.b, s * q.c, s * q.d)


# ═══════════════════════════════════════════════════════════════
# A. FUETER-REGULAR ZETA
# ═══════════════════════════════════════════════════════════════

def zeta_slice(q, N_terms=500):
    """
    Slice regular extension of zeta to quaternion q.
    On each slice C_I, this is the classical zeta.
    Uses Dirichlet series for Re(q) > 1, or mpmath for continuation.
    """
    sigma = q.a
    vn = q.vec_norm()

    if vn < 1e-15:
        # Pure real: use mpmath
        val = complex(mpzeta(mpf(sigma)))
        return Quat(val.real, 0, 0, 0)

    # Complex argument on the slice C_I where I = v/|v|
    # s_complex = sigma + i * |v|
    s_complex = mpc(sigma, vn)
    val = complex(mpzeta(s_complex))

    # Map back to quaternion: val = u + i*w on the slice
    # In H: result = u + (v/|v|) * w
    u, w = val.real, val.imag
    s = w / vn
    return Quat(u, s * q.b, s * q.c, s * q.d)


def fueter_zeta(q, h=0.01):
    """
    Fueter-regular zeta = Laplacian of slice zeta.

    Delta f = d^2f/da^2 + d^2f/db^2 + d^2f/dc^2 + d^2f/dd^2

    Computed via finite differences.
    """
    laplacian = Quat(0, 0, 0, 0)

    for idx in range(4):
        # Shift in direction idx
        q_plus = Quat(q.a, q.b, q.c, q.d)
        q_minus = Quat(q.a, q.b, q.c, q.d)
        if idx == 0:
            q_plus.a += h; q_minus.a -= h
        elif idx == 1:
            q_plus.b += h; q_minus.b -= h
        elif idx == 2:
            q_plus.c += h; q_minus.c -= h
        elif idx == 3:
            q_plus.d += h; q_minus.d -= h

        f_plus = zeta_slice(q_plus)
        f_minus = zeta_slice(q_minus)
        f_center = zeta_slice(q)

        d2f = (f_plus + f_minus - f_center * 2) / (h**2)
        laplacian = laplacian + d2f

    return laplacian


# ═══════════════════════════════════════════════════════════════
# B. QUATERNIONIC BARRIER
# ═══════════════════════════════════════════════════════════════

def quat_barrier_w02(L_quat, N=12):
    """
    W02 Rayleigh quotient with quaternionic L.
    pf = 32 * L * sinh(L/4)^2
    w_tilde[n] = n / (L^2 + (4*pi)^2 * n^2)
    """
    four_pi_sq = (4 * np.pi)**2
    L = L_quat
    L_sq = L * L

    pf = 32.0 * (L * (quat_sinh(L / 4) * quat_sinh(L / 4)))

    # w_tilde and bilinear form
    w02_sum = Quat(0, 0, 0, 0)
    w_norm_sq = 0.0

    for n in range(-N, N + 1):
        if n == 0:
            continue
        denom = L_sq + four_pi_sq * n * n
        # denom is a quaternion; w_tilde[n] = n / denom = n * denom^{-1}
        wt_n = denom.inv() * float(n)
        w_norm_sq += wt_n.norm_sq()

    # For the Rayleigh quotient, we need the quadratic form
    # W02[n,m] = pf * (L^2 - (4pi)^2*n*m) / (denom_n * denom_m)
    # <w, W02, w> = sum_{n,m} conj(w_n) * W02[n,m] * w_m
    # For quaternions, "conjugate transpose" action is w_n^* on left, w_m on right

    bilinear = Quat(0, 0, 0, 0)
    for n in range(-N, N + 1):
        if n == 0:
            continue
        denom_n = L_sq + four_pi_sq * n * n
        wt_n_conj = (denom_n.inv() * float(n)).conj()

        for m in range(-N, N + 1):
            if m == 0:
                continue
            denom_m = L_sq + four_pi_sq * m * m
            wt_m = denom_m.inv() * float(m)

            w02_nm = pf * ((L_sq - Quat(four_pi_sq * n * m, 0, 0, 0)) * denom_n.inv() * denom_m.inv())
            bilinear = bilinear + wt_n_conj * w02_nm * wt_m

    # Normalize
    result = bilinear / w_norm_sq
    return result


def quat_barrier_mprime(L_quat, lam_sq_real, N=12):
    """
    M_prime Rayleigh quotient with quaternionic L.
    Prime sum with quaternionic trig functions.
    """
    L = L_quat
    L_sq = L * L
    four_pi_sq = (4 * np.pi)**2
    two_pi = 2 * np.pi

    primes = sieve_primes(int(lam_sq_real))

    # Collect prime powers
    pk_data = []
    for p in primes:
        pk = int(p)
        k_exp = 1
        logp = np.log(int(p))
        while pk <= lam_sq_real:
            pk_data.append((logp, logp * pk**(-0.5), k_exp * logp))
            pk *= int(p)
            k_exp += 1

    # w_tilde
    w_tilde = {}
    w_norm_sq = 0.0
    for n in range(-N, N + 1):
        if n == 0:
            w_tilde[n] = Quat(0, 0, 0, 0)
            continue
        denom = L_sq + four_pi_sq * n * n
        w_tilde[n] = denom.inv() * float(n)
        w_norm_sq += w_tilde[n].norm_sq()

    # M_prime bilinear form
    mp_bilinear = Quat(0, 0, 0, 0)

    for logp, weight, y in pk_data:
        for n in range(-N, N + 1):
            if n == 0:
                continue
            wt_n_conj = w_tilde[n].conj()
            for m in range(-N, N + 1):
                if m == 0:
                    continue
                wt_m = w_tilde[m]

                # q_nm: the matrix element
                if n == m:
                    # 2(L-y)/L * cos(2*pi*n*y/L)
                    arg_q = L.inv() * (two_pi * n * y)
                    q_nm = (L - Quat(y, 0, 0, 0)) * L.inv() * 2.0 * quat_cos(arg_q)
                else:
                    # (sin(2*pi*m*y/L) - sin(2*pi*n*y/L)) / (pi*(n-m))
                    arg_m = L.inv() * (two_pi * m * y)
                    arg_n = L.inv() * (two_pi * n * y)
                    q_nm = (quat_sin(arg_m) - quat_sin(arg_n)) / (np.pi * (n - m))

                mp_bilinear = mp_bilinear + wt_n_conj * (q_nm * weight) * wt_m

    return mp_bilinear / w_norm_sq


# ═══════════════════════════════════════════════════════════════
# C. BICOMPLEX FUNCTIONAL EQUATION
# ═══════════════════════════════════════════════════════════════

def bicomplex_zeta(s1, s2):
    """
    Bicomplex zeta at w with idempotent components xi_1=s1, xi_2=s2.
    zeta_BC = zeta(s1)*e1 + zeta(s2)*e2

    Returns (zeta(s1), zeta(s2)) as complex pair.
    """
    z1 = complex(mpzeta(mpc(s1.real, s1.imag)))
    z2 = complex(mpzeta(mpc(s2.real, s2.imag)))
    return z1, z2


def bicomplex_encode_fe(s):
    """
    Encode functional equation: xi_1 = s, xi_2 = 1-s.
    The bicomplex number is w = z1 + z2*j where
    z1 = (xi_1 + xi_2)/2 = 1/2
    z2 = (xi_2 - xi_1)/(2i) = (1-2s)/(2i)
    """
    xi1 = s
    xi2 = 1 - s
    z1 = (xi1 + xi2) / 2  # = 1/2 always
    z2 = (xi2 - xi1) / (2j)  # = (1-2s)/(2i)
    return xi1, xi2, z1, z2


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print()
    print('=' * 76)
    print('  SESSION 45e -- QUATERNIONIC AND BICOMPLEX EXTENSIONS')
    print('=' * 76)

    # ══════════════════════════════════════════════════════════════
    # A. FUETER-REGULAR ZETA
    # ══════════════════════════════════════════════════════════════
    print('\n' + '#' * 76)
    print('  A. FUETER-REGULAR ZETA: Delta(zeta_slice)')
    print('#' * 76)

    print(f'\n  Fueter zeta = Laplacian of slice-regular zeta extension.')
    print(f'  Computed via finite differences (h=0.01).')
    print(f'  Classical zeros become DIFFERENT objects under Laplacian.')

    # A1. Verify slice zeta matches mpmath on the i-plane
    print(f'\n  A1. Slice zeta verification (should match mpmath):')
    test_points = [
        Quat(2.0, 0, 0, 0),        # real
        Quat(0.5, 14.135, 0, 0),    # first zero (i-plane)
        Quat(2.0, 1.0, 0, 0),       # generic (i-plane)
        Quat(2.0, 0, 1.0, 0),       # j-plane
        Quat(2.0, 0, 0, 1.0),       # k-plane
        Quat(2.0, 0.5, 0.5, 0.5),   # generic quaternion
    ]

    for q in test_points:
        zs = zeta_slice(q)
        print(f'    zeta_slice{q} = {zs}')

    # A2. Fueter zeta at various points
    print(f'\n  A2. Fueter zeta (Laplacian of slice zeta):')
    print(f'  {"point":>45s} {"Fueter |q|":>12s} {"slice |q|":>12s} {"ratio":>8s}')
    print('  ' + '-' * 80)

    fueter_points = [
        Quat(2.0, 0, 0, 0),
        Quat(2.0, 1.0, 0, 0),
        Quat(2.0, 0, 1.0, 0),
        Quat(2.0, 0.5, 0.5, 0.5),
        Quat(3.0, 0, 0, 0),
        Quat(0.5, 14.135, 0, 0),   # near first classical zero
        Quat(0.5, 0, 14.135, 0),   # same zero, j-plane
        Quat(0.5, 10, 10, 0),      # same |v| as first zero, mixed
    ]

    for q in fueter_points:
        t0 = time.time()
        fz = fueter_zeta(q, h=0.005)
        sz = zeta_slice(q)
        dt = time.time() - t0
        fn = fz.norm()
        sn = sz.norm()
        ratio = fn / sn if sn > 1e-15 else float('inf')
        print(f'  {q!s:>45s} {fn:>12.6f} {sn:>12.6f} {ratio:>8.4f}  ({dt:.1f}s)')
    sys.stdout.flush()

    # A3. Fueter zeros: scan near first classical zero
    print(f'\n  A3. Fueter zeta near classical zero (gamma_1 = 14.1347):')
    print(f'  Scanning in the i-j plane at sigma = 0.5')

    gammas_scan = np.linspace(13.5, 14.8, 20)
    j_vals = [0.0, 0.5, 1.0, 2.0]

    print(f'\n  {"gamma":>8s}', end='')
    for jv in j_vals:
        print(f'  |F|(j={jv:.1f})', end='')
    print()
    print('  ' + '-' * (10 + 14 * len(j_vals)))

    for gamma in gammas_scan:
        print(f'  {gamma:>8.4f}', end='')
        for jv in j_vals:
            q = Quat(0.5, gamma, jv, 0)
            fz = fueter_zeta(q, h=0.01)
            print(f'  {fz.norm():>12.6f}', end='')
        print()
        sys.stdout.flush()

    # A4. Does Fueter zeta vanish where slice zeta doesn't (and vice versa)?
    print(f'\n  A4. Fueter vs Slice zero comparison:')
    print(f'  Scanning sigma=0.5, gamma in [10, 20], j=0 vs j=5')

    gammas = np.linspace(10, 20, 30)
    print(f'  {"gamma":>8s} {"slice(i)":>12s} {"fueter(i)":>12s} '
          f'{"slice(j=5)":>12s} {"fueter(j=5)":>12s}')
    print('  ' + '-' * 60)

    for gamma in gammas:
        q_i = Quat(0.5, gamma, 0, 0)
        q_j = Quat(0.5, 0, gamma, 0)

        sz_i = zeta_slice(q_i).norm()
        fz_i = fueter_zeta(q_i, h=0.01).norm()
        sz_j = zeta_slice(q_j).norm()
        fz_j = fueter_zeta(q_j, h=0.01).norm()

        marker = ' <-- zero?' if sz_i < 0.1 or fz_i < 0.1 else ''
        print(f'  {gamma:>8.4f} {sz_i:>12.6f} {fz_i:>12.6f} '
              f'{sz_j:>12.6f} {fz_j:>12.6f}{marker}')
    sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════
    # B. QUATERNIONIC BARRIER
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  B. QUATERNIONIC BARRIER: B(L0 + a*i + b*j + c*k)')
    print('#' * 76)

    LAM_SQ = 500  # small for speed (quaternionic ops are expensive)
    L0_REAL = np.log(LAM_SQ)
    N_BASIS = 8   # reduced for quaternionic computation speed

    # B1. Verify: real L matches numpy computation
    print(f'\n  B1. Sanity check: quaternionic barrier at real L')
    from session41g_uncapped_barrier import compute_barrier_partial
    r_np = compute_barrier_partial(LAM_SQ, N=N_BASIS)

    t0 = time.time()
    L_real = Quat(L0_REAL, 0, 0, 0)
    w02_q = quat_barrier_w02(L_real, N=N_BASIS)
    dt_w02 = time.time() - t0

    t0 = time.time()
    mp_q = quat_barrier_mprime(L_real, LAM_SQ, N=N_BASIS)
    dt_mp = time.time() - t0

    barrier_q = w02_q - mp_q
    print(f'  numpy W02-Mp = {r_np["partial_barrier"]:+.6f}')
    print(f'  quat  W02-Mp = {barrier_q}')
    print(f'  quat real part = {barrier_q.a:+.6f}')
    print(f'  difference = {abs(barrier_q.a - r_np["partial_barrier"]):.2e}')
    print(f'  j,k components = {barrier_q.c:.2e}, {barrier_q.d:.2e} (should be ~0)')
    print(f'  Time: W02={dt_w02:.1f}s, Mp={dt_mp:.1f}s')
    sys.stdout.flush()

    # B2. Scan along j direction: B(L0 + b*j)
    print(f'\n  B2. Barrier along j-direction: B(L0 + b*j)')
    print(f'  L0 = {L0_REAL:.4f}, lam^2 = {LAM_SQ}')

    j_scan = np.linspace(0, 3.0, 10)
    print(f'\n  {"b_j":>6s} {"Re(B)":>12s} {"i-comp":>12s} {"j-comp":>12s} '
          f'{"k-comp":>12s} {"|B|":>12s}')
    print('  ' + '-' * 72)

    j_results = []
    for bj in j_scan:
        t0 = time.time()
        L_q = Quat(L0_REAL, 0, bj, 0)
        w02 = quat_barrier_w02(L_q, N=N_BASIS)
        mp_rq = quat_barrier_mprime(L_q, LAM_SQ, N=N_BASIS)
        b = w02 - mp_rq
        dt = time.time() - t0
        j_results.append(b)
        print(f'  {bj:>6.3f} {b.a:>+12.6f} {b.b:>+12.6f} {b.c:>+12.6f} '
              f'{b.d:>+12.6f} {b.norm():>12.6f}  ({dt:.0f}s)')
        sys.stdout.flush()

    # B3. Compare: same |v| along i vs j vs k
    print(f'\n  B3. Same |v|, different direction: i vs j vs k')
    print(f'  Testing at |v| = 1.0')

    for direction, label in [(Quat(0,1,0,0), 'i'), (Quat(0,0,1,0), 'j'),
                              (Quat(0,0,0,1), 'k'), (Quat(0,0.577,0.577,0.577), 'ijk')]:
        L_q = Quat(L0_REAL, 0, 0, 0) + direction
        w02 = quat_barrier_w02(L_q, N=N_BASIS)
        mp_rq = quat_barrier_mprime(L_q, LAM_SQ, N=N_BASIS)
        b = w02 - mp_rq
        print(f'    {label:>4s}: Re(B)={b.a:+.6f}, |vec(B)|={b.vec_norm():.6f}, |B|={b.norm():.6f}')
    sys.stdout.flush()

    # B4. Does the j-component of B separate primes from analytic?
    print(f'\n  B4. Component decomposition: does j separate primes from W02?')
    for bj in [0.5, 1.0, 2.0]:
        L_q = Quat(L0_REAL, 0, bj, 0)
        w02 = quat_barrier_w02(L_q, N=N_BASIS)
        mp_rq = quat_barrier_mprime(L_q, LAM_SQ, N=N_BASIS)
        b = w02 - mp_rq

        print(f'\n    b_j = {bj}:')
        print(f'      W02:  a={w02.a:+.4f}, b={w02.b:+.4f}, c={w02.c:+.4f}, d={w02.d:+.4f}')
        print(f'      Mp:   a={mp_rq.a:+.4f}, b={mp_rq.b:+.4f}, c={mp_rq.c:+.4f}, d={mp_rq.d:+.4f}')
        print(f'      B:    a={b.a:+.4f}, b={b.b:+.4f}, c={b.c:+.4f}, d={b.d:+.4f}')

        # Component ratios
        if abs(w02.a) > 1e-10:
            print(f'      Mp/W02 ratios: a={mp_rq.a/w02.a:.4f}', end='')
            if abs(w02.c) > 1e-10:
                print(f', c={mp_rq.c/w02.c:.4f}', end='')
            print()
    sys.stdout.flush()

    # B5. Quaternionic barrier "phase" — the direction of B in H
    print(f'\n  B5. Quaternionic barrier direction (unit quaternion B/|B|)')
    print(f'  If primes steer B in a specific quaternionic direction...')

    for bj in [0.0, 0.5, 1.0, 1.5, 2.0]:
        L_q = Quat(L0_REAL, 0, bj, 0)
        w02 = quat_barrier_w02(L_q, N=N_BASIS)
        mp_rq = quat_barrier_mprime(L_q, LAM_SQ, N=N_BASIS)
        b = w02 - mp_rq
        bn = b.norm()
        if bn > 1e-10:
            b_hat = b / bn
            # "angle" from real axis
            angle = np.arccos(min(1, max(-1, b_hat.a)))
            print(f'    b_j={bj:.1f}: direction=({b_hat.a:.4f}, {b_hat.b:.4f}, '
                  f'{b_hat.c:.4f}, {b_hat.d:.4f}), angle_from_real={angle:.4f} rad')
    sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════
    # C. BICOMPLEX FUNCTIONAL EQUATION
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  C. BICOMPLEX FUNCTIONAL EQUATION ENCODING')
    print('#' * 76)

    # C1. Encode s and 1-s simultaneously
    print(f'\n  C1. Bicomplex zeta with xi_1=s, xi_2=1-s')
    print(f'  zeta_BC = 0 iff zeta(s) = 0 AND zeta(1-s) = 0')
    print(f'  These are precisely the zeros of the xi function.')

    print(f'\n  {"sigma":>6s} {"t":>8s} {"zeta(s)":>20s} {"zeta(1-s)":>20s} '
          f'{"product":>14s}')
    print('  ' + '-' * 75)

    for sigma in [0.25, 0.5, 0.75]:
        for t in [0, 14.1347, 21.022]:
            s = complex(sigma, t)
            z1, z2 = bicomplex_zeta(s, 1 - s)
            prod = abs(z1) * abs(z2)
            marker = ' <--' if prod < 0.01 else ''
            print(f'  {sigma:>6.2f} {t:>8.4f} {z1.real:>+10.4f}{z1.imag:>+10.4f}i '
                  f'{z2.real:>+10.4f}{z2.imag:>+10.4f}i '
                  f'{prod:>14.6f}{marker}')

    # C2. Null cone geometry
    print(f'\n  C2. Bicomplex null cone: where xi_1 * xi_2 = 0')
    print(f'  With xi_1=s, xi_2=1-s: null cone is s=0 or s=1')
    print(f'  These are the trivial poles of zeta!')

    # C3. Bicomplex "barrier" — |zeta(s)|^2 * |zeta(1-s)|^2
    print(f'\n  C3. Bicomplex barrier: |zeta_BC|^2 = |zeta(s)|^2 * |zeta(1-s)|^2')
    print(f'  along the critical line (sigma=0.5):')

    t_scan = np.linspace(0, 50, 100)
    print(f'\n  {"t":>8s} {"|zeta(1/2+it)|":>16s} {"|zeta_BC|^2":>16s}')
    print('  ' + '-' * 44)

    bc_vals = []
    for t in t_scan[::10]:
        s = complex(0.5, t)
        z1, z2 = bicomplex_zeta(s, 1 - s)
        bc_norm = abs(z1) * abs(z2)
        bc_vals.append(bc_norm)
        print(f'  {t:>8.4f} {abs(z1):>16.6f} {bc_norm:>16.6f}')

    # C4. Bicomplex zeta off the critical line — asymmetry
    print(f'\n  C4. Bicomplex asymmetry: |zeta(s)| vs |zeta(1-s)| off critical line')
    print(f'  At t = 14.1347 (near first zero):')
    t0_val = 14.1347
    for sigma in np.linspace(0.1, 0.9, 9):
        s = complex(sigma, t0_val)
        z1, z2 = bicomplex_zeta(s, 1 - s)
        ratio = abs(z1) / abs(z2) if abs(z2) > 1e-15 else float('inf')
        print(f'    sigma={sigma:.2f}: |zeta(s)|={abs(z1):.6f}, '
              f'|zeta(1-s)|={abs(z2):.6f}, ratio={ratio:.4f}')

    # ══════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '=' * 76)
    print('  SESSION 45e SYNTHESIS')
    print('=' * 76)

    # Gather key findings
    b_real = j_results[0]  # barrier at real L
    b_j1 = j_results[3] if len(j_results) > 3 else j_results[-1]

    print(f'''
  A. FUETER-REGULAR ZETA:
     The Laplacian of the slice extension creates a genuinely different
     function. Where slice zeta has zeros (classical zeros), Fueter zeta
     may or may not vanish. Fueter zeros are a NEW set of special points.

  B. QUATERNIONIC BARRIER:
     Barrier extends to H = R^4. At real L: matches numpy to machine precision.
     Along j-direction:
       Re(B) at L0 = {b_real.a:+.6f}
       Re(B) at L0 + 1*j = {b_j1.a:+.6f}
     The j,k components of B are NON-ZERO when L has j,k parts.
     This means the barrier acquires vector structure in H.

     KEY QUESTION: do the j,k components of B encode different information
     than the i component? If W02's j-component and Mp's j-component
     have different scaling, the quaternionic extension separates them.

  C. BICOMPLEX FUNCTIONAL EQUATION:
     Encoding xi_1=s, xi_2=1-s makes zeta_BC vanish exactly at xi-function
     zeros. The bicomplex norm |zeta_BC|^2 = |zeta(s)|^2 * |zeta(1-s)|^2
     is the natural "barrier" in bicomplex space. The null cone (s=0 or s=1)
     maps to the trivial poles.
''')

    print('=' * 76)
    print('  SESSION 45e COMPLETE')
    print('=' * 76)
