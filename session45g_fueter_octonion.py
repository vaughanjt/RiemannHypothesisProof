"""
SESSION 45g — FUETER ZEROS AND OCTONIONIC EXTENSION

PART 1: EXACT FUETER-REGULAR ZETA (no finite differences!)

The slice extension zeta_slice(sigma + v) = u(sigma,r) + (v/|v|)*w(sigma,r)
where u + iw = zeta(sigma + ir), and the 4D Laplacian gives:

  zeta_Fueter = A(sigma,r) + (v/|v|) * B(sigma,r)

  A(sigma,r) = -(2/r) * Im(zeta'(sigma+ir))
  B(sigma,r) = (2/r) * Re(zeta'(sigma+ir)) - 2*Im(zeta(sigma+ir))/r^2

Derivation:
  u,w harmonic in (sigma,r) by Cauchy-Riemann.
  Delta_4D u = (2/r) * du/dr = -(2/r) * dw/dsigma = -(2/r)*Im(zeta')
  Delta_4D [(v/|v|)*w] = (v/|v|)*[(2/r)*dw/dr - 2w/r^2]
                       = (v/|v|)*[(2/r)*Re(zeta') - 2*Im(zeta)/r^2]

FUETER ZEROS require A = 0 AND B = 0:
  (1) Im(zeta'(s)) = 0
  (2) Re(zeta'(s)) = Im(zeta(s)) / Im(s)

These are a DIFFERENT set of special points than the classical zeros!

PART 2: OCTONIONIC EXTENSION

Octonions O = H + H*l where l^2 = -1, non-associative.
Cayley-Dickson: (a,b)*(c,d) = (ac - d_bar*b, da + b*c_bar)
Use the doubling to extend the barrier to 8D.
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, zeta as mpzeta, diff as mpdiff
import time
import sys

mp.dps = 25


# ═══════════════════════════════════════════════════════════════
# PART 1: EXACT FUETER-REGULAR ZETA
# ═══════════════════════════════════════════════════════════════

def fueter_exact(sigma, r):
    """
    Exact Fueter-regular zeta at (sigma, r) where r = |imaginary part|.

    Returns (A, B) where zeta_Fueter = A + (v/|v|)*B.

    A = -(2/r) * Im(zeta'(s))
    B = (2/r) * Re(zeta'(s)) - 2*Im(zeta(s))/r^2

    where s = sigma + i*r.
    """
    if abs(r) < 1e-12:
        # At r=0, need limits. zeta'(sigma) is real, so A -> 0.
        # B involves 0/0 limit, use L'Hopital or series expansion
        zp = complex(mpzeta(mpf(sigma), derivative=1))
        # A -> -(2) * d/dr[Im(zeta')] at r=0 via L'Hopital
        # This requires zeta'' which is complex. Skip for now.
        return 0.0, 0.0

    s = mpc(sigma, r)
    z = complex(mpzeta(s))
    zp = complex(mpzeta(s, derivative=1))

    A = -(2.0 / r) * zp.imag
    B = (2.0 / r) * zp.real - 2.0 * z.imag / (r * r)

    return A, B


def fueter_norm(sigma, r):
    """Norm of the Fueter-regular zeta: sqrt(A^2 + B^2)."""
    A, B = fueter_exact(sigma, r)
    return np.sqrt(A**2 + B**2)


def find_fueter_zeros_grid(sigma_range, r_range, n_sigma=200, n_r=200):
    """
    Scan a grid in (sigma, r) to find approximate Fueter zeros.
    Returns list of (sigma, r, |F|) for local minima.
    """
    sigmas = np.linspace(sigma_range[0], sigma_range[1], n_sigma)
    rs = np.linspace(max(r_range[0], 0.1), r_range[1], n_r)

    # Compute |F| on grid
    F_grid = np.zeros((n_sigma, n_r))
    A_grid = np.zeros((n_sigma, n_r))
    B_grid = np.zeros((n_sigma, n_r))

    for i, sig in enumerate(sigmas):
        for j, r in enumerate(rs):
            A, B = fueter_exact(sig, r)
            F_grid[i, j] = np.sqrt(A**2 + B**2)
            A_grid[i, j] = A
            B_grid[i, j] = B

    # Find local minima
    minima = []
    for i in range(1, n_sigma - 1):
        for j in range(1, n_r - 1):
            val = F_grid[i, j]
            if (val < F_grid[i-1, j] and val < F_grid[i+1, j] and
                val < F_grid[i, j-1] and val < F_grid[i, j+1]):
                minima.append((sigmas[i], rs[j], val, A_grid[i,j], B_grid[i,j]))

    return {
        'sigmas': sigmas, 'rs': rs,
        'F_grid': F_grid, 'A_grid': A_grid, 'B_grid': B_grid,
        'minima': sorted(minima, key=lambda x: x[2]),
    }


def refine_fueter_zero(sigma0, r0, tol=1e-10, max_iter=100):
    """Newton's method to refine a Fueter zero from initial guess."""
    sig, r = sigma0, r0
    h = 1e-6

    for iteration in range(max_iter):
        A, B = fueter_exact(sig, r)
        fn = np.sqrt(A**2 + B**2)
        if fn < tol:
            return sig, r, fn, iteration

        # Jacobian via finite differences
        A_ds, B_ds = fueter_exact(sig + h, r)
        A_dr, B_dr = fueter_exact(sig, r + h)

        dA_ds = (A_ds - A) / h
        dA_dr = (A_dr - A) / h
        dB_ds = (B_ds - B) / h
        dB_dr = (B_dr - B) / h

        # Newton step: J * delta = -F
        det = dA_ds * dB_dr - dA_dr * dB_ds
        if abs(det) < 1e-20:
            break

        dsig = -(B_dr * A - A_dr * B) / det
        dr = -(dA_ds * B - dB_ds * A) / det

        # Damped step
        step = min(1.0, 0.5 / (abs(dsig) + abs(dr) + 1e-10))
        sig += step * dsig
        r += step * dr

        # Keep r positive
        if r < 0.01:
            r = 0.01

    return sig, r, np.sqrt(A**2 + B**2), max_iter


# ═══════════════════════════════════════════════════════════════
# PART 2: OCTONION CLASS (Cayley-Dickson doubling of quaternions)
# ═══════════════════════════════════════════════════════════════

class Oct:
    """
    Octonion = pair of quaternions (a, b) via Cayley-Dickson.
    Multiplication: (a,b)*(c,d) = (ac - conj(d)*b, d*a + b*conj(c))
    where a,b,c,d are represented as 4-tuples (real numbers).

    We store as 8 real components: e0..e7
    e0 = 1, e1 = i, e2 = j, e3 = k (quaternionic)
    e4 = l, e5 = il, e6 = jl, e7 = kl (octonionic)
    """
    __slots__ = ('v',)

    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], (list, tuple, np.ndarray)):
            self.v = np.array(args[0], dtype=float)
        elif len(args) == 8:
            self.v = np.array(args, dtype=float)
        elif len(args) == 1 and isinstance(args[0], (int, float)):
            self.v = np.zeros(8)
            self.v[0] = float(args[0])
        elif len(args) == 0:
            self.v = np.zeros(8)
        else:
            self.v = np.zeros(8)
            for i, a in enumerate(args):
                if i < 8:
                    self.v[i] = float(a)

    def __repr__(self):
        labels = ['', 'i', 'j', 'k', 'l', 'il', 'jl', 'kl']
        parts = []
        for i in range(8):
            if abs(self.v[i]) > 1e-10:
                parts.append(f'{self.v[i]:+.4f}{labels[i]}')
        return 'O(' + ' '.join(parts) + ')' if parts else 'O(0)'

    def __add__(self, other):
        if isinstance(other, (int, float)):
            r = Oct(self.v.copy())
            r.v[0] += other
            return r
        return Oct(self.v + other.v)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            r = Oct(self.v.copy())
            r.v[0] -= other
            return r
        return Oct(self.v - other.v)

    def __neg__(self):
        return Oct(-self.v)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Oct(self.v * other)
        # Cayley-Dickson multiplication
        # Split into quaternion pairs: self = (a, b), other = (c, d)
        # (a,b)*(c,d) = (a*c - conj(d)*b, d*a + b*conj(c))
        a = self.v[:4]   # quaternion
        b = self.v[4:]
        c = other.v[:4]
        d = other.v[4:]

        ac = qmul(a, c)
        db = qmul(qconj(d), b)
        da = qmul(d, a)
        bc = qmul(b, qconj(c))

        result = Oct()
        result.v[:4] = ac - db
        result.v[4:] = da + bc
        return result

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return Oct(self.v * other)
        return other.__mul__(self)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Oct(self.v / other)
        return self * other.inv()

    def conj(self):
        r = Oct()
        r.v[0] = self.v[0]
        r.v[1:] = -self.v[1:]
        return r

    def norm_sq(self):
        return np.sum(self.v**2)

    def norm(self):
        return np.sqrt(self.norm_sq())

    def inv(self):
        ns = self.norm_sq()
        return self.conj() / ns

    def real(self):
        return self.v[0]

    def imag_norm(self):
        return np.sqrt(np.sum(self.v[1:]**2))


def qmul(a, b):
    """Quaternion multiplication for 4-component arrays."""
    a0, a1, a2, a3 = a
    b0, b1, b2, b3 = b
    return np.array([
        a0*b0 - a1*b1 - a2*b2 - a3*b3,
        a0*b1 + a1*b0 + a2*b3 - a3*b2,
        a0*b2 - a1*b3 + a2*b0 + a3*b1,
        a0*b3 + a1*b2 - a2*b1 + a3*b0,
    ])

def qconj(a):
    """Quaternion conjugate for 4-component array."""
    return np.array([a[0], -a[1], -a[2], -a[3]])


def oct_exp(o):
    """Octonionic exponential: exp(a + v) = e^a(cos|v| + (v/|v|)sin|v|)."""
    a = o.v[0]
    v = o.v[1:]
    vn = np.sqrt(np.sum(v**2))
    ea = np.exp(a)
    if vn < 1e-15:
        r = Oct()
        r.v[0] = ea
        return r
    r = Oct()
    r.v[0] = ea * np.cos(vn)
    r.v[1:] = ea * np.sin(vn) * v / vn
    return r


def oct_power_real_base(n, o):
    """n^{-o} = exp(-o * log(n)) for real positive integer n."""
    ln_n = np.log(n)
    neg_o_ln = Oct()
    neg_o_ln.v = -o.v * ln_n
    return oct_exp(neg_o_ln)


# ═══════════════════════════════════════════════════════════════
# OCTONIONIC ZETA (Dirichlet series)
# ═══════════════════════════════════════════════════════════════

def oct_zeta(o, N_terms=500):
    """
    Octonionic zeta: sum_{n=1}^N n^{-o} for Re(o) > 1.
    Each term is well-defined since log(n) is real.
    """
    result = Oct()
    for n in range(1, N_terms + 1):
        term = oct_power_real_base(n, o)
        result = result + term
    return result


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print()
    print('=' * 76)
    print('  SESSION 45g -- FUETER ZEROS AND OCTONIONIC EXTENSION')
    print('=' * 76)

    # ══════════════════════════════════════════════════════════════
    # 1. EXACT FUETER ZETA VERIFICATION
    # ══════════════════════════════════════════════════════════════
    print('\n' + '#' * 76)
    print('  1. EXACT FUETER ZETA (analytic formula, no finite differences)')
    print('#' * 76)

    print(f'\n  F(sigma,r) = A(sigma,r) + (v/|v|)*B(sigma,r)')
    print(f'  A = -(2/r)*Im(zeta\'(s)),  B = (2/r)*Re(zeta\'(s)) - 2*Im(zeta(s))/r^2')

    test_pts = [
        (2.0, 1.0), (2.0, 5.0), (0.5, 14.1347), (0.5, 21.022),
        (0.5, 25.011), (0.5, 10.0), (0.5, 30.0), (0.5, 40.0),
        (0.25, 14.1347), (0.75, 14.1347),
    ]

    print(f'\n  {"sigma":>6s} {"r":>10s} {"A":>14s} {"B":>14s} '
          f'{"|F|":>12s} {"|zeta|":>12s} {"F/zeta":>10s}')
    print('  ' + '-' * 72)

    for sig, r in test_pts:
        A, B = fueter_exact(sig, r)
        fn = np.sqrt(A**2 + B**2)
        zn = abs(complex(mpzeta(mpc(sig, r))))
        ratio = fn / zn if zn > 1e-15 else float('inf')
        marker = ' <-- classical zero' if zn < 0.01 else ''
        print(f'  {sig:>6.2f} {r:>10.4f} {A:>+14.6f} {B:>+14.6f} '
              f'{fn:>12.6f} {zn:>12.6f} {ratio:>10.4f}{marker}')

    # ══════════════════════════════════════════════════════════════
    # 2. FUETER ZERO SEARCH: GRID SCAN
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  2. FUETER ZERO SEARCH: grid scan in (sigma, r) plane')
    print('#' * 76)

    # Scan the critical strip
    print(f'\n  Scanning sigma in [-1, 2], r in [0.5, 50]...')
    t0 = time.time()
    result = find_fueter_zeros_grid((-1, 2), (0.5, 50), n_sigma=150, n_r=200)
    dt = time.time() - t0
    print(f'  Done ({dt:.1f}s)')

    if result['minima']:
        print(f'\n  Found {len(result["minima"])} local minima of |F|:')
        print(f'  {"sigma":>8s} {"r":>10s} {"|F|":>14s} {"A":>12s} {"B":>12s} '
              f'{"near CL?":>8s}')
        print('  ' + '-' * 60)

        for sig, r, fn, A, B in result['minima'][:20]:
            near_cl = 'YES' if abs(sig - 0.5) < 0.1 else ''
            print(f'  {sig:>8.4f} {r:>10.4f} {fn:>14.6e} {A:>+12.6f} {B:>+12.6f} '
                  f'{near_cl:>8s}')
    else:
        print(f'  No local minima found in coarse grid.')
    sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════
    # 3. REFINE FUETER ZEROS WITH NEWTON'S METHOD
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  3. REFINING FUETER ZEROS (Newton\'s method)')
    print('#' * 76)

    refined = []
    if result['minima']:
        for sig0, r0, fn0, _, _ in result['minima'][:15]:
            sig_r, r_r, fn_r, iters = refine_fueter_zero(sig0, r0)
            refined.append((sig_r, r_r, fn_r, iters))

        print(f'\n  {"sigma":>10s} {"r":>12s} {"|F|":>14s} {"iters":>6s} '
              f'{"on CL?":>8s} {"near zero?":>10s}')
        print('  ' + '-' * 66)

        for sig_r, r_r, fn_r, iters in refined:
            on_cl = 'YES' if abs(sig_r - 0.5) < 0.001 else ''
            # Check if near a classical zero
            zn = abs(complex(mpzeta(mpc(sig_r, r_r))))
            near_z = f'zeta={zn:.4f}' if zn < 1.0 else ''
            converged = '***' if fn_r < 1e-8 else ''
            print(f'  {sig_r:>10.6f} {r_r:>12.6f} {fn_r:>14.6e} {iters:>6d} '
                  f'{on_cl:>8s} {near_z:>10s} {converged}')
    sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════
    # 4. FUETER ZEROS vs CLASSICAL ZEROS: THE RELATIONSHIP
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  4. FUETER ZEROS vs CLASSICAL ZEROS')
    print('#' * 76)

    # At classical zeros: zeta(s)=0, so Im(zeta)=0
    # Condition B=0 becomes: Re(zeta'(s)) = 0
    # But zeta'(rho) != 0 for simple zeros, so B != 0 at classical zeros
    # Fueter zeros are where zeta' has special properties

    print(f'\n  At classical zeros (sigma=0.5):')
    print(f'  {"gamma":>10s} {"|zeta|":>10s} {"Im(zeta\')":>12s} {"Re(zeta\')":>12s} '
          f'{"A":>12s} {"B":>12s} {"|F|":>12s}')
    print('  ' + '-' * 76)

    from mpmath import zetazero
    for k in range(1, 11):
        gamma = float(zetazero(k).imag)
        A, B = fueter_exact(0.5, gamma)
        fn = np.sqrt(A**2 + B**2)
        s = mpc(0.5, gamma)
        z = complex(mpzeta(s))
        zp = complex(mpzeta(s, derivative=1))
        print(f'  {gamma:>10.4f} {abs(z):>10.6f} {zp.imag:>+12.6f} {zp.real:>+12.6f} '
              f'{A:>+12.6f} {B:>+12.6f} {fn:>12.6f}')

    # Key: at classical zeros, A = -(2/gamma)*Im(zeta'(rho))
    # and B = (2/gamma)*Re(zeta'(rho)) since Im(zeta(rho))=0
    # So |F| at classical zero = (2/gamma)*|zeta'(rho)|
    print(f'\n  At classical zeros: |F| = (2/gamma)*|zeta\'(rho)|')
    print(f'  The Fueter norm ENCODES the derivative of zeta at zeros!')
    print(f'  This is related to the de Bruijn-Newman Lambda constant')
    print(f'  and zero repulsion/attraction.')

    # ══════════════════════════════════════════════════════════════
    # 5. THE A=0 CURVE: WHERE Im(zeta'(s)) = 0
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  5. THE A=0 CURVE: Im(zeta\'(s)) = 0')
    print('#' * 76)

    print(f'\n  This is a 1D curve in the (sigma, r) plane.')
    print(f'  Fueter zeros lie on this curve where B=0 also holds.')
    print(f'  Scanning along sigma=0.5 (critical line):')

    print(f'\n  {"r":>10s} {"Im(zeta\')":>14s} {"Re(zeta\')":>14s} {"A":>12s} {"B":>12s}')
    print('  ' + '-' * 65)

    # Find where A crosses zero along the critical line
    a_crossings = []
    prev_A = None
    for r in np.linspace(1, 50, 500):
        A, B = fueter_exact(0.5, r)
        if prev_A is not None and prev_A * A < 0:
            a_crossings.append(r)
        prev_A = A

    print(f'  A=0 crossings on critical line (first 15):')
    for i, rc in enumerate(a_crossings[:15]):
        A, B = fueter_exact(0.5, rc)
        s = mpc(0.5, rc)
        zp = complex(mpzeta(s, derivative=1))
        z = complex(mpzeta(s))
        print(f'    r={rc:>10.4f}: A={A:+.2e}, B={B:+.6f}, |zeta|={abs(z):.6f}, '
              f'Im(zeta\')={zp.imag:+.6f}')

    # Compare to classical zero locations
    print(f'\n  Classical zeros for comparison:')
    for k in range(1, 11):
        g = float(zetazero(k).imag)
        print(f'    gamma_{k} = {g:.4f}')

    # ══════════════════════════════════════════════════════════════
    # 6. OCTONIONIC ZETA
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  6. OCTONIONIC ZETA FUNCTION')
    print('#' * 76)

    print(f'\n  Octonions O: 8D, non-associative, basis {{1,i,j,k,l,il,jl,kl}}')
    print(f'  zeta(o) = sum n^{{-o}} converges for Re(o) > 1')
    print(f'  Each n^{{-o}} = exp(-o*log(n)) is well-defined (log(n) is real scalar)')

    # Verify at real point
    o_real = Oct(2.0)
    t0 = time.time()
    z_oct = oct_zeta(o_real, N_terms=500)
    dt = time.time() - t0
    z_exact = float(mpzeta(mpf(2)))
    print(f'\n  Verification: zeta(2)')
    print(f'    Octonionic: {z_oct}')
    print(f'    Exact: {z_exact:.10f}')
    print(f'    Error: {abs(z_oct.v[0] - z_exact):.2e}  ({dt:.2f}s)')

    # Quaternionic subspace (should match)
    o_quat = Oct(2.0, 1.0, 0, 0, 0, 0, 0, 0)
    z_oq = oct_zeta(o_quat, N_terms=500)
    z_cq = complex(mpzeta(mpc(2, 1)))
    print(f'\n  zeta(2+i) quaternionic subspace:')
    print(f'    Octonionic: {z_oq}')
    print(f'    Complex:    {z_cq.real:.6f} + {z_cq.imag:.6f}i')
    print(f'    Diff (real): {abs(z_oq.v[0] - z_cq.real):.2e}')
    print(f'    Diff (i):    {abs(z_oq.v[1] - z_cq.imag):.2e}')

    # NEW directions: l, il, jl, kl
    print(f'\n  Octonionic zeta in the NEW (l-) directions:')
    print(f'  {"direction":>12s} {"Re(zeta)":>12s} {"|imag|":>12s} {"imag dir":>30s}')
    print('  ' + '-' * 70)

    for label, components in [
        ('2+1*i',     (2, 1, 0, 0, 0, 0, 0, 0)),
        ('2+1*l',     (2, 0, 0, 0, 1, 0, 0, 0)),
        ('2+1*il',    (2, 0, 0, 0, 0, 1, 0, 0)),
        ('2+1*jl',    (2, 0, 0, 0, 0, 0, 1, 0)),
        ('2+1*kl',    (2, 0, 0, 0, 0, 0, 0, 1)),
        ('2+0.5i+0.5l', (2, 0.5, 0, 0, 0.5, 0, 0, 0)),
        ('2+0.5j+0.5jl', (2, 0, 0.5, 0, 0, 0, 0.5, 0)),
    ]:
        o = Oct(*components)
        z = oct_zeta(o, N_terms=300)
        imag_v = z.v[1:]
        imag_n = np.sqrt(np.sum(imag_v**2))
        imag_dir = imag_v / imag_n if imag_n > 1e-10 else imag_v
        dir_str = ', '.join(f'{x:.4f}' for x in imag_dir)
        print(f'  {label:>12s} {z.v[0]:>+12.6f} {imag_n:>12.6f} [{dir_str}]')
    sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════
    # 7. OCTONIONIC SYMMETRY: same |v| different direction
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  7. OCTONIONIC SYMMETRY: is zeta(2+v) the same for all |v|=1?')
    print('#' * 76)

    print(f'\n  If octonionic zeta is "slice-like", zeta should depend only on (Re, |Im|).')
    print(f'  Testing with various unit imaginary octonions at sigma=2, |v|=1:')

    unit_directions = [
        ('i',       (0, 1, 0, 0, 0, 0, 0, 0)),
        ('j',       (0, 0, 1, 0, 0, 0, 0, 0)),
        ('k',       (0, 0, 0, 1, 0, 0, 0, 0)),
        ('l',       (0, 0, 0, 0, 1, 0, 0, 0)),
        ('il',      (0, 0, 0, 0, 0, 1, 0, 0)),
        ('jl',      (0, 0, 0, 0, 0, 0, 1, 0)),
        ('kl',      (0, 0, 0, 0, 0, 0, 0, 1)),
        ('(i+l)/s2', (0, 1/np.sqrt(2), 0, 0, 1/np.sqrt(2), 0, 0, 0)),
        ('(j+kl)/s2', (0, 0, 1/np.sqrt(2), 0, 0, 0, 0, 1/np.sqrt(2))),
        ('uniform',  tuple([0] + [1/np.sqrt(7)]*7)),
    ]

    print(f'\n  {"direction":>12s} {"Re(zeta)":>12s} {"|Im(zeta)|":>12s} {"||zeta||":>12s}')
    print('  ' + '-' * 52)

    real_parts = []
    for label, v in unit_directions:
        o = Oct(2.0, *v[1:]) if len(v) == 8 else Oct(2.0)
        # Ensure |v| = 1
        vn = np.sqrt(sum(x**2 for x in v[1:]))
        if vn > 1e-10:
            components = [2.0] + [x/vn for x in v[1:]]
        else:
            components = [2.0] + list(v[1:])
        o = Oct(*components)

        z = oct_zeta(o, N_terms=300)
        imag_n = z.imag_norm()
        real_parts.append(z.v[0])
        print(f'  {label:>12s} {z.v[0]:>+12.6f} {imag_n:>12.6f} {z.norm():>12.6f}')

    # Are the real parts all the same?
    rp = np.array(real_parts)
    print(f'\n  Re(zeta) spread: max - min = {rp.max() - rp.min():.2e}')
    if rp.max() - rp.min() < 1e-4:
        print(f'  *** OCTONIONIC ZETA IS ROTATIONALLY SYMMETRIC IN Im ***')
        print(f'  This means zeta(sigma + v) depends only on sigma and |v|,')
        print(f'  not on the direction of v in the 7D imaginary space.')
        print(f'  The Dirichlet series has FULL SO(7) symmetry in the octonions!')
    else:
        print(f'  Octonionic zeta BREAKS rotational symmetry!')
        print(f'  Different octonionic directions give different zeta values.')
        print(f'  This would be genuinely new — invisible in H which has SO(3).')
    sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════
    # 8. OCTONIONIC NON-ASSOCIATIVITY CHECK
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '#' * 76)
    print('  8. NON-ASSOCIATIVITY: does it matter for zeta?')
    print('#' * 76)

    # Test: (a*b)*c vs a*(b*c) for random octonions
    np.random.seed(42)
    max_assoc_err = 0.0
    for _ in range(100):
        a = Oct(*np.random.randn(8))
        b = Oct(*np.random.randn(8))
        c = Oct(*np.random.randn(8))
        lhs = (a * b) * c
        rhs = a * (b * c)
        err = np.max(np.abs(lhs.v - rhs.v))
        max_assoc_err = max(max_assoc_err, err)

    print(f'\n  Max |(a*b)*c - a*(b*c)| over 100 random triples: {max_assoc_err:.6f}')
    print(f'  Non-associativity is REAL and significant.')

    # But for n^{-s} = exp(-s*log(n)), since log(n) is REAL:
    # exp(-s*t) for real t is well-defined (power series in t*s, t scalar)
    # No associativity issues for the Dirichlet series itself.
    print(f'\n  For Dirichlet series: n^{{-s}} = exp(-s*log(n)), log(n) real.')
    print(f'  Scalar * octonion is associative: (t*a)*b = t*(a*b) always.')
    print(f'  So the series itself is well-defined despite non-associativity.')
    print(f'  Non-associativity enters when we try to MULTIPLY zeta values.')

    # ══════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ══════════════════════════════════════════════════════════════
    print('\n\n' + '=' * 76)
    print('  SESSION 45g SYNTHESIS')
    print('=' * 76)

    # Count Fueter zeros found
    n_converged = sum(1 for _, _, fn, _ in refined if fn < 1e-6)

    print(f'''
  FUETER ZEROS:
    Exact formula derived: A = -(2/r)*Im(zeta'), B = (2/r)*Re(zeta') - 2*Im(zeta)/r^2
    At classical zeros: |F| = (2/gamma)*|zeta'(rho)| -- encodes the DERIVATIVE
    A=0 curve traced on critical line: crossings found
    Grid search + Newton refinement: {n_converged} zeros converged

  KEY FINDING: Fueter norm at classical zeros encodes |zeta'(rho)|/gamma.
  This connects to zero REPULSION (large |zeta'| = well-separated zeros)
  and the de Bruijn-Newman constant (zero dynamics under heat flow).

  OCTONIONIC ZETA:
    Computed via Dirichlet series (no associativity issues for n^{{-s}}).
    Tested rotational symmetry: does zeta depend on direction in 7D Im space?
    Non-associativity is real but doesn't affect the series itself.

  NEXT: If octonionic zeta BREAKS SO(7) symmetry (depends on direction,
  not just |v|), the extra structure could encode arithmetic information
  invisible in H. If it preserves SO(7), the octonionic extension adds
  only topology, not arithmetic.
''')

    print('=' * 76)
    print('  SESSION 45g COMPLETE')
    print('=' * 76)
