"""
SESSION 48e -- CONJUGATE POISSON KERNEL DEEP DIVE

Discovery from 48d: w(n) = n/(L^2 + 16*pi^2*n^2) IS the conjugate Poisson
kernel at height y = L/(4*pi) in the upper half-plane.

Q_y(n) = n / (y^2 + n^2)  [conjugate Poisson, the Hilbert transform]
P_y(n) = y / (y^2 + n^2)  [regular Poisson kernel]

Both are Fourier coefficients:
  P_y = Fourier[e^{-2*pi*y*|k|}]  at frequency k
  Q_y = Fourier[-i*sign(k)*e^{-2*pi*y*|k|}]  at frequency k

So <w_L, Q_W w_L> computes Q_W evaluated on the HILBERT TRANSFORM
of the delta function at height y = L/(4*pi).

QUESTIONS:
1. What is Q_W as an operator when w_L = Q_y? Is there a natural
   operator T such that Q_W = T acting on conjugate Poisson family?
2. Is B(L) = F(y) for a simple function F on (0, inf)?
3. Does B(L) have a nice form as y -> 0 or y -> inf?
4. What is the automorphic interpretation: is w_L a Whittaker function?
"""

import numpy as np
import sys
import time

sys.path.insert(0, '.')
from connes_crossterm import build_all


def conjugate_poisson_kernel(y, n_values):
    """Conjugate Poisson kernel at height y: Q_y(n) = n/(y^2 + n^2)."""
    return n_values / (y**2 + n_values**2)


def poisson_kernel(y, n_values):
    """Regular Poisson kernel at height y: P_y(n) = y/(y^2 + n^2)."""
    return y / (y**2 + n_values**2)


def barrier_at_y(y, N=None, use_regular_poisson=False):
    """
    Compute <kernel_y, Q_W kernel_y> as a function of height y.

    y = L/(4*pi) where L = log(lam^2).
    So lam^2 = exp(4*pi*y).
    """
    L_f = 4 * np.pi * y
    lam_sq = np.exp(L_f)
    if N is None:
        N = max(15, round(6 * L_f))

    W02, M, QW = build_all(lam_sq, N)

    ns = np.arange(-N, N + 1, dtype=float)

    # Scale factor: original w(n) = n/(L^2 + 16*pi^2*n^2) = (1/16pi^2) * Q_y(n)
    # where y = L/(4*pi). So w_L is a scaled conjugate Poisson kernel.
    if use_regular_poisson:
        kernel = poisson_kernel(y, ns)
        # Poisson kernel has P_y(0) = 1/y, but our w has w[0] = 0
        # For symmetric comparison, keep n=0 term for Poisson
        kernel_type = 'Poisson'
    else:
        kernel = conjugate_poisson_kernel(y, ns)
        kernel[N] = 0.0  # conjugate Poisson has 0 at n=0 (it's odd)
        kernel_type = 'Conjugate Poisson'

    if np.linalg.norm(kernel) > 0:
        kernel_hat = kernel / np.linalg.norm(kernel)
    else:
        kernel_hat = kernel

    w02 = float(kernel_hat @ W02 @ kernel_hat)
    m = float(kernel_hat @ M @ kernel_hat)
    b = float(kernel_hat @ QW @ kernel_hat)

    return {
        'y': y, 'L': L_f, 'lam_sq': lam_sq, 'N': N,
        'W02': w02, 'M': m, 'B': b, 'kernel_type': kernel_type,
    }


def compare_kernels(y_values):
    """Compare barrier computed with Poisson vs conjugate Poisson kernels."""
    print('\n  -- KERNEL COMPARISON --')
    print(f'\n  {"y":>8} {"L":>8} {"lam^2":>10} {"B (conj Poisson)":>18} {"B (Poisson)":>14}')
    print('  ' + '-' * 64)

    for y in y_values:
        t0 = time.time()
        r_conj = barrier_at_y(y, use_regular_poisson=False)
        r_reg = barrier_at_y(y, use_regular_poisson=True)
        dt = time.time() - t0
        print(f'  {y:8.4f} {r_conj["L"]:8.3f} {r_conj["lam_sq"]:10.1f} '
              f'{r_conj["B"]:18.6f} {r_reg["B"]:14.6f}  ({dt:.0f}s)')


def fine_grained_y_scan(y_min, y_max, n_points):
    """
    Scan B(L) as a function of y = L/(4*pi).

    This reveals the shape of B as a function of height in the upper half-plane.
    """
    print(f'\n  -- Y-SCAN: y in [{y_min}, {y_max}], {n_points} points --')
    print(f'\n  {"y":>8} {"L":>8} {"lam^2":>12} {"B(y)":>12} {"|W02|":>10} {"|M|":>10}')
    print('  ' + '-' * 66)

    y_values = np.linspace(y_min, y_max, n_points)
    results = []

    for y in y_values:
        t0 = time.time()
        r = barrier_at_y(y)
        results.append(r)
        print(f'  {y:8.4f} {r["L"]:8.3f} {r["lam_sq"]:12.2f} '
              f'{r["B"]:12.6f} {abs(r["W02"]):10.4f} {abs(r["M"]):10.4f}  ({time.time()-t0:.0f}s)')

    return results


def check_monotonicity(results):
    """Check if B(y) has any monotonicity or special structure."""
    ys = [r['y'] for r in results]
    Bs = [r['B'] for r in results]

    print('\n  -- STRUCTURE ANALYSIS --')

    # First differences
    print(f'\n  {"y":>8} {"B(y)":>10} {"dB/dy":>12} {"d^2B":>12}')
    print('  ' + '-' * 44)

    for i, (y, b) in enumerate(zip(ys, Bs)):
        if 0 < i < len(Bs) - 1:
            dB = (Bs[i+1] - Bs[i-1]) / (ys[i+1] - ys[i-1])
            d2B = (Bs[i+1] - 2*Bs[i] + Bs[i-1]) / ((ys[i+1] - ys[i]) * (ys[i] - ys[i-1]))
            print(f'  {y:8.4f} {b:10.6f} {dB:12.6f} {d2B:12.6f}')
        else:
            print(f'  {y:8.4f} {b:10.6f}')

    # Check for extrema
    min_idx = np.argmin(Bs)
    max_idx = np.argmax(Bs)
    print(f'\n  Minimum B: {Bs[min_idx]:.6f} at y={ys[min_idx]:.4f}')
    print(f'  Maximum B: {Bs[max_idx]:.6f} at y={ys[max_idx]:.4f}')

    # Is B(y) monotone?
    diffs = np.diff(Bs)
    sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
    print(f'  Sign changes in dB/dy: {sign_changes}')
    if sign_changes == 0:
        print(f'  B(y) is MONOTONIC')
    else:
        print(f'  B(y) has {sign_changes} extrema (non-monotonic)')


def automorphic_interpretation_test(y_values):
    """
    Test specific y values related to automorphic structure:
    - y = 1/2, 1, 2 (boundary of standard fundamental domain)
    - y related to Heegner numbers (CM points project to these heights)
    - y = sqrt(3)/2 (corner of fundamental domain)
    """
    print('\n\n  -- AUTOMORPHIC Y VALUES --')
    print(f'\n  Testing B(y) at heights related to SL(2,Z) fundamental domain:')

    special_y = [
        (0.289, 'y=sqrt(1/12) ~ corner'),
        (0.433, 'y=sqrt(3)/4 ~ corner'),
        (0.500, 'y=1/2 (boundary)'),
        (0.866, 'y=sqrt(3)/2 (corner)'),
        (1.000, 'y=1 (above unit circle)'),
    ]

    print(f'\n  {"y":>8} {"description":<35} {"B(y)":>12} {"lam^2":>10}')
    print('  ' + '-' * 68)
    for y, desc in special_y:
        try:
            r = barrier_at_y(y)
            print(f'  {y:8.4f} {desc:<35} {r["B"]:12.6f} {r["lam_sq"]:10.1f}')
        except Exception as e:
            print(f'  {y:8.4f} {desc:<35} ERROR: {e}')


if __name__ == '__main__':
    print()
    print('#' * 72)
    print('  SESSION 48e -- CONJUGATE POISSON KERNEL DEEP DIVE')
    print('#' * 72)

    print("""
  The test vector w_L = conjugate Poisson kernel at height y = L/(4*pi).
  B(L) = <w_L, Q_W w_L> computes Q_W on the harmonic conjugate.

  We scan B as a function of y to understand its shape and look for
  automorphic structure.
  """)

    # Quick coverage of wide y range
    y_small = [0.20, 0.25, 0.30, 0.40, 0.50]

    print('=' * 72)
    print('  A. Y-SCAN (small to large)')
    print('=' * 72)

    results = fine_grained_y_scan(0.20, 0.50, 5)

    check_monotonicity(results)

    # Kernel comparison (Poisson vs conjugate Poisson)
    print('\n\n' + '=' * 72)
    print('  B. POISSON vs CONJUGATE POISSON')
    print('=' * 72)

    compare_kernels([0.25, 0.40])

    # Automorphic special values
    print('\n\n' + '=' * 72)
    print('  C. AUTOMORPHIC HEIGHTS')
    print('=' * 72)

    automorphic_interpretation_test([0.433, 0.5, 0.866, 1.0])

    print('\n\n' + '=' * 72)
    print('  SUMMARY')
    print('=' * 72)
    print("""
  Key questions answered:
  1. Is B(y) monotonic? [see structure analysis]
  2. Does B(y) differ dramatically with Poisson vs conjugate Poisson? [kernel comparison]
  3. Are there special heights where B has special values? [automorphic test]

  Next: if B(y) has clean structure (monotonic, simple functional form,
  or specific values at special heights), that's a lead. If it's noisy,
  the conjugate Poisson observation was superficial.
  """)
    print()
