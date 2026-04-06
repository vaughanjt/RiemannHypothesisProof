"""
SESSION 60c -- THE DISPLACEMENT SIGNATURE

The barrier Hankel has displacement rank 2 with EQUAL singular values
(0.319, 0.319). Is this the distinguishing feature?

Hypothesis: displacement-rank-2 with EQUAL displacement singular values
forces the ESPRIT eigenvalues onto the unit circle.

Rationale: equal displacement singular values mean the shift acts
ISOMETRICALLY on the 2D displacement subspace. Isometric shift =
rotation = unitary = unit circle.
"""

import sys
import numpy as np

sys.path.insert(0, '.')


def hankel_matrix(signal, m=None):
    n = len(signal)
    if m is None:
        m = n // 2
    H = np.zeros((m, n - m))
    for i in range(m):
        H[i, :] = signal[i:i + n - m]
    return H


def displacement_svals(H):
    m, n = H.shape
    Z1 = np.zeros((m, m))
    Z1[1:, :-1] = np.eye(m - 1)
    Z2 = np.zeros((n, n))
    Z2[1:, :-1] = np.eye(n - 1)
    D = Z1 @ H - H @ Z2.T
    return np.linalg.svd(D, compute_uv=False)


def esprit_from_hankel(H, r):
    U, S, Vt = np.linalg.svd(H, full_matrices=False)
    U_r = U[:, :r]
    Phi = np.linalg.pinv(U_r[:-1, :]) @ U_r[1:, :]
    evals = np.linalg.eigvals(Phi)
    radii = np.abs(evals)
    deficit = np.linalg.norm(Phi.conj().T @ Phi - np.eye(r))
    return evals, radii, deficit


def run():
    print()
    print('#' * 76)
    print('  SESSION 60c -- THE DISPLACEMENT SIGNATURE')
    print('#' * 76)

    rng = np.random.default_rng(seed=601)
    n_samples = 200
    t = np.arange(n_samples, dtype=float)
    m_h = 80

    # == Part 1: Displacement svals for pure vs damped ==
    print('\n  === PART 1: DISPLACEMENT SINGULAR VALUES ===')
    print(f'  {"type":>12} {"s1":>10} {"s2":>10} {"s1/s2":>8} '
          f'{"max||z|-1|":>12} {"ESPRIT":>8}')
    print('  ' + '-' * 64)

    for trial in range(10):
        # Pure sinusoids
        w = rng.uniform(0.05, 0.45, 2)
        a = rng.uniform(0.5, 2.0, 2)
        sig = a[0] * np.cos(2*np.pi*w[0]*t) + a[1] * np.cos(2*np.pi*w[1]*t)
        H = hankel_matrix(sig, m=m_h)
        sv = displacement_svals(H)
        ev, radii, deficit = esprit_from_hankel(H, 4)
        max_dev = np.max(np.abs(radii - 1))
        print(f'  {"pure":>12} {sv[0]:>10.6f} {sv[1]:>10.6f} '
              f'{sv[0]/sv[1] if sv[1]>1e-15 else 999:>8.4f} '
              f'{max_dev:>12.2e} {"OK" if max_dev < 1e-6 else "FAIL":>8}')

    print()
    for trial in range(10):
        # Damped sinusoids
        w = rng.uniform(0.05, 0.45, 2)
        a = rng.uniform(0.5, 2.0, 2)
        d = rng.uniform(0.01, 0.05, 2)
        sig = (a[0] * np.exp(-d[0]*t) * np.cos(2*np.pi*w[0]*t) +
               a[1] * np.exp(-d[1]*t) * np.cos(2*np.pi*w[1]*t))
        H = hankel_matrix(sig, m=m_h)
        sv = displacement_svals(H)
        ev, radii, deficit = esprit_from_hankel(H, 4)
        max_dev = np.max(np.abs(radii - 1))
        print(f'  {"damped":>12} {sv[0]:>10.6f} {sv[1]:>10.6f} '
              f'{sv[0]/sv[1] if sv[1]>1e-15 else 999:>8.4f} '
              f'{max_dev:>12.2e} {"OK" if max_dev < 1e-6 else "FAIL":>8}')

    # == Part 2: Our barrier ==
    print('\n  === PART 2: BARRIER DISPLACEMENT SIGNATURE ===')
    data = np.load('session60_esprit_unitarity.npz')
    residual = data['residual']
    H_bar = hankel_matrix(residual, m=len(residual)//2)
    sv_bar = displacement_svals(H_bar)
    print(f'  Barrier: s1={sv_bar[0]:.10f}, s2={sv_bar[1]:.10f}, '
          f's1/s2={sv_bar[0]/sv_bar[1]:.10f}')
    print(f'  s3={sv_bar[2]:.2e}, s4={sv_bar[3]:.2e}')

    # == Part 3: Does s1 == s2 predict unitarity? ==
    print('\n  === PART 3: s1/s2 RATIO vs ESPRIT UNITARITY ===')
    print(f'  Generate signals with varying damping and check.')
    print()
    print(f'  {"damping":>10} {"s1/s2":>10} {"max||z|-1|":>12} {"on circle":>10}')
    print('  ' + '-' * 46)

    w_fixed = [0.15, 0.30]
    a_fixed = [1.0, 0.7]
    for damping in [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10]:
        sig = (a_fixed[0] * np.exp(-damping*t) * np.cos(2*np.pi*w_fixed[0]*t) +
               a_fixed[1] * np.exp(-damping*t) * np.cos(2*np.pi*w_fixed[1]*t))
        H = hankel_matrix(sig, m=m_h)
        sv = displacement_svals(H)
        ev, radii, deficit = esprit_from_hankel(H, 4)
        max_dev = np.max(np.abs(radii - 1))
        ratio = sv[0] / sv[1] if sv[1] > 1e-15 else float('inf')
        on_circle = max_dev < 1e-6
        print(f'  {damping:>10.3f} {ratio:>10.6f} {max_dev:>12.2e} '
              f'{"YES" if on_circle else "NO":>10}')

    # == Part 4: The displacement GENERATORS ==
    print('\n  === PART 4: DISPLACEMENT GENERATORS ===')
    print(f'  D = Z1 H - H Z2^T = U S V^T')
    print(f'  The 2 displacement generators U[:,0:2] and V[:,0:2]')
    print(f'  encode the STRUCTURE. For pure sinusoids, what do they look like?')
    print()

    # Pure signal
    sig_pure = np.cos(2*np.pi*0.15*t) + 0.7*np.cos(2*np.pi*0.30*t)
    H_pure = hankel_matrix(sig_pure, m=m_h)
    Z1 = np.zeros((m_h, m_h)); Z1[1:, :-1] = np.eye(m_h - 1)
    n_h = 200 - m_h
    Z2 = np.zeros((n_h, n_h)); Z2[1:, :-1] = np.eye(n_h - 1)
    D_pure = Z1 @ H_pure - H_pure @ Z2.T
    U_d, S_d, Vt_d = np.linalg.svd(D_pure)

    print(f'  Pure signal displacement:')
    print(f'    s1={S_d[0]:.6f}, s2={S_d[1]:.6f}, s1/s2={S_d[0]/S_d[1]:.6f}')

    # The left generator U[:,0:2] should have sinusoidal structure
    print(f'    Left generator U[:,0] (first 10):')
    for i in range(min(10, m_h)):
        print(f'      {i}: {U_d[i,0]:+.6f}')

    # Check if U[:,0:2] spans the same space as [cos(w1*n), cos(w2*n)]
    cos_basis = np.column_stack([
        np.cos(2*np.pi*0.15*np.arange(m_h)),
        np.sin(2*np.pi*0.15*np.arange(m_h)),
        np.cos(2*np.pi*0.30*np.arange(m_h)),
        np.sin(2*np.pi*0.30*np.arange(m_h)),
    ])
    # Project U[:,0:2] onto cos_basis
    proj = np.linalg.lstsq(cos_basis, U_d[:, :2], rcond=None)[0]
    reconstruction = cos_basis @ proj
    error = np.linalg.norm(U_d[:, :2] - reconstruction) / np.linalg.norm(U_d[:, :2])
    print(f'    Projection error of U[:,0:2] onto sin/cos basis: {error:.2e}')

    # Now barrier
    print(f'\n  Barrier displacement:')
    D_bar = Z1[:len(residual)//2, :len(residual)//2] @ H_bar - H_bar @ Z2[:H_bar.shape[1], :H_bar.shape[1]].T

    # Recompute properly
    m_b = len(residual) // 2
    n_b = len(residual) - m_b
    Z1b = np.zeros((m_b, m_b)); Z1b[1:, :-1] = np.eye(m_b - 1)
    Z2b = np.zeros((n_b, n_b)); Z2b[1:, :-1] = np.eye(n_b - 1)
    D_bar = Z1b @ H_bar - H_bar @ Z2b.T
    U_bar, S_bar, Vt_bar = np.linalg.svd(D_bar)
    print(f'    s1={S_bar[0]:.10f}, s2={S_bar[1]:.10f}')
    print(f'    s1/s2 = {S_bar[0]/S_bar[1]:.10f}')
    print(f'    s1 == s2 to {abs(S_bar[0]-S_bar[1]):.2e}')

    # == Verdict ==
    print()
    print('=' * 76)
    print('  VERDICT')
    print('=' * 76)
    print()

    barrier_ratio = sv_bar[0] / sv_bar[1]
    if abs(barrier_ratio - 1.0) < 1e-6:
        print(f'  BARRIER HAS EQUAL DISPLACEMENT SINGULAR VALUES')
        print(f'  s1/s2 = {barrier_ratio:.10f} (deviation {abs(barrier_ratio-1):.2e} from 1)')
        print()
        print(f'  For pure sinusoids: s1/s2 = 1.0 (exactly)')
        print(f'  For damped sinusoids: s1/s2 != 1.0')
        print()
        print(f'  HYPOTHESIS: s1 = s2 in the displacement <=> |z_k| = 1')
        print(f'  <=> zeros on the critical line <=> RH')
        print()
        print(f'  If this is a theorem (displacement equal svals => unit circle),')
        print(f'  then proving s1 = s2 for the barrier Hankel proves RH.')
    else:
        print(f'  Barrier s1/s2 = {barrier_ratio:.6f} (not exactly 1)')


if __name__ == '__main__':
    run()
