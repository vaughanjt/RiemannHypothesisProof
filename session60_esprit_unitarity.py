"""
SESSION 60 -- ESPRIT UNITARITY FROM STRUCTURE

The pipeline: Primes -> Barrier B(L) -> ESPRIT -> Zeros
RH <=> ESPRIT rotation matrix is unitary <=> all frequencies real

Session 42 found the ESPRIT matrix was "nearly unitary" (deviation 0.047).
Session 56-57 found M has Lorentzian signature + parity decomposition.
Session 59b confirmed M has exact Cauchy off-diagonal structure.

ESPRIT (Roy & Kailath 1989) was designed for exactly this matrix type.
For a signal x(t) = sum A_k exp(i*w_k*t):
  - Build Hankel matrix H from signal samples
  - SVD: H = U S V^T, keep top-r singular vectors
  - Extract rotation: Phi = U_up^+ U_down (shift-invariance)
  - Eigenvalues of Phi = exp(i*w_k*dt) = frequencies on unit circle

If signal is purely sinusoidal (no damping): Phi is unitary.
If signal has damping: |eigenvalues of Phi| != 1.

Plan:
  1. Compute B(L) at dense L grid using build_all_fast
  2. Build ESPRIT Hankel matrix
  3. Extract rotation matrix Phi
  4. Measure unitarity: ||Phi^H Phi - I||
  5. Compare eigenvalues of Phi to unit circle
  6. Track unitarity vs lambda (does it improve with more primes?)
  7. Test: does the Cauchy/Lorentzian structure predict unitarity?
"""

import sys
import time

import numpy as np

sys.path.insert(0, '.')
from session49c_weil_residual import build_all_fast


def barrier_at_L(L_val):
    """Compute B(L) = min eigenvalue of Q_W (the barrier)."""
    lam_sq = max(2, int(round(np.exp(L_val))))
    N = max(15, round(6 * L_val))
    L_eff = np.log(lam_sq)
    W02, M, QW = build_all_fast(lam_sq, N)
    ns = np.arange(-N, N + 1, dtype=float)

    # Odd direction barrier (w_hat)
    w = ns / (L_eff ** 2 + (4 * np.pi) ** 2 * ns ** 2)
    w[N] = 0.0
    w_hat = w / np.linalg.norm(w)
    b_odd = float(w_hat @ QW @ w_hat)

    # Even direction barrier (u_hat)
    u = 1.0 / (L_eff ** 2 + (4 * np.pi) ** 2 * ns ** 2)
    u_hat = u / np.linalg.norm(u)
    b_even = float(u_hat @ QW @ u_hat)

    # Full barrier (min eigenvalue)
    evals = np.linalg.eigvalsh(QW)
    b_full = float(evals[0])

    return b_odd, b_even, b_full


def esprit(signal, r, dt=1.0):
    """
    ESPRIT algorithm for frequency extraction.

    signal: 1D array of samples
    r: number of sinusoidal components to extract
    dt: sample spacing

    Returns: (frequencies, Phi, unitarity_deficit)
    """
    n = len(signal)
    m = n // 2  # Hankel matrix size

    # Build Hankel matrix
    H = np.zeros((m, n - m))
    for i in range(m):
        H[i, :] = signal[i:i + n - m]

    # SVD
    U, S, Vt = np.linalg.svd(H, full_matrices=False)

    # Keep top r components
    U_r = U[:, :r]

    # Shift-invariance: U_up and U_down
    U_up = U_r[:-1, :]    # rows 0..m-2
    U_down = U_r[1:, :]   # rows 1..m-1

    # Rotation matrix: Phi = pinv(U_up) @ U_down
    Phi = np.linalg.pinv(U_up) @ U_down

    # Eigenvalues
    evals = np.linalg.eigvals(Phi)

    # Frequencies from eigenvalues: z = exp(i*w*dt)
    freqs = np.angle(evals) / dt

    # Unitarity deficit
    PhiH = Phi.conj().T
    deficit = np.linalg.norm(PhiH @ Phi - np.eye(r))

    # How close eigenvalues are to unit circle
    radii = np.abs(evals)

    return freqs, evals, Phi, deficit, radii, S[:r+5]


def run():
    print()
    print('#' * 76)
    print('  SESSION 60 -- ESPRIT UNITARITY FROM STRUCTURE')
    print('#' * 76)

    # == Part 1: Compute barrier signal ==
    print('\n  === PART 1: BARRIER SIGNAL ===')

    # Dense L grid
    n_pts = 300
    L_min, L_max = 1.5, 6.5
    L_vals = np.linspace(L_min, L_max, n_pts)
    dL = L_vals[1] - L_vals[0]

    print(f'  Computing B_odd(L) at {n_pts} points, L in [{L_min}, {L_max}]')
    print(f'  dL = {dL:.4f}')

    t0 = time.time()
    signal_odd = np.zeros(n_pts)
    signal_even = np.zeros(n_pts)
    for i, L in enumerate(L_vals):
        b_odd, b_even, b_full = barrier_at_L(float(L))
        signal_odd[i] = b_odd
        signal_even[i] = b_even
        if i % 50 == 0:
            print(f'    L={L:.2f}: B_odd={b_odd:.6f}, B_even={b_even:.6f}',
                  flush=True)

    dt = time.time() - t0
    print(f'  Signal computed in {dt:.1f}s')

    # Detrend: remove smooth polynomial trend to isolate oscillations
    # Fit and subtract cubic trend
    coeffs = np.polyfit(L_vals, signal_odd, 3)
    trend = np.polyval(coeffs, L_vals)
    residual = signal_odd - trend

    print(f'\n  Signal stats:')
    print(f'    B_odd range: [{signal_odd.min():.6f}, {signal_odd.max():.6f}]')
    print(f'    Trend range: [{trend.min():.6f}, {trend.max():.6f}]')
    print(f'    Residual range: [{residual.min():.6f}, {residual.max():.6f}]')
    print(f'    Residual RMS: {np.sqrt((residual**2).mean()):.6f}')
    sys.stdout.flush()

    # == Part 2: ESPRIT at multiple model orders ==
    print('\n  === PART 2: ESPRIT FREQUENCY EXTRACTION ===')

    # Known zeta zeros for comparison
    known_zeros = [14.1347, 21.0220, 25.0109, 30.4249, 32.9351,
                   37.5862, 40.9187, 43.3271, 48.0052, 49.7738]

    for r in [5, 8, 10, 15, 20]:
        freqs, evals, Phi, deficit, radii, svals = esprit(
            residual, r, dt=dL)

        # Sort by frequency
        order = np.argsort(np.abs(freqs))
        freqs_sorted = freqs[order]
        radii_sorted = radii[order]
        evals_sorted = evals[order]

        # Match to known zeros (positive frequencies only)
        pos_freqs = sorted([f for f in freqs_sorted if f > 5])

        print(f'\n  r = {r}:')
        print(f'    Unitarity deficit ||Phi^H Phi - I||: {deficit:.6f}')
        print(f'    Eigenvalue radii: min={radii.min():.6f}, '
              f'max={radii.max():.6f}, mean={radii.mean():.6f}')
        print(f'    Singular values: {svals[:5]}')
        print(f'    Positive frequencies > 5:')
        for f in pos_freqs[:8]:
            # Find closest known zero
            dists = [abs(f - z) for z in known_zeros]
            best = known_zeros[np.argmin(dists)]
            err = abs(f - best)
            print(f'      {f:>8.3f}  (nearest zero: {best:.4f}, err: {err:.3f})')
    sys.stdout.flush()

    # == Part 3: Unitarity vs signal length ==
    print('\n  === PART 3: UNITARITY vs SIGNAL LENGTH ===')
    print(f'  Fix r=10, vary the L range (more data = more primes).')
    print()

    r = 10
    print(f'  {"L_max":>6} {"n_pts":>6} {"deficit":>10} {"mean |z|":>10} '
          f'{"max ||z|-1|":>12}')
    print('  ' + '-' * 50)

    for L_max_test in [4.0, 5.0, 6.0, 6.5]:
        n_test = int((L_max_test - L_min) / dL) + 1
        if n_test < 2 * r + 5:
            continue
        sig_test = residual[:n_test]
        if len(sig_test) < 2 * r + 5:
            continue
        f, ev, Ph, defi, rad, sv = esprit(sig_test, r, dt=dL)
        max_dev = np.max(np.abs(rad - 1.0))
        print(f'  {L_max_test:>6.1f} {n_test:>6d} {defi:>10.6f} '
              f'{rad.mean():>10.6f} {max_dev:>12.6f}')
    sys.stdout.flush()

    # == Part 4: ESPRIT on even barrier ==
    print('\n  === PART 4: ESPRIT ON EVEN BARRIER ===')
    coeffs_even = np.polyfit(L_vals, signal_even, 3)
    trend_even = np.polyval(coeffs_even, L_vals)
    residual_even = signal_even - trend_even

    r = 10
    freqs_e, evals_e, Phi_e, deficit_e, radii_e, svals_e = esprit(
        residual_even, r, dt=dL)

    print(f'  r = {r}:')
    print(f'    Unitarity deficit: {deficit_e:.6f}')
    print(f'    Radii: min={radii_e.min():.6f}, max={radii_e.max():.6f}')
    pos_freqs_e = sorted([f for f in freqs_e if f > 5])
    print(f'    Positive frequencies > 5:')
    for f in pos_freqs_e[:8]:
        dists = [abs(f - z) for z in known_zeros]
        best = known_zeros[np.argmin(dists)]
        err = abs(f - best)
        print(f'      {f:>8.3f}  (nearest zero: {best:.4f}, err: {err:.3f})')
    sys.stdout.flush()

    # == Part 5: Structure test — does Cauchy structure predict unitarity? ==
    print('\n  === PART 5: STRUCTURAL ANALYSIS OF ESPRIT MATRIX ===')
    print(f'  Analyzing the ESPRIT Phi matrix at r=10 on odd barrier.')
    print()

    r = 10
    _, evals_phi, Phi_phi, deficit_phi, radii_phi, _ = esprit(
        residual, r, dt=dL)

    # Is Phi normal? (Phi^H Phi = Phi Phi^H for normal matrices)
    PhiH = Phi_phi.conj().T
    comm = np.linalg.norm(PhiH @ Phi_phi - Phi_phi @ PhiH)
    print(f'  ||Phi^H Phi - Phi Phi^H|| (normality): {comm:.6e}')
    print(f'  ||Phi^H Phi - I|| (unitarity):          {deficit_phi:.6e}')

    # Eigenvalue analysis
    print(f'\n  Eigenvalues of Phi:')
    print(f'  {"idx":>4} {"Re":>10} {"Im":>10} {"|z|":>10} {"||z|-1|":>10} '
          f'{"freq":>10}')
    print('  ' + '-' * 56)
    for i, ev in enumerate(sorted(evals_phi, key=lambda z: np.angle(z))):
        freq = np.angle(ev) / dL
        print(f'  {i:>4d} {ev.real:>+10.6f} {ev.imag:>+10.6f} '
              f'{abs(ev):>10.6f} {abs(abs(ev)-1):>10.6f} {freq:>+10.3f}')

    # == Verdict ==
    print()
    print('=' * 76)
    print('  VERDICT')
    print('=' * 76)
    print()

    best_deficit = deficit_phi
    best_max_dev = np.max(np.abs(radii_phi - 1.0))

    if best_max_dev < 0.01:
        print(f'  ESPRIT eigenvalues are ON the unit circle to {best_max_dev:.1e}.')
        print(f'  All extracted frequencies are REAL.')
        print(f'  Consistent with RH (no off-line zeros detected).')
        print()
        print(f'  To turn this into a proof:')
        print(f'    1. Prove the Hankel matrix from M has displacement rank 2')
        print(f'    2. Prove displacement-rank-2 Hankel -> unitary ESPRIT Phi')
        print(f'    3. Unitary Phi -> real frequencies -> zeros on critical line')
    elif best_max_dev < 0.1:
        print(f'  ESPRIT eigenvalues are NEAR the unit circle (max dev {best_max_dev:.4f}).')
        print(f'  Deviation could be: truncation, finite signal, or off-line zeros.')
        print(f'  At this precision, consistent with RH but not conclusive.')
    else:
        print(f'  ESPRIT eigenvalues deviate significantly from unit circle.')
        print(f'  Max deviation: {best_max_dev:.4f}')

    np.savez('session60_esprit_unitarity.npz',
             L_vals=L_vals, signal_odd=signal_odd, signal_even=signal_even,
             residual=residual)
    print(f'\n  Data saved to session60_esprit_unitarity.npz')


if __name__ == '__main__':
    run()
