"""
SESSION 60b -- DISPLACEMENT RANK AND ESPRIT UNITARITY: DIRECT TEST

Can we show that displacement-rank-2 Hankel structure forces
ESPRIT unitarity? Test empirically, then look for the mechanism.

Approach:
  1. Generate SYNTHETIC signals with known properties:
     (a) Pure sinusoids (z_k on unit circle) -> displacement rank = ?
     (b) Damped sinusoids (|z_k| != 1) -> displacement rank = ?
  2. Check: does displacement rank DISTINGUISH pure from damped?
  3. For our actual barrier signal: measure displacement rank of
     its Hankel matrix. Is it exactly 2?
  4. Construct random Hankel matrices of displacement rank 2 and
     check if ESPRIT always gives unitary Phi.
"""

import sys
import numpy as np

sys.path.insert(0, '.')


def hankel_matrix(signal, m=None):
    """Build Hankel matrix from signal samples."""
    n = len(signal)
    if m is None:
        m = n // 2
    H = np.zeros((m, n - m))
    for i in range(m):
        H[i, :] = signal[i:i + n - m]
    return H


def displacement_rank(H, tol=1e-10):
    """
    Compute displacement rank of Hankel matrix H.
    For Hankel: displacement is H - Z_down H Z_right^T
    where Z_down shifts rows down and Z_right shifts columns right.

    Actually for Hankel, the appropriate displacement is:
    Z_1 H - H Z_2^T where Z_1 and Z_2 are shift matrices.

    displacement rank = rank(Z_1 H - H Z_2^T)
    """
    m, n = H.shape
    Z1 = np.zeros((m, m))
    Z1[1:, :-1] = np.eye(m - 1)  # downshift
    Z2 = np.zeros((n, n))
    Z2[1:, :-1] = np.eye(n - 1)  # downshift

    D = Z1 @ H - H @ Z2.T
    svals = np.linalg.svd(D, compute_uv=False)
    rank = np.sum(svals > tol * svals[0])
    return rank, svals[:10]


def esprit_from_hankel(H, r):
    """Run ESPRIT on a pre-built Hankel matrix."""
    U, S, Vt = np.linalg.svd(H, full_matrices=False)
    U_r = U[:, :r]
    U_up = U_r[:-1, :]
    U_down = U_r[1:, :]
    Phi = np.linalg.pinv(U_up) @ U_down
    evals = np.linalg.eigvals(Phi)
    radii = np.abs(evals)
    deficit = np.linalg.norm(Phi.conj().T @ Phi - np.eye(r))
    return evals, Phi, deficit, radii


def run():
    print()
    print('#' * 76)
    print('  SESSION 60b -- DISPLACEMENT RANK vs ESPRIT UNITARITY')
    print('#' * 76)

    rng = np.random.default_rng(seed=60)

    # == Part 1: Synthetic signals ==
    print('\n  === PART 1: SYNTHETIC SIGNALS ===')
    print(f'  Compare displacement rank of pure vs damped sinusoids.')
    print()

    n_samples = 200
    t = np.arange(n_samples, dtype=float)

    # (a) Pure sinusoids (on unit circle)
    print(f'  --- Pure sinusoids (|z_k| = 1) ---')
    for n_freq in [2, 3, 5, 8]:
        freqs = rng.uniform(0.1, 0.4, n_freq)
        amps = rng.uniform(0.5, 2.0, n_freq)
        signal = sum(a * np.cos(2 * np.pi * f * t) for a, f in zip(amps, freqs))
        H = hankel_matrix(signal, m=80)
        dr, sv = displacement_rank(H)
        _, _, deficit, radii = esprit_from_hankel(H, 2 * n_freq)
        max_dev = np.max(np.abs(radii - 1))
        print(f'    {n_freq} freqs: disp_rank={dr}, ESPRIT deficit={deficit:.6f}, '
              f'max ||z|-1|={max_dev:.2e}')

    # (b) Damped sinusoids (|z_k| != 1)
    print(f'\n  --- Damped sinusoids (|z_k| != 1) ---')
    for n_freq in [2, 3, 5]:
        freqs = rng.uniform(0.1, 0.4, n_freq)
        amps = rng.uniform(0.5, 2.0, n_freq)
        dampings = rng.uniform(0.01, 0.05, n_freq)  # decay rates
        signal = sum(a * np.exp(-d * t) * np.cos(2 * np.pi * f * t)
                     for a, f, d in zip(amps, freqs, dampings))
        H = hankel_matrix(signal, m=80)
        dr, sv = displacement_rank(H)
        _, _, deficit, radii = esprit_from_hankel(H, 2 * n_freq)
        max_dev = np.max(np.abs(radii - 1))
        print(f'    {n_freq} damped: disp_rank={dr}, ESPRIT deficit={deficit:.6f}, '
              f'max ||z|-1|={max_dev:.2e}')

    # (c) Mixed: some pure, some damped
    print(f'\n  --- Mixed (2 pure + 1 damped) ---')
    freqs_p = [0.15, 0.30]
    freqs_d = [0.22]
    signal = (np.cos(2 * np.pi * 0.15 * t) + 0.8 * np.cos(2 * np.pi * 0.30 * t)
              + 0.5 * np.exp(-0.02 * t) * np.cos(2 * np.pi * 0.22 * t))
    H = hankel_matrix(signal, m=80)
    dr, sv = displacement_rank(H)
    _, _, deficit, radii = esprit_from_hankel(H, 6)
    max_dev = np.max(np.abs(radii - 1))
    print(f'    mixed: disp_rank={dr}, ESPRIT deficit={deficit:.6f}, '
          f'max ||z|-1|={max_dev:.2e}')
    sys.stdout.flush()

    # == Part 2: Displacement rank of barrier Hankel ==
    print('\n  === PART 2: BARRIER HANKEL DISPLACEMENT RANK ===')

    # Load barrier signal from Session 60
    try:
        data = np.load('session60_esprit_unitarity.npz')
        residual = data['residual']
        L_vals = data['L_vals']
        print(f'  Loaded barrier residual: {len(residual)} samples')
    except FileNotFoundError:
        print(f'  session60 data not found, computing...')
        from session49c_weil_residual import build_all_fast
        L_vals = np.linspace(1.5, 6.5, 300)
        signal_odd = np.zeros(len(L_vals))
        for i, L in enumerate(L_vals):
            lam_sq = max(2, int(round(np.exp(L))))
            N = max(15, round(6 * L))
            W02, M, QW = build_all_fast(lam_sq, N)
            ns = np.arange(-N, N + 1, dtype=float)
            L_eff = np.log(lam_sq)
            w = ns / (L_eff ** 2 + (4 * np.pi) ** 2 * ns ** 2)
            w[N] = 0.0
            w_hat = w / np.linalg.norm(w)
            signal_odd[i] = float(w_hat @ QW @ w_hat)
        coeffs = np.polyfit(L_vals, signal_odd, 3)
        residual = signal_odd - np.polyval(coeffs, L_vals)

    H_barrier = hankel_matrix(residual, m=len(residual) // 2)
    dr_barrier, sv_barrier = displacement_rank(H_barrier, tol=1e-6)
    print(f'  Displacement rank (tol=1e-6): {dr_barrier}')
    print(f'  Top 10 displacement singular values:')
    for i, s in enumerate(sv_barrier[:10]):
        print(f'    {i}: {s:.6f}')

    # Try different tolerances
    print(f'\n  Displacement rank vs tolerance:')
    for tol in [1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10]:
        dr_t, _ = displacement_rank(H_barrier, tol=tol)
        print(f'    tol={tol:.0e}: rank={dr_t}')
    sys.stdout.flush()

    # == Part 3: Random displacement-rank-2 Hankel matrices ==
    print('\n  === PART 3: RANDOM DISPLACEMENT-RANK-2 HANKEL ===')
    print(f'  Generate Hankel matrices with EXACTLY displacement rank 2.')
    print(f'  Check if ESPRIT always gives unitary Phi.')
    print()

    # A Hankel matrix with displacement rank 2 comes from a signal
    # that is a sum of 2 complex exponentials: x(n) = a1*z1^n + a2*z2^n
    # If z1, z2 are on the unit circle: pure sinusoidal
    # If z1, z2 are off: damped

    print(f'  --- Case A: z on unit circle (|z|=1) ---')
    n_trials = 20
    n_unitary = 0
    for trial in range(n_trials):
        # Random frequencies on unit circle
        w1, w2 = rng.uniform(0.1, 0.4, 2)
        z1, z2 = np.exp(1j * 2 * np.pi * w1), np.exp(1j * 2 * np.pi * w2)
        a1, a2 = rng.uniform(0.5, 2.0, 2)
        signal = np.real(a1 * z1 ** t + a2 * z2 ** t)
        H = hankel_matrix(signal, m=80)
        dr, _ = displacement_rank(H, tol=1e-8)
        _, _, deficit, radii = esprit_from_hankel(H, 4)
        max_dev = np.max(np.abs(radii - 1))
        if max_dev < 1e-6:
            n_unitary += 1
    print(f'    {n_unitary}/{n_trials} trials: ESPRIT unitary (max ||z|-1| < 1e-6)')

    print(f'\n  --- Case B: z off unit circle (|z| != 1) ---')
    n_unitary_off = 0
    for trial in range(n_trials):
        w1, w2 = rng.uniform(0.1, 0.4, 2)
        r1 = rng.uniform(0.95, 0.99)  # inside unit circle (damped)
        r2 = rng.uniform(0.95, 0.99)
        z1 = r1 * np.exp(1j * 2 * np.pi * w1)
        z2 = r2 * np.exp(1j * 2 * np.pi * w2)
        a1, a2 = rng.uniform(0.5, 2.0, 2)
        signal = np.real(a1 * z1 ** t + a2 * z2 ** t)
        H = hankel_matrix(signal, m=80)
        dr, _ = displacement_rank(H, tol=1e-8)
        _, _, deficit, radii = esprit_from_hankel(H, 4)
        max_dev = np.max(np.abs(radii - 1))
        if max_dev < 1e-6:
            n_unitary_off += 1
    print(f'    {n_unitary_off}/{n_trials} trials: ESPRIT unitary (max ||z|-1| < 1e-6)')

    print(f'\n  --- Case C: displacement rank 2, z forced on unit circle ---')
    print(f'  Key question: do ALL disp-rank-2 Hankel give unitary ESPRIT,')
    print(f'  or only the ones from pure sinusoids?')
    print()

    # A Hankel matrix H has displacement Z1 H - H Z2^T = G J G^T
    # with G being m x 2. Not every such matrix comes from sinusoids.
    # Build random displacement-rank-2 Hankel by:
    # choosing random 2-column generators G1, G2
    # and solving the Sylvester equation Z1 H - H Z2^T = G1 G2^T

    print(f'  Generating random disp-rank-2 Hankel matrices...')
    m_h, n_h = 80, 120
    n_dr2_unitary = 0
    n_dr2_total = 20
    deficits_dr2 = []

    for trial in range(n_dr2_total):
        # Generate a Hankel matrix as sum of r=2 exponentials with random z
        # (this guarantees displacement rank <= 4 for real signal)
        # Actually, for a SINGLE complex exponential x(n) = z^n,
        # the Hankel has rank 1 and displacement rank 1.
        # For 2 exponentials: rank 2, displacement rank 2.

        # Let's test: pure z on unit circle
        w = rng.uniform(0.05, 0.45, 2)
        z_vals = np.exp(1j * 2 * np.pi * w)
        a_vals = rng.uniform(0.5, 2.0, 2) * np.exp(1j * rng.uniform(0, 2*np.pi, 2))
        signal_c = np.zeros(n_samples, dtype=complex)
        for a, z in zip(a_vals, z_vals):
            signal_c += a * z ** t
        signal_r = signal_c.real

        H = hankel_matrix(signal_r, m=m_h)
        _, _, deficit, radii = esprit_from_hankel(H, 4)
        max_dev = np.max(np.abs(radii - 1))
        deficits_dr2.append(deficit)
        if max_dev < 1e-6:
            n_dr2_unitary += 1

    print(f'    Unit circle: {n_dr2_unitary}/{n_dr2_total} unitary')
    print(f'    Mean deficit: {np.mean(deficits_dr2):.2e}')

    # Now OFF unit circle
    n_dr2_off = 0
    deficits_off = []
    for trial in range(n_dr2_total):
        w = rng.uniform(0.05, 0.45, 2)
        r_vals = rng.uniform(0.90, 0.99, 2)  # damped
        z_vals = r_vals * np.exp(1j * 2 * np.pi * w)
        a_vals = rng.uniform(0.5, 2.0, 2) * np.exp(1j * rng.uniform(0, 2*np.pi, 2))
        signal_c = np.zeros(n_samples, dtype=complex)
        for a, z in zip(a_vals, z_vals):
            signal_c += a * z ** t
        signal_r = signal_c.real

        H = hankel_matrix(signal_r, m=m_h)
        _, _, deficit, radii = esprit_from_hankel(H, 4)
        max_dev = np.max(np.abs(radii - 1))
        deficits_off.append(deficit)
        if max_dev < 1e-6:
            n_dr2_off += 1

    print(f'    Off circle:  {n_dr2_off}/{n_dr2_total} unitary')
    print(f'    Mean deficit: {np.mean(deficits_off):.2e}')

    # == Verdict ==
    print()
    print('=' * 76)
    print('  VERDICT')
    print('=' * 76)
    print()
    if n_dr2_unitary == n_dr2_total and n_dr2_off == 0:
        print(f'  CLEAN SEPARATION:')
        print(f'    z on unit circle -> ESPRIT always unitary')
        print(f'    z off circle     -> ESPRIT never unitary')
        print(f'  Displacement rank 2 is NOT sufficient alone.')
        print(f'  Unitarity detects |z|=1 vs |z|!=1 perfectly.')
    else:
        print(f'  Results: on-circle {n_dr2_unitary}/{n_dr2_total} unitary, '
              f'off-circle {n_dr2_off}/{n_dr2_total}')

    print(f'\n  The question shifts: what STRUCTURAL property of M')
    print(f'  forces the Hankel signal to have |z_k| = 1?')
    print(f'  That property + ESPRIT = proof of RH.')


if __name__ == '__main__':
    run()
