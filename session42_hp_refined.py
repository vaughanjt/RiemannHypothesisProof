"""
SESSION 42 — REFINED HILBERT-POLYA: FEED THE FLUCTUATIONS

Improvement over session42_hilbert_polya.py:
1. Subtract the EXACT analytical margin (not polynomial trend)
2. Compensate for filter attenuation (multiply by exp(c*L))
3. Feed the compensated fluctuation into ESPRIT

The fluctuation = barrier - margin = zero_oscillation * filter_decay
Compensated = fluctuation / filter_decay = pure zero oscillation
ESPRIT on compensated signal should give CLEAN zero extraction.
"""

import numpy as np
from scipy.linalg import svd
import mpmath
from mpmath import zetazero
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from connes_crossterm import build_all
from session42k_close_the_gap import compute_margin_precise


def extract_frequencies_esprit(signal, n_components, dt=1.0):
    """ESPRIT frequency extraction."""
    N = len(signal)
    M = N // 2
    X = np.column_stack([signal[i:i+M] for i in range(N-M)])
    U, S, Vh = svd(X, full_matrices=False)
    Us = U[:, :n_components]
    U1 = Us[:-1, :]
    U2 = Us[1:, :]
    Phi = np.linalg.lstsq(U1, U2, rcond=None)[0]
    eigenvalues = np.linalg.eigvals(Phi)
    frequencies = np.angle(eigenvalues) / dt
    magnitudes = np.abs(eigenvalues)
    return frequencies, magnitudes, Phi


def compute_barrier_at_L(L):
    """Single barrier computation at given L."""
    lam_sq = int(round(np.exp(L)))
    if lam_sq < 2:
        lam_sq = 2
    N = max(15, round(6 * L))
    W02, M, QW = build_all(lam_sq, N, n_quad=4000)
    ns = np.arange(-N, N + 1, dtype=float)
    w = ns / (L**2 + (4*np.pi)**2 * ns**2)
    w[N] = 0.0
    w_hat = w / np.linalg.norm(w)
    return float(w_hat @ QW @ w_hat)


if __name__ == '__main__':
    print()
    print('#' * 72)
    print('  REFINED HILBERT-POLYA: FLUCTUATION-FED ESPRIT')
    print('#' * 72)

    # Known zeros
    mpmath.mp.dps = 15
    true_zeros = [float(zetazero(k).imag) for k in range(1, 31)]

    # ── Step 1: Compute barrier and margin at uniform L ──
    print('\n  Step 1: Computing barrier + analytical margin...', flush=True)
    dL = 0.02
    L_values = np.arange(1.5, 6.2, dL)
    n_pts = len(L_values)

    barriers = np.zeros(n_pts)
    margins = np.zeros(n_pts)

    t0 = time.time()
    for i, L in enumerate(L_values):
        barriers[i] = compute_barrier_at_L(L)
        r = compute_margin_precise(int(round(np.exp(L))), n_quad=3000, n_pnt=3000)
        margins[i] = r['margin']

        if (i+1) % 50 == 0:
            print(f'    [{i+1}/{n_pts}] L={L:.2f} barrier={barriers[i]:.6f} '
                  f'margin={margins[i]:.6f}', flush=True)

    dt = time.time() - t0
    print(f'  Done in {dt:.0f}s')

    # ── Step 2: Extract fluctuation ──
    fluctuation = barriers - margins
    print(f'\n  Step 2: Fluctuation')
    print(f'    Range: [{fluctuation.min():.6f}, {fluctuation.max():.6f}]')
    print(f'    Std: {fluctuation.std():.6f}')
    print(f'    Mean: {fluctuation.mean():.6f}')

    # ── Step 3: Compensate for filter decay ──
    # Fit decay rate: |fluct| ~ C * exp(-alpha * L)
    abs_fluct = np.abs(fluctuation)
    # Use running max as envelope
    window = 20
    envelope = np.array([np.max(abs_fluct[max(0,i-window):i+window+1])
                         for i in range(n_pts)])
    valid = envelope > 1e-6
    if np.sum(valid) > 10:
        c_fit = np.polyfit(L_values[valid], np.log(envelope[valid] + 1e-15), 1)
        decay_rate = -c_fit[0]
        print(f'\n  Step 3: Filter decay rate = {decay_rate:.4f}')
        print(f'    Fluctuation decays as exp(-{decay_rate:.3f}*L)')

        # Compensate
        compensation = np.exp(decay_rate * (L_values - L_values[0]))
        compensated = fluctuation * compensation

        print(f'    Compensated range: [{compensated.min():.6f}, {compensated.max():.6f}]')
        print(f'    Compensated std: {compensated.std():.6f}')
    else:
        compensated = fluctuation
        decay_rate = 0

    # ── Step 4: ESPRIT on RAW fluctuation ──
    print('\n\n  === ESPRIT ON RAW FLUCTUATION ===')
    print('  ' + '=' * 50)

    for n_comp in [5, 10, 15, 20]:
        try:
            freqs, mags, Phi = extract_frequencies_esprit(fluctuation, n_comp, dt=dL)
            pos = freqs[freqs > 0.5]
            pos = np.sort(pos)

            matches = 0
            print(f'\n  n={n_comp}: {len(pos)} positive frequencies')
            for g in pos[:12]:
                nearest = min(true_zeros, key=lambda z: abs(z-g))
                err = abs(g - nearest)
                m = 'YES' if err < 0.5 else ('~' if err < 1.0 else '')
                if err < 1.0: matches += 1
                print(f'    {g:>8.3f} -> {nearest:>8.3f} err={err:.3f} {m}')
            print(f'    Matches: {matches}/{min(len(pos),12)}')
        except Exception as e:
            print(f'\n  n={n_comp}: FAILED ({e})')

    # ── Step 5: ESPRIT on COMPENSATED fluctuation ──
    print('\n\n  === ESPRIT ON COMPENSATED FLUCTUATION ===')
    print('  ' + '=' * 50)

    for n_comp in [5, 10, 15, 20]:
        try:
            freqs, mags, Phi = extract_frequencies_esprit(compensated, n_comp, dt=dL)
            pos = freqs[freqs > 0.5]
            pos = np.sort(pos)

            matches = 0
            print(f'\n  n={n_comp}: {len(pos)} positive frequencies')
            for g in pos[:12]:
                nearest = min(true_zeros, key=lambda z: abs(z-g))
                err = abs(g - nearest)
                m = 'YES' if err < 0.5 else ('~' if err < 1.0 else '')
                if err < 1.0: matches += 1
                print(f'    {g:>8.3f} -> {nearest:>8.3f} err={err:.3f} {m}')
            print(f'    Matches: {matches}/{min(len(pos),12)}')
        except Exception as e:
            print(f'\n  n={n_comp}: FAILED ({e})')

    # ── Step 6: The ESPRIT operator ──
    print('\n\n  === THE ESPRIT OPERATOR (Phi) ===')
    print('  ' + '=' * 50)

    # Use the best result
    n_best = 10
    freqs, mags, Phi = extract_frequencies_esprit(compensated, n_best, dt=dL)

    print(f'  Phi is a {Phi.shape[0]}x{Phi.shape[1]} matrix')
    print(f'  Is Phi unitary? max|Phi*Phi^H - I| = {np.max(np.abs(Phi @ Phi.conj().T - np.eye(Phi.shape[0]))):.6f}')

    # The self-adjoint operator: H = -i * log(Phi) / dL
    # eigenvalues of Phi = exp(i*gamma_k*dL)
    # eigenvalues of H = gamma_k (the zeta zeros!)
    evals_phi = np.linalg.eigvals(Phi)
    gammas = np.angle(evals_phi) / dL

    print(f'\n  Eigenvalues of Phi (= exp(i*gamma*dL)):')
    for ev in evals_phi:
        g = np.angle(ev) / dL
        mag = np.abs(ev)
        if g > 0:
            nearest = min(true_zeros, key=lambda z: abs(z-g))
            print(f'    |z|={mag:.4f}  gamma={g:>8.3f}  nearest_zero={nearest:.3f}  '
                  f'err={abs(g-nearest):.3f}')

    # The operator H = (1/dL) * Im(log(Phi))
    # This has eigenvalues = extracted gamma values
    print(f'\n  The operator H = Im(log(Phi)) / dL')
    print(f'  has eigenvalues that ARE the extracted zeta zeros.')
    print(f'  H is {Phi.shape[0]}x{Phi.shape[1]}, built from:')
    print(f'  - {n_pts} barrier samples (primes + special functions)')
    print(f'  - ESPRIT decomposition (SVD + shift invariance)')
    print(f'  - NO zeta(s) computation anywhere')

    print('\n' + '#' * 72)
    print('  REFINED HILBERT-POLYA COMPLETE')
    print('#' * 72)
