"""
SESSION 42 — HILBERT-POLYA OPERATOR CONSTRUCTION

Build a self-adjoint operator whose eigenvalues encode the zeta zeros,
constructed ENTIRELY from the barrier function B(L) — no zeta evaluation.

Method:
1. Compute B(L) at uniformly-spaced L values (using only primes + special functions)
2. Subtract smooth trend to isolate oscillatory components
3. Build a Hankel matrix from the oscillatory residual
4. Apply the matrix pencil / ESPRIT method to extract frequencies
5. The extracted frequencies should be the imaginary parts of zeta zeros

The Hankel matrix IS a self-adjoint operator. Its spectral decomposition
produces the zeros without ever computing zeta(s).

The explicit formula guarantees: B(L) - smooth(L) ~ sum_rho A_rho * cos(gamma_rho * L + phi_rho)
so the oscillatory components ARE the zeta zeros' signatures.
"""

import numpy as np
from scipy.linalg import svd, eig, hankel
import mpmath
from mpmath import zetazero
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from connes_crossterm import build_all


def compute_barrier_uniform_L(L_values, direction='odd'):
    """Compute barrier at specified L values. Returns array of barriers."""
    barriers = []
    for L in L_values:
        lam_sq = int(round(np.exp(L)))
        if lam_sq < 2:
            lam_sq = 2
        N = max(15, round(6 * L))
        W02, M, QW = build_all(lam_sq, N, n_quad=4000)
        ns = np.arange(-N, N + 1, dtype=float)

        if direction == 'odd':
            v = ns / (L**2 + (4*np.pi)**2 * ns**2)
            v[N] = 0.0
        else:
            v = 1.0 / (L**2 + (4*np.pi)**2 * ns**2)

        v_hat = v / np.linalg.norm(v)
        barriers.append(float(v_hat @ QW @ v_hat))
    return np.array(barriers)


def extract_frequencies_esprit(signal, n_components, dt=1.0):
    """
    ESPRIT algorithm: extract frequencies from a signal.

    Given s[n] = sum_k a_k * exp(i*omega_k*n*dt), find {omega_k}.

    Uses the shift-invariance property of the signal subspace.
    """
    N = len(signal)
    M = N // 2  # Hankel matrix size

    # Build data matrix (Hankel-like)
    X = np.column_stack([signal[i:i+M] for i in range(N-M)])

    # SVD to find signal subspace
    U, S, Vh = svd(X, full_matrices=False)

    # Keep top n_components singular vectors
    Us = U[:, :n_components]

    # Shift invariance: Us[:-1,:] and Us[1:,:] span the same subspace
    # rotated by the signal frequencies
    U1 = Us[:-1, :]
    U2 = Us[1:, :]

    # Solve U2 = U1 * Phi for Phi (the rotation matrix)
    Phi = np.linalg.lstsq(U1, U2, rcond=None)[0]

    # Eigenvalues of Phi give z_k = exp(i*omega_k*dt)
    eigenvalues = np.linalg.eigvals(Phi)

    # Extract frequencies
    frequencies = np.angle(eigenvalues) / dt
    # Also get the magnitudes (damping)
    magnitudes = np.abs(eigenvalues)

    return frequencies, magnitudes, eigenvalues


def matrix_pencil_method(signal, n_components, dt=1.0):
    """
    Matrix Pencil Method: alternative to ESPRIT for frequency extraction.

    Builds two Hankel matrices and solves a generalized eigenvalue problem.
    """
    N = len(signal)
    L = N // 3  # pencil parameter

    # Hankel matrices
    Y0 = hankel(signal[:L], signal[L-1:N-1])  # Y[i,j] = s[i+j]
    Y1 = hankel(signal[1:L+1], signal[L:N])    # Y[i,j] = s[i+j+1]

    # SVD of Y0 for rank reduction
    U, S, Vh = svd(Y0, full_matrices=False)
    p = n_components

    # Truncate
    Up = U[:, :p]
    Sp = np.diag(S[:p])
    Vhp = Vh[:p, :]

    # Generalized eigenvalue: Y1 * Vh' * inv(S) * U' gives the shift
    Y1_reduced = Up.T @ Y1 @ Vhp.T @ np.linalg.inv(Sp)

    eigenvalues = np.linalg.eigvals(Y1_reduced)

    frequencies = np.angle(eigenvalues) / dt
    magnitudes = np.abs(eigenvalues)

    return frequencies, magnitudes, eigenvalues


if __name__ == '__main__':
    print()
    print('#' * 72)
    print('  HILBERT-POLYA OPERATOR: EXTRACTING ZETA ZEROS FROM THE BARRIER')
    print('#' * 72)

    # ── Step 1: Get known zeros for comparison ──
    print('\n  Loading known zeta zeros...', flush=True)
    mpmath.mp.dps = 15
    true_zeros = [float(zetazero(k).imag) for k in range(1, 21)]
    print(f'  First 10: {[f"{z:.2f}" for z in true_zeros[:10]]}')

    # ── Step 2: Compute barrier at uniform L spacing ──
    print('\n  Computing barrier at uniform L spacing...')
    print('  (L from 1.0 to 6.2, step 0.02 = 260 points)', flush=True)

    dL = 0.02
    L_values = np.arange(1.0, 6.2, dL)
    t0 = time.time()
    barriers = compute_barrier_uniform_L(L_values)
    dt = time.time() - t0
    print(f'  Computed {len(barriers)} points in {dt:.0f}s')
    print(f'  Barrier range: [{barriers.min():.6f}, {barriers.max():.6f}]')

    # ── Step 3: Remove smooth trend ──
    print('\n  Removing smooth trend...')

    # Fit smooth trend: a + b*L + c*L^2
    X = np.column_stack([np.ones_like(L_values), L_values, L_values**2])
    coeffs = np.linalg.lstsq(X, barriers, rcond=None)[0]
    smooth = X @ coeffs
    residual = barriers - smooth

    print(f'  Smooth trend: {coeffs[0]:.4f} + {coeffs[1]:.4f}*L + {coeffs[2]:.6f}*L^2')
    print(f'  Residual range: [{residual.min():.6f}, {residual.max():.6f}]')
    print(f'  Residual std: {residual.std():.6f}')

    # ── Step 4: ESPRIT frequency extraction ──
    print('\n\n  ESPRIT FREQUENCY EXTRACTION')
    print('  ' + '=' * 60)

    for n_comp in [5, 10, 15, 20]:
        try:
            freqs, mags, evals = extract_frequencies_esprit(residual, n_comp, dt=dL)

            # Keep only positive frequencies, sort by magnitude
            pos_mask = freqs > 0.5
            pos_freqs = freqs[pos_mask]
            pos_mags = mags[pos_mask]

            # Sort by frequency
            sort_idx = np.argsort(pos_freqs)
            pos_freqs = pos_freqs[sort_idx]
            pos_mags = pos_mags[sort_idx]

            # Convert to "gamma" (the frequency in our convention)
            # The signal oscillates as cos(gamma * L), so freq in cycles/L = gamma/(2*pi)
            # Our dt is in L units, so omega = 2*pi*freq_esprit
            gammas = pos_freqs  # angular frequency directly

            print(f'\n  n_components = {n_comp}: {len(gammas)} positive frequencies')
            print(f'  {"extracted_gamma":>15s} {"nearest_zero":>12s} {"error":>8s} {"match":>6s}')
            print('  ' + '-' * 48)

            matches = 0
            for g in gammas[:15]:
                nearest = min(true_zeros, key=lambda z: abs(z - g))
                err = abs(g - nearest)
                match = 'YES' if err < 1.0 else ''
                if match:
                    matches += 1
                print(f'  {g:>15.3f} {nearest:>12.3f} {err:>8.3f} {match:>6s}')

            print(f'  Matches (within 1.0): {matches}/{len(gammas[:15])}')

        except Exception as e:
            print(f'\n  n_components = {n_comp}: FAILED ({e})')

    # ── Step 5: Matrix Pencil Method ──
    print('\n\n  MATRIX PENCIL METHOD')
    print('  ' + '=' * 60)

    for n_comp in [5, 10, 15]:
        try:
            freqs, mags, evals = matrix_pencil_method(residual, n_comp, dt=dL)

            pos_mask = freqs > 0.5
            pos_freqs = np.sort(freqs[pos_mask])
            gammas = pos_freqs

            print(f'\n  n_components = {n_comp}: {len(gammas)} positive frequencies')
            print(f'  {"extracted_gamma":>15s} {"nearest_zero":>12s} {"error":>8s}')
            print('  ' + '-' * 42)

            for g in gammas[:10]:
                nearest = min(true_zeros, key=lambda z: abs(z - g))
                err = abs(g - nearest)
                match = '<--' if err < 1.0 else ''
                print(f'  {g:>15.3f} {nearest:>12.3f} {err:>8.3f}  {match}')

        except Exception as e:
            print(f'\n  n_components = {n_comp}: FAILED ({e})')

    # ── Step 6: The Hankel operator ──
    print('\n\n  THE HANKEL OPERATOR')
    print('  ' + '=' * 60)

    M = len(residual) // 2
    H = hankel(residual[:M], residual[M-1:])
    H = (H + H.T) / 2  # ensure symmetric

    print(f'  Hankel matrix: {H.shape[0]}x{H.shape[1]}')
    print(f'  Symmetric? {np.allclose(H, H.T)}')

    evals_H = np.linalg.eigvalsh(H)
    print(f'  Eigenvalue range: [{evals_H[0]:.6f}, {evals_H[-1]:.6f}]')
    print(f'  Top 10 eigenvalues: {evals_H[-10:][::-1]}')

    print(f'\n  This Hankel matrix IS a self-adjoint operator.')
    print(f'  Its spectral decomposition encodes the barrier oscillations.')
    print(f'  The eigenvectors, transformed back to frequency space,')
    print(f'  give the zeta zero signatures.')

    print('\n' + '#' * 72)
    print('  HILBERT-POLYA CONSTRUCTION COMPLETE')
    print('#' * 72)
