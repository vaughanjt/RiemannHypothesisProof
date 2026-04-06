"""
SESSION 60d -- PALINDROMIC KERNEL TEST

Markovsky (2014): signal has all poles on unit circle <=> Hankel kernel
contains a palindromic vector (c_0, c_1, ..., c_r) = (c_r, ..., c_1, c_0).

If the barrier Hankel has palindromic kernel, that's equivalent to:
  all oscillation frequencies are real = zeros on critical line = RH.

Test:
  1. Build Hankel of barrier signal
  2. Find kernel (null space)
  3. Check for palindromic vectors
  4. If palindromic, check if this follows from M's antisymmetry b_{-n} = -b_n
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


def palindromic_score(v):
    """How palindromic is v? Returns 1.0 for perfect palindrome, 0 for anti."""
    v_rev = v[::-1]
    if np.linalg.norm(v) < 1e-15:
        return 0.0
    # Check both palindromic (v = v_rev) and anti-palindromic (v = -v_rev)
    score_pal = np.abs(np.dot(v, v_rev)) / (np.linalg.norm(v) * np.linalg.norm(v_rev))
    return score_pal


def run():
    print()
    print('#' * 76)
    print('  SESSION 60d -- PALINDROMIC KERNEL TEST')
    print('#' * 76)

    # Load barrier signal
    data = np.load('session60_esprit_unitarity.npz')
    residual = data['residual']
    L_vals = data['L_vals']
    signal_odd = data['signal_odd']

    print(f'  Loaded: {len(residual)} samples, L in [{L_vals[0]:.2f}, {L_vals[-1]:.2f}]')

    # == Part 1: Hankel kernel of barrier residual ==
    print('\n  === PART 1: HANKEL KERNEL OF BARRIER ===')

    for m_frac in [0.3, 0.4, 0.5]:
        m = int(len(residual) * m_frac)
        H = hankel_matrix(residual, m)
        U, S, Vt = np.linalg.svd(H)

        # Kernel = right singular vectors with small singular values
        # Find rank (number of significant singular values)
        tol = 1e-6 * S[0]
        rank = np.sum(S > tol)
        kernel_dim = min(H.shape) - rank

        print(f'\n  m/n = {m_frac}: H is {H.shape[0]}x{H.shape[1]}, '
              f'rank={rank}, kernel dim={kernel_dim}')
        s_last = S[rank-1] if rank > 0 else 0
        s_next = S[rank] if rank < len(S) else 0
        print(f'  Singular values: {S[0]:.4f}, {S[1]:.4f}, ..., '
              f'{s_last:.4e}, {s_next:.4e}')

        if kernel_dim > 0:
            # Kernel vectors are last rows of Vt
            kernel_vecs = Vt[rank:, :]
            for i in range(min(kernel_dim, 5)):
                v = kernel_vecs[i]
                ps = palindromic_score(v)
                print(f'  Kernel vec {i}: palindromic score = {ps:.8f}')
        else:
            # No exact kernel; check near-kernel (smallest singular vectors)
            print(f'  No exact kernel. Checking near-kernel vectors:')
            for i in range(min(5, len(S))):
                v = Vt[-(i+1), :]
                ps = palindromic_score(v)
                print(f'  Near-kernel vec {i} (sv={S[-(i+1)]:.4e}): '
                      f'palindromic score = {ps:.8f}')

    # == Part 2: Test with known pure sinusoidal signal ==
    print('\n  === PART 2: CONTROL — PURE SINUSOIDAL SIGNAL ===')
    t = np.arange(300)
    sig_pure = np.cos(0.15 * t) + 0.7 * np.cos(0.30 * t) + 0.3 * np.cos(0.22 * t)
    H_pure = hankel_matrix(sig_pure, 120)
    U_p, S_p, Vt_p = np.linalg.svd(H_pure)
    tol_p = 1e-6 * S_p[0]
    rank_p = np.sum(S_p > tol_p)
    print(f'  Pure signal: rank={rank_p}, kernel dim={min(H_pure.shape)-rank_p}')
    kernel_p = Vt_p[rank_p:, :]
    for i in range(min(3, kernel_p.shape[0])):
        ps = palindromic_score(kernel_p[i])
        print(f'  Kernel vec {i}: palindromic score = {ps:.8f}')

    # == Part 3: Test with damped sinusoidal signal ==
    print('\n  === PART 3: CONTROL — DAMPED SINUSOIDAL SIGNAL ===')
    sig_damp = (np.exp(-0.02*t) * np.cos(0.15*t) +
                0.7 * np.exp(-0.01*t) * np.cos(0.30*t))
    H_damp = hankel_matrix(sig_damp, 120)
    U_d, S_d, Vt_d = np.linalg.svd(H_damp)
    tol_d = 1e-6 * S_d[0]
    rank_d = np.sum(S_d > tol_d)
    print(f'  Damped signal: rank={rank_d}, kernel dim={min(H_damp.shape)-rank_d}')
    kernel_d = Vt_d[rank_d:, :]
    for i in range(min(3, kernel_d.shape[0])):
        ps = palindromic_score(kernel_d[i])
        print(f'  Kernel vec {i}: palindromic score = {ps:.8f}')

    # == Part 4: ESPRIT characteristic polynomial ==
    print('\n  === PART 4: ESPRIT CHARACTERISTIC POLYNOMIAL ===')
    print(f'  Extract characteristic polynomial from ESPRIT and check palindromic.')

    for r in [6, 8, 10]:
        H = hankel_matrix(residual, len(residual) // 2)
        U, S, Vt = np.linalg.svd(H, full_matrices=False)
        U_r = U[:, :r]
        Phi = np.linalg.pinv(U_r[:-1, :]) @ U_r[1:, :]
        evals = np.linalg.eigvals(Phi)

        # Characteristic polynomial
        char_poly = np.poly(evals)  # coefficients of det(zI - Phi)
        char_poly_real = char_poly.real

        # Check palindromic: c_k = c_{r-k}
        n_c = len(char_poly_real)
        pal_error = 0.0
        for k in range(n_c):
            pal_error = max(pal_error,
                           abs(char_poly_real[k] - char_poly_real[n_c - 1 - k]))

        # Also check: are eigenvalues symmetric about unit circle?
        radii = np.abs(evals)
        on_circle = np.max(np.abs(radii - 1.0))

        print(f'\n  r={r}:')
        print(f'    Char poly coefficients: {char_poly_real}')
        print(f'    Palindromic error: {pal_error:.2e}')
        print(f'    Max |z|-1 deviation: {on_circle:.6f}')
        print(f'    Palindromic: {"YES" if pal_error < 1e-6 else "NO"}')
        print(f'    On unit circle: {"YES" if on_circle < 0.02 else "APPROX" if on_circle < 0.1 else "NO"}')

    # == Part 5: Does parity of M force palindromic? ==
    print('\n  === PART 5: PARITY SYMMETRY -> PALINDROMIC? ===')
    print(f'  M has b_{{-n}} = -b_n (antisymmetric generators).')
    print(f'  The barrier B(L) uses an ODD test function w_hat.')
    print(f'  Odd function x Loewner matrix with antisymmetric b:')
    print(f'    B(L) = w^T M w where w is odd and M has b_{{-n}} = -b_n')
    print()

    # The explicit formula gives B(L) = sum cos(gamma*L) terms
    # (no sin terms, because of the even/odd structure)
    # cos(gamma*L) is EVEN in L -> palindromic in samples

    # Check: is B(L) even around its midpoint?
    # B(L) for L in [L_min, L_max], centered at L_mid
    L_mid = (L_vals[0] + L_vals[-1]) / 2
    n_half = len(residual) // 2

    # Check symmetry of residual around midpoint
    res_first = residual[:n_half]
    res_last = residual[-n_half:][::-1]
    sym_error = np.linalg.norm(res_first - res_last) / np.linalg.norm(residual)
    asym_error = np.linalg.norm(res_first + res_last) / np.linalg.norm(residual)

    print(f'  Signal symmetry around L_mid = {L_mid:.2f}:')
    print(f'    Even component (f(L) = f(-L)):  {1 - sym_error:.6f}')
    print(f'    Odd component (f(L) = -f(-L)):   {1 - asym_error:.6f}')
    print(f'    (Palindromic kernel requires even-dominant signal)')

    # == Verdict ==
    print()
    print('=' * 76)
    print('  VERDICT')
    print('=' * 76)


if __name__ == '__main__':
    run()
