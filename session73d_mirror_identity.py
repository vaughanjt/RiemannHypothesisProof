"""
SESSION 73d -- DRILLING INTO THE SPECTRAL MIRROR (FIXED)

Uses build_all_fast for exact M, then decomposes into L_pure + D.
Tests whether the spectral mirror is arithmetic (prime-specific) or generic.
"""

import sys
import numpy as np

sys.path.insert(0, '.')
from session41g_uncapped_barrier import sieve_primes
from session49c_weil_residual import (
    build_all_fast, _compute_alpha, _compute_wr_diag
)


def decompose_exact(lam_sq):
    """Use build_all_fast for exact M, then extract L_pure and D."""
    L = float(np.log(lam_sq))
    N = max(15, round(6 * L))
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)

    _, M, QW = build_all_fast(lam_sq, N)

    # L_pure = off-diagonal of M + finite-diff diagonal
    Lp = M.copy()
    for i in range(dim):
        Lp[i, i] = 0
    # Fill diagonal with finite-diff estimate of the Cauchy limit
    # Use row average approach: L_diag[i] = mean of off-diagonal row
    # Actually, just use the original 59b decomposition approach
    # The cleanest: L_pure = M - diag(D), where D = diag(M) - Cauchy_diag_limit
    # Cauchy_diag_limit from B_n finite difference

    wr = _compute_wr_diag(L, N)
    alpha = _compute_alpha(L, N)
    primes = sieve_primes(int(lam_sq))
    B_prime = np.zeros(dim)
    for p in primes:
        pk = int(p)
        logp = np.log(p)
        while pk <= lam_sq:
            w = logp * pk ** (-0.5)
            y = np.log(pk)
            B_prime += w * np.sin(2 * np.pi * ns * y / L) / np.pi
            pk *= int(p)
    B_n = alpha + B_prime

    # Cauchy diagonal limit: B'(n) by centered finite difference
    Lp_diag = np.zeros(dim)
    for i in range(dim):
        if 0 < i < dim - 1:
            Lp_diag[i] = (B_n[i + 1] - B_n[i - 1]) / 2
        elif i == 0:
            Lp_diag[i] = B_n[1] - B_n[0]
        else:
            Lp_diag[i] = B_n[-1] - B_n[-2]

    # L_pure: off-diagonal from M, diagonal from Cauchy limit
    for i in range(dim):
        Lp[i, i] = Lp_diag[i]

    # D = diag part not in Cauchy limit
    Dd = np.diag(M) - Lp_diag

    # Verify: M = Lp + diag(Dd)
    M_check = Lp + np.diag(Dd)
    err = np.max(np.abs(M - M_check))

    return Lp, Dd, M, QW, N, L, dim, ns, err


def get_near_zero_subspace(M, threshold=0.01):
    evals, evecs = np.linalg.eigh(M)
    nz_mask = np.abs(evals) < threshold
    return evecs[:, nz_mask], evals[nz_mask], evals, evecs


def mirror_metric(Lp, Dd, V_nz):
    if V_nz.shape[1] == 0:
        return float('inf'), 0, 0, 0
    L_nz = V_nz.T @ Lp @ V_nz
    D_nz = V_nz.T @ np.diag(Dd) @ V_nz
    M_nz = L_nz + D_nz
    nL = np.linalg.norm(L_nz)
    nD = np.linalg.norm(D_nz)
    nM = np.linalg.norm(M_nz)
    cancel = nM / nL if nL > 0 else float('inf')
    return cancel, nL, nD, nM


def build_cramer_M(lam_sq, cramer_primes):
    """Build M using Cramer primes instead of actual primes."""
    L = float(np.log(lam_sq))
    N = max(15, round(6 * L))
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)

    wr = _compute_wr_diag(L, N)
    alpha = _compute_alpha(L, N)

    # Build M_prime from Cramer primes
    M_prime = np.zeros((dim, dim))
    for p in cramer_primes:
        pk = int(p)
        logp = np.log(float(p))
        while pk <= lam_sq:
            w = logp * pk ** (-0.5)
            y = np.log(float(pk))
            # v_n = sqrt(w) * [kernel function at (n, y)]
            # M_prime += w * outer(v, v) style
            phase = 2 * np.pi * ns * y / L
            cos_phase = np.cos(phase)
            sin_phase = np.sin(phase)
            # The (n,m) entry of M_prime for this prime power:
            # w * (L - y) / L * sinc-like kernel
            for i in range(dim):
                for j in range(dim):
                    ni = ns[i]
                    nj = ns[j]
                    if ni == nj:
                        M_prime[i, j] += w * 2 * (L - y) / L * np.cos(2*np.pi*ni*y/L)
                    else:
                        M_prime[i, j] += w * (np.sin(2*np.pi*nj*y/L) - np.sin(2*np.pi*ni*y/L)) / (np.pi*(ni-nj))
            pk *= int(p)

    # Build full M = diag(wr) + alpha_offdiag + M_prime
    M = np.zeros((dim, dim))
    for i in range(dim):
        M[i, i] = wr[abs(int(ns[i]))]
    # alpha off-diagonal
    for i in range(dim):
        for j in range(dim):
            if i != j:
                M[i, j] += (alpha[j] - alpha[i]) / (ns[i] - ns[j])
    M += M_prime

    return M, N, dim


def build_cramer_M_fast(lam_sq, cramer_primes):
    """Faster Cramer M build using vectorized operations."""
    L = float(np.log(lam_sq))
    N = max(15, round(6 * L))
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)

    wr = _compute_wr_diag(L, N)
    alpha = _compute_alpha(L, N)

    # Alpha off-diagonal (Cauchy structure)
    nm = ns[:, None] - ns[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        alpha_offdiag = (alpha[None, :] - alpha[:, None]) / nm
    np.fill_diagonal(alpha_offdiag, 0)

    # M_prime from Cramer primes
    M_prime = np.zeros((dim, dim))
    for p in cramer_primes:
        pk = int(p)
        logp = np.log(float(p))
        while pk <= lam_sq:
            w = logp * pk ** (-0.5)
            y = np.log(float(pk))
            phase = 2 * np.pi * ns * y / L
            sin_phase = np.sin(phase)
            cos_phase = np.cos(phase)

            # Off-diagonal: w * (sin(phase_m) - sin(phase_n)) / (pi*(n-m))
            sin_diff = sin_phase[None, :] - sin_phase[:, None]
            with np.errstate(divide='ignore', invalid='ignore'):
                offdiag = w * sin_diff / (np.pi * nm)
            np.fill_diagonal(offdiag, 0)

            # Diagonal: w * 2*(L-y)/L * cos(phase_n)
            diag = w * 2 * (L - y) / L * cos_phase

            M_prime += offdiag + np.diag(diag)
            pk *= int(p)

    M = np.diag(np.array([wr[abs(int(n))] for n in ns])) + alpha_offdiag + M_prime
    M = (M + M.T) / 2

    return M, N, dim


def run():
    print()
    print('#' * 76)
    print('  SESSION 73d -- DRILLING INTO THE SPECTRAL MIRROR')
    print('#' * 76)

    # ==================================================================
    # TEST 1: Verify exact decomposition
    # ==================================================================
    print(f'\n  === BASELINE: EXACT DECOMPOSITION ===\n')

    lam_sq = 1000
    Lp, Dd, M, QW, N, L, dim, ns, err = decompose_exact(lam_sq)
    V_nz, evals_nz, evals_all, evecs_all = get_near_zero_subspace(M)
    cancel, nL, nD, nM = mirror_metric(Lp, Dd, V_nz)

    n_pos = np.sum(evals_all > 1e-10)
    print(f'  lam^2={lam_sq}, dim={dim}, reconstruction err: {err:.2e}')
    print(f'  #pos eigenvalues: {n_pos}')
    print(f'  Near-zero subspace: {V_nz.shape[1]} dimensions')
    print(f'  Mirror: ||L+D||/||L|| = {cancel:.6e}')
    print(f'  ||L_nz||={nL:.4f}, ||D_nz||={nD:.4f}, ||M_nz||={nM:.6f}')
    sys.stdout.flush()

    # ==================================================================
    # TEST 2: Cramer primes — does the mirror survive?
    # ==================================================================
    print(f'\n  === TEST 2: CRAMER MODEL ===\n')

    actual_primes = list(sieve_primes(int(lam_sq)))
    n_primes = len(actual_primes)
    np.random.seed(42)

    print(f'  {n_primes} actual primes up to {lam_sq}')
    print(f'  {"trial":>5} {"#pos":>5} {"#nz":>5} {"mirror":>12} {"Lorentzian":>11}')
    print('  ' + '-' * 42)

    for trial in range(10):
        cramer = []
        n = 2
        while len(cramer) < n_primes and n < 5 * lam_sq:
            if np.random.random() < 1.0 / max(np.log(n), 1):
                cramer.append(n)
            n += 1
        cramer = cramer[:n_primes]

        M_c, N_c, dim_c = build_cramer_M_fast(lam_sq, cramer)
        evals_c = np.linalg.eigvalsh(M_c)
        n_pos_c = np.sum(evals_c > 1e-10)

        # Decompose Cramer M the same way
        Lp_c, Dd_c, _, _, _, _, _, _, _ = decompose_exact.__wrapped__(lam_sq) if hasattr(decompose_exact, '__wrapped__') else (None, None, None, None, None, None, None, None, None)

        # Actually just check near-zero subspace of M_c
        V_nz_c, _, _, _ = get_near_zero_subspace(M_c)
        n_nz_c = V_nz_c.shape[1]
        lor_c = n_pos_c <= 1

        print(f'  {trial+1:>5d} {n_pos_c:>5d} {n_nz_c:>5d} {"---":>12} '
              f'{"YES" if lor_c else "no":>11}')
    sys.stdout.flush()

    # ==================================================================
    # TEST 3: Archimedean-only (no primes)
    # ==================================================================
    print(f'\n  === TEST 3: ARCHIMEDEAN ONLY ===\n')

    M_arch, N_arch, dim_arch = build_cramer_M_fast(lam_sq, [])
    evals_arch = np.linalg.eigvalsh(M_arch)
    n_pos_arch = np.sum(evals_arch > 1e-10)
    V_nz_arch, _, _, _ = get_near_zero_subspace(M_arch)

    print(f'  No primes: #pos = {n_pos_arch}, #near-zero = {V_nz_arch.shape[1]}')
    print(f'  Eigenvalue range: [{evals_arch.min():.4f}, {evals_arch.max():.4f}]')
    print(f'  Lorentzian? {n_pos_arch <= 1}')
    sys.stdout.flush()

    # ==================================================================
    # TEST 4: Add primes one at a time
    # ==================================================================
    print(f'\n  === TEST 4: PRIME-BY-PRIME EMERGENCE ===\n')

    print(f'  {"#primes":>8} {"last_p":>8} {"#pos":>5} {"#nz":>5} {"Lorentzian":>11}')
    print('  ' + '-' * 42)

    for n_p in [0, 1, 2, 3, 5, 10, 20, 50, 100, n_primes]:
        subset = actual_primes[:n_p]
        M_s, _, dim_s = build_cramer_M_fast(lam_sq, subset)
        evals_s = np.linalg.eigvalsh(M_s)
        n_pos_s = np.sum(evals_s > 1e-10)
        V_nz_s, _, _, _ = get_near_zero_subspace(M_s)
        last_p = int(subset[-1]) if subset else 0
        lor = n_pos_s <= 1
        print(f'  {n_p:>8d} {last_p:>8d} {n_pos_s:>5d} {V_nz_s.shape[1]:>5d} '
              f'{"YES" if lor else "no":>11}')
    sys.stdout.flush()

    # ==================================================================
    # TEST 5: Near-zero eigenvector structure
    # ==================================================================
    print(f'\n  === TEST 5: NEAR-ZERO EIGENVECTOR STRUCTURE ===\n')

    V_nz, evals_nz, evals_all, evecs_all = get_near_zero_subspace(M)
    n_nz = V_nz.shape[1]

    if n_nz > 0:
        # Parity
        parity = np.zeros(n_nz)
        for j in range(n_nz):
            v = V_nz[:, j]
            even_e = v[N]**2 + sum((v[N+k]+v[N-k])**2/2 for k in range(1, N+1))
            odd_e = sum((v[N+k]-v[N-k])**2/2 for k in range(1, N+1))
            parity[j] = even_e - odd_e

        n_even = np.sum(parity > 0.5)
        n_odd = np.sum(parity < -0.5)
        print(f'  {n_nz} near-zero eigenvectors: {n_even} even, {n_odd} odd, '
              f'{n_nz-n_even-n_odd} mixed')

        # Frequency content
        freqs = np.abs(ns)
        mean_freq = np.array([np.sum(freqs * V_nz[:, j]**2) for j in range(n_nz)])
        print(f'  Mean |n|: min={mean_freq.min():.1f}, max={mean_freq.max():.1f}, '
              f'mean={mean_freq.mean():.1f} (N={N})')

        # Are they high-frequency? (prolate null space)
        print(f'  Bandwidth W = pi*N/L = {np.pi*N/L:.1f}')
        print(f'  Signal space dim ~ 2W/pi = {2*np.pi*N/(np.pi*L):.0f}')
        print(f'  Near-zero dim = {n_nz} vs null dim = {dim - round(2*np.pi*N/(np.pi*L)):.0f}')
    sys.stdout.flush()

    # ==================================================================
    # TEST 6: Mirror quality by parity subspace
    # ==================================================================
    print(f'\n  === TEST 6: MIRROR BY PARITY ===\n')

    dim_even = N + 1
    P_even = np.zeros((dim, dim_even))
    P_even[N, 0] = 1.0
    for k in range(1, N + 1):
        P_even[N + k, k] = 1.0 / np.sqrt(2)
        P_even[N - k, k] = 1.0 / np.sqrt(2)

    dim_odd = N
    P_odd = np.zeros((dim, dim_odd))
    for k in range(1, N + 1):
        P_odd[N + k, k - 1] = 1.0 / np.sqrt(2)
        P_odd[N - k, k - 1] = -1.0 / np.sqrt(2)

    for label, P, d in [('EVEN', P_even, dim_even), ('ODD', P_odd, dim_odd)]:
        M_sub = P.T @ M @ P
        Lp_sub = P.T @ Lp @ P
        Dd_sub = np.diag(P.T @ np.diag(Dd) @ P)

        evals_sub = np.linalg.eigvalsh(M_sub)
        n_pos_sub = np.sum(evals_sub > 1e-10)

        V_nz_sub, _, _, _ = get_near_zero_subspace(M_sub)
        cancel_sub, nL_sub, nD_sub, nM_sub = mirror_metric(Lp_sub, Dd_sub, V_nz_sub)

        print(f'  {label} (dim={d}): #pos={n_pos_sub}, #nz={V_nz_sub.shape[1]}, '
              f'mirror={cancel_sub:.4e}')
        if V_nz_sub.shape[1] > 0:
            print(f'    ||L||={nL_sub:.4f}, ||D||={nD_sub:.4f}, ||M||={nM_sub:.6f}')
    sys.stdout.flush()

    # ==================================================================
    # TEST 7: Lambda scaling of mirror quality
    # ==================================================================
    print(f'\n  === TEST 7: MIRROR vs LAMBDA ===\n')

    print(f'  {"lam^2":>8} {"dim":>5} {"#nz":>5} {"cancel":>12} {"||M_nz||":>12}')
    print('  ' + '-' * 46)

    for lam_sq_t in [50, 100, 200, 500, 1000, 2000, 5000, 10000]:
        try:
            Lp_t, Dd_t, M_t, _, N_t, L_t, dim_t, _, err_t = decompose_exact(lam_sq_t)
            V_nz_t, _, _, _ = get_near_zero_subspace(M_t)
            cancel_t, _, _, nM_t = mirror_metric(Lp_t, Dd_t, V_nz_t)
            print(f'  {lam_sq_t:>8d} {dim_t:>5d} {V_nz_t.shape[1]:>5d} '
                  f'{cancel_t:>12.6e} {nM_t:>12.6f}')
        except Exception as e:
            print(f'  {lam_sq_t:>8d} ERROR: {e}')
    sys.stdout.flush()

    # ==================================================================
    # VERDICT
    # ==================================================================
    print()
    print('=' * 76)
    print('  SESSION 73d VERDICT')
    print('=' * 76)


if __name__ == '__main__':
    run()
