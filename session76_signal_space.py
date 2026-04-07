"""
SESSION 76 -- DEEP PROBE OF THE 15-DIM SIGNAL SPACE

Session 73 discovered: M has ~15 "active" dimensions (the signal space)
and ~67 "near-zero" dimensions (the null space / spectral mirror).

The null space is the EASY part: L_pure and D cancel to 0.077%.
The signal space is WHERE RH LIVES.

HANDOFF.json says "governed by first ~12 zeros" -- test this.

Six probes:
  1. Signal-space basis: what are the ~15 eigenvectors? Structure in n-basis?
  2. Prolate connection: is signal space = Slepian signal space (Shannon 2WT)?
  3. Zero projection: how much signal space is spanned by zero directions?
  4. Boundary determination: why ~15? What sets the cutoff?
  5. Coupling anatomy in signal basis: which components dominate the ratio?
  6. Signal vs null RH anatomy: what makes M neg-def (odd) on signal space?

Infrastructure: build_all_fast (49c), odd_block (62b/75b), decompose_exact (73d).
"""

import sys
import numpy as np
import mpmath

sys.path.insert(0, '.')
from session41g_uncapped_barrier import sieve_primes
from session49c_weil_residual import (
    build_all_fast, _compute_alpha, _compute_wr_diag
)


def odd_block(M, N):
    """Project M onto the odd (n -> -n antisymmetric) subspace."""
    dim = 2 * N + 1
    P = np.zeros((dim, N))
    for n in range(1, N + 1):
        P[N + n, n - 1] = 1.0 / np.sqrt(2)
        P[N - n, n - 1] = -1.0 / np.sqrt(2)
    return P.T @ M @ P, P


def even_block(M, N):
    """Project M onto the even (n -> -n symmetric) subspace."""
    dim = 2 * N + 1
    dim_even = N + 1
    P = np.zeros((dim, dim_even))
    P[N, 0] = 1.0
    for k in range(1, N + 1):
        P[N + k, k] = 1.0 / np.sqrt(2)
        P[N - k, k] = 1.0 / np.sqrt(2)
    return P.T @ M @ P, P


def decompose_exact(lam_sq):
    """Decompose M = L_pure + diag(D). Returns both plus metadata."""
    L = float(np.log(lam_sq))
    N = max(15, round(6 * L))
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)

    _, M, QW = build_all_fast(lam_sq, N)

    # Compute B_n for Cauchy diagonal limit
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
    Lp = M.copy()
    for i in range(dim):
        Lp[i, i] = Lp_diag[i]

    D = np.diag(M) - Lp_diag

    return Lp, D, M, QW, N, L, dim, ns, B_n


def get_spectral_groups(M, threshold=0.01):
    """Split spectrum into top (positive), near-zero, bulk negative."""
    evals, evecs = np.linalg.eigh(M)
    top_idx = [len(evals) - 1]  # largest eigenvalue
    nz_mask = np.abs(evals) < threshold
    nz_mask[top_idx[0]] = False  # exclude top even if near boundary
    bulk_mask = (evals < -threshold)

    return {
        'evals': evals, 'evecs': evecs,
        'top_idx': top_idx,
        'nz_idx': np.where(nz_mask)[0],
        'bulk_idx': np.where(bulk_mask)[0],
    }


def run():
    print()
    print('#' * 76)
    print('  SESSION 76 -- DEEP PROBE OF THE 15-DIM SIGNAL SPACE')
    print('#' * 76)

    # ======================================================================
    # PROBE 1: Signal-space basis and eigenvector structure
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 1: SIGNAL-SPACE BASIS STRUCTURE')
    print(f'{"="*76}\n')

    lam_sq = 1000
    Lp, Dd, M, QW, N, L, dim, ns, B_n = decompose_exact(lam_sq)
    groups = get_spectral_groups(M)
    evals, evecs = groups['evals'], groups['evecs']

    bulk_idx = groups['bulk_idx']
    nz_idx = groups['nz_idx']
    top_idx = groups['top_idx']

    n_signal = len(bulk_idx) + len(top_idx)  # signal = bulk neg + top pos
    n_null = len(nz_idx)

    print(f'  lam^2={lam_sq}, dim={dim}, N={N}, L={L:.4f}')
    print(f'  Signal space: {n_signal} dims ({len(bulk_idx)} bulk neg + {len(top_idx)} top pos)')
    print(f'  Null space:   {n_null} dims (near-zero, spectral mirror)')
    print()

    # Signal-space eigenvectors: frequency content
    signal_idx = np.concatenate([bulk_idx, top_idx])
    V_signal = evecs[:, signal_idx]
    signal_evals = evals[signal_idx]

    freqs = np.abs(ns)
    print(f'  Signal eigenvector frequency content:')
    print(f'  {"idx":>4} {"eigenvalue":>14} {"mean|n|":>8} {"max|n|":>8} '
          f'{"n=0 wt":>8} {"parity":>8}')
    print('  ' + '-' * 56)

    for j in range(len(signal_idx)):
        v = V_signal[:, j]
        mean_n = np.sum(freqs * v**2)
        max_n_idx = np.argmax(np.abs(v))
        max_n = abs(ns[max_n_idx])
        n0_wt = v[N]**2  # weight at n=0
        # parity
        even_e = v[N]**2 + sum((v[N+k]+v[N-k])**2/2 for k in range(1, N+1))
        parity = 'even' if even_e > 0.9 else ('odd' if even_e < 0.1 else f'{even_e:.2f}')
        print(f'  {j:>4d} {signal_evals[j]:>+14.6e} {mean_n:>8.1f} {max_n:>8.0f} '
              f'{n0_wt:>8.4f} {parity:>8}')
    sys.stdout.flush()

    # Localization: how spread out is each signal eigenvector?
    print(f'\n  Participation ratio (1/sum(v^4), 1=delta, dim=uniform):')
    for j in range(len(signal_idx)):
        v = V_signal[:, j]
        pr = 1.0 / np.sum(v**4)
        print(f'    eig {j}: PR = {pr:.1f} / {dim}')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 2: Prolate / Slepian connection
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 2: PROLATE / SLEPIAN CONNECTION')
    print(f'{"="*76}\n')

    # The prolate (Slepian) concentration problem: signals bandlimited to
    # [-W, W] and timelimited to [-T, T]. Shannon number = 2WT/pi.
    # In our setup: "frequency" = n (integer, |n| <= N), "time" = y/L in [0,1].
    # The Cauchy kernel sin(2*pi*n*y/L) / (pi*n) has bandwidth W = 2*pi*N/L.
    # The time window is T = L (the log-cutoff).
    # Shannon number = 2 * (2*pi*N/L) * L / (2*pi) = 2*N.
    # But that's the full bandwidth. The EFFECTIVE bandwidth is smaller.
    #
    # Alternative: the Slepian matrix for our problem is the prolate operator
    # projected onto integer frequencies: C[n,m] = sin(pi*(n-m)*c) / (pi*(n-m))
    # where c = L/(2*pi) (fraction of circle).
    # Shannon number = 2*N*c = 2*N*L/(2*pi) = N*L/pi.

    for lam_sq_test in [200, 500, 1000, 2000, 5000, 10000]:
        L_t = np.log(lam_sq_test)
        N_t = max(15, round(6 * L_t))
        dim_t = 2 * N_t + 1

        # Build Slepian concentration matrix for bandwidth N, time-window L
        # C[n,m] = integral_0^L e^{2*pi*i*(n-m)*y/L} dy / L = sinc((n-m)*1) = delta_{n,m}
        # No, that's trivial. The right formulation:
        #
        # Our M has "frequencies" n = -N..N and the kernel concentrates
        # on the interval [0, L] in log-space. The Slepian concentration
        # parameter is c = L / (2*pi) for the sinc kernel.
        #
        # Prolate matrix: P[n,m] = sin(2*pi*(n-m)*c) / (pi*(n-m)) for n != m
        #                 P[n,n] = 2*c
        # where c is the concentration parameter.
        # For our Cauchy kernel with period L: c = 1 (signals on [0, L] with
        # freq spacing 1/L). Shannon number = 2*N*c.
        # But that gives Shannon = 2*N = dim-1, which is the full dimension.
        #
        # The RIGHT analogy: the prolate operator for our specific kernel.
        # The Cauchy off-diagonal B_n/(n-m) concentrates on modes where
        # B_n varies slowly. The "signal space" = modes where B_n has
        # significant Fourier content.
        #
        # Actually: let's just build the Slepian concentration matrix
        # for the interval [0, 1] (normalized) and see if its eigenvalue
        # transition matches our signal/null split.

        # Slepian matrix for discrete frequencies n in {-N,...,N}
        # concentrated on fraction f of the period
        f = 1.0  # full period -- this would be trivial
        # The real question: what fraction of the "B_n bandwidth" is active?

        # Instead: compute the ACTUAL signal/null split
        _, M_t, _ = build_all_fast(lam_sq_test, N_t)
        evals_t = np.linalg.eigvalsh(M_t)
        n_nz = np.sum(np.abs(evals_t) < 0.01)
        n_signal_t = dim_t - n_nz
        shannon = N_t * L_t / np.pi  # candidate Shannon number

        print(f'  lam^2={lam_sq_test:>6d}: dim={dim_t:>3d}, N={N_t:>3d}, L={L_t:.3f}, '
              f'signal={n_signal_t:>3d}, null={n_nz:>3d}, '
              f'N*L/pi={shannon:.1f}, 2*L/pi={2*L_t/np.pi:.1f}')
    sys.stdout.flush()

    # Build actual prolate concentration matrix and compare eigenvalue structure
    print(f'\n  Prolate concentration matrix eigenvalues vs M spectrum:')
    lam_sq = 1000
    Lp, Dd, M, QW, N, L, dim, ns, B_n = decompose_exact(lam_sq)

    # Prolate: C[n,m] = sin(pi*(n-m)*c) / (pi*(n-m)), C[n,n] = c
    # concentration parameter c = fraction of "bandwidth" that's active
    # Try c = L/(2*pi*N) -- fraction of full freq range that's "populated"
    c_vals = [0.1, 0.2, L / (2 * np.pi), 1 / np.pi, 0.5]
    for c in c_vals:
        C_mat = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                if i == j:
                    C_mat[i, j] = c
                else:
                    diff = ns[i] - ns[j]
                    C_mat[i, j] = np.sin(np.pi * diff * c) / (np.pi * diff)
        prol_evals = np.linalg.eigvalsh(C_mat)
        n_above = np.sum(prol_evals > 0.5)
        n_transition = np.sum((prol_evals > 0.01) & (prol_evals < 0.99))
        print(f'    c={c:.4f}: {n_above} above 0.5, {n_transition} in transition, '
              f'Shannon = {2*N*c:.1f}')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 3: Zero projection -- does zero-space span signal space?
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 3: ZETA ZERO PROJECTION')
    print(f'{"="*76}\n')

    # Build "zero vectors": z_k[n] = cos(gamma_k * 2*pi*n/L) for even,
    #                        z_k[n] = sin(gamma_k * 2*pi*n/L) for odd
    # These are the directions that the zeta zeros "activate" in the matrix.

    mpmath.mp.dps = 30
    n_zeros_load = 50
    zeros = [float(mpmath.zetazero(k).imag) for k in range(1, n_zeros_load + 1)]

    # For each zero gamma_k, construct the vector in our n-basis
    # The Weil explicit formula contributes terms like cos(gamma_k * y)
    # where y = 2*pi*n/L, so the n-space vector is:
    #   v_cos[n] = cos(gamma_k * 2*pi*n/L)  (even part)
    #   v_sin[n] = sin(gamma_k * 2*pi*n/L)  (odd part)

    signal_evecs = evecs[:, signal_idx]
    V_signal_proj = signal_evecs  # columns are signal eigenvectors

    print(f'  Signal space dim = {n_signal}')
    print(f'  Testing: how much signal space is spanned by first K zero directions?\n')

    print(f'  {"K zeros":>8} {"cos+sin vecs":>12} {"explained var":>14} '
          f'{"max residual":>14} {"min sval":>12}')
    print('  ' + '-' * 66)

    for K in [3, 5, 8, 10, 12, 15, 20, 30, 50]:
        if K > len(zeros):
            break
        # Build zero basis: cos and sin for each gamma_k
        zero_vecs = []
        for k in range(K):
            gamma = zeros[k]
            v_cos = np.cos(gamma * 2 * np.pi * ns / L)
            v_sin = np.sin(gamma * 2 * np.pi * ns / L)
            v_cos /= np.linalg.norm(v_cos)
            v_sin /= np.linalg.norm(v_sin)
            zero_vecs.append(v_cos)
            zero_vecs.append(v_sin)

        Z = np.column_stack(zero_vecs)
        # Project signal eigenvectors onto zero-space
        # explained = ||proj_Z(v_signal)||^2 / ||v_signal||^2
        Q, _ = np.linalg.qr(Z)  # orthonormal basis of zero-space
        proj = Q @ (Q.T @ V_signal_proj)
        explained = np.sum(proj**2) / np.sum(V_signal_proj**2)

        # Per-eigenvector residual
        residuals = []
        for j in range(V_signal_proj.shape[1]):
            v = V_signal_proj[:, j]
            p = Q @ (Q.T @ v)
            residuals.append(np.linalg.norm(v - p))
        max_resid = max(residuals)

        # Singular values of Z^T V_signal
        sv = np.linalg.svd(Q.T @ V_signal_proj, compute_uv=False)
        min_sv = sv.min() if len(sv) > 0 else 0

        print(f'  {K:>8d} {2*K:>12d} {explained:>14.6f} {max_resid:>14.6e} '
              f'{min_sv:>12.6f}')
    sys.stdout.flush()

    # Which zeros matter most? Project each signal eigenvector onto individual zeros
    print(f'\n  Per-zero overlap with signal space (lam^2={lam_sq}):')
    print(f'  {"zero#":>6} {"gamma_k":>12} {"overlap^2":>12} {"cumulative":>12}')
    print('  ' + '-' * 46)

    cumulative = 0
    total_signal_norm2 = np.sum(V_signal_proj**2)
    for k in range(min(20, len(zeros))):
        gamma = zeros[k]
        v_cos = np.cos(gamma * 2 * np.pi * ns / L)
        v_sin = np.sin(gamma * 2 * np.pi * ns / L)
        v_cos /= np.linalg.norm(v_cos)
        v_sin /= np.linalg.norm(v_sin)

        overlap = 0
        for j in range(V_signal_proj.shape[1]):
            v = V_signal_proj[:, j]
            overlap += (v @ v_cos)**2 + (v @ v_sin)**2
        overlap /= total_signal_norm2
        cumulative += overlap
        print(f'  {k+1:>6d} {gamma:>12.6f} {overlap:>12.6f} {cumulative:>12.6f}')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 4: What determines the boundary at ~15 dimensions?
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 4: WHAT SETS THE SIGNAL/NULL BOUNDARY?')
    print(f'{"="*76}\n')

    # Hypothesis A: signal dim = number of zeros gamma_k < L/2
    # Hypothesis B: signal dim = 2*L/pi (some Shannon-like number)
    # Hypothesis C: signal dim = number of zeros gamma_k < pi*N/L
    # Hypothesis D: signal dim tracks with prolate transition width
    # Hypothesis E: signal dim = floor(L) or ceil(L) or round(L)

    print(f'  {"lam^2":>8} {"L":>8} {"N":>4} {"dim":>5} {"#signal":>8} '
          f'{"2L/pi":>8} {"#zeros<L":>9} {"#zeros<2L":>9} {"floor(2L)":>9}')
    print('  ' + '-' * 82)

    for lam_sq_test in [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]:
        L_t = np.log(lam_sq_test)
        N_t = max(15, round(6 * L_t))
        dim_t = 2 * N_t + 1

        _, M_t, _ = build_all_fast(lam_sq_test, N_t)
        evals_t = np.linalg.eigvalsh(M_t)

        # Count signal (|eig| > 0.01) including top positive
        n_signal_t = np.sum(np.abs(evals_t) > 0.01)

        # Count zeros below various thresholds
        n_zeros_below_L = sum(1 for g in zeros if g < L_t)
        n_zeros_below_2L = sum(1 for g in zeros if g < 2 * L_t)
        shannon = 2 * L_t / np.pi

        print(f'  {lam_sq_test:>8d} {L_t:>8.3f} {N_t:>4d} {dim_t:>5d} {n_signal_t:>8d} '
              f'{shannon:>8.2f} {n_zeros_below_L:>9d} {n_zeros_below_2L:>9d} '
              f'{int(np.floor(2*L_t)):>9d}')
    sys.stdout.flush()

    # Refined: use multiple thresholds to see if signal dim is robust
    print(f'\n  Signal dim sensitivity to threshold (lam^2=1000):')
    lam_sq = 1000
    _, M_t, _ = build_all_fast(lam_sq, max(15, round(6 * np.log(lam_sq))))
    evals_t = np.linalg.eigvalsh(M_t)
    for thresh in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8]:
        n_sig = np.sum(np.abs(evals_t) > thresh)
        print(f'    thresh={thresh:.0e}: {n_sig} signal dims')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 5: Coupling anatomy in signal basis
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 5: COUPLING ANATOMY IN SIGNAL BASIS')
    print(f'{"="*76}\n')

    # At Schur step 0 of M_odd: a_1, c, B.
    # coupling = c^T (-B^{-1}) c = sum_k (c.u_k)^2 / |lam_k|
    # Decompose this sum by signal vs null eigenvectors of M (not B).

    lam_sq = 1000
    Lp, Dd, M, QW, N, L, dim, ns, B_n = decompose_exact(lam_sq)
    Mo, P_odd = odd_block(M, N)

    a1 = Mo[0, 0]
    c = Mo[0, 1:]
    B = Mo[1:, 1:]

    B_evals, B_evecs = np.linalg.eigh(B)
    coupling_total = 0
    coupling_by_component = []

    for k in range(len(B_evals)):
        proj = float(c @ B_evecs[:, k])
        contrib = -proj**2 / B_evals[k]  # positive since B_evals < 0
        coupling_total += contrib
        coupling_by_component.append((B_evals[k], proj, contrib))

    margin = abs(a1) - coupling_total
    print(f'  M_odd Schur step 0 (lam^2={lam_sq}):')
    print(f'    a_1 = {a1:+.8f}')
    print(f'    coupling = {coupling_total:.8f}')
    print(f'    margin = {margin:.4e}')
    print(f'    ratio = {coupling_total / abs(a1):.10f}')
    print()

    # Sort by contribution magnitude
    coupling_by_component.sort(key=lambda x: -abs(x[2]))

    print(f'  Top 20 coupling contributions (sorted by |contribution|):')
    print(f'  {"rank":>5} {"B_eig":>14} {"projection":>14} {"contribution":>14} {"cum%":>8}')
    print('  ' + '-' * 60)

    cum = 0
    for rank, (be, proj, cont) in enumerate(coupling_by_component[:20]):
        cum += cont
        cum_pct = cum / coupling_total * 100
        print(f'  {rank+1:>5d} {be:>+14.6e} {proj:>14.6e} {cont:>14.6e} {cum_pct:>8.2f}%')

    # Now: which of these B-eigenvectors live in signal space vs null space?
    # Project B-eigenvectors back to full M space via P_odd
    print(f'\n  Signal vs null decomposition of top coupling components:')
    print(f'  {"rank":>5} {"contribution":>14} {"signal_frac":>12} {"null_frac":>12}')
    print('  ' + '-' * 48)

    groups = get_spectral_groups(M)
    V_signal_full = groups['evecs'][:, np.concatenate([groups['bulk_idx'], groups['top_idx']])]
    V_null_full = groups['evecs'][:, groups['nz_idx']]

    for rank in range(min(10, len(coupling_by_component))):
        be, proj, cont = coupling_by_component[rank]
        # B-eigenvector in odd subspace -> lift to full space
        b_vec_odd = B_evecs[:, np.argsort(B_evals)[rank]]  # sorted by eigenvalue
        b_vec_odd = coupling_by_component[rank]  # actually need the right index
        # Redo: get the right eigenvector
        idx_sorted = np.argsort(-np.abs([x[2] for x in coupling_by_component]))

    # Simpler approach: just decompose c itself into signal vs null
    c_full = P_odd[:, 0]  # the n=1 odd basis vector in full space
    # Wait, c is the first row of Mo minus (0,0), which is P_odd^T @ M @ P_odd
    # The coupling vector c lives in the (N-1)-dim B subspace of M_odd
    # Lift c to full dim: c_lifted = P_odd[:,1:] @ c (skip the n=1 column)
    c_lifted = P_odd[:, 1:] @ c

    c_signal = V_signal_full.T @ c_lifted
    c_null = V_null_full.T @ c_lifted
    print(f'\n  Coupling vector c decomposition:')
    print(f'    ||c|| = {np.linalg.norm(c_lifted):.6f}')
    print(f'    signal component: {np.linalg.norm(c_signal):.6f} ({np.linalg.norm(c_signal)**2/np.linalg.norm(c_lifted)**2*100:.2f}%)')
    print(f'    null component:   {np.linalg.norm(c_null):.6f} ({np.linalg.norm(c_null)**2/np.linalg.norm(c_lifted)**2*100:.2f}%)')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 6: Signal vs null RH anatomy
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 6: SIGNAL vs NULL RH ANATOMY')
    print(f'{"="*76}\n')

    # On the null space: L and D cancel to 0.077% -- easy, M ~ 0.
    # On the signal space: M is NOT near-zero. What makes it work?

    V_sig = evecs[:, signal_idx]
    V_nul = evecs[:, nz_idx]

    # M restricted to signal space
    M_sig = V_sig.T @ M @ V_sig
    L_sig = V_sig.T @ Lp @ V_sig
    D_sig = V_sig.T @ np.diag(Dd) @ V_sig

    # M restricted to null space
    M_nul = V_nul.T @ M @ V_nul
    L_nul = V_nul.T @ Lp @ V_nul
    D_nul = V_nul.T @ np.diag(Dd) @ V_nul

    print(f'  SIGNAL SPACE ({n_signal} dims):')
    print(f'    ||M_sig|| = {np.linalg.norm(M_sig):.4f}')
    print(f'    ||L_sig|| = {np.linalg.norm(L_sig):.4f}')
    print(f'    ||D_sig|| = {np.linalg.norm(D_sig):.4f}')
    print(f'    cancel = ||L+D|| / ||L|| = {np.linalg.norm(L_sig + D_sig)/np.linalg.norm(L_sig):.6f}')
    print(f'    tr(M_sig) = {np.trace(M_sig):.6f}')
    print(f'    tr(L_sig) = {np.trace(L_sig):.6f}')
    print(f'    tr(D_sig) = {np.trace(D_sig):.6f}')
    eig_sig = np.linalg.eigvalsh(M_sig)
    print(f'    eigenvalues: [{eig_sig.min():.4f}, {eig_sig.max():.4f}]')
    print(f'    #pos = {np.sum(eig_sig > 1e-10)}, #neg = {np.sum(eig_sig < -1e-10)}')

    print(f'\n  NULL SPACE ({n_null} dims):')
    print(f'    ||M_nul|| = {np.linalg.norm(M_nul):.6e}')
    print(f'    ||L_nul|| = {np.linalg.norm(L_nul):.4f}')
    print(f'    ||D_nul|| = {np.linalg.norm(D_nul):.4f}')
    print(f'    cancel = ||L+D|| / ||L|| = {np.linalg.norm(L_nul + D_nul)/np.linalg.norm(L_nul):.6e}')
    eig_nul = np.linalg.eigvalsh(M_nul)
    print(f'    eigenvalues: [{eig_nul.min():.4e}, {eig_nul.max():.4e}]')

    # Cross-coupling between signal and null
    M_cross = V_sig.T @ M @ V_nul
    print(f'\n  CROSS-COUPLING (signal-null):')
    print(f'    ||M_cross|| = {np.linalg.norm(M_cross):.6e}')
    print(f'    max |entry| = {np.max(np.abs(M_cross)):.6e}')
    print(f'    (should be near-zero since these are eigenvector subspaces)')
    sys.stdout.flush()

    # Signal space by parity
    print(f'\n  Signal space by parity:')
    Mo_full, P_o = odd_block(M, N)
    Me_full, P_e = even_block(M, N)

    # Odd signal: eigenvalues of Mo that are "large"
    eo = np.linalg.eigvalsh(Mo_full)
    n_sig_odd = np.sum(np.abs(eo) > 0.01)
    n_nul_odd = np.sum(np.abs(eo) < 0.01)

    ee = np.linalg.eigvalsh(Me_full)
    n_sig_even = np.sum(np.abs(ee) > 0.01)
    n_nul_even = np.sum(np.abs(ee) < 0.01)

    print(f'    Odd:  dim={N}, signal={n_sig_odd}, null={n_nul_odd}')
    print(f'    Even: dim={N+1}, signal={n_sig_even}, null={n_nul_even}')
    print(f'    Total signal: {n_sig_odd + n_sig_even} (check vs {n_signal})')

    # What's the STRUCTURE of M on the odd signal subspace?
    eo_all, ev_o = np.linalg.eigh(Mo_full)
    sig_mask_o = np.abs(eo_all) > 0.01
    V_sig_o = ev_o[:, sig_mask_o]
    Mo_sig = V_sig_o.T @ Mo_full @ V_sig_o

    Lo, _ = odd_block(Lp, N)
    Do_diag = np.diag(odd_block(np.diag(Dd), N)[0])
    Lo_sig = V_sig_o.T @ Lo @ V_sig_o
    Do_sig = V_sig_o.T @ np.diag(Do_diag) @ V_sig_o

    print(f'\n  M_odd on signal subspace ({n_sig_odd} dims):')
    print(f'    eigenvalues: [{eo_all[sig_mask_o].min():.4f}, {eo_all[sig_mask_o].max():.4f}]')
    print(f'    all negative: {np.all(eo_all[sig_mask_o] < 0)}')
    print(f'    cancel L+D: {np.linalg.norm(Lo_sig + Do_sig)/np.linalg.norm(Lo_sig):.6f}')
    print(f'    tr(L_sig) = {np.trace(Lo_sig):.4f}')
    print(f'    tr(D_sig) = {np.trace(Do_sig):.4f}')
    print(f'    tr(M_sig) = {np.trace(Mo_sig):.4f}')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 6b: Lambda scaling of signal-space eigenvalue structure
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 6b: SIGNAL EIGENVALUE SCALING WITH LAMBDA')
    print(f'{"="*76}\n')

    print(f'  {"lam^2":>8} {"L":>8} {"#sig":>5} {"eig_min":>12} {"eig_max":>12} '
          f'{"|eig_2/eig_1|":>14} {"tr(M_sig)":>12}')
    print('  ' + '-' * 78)

    for lam_sq_test in [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]:
        L_t = np.log(lam_sq_test)
        N_t = max(15, round(6 * L_t))
        _, M_t, _ = build_all_fast(lam_sq_test, N_t)
        evals_t = np.linalg.eigvalsh(M_t)

        sig_mask = np.abs(evals_t) > 0.01
        n_sig = np.sum(sig_mask)
        sig_evals = evals_t[sig_mask]

        if len(sig_evals) >= 2:
            eig_min = sig_evals.min()
            eig_max = sig_evals.max()
            # Second largest by magnitude
            sorted_by_mag = sorted(sig_evals, key=abs, reverse=True)
            ratio_21 = abs(sorted_by_mag[1]) / abs(sorted_by_mag[0]) if abs(sorted_by_mag[0]) > 0 else 0
            tr_sig = sig_evals.sum()
        else:
            eig_min = eig_max = sig_evals[0] if len(sig_evals) == 1 else 0
            ratio_21 = 0
            tr_sig = sig_evals.sum() if len(sig_evals) > 0 else 0

        print(f'  {lam_sq_test:>8d} {L_t:>8.3f} {n_sig:>5d} {eig_min:>+12.4f} '
              f'{eig_max:>+12.4f} {ratio_21:>14.6e} {tr_sig:>+12.4f}')
    sys.stdout.flush()

    # ======================================================================
    # VERDICT
    # ======================================================================
    print()
    print('=' * 76)
    print('  SESSION 76 VERDICT')
    print('=' * 76)
    print()
    print('  1. Signal dim at standard N=6*L is ~16-17, CONSTANT across lambda.')
    print('     BUT: 76b/76c showed this is an ARTIFACT of the N=6*L choice.')
    print('     At N=150, signal dim = 98 (not 17). True picture: compact operator')
    print('     with continuous spectral band filling in as truncation grows.')
    print()
    print('  2. WHAT IS CONVERGED (from 76c):')
    print('     - eig_max = +38.46 (1 positive eigenvalue, stable to all N)')
    print('     - eig_2 = -24.6 (stable)')
    print('     - 73 near-zero eigenvalues in [1e-8, 1e-4]: the TRUE spectral mirror')
    print('     - Lorentzian property: ROBUST across all tested N')
    print()
    print('  3. Zeta zeros do NOT efficiently span signal space:')
    print('     Each zero contributes ~2% uniformly. Need ~50 zeros for full span.')
    print('     Signal eigenvectors are superpositions of many zero modes.')
    print()
    print('  4. Coupling vector is 52% signal, 48% null -- lives in BOTH subspaces.')
    print('     97.3% of coupling from one B-eigenvector (confirms S62).')
    print()
    print('  5. On signal space: L and D REINFORCE (cancel ratio 1.98)')
    print('     On null space: L and D CANCEL (cancel ratio 7.7e-4)')
    print('     The mirror is real but applies to a FIXED 73-dim subspace,')
    print('     not to "80% of the spectrum" as previously stated.')
    print()
    print('  6. The N-scaling is: signal ~ 0.12 * N^1.33 (power law)')
    print('     Fraction of dim: 15% at N=20, 33% at N=150 (growing)')
    print()
    print('  BOTTOM LINE: The "15-dim signal space" was a truncation artifact.')
    print('  The REAL structural invariants are:')
    print('    (a) Exactly 1 positive eigenvalue (Lorentzian), converged')
    print('    (b) 73 near-zero eigenvalues (spectral mirror), converged')
    print('    (c) A growing bulk negative band [-10,-1] (continuous spectrum)')
    print()


if __name__ == '__main__':
    run()
