"""
SESSION 38b — THE SILENT MODE IDENTITY AND ITS GENERALIZATION

On silent modes: M*phi = 0 exactly. The explicit formula forces
M_diag + M_alpha + M_prime = 0 on these directions.

On seeing modes: M*phi < 0. The explicit formula gives
M*phi = -(spectral sum over zeros) < 0.

QUESTION: What algebraic property of M forces:
  (a) Silent modes: M = 0 (kernel of Mellin transform at zeros)
  (b) Seeing modes: M < 0 (Mellin transform hits zeros, giving positive Q_W)
  (c) NO positive eigenvalues

APPROACH:
1. Characterize the silent modes explicitly — what are they in the Fourier basis?
2. Characterize the seeing modes — how do they relate to zeta zeros?
3. Look for an algebraic decomposition: M = -G^H * G + (something zero on null(W02))
   If such a decomposition exists, M <= 0 on null(W02) follows algebraically.
4. The matrix G would be the "zero-hitting map" — its rows indexed by zeros,
   columns by null(W02) basis, entries = Mellin transform values.
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, log, pi, exp, cos, sin, euler, digamma, hyp2f1, sinh, zetazero
import time
import sys
sys.path.insert(0, '.')
from connes_crossterm import build_all


def characterize_silent_modes(lam_sq, N=None, n_zeros=30):
    """
    Silent modes are eigenvectors of M|null(W02) with eigenvalue = 0.
    These have Mellin transform = 0 at every zeta zero in the bandwidth.

    Characterize them: what do they look like in the Fourier basis?
    """
    if N is None:
        L_f = np.log(lam_sq)
        N = max(15, round(6 * L_f))
    dim = 2 * N + 1
    L_f = np.log(lam_sq)
    ns = np.arange(-N, N + 1, dtype=float)

    W02, M, QW = build_all(lam_sq, N, n_quad=10000)

    ew, ev = np.linalg.eigh(W02)
    thresh = np.max(np.abs(ew)) * 1e-10
    P_null = ev[:, np.abs(ew) <= thresh]
    d_null = P_null.shape[1]

    M_null = P_null.T @ M @ P_null
    evals, evecs = np.linalg.eigh(M_null)

    # Split into seeing (|eig| > 0.001) and silent (|eig| < 0.001)
    seeing_idx = np.where(np.abs(evals) > 0.001)[0]
    silent_idx = np.where(np.abs(evals) <= 0.001)[0]

    print(f"SILENT MODE CHARACTERIZATION: lam^2={lam_sq}, dim={dim}", flush=True)
    print(f"  null(W02) dim: {d_null}", flush=True)
    print(f"  Seeing modes: {len(seeing_idx)} (eigenvalues {evals[seeing_idx[0]]:.4f} to {evals[seeing_idx[-1]]:.4f})" if len(seeing_idx) > 0 else "  Seeing: 0", flush=True)
    print(f"  Silent modes: {len(silent_idx)}", flush=True)

    # Compute Mellin transform at zeta zeros
    mp.dps = 30
    gammas = []
    for j in range(1, n_zeros + 1):
        z = zetazero(j)
        gammas.append(float(z.imag))

    def mellin_at_zero(phi_full, gamma):
        """Compute integral_0^L g(x) * exp(i*gamma*x) dx where g = sum phi_n omega_n."""
        n_pts = 2000
        dx = L_f / n_pts
        result = 0.0 + 0.0j
        for k in range(n_pts):
            x = dx * (k + 0.5)
            g_x = sum(phi_full[idx] * 2 * (1 - x/L_f) * np.cos(2*np.pi*ns[idx]*x/L_f)
                      for idx in range(dim))
            result += g_x * np.exp(1j * gamma * x) * dx
        return result

    # Build the ZERO-HITTING MATRIX G
    # G[j, k] = Mellin transform of k-th null basis vector at j-th zero
    print(f"\n  Building zero-hitting matrix G ({n_zeros} zeros x {d_null} null modes)...", flush=True)

    G = np.zeros((n_zeros, d_null), dtype=complex)
    for k in range(d_null):
        phi_full = P_null[:, k]
        for j, gamma in enumerate(gammas):
            G[j, k] = mellin_at_zero(phi_full, gamma)

    # The spectral decomposition should give:
    # M_null = -G^H * G  (on null(W02), up to corrections from high zeros)
    GHG = np.real(G.conj().T @ G)

    # Compare -G^H*G with M_null
    print(f"\n  Comparing M_null with -G^H*G:", flush=True)
    print(f"  ||M_null||_F = {np.linalg.norm(M_null, 'fro'):.4f}", flush=True)
    print(f"  ||G^H*G||_F = {np.linalg.norm(GHG, 'fro'):.4f}", flush=True)
    diff = M_null + GHG  # Should be near zero if M = -G^H*G
    print(f"  ||M_null + G^H*G||_F = {np.linalg.norm(diff, 'fro'):.6f}", flush=True)
    rel_err = np.linalg.norm(diff, 'fro') / np.linalg.norm(M_null, 'fro')
    print(f"  Relative error: {rel_err:.6f} ({rel_err*100:.2f}%)", flush=True)

    # Eigenvalues of -G^H*G (should be <= 0)
    evals_GHG = np.linalg.eigvalsh(-GHG)
    print(f"\n  Eigenvalues of -G^H*G: [{np.min(evals_GHG):.4f}, {np.max(evals_GHG):.6e}]", flush=True)
    print(f"  -G^H*G is NSD: {np.max(evals_GHG) < 1e-6}", flush=True)
    print(f"  (This is AUTOMATIC: G^H*G is PSD, so -G^H*G is NSD)", flush=True)

    # How well does -G^H*G approximate M_null?
    # The seeing modes should be captured; the silent modes should be in ker(G)
    evals_M_sorted = np.sort(evals)
    evals_GHG_sorted = np.sort(evals_GHG)

    print(f"\n  Eigenvalue comparison (sorted):", flush=True)
    print(f"  {'idx':>4} {'eig(M_null)':>12} {'eig(-G^H*G)':>12} {'diff':>12}", flush=True)
    for i in range(min(20, d_null)):
        print(f"  {i:>4} {evals_M_sorted[i]:>+12.4f} {evals_GHG_sorted[i]:>+12.4f} "
              f"{evals_M_sorted[i]-evals_GHG_sorted[i]:>+12.4e}", flush=True)
    if d_null > 25:
        print(f"  ...", flush=True)
    for i in range(max(d_null-5, 20), d_null):
        print(f"  {i:>4} {evals_M_sorted[i]:>+12.6e} {evals_GHG_sorted[i]:>+12.6e} "
              f"{evals_M_sorted[i]-evals_GHG_sorted[i]:>+12.4e}", flush=True)

    # Rank of G
    sv_G = np.linalg.svd(G, compute_uv=False)
    rank_G = np.sum(sv_G > 1e-10 * sv_G[0])
    print(f"\n  Rank of G: {rank_G} (out of min({n_zeros}, {d_null}) = {min(n_zeros, d_null)})", flush=True)
    print(f"  Top 10 singular values: {', '.join(f'{s:.4f}' for s in sv_G[:10])}", flush=True)

    # The NULL SPACE of G = the silent modes
    null_G_dim = d_null - rank_G
    print(f"  dim(ker(G)) = {null_G_dim} (should match #silent = {len(silent_idx)})", flush=True)

    return G, evals, evecs, gammas


def test_decomposition_more_zeros(lam_sq, N=None):
    """
    If M_null = -G^H*G exactly, then adding more zeros to G should
    improve the approximation. Test with increasing numbers of zeros.
    """
    if N is None:
        L_f = np.log(lam_sq)
        N = max(15, round(6 * L_f))
    dim = 2 * N + 1
    L_f = np.log(lam_sq)
    ns = np.arange(-N, N + 1, dtype=float)

    W02, M, QW = build_all(lam_sq, N, n_quad=10000)

    ew, ev = np.linalg.eigh(W02)
    thresh = np.max(np.abs(ew)) * 1e-10
    P_null = ev[:, np.abs(ew) <= thresh]
    d_null = P_null.shape[1]

    M_null = P_null.T @ M @ P_null
    M_norm = np.linalg.norm(M_null, 'fro')

    def mellin_at_zero(phi_full, gamma):
        n_pts = 2000
        dx = L_f / n_pts
        result = 0.0 + 0.0j
        for k in range(n_pts):
            x = dx * (k + 0.5)
            g_x = sum(phi_full[idx] * 2 * (1 - x/L_f) * np.cos(2*np.pi*ns[idx]*x/L_f)
                      for idx in range(dim))
            result += g_x * np.exp(1j * gamma * x) * dx
        return result

    mp.dps = 30
    print(f"\nDECOMPOSITION CONVERGENCE: M_null = -G^H*G", flush=True)
    print(f"  lam^2={lam_sq}, null_dim={d_null}", flush=True)
    print(f"  {'#zeros':>7} {'||M+G^HG||_F':>14} {'rel_error':>10} {'rank(G)':>8}", flush=True)

    all_gammas = []
    G_rows = []
    for j in range(1, 51):
        gamma = float(zetazero(j).imag)
        all_gammas.append(gamma)

        # Compute new row of G
        row = np.zeros(d_null, dtype=complex)
        for k in range(d_null):
            row[k] = mellin_at_zero(P_null[:, k], gamma)
        G_rows.append(row)

        if j in [1, 2, 3, 5, 8, 10, 15, 20, 30, 50]:
            G = np.array(G_rows)
            GHG = np.real(G.conj().T @ G)
            diff = M_null + GHG
            err = np.linalg.norm(diff, 'fro')
            rel = err / M_norm
            sv = np.linalg.svd(G, compute_uv=False)
            rank = np.sum(sv > 1e-10 * sv[0])
            print(f"  {j:>7} {err:>14.6f} {rel:>10.6f} {rank:>8}", flush=True)

    # THE KEY: if M_null = -G^H*G EXACTLY (with enough zeros),
    # then M_null is AUTOMATICALLY NSD (since G^H*G is PSD).
    # This would be a PROOF of Q_W >= 0 at this bandwidth!
    #
    # But does it converge? And does it require knowing the zeros
    # (which requires RH)?
    #
    # The decomposition M = -G^H*G uses zeros as the basis for G.
    # If we could construct G from PRIMES (not zeros), we'd have
    # a non-circular proof.

    return


def algebraic_structure_of_silent(lam_sq, N=None):
    """
    Study the ALGEBRAIC structure of silent modes more carefully.

    Silent modes satisfy: for all zeta zeros rho in the bandwidth,
    the Mellin transform of the test function vanishes at rho.

    This is a set of LINEAR constraints on the Fourier coefficients:
    sum_n phi_n * omega_n_hat(rho) = 0  for each rho

    The number of constraints = 2 * (number of zeros in window)
    (real and imaginary parts, or equivalently, rho and bar(rho))

    The silent space = null(W02) intersect (intersection of zero hyperplanes)
    Its dimension = null_dim - 2*(#zeros in window) (generically)

    THIS IS THE KEY STRUCTURE: the seeing modes live in a subspace
    whose dimension equals 2*(#zeros), and on this subspace,
    M = -G^H*G (the Gram matrix of Mellin values at zeros).

    The decomposition M = -G^H*G is EXACTLY the explicit formula
    written as a sum of squares!

    IF this decomposition accounts for ALL of M (no residual),
    then M <= 0 follows ALGEBRAICALLY.
    """
    if N is None:
        L_f = np.log(lam_sq)
        N = max(15, round(6 * L_f))
    dim = 2 * N + 1
    L_f = np.log(lam_sq)
    ns = np.arange(-N, N + 1, dtype=float)

    W02, M, QW = build_all(lam_sq, N, n_quad=10000)

    ew, ev = np.linalg.eigh(W02)
    thresh = np.max(np.abs(ew)) * 1e-10
    P_null = ev[:, np.abs(ew) <= thresh]
    d_null = P_null.shape[1]

    M_null = P_null.T @ M @ P_null

    print(f"\nALGEBRAIC STRUCTURE OF SILENT MODES: lam^2={lam_sq}", flush=True)
    print(f"  dim={dim}, null_dim={d_null}", flush=True)

    # Count eigenvalues by regime
    evals = np.linalg.eigvalsh(M_null)
    n_seeing = np.sum(np.abs(evals) > 0.001)
    n_silent = d_null - n_seeing

    print(f"  Seeing modes: {n_seeing}", flush=True)
    print(f"  Silent modes: {n_silent}", flush=True)
    print(f"  Implied #zero pairs: ~{n_seeing // 2}", flush=True)

    # The critical prediction: n_seeing should equal 2*(#zeros in effective window)
    # Effective window: zeros whose Mellin transform has significant magnitude
    # For band-limited functions of bandwidth L, significant means gamma < ~N*2*pi/L

    effective_height = N * 2 * np.pi / L_f
    estimated_zeros = effective_height * np.log(effective_height) / (2 * np.pi)

    print(f"\n  Effective height: {effective_height:.1f}", flush=True)
    print(f"  Estimated #zeros in window: {estimated_zeros:.1f}", flush=True)
    print(f"  Predicted seeing modes (2*#zeros): {2*estimated_zeros:.1f}", flush=True)
    print(f"  Actual seeing modes: {n_seeing}", flush=True)
    print(f"  Match: {'CLOSE' if abs(n_seeing - 2*estimated_zeros) < 5 else 'OFF'}", flush=True)

    # THE CRITICAL QUESTION:
    # Does M_null = -G^H*G with G = (Mellin values at zeros)?
    # If yes: M_null <= 0 is AUTOMATIC (Gram matrix is PSD).
    # If no: there's a residual, and the residual must also be NSD.
    #
    # The explicit formula says: <phi, M phi> = -sum_rho |g-hat(rho)|^2 + (high zeros)
    # The "high zeros" are zeros with gamma >> effective_height.
    # Their contribution MUST be non-positive (they're also |g-hat|^2 terms).
    # So the residual from truncating the zero sum is also NSD.
    #
    # This means: M_null = -G_finite^H * G_finite - R
    # where G_finite uses finitely many zeros, and R >= 0 (from high zeros).
    # Both terms are NSD, so M_null is NSD.
    #
    # BUT: the decomposition M = -sum_rho |g-hat(rho)|^2 ASSUMES
    # each term is |g-hat(rho)|^2 (not a cross-product).
    # This requires rho to be on the critical line = RH!

    print(f"\n  THE DECOMPOSITION:", flush=True)
    print(f"  M_null = -G^H*G  where G[j,k] = Mellin(null_vec_k, zero_j)", flush=True)
    print(f"  This is M = -(sum of squares), automatically NSD.", flush=True)
    print(f"", flush=True)
    print(f"  BUT: writing the spectral sum as sum of |g-hat(rho)|^2", flush=True)
    print(f"  requires rho = 1/2 + i*gamma (on critical line).", flush=True)
    print(f"  For off-line rho: the term is g-hat(rho)*conj(g-hat(1-bar(rho))),", flush=True)
    print(f"  which is NOT |...|^2 and can be negative.", flush=True)
    print(f"", flush=True)
    print(f"  SO: M = -G^H*G is equivalent to RH.", flush=True)
    print(f"  The sum-of-squares decomposition IS the hypothesis.", flush=True)
    print(f"", flush=True)
    print(f"  HOWEVER: we could try to find a DIFFERENT sum-of-squares", flush=True)
    print(f"  decomposition that doesn't use zero locations.", flush=True)
    print(f"  If M = -A^T*A for some A built from PRIMES (not zeros),", flush=True)
    print(f"  that would be a non-circular proof of M <= 0.", flush=True)

    # Can we find such an A from the prime structure?
    # M_prime = sum_{pk} w(pk) * T(pk)
    # Each T(pk) has rank ~2. Can we decompose M_prime into -B^T*B?
    # Only if M_prime <= 0 on null(W02), which we showed is FALSE
    # (M_prime has positive eigenvalues on null(W02)).
    # So M_prime alone doesn't factor as -B^T*B.
    # We need M_diag + M_alpha + M_prime = -(something PSD).

    # The diagonal part M_diag has positive entries (for small |n|).
    # So M_diag is NOT NSD.
    # M_alpha is small.
    # M_prime is indefinite on null(W02).
    # Only the SUM is NSD.

    # THE GENERALIZATION OF THE SILENT MODE IDENTITY:
    # On silent modes: M_diag + M_alpha + M_prime = 0
    # On seeing modes: M_diag + M_alpha + M_prime = -(positive)
    # On ALL modes:    M_diag + M_alpha + M_prime <= 0
    #
    # Can we prove the INEQUALITY from the EQUALITY?
    # The equality (on silent modes) says: the analytic and prime parts cancel.
    # The inequality (on seeing modes) says: the analytic part wins.
    #
    # But Session 35 showed the OPPOSITE: on seeing modes, M_prime is the
    # dominant negative contributor, not M_analytic.
    # The analytic part (M_diag + M_alpha) is POSITIVE on some seeing modes.
    # M_prime overcomes this positivity.
    #
    # So the seeing mode negativity comes from M_prime being MORE negative
    # than M_analytic is positive, in specific directions.

    return n_seeing, n_silent


if __name__ == "__main__":
    print("SESSION 38b — SILENT MODE IDENTITY GENERALIZATION", flush=True)
    print("=" * 80, flush=True)

    # 1. Characterize silent modes and build G matrix
    G, evals, evecs, gammas = characterize_silent_modes(50, n_zeros=30)

    # 2. Convergence of M = -G^H*G with increasing zeros
    test_decomposition_more_zeros(50)

    # 3. Algebraic structure
    for lam_sq in [50, 200]:
        algebraic_structure_of_silent(lam_sq)

    print(f"\nDone.", flush=True)
