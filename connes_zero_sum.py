"""
Session 30 iteration 4: Q_W = -(zero sum) from the Weil explicit formula.

THE KEY RELATION:
  Q_W(f, g) = -sum_rho f_hat(rho) * conj(g_hat(rho))

where the sum is over nontrivial zeros rho of zeta (with appropriate signs/weights).

This means Q_W is a NEGATIVE of a sampling operator at the zeta zeros.
By Slepian concentration: sampling at ~D points in bandwidth B gives
effective rank ~B (the time-bandwidth product).

VERIFY: compute the zero sum Z directly and compare with Q_W.
If they match, Gap 1 is closed by the Slepian theorem.
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, log, pi, zeta, exp, nstr)
import time

mp.dps = 30


def build_zero_sum(N_val, L_f, gammas, n_zeros_use):
    """Build Z_{nm} = sum_rho V_n(rho) * conj(V_m(rho))

    In the multiplicative Fourier basis V_n(u) = u^{2*pi*i*n/L},
    evaluated at rho = 1/2 + i*gamma_k:

    V_n(rho) involves the Mellin transform evaluated at rho.
    For our finite basis on [lambda^{-1}, lambda]:

    The contribution of zero rho to Q_W is related to:
    phi_k(n) = exp(2*pi*i*n*gamma_k/L) (the "frequency" at the zero)

    Actually, for the Weil explicit formula on the V_n basis:
    Q_W ~ -sum_k |phi_k><phi_k| (rank-1 per zero pair)

    But with proper weights. Let me compute it directly.
    """
    dim = 2*N_val + 1

    # Each zero gamma_k contributes a rank-2 term (cos + sin):
    # phi_k(n) = exp(2*pi*i*n*gamma_k/L)
    # Z += w_k * |phi_k><phi_k*|

    # Build the zero sum matrix
    Z = np.zeros((dim, dim))

    for k in range(n_zeros_use):
        gamma = gammas[k]
        # Vector: phi_k(n) = exp(2*pi*i*n*gamma/L) for n = -N,...,N
        phi = np.array([np.exp(2j*np.pi*n*gamma/L_f) for n in range(-N_val, N_val+1)])

        # Contribution: phi * phi^H (rank 1 in complex, rank 2 in real)
        # Weight: 1 (for now; proper weight from explicit formula needed)
        Z += np.real(np.outer(phi, np.conj(phi)))

    return Z


if __name__ == "__main__":
    print("Q_W AS ZERO SUM — VERIFICATION")
    print("=" * 70)

    gammas = np.load("_zeros_500.npy")

    # ================================================================
    # PART 1: Build Q_W and the zero sum, compare
    # ================================================================
    print("\nPART 1: ZERO SUM MATRIX vs Q_W")
    print("-" * 70)

    # For small lambda, build Q_W the standard way and compare with zero sum
    from connes_h1h2_correct import build_QW

    for lam_sq in [14, 50]:
        L_f = np.log(lam_sq)
        N = round(8 * L_f)
        dim = 2*N + 1
        bw = np.pi * N / L_f

        t0 = time.time()
        QW = build_QW(lam_sq, N)
        dt = time.time() - t0

        evals_qw = np.linalg.eigvalsh(QW)

        # Zero sum with different number of zeros
        for n_z in [2, 5, 10, 20, 50, 100]:
            Z = build_zero_sum(N, L_f, gammas, n_z)

            # Compare -Z with Q_W
            # They won't match exactly (weights, pole terms, etc.)
            # But the RANK STRUCTURE should be similar

            _, sv_z, _ = np.linalg.svd(Z)
            rank_z = np.sum(sv_z > sv_z[0] * 1e-4)

            # Correlation between Q_W and -Z eigenspaces
            evals_z, evecs_z = np.linalg.eigh(-Z)

            # Overlap of Q_W min eigenvector with Z eigenvectors
            xi_qw = np.linalg.eigh(QW)[1][:, 0]
            overlaps = np.abs(evecs_z.T @ xi_qw)
            max_overlap = np.max(overlaps)

            if n_z in [2, 10, 50, 100]:
                print(f"  lam^2={lam_sq}, N={N}, {n_z} zeros: rank(Z)={rank_z}, "
                      f"max_overlap(xi_QW, Z_eigvecs)={max_overlap:.4f}")

    # ================================================================
    # PART 2: Rank of the zero sum vs bandwidth
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 2: RANK OF ZERO SUM Z vs BANDWIDTH")
    print("-" * 70)

    print(f"{'N':>4} {'dim':>5} {'BW':>6} {'n_zeros':>8} {'rank(Z)':>8} {'BW/rank':>8}")
    print("-" * 45)

    for N in [15, 20, 25, 30, 40, 50]:
        L_f = np.log(200)  # fixed lambda
        dim = 2*N + 1
        bw = np.pi * N / L_f

        Z = build_zero_sum(N, L_f, gammas, 100)  # use 100 zeros
        _, sv_z, _ = np.linalg.svd(Z)
        rank_z = np.sum(sv_z > sv_z[0] * 1e-4)

        print(f"{N:>4} {dim:>5} {bw:>6.1f} {100:>8} {rank_z:>8} {bw/rank_z:>8.2f}")

    # ================================================================
    # PART 3: The 26x26 signal-subspace Q_W — the secular equation
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 3: SIGNAL SUBSPACE Q_W (26x26)")
    print("-" * 70)

    for lam_sq in [50, 200, 1000]:
        L_f = np.log(lam_sq)
        N = round(8 * L_f)
        dim = 2*N + 1

        t0 = time.time()

        # Build M and decompose
        from connes_crossterm import build_all
        W02, M, QW = build_all(lam_sq, N)

        evals_m, evecs_m = np.linalg.eigh(M)
        abs_evals = np.abs(evals_m)
        threshold = np.max(abs_evals) * 1e-4
        signal_idx = np.where(abs_evals >= threshold)[0]
        P_signal = evecs_m[:, signal_idx]

        # Q_W restricted to signal subspace
        QW_signal = P_signal.T @ QW @ P_signal
        evals_qw_sig = np.linalg.eigvalsh(QW_signal)

        # Full Q_W for comparison
        evals_qw_full = np.linalg.eigvalsh(QW)

        n_sig = len(signal_idx)
        print(f"\n  lam^2={lam_sq}, signal dim={n_sig} ({time.time()-t0:.0f}s):")
        print(f"    QW|signal eigenvalues (bottom 5): {', '.join(f'{e:.4e}' for e in evals_qw_sig[:5])}")
        print(f"    QW|signal eigenvalues (top 5):    {', '.join(f'{e:.4e}' for e in evals_qw_sig[-5:])}")
        print(f"    Full QW eps_0 = {evals_qw_full[0]:.6e}")
        print(f"    min(QW|signal) = {evals_qw_sig[0]:.6e}")
        print(f"    min(QW|signal) / eps_0 = {evals_qw_sig[0]/evals_qw_full[0]:.2f}")

        # Is min(QW|signal) = eps_0?
        # If yes: the signal subspace captures the minimum eigenvalue exactly.
        # If no: the full optimization mixes signal and null to get something smaller.

    print(f"\n{'='*70}")
    print("SYNTHESIS")
    print("=" * 70)
