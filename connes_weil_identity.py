"""
Session 30 iteration 10: Verify the Weil explicit formula identity.

The Weil explicit formula says (for test functions on [lambda^{-1}, lambda]):

Q_W(f,g) = sum_rho f_hat(rho) * conj(g_hat(rho_bar))

where the sum is over nontrivial zeros rho = 1/2 + i*gamma of zeta,
with appropriate multiplicity and sign conventions.

In the V_n basis: V_n(u) = u^{2*pi*i*n/L} on [lambda^{-1}, lambda].

So: (Q_W)_{nm} = sum_rho V_n_hat(rho) * conj(V_m_hat(rho))

where V_n_hat(rho) is the Mellin transform of V_n at rho.

For V_n on [lambda^{-1}, lambda]:
V_n_hat(s) = integral_{lambda^{-1}}^{lambda} u^{2*pi*i*n/L} * u^{-s} du/u
           = integral_{-L/2}^{L/2} e^{(2*pi*i*n/L - s)*t} dt
           = L * sinc((s - 2*pi*n/L) * L / (2*pi))

Wait, more carefully:
= [e^{(2*pi*i*n/L - s)*t}]_{-L/2}^{L/2} / (2*pi*i*n/L - s)
= 2*sinh((2*pi*i*n/L - s)*L/2) / (2*pi*i*n/L - s)

At s = 1/2 + i*gamma:
= 2*sinh((2*pi*i*n/L - 1/2 - i*gamma)*L/2) / (2*pi*i*n/L - 1/2 - i*gamma)

This is exact. Let me compute it and build Q_W from the zero sum.
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, log, pi, sinh, nstr, zeta)
import time
import sys
sys.path.insert(0, '.')
from connes_h1h2_correct import build_QW

mp.dps = 30


def V_hat(n, s, L_f):
    """Mellin transform of V_n(u) = u^{2*pi*i*n/L} on [lambda^{-1}, lambda]."""
    freq = 2j * np.pi * n / L_f
    arg = freq - complex(s)
    if abs(arg) < 1e-14:
        return complex(L_f)  # limit as arg -> 0
    return complex(2 * np.sinh(arg * L_f / 2) / arg)


def build_zero_sum_weighted(N_val, L_f, gammas, n_zeros):
    """Build Q_W from the Weil explicit formula zero sum.

    Q_W_nm = sum_rho V_n_hat(rho) * conj(V_m_hat(rho))

    where rho = 1/2 + i*gamma (counting multiplicities).
    Each pair (rho, rho_bar) contributes 2*Re terms.
    """
    dim = 2 * N_val + 1
    Z = np.zeros((dim, dim))

    for k in range(n_zeros):
        gamma = gammas[k]
        rho = 0.5 + 1j * gamma

        # Compute V_hat at rho for all n
        v = np.array([V_hat(n, rho, L_f) for n in range(-N_val, N_val + 1)])

        # Contribution: v * v^H (complex outer product, take real part)
        # Both rho and conj(rho) = 1/2 - i*gamma contribute
        Z += np.real(np.outer(v, np.conj(v)))

    return Z


if __name__ == "__main__":
    print("WEIL EXPLICIT FORMULA IDENTITY VERIFICATION")
    print("=" * 70)

    gammas = np.load("_zeros_500.npy")

    for lam_sq in [14, 50]:
        L_f = np.log(lam_sq)
        N = round(8 * L_f)
        dim = 2 * N + 1

        print(f"\nlam^2={lam_sq}, N={N}, dim={dim}")
        print("-" * 50)

        # Build Q_W the standard way
        t0 = time.time()
        QW = build_QW(lam_sq, N)
        dt1 = time.time() - t0
        evals_qw = np.linalg.eigvalsh(QW)
        print(f"  Q_W built ({dt1:.0f}s): eps_0={evals_qw[0]:.4e}, eps_max={evals_qw[-1]:.4e}")

        # Build from zero sum with increasing number of zeros
        for n_z in [5, 10, 20, 50, 100, 200, 500]:
            if n_z > len(gammas):
                break
            Z = build_zero_sum_weighted(N, L_f, gammas, n_z)

            # Compare Z with Q_W
            diff = QW - Z
            rel_err = np.linalg.norm(diff) / np.linalg.norm(QW)

            # Eigenvalue comparison
            evals_z = np.linalg.eigvalsh(Z)
            ev_diff = np.max(np.abs(np.sort(evals_qw) - np.sort(evals_z)))

            # Rank of Z
            _, sv_z, _ = np.linalg.svd(Z)
            rank_z = np.sum(sv_z > sv_z[0] * 1e-4)

            print(f"  Z({n_z:>3} zeros): rel_err={rel_err:.4e}, "
                  f"max_ev_diff={ev_diff:.4e}, rank(Z)={rank_z:>3}")

        # Also check: does Q_W - Z converge as n_z grows?
        print(f"\n  Convergence of Q_W - Z_{n_z}:")
        prev_norm = None
        for n_z in [10, 50, 100, 200, 500]:
            if n_z > len(gammas):
                break
            Z = build_zero_sum_weighted(N, L_f, gammas, n_z)
            norm_diff = np.linalg.norm(QW - Z, 'fro')
            ratio = norm_diff / prev_norm if prev_norm else 0
            prev_norm = norm_diff
            print(f"    n_z={n_z:>3}: ||Q_W - Z|| = {norm_diff:.6e}, "
                  f"ratio to prev = {ratio:.4f}")

    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print("=" * 70)
    print("""
If Q_W = Z (zero sum): the Weil explicit formula is verified.
  => rank(Q_W) = rank(Z) <= number of zeros in effective bandwidth
  => The bounded rank follows from the zero counting function

If Q_W != Z: the proper Weil formula has additional terms (poles, gamma factors)
  that need to be included.

The CONVERGENCE RATE of Q_W - Z tells us how many zeros are needed
to capture Q_W. Fast convergence = Q_W is well-approximated by
few zeros = bounded effective rank.
""")
