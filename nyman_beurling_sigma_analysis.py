"""
Session 28b: Analyze what controls sigma_min of the NB Gram matrix.

Key insight from the Mellin derivation:
  f_hat_j(s) = 1/(j(s-1)) - j^{-s}*zeta(s)/s = A_j(s) + B_j(s)

  G = G^{AA} + G^{BB} + G^{cross}
  G^{AA}_{jk} = 1/(jk) = rank-1 matrix (outer product of w = (1/j))

On the subspace w_perp = {c : sum c_j/j = 0}, G^{AA} vanishes, so:
  sigma_min(G) = sigma_min(G^{BB} + G^{cross}) restricted to w_perp

Questions:
1. Is the min eigenvector in w_perp? (i.e., does sum (v_min)_j / j = 0?)
2. What are |D(1/2)|, |D(1/2+i*gamma_1)|, etc. for the min eigvec?
3. Can we compute G^{BB} and G^{cross} separately?
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, zeta, pi, nstr, power
import time

mp.dps = 30


def build_gram(N, n_grid=500000):
    x = np.linspace(1.0/n_grid, 1.0, n_grid); dx = x[1]-x[0]
    fp = np.zeros((N, n_grid))
    for k in range(1, N+1):
        v = 1.0/(k*x); fp[k-1] = v - np.floor(v)
    return (fp @ fp.T) * dx


def f_hat(j, s):
    """f_hat_j(s) = 1/(j(s-1)) - j^{-s} zeta(s)/s"""
    return 1/(mpf(j)*(s-1)) - power(mpf(j),-s)*zeta(s)/s


if __name__ == "__main__":
    print("SIGMA_MIN STRUCTURAL ANALYSIS")
    print("=" * 70)

    for N in [20, 50, 100]:
        t0 = time.time()
        G = build_gram(N, n_grid=500000)
        evals, evecs = np.linalg.eigh(G)
        v_min = evecs[:, 0]  # minimum eigenvector

        print(f"\nN = {N}, sigma_min = {evals[0]:.6e}")

        # 1. Is v_min in w_perp?
        w = np.array([1.0/j for j in range(1, N+1)])
        S = np.dot(v_min, w)  # sum v_j / j
        w_norm = np.linalg.norm(w)
        cos_angle = abs(S) / w_norm  # cosine of angle to w direction
        print(f"  S = sum(v_j/j) = {S:.6e}")
        print(f"  |cos(angle to w)| = {cos_angle:.6e}")
        print(f"  v_min {'IS' if cos_angle < 0.01 else 'is NOT'} in w_perp")

        # 2. Dirichlet polynomial at key points
        # D(s) = sum c_j j^{-s}
        for t_val in [0, 1, 5, 14.13, 25.01]:
            s = mpc(0.5, t_val)
            D_val = sum(v_min[j] * float(power(mpf(j+1), -s).real) +
                       1j * v_min[j] * float(power(mpf(j+1), -s).imag)
                       for j in range(N))
            print(f"  |D(1/2+i*{t_val:>5.2f})| = {abs(D_val):.6e}")

        # 3. G^{AA} contribution (rank-1): w * w^T
        G_AA = np.outer(w, w)  # 1/(jk)
        qf_AA = v_min @ G_AA @ v_min  # = S^2
        qf_total = v_min @ G @ v_min  # = sigma_min

        # G^{BB} + G^{cross} contribution
        qf_rest = qf_total - qf_AA

        print(f"  Quadratic form decomposition:")
        print(f"    total (sigma_min) = {qf_total:.6e}")
        print(f"    G^AA (pole-pole) = {qf_AA:.6e} (= S^2 = {S**2:.6e})")
        print(f"    G^BB + G^cross   = {qf_rest:.6e}")
        print(f"    Fraction from poles: {qf_AA/qf_total:.4f}")

        # 4. Structure of v_min: where is the weight?
        energy_low = np.sum(v_min[:N//4]**2)
        energy_mid = np.sum(v_min[N//4:3*N//4]**2)
        energy_high = np.sum(v_min[3*N//4:]**2)
        print(f"  v_min energy: low(j<N/4)={energy_low:.4f}, "
              f"mid={energy_mid:.4f}, high(j>3N/4)={energy_high:.4f}")

        # 5. Check: does v_min look like it's trying to make D*zeta small near t=0?
        # Sum c_j/sqrt(j) = D(1/2)
        D_half = np.dot(v_min, 1.0/np.sqrt(np.arange(1, N+1)))
        print(f"  D(1/2) = sum(v_j/sqrt(j)) = {D_half:.6e}")

        # 6. Eigenvalue spacing at bottom
        print(f"  Bottom 5 eigenvalues: {', '.join(f'{e:.4e}' for e in evals[:5])}")

        # 7. The key ratio: sigma_min * N^2
        print(f"  sigma_min * N^2 = {evals[0] * N**2:.6f}")

        print(f"  ({time.time()-t0:.1f}s)")

    print(f"\n{'='*70}")
    print("KEY INSIGHT:")
    print("  If v_min IS in w_perp (S=0): sigma_min = min of G^BB on w_perp")
    print("  Then sigma_min = min integral |D|^2 |zeta|^2 / (1/4+t^2) dt")
    print("  And the N^{-2} rate comes from the Dirichlet polynomial structure")
