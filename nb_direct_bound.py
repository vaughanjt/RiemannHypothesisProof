"""
Session 29b: Direct analytical bound for sigma_min via Mellin integral.

INSIGHT FROM PART 5: Mobius inversion in Mellin space FAILS because
the Dirichlet characters j^{-s} are linearly independent on Re(s)=1/2.
The Euler product pattern is a REAL-SPACE phenomenon, not Mellin.

NEW APPROACH: Bound sigma_min directly from the Mellin representation.

On w_perp (pole terms vanish), the quadratic form becomes:
  c^T G c = (1/2pi) integral |D(1/2+it)|^2 |zeta(1/2+it)|^2 / (1/4+t^2) dt

where D(s) = sum c_j j^{-s}, ||c|| = 1, sum c_j/j = 0.

KEY QUESTION: How small can this integral be?

APPROACH A: Split integral into regions and use Montgomery-Vaughan
APPROACH B: Use zeta lower bounds on specific intervals
APPROACH C: Halász-Montgomery method (large values of D)
APPROACH D: Use Parseval in both spaces to get a "sandwich"

Also: test whether sigma_min = dist_N^2 * f(N) for some simple f(N).
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, zeta, pi, power, nstr
import time
from math import gcd

mp.dps = 30


def build_gram(N, n_grid=500000):
    x = np.linspace(1.0/n_grid, 1.0, n_grid); dx = x[1]-x[0]
    fp = np.zeros((N, n_grid))
    for k in range(1, N+1):
        v = 1.0/(k*x); fp[k-1] = v - np.floor(v)
    return (fp @ fp.T) * dx


def f_hat(j, s):
    return 1/(mpf(j)*(s-1)) - power(mpf(j),-s)*zeta(s)/s


if __name__ == "__main__":
    print("SESSION 29b: DIRECT BOUND FOR SIGMA_MIN")
    print("=" * 70)

    # ================================================================
    # PART A: Detailed regional analysis of sigma_min integral
    # ================================================================
    print("\nPART A: Regional analysis of sigma_min contribution")
    print("-" * 70)

    for N in [30, 50, 100, 150, 200]:
        t0 = time.time()
        G = build_gram(N, n_grid=max(500000, N*5000))
        evals, evecs = np.linalg.eigh(G)
        v_min = evecs[:, 0]
        sigma_min = evals[0]

        # Verify on w_perp
        S = np.dot(v_min, 1.0/np.arange(1, N+1))

        # Compute integrand on fine grid of t
        t_max = 5 * N
        n_t = min(3000, max(1000, 10*N))
        t_pos = np.linspace(0.1, t_max, n_t)

        integrand_vals = np.zeros(n_t)
        D_sq_vals = np.zeros(n_t)
        zeta_sq_vals = np.zeros(n_t)
        weight_vals = np.zeros(n_t)

        for i, t in enumerate(t_pos):
            s = mpc(0.5, t)
            D_val = sum(v_min[j] * power(mpf(j+1), -s) for j in range(N))
            z_val = zeta(s)
            D_sq_vals[i] = float(abs(D_val)**2)
            zeta_sq_vals[i] = float(abs(z_val)**2)
            weight_vals[i] = 1.0 / (0.25 + t**2)
            integrand_vals[i] = D_sq_vals[i] * zeta_sq_vals[i] * weight_vals[i]

        # Integrate in regions
        dt = t_pos[1] - t_pos[0]
        regions = [
            (0, 1, "near zero"),
            (1, N/2, "low freq"),
            (N/2, N, "mid freq"),
            (N, 2*N, "MV regime"),
            (2*N, 5*N, "tail"),
        ]

        total_integral = 2 * np.trapezoid(integrand_vals, t_pos) / (2*float(pi))

        print(f"\nN={N}, sigma_min={sigma_min:.4e}, S={S:.2e}")
        print(f"  {'Region':>15} {'I_region':>12} {'%':>6} {'avg|D|^2':>12} {'avg|z|^2':>10} {'avg wt':>10}")

        for lo, hi, name in regions:
            mask = (t_pos >= lo) & (t_pos < hi)
            if np.sum(mask) < 2:
                continue
            I_r = 2 * np.trapezoid(integrand_vals[mask], t_pos[mask]) / (2*float(pi))
            avg_D2 = np.mean(D_sq_vals[mask])
            avg_z2 = np.mean(zeta_sq_vals[mask])
            avg_w = np.mean(weight_vals[mask])
            pct = 100 * I_r / total_integral if total_integral > 0 else 0
            print(f"  {name:>15} {I_r:>12.4e} {pct:>5.1f}% {avg_D2:>12.4e} {avg_z2:>10.2f} {avg_w:>10.2e}")

        print(f"  Total integral: {total_integral:.4e} (sigma_min: {sigma_min:.4e}, ratio: {total_integral/sigma_min:.4f})")

        # KEY: What is the MV mean square of D on [T, 2T]?
        # MV: (1/T) integral_T^{2T} |D|^2 dt ≈ sum |c_j|^2 = 1
        # But the minimizer may concentrate c differently
        sum_cj2 = np.sum(v_min**2)  # should be 1
        sum_cj2_j = np.sum(v_min**2 / np.arange(1, N+1))

        for T in [N, 2*N, 5*N]:
            mask_T = (t_pos >= T) & (t_pos < 2*T)
            if np.sum(mask_T) < 2:
                continue
            mean_D2 = np.mean(D_sq_vals[mask_T])
            print(f"  MV check [T={T:.0f}, 2T]: mean|D|^2 = {mean_D2:.4e}, "
                  f"MV prediction = {sum_cj2:.4e}, ratio = {mean_D2/sum_cj2:.4f}")

        print(f"  ({time.time()-t0:.1f}s)")

    # ================================================================
    # PART B: Relationship between sigma_min and dist_N^2
    # ================================================================
    print(f"\n{'='*70}")
    print("PART B: sigma_min vs dist_N^2 relationship")
    print("-" * 70)

    print(f"\n{'N':>5} {'sigma_min':>12} {'dist_N^2':>12} {'sig/dist':>10} "
          f"{'sig*N^2':>10} {'dist*N^2':>10} {'ln(N)':>8}")
    print("-" * 80)

    for N in [10, 20, 30, 50, 75, 100, 150, 200]:
        t0 = time.time()
        G = build_gram(N, n_grid=max(500000, N*5000))
        evals = np.linalg.eigvalsh(G)
        sigma_min = evals[0]

        # dist_N^2 = Schur complement for j=N
        G_sub = G[:N-1, :N-1]
        g_cross = G[N-1, :N-1]
        coeffs = np.linalg.solve(G_sub, g_cross)
        dist_N_sq = G[N-1, N-1] - np.dot(g_cross, coeffs)

        ratio = sigma_min / dist_N_sq if dist_N_sq > 0 else 0
        print(f"{N:>5} {sigma_min:>12.4e} {dist_N_sq:>12.4e} {ratio:>10.4f} "
              f"{sigma_min*N**2:>10.4f} {dist_N_sq*N**2:>10.4f} {np.log(N):>8.3f}")

    # ================================================================
    # PART C: Test sigma_min >= c * min_j(dist_j^2)
    # ================================================================
    print(f"\n{'='*70}")
    print("PART C: sigma_min vs min_j(dist_j^2)")
    print("-" * 70)

    for N in [20, 50, 100, 200]:
        G = build_gram(N, n_grid=max(500000, N*5000))
        evals = np.linalg.eigvalsh(G)
        sigma_min = evals[0]

        # Compute all dist_j^2
        dist_sq = np.zeros(N)
        for j in range(1, N+1):
            if j == 1:
                dist_sq[0] = G[0, 0]
            else:
                G_sub = G[:j-1, :j-1]
                g_cross = G[j-1, :j-1]
                coeffs = np.linalg.solve(G_sub, g_cross)
                dist_sq[j-1] = G[j-1, j-1] - np.dot(g_cross, coeffs)

        min_dist = np.min(dist_sq)
        j_min = np.argmin(dist_sq) + 1

        print(f"N={N:>4}: sigma_min={sigma_min:.4e}, min(dist_j^2)={min_dist:.4e} at j={j_min}, "
              f"ratio={sigma_min/min_dist:.4f}")

    # ================================================================
    # PART D: Is sigma_min ~ det(G)^{1/N} ?  (geometric mean of eigenvalues)
    # ================================================================
    print(f"\n{'='*70}")
    print("PART D: sigma_min vs geometric mean of eigenvalues")
    print("-" * 70)

    for N in [20, 50, 100]:
        G = build_gram(N, n_grid=max(500000, N*5000))
        evals = np.linalg.eigvalsh(G)
        sigma_min = evals[0]
        geo_mean = np.exp(np.mean(np.log(evals)))
        det_per_N = np.exp(np.sum(np.log(evals)) / N)

        print(f"N={N:>4}: sigma_min={sigma_min:.4e}, geo_mean={geo_mean:.4e}, "
              f"ratio={sigma_min/geo_mean:.4f}")

    # ================================================================
    # PART E: Eigenvalue distribution — is there a gap formula?
    # ================================================================
    print(f"\n{'='*70}")
    print("PART E: Eigenvalue distribution near sigma_min")
    print("-" * 70)

    for N in [50, 100, 200]:
        G = build_gram(N, n_grid=max(500000, N*5000))
        evals = np.linalg.eigvalsh(G)

        print(f"\nN={N}: bottom 10 eigenvalues * N^2:")
        for i in range(10):
            print(f"  sigma_{i+1} * N^2 = {evals[i]*N**2:>10.4f}  "
                  f"(ratio to prev: {evals[i]/evals[i-1] if i>0 else 0:>8.4f})")

        # Check: does sigma_k ~ c_k / N^2 for fixed k?
        print(f"  Ratios sigma_k/sigma_1: ", end="")
        for i in range(5):
            print(f"{evals[i]/evals[0]:.2f} ", end="")
        print()

    # ================================================================
    # PART F: The Bessel inequality approach
    # ================================================================
    print(f"\n{'='*70}")
    print("PART F: BESSEL INEQUALITY APPROACH")
    print("-" * 70)
    print("""
ANALYTICAL ARGUMENT:

For c in w_perp with ||c|| = 1, the Mellin representation gives:
  c^T G c = (1/2pi) integral |D(1/2+it)|^2 * |zeta(1/2+it)|^2 / (1/4+t^2) dt

where D(s) = sum c_j j^{-s}.

Step 1: On the real line (t large), MV gives:
  (1/T) integral_T^{2T} |D(1/2+it)|^2 dt -> sum |c_j|^2 = 1  as T -> inf

Step 2: The weight |zeta(1/2+it)|^2 / (1/4+t^2):
  - Average of |zeta(1/2+it)|^2 on [T, 2T] ~ log(T)
  - Weight 1/(1/4+t^2) ~ 1/T^2 on this interval
  - Combined: ~ log(T) / T^2

Step 3: The tail contribution (t > N):
  integral_N^inf |D|^2 |zeta|^2 / (1/4+t^2) dt
  >= sum_{k: N <= 2^k N <= ???} integral_{2^k N}^{2^{k+1} N} |D|^2 * |zeta|^2 / t^2 dt
  ~ sum_k (mean |D|^2) * (mean |zeta|^2) * integral dt/t^2
  ~ sum_k 1 * log(2^k N) * 1/(2^k N)
  = sum_k (k*log2 + logN) / (2^k * N)
  = (1/N) * (log(N) * sum 1/2^k + log(2) * sum k/2^k)
  = (1/N) * (log(N) * 1 + log(2) * 2)
  ~ log(N) / N

But we need ~ 1/N^2. So the tail alone gives MORE than enough!
Unless the minimizer D makes |D|^2 much smaller than 1 in the tail...
""")

    # Test: does the minimizing D have |D|^2 << 1 for large t?
    print("Test: mean |D(1/2+it)|^2 for minimizing eigenvector in various regions")
    for N in [50, 100]:
        G = build_gram(N, n_grid=500000)
        evals, evecs = np.linalg.eigh(G)
        v_min = evecs[:, 0]
        sigma_min = evals[0]

        for T_lo, T_hi in [(N, 2*N), (2*N, 4*N), (4*N, 8*N), (10*N, 20*N)]:
            t_pts = np.linspace(T_lo, T_hi, 200)
            D2_vals = []
            for t in t_pts:
                s = mpc(0.5, t)
                D = sum(v_min[j] * power(mpf(j+1), -s) for j in range(N))
                D2_vals.append(float(abs(D)**2))

            mean_D2 = np.mean(D2_vals)
            MV_pred = np.sum(v_min**2)  # = 1
            print(f"  N={N}, [{T_lo},{T_hi}]: mean|D|^2 = {mean_D2:.4e} "
                  f"(MV pred: {MV_pred:.4f}, ratio: {mean_D2/MV_pred:.4f})")

    # ================================================================
    # PART G: The key computation — can we prove sigma_min >= c/N^2
    # by showing the minimizer can't avoid the weight everywhere?
    # ================================================================
    print(f"\n{'='*70}")
    print("PART G: COMPLETENESS ARGUMENT TEST")
    print("-" * 70)
    print("""
STRATEGY: sigma_min >= c/N^2 if we can show:

For all unit vectors c in w_perp (dimension N-1):
  integral |D(1/2+it)|^2 * W(t) dt >= c_0 / N^2

where W(t) = |zeta(1/2+it)|^2 / (1/4+t^2).

This is equivalent to: the operator
  T_W[c](t) = W(t)^{1/2} * D_c(1/2+it)  (maps R^N to L^2(R))
has sigma_min(T_W) >= c_0/N.

By MV, T_1[c](t) = D_c(1/2+it) has sigma_min ~ 1 (large T limit).
The weight W "deforms" this.

KEY: W(t) ~ log(t)/t^2 for large t, so the effective space shrinks.
The question is: how much does the N-dimensional space R^N "see" through W?
""")

    # Compute the "effective dimension" seen through W
    for N in [20, 50]:
        G = build_gram(N, n_grid=500000)
        evals = np.linalg.eigvalsh(G)

        # Participation ratio: (sum lambda_i)^2 / sum lambda_i^2
        PR = np.sum(evals)**2 / np.sum(evals**2)
        print(f"N={N}: participation ratio = {PR:.2f} of {N}")
        print(f"  Bottom 5 * N^2: {', '.join(f'{e*N**2:.3f}' for e in evals[:5])}")
        print(f"  Top 5: {', '.join(f'{e:.4f}' for e in evals[-5:])}")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)
