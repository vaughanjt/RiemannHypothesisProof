"""
Session 29h: Attack Connes H1 (eigenvector freezing) and H2 (spectral gap).

FROM THE PROOF SKELETON (proof_skeleton.tex, session 25):
- H1: Eigenvector components |xi_n| decay geometrically: |xi_n| <= C * r^|n|
- H2: Spectral gap (eps_1 - eps_0)/|eps_0| >= c > 0 uniformly in lambda
- H3: eps_0 <= C*exp(-cL) — PROVED via Beckermann-Townsend

THE ARGUMENT:
  tau has displacement rank 2 with generators from b(z) analytic in Bernstein ellipse.
  By BT, singular values decay geometrically.

FOR H1: Need to show eigenvectors inherit the analyticity of the generators.
  Key: for Cauchy-like matrices, eigenvector components are evaluations of
  a rational function at the nodes. If the nodes are equispaced and the
  rational function is analytic in the Bernstein ellipse, the components
  decay geometrically.

FOR H2: BT gives bounds on ALL singular values, but the spectral gap requires
  showing the SECOND-smallest eigenvalue is bounded away from zero.
  This follows if tau is a small perturbation of a matrix with known gap.

NUMERICAL VERIFICATION at multiple lambda values.
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, zeta, pi, log, euler, nstr, power, psi, hyp2f1, exp
import time

mp.dps = 50


def build_tau(lam_sq, N):
    """Build the Weil matrix tau from Connes arXiv:2511.22755, Lemma 5.1.

    tau_{nn} = a_n
    tau_{nm} = (b_n - b_m)/(n - m) for n != m

    a_n = Re(psi(1/4 + pi*i*n/L)) + (L/2)  (approximate — see Prop 4.2)
    b_n = Im(psi(1/4 + pi*i*n/L))
    L = log(lam_sq)
    """
    L = float(log(mpf(lam_sq)))
    size = 2*N + 1
    indices = list(range(-N, N+1))

    # Compute b_n from Prop 4.2
    b = np.zeros(size)
    a = np.zeros(size)

    for idx, n in enumerate(indices):
        if n == 0:
            # b_0 = 0 by antisymmetry
            b[idx] = 0.0
            s_val = mpc(0.25, 0)
        else:
            s_val = mpc(0.25, float(pi) * n / L)

        psi_val = psi(0, s_val)
        a[idx] = float(psi_val.real) + L/2
        b[idx] = float(psi_val.imag)

    # Build tau
    tau = np.zeros((size, size))
    for i in range(size):
        tau[i, i] = a[i]
        for j in range(size):
            if i != j:
                tau[i, j] = (b[i] - b[j]) / (indices[i] - indices[j])

    return tau, indices, a, b


if __name__ == "__main__":
    print("SESSION 29h: CONNES H1/H2 ATTACK")
    print("=" * 70)

    # ================================================================
    # PART 1: Verify eigenvector decay (H1) at multiple lambda
    # ================================================================
    print("\nPART 1: EIGENVECTOR GEOMETRIC DECAY (H1)")
    print("-" * 70)

    N = 40  # Large enough for freezing

    for lam_sq in [14, 30, 50, 100, 200]:
        t0 = time.time()
        L = np.log(lam_sq)
        tau, indices, a, b = build_tau(lam_sq, N)

        evals, evecs = np.linalg.eigh(tau)
        # eps_0 is the smallest eigenvalue
        eps_0 = evals[0]
        xi_0 = evecs[:, 0]  # eigenvector for eps_0

        # Measure geometric decay rate
        # |xi_n| ~ C * r^|n|
        abs_xi = np.abs(xi_0)
        center = N  # index of n=0 in the array

        # Fit decay rate from |xi_n| for |n| > 5
        ns = np.arange(5, N+1)
        log_xi_pos = np.log(abs_xi[center + ns] + 1e-300)
        log_xi_neg = np.log(abs_xi[center - ns] + 1e-300)
        log_xi_avg = (log_xi_pos + log_xi_neg) / 2

        # Linear fit: log|xi_n| ~ log(C) - |n| * log(1/r)
        valid = log_xi_avg > -50  # exclude zeros
        if np.sum(valid) > 2:
            slope, intercept = np.polyfit(ns[valid], log_xi_avg[valid], 1)
            r = np.exp(slope)  # decay rate
        else:
            r = 0
            slope = -np.inf

        # Bernstein ellipse prediction: r = exp(-L/(8*pi*N))
        r_predicted = np.exp(-L / (8 * np.pi * N))

        print(f"  lam^2={lam_sq:>4}, L={L:.3f}: eps_0={eps_0:.4e}, "
              f"r_meas={r:.4f}, r_BT={r_predicted:.4f}, "
              f"ratio={r/r_predicted:.2f} ({time.time()-t0:.1f}s)")

        # Show first few components
        if lam_sq in [14, 50]:
            print(f"    |xi_n| for n=0..10: {', '.join(f'{abs_xi[center+k]:.4f}' for k in range(11))}")
            ratios_str = ', '.join(f'{abs_xi[center+k+1]/max(abs_xi[center+k],1e-20):.4f}' for k in range(10))
            print(f"    Ratios |xi_{{n+1}}/xi_n|: {ratios_str}")

    # ================================================================
    # PART 2: Eigenvector freezing — convergence as N grows
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 2: EIGENVECTOR FREEZING AS N GROWS")
    print("-" * 70)

    lam_sq = 50
    L = np.log(lam_sq)

    prev_xi = None
    for N in [10, 20, 30, 40, 50, 60]:
        tau, indices, a, b = build_tau(lam_sq, N)
        evals, evecs = np.linalg.eigh(tau)
        xi_0 = evecs[:, 0]
        eps_0 = evals[0]
        center = N

        # Compare with previous N
        if prev_xi is not None:
            # Align the common part
            prev_N = (len(prev_xi) - 1) // 2
            common = min(prev_N, N)
            # Extract central 2*common+1 components
            curr_central = xi_0[N-common:N+common+1]
            prev_central = prev_xi[prev_N-common:prev_N+common+1]

            # Normalize for sign
            if np.dot(curr_central, prev_central) < 0:
                curr_central = -curr_central

            diff = np.linalg.norm(curr_central - prev_central)
            print(f"  N={N:>3}: eps_0={eps_0:.6e}, "
                  f"||xi_N - xi_{{N-10}}||_common = {diff:.4e}")
        else:
            print(f"  N={N:>3}: eps_0={eps_0:.6e}")

        prev_xi = xi_0.copy()

    # ================================================================
    # PART 3: Spectral gap (H2)
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 3: SPECTRAL GAP (H2)")
    print("-" * 70)

    N = 40

    print(f"{'lam^2':>6} {'eps_0':>12} {'eps_1':>12} {'gap':>12} "
          f"{'gap/|eps_0|':>12} {'eps_2':>12}")
    print("-" * 70)

    for lam_sq in [10, 14, 20, 30, 50, 75, 100, 150, 200, 300, 500]:
        tau, indices, a, b = build_tau(lam_sq, N)
        evals = np.linalg.eigvalsh(tau)
        eps_0, eps_1, eps_2 = evals[0], evals[1], evals[2]
        gap = eps_1 - eps_0
        gap_ratio = gap / abs(eps_0) if abs(eps_0) > 1e-20 else float('inf')

        print(f"{lam_sq:>6} {eps_0:>12.4e} {eps_1:>12.4e} {gap:>12.4e} "
              f"{gap_ratio:>12.2f} {eps_2:>12.4e}")

    # ================================================================
    # PART 4: Displacement rank verification at high precision
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 4: DISPLACEMENT RANK (high precision)")
    print("-" * 70)

    mp.dps = 80
    N_small = 20

    for lam_sq in [14, 50, 100]:
        L = float(log(mpf(lam_sq)))
        size = 2*N_small + 1
        indices = list(range(-N_small, N_small+1))

        # High-precision tau
        b_hp = np.zeros(size)
        a_hp = np.zeros(size)
        for idx, n in enumerate(indices):
            s_val = mpc(0.25, float(pi) * n / L)
            psi_val = psi(0, s_val)
            a_hp[idx] = float(psi_val.real) + L/2
            b_hp[idx] = float(psi_val.imag)

        tau_hp = np.zeros((size, size))
        for i in range(size):
            tau_hp[i, i] = a_hp[i]
            for j in range(size):
                if i != j:
                    tau_hp[i, j] = (b_hp[i] - b_hp[j]) / (indices[i] - indices[j])

        # Displacement: D*tau - tau*D
        D = np.diag(np.array(indices, dtype=float))
        disp = D @ tau_hp - tau_hp @ D

        _, svd_vals, _ = np.linalg.svd(disp)
        print(f"  lam^2={lam_sq:>4}: top SVs of D*tau-tau*D: "
              f"{svd_vals[0]:.4e}, {svd_vals[1]:.4e}, {svd_vals[2]:.4e}, "
              f"{svd_vals[3]:.4e}")
        print(f"    sigma_3/sigma_1 = {svd_vals[2]/svd_vals[0]:.4e} "
              f"(rank 2 if << 1)")

    mp.dps = 50

    # ================================================================
    # PART 5: The key test — does BT bound actually hold for eigenvalues?
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 5: BT EIGENVALUE DECAY TEST")
    print("-" * 70)

    N = 40

    for lam_sq in [14, 50, 100]:
        L = np.log(lam_sq)
        rho = np.exp(L / (8 * np.pi * N))
        tau, indices, a, b = build_tau(lam_sq, N)
        evals = np.linalg.eigvalsh(tau)

        # Sort by absolute value (descending) for comparison with BT
        evals_sorted = np.sort(np.abs(evals))[::-1]

        print(f"\n  lam^2={lam_sq}, L={L:.3f}, rho={rho:.6f}")
        print(f"  BT prediction: sigma_k <= C * rho^{{-k}}")
        print(f"  {'k':>4} {'|eig_k|':>12} {'C*rho^{-k}':>12} {'ratio':>10}")

        # Estimate C from the largest eigenvalue
        C_est = evals_sorted[0]

        for k in [1, 2, 3, 5, 10, 20, 40, 60, 80]:
            if k <= len(evals_sorted):
                bt_pred = C_est * rho**(-k)
                ratio = evals_sorted[k-1] / bt_pred if bt_pred > 1e-20 else 0
                print(f"  {k:>4} {evals_sorted[k-1]:>12.4e} {bt_pred:>12.4e} {ratio:>10.4f}")

    # ================================================================
    # PART 6: H1 proof attempt — eigenvector as rational interpolant
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 6: H1 PROOF MECHANISM")
    print("-" * 70)

    # For a Cauchy-like matrix tau with tau_{nm} = (b_n - b_m)/(n - m):
    # The eigenvector equation tau * xi = eps * xi gives:
    #   sum_{m != n} (b_n - b_m)/(n-m) * xi_m + a_n * xi_n = eps * xi_n
    #
    # Define the function: f(z) = sum_m xi_m / (z - m)  (partial fractions)
    # Then: sum_m xi_m * (b_n - b_m)/(n-m) = sum_m xi_m * b_n/(n-m) - sum_m xi_m * b_m/(n-m)
    #      = b_n * f(n) - sum_m b_m * xi_m / (n-m)
    #
    # Not quite what we need. Let me try differently.
    #
    # The Cauchy matrix C_{nm} = 1/(n-m) has eigenvectors related to
    # discrete Chebyshev polynomials. For the MODIFIED Cauchy matrix
    # with (b_n - b_m)/(n-m), the structure is richer.
    #
    # KEY OBSERVATION: b(z) is analytic in a strip |Im(z)| < L/(4*pi)
    # (from the nearest pole of psi at distance L/(4*pi) from the real axis).
    # The discrete derivative (b_n - b_m)/(n-m) ~ b'(xi_nm) for some xi_nm in [m,n].
    # So tau is approximately a discretization of b'(z) evaluated on pairs from [-N,N].
    #
    # For such discretizations, the eigenvectors inherit the analyticity of b(z):
    # they are essentially the kernel functions of the underlying integral operator.

    lam_sq = 50
    N = 40
    tau, indices, a, b = build_tau(lam_sq, N)
    evals, evecs = np.linalg.eigh(tau)
    xi_0 = evecs[:, 0]
    center = N

    # Test: does xi_0 correspond to evaluation of a smooth function?
    # If xi_n = g(n) for some function g analytic in the Bernstein ellipse,
    # then the Chebyshev coefficients of g decay geometrically.

    # Map indices [-N, N] to [-1, 1] for Chebyshev analysis
    x_nodes = np.array(indices) / N  # in [-1, 1]

    # Compute Chebyshev coefficients via DCT
    from scipy.fft import dct
    # Chebyshev interpolation on equispaced points
    n_pts = len(xi_0)
    cheb_coeffs = np.abs(np.fft.fft(xi_0)) / n_pts * 2
    cheb_coeffs[0] /= 2

    print(f"Chebyshev analysis of xi_0 (lam^2={lam_sq}, N={N}):")
    print(f"  First 20 |Chebyshev coefficients|:")
    for k in range(min(20, len(cheb_coeffs))):
        print(f"    k={k:>3}: |c_k| = {cheb_coeffs[k]:.6e}")

    # Fit decay rate of Chebyshev coefficients
    log_cheb = np.log(cheb_coeffs[1:30] + 1e-300)
    ks = np.arange(1, 30)
    valid_c = log_cheb > -40
    if np.sum(valid_c) > 2:
        slope_c, _ = np.polyfit(ks[valid_c], log_cheb[valid_c], 1)
        rho_cheb = np.exp(-slope_c)
        L = np.log(lam_sq)
        rho_BT = np.exp(L / (8 * np.pi * N))
        print(f"\n  Chebyshev decay rate: rho_cheb = {rho_cheb:.4f}")
        print(f"  BT prediction:        rho_BT = {rho_BT:.4f}")
        print(f"  Ratio: {rho_cheb/rho_BT:.4f}")

    print(f"\n{'='*70}")
    print("H1/H2 STATUS")
    print("=" * 70)
