"""Schur algorithm: extract Jacobi parameters from (xi'/xi).

NON-CIRCULAR pipeline:
  mpmath.zeta (definition, no zeros) -> (xi'/xi)(s) -> T(w) -> Schur -> {alpha_k, beta_k}

T(w) = Stieltjes transform of the squared-zero measure:
  T(w) = sum_n 1/(w - gamma_n^2)

At w = -R (negative real axis, R > 0):
  T(-R) = 1/(2*sqrt(R)) * (xi'/xi)(1/2 - sqrt(R))

The Schur algorithm extracts the J-fraction coefficients:
  T(w) = N / (w - alpha_1 - beta_1^2 / (w - alpha_2 - beta_2^2 / ...))

where N = total number of zeros (infinite, but we work with the partial fraction).

For the NORMALIZED measure (divide by N):
  T_norm(w) = 1 / (w - alpha_1 - beta_1^2 / (w - alpha_2 - ...))
"""

import numpy as np
import mpmath
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import time


def xi_log_deriv(s, dps=50):
    """Compute (xi'/xi)(s) via mpmath. NON-CIRCULAR.

    xi(s) = (1/2)*s*(s-1)*pi^{-s/2}*Gamma(s/2)*zeta(s)

    Uses mpmath's built-in zeta (Euler product + functional equation).
    """
    with mpmath.workdps(dps):
        s = mpmath.mpc(s)

        def xi_func(s_val):
            return (s_val * (s_val - 1) / 2
                    * mpmath.power(mpmath.pi, -s_val/2)
                    * mpmath.gamma(s_val/2)
                    * mpmath.zeta(s_val))

        xi_val = xi_func(s)
        xi_deriv = mpmath.diff(xi_func, s)

        if abs(xi_val) < mpmath.mpf(10)**(-dps + 5):
            return complex(mpmath.mpf('nan'))

        return complex(xi_deriv / xi_val)


def T_from_xi(R, dps=50):
    """Compute T(-R) from (xi'/xi). NON-CIRCULAR.

    T(-R) = 1/(2*sqrt(R)) * (xi'/xi)(1/2 - sqrt(R))
    """
    with mpmath.workdps(dps):
        sqrt_R = mpmath.sqrt(mpmath.mpf(str(R)))
        s = mpmath.mpf('0.5') - sqrt_R

        xld = xi_log_deriv(s, dps=dps)
        if np.isnan(xld.real):
            return float('nan')

        return float(mpmath.re(mpmath.mpc(xld) / (2 * sqrt_R)))


def T_from_zeros(R, gammas):
    """Compute T(-R) from known zeros. FOR VERIFICATION ONLY."""
    return np.sum(-1.0 / (R + gammas**2))


def schur_algorithm(T_func, R_values, n_levels, dps=50):
    """Schur algorithm: extract J-fraction coefficients from T(w) on negative real axis.

    T(w) for the normalized measure satisfies:
      T(w) = 1 / (w - alpha_1 - beta_1^2 * T_1(w))

    where T_1 is the "Schur complement" satisfying the same structure.

    At w = -R for large R:
      T(-R) ~ -1/R  (for normalized measure with mass 1)

    Extraction:
      alpha_1 = lim_{R->inf} [-R - 1/T(-R)]
      Practically: fit -R - 1/T(-R) as a function of 1/R and extrapolate.

    Then: T_1(-R) = [T(-R)^{-1} - (-R - alpha_1)] / (-beta_1^2)
    where beta_1^2 = lim_{R->inf} R * [T(-R)^{-1} - (-R - alpha_1)]  ... needs care.
    """
    alphas = []
    betas = []

    # Start with T_current = T (the input function)
    # We store T values at each R
    T_current = np.array([T_func(R) for R in R_values])

    print(f"  Initial T(-R) values:")
    for i in range(0, len(R_values), max(1, len(R_values)//5)):
        print(f"    R={R_values[i]:>12.1f}  T={T_current[i]:>15.10f}")

    for level in range(n_levels):
        # Extract alpha_k from the large-R behavior of -R - 1/T(-R)
        # For T(-R) = 1/(-R - alpha - beta^2 * T_next(-R)):
        #   1/T(-R) = -R - alpha - beta^2 * T_next(-R)
        #   -R - 1/T(-R) = alpha + beta^2 * T_next(-R)
        # For large R: T_next(-R) ~ -1/R, so:
        #   -R - 1/T(-R) ~ alpha - beta^2/R
        # Linear in 1/R: intercept = alpha, slope = -beta^2

        inv_T = 1.0 / T_current
        f_R = -R_values - inv_T  # This should approach alpha + beta^2/R

        # Use the largest R values for the fit
        n_fit = min(len(R_values), 8)
        idx_large = np.argsort(R_values)[-n_fit:]  # largest R indices

        x_fit = 1.0 / R_values[idx_large]
        y_fit = f_R[idx_large]

        # Linear fit: y = alpha + (-beta^2) * x
        if len(x_fit) >= 2:
            coeffs = np.polyfit(x_fit, y_fit, 1)
            alpha_k = coeffs[1]  # intercept
            neg_beta_sq = coeffs[0]  # slope = -beta^2
            beta_sq = -neg_beta_sq
        else:
            alpha_k = y_fit[0]
            beta_sq = 0.0

        alphas.append(float(alpha_k))

        if level < n_levels - 1:
            betas.append(float(np.sqrt(max(beta_sq, 0))))

            # Compute T_next
            residual = inv_T - (-R_values - alpha_k)
            # residual = -beta^2 * T_next(-R)
            if beta_sq > 1e-30:
                T_next = residual / (-beta_sq)
            else:
                T_next = np.zeros_like(T_current)
                break

            # Check for numerical breakdown
            if np.any(np.isnan(T_next)) or np.any(np.isinf(T_next)):
                print(f"  Level {level+1}: numerical breakdown")
                break

            T_current = T_next

        print(f"  Level {level+1}: alpha={alpha_k:.4f}"
              + (f", beta={np.sqrt(max(beta_sq,0)):.4f}" if level < n_levels-1 else ""))

    return np.array(alphas), np.array(betas)


def run_schur():
    """Main: Schur algorithm to extract Jacobi params non-circularly."""

    print("=" * 70)
    print("SCHUR ALGORITHM: JACOBI PARAMS FROM (xi'/xi)")
    print("=" * 70)
    print("  Pipeline: mpmath.zeta -> (xi'/xi) -> T(w) -> Schur -> {alpha, beta}")
    print("  NO ZEROS USED in the extraction.")

    all_zeros = np.load('_zeros_200.npy')

    # Step 1: Verify T(-R) from xi'/xi matches T(-R) from zeros
    print("\n[1/3] Verifying T(-R): xi-derived vs zero-derived")

    R_test = [100, 1000, 10000, 100000, 1000000]
    dps = 50

    print(f"\n  {'R':>10}  {'T_xi':>18}  {'T_zeros(200)':>18}  {'diff':>12}")
    for R in R_test:
        t_xi = T_from_xi(R, dps=dps)
        t_z = T_from_zeros(R, all_zeros)
        diff = t_xi - t_z
        print(f"  {R:>10}  {t_xi:>18.12f}  {t_z:>18.12f}  {diff:>12.4e}")

    print("\n  Note: diff comes from zeros beyond N=200 (tail of infinite sum)")

    # Step 2: Run Schur algorithm on xi-derived T(-R)
    print("\n[2/3] Schur algorithm (from xi'/xi, non-circular)")

    # Use log-spaced R values for good conditioning
    R_values = np.logspace(2, 8, 30)  # R from 100 to 10^8

    print(f"  Evaluating T(-R) at {len(R_values)} points via (xi'/xi)...")
    t0 = time.time()
    T_xi_values = np.array([T_from_xi(R, dps=dps) for R in R_values])
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # Normalize: the total "mass" is sum_n 1 = infinity for all zeros.
    # For the first N zeros, T ~ -N/R for large R.
    # From xi: T includes ALL zeros, so T ~ -N(T_max)/R is unclear.
    # Let's check what "N" the xi-derived T gives.
    N_est = -T_xi_values[-1] * R_values[-1]  # T(-R) ~ -N/R for large R
    print(f"  Estimated N from large-R limit: {N_est:.1f}")
    print(f"  (This is the effective number of zeros contributing)")

    # Normalize T by N_est so the measure has mass 1
    T_normalized = T_xi_values / N_est

    print(f"\n  Running Schur extraction (normalized T)...")
    n_levels = 10
    alphas_xi, betas_xi = schur_algorithm(
        lambda R: T_from_xi(R, dps=dps) / N_est,
        R_values, n_levels, dps=dps)

    # Step 3: Compare with Jacobi params from known zeros
    print(f"\n[3/3] Comparison with zero-derived Jacobi params")

    # Compute Jacobi matrix of squared zeros directly
    from jacobi_zeta_v2 import lanczos_full_reorth, verify_jacobi
    gammas = all_zeros[:200]
    w = gammas**2  # squared zeros
    w_sorted = np.sort(w)
    w_centered = w_sorted - np.mean(w_sorted)

    # Normalize
    a_z, b_z = lanczos_full_reorth(w_sorted / np.mean(w_sorted))
    err = verify_jacobi(a_z, b_z, np.sort(w_sorted / np.mean(w_sorted)))
    print(f"  Zero-derived Jacobi (N=200 squared zeros): err={err:.2e}")

    # Also do unnormalized
    a_z_raw, b_z_raw = lanczos_full_reorth(w_sorted)
    err_raw = verify_jacobi(a_z_raw, b_z_raw, w_sorted)
    print(f"  Zero-derived Jacobi (raw squared zeros): err={err_raw:.2e}")

    n_compare = min(len(alphas_xi), len(a_z_raw))
    print(f"\n  {'level':>6}  {'alpha_xi':>14}  {'alpha_zeros':>14}  {'beta_xi':>14}  {'beta_zeros':>14}")
    print(f"  {'-'*6}  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*14}")
    for k in range(n_compare):
        a_xi = alphas_xi[k] if k < len(alphas_xi) else float('nan')
        a_zr = a_z_raw[k]
        b_xi = betas_xi[k] if k < len(betas_xi) else float('nan')
        b_zr = b_z_raw[k] if k < len(b_z_raw) else float('nan')
        print(f"  {k+1:>6}  {a_xi:>14.4f}  {a_zr:>14.4f}  {b_xi:>14.4f}  {b_zr:>14.4f}")

    # Key question: do the xi-derived params match the zero-derived ones?
    # They WON'T match exactly because:
    # - xi-derived uses ALL zeros (infinite), zero-derived uses N=200
    # - Different normalization
    # But the STRUCTURE should be similar.

    # -- Plots --
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) T(-R) comparison
    ax = axes[0, 0]
    T_zero_vals = np.array([T_from_zeros(R, all_zeros) for R in R_values])
    ax.semilogx(R_values, T_xi_values, 'b-', linewidth=2, label='From (xi\'/xi) [no zeros]')
    ax.semilogx(R_values, T_zero_vals, 'r--', linewidth=1.5, label='From 200 zeros')
    ax.set_xlabel('R')
    ax.set_ylabel('T(-R)')
    ax.set_title('Stieltjes transform: non-circular vs zero-derived')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (0,1) Difference (tail contribution)
    ax = axes[0, 1]
    diff = T_xi_values - T_zero_vals
    ax.semilogx(R_values, diff, 'g-', linewidth=1.5)
    ax.set_xlabel('R')
    ax.set_ylabel('T_xi - T_zeros')
    ax.set_title('Tail contribution (zeros > 200th)')
    ax.grid(True, alpha=0.3)

    # (1,0) Extracted alpha_k
    ax = axes[1, 0]
    ks = np.arange(1, len(alphas_xi)+1)
    ax.plot(ks, alphas_xi, 'bo-', markersize=6, linewidth=1.5, label='From (xi\'/xi)')
    if len(a_z_raw) >= len(ks):
        ax.plot(ks, a_z_raw[:len(ks)], 'rs-', markersize=6, linewidth=1.5, label='From 200 zeros')
    ax.set_xlabel('k')
    ax.set_ylabel('alpha_k')
    ax.set_title('Diagonal Jacobi: xi-derived vs zero-derived')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (1,1) Extracted beta_k
    ax = axes[1, 1]
    ks_b = np.arange(1, len(betas_xi)+1)
    ax.plot(ks_b, betas_xi, 'bo-', markersize=6, linewidth=1.5, label='From (xi\'/xi)')
    if len(b_z_raw) >= len(ks_b):
        ax.plot(ks_b, b_z_raw[:len(ks_b)], 'rs-', markersize=6, linewidth=1.5, label='From 200 zeros')
    ax.set_xlabel('k')
    ax.set_ylabel('beta_k')
    ax.set_title('Off-diagonal Jacobi: xi-derived vs zero-derived')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Schur Algorithm: Non-Circular Jacobi Extraction', fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig('jacobi_schur.png', dpi=150, bbox_inches='tight')
    print("\n  Saved: jacobi_schur.png")
    plt.close(fig)

    save_data = {
        'alphas_xi': alphas_xi.tolist(),
        'betas_xi': betas_xi.tolist(),
        'N_estimated': float(N_est),
        'R_values': R_values.tolist(),
        'T_xi': T_xi_values.tolist(),
    }
    with open('jacobi_schur.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    print("  Saved: jacobi_schur.json")


if __name__ == '__main__':
    run_schur()
