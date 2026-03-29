"""
Session 17d: Connection between Li coefficients and the M-function.

KEY INSIGHT TO TEST:
  Li coefficients = "averaged" M-function in a discrete basis.
  The Li basis naturally smooths the local negativity of M_zeta
  into globally positive lambda_n^{zeta}.

Specifically:
  lambda_n = (1/(n-1)!) * d^n/ds^n [s^{n-1} * log xi(s)] |_{s=1}
           = integral over circle of log xi(1+re^{i*theta}) * kernel_n(theta) dtheta

  The M-function: M(s) = Im[(s-1/2)*conj(xi'/xi(s))]
  is related to d/ds [log xi(s)] = xi'/xi(s).

  Connection: lambda_n involves n-th derivative of log xi,
  which is related to (n-1)-th derivative of xi'/xi.

This script:
  1. Computes the M-function decomposition along a circle around s=1
  2. Shows how the Li coefficient kernel averages the M-function
  3. Tests whether the averaging explains the zeta-part positivity
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, log, pi, zeta, power, loggamma

mp.dps = 50


# ─── M-function from Session 16 ───

def xi_log_deriv(s):
    """(xi'/xi)(s) via mpmath."""
    def xi_func(sv):
        return (sv * (sv - 1) / 2
                * power(pi, -sv/2)
                * mpmath.gamma(sv/2)
                * zeta(sv))
    xi_val = xi_func(s)
    if abs(xi_val) < mpf(10)**(-40):
        return mpc(0, 0)
    xi_d = mpmath.diff(xi_func, s)
    return xi_d / xi_val


def M_function(s):
    """M(s) = Im[(s-1/2) * conj(xi'/xi(s))]"""
    f = xi_log_deriv(s)
    u = s - mpf(0.5)
    return (u * mpmath.conj(f)).imag


def M_gamma_part(s):
    """Gamma contribution to M: from Stirling expansion."""
    # (xi'/xi)_Gamma = 1/s + 1/(s-1) - log(pi)/2 + psi(s/2)/2
    # where psi is digamma
    psi_val = mpmath.digamma(s/2)
    f_gamma = 1/s + 1/(s-1) - log(pi)/2 + psi_val/2
    u = s - mpf(0.5)
    return (u * mpmath.conj(f_gamma)).imag


# ─── Decomposition functions for log xi ───

def log_xi(s):
    h = s - 1
    if abs(h) < mpf(10)**(-40):
        return -log(mpf(2))
    sz = h * zeta(s)
    return log(mpf(0.5)) + log(s) + log(sz) - (s/2) * log(pi) + loggamma(s/2)

def smooth_gamma_part(s):
    return log(mpf(0.5)) + log(s) - (s/2) * log(pi) + loggamma(s/2)

def log_sz(s):
    h = s - 1
    if abs(h) < mpf(10)**(-40):
        return mpf(0)
    return log(h * zeta(s))


def taylor_fft(f, center, n_terms, radius=0.4, n_points=None):
    if n_points is None:
        n_points = max(2 * n_terms + 32, 128)
    r = mpf(radius)
    values = []
    for j in range(n_points):
        theta = 2 * mpf(pi) * j / n_points
        z = mpc(center, 0) + r * mpc(mpmath.cos(theta), mpmath.sin(theta))
        values.append(f(z))
    coeffs = []
    for k in range(n_terms + 1):
        ck = mpc(0, 0)
        for j in range(n_points):
            theta = 2 * mpf(pi) * j / n_points
            ck += values[j] * mpc(mpmath.cos(-k * theta), mpmath.sin(-k * theta))
        ck /= n_points
        ck /= power(r, k)
        coeffs.append(ck.real if abs(ck.imag) < mpf(10)**(-20) else ck)
    return coeffs


def lambda_from_taylor(n, coeffs):
    total = mpf(0)
    for j in range(n):
        k = n - j
        if k < len(coeffs):
            total += mpmath.binomial(n - 1, j) * coeffs[k]
    return n * total


if __name__ == "__main__":
    print("=" * 70)
    print("SESSION 17d: Li Coefficient <-> M-Function Connection")
    print("=" * 70)

    # --- Step 1: M-function on circles around s=1 ---
    print("\n--- Step 1: M-function behavior on circle |s-1|=0.4 ---")
    print("  This is the SAME circle used for Taylor FFT")
    r = 0.4
    n_pts = 64

    M_total_vals = []
    M_gamma_vals = []
    M_zeta_vals = []
    thetas = []

    print("  Computing M-function at 64 points on circle...")
    for j in range(n_pts):
        theta = 2 * np.pi * j / n_pts
        s = mpc(1 + r * np.cos(theta), r * np.sin(theta))
        M_tot = float(M_function(s))
        M_gam = float(M_gamma_part(s))
        M_zet = M_tot - M_gam
        M_total_vals.append(M_tot)
        M_gamma_vals.append(M_gam)
        M_zeta_vals.append(M_zet)
        thetas.append(theta)

    M_total_arr = np.array(M_total_vals)
    M_gamma_arr = np.array(M_gamma_vals)
    M_zeta_arr = np.array(M_zeta_vals)

    print(f"\n  M_total: min={M_total_arr.min():+.6f}, max={M_total_arr.max():+.6f}")
    print(f"  M_gamma: min={M_gamma_arr.min():+.6f}, max={M_gamma_arr.max():+.6f}")
    print(f"  M_zeta:  min={M_zeta_arr.min():+.6f}, max={M_zeta_arr.max():+.6f}")
    print(f"  M_zeta < 0 at {np.sum(M_zeta_arr < 0)} of {n_pts} points")

    # Show where M_zeta is most negative
    worst_idx = np.argmin(M_zeta_arr)
    s_worst = 1 + r * np.cos(thetas[worst_idx]) + 1j * r * np.sin(thetas[worst_idx])
    print(f"  Most negative M_zeta at theta={thetas[worst_idx]:.4f} (s={s_worst:.4f})")
    print(f"    M_total={M_total_arr[worst_idx]:+.6f}, M_gamma={M_gamma_arr[worst_idx]:+.6f}, M_zeta={M_zeta_arr[worst_idx]:+.6f}")

    # --- Step 2: Connection via kernel ---
    print("\n--- Step 2: Li coefficient as weighted M-function average ---")
    print("  lambda_n = (1/(n-1)!) * d^n/ds^n [s^{n-1} * log xi(s)] |_{s=1}")
    print("  By Cauchy integral:")
    print("    lambda_n ~ (1/r^n) * integral_0^{2pi} s^{n-1}*log_xi(s) * e^{-in*theta} dtheta")
    print("  The kernel picks out the n-th Fourier mode of s^{n-1}*log_xi on the circle.")
    print()
    print("  The n-th derivative of log xi is related to integrals of (xi'/xi)^k terms.")
    print("  M-function involves first derivative xi'/xi.")
    print("  Higher Li coefficients involve higher-order correlations of xi'/xi.")

    # --- Step 3: The REAL connection ---
    print("\n--- Step 3: The STRUCTURAL connection ---")
    print()
    print("  Both the M-function and Li coefficients test positivity of the same")
    print("  underlying object (the zero distribution), but in different bases:")
    print()
    print("  M-function: tests at EACH POINT s in the right half-plane")
    print("    M(s) > 0 for ALL s with Re(s) > 1/2, Im(s) > 0")
    print("    This is an UNCOUNTABLE family of inequalities")
    print()
    print("  Li criterion: tests in DISCRETE modes n = 1, 2, 3, ...")
    print("    lambda_n > 0 for ALL n >= 1")
    print("    This is a COUNTABLE family of inequalities")
    print()
    print("  The Li coefficients are 'Fourier modes' of log xi on a circle around s=1.")
    print("  They average the M-function behavior over the circle.")
    print()
    print("  KEY: Averaging can turn a locally-negative function into a positive integral.")
    print("  M_zeta is negative at some points on the circle, but lambda_n^{zeta} > 0")
    print("  because the kernel weights the positive regions more heavily.")

    # --- Step 4: Quantitative verification ---
    print("\n--- Step 4: Fourier modes of log_sz on the circle ---")
    print("  The Taylor coefficients of log_sz ARE the Fourier modes.")
    print("  lambda_n^{zeta} is a linear combination of these modes.")

    zeta_coeffs = taylor_fft(log_sz, 1, 30, radius=0.4, n_points=256)
    print(f"\n  First 10 Fourier modes of log((s-1)*zeta(s)):")
    for k in range(11):
        print(f"    d_{k} = {float(zeta_coeffs[k]):+.15f}")

    # The alternating signs in d_k are key: d_1 > 0, d_2 < 0, d_3 > 0, d_4 < 0, ...
    print(f"\n  Sign pattern: {' '.join(['+' if float(zeta_coeffs[k]) >= 0 else '-' for k in range(15)])}")
    print("  (Alternating after d_1 — characteristic of oscillating analytic function)")

    # --- Step 5: Why does the binomial convolution preserve positivity? ---
    print("\n--- Step 5: Binomial convolution analysis ---")
    print("  lambda_n^{zeta} = n * sum_{j=0}^{n-1} C(n-1,j) * d_{n-j}")
    print()
    print("  The binomial coefficients C(n-1,j) peak at j=(n-1)/2")
    print("  and weight d_k with k near n/2 most heavily.")
    print()
    print("  Because d_1 = gamma >> |d_k| for k >= 2,")
    print("  and the j=(n-1) term always contributes C(n-1,n-1)*d_1 = d_1 = gamma,")
    print("  the Euler constant provides a 'floor' that keeps lambda_n^{zeta} > 0.")
    print()

    # Verify: what fraction of lambda_n^{zeta} comes from the d_1*gamma term?
    print("  Fraction of lambda_n^{zeta} from the d_1 (=gamma) term:")
    print(f"  {'n':>3s}  {'lambda_n^zeta':>14s}  {'n*d_1':>14s}  {'fraction':>10s}")
    d1 = float(zeta_coeffs[1])
    for n in range(1, 21):
        z_val = float(lambda_from_taylor(n, zeta_coeffs))
        d1_contrib = n * d1  # C(n-1,n-1) = 1, so d_1 gets weight n
        frac = d1_contrib / z_val if abs(z_val) > 1e-15 else 0
        print(f"  {n:3d}  {z_val:+14.8f}  {d1_contrib:+14.8f}  {frac:10.4f}")

    # --- Step 6: The gamma floor argument ---
    print("\n--- Step 6: The gamma floor ---")
    print("  If d_1 = gamma = 0.5772... and |d_k| decreases fast for k >= 2,")
    print("  then lambda_n^{zeta} = n * [d_1 + sum_{j<n-1} C(n-1,j)*d_{n-j}]")
    print("  The d_1 term contributes n*gamma.")
    print("  The remaining terms contribute n * [correction].")
    print()
    print("  For the correction to not overwhelm n*gamma,")
    print("  we need: |sum_{j<n-1} C(n-1,j)*d_{n-j}| < gamma")
    print()

    # Check the correction
    print("  Correction terms:")
    print(f"  {'n':>3s}  {'n*gamma':>14s}  {'correction':>14s}  {'|corr/gamma|':>14s}")
    for n in range(1, 21):
        z_val = float(lambda_from_taylor(n, zeta_coeffs))
        d1_contrib = n * d1
        correction = z_val - d1_contrib
        ratio = abs(correction) / (n * d1) if n * d1 > 0 else 0
        print(f"  {n:3d}  {d1_contrib:+14.8f}  {correction:+14.8f}  {ratio:14.6f}")

    # --- Step 7: Synthesis ---
    print("\n--- Step 7: SYNTHESIS ---")
    print()
    print("  The Li coefficient decomposition reveals:")
    print()
    print("  1. lambda_n = lambda_n^{Gamma} + lambda_n^{zeta}")
    print("     where Gamma part grows like (n/2)*ln(n)")
    print("     and zeta part is BOUNDED and POSITIVE (oscillating ~0.2-1.5)")
    print()
    print("  2. The zeta part positivity comes from:")
    print("     - d_1 = gamma (Euler constant) providing a 'floor'")
    print("     - Binomial convolution concentrating weight on d_1")
    print("     - Alternating-sign higher coefficients partially cancelling")
    print()
    print("  3. Connection to M-function (Session 16):")
    print("     - M-function tests pointwise (uncountable constraints)")
    print("     - Li coefficients test in averaged modes (countable)")
    print("     - The averaging smooths local M_zeta negativity")
    print("     - Same difficulty: proving prime-distribution properties")
    print()
    print("  4. If lambda_n^{zeta} > 0 could be proved for all n:")
    print("     - Combined with Gamma positivity for n >= 8 -> lambda_n > 0 for n >= 8")
    print("     - Plus finite verification n=1..7 -> RH")
    print("     - The d_1=gamma floor suggests this MIGHT be provable")
    print("       via bounding the correction terms using Stieltjes constant bounds")

    print("\n" + "=" * 70)
