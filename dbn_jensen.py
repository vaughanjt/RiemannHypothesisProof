"""Jensen polynomials for the xi function and connection to dBN dynamics.

Griffin-Ono-Rolen-Zagier (2019) proved that for each degree d, the
Jensen polynomial J_d^n(xi; x) has only real zeros for n >= n_0(d).
This is a necessary condition for xi to be in the Laguerre-Polya class.

The Taylor coefficients a_n of Xi(t) = xi(1/2 + it) = sum (-1)^n a_n t^{2n}
satisfy a_n > 0 and the Turan inequalities a_n^2 >= a_{n-1}*a_{n+1}.

Connection to our dBN work:
- The Turan inequalities are the d=2 Jensen polynomial condition
- Our log-concavity f''(z) < 0 is the continuous analogue along the critical line
- The GORZ Hermite polynomial limit connects to GUE (same universality class
  as the Coulomb ODE dynamics)

Plan:
1. Compute a_n = xi^{(2n)}(1/2) / (2n)! for n = 0, ..., N_max
2. Verify Turan inequalities and compute the ratios
3. Build Jensen polynomials and find their zeros
4. Find n_0(d) thresholds and compare to GORZ predictions
5. Connect the Turan ratios to our log-concavity margins
"""

import numpy as np
import mpmath
from itertools import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import time


def compute_xi_taylor_coefficients(n_max, dps=60):
    """Compute the Taylor coefficients a_n of Xi(t) = xi(1/2+it).

    Xi(t) = sum_{n=0}^{inf} (-1)^n a_n t^{2n}

    where a_n = xi^{(2n)}(1/2) / (2n)! > 0.

    Uses Cauchy integral formula via FFT: evaluate Xi on a circle
    in the complex t-plane and extract coefficients.
    """
    with mpmath.workdps(dps):
        def xi_eval(t_complex):
            """Xi(t) = xi(1/2 + it) for complex t."""
            s = mpmath.mpf('0.5') + mpmath.mpc(0, 1) * t_complex
            return (s * (s - 1) / 2
                    * mpmath.power(mpmath.pi, -s/2)
                    * mpmath.gamma(s/2)
                    * mpmath.zeta(s))

        # Evaluate Xi(t) on a circle of radius R in the t-plane
        # Use N_pts points for FFT
        N_pts = max(4 * n_max + 16, 256)
        R = mpmath.mpf('10')  # radius (Xi is entire, any R works)

        print(f"  Evaluating Xi on {N_pts}-point circle, radius R={R}...")
        vals = []
        for k in range(N_pts):
            theta = 2 * mpmath.pi * k / N_pts
            t_k = R * mpmath.exp(mpmath.mpc(0, 1) * theta)
            val = xi_eval(t_k)
            vals.append(complex(val))

        vals = np.array(vals)

        # FFT to get Taylor coefficients
        # Xi(t) = sum_m c_m t^m, and c_m = (1/R^m) * (1/N) * sum_k Xi(R*e^{2pi i k/N}) * e^{-2pi i km/N}
        fft_vals = np.fft.fft(vals) / N_pts

        # c_m = fft_vals[m] / R^m
        # Since Xi is even, c_m = 0 for odd m, c_{2n} = (-1)^n a_n
        a = []
        R_float = float(R)
        for n in range(n_max + 1):
            m = 2 * n
            if m < N_pts:
                c_m = fft_vals[m] / R_float**m
                a_n = float(np.real(c_m)) * (-1)**n
                a.append(a_n)
            else:
                a.append(0.0)

        return a


def turan_ratios(a):
    """Compute the Turan ratios T_n = a_n^2 / (a_{n-1} * a_{n+1}).

    The Turan inequality states T_n >= 1 for all n >= 1.
    """
    ratios = []
    for n in range(1, len(a) - 1):
        if a[n-1] != 0 and a[n+1] != 0:
            T_n = a[n]**2 / (a[n-1] * a[n+1])
            ratios.append(T_n)
        else:
            ratios.append(float('nan'))
    return ratios


def jensen_polynomial(a, d, n):
    """Build the Jensen polynomial J_d^n(x) = sum_{j=0}^d C(d,j) a_{n+j} x^j.

    Returns polynomial coefficients [c_0, c_1, ..., c_d] where
    J_d^n(x) = c_0 + c_1*x + ... + c_d*x^d.
    """
    from math import comb
    coeffs = []
    for j in range(d + 1):
        if n + j < len(a):
            coeffs.append(comb(d, j) * a[n + j])
        else:
            coeffs.append(0.0)
    return coeffs


def polynomial_roots(coeffs):
    """Find roots of polynomial with given coefficients.

    coeffs = [c_0, c_1, ..., c_d] for c_0 + c_1*x + ... + c_d*x^d.
    """
    # numpy.roots expects [c_d, ..., c_1, c_0] (highest degree first)
    coeffs_rev = list(reversed(coeffs))
    if abs(coeffs_rev[0]) < 1e-30:
        return np.array([])
    return np.roots(coeffs_rev)


def check_all_real_roots(roots, tol=1e-6):
    """Check if all roots are real (imaginary part < tol)."""
    if len(roots) == 0:
        return True
    return np.all(np.abs(roots.imag) < tol * (1 + np.abs(roots.real)))


def find_n0_threshold(a, d, n_max=None):
    """Find the smallest n such that J_d^n has only real roots for all n' >= n.

    Returns n_0 and the detailed scan results.
    """
    if n_max is None:
        n_max = len(a) - d - 1

    results = []
    for n in range(n_max + 1):
        if n + d >= len(a):
            break
        coeffs = jensen_polynomial(a, d, n)
        roots = polynomial_roots(coeffs)
        all_real = check_all_real_roots(roots)
        results.append({
            'n': n,
            'all_real': all_real,
            'roots': roots,
            'max_imag': float(np.max(np.abs(roots.imag))) if len(roots) > 0 else 0.0,
        })

    # Find n_0: smallest n such that all n' >= n have all_real = True
    n0 = None
    for i in range(len(results) - 1, -1, -1):
        if not results[i]['all_real']:
            n0 = results[i]['n'] + 1
            break
    if n0 is None:
        n0 = 0  # all are hyperbolic

    return n0, results


def run_jensen_analysis():
    """Full Jensen polynomial analysis."""

    print("=" * 70)
    print("JENSEN POLYNOMIALS FOR THE XI FUNCTION")
    print("=" * 70)

    # Step 1: Compute Taylor coefficients
    n_max = 30
    dps = 60
    print(f"\n[1/5] Computing Taylor coefficients a_0, ..., a_{n_max}")
    print(f"  Working precision: {dps} digits")

    t0 = time.time()
    a = compute_xi_taylor_coefficients(n_max, dps=dps)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # Show first few coefficients
    print(f"\n  First coefficients a_n:")
    print(f"  {'n':>4}  {'a_n':>20}  {'log10|a_n|':>12}")
    for n in range(min(20, len(a))):
        la = np.log10(abs(a[n])) if a[n] != 0 else float('-inf')
        print(f"  {n:>4}  {a[n]:>20.12e}  {la:>12.2f}")

    # Check positivity
    n_positive = sum(1 for x in a if x > 0)
    n_negative = sum(1 for x in a if x < 0)
    print(f"\n  Positive: {n_positive}, Negative: {n_negative}, Zero: {len(a) - n_positive - n_negative}")
    if n_negative > 0:
        first_neg = next(i for i, x in enumerate(a) if x < 0)
        print(f"  WARNING: First negative a_n at n = {first_neg}")
        print(f"  (May be precision loss for large n)")

    # Step 2: Turan inequalities
    print(f"\n[2/5] Turan inequalities: a_n^2 / (a_{{n-1}} * a_{{n+1}}) >= 1")

    # Only use coefficients we trust (before precision loss)
    n_trust = n_positive if n_negative > 0 else len(a)
    a_good = a[:n_trust]

    ratios = turan_ratios(a_good)
    print(f"  Computing for n = 1 to {len(ratios)}")

    n_satisfied = sum(1 for r in ratios if r >= 1.0)
    print(f"  Turan inequality satisfied: {n_satisfied}/{len(ratios)}")

    print(f"\n  {'n':>4}  {'T_n':>14}  {'T_n - 1':>14}  {'status':>8}")
    for n, r in enumerate(ratios, start=1):
        status = "OK" if r >= 1.0 else "FAIL"
        print(f"  {n:>4}  {r:>14.8f}  {r-1:>14.8f}  {status:>8}")
        if n >= 25:
            break

    # Step 3: Jensen polynomials for d = 2, 3, 4, 5, ...
    print(f"\n[3/5] Jensen polynomial hyperbolicity thresholds n_0(d)")
    print(f"  J_d^n(x) has only real roots for n >= n_0(d)")

    d_values = list(range(2, 20))
    thresholds = {}

    print(f"\n  {'d':>4}  {'n_0(d)':>8}  {'first complex at n':>20}")
    for d in d_values:
        if d >= len(a_good) - 1:
            break
        n0, scan = find_n0_threshold(a_good, d)
        thresholds[d] = n0

        # Find the specific n values with complex roots
        complex_ns = [r['n'] for r in scan if not r['all_real']]
        complex_str = str(complex_ns[:5]) if complex_ns else "none"

        print(f"  {d:>4}  {n0:>8}  {complex_str:>20}")

    # Step 4: Detailed root analysis for small d
    print(f"\n[4/5] Detailed Jensen polynomial roots for d = 3, 4, 5")

    for d in [3, 4, 5]:
        print(f"\n  d = {d}:")
        print(f"  {'n':>4}  {'all_real':>10}  {'max |Im|':>12}  {'roots (real part)':>30}")
        for n in range(min(15, len(a_good) - d)):
            coeffs = jensen_polynomial(a_good, d, n)
            roots = polynomial_roots(coeffs)
            all_real = check_all_real_roots(roots)
            max_imag = float(np.max(np.abs(roots.imag))) if len(roots) > 0 else 0.0
            real_parts = sorted(roots.real)
            real_str = ', '.join(f'{r:.4f}' for r in real_parts[:4])
            print(f"  {n:>4}  {str(all_real):>10}  {max_imag:>12.2e}  {real_str:>30}")

    # Step 5: Connection to log-concavity and dBN dynamics
    print(f"\n[5/5] Connection to log-concavity and dBN dynamics")

    # The Turan ratio T_n = a_n^2/(a_{n-1}*a_{n+1}) is the d=2 Jensen condition.
    # Our log-concavity f''(z) < 0 is the continuous analogue.
    # The GORZ theorem says J_d^n -> Hermite polynomial as n -> inf.
    # Hermite polynomials have only real roots (they're the GUE wave functions).

    print("\n  BRIDGE: Taylor coefficients <-> log-concavity")
    print("  The Turan ratio T_n measures 'discrete log-concavity' of the")
    print("  sequence {a_n}. Our f''(z) < 0 measures 'continuous log-concavity'")
    print("  of |xi(1/2+iz)| along the critical line.")
    print()

    # Compute the "Turan margin" = T_n - 1 and see if it correlates
    # with our log-concavity margin
    print("  Turan margins (T_n - 1) vs n:")
    margins = [(n+1, r-1) for n, r in enumerate(ratios) if not np.isnan(r)]

    if margins:
        ns_m = [m[0] for m in margins]
        vals_m = [m[1] for m in margins]
        print(f"  Min margin: {min(vals_m):.8f} at n = {ns_m[vals_m.index(min(vals_m))]}")
        print(f"  Max margin: {max(vals_m):.8f} at n = {ns_m[vals_m.index(max(vals_m))]}")

    # The GORZ Hermite limit: as n -> inf,
    # J_d^n(x) / a_n -> He_d(x * sqrt(n) / sigma)
    # where sigma^2 = lim n*(T_n - 1)
    # This "width" parameter connects to the GUE spacing distribution

    if len(ratios) >= 5:
        sigma_sq_estimates = []
        for n, r in enumerate(ratios, start=1):
            if not np.isnan(r) and r > 1:
                sigma_sq_estimates.append(n * (r - 1))

        if sigma_sq_estimates:
            print(f"\n  GORZ width parameter: sigma^2 = lim n*(T_n - 1)")
            print(f"  Estimates: n*(T_n-1) for n = 1..{len(sigma_sq_estimates)}")
            for i in [0, 4, 9, 14, 19, min(24, len(sigma_sq_estimates)-1)]:
                if i < len(sigma_sq_estimates):
                    print(f"    n={i+1}: {sigma_sq_estimates[i]:.6f}")

            # Check convergence
            if len(sigma_sq_estimates) >= 10:
                recent = sigma_sq_estimates[-5:]
                print(f"  Last 5 values: {[f'{x:.4f}' for x in recent]}")
                print(f"  Apparent limit: sigma^2 ~ {np.mean(recent):.4f}")

    # -- Plots --
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) Taylor coefficients a_n
    ax = axes[0, 0]
    ns = list(range(len(a_good)))
    ax.semilogy(ns, [abs(x) for x in a_good], 'bo-', markersize=4, linewidth=1)
    ax.set_xlabel('n')
    ax.set_ylabel('|a_n|')
    ax.set_title('Taylor coefficients of Xi(t) = xi(1/2+it)')
    ax.grid(True, alpha=0.3)

    # (0,1) Turan ratios
    ax = axes[0, 1]
    if ratios:
        ns_r = list(range(1, len(ratios) + 1))
        ax.plot(ns_r, ratios, 'ro-', markersize=4, linewidth=1)
        ax.axhline(y=1.0, color='blue', linestyle='--', linewidth=1.5,
                   label='Turan bound (T_n >= 1)')
        ax.set_xlabel('n')
        ax.set_ylabel('T_n = a_n^2 / (a_{n-1} * a_{n+1})')
        ax.set_title('Turan ratios (d=2 Jensen condition)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # (1,0) n_0(d) thresholds
    ax = axes[1, 0]
    if thresholds:
        ds = sorted(thresholds.keys())
        n0s = [thresholds[d] for d in ds]
        ax.plot(ds, n0s, 'gs-', markersize=6, linewidth=1.5)
        ax.set_xlabel('d (Jensen polynomial degree)')
        ax.set_ylabel('n_0(d) (hyperbolicity threshold)')
        ax.set_title('GORZ thresholds: J_d^n hyperbolic for n >= n_0(d)')
        ax.grid(True, alpha=0.3)

    # (1,1) GORZ width parameter n*(T_n - 1)
    ax = axes[1, 1]
    if sigma_sq_estimates:
        ns_s = list(range(1, len(sigma_sq_estimates) + 1))
        ax.plot(ns_s, sigma_sq_estimates, 'mo-', markersize=3, linewidth=1)
        ax.set_xlabel('n')
        ax.set_ylabel('n * (T_n - 1)')
        ax.set_title('GORZ width: n*(T_n-1) -> sigma^2 (Hermite scaling)')
        ax.grid(True, alpha=0.3)

    fig.suptitle('Jensen Polynomials and Turan Inequalities for xi', fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig('dbn_jensen.png', dpi=150, bbox_inches='tight')
    print("\n  Saved: dbn_jensen.png")
    plt.close(fig)

    # Save data
    save_data = {
        'a_n': a_good,
        'turan_ratios': ratios,
        'thresholds': {str(k): v for k, v in thresholds.items()},
        'n_trust': n_trust,
    }
    with open('dbn_jensen.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    print("  Saved: dbn_jensen.json")

    return a_good, ratios, thresholds


if __name__ == '__main__':
    a, ratios, thresholds = run_jensen_analysis()
