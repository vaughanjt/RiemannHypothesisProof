"""
DE BRUIJN-NEWMAN ATTACK — Λ = 0 from heat flow dynamics.

RH <=> Lambda <= 0. Known: Lambda >= 0 (Rodgers-Tao 2020).
So RH <=> Lambda = 0 exactly.

The heat flow: H_t(z) = integral Xi(z+iy) exp(-y^2/(4t)) dy / sqrt(4*pi*t)

H_t has only real zeros for t > Lambda.
At t = 0: H_0 = Xi, and its zeros are the zeta zeros.

The zeros of H_t move under the heat flow. The dynamics:
- Each zero z_k(t) satisfies an ODE (the "electrostatic" model)
- Zeros repel each other on the real line
- RH says: at t=0, all zeros are already real (Λ = 0)

APPROACH: Compute H_t for small t and track zero dynamics.
If zeros approach the real line MONOTONICALLY as t -> 0+,
this is evidence for Λ = 0.

The key computation: H_t(z) for complex z near the critical line,
at small t values.
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, zeta, pi, exp, sqrt, gamma, log, nstr
import time

mp.dps = 30


def xi_function(z):
    """The Riemann Xi function: Xi(z) = (1/2)*s*(s-1)*pi^{-s/2}*Gamma(s/2)*zeta(s)
    where s = 1/2 + iz.
    """
    z_mp = mpc(z)
    s = mpf('0.5') + mpc(0, 1) * z_mp

    try:
        val = mpf('0.5') * s * (s - 1) * mpmath.power(pi, -s/2) * gamma(s/2) * zeta(s)
        return complex(val)
    except:
        return 0.0


def H_t(z, t, n_quad=500):
    """Heat-evolved Xi function.

    H_t(z) = (1/sqrt(4*pi*t)) * integral Xi(z+iy) exp(-y^2/(4t)) dy

    For t > 0: this smooths Xi, pushing zeros onto the real line.
    For t = 0: H_0 = Xi.
    """
    if t <= 0:
        return xi_function(z)

    z_c = complex(z)
    sigma = np.sqrt(2 * t)
    y_max = 6 * sigma  # 6-sigma cutoff
    y_grid = np.linspace(-y_max, y_max, n_quad)
    dy = y_grid[1] - y_grid[0]

    total = 0.0
    for y in y_grid:
        xi_val = xi_function(z_c + 1j * y)
        gaussian = np.exp(-y**2 / (4 * t))
        total += xi_val * gaussian

    return total * dy / np.sqrt(4 * np.pi * t)


if __name__ == "__main__":
    print("DE BRUIJN-NEWMAN ATTACK")
    print("=" * 70)

    gammas = np.load("_zeros_500.npy")

    # ================================================================
    # PART 1: Verify Xi zeros
    # ================================================================
    print("\nPART 1: Xi function at known zeros")
    print("-" * 50)

    for k in range(5):
        val = xi_function(gammas[k])
        print(f"  Xi(gamma_{k+1}={gammas[k]:.6f}) = {val:.6e}")

    # ================================================================
    # PART 2: H_t at real points near gamma_1, for decreasing t
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 2: H_t NEAR gamma_1 FOR DECREASING t")
    print("-" * 70)

    gamma_1 = gammas[0]

    print(f"\n  z = gamma_1 = {gamma_1:.6f} (real zero of Xi)")
    print(f"  {'t':>8} {'H_t(gamma_1)':>18} {'|H_t|':>12}")
    print("  " + "-" * 45)

    for t in [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]:
        val = H_t(gamma_1, t, n_quad=300)
        print(f"  {t:>8.3f} {val.real:>18.10f} {abs(val):>12.4e}")

    # ================================================================
    # PART 3: Zero tracking — where are H_t's zeros?
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 3: ZERO TRACKING — H_t zeros near gamma_1")
    print("-" * 70)

    # For each t: find the zero of H_t nearest to gamma_1
    # Scan on the real line near gamma_1

    print(f"  {'t':>8} {'zero_real':>14} {'zero_imag':>14} {'|zero-g1|':>12}")
    print("  " + "-" * 55)

    for t in [0.2, 0.1, 0.05, 0.02, 0.01]:
        # Scan real line near gamma_1
        z_scan = np.linspace(gamma_1 - 2, gamma_1 + 2, 500)
        h_vals = np.array([H_t(z, t, n_quad=200).real for z in z_scan])

        # Find sign changes
        zeros = []
        for i in range(len(h_vals)-1):
            if h_vals[i] * h_vals[i+1] < 0:
                # Bisect
                lo, hi = z_scan[i], z_scan[i+1]
                for _ in range(40):
                    mid = (lo + hi) / 2
                    if H_t(mid, t, n_quad=200).real * H_t(lo, t, n_quad=200).real < 0:
                        hi = mid
                    else:
                        lo = mid
                zeros.append((lo+hi)/2)

        if zeros:
            nearest = min(zeros, key=lambda z: abs(z - gamma_1))
            print(f"  {t:>8.3f} {nearest:>14.8f} {'0':>14} {abs(nearest-gamma_1):>12.4e}")
        else:
            print(f"  {t:>8.3f}  (no real zero found near gamma_1)")

    # ================================================================
    # PART 4: The critical test — H_t with COMPLEX z
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 4: H_t OFF THE REAL LINE (complex zeros)")
    print("-" * 70)

    # If Lambda > 0: H_t has complex zeros for t < Lambda.
    # If Lambda = 0: H_t has only real zeros for ALL t >= 0.
    # Test: at t = 0.01, does H_t have zeros with Im(z) != 0?

    t_test = 0.01
    print(f"\n  t = {t_test}, scanning Im(z) = 0 to 0.5 near Re(z) = gamma_1")

    for im_z in [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
        z_test = gamma_1 + 1j * im_z
        val = H_t(z_test, t_test, n_quad=200)
        print(f"    z = {gamma_1:.4f} + {im_z:.2f}i: "
              f"H_t = {val.real:>12.6f} + {val.imag:>12.6f}i, |H_t| = {abs(val):.6e}")

    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print("=" * 70)
    print("""
If H_t has zeros ONLY on the real line for all t > 0:
  => Lambda <= 0 => RH (combined with Lambda >= 0 from Rodgers-Tao)

The heat flow SMOOTHS Xi. As t -> 0+:
  - H_t -> Xi (the original function)
  - Zeros of H_t approach zeros of Xi
  - If zeros stay real during the limit: RH

The de Bruijn-Newman constant Lambda is the INFIMUM of t where
all zeros become real. Proving Lambda = 0 means they were
ALWAYS real — which IS the Riemann Hypothesis.
""")
