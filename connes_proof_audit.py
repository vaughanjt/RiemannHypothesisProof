"""
Session 26a: Systematic audit of proof_skeleton.tex against numerical data.

Checks every auditable claim in the proof:
1. Section 1: Displacement structure (already verified)
2. Section 2: Bernstein ellipse parameter rho, Beckermann-Townsend application
3. Section 3: H1 eigenvector decay rate, H2 gap bound logic
4. Section 4: The critical Fourier bump argument — does |xi_hat(gamma_k)| ~ eps_0?

The CRITICAL CHECK is #4: our Session 25 data showed |F_T[xi](gamma_k)| ~ O(1),
NOT O(eps_0). If the proof claims |xi_hat(gamma_k)| <= C*eps_0 (line 192),
this directly contradicts our measurements.
"""

import mpmath
from mpmath import mp, mpf, pi, log, exp, nstr, fabs, sqrt
import numpy as np

mp.dps = 30

print("PROOF SKELETON AUDIT")
print("=" * 80)

# =====================================================================
# CHECK 1: Bernstein ellipse parameter (Section 2, line 62)
# Skeleton claims: rho = exp(d/(2N)) = exp(L/(8*pi*N))
# Standard formula: rho = (d/N) + sqrt((d/N)^2 + 1) ~ exp(d/N) for small d/N
# The factor of 2 (d/(2N) vs d/N) comes from taking half the distance to pole
# =====================================================================
print("\nCHECK 1: Bernstein ellipse parameter rho")
print("-" * 60)
for lam_sq in [14, 50, 100]:
    L = float(log(mpf(lam_sq)))
    N = 30
    d = L / (4 * np.pi)  # nearest pole distance

    # Skeleton's formula
    rho_skeleton = np.exp(d / (2 * N))  # = exp(L/(8*pi*N))

    # Standard Bernstein: rho = d/N + sqrt((d/N)^2 + 1)
    rho_standard = d / N + np.sqrt((d / N) ** 2 + 1)

    # At half distance (conservative)
    rho_half = (d / 2) / N + np.sqrt(((d / 2) / N) ** 2 + 1)

    print(f"  lam^2={lam_sq}: d={d:.4f}, d/N={d/N:.6f}")
    print(f"    rho_skeleton = exp(d/(2N)) = {rho_skeleton:.8f}")
    print(f"    rho_standard = {rho_standard:.8f}  (full distance)")
    print(f"    rho_half     = {rho_half:.8f}  (half distance)")
    print(f"    ln(rho_skel) = {np.log(rho_skeleton):.6f}")
    print(f"    ln(rho_std)  = {np.log(rho_standard):.6f}")

print("\n  VERDICT: Skeleton uses half-distance (conservative by ~2x).")
print("  Both give rho > 1, so c > 0 regardless. OK for proof.\n")

# =====================================================================
# CHECK 2: H1 eigenvector decay (Section 3.1, line 97)
# Skeleton claims: |xi_n| <= C * rho^{-|n|}
# Our data (Session 25): xi_n ~ 0.88^|n| for lam^2=14
# rho_skeleton ~ 1.007, so rho^{-1} ~ 0.993 — much weaker than 0.88
# =====================================================================
print("CHECK 2: Eigenvector decay rate")
print("-" * 60)
for lam_sq in [14, 50]:
    L = float(log(mpf(lam_sq)))
    N = 30
    rho = np.exp(L / (8 * np.pi * N))
    r_predicted = 1.0 / rho

    # Measured rates from Session 25 eigenvector data
    if lam_sq == 14:
        r_measured = 0.88  # |xi_1/xi_0| ~ 0.88
    else:
        r_measured = 0.93  # |xi_1/xi_0| ~ 0.93

    print(f"  lam^2={lam_sq}:")
    print(f"    Predicted r = rho^{{-1}} = {r_predicted:.6f}")
    print(f"    Measured  r ~ {r_measured:.2f}")
    print(f"    Measured is FASTER than predicted (good — bound is valid)")
    print(f"    Measured c' = {-np.log(r_measured):.4f}")
    print(f"    Predicted c' = {np.log(rho):.6f}")

print("\n  VERDICT: Bound is valid but very loose (predicts r~0.993, actual r~0.88).")
print("  The proof only needs r < 1, which is satisfied.\n")

# =====================================================================
# CHECK 3: H2 spectral gap logic (Section 3.2, lines 117-127)
# Skeleton claims: eigenvalue-singular value interlacing gives
#   eps_1 - eps_0 >= sigma_2 - sigma_{2N+1} >= C(rho^{-1} - rho^{-(2N+1)})
# ISSUE: For PD matrix, eigenvalues = singular values (ordered differently)
#   sigma_1 >= ... >= sigma_M = eps_0 (smallest)
#   sigma_{M-1} = eps_1 (second smallest)
# The Beckermann bound gives UPPER bounds on sigma_k, not lower bounds.
# The ratio argument (gap/eps_0 ~ rho-1) assumes bound is tight.
# =====================================================================
print("CHECK 3: H2 spectral gap — is the bound rigorous?")
print("-" * 60)
print("  The skeleton claims: (eps_1-eps_0)/|eps_0| >= rho - 1 > 0")
print("  Beckermann gives: sigma_k <= C * rho^{-k} (UPPER bounds)")
print()
print("  For the gap ratio = eps_1/eps_0 - 1:")
print("    If sigma_k ~ C*rho^{-k} (approximately tight):")
print("      eps_0 = sigma_M ~ C*rho^{-M}")
print("      eps_1 = sigma_{M-1} ~ C*rho^{-(M-1)}")
print("      ratio ~ rho - 1")
print("    But 'approximately tight' is NOT a rigorous bound.")
print("    Beckermann ONLY gives upper bounds.")
print()
print("  VERDICT: **GAP BOUND IS NOT RIGOROUS as stated.**")
print("  The Bernstein bound cannot prove gap >= rho-1.")
print("  It can prove eps_0 is small, but not that eps_1 is separated.\n")

# =====================================================================
# CHECK 4: THE CRITICAL CHECK — Section 4, line 192
# Skeleton claims: |xi_hat(gamma_k)| <= C*eps_0 + O(exp(-c'N))
# Session 25 data at dps=120:
#   |F_T[xi](gamma_1)| ~ 0.083 (lam^2=14), eps_0 ~ 5.86e-50
#   Ratio ~ 10^48
# =====================================================================
print("CHECK 4: CRITICAL — |xi_hat(gamma_k)| <= C*eps_0?")
print("-" * 60)
print("  The proof's Section 4 (line 192) claims:")
print("    |xi_hat(gamma_k)| <= C*eps_0 + O(exp(-c'N))")
print()
print("  Session 25 HIGH-PRECISION measurements (dps=120):")
print("  lam^2   |F_T(gamma_1)|   |eps_0|         ratio")
print("  -----   --------------   --------        ------")
print("    14       0.083          5.86e-50       1.4e+48")
print("    30       0.403          1.58e-66       2.6e+65")
print("    50       0.274          3.69e-74       7.4e+72")
print()
print("  |F_T| is O(1), NOT O(eps_0).")
print("  The ratio F_T/eps_0 GROWS with lambda.")
print()
print("  The variational equation gives Q_W(xi, f) = eps_0 * <xi, f>_T.")
print("  The explicit formula decomposes Q_W into sum over zeros.")
print("  But isolating the k-th zero requires controlling the leakage")
print("  sum_{j!=k} |xi_hat(gamma_j) * f_hat(gamma_j)|.")
print("  Since |xi_hat(gamma_j)| ~ O(1) for ALL j, the leakage is O(1),")
print("  which is >> eps_0.")
print()
print("  **THIS IS THE SAME MELLIN/FOURIER GAP FROM SESSION 23.**")
print("  The proof claims the gap is closed, but our data shows it is not.\n")
print("  VERDICT: **CRITICAL GAP — line 192 is FALSE.**")
print("  |xi_hat(gamma_k)| ~ O(1), not O(eps_0).")
print("  The variational equation cannot isolate individual zeros.\n")

# =====================================================================
# CHECK 5: Section 4 — does H_{lambda,N} have all real zeros?
# =====================================================================
print("CHECK 5: H_{lambda,N} has all real zeros")
print("-" * 60)
print("  Proof cites Theorem 1.1(i) of arXiv:2511.22755:")
print("  D_log^{(lambda,N)} is self-adjoint => real spectrum => real zeros of det.")
print("  This is the spectral theorem — correct IF the base paper's Thm 1.1(i)")
print("  establishes self-adjointness.")
print()
print("  VERDICT: Depends on base paper. Cannot verify independently.\n")

# =====================================================================
# CHECK 6: Minor — typo in bump definition (line 174)
# =====================================================================
print("CHECK 6: Typo in bump definition (line 174)")
print("-" * 60)
print("  f(u) = (1/2pi) int phi_k(t) u^{-it} du/u")
print("  Should be: ... dt (not du/u)")
print("  VERDICT: Typo. Does not affect the argument.\n")

# =====================================================================
# SUMMARY
# =====================================================================
print("=" * 80)
print("AUDIT SUMMARY")
print("=" * 80)
print()
print("VERIFIED (no issues):")
print("  [OK] Section 1: Displacement rank = 2")
print("  [OK] Section 2: Pole at L/(4pi), rho > 1, c > 0")
print("  [OK] Section 2: |eps_0| <= C*exp(-cL) with c > 0")
print("  [OK] Section 3.1: Eigenvector decay (numerically; analytic arg is loose)")
print("  [OK] Section 4: H has all real zeros (cites base paper)")
print()
print("ISSUES (minor):")
print("  [FLAG] Section 2: Beckermann-Townsend theorem — need to verify exact statement")
print("  [FLAG] Section 3.1: 'Linear recurrence' from displacement — stated without proof")
print("  [TYPO] Section 4, line 174: du/u should be dt")
print()
print("ISSUES (major):")
print("  [GAP] Section 3.2 (H2): Gap bound not rigorous —")
print("        Bernstein gives upper bounds only, cannot prove gap >= rho-1")
print()
print("ISSUES (critical):")
print("  [FATAL] Section 4, line 192: |xi_hat(gamma_k)| <= C*eps_0")
print("          CONTRADICTED BY DATA: |xi_hat| ~ O(1), eps_0 ~ 10^{-50}")
print("          This is the Mellin/Fourier gap from Session 23.")
print("          The variational equation cannot isolate individual zeros.")
print("          THE PROOF OF UNIFORM CONVERGENCE H -> Xi IS BROKEN.")
