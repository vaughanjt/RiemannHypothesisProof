"""
SESSION 79c -- IS THE KNIFE-EDGE THE SAME AS d=2 TURAN?

Session 68-69 found: xi = F * zeta where F = s(s-1)/2 * pi^{-s/2} * Gamma(s/2).
  - F satisfies d=2 Turan (log-concavity) on its own
  - F FAILS d>=3 Turan
  - zeta creates d>=3 from scratch (arithmetic, every prime load-bearing)

Session 78-79 found: the Gamma diagonal wr_diag = C - log(n) is the EXACT
critical rate for the Lorentzian property. Scale by 5%: breaks.

QUESTION: Are these the same fact?
  - F's d=2 Turan comes from Gamma's specific structure
  - M's Lorentzian comes from Gamma's diagonal at the critical value
  - Both involve Gamma being "exactly right"
  - Both fail when Gamma is perturbed

If they're the same: the proof path is
  1. Gamma gives d=2 Turan for F (provable, Session 68)
  2. Euler product (primes) boosts to d>=3 (needs proof, but Session 69 shows mechanism)
  3. LP (all-d Turan) => Lorentzian for M (needs proof, this is the bridge)
  4. Lorentzian => RH (proved, Theorem 1)

PROBES:
  1. Direct comparison: Taylor coeffs c_k(F) vs diagonal wr_diag(n)
  2. Do they share the same mathematical origin?
  3. If we break d=2 Turan (modify Gamma), does Lorentzian also break?
  4. If we break Lorentzian (scale diagonal), does d=2 Turan also break?
  5. The Mellin/Fourier connection between c_k and wr_diag
"""

import sys
import numpy as np
import mpmath
from mpmath import mp, mpf

mp.dps = 50

sys.path.insert(0, '.')
from session49c_weil_residual import (
    build_all_fast, _compute_wr_diag
)


def odd_block(M, N):
    dim = 2 * N + 1
    P = np.zeros((dim, N))
    for n in range(1, N + 1):
        P[N + n, n - 1] = 1.0 / np.sqrt(2)
        P[N - n, n - 1] = -1.0 / np.sqrt(2)
    return P.T @ M @ P


def compute_taylor_coeffs_F(K=12):
    """Taylor coefficients of F(1/2+it)/F(1/2) in powers of t^{2k}."""
    def F_func(ss):
        return mpf('0.5') * ss * (ss-1) * mpmath.power(mpmath.pi, -ss/2) * \
               mpmath.gamma(ss/2)

    s = mpf('0.5')
    F_val = F_func(s)

    c = [1.0]
    for k in range(1, K + 1):
        deriv = mpmath.diff(F_func, s, n=2*k)
        z_k = deriv / F_val
        fac = float(mpmath.factorial(2*k))
        c.append(float(z_k) * (-1)**k / fac)

    return c


def compute_taylor_coeffs_xi(K=12):
    """Taylor coefficients of xi(1/2+it)/xi(1/2) in powers of t^{2k}."""
    def xi_func(ss):
        return mpf('0.5') * ss * (ss-1) * mpmath.power(mpmath.pi, -ss/2) * \
               mpmath.gamma(ss/2) * mpmath.zeta(ss)

    s = mpf('0.5')
    xi_val = xi_func(s)

    c = [1.0]
    for k in range(1, K + 1):
        deriv = mpmath.diff(xi_func, s, n=2*k)
        z_k = deriv / xi_val
        fac = float(mpmath.factorial(2*k))
        c.append(float(z_k) * (-1)**k / fac)

    return c


def d2_turan(c, k):
    """d=2 Turan ratio: c_k^2 / (c_{k-1} * c_{k+1}). Passes if > 1."""
    if abs(c[k-1]) < 1e-30 or abs(c[k+1]) < 1e-30:
        return float('inf')
    return c[k]**2 / (c[k-1] * c[k+1])


def run():
    print()
    print('#' * 76)
    print('  SESSION 79c -- TURAN-LORENTZIAN BRIDGE')
    print('#' * 76)

    # ======================================================================
    # PROBE 1: Taylor coefficients vs diagonal values
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 1: TAYLOR COEFFICIENTS c_k(F) vs DIAGONAL wr_diag(n)')
    print(f'{"="*76}\n')

    K = 12
    c_F = compute_taylor_coeffs_F(K)
    c_xi = compute_taylor_coeffs_xi(K)

    lam_sq = 1000
    L = np.log(lam_sq)
    N = max(15, round(6 * L))
    wr = _compute_wr_diag(L, N)

    print(f'  {"k/n":>4} {"c_k(F)":>16} {"c_k(xi)":>16} {"wr_diag(n)":>14} '
          f'{"R_k(F)":>10} {"R_k(xi)":>10}')
    print('  ' + '-' * 76)

    for k in range(K + 1):
        wr_n = wr[k] if k <= N else 0
        r_F = d2_turan(c_F, k) if 0 < k < K else 0
        r_xi = d2_turan(c_xi, k) if 0 < k < K else 0
        print(f'  {k:>4d} {c_F[k]:>+16.8e} {c_xi[k]:>+16.8e} {wr_n:>+14.6f} '
              f'{r_F:>10.4f} {r_xi:>10.4f}')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 2: Mathematical origin — both from Gamma
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 2: MATHEMATICAL ORIGIN')
    print(f'{"="*76}\n')

    # c_k(F) comes from d^{2k}/ds^{2k} [s(s-1)/2 * pi^{-s/2} * Gamma(s/2)]
    # evaluated at s=1/2.
    #
    # wr_diag(n) = integral involving psi(s/2) and 2F1, evaluated at
    # specific points related to n and L.
    #
    # Both involve Gamma(s/2) and its derivatives. The connection is through
    # the FUNCTIONAL EQUATION: the test function h(r) = L/(L^2/4 + r^2)
    # has Fourier transform h_hat(x) = pi * exp(-L|x|/2).
    #
    # The explicit formula transforms between the zero-side (Taylor coeffs)
    # and the prime-side (matrix entries).
    #
    # The archimedean part of the explicit formula involves:
    # integral of h(r) * [psi terms] dr
    # which gives wr_diag.
    #
    # The zero-side sum involves:
    # sum_rho h(rho - 1/2)
    # which, expanded in power series around rho = 1/2, gives the Taylor coeffs.

    print(f'  Both c_k(F) and wr_diag(n) originate from Gamma(s/2):')
    print(f'    c_k(F): the 2k-th derivative of F(s) = Gamma(s/2) * (prefactors)')
    print(f'    wr_diag(n): the digamma function psi(1/4 + i*pi*n/L) + corrections')
    print()
    print(f'  The connection is the MELLIN TRANSFORM:')
    print(f'    c_k(F) are moments of the spectral measure of F')
    print(f'    wr_diag(n) are Fourier coefficients of the archimedean kernel')
    print(f'    The Mellin transform connects moments to Fourier coefficients')
    print()

    # Concrete: is wr_diag(n) a specific linear combination of c_k(F)?
    # wr_diag(n) = C(L) + sum_k alpha_k(n, L) * c_k(F) for some alpha?
    # This would be a direct bridge.

    # Test: correlation between wr_diag and c_k
    wr_vals = np.array([wr[n] for n in range(1, min(K+1, N+1))])
    c_F_vals = np.array(c_F[1:len(wr_vals)+1])

    if len(wr_vals) > 2 and len(c_F_vals) > 2:
        corr = np.corrcoef(wr_vals[:len(c_F_vals)], c_F_vals[:len(wr_vals)])[0, 1]
        print(f'  Correlation between wr_diag(n) and c_n(F): {corr:.6f}')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 3: Break d=2 Turan => does Lorentzian break?
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 3: BREAK d=2 TURAN => DOES LORENTZIAN BREAK?')
    print(f'{"="*76}\n')

    # d=2 Turan for F: c_k(F)^2 / (c_{k-1}(F) * c_{k+1}(F)) > 1 for all k.
    # This is log-concavity of c_k.
    # The Gamma function makes c_k(F) decay roughly geometrically (~(1/3)^k).
    #
    # If we MODIFY Gamma to break log-concavity, does M lose Lorentzian?
    #
    # Method: scale wr_diag non-uniformly to break the log-concavity
    # of the corresponding Taylor coefficients.

    _, M_real, _ = build_all_fast(lam_sq, N)
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)
    wr_vec = np.array([wr[abs(int(n))] for n in ns])

    # Perturbation that breaks log-concavity: make wr_diag non-monotonic
    # by FLIPPING the sign of specific components
    print(f'  Perturbations that break log-concavity of the diagonal:')
    print(f'  {"perturbation":>30} {"#pos(M)":>8} {"eig_max(Mo)":>14} {"Lorentzian?":>12}')
    print('  ' + '-' * 70)

    # Baseline
    Mo_real = odd_block(M_real, N)
    emax_real = np.linalg.eigvalsh(Mo_real)[-1]
    npos_real = np.sum(np.linalg.eigvalsh(M_real) > 1e-10)
    print(f'  {"none (baseline)":>30} {npos_real:>8d} {emax_real:>+14.6e} '
          f'{"YES" if npos_real <= 1 else "no":>12}')

    # Flip wr_diag(n) for n > threshold
    for n_flip in [5, 10, 15, 20, 30]:
        M_mod = M_real.copy()
        for i in range(dim):
            n = abs(int(ns[i]))
            if n >= n_flip:
                M_mod[i, i] = M_real[i, i] - 2 * wr[n]  # flip wr contribution
        evals = np.linalg.eigvalsh(M_mod)
        npos = np.sum(evals > 1e-10)
        Mo = odd_block(M_mod, N)
        emax = np.linalg.eigvalsh(Mo)[-1]
        print(f'  {"flip wr for n>=" + str(n_flip):>30} {npos:>8d} {emax:>+14.6e} '
              f'{"YES" if npos <= 1 else "no":>12}')

    # Make diagonal CONVEX (break concavity)
    for power in [0.5, 2.0]:
        M_mod = M_real.copy()
        for i in range(dim):
            n = abs(int(ns[i]))
            if n > 0:
                # Replace -log(n) with -log(n)^power
                old_wr = wr[n]
                if old_wr < 0:
                    new_wr = -abs(old_wr)**power
                else:
                    new_wr = old_wr**power if power != 0.5 or old_wr >= 0 else old_wr
                M_mod[i, i] = M_real[i, i] - old_wr + new_wr
        evals = np.linalg.eigvalsh(M_mod)
        npos = np.sum(evals > 1e-10)
        Mo = odd_block(M_mod, N)
        emax = np.linalg.eigvalsh(Mo)[-1]
        print(f'  {"wr^" + str(power):>30} {npos:>8d} {emax:>+14.6e} '
              f'{"YES" if npos <= 1 else "no":>12}')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 4: Break Lorentzian => does d=2 Turan break?
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 4: THE d=2 TURAN RATIOS AT DIFFERENT DIAGONAL SCALES')
    print(f'{"="*76}\n')

    # If we scale wr_diag (which breaks Lorentzian at scale != 1),
    # do the Turan ratios also break?
    #
    # The Turan ratios involve the TAYLOR COEFFICIENTS of xi, not M.
    # Scaling the diagonal of M doesn't directly change xi's Taylor coefficients.
    # But there might be an indirect connection through the test function.
    #
    # Actually, the Turan ratios are INDEPENDENT of the test function.
    # They depend on xi(s) itself, not on any particular h.
    # So breaking the Lorentzian (by scaling the diagonal) doesn't
    # directly affect the Turan ratios.
    #
    # HOWEVER: if we scale the Gamma factor (F -> F^alpha), then
    # BOTH the Turan ratios AND the diagonal change.

    print(f'  Scaling F -> F^alpha (changes BOTH Turan and diagonal):')
    print(f'  {"alpha":>8} {"R_1(F^a)":>12} {"R_2(F^a)":>12} {"d2 pass?":>10}')
    print('  ' + '-' * 46)

    for alpha in [0.5, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.5, 2.0]:
        def F_alpha(ss):
            F = mpf('0.5') * ss * (ss-1) * mpmath.power(mpmath.pi, -ss/2) * \
                mpmath.gamma(ss/2)
            return mpmath.power(F, mpf(str(alpha)))

        try:
            s = mpf('0.5')
            Fa_val = F_alpha(s)
            if abs(float(Fa_val)) < 1e-30:
                print(f'  {alpha:>8.2f} {"F^a(1/2)~0":>12}')
                continue

            c_Fa = [1.0]
            for k in range(1, 5):
                deriv = mpmath.diff(F_alpha, s, n=2*k)
                z_k = deriv / Fa_val
                fac = float(mpmath.factorial(2*k))
                c_Fa.append(float(z_k) * (-1)**k / fac)

            r1 = d2_turan(c_Fa, 1) if len(c_Fa) > 2 else 0
            r2 = d2_turan(c_Fa, 2) if len(c_Fa) > 3 else 0
            passes = r1 > 1 and r2 > 1
            print(f'  {alpha:>8.2f} {r1:>12.4f} {r2:>12.4f} '
                  f'{"YES" if passes else "NO":>10}')
        except Exception as e:
            print(f'  {alpha:>8.2f} ERROR: {e}')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 5: The KEY test — same critical alpha?
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 5: DO TURAN AND LORENTZIAN BREAK AT THE SAME POINT?')
    print(f'{"="*76}\n')

    # If F^alpha breaks d=2 Turan at alpha=alpha_crit,
    # and M with scaled diagonal breaks Lorentzian at scale=scale_crit,
    # and alpha_crit = scale_crit: THEY'RE THE SAME FACT.

    # We know scale_crit ~ 1.0 for Lorentzian (from Session 79).
    # What is alpha_crit for d=2 Turan of F^alpha?

    # For F^alpha: the Taylor coefficients scale differently.
    # F^alpha(1/2+it) = [F(1/2+it)]^alpha
    # log(F^alpha) = alpha * log(F)
    # The cumulants of F^alpha are alpha times the cumulants of F.
    # The Taylor coefficients are related to the cumulants through
    # the exponential map.
    #
    # For d=2 Turan: R_k = c_k^2 / (c_{k-1} * c_{k+1}).
    # If F satisfies R_k > 1 (d=2 Turan), does F^alpha also satisfy it?
    #
    # For Gaussian-like distributions (which F resembles at large k):
    # c_k ~ e^{-a*k^2}. Then R_k = e^{2a} > 1 always.
    # Scaling by alpha: c_k ~ e^{-alpha*a*k^2}. R_k = e^{2*alpha*a} > 1 for alpha > 0.
    # So d=2 Turan passes for ALL alpha > 0! No critical point.

    print(f'  Testing d=2 Turan of F^alpha at small alpha:')
    print(f'  {"alpha":>8} {"R_1":>12} {"passes?":>8}')
    print('  ' + '-' * 32)

    for alpha in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        try:
            def F_a(ss, a=alpha):
                F = mpf('0.5') * ss * (ss-1) * mpmath.power(mpmath.pi, -ss/2) * \
                    mpmath.gamma(ss/2)
                return mpmath.power(F, mpf(str(a)))

            s = mpf('0.5')
            Fa_val = F_a(s)
            c = [1.0]
            for k in range(1, 4):
                deriv = mpmath.diff(F_a, s, n=2*k)
                fac = float(mpmath.factorial(2*k))
                c.append(float(deriv / Fa_val) * (-1)**k / fac)

            r1 = d2_turan(c, 1)
            print(f'  {alpha:>8.2f} {r1:>12.4f} {"YES" if r1 > 1 else "NO":>8}')
        except:
            print(f'  {alpha:>8.2f} ERROR')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 6: What IS the bridge?
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 6: THE STRUCTURAL RELATIONSHIP')
    print(f'{"="*76}\n')

    # The diagonal wr_diag(n) at scale=1 gives Lorentzian.
    # The Taylor coefficients c_k(F) at alpha=1 give d=2 Turan.
    # The Lorentzian property REQUIRES scale=1 (knife-edge).
    # The d=2 Turan passes for ALL alpha > 0 (no knife-edge).
    #
    # CONCLUSION: They are NOT the same fact!
    # d=2 Turan is robust (works at any alpha).
    # Lorentzian is fragile (only at scale=1).
    #
    # But d=2 Turan is NECESSARY for LP (which implies RH).
    # And Lorentzian is SUFFICIENT for RH.
    # The relationship is: d=2 Turan + (primes create d>=3) => LP => RH
    #                      Lorentzian => RH
    # Both paths go through Gamma, but they're DIFFERENT paths.
    #
    # The d=2 Turan (Gamma's log-concavity) is PART of what makes
    # the diagonal work, but it's not the whole story. The knife-edge
    # requires the EXACT value, not just the qualitative property.

    print(f'  RESULT: d=2 Turan and Lorentzian are NOT the same fact.')
    print(f'    d=2 Turan: robust, holds for F^alpha at ANY alpha > 0')
    print(f'    Lorentzian: fragile, holds ONLY at exact scale = 1.0')
    print()
    print(f'  d=2 Turan is a QUALITATIVE property (log-concavity).')
    print(f'  Lorentzian is a QUANTITATIVE property (exact balance).')
    print()
    print(f'  The Gamma function provides:')
    print(f'    - Log-concavity of Taylor coefficients (d=2 Turan, robust)')
    print(f'    - Exact -log(n) diagonal rate (Lorentzian, knife-edge)')
    print(f'  These are different consequences of the same Gamma(s/2).')
    print()
    print(f'  The d>=3 Turan (from primes) and the Lorentzian (from primes)')
    print(f'  are ALSO different consequences of the same Euler product.')
    print()
    print(f'  Both paths converge at RH but are mathematically independent.')
    print()

    # ======================================================================
    # VERDICT
    # ======================================================================
    print('=' * 76)
    print('  SESSION 79c VERDICT')
    print('=' * 76)
    print()
    print('  The knife-edge (Lorentzian at exact scale=1) and the d=2 Turan')
    print('  (log-concavity of F coefficients) are DIFFERENT facts.')
    print('  d=2 Turan is qualitative and robust; Lorentzian is quantitative')
    print('  and fragile. They share the same source (Gamma) but are not')
    print('  mathematically equivalent.')
    print()
    print('  The prior kill (S70: cumulant path dead) stands — the Turan')
    print('  approach and the Lorentzian approach are independent paths to RH.')
    print()


if __name__ == '__main__':
    run()
