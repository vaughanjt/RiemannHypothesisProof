"""
SESSION 35 — PROVE M_diag DOMINATES M_prime ON orth(Poisson kernel)

THE CORE CLAIM (from Session 34):
  M = M_analytic + M_prime  where M_analytic = M_diag + M_alpha
  On orth(v_+), M_analytic is strongly negative (trace ~ -65)
  On orth(v_+), ||M_prime||_op ~ 1.75
  => Per-eigenvalue domination: each eigenvalue of M_analytic on orth(v_+)
     is more negative than -||M_prime||_op
  => M <= 0 on orth(v_+) without circularity

PHASE 1: Numerical characterization across lambda values
  - Eigenvalue spectra of M_analytic restricted to orth(v_+)
  - M_prime operator norm on orth(v_+)
  - Per-eigenvalue domination margins
  - Scaling behavior as lambda grows

PHASE 2: Asymptotic analysis
  - wr_diag[n] asymptotics via digamma
  - alpha[n] asymptotics
  - M_prime bound via PNT/partial summation

PHASE 3: Grok's deformation idea (Q_t = W02 - t*M)
  - Track eigenvalue flow along t in [0,1]
  - Check if smallest eigenvalue on null block is monotone
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, log, pi, exp, cos, sin, euler, digamma, hyp2f1, sinh
import time
import json
import sys
sys.path.insert(0, '.')
from connes_crossterm import build_all
from session33_sieve_bypass import compute_M_decomposition


def get_v_plus_and_projector(M, dim):
    """Extract the positive eigenvector v_+ and orthogonal projector."""
    evals, evecs = np.linalg.eigh(M)
    v_plus = evecs[:, -1]
    lambda_plus = evals[-1]
    P_orth = np.eye(dim) - np.outer(v_plus, v_plus)
    return v_plus, lambda_plus, P_orth


def restrict_to_orth(A, P_orth, tol=1e-12):
    """Restrict matrix A to orth(v_+), return non-zero eigenvalues."""
    A_orth = P_orth @ A @ P_orth
    evals = np.linalg.eigvalsh(A_orth)
    return evals[np.abs(evals) > tol]


def phase1_numerical_characterization(lam_sq_values):
    """
    For each lambda: compute eigenvalue spectra of M_analytic and M_prime
    restricted to orth(v_+), measure per-eigenvalue domination.
    """
    print("=" * 80)
    print("PHASE 1: NUMERICAL CHARACTERIZATION OF M_diag DOMINATION")
    print("=" * 80)

    results = []

    for lam_sq in lam_sq_values:
        L_f = np.log(lam_sq)
        N = max(15, round(6 * L_f))
        dim = 2 * N + 1
        t0 = time.time()

        print(f"\n{'-' * 70}")
        print(f"lam^2 = {lam_sq}, N = {N}, dim = {dim}")
        print(f"{'-' * 70}")

        # Build matrices
        W02, M, QW = build_all(lam_sq, N)
        M_diag, M_alpha, M_prime, M_full, primes = compute_M_decomposition(lam_sq, N)
        M_analytic = M_diag + M_alpha  # The non-prime part

        # Get v_+ and projector
        v_plus, lambda_plus, P_orth = get_v_plus_and_projector(M, dim)

        # Eigenvalues of each component on orth(v_+)
        evals_analytic = restrict_to_orth(M_analytic, P_orth)
        evals_prime = restrict_to_orth(M_prime, P_orth)
        evals_diag = restrict_to_orth(M_diag, P_orth)
        evals_M = restrict_to_orth(M, P_orth)

        # Key metrics
        max_analytic = np.max(evals_analytic)
        min_analytic = np.min(evals_analytic)
        mean_analytic = np.mean(evals_analytic)
        prime_opnorm = np.max(np.abs(evals_prime))  # ||M_prime|| on orth(v_+)
        max_prime = np.max(evals_prime)

        # Per-eigenvalue domination margin
        # For M <= 0 on orth(v_+), need: lambda_i(M_analytic) + lambda_i(M_prime) < 0
        # Since eigenvalues aren't ordered the same way, use Weyl:
        # max(M) <= max(M_analytic) + max(M_prime) on orth(v_+)
        weyl_bound = max_analytic + max_prime

        # Stronger: every eigenvalue of M on orth(v_+) should be negative
        max_M_orth = np.max(evals_M)
        all_neg = max_M_orth < 1e-10

        # Domination ratio: how many times over does M_analytic dominate?
        domination_ratio = abs(mean_analytic) / prime_opnorm if prime_opnorm > 1e-15 else float('inf')

        # Per-eigenvalue: what fraction of M_analytic eigenvalues beat ||M_prime||?
        n_dominating = np.sum(evals_analytic < -prime_opnorm)
        frac_dominating = n_dominating / len(evals_analytic)

        # The HARDEST eigenvalue: the one closest to zero in M_analytic
        hardest_margin = -max_analytic - max_prime  # positive means domination holds

        elapsed = time.time() - t0

        print(f"  M_analytic on orth(v_+):")
        print(f"    max eigenvalue:  {max_analytic:>+12.6f}")
        print(f"    min eigenvalue:  {min_analytic:>+12.6f}")
        print(f"    mean eigenvalue: {mean_analytic:>+12.6f}")
        print(f"    trace:           {np.sum(evals_analytic):>+12.6f}")
        print(f"")
        print(f"  M_prime on orth(v_+):")
        print(f"    max eigenvalue:  {max_prime:>+12.6f}")
        print(f"    operator norm:   {prime_opnorm:>12.6f}")
        print(f"")
        print(f"  M_diag on orth(v_+):")
        print(f"    max eigenvalue:  {np.max(evals_diag):>+12.6f}")
        print(f"    min eigenvalue:  {np.min(evals_diag):>+12.6f}")
        print(f"")
        print(f"  DOMINATION METRICS:")
        print(f"    |mean(M_analytic)| / ||M_prime|| = {domination_ratio:.2f}x")
        print(f"    Eigenvalues beating ||M_prime||:   {n_dominating}/{len(evals_analytic)} ({frac_dominating:.1%})")
        print(f"    Hardest margin (max_anal + max_prime): {hardest_margin:>+.6f} {'DOMINATES' if hardest_margin > 0 else 'FAILS'}")
        print(f"    Weyl bound (max_anal + max_prime):     {weyl_bound:>+.6f}")
        print(f"    Full M on orth(v_+) max eigenvalue:    {max_M_orth:>+.4e}  {'ALL NEG' if all_neg else 'HAS POS'}")
        print(f"    Time: {elapsed:.1f}s")

        # Store sorted eigenvalue spectra
        results.append({
            'lam_sq': lam_sq,
            'dim': dim,
            'lambda_plus': float(lambda_plus),
            'evals_analytic_orth': sorted(evals_analytic.tolist()),
            'evals_prime_orth': sorted(evals_prime.tolist()),
            'evals_M_orth': sorted(evals_M.tolist()),
            'max_analytic': float(max_analytic),
            'prime_opnorm': float(prime_opnorm),
            'max_prime': float(max_prime),
            'domination_ratio': float(domination_ratio),
            'frac_dominating': float(frac_dominating),
            'hardest_margin': float(hardest_margin),
            'all_neg': bool(all_neg),
        })

    return results


def phase1b_eigenvalue_profiles(lam_sq_values):
    """
    Show the full sorted eigenvalue profiles side by side.
    M_analytic vs M_prime on orth(v_+) — the domination picture.
    """
    print("\n" + "=" * 80)
    print("PHASE 1b: EIGENVALUE PROFILES (sorted, restricted to orth(v_+))")
    print("=" * 80)

    for lam_sq in lam_sq_values:
        L_f = np.log(lam_sq)
        N = max(15, round(6 * L_f))
        dim = 2 * N + 1

        W02, M, QW = build_all(lam_sq, N)
        M_diag, M_alpha, M_prime, M_full, primes = compute_M_decomposition(lam_sq, N)
        M_analytic = M_diag + M_alpha

        v_plus, lambda_plus, P_orth = get_v_plus_and_projector(M, dim)
        evals_a = np.sort(restrict_to_orth(M_analytic, P_orth))
        evals_p = np.sort(restrict_to_orth(M_prime, P_orth))
        evals_M = np.sort(restrict_to_orth(M, P_orth))

        d = min(len(evals_a), len(evals_p), len(evals_M))
        evals_a = evals_a[:d]
        evals_p = evals_p[:d]
        evals_M = evals_M[:d]

        print(f"\nlam^2={lam_sq}, dim={dim}, orth dim={d}")
        print(f"  {'idx':>4} {'M_analytic':>12} {'M_prime':>12} {'M_full':>12} {'margin':>12}")
        for i in range(d):
            margin = -evals_a[i] - abs(evals_p[d-1-i])  # pessimistic pairing
            print(f"  {i:>4} {evals_a[i]:>+12.6f} {evals_p[i]:>+12.6f} {evals_M[i]:>+12.6f} {margin:>+12.6f}")


def phase2_digamma_asymptotics(lam_sq_values):
    """
    Derive and verify asymptotic formulas for wr_diag[n] and alpha[n].

    wr_diag[n] involves: w_const + integral of (exp(x/2)*omega_n(x) - omega_0)/(exp(x)-exp(-x)) dx
    where omega_n(x) = 2(1 - x/L) cos(2*pi*n*x/L)

    For large |n|: the cos(2*pi*n*x/L) oscillates rapidly.
    The integral ≈ -(the DC component of the integrand) for generic n.

    alpha[n] for large |n|:
    digamma(pi*i*|n|/L + 1/4) ~ log(pi*|n|/L) + O(L/|n|) for |n| >> L/pi
    """
    print("\n" + "=" * 80)
    print("PHASE 2: DIGAMMA ASYMPTOTICS FOR wr_diag[n] AND alpha[n]")
    print("=" * 80)

    for lam_sq in lam_sq_values:
        L = float(np.log(lam_sq))
        N = max(15, round(6 * L))

        print(f"\nlam^2 = {lam_sq}, L = {L:.4f}, N = {N}")

        # Compute exact values
        mp.dps = 50
        L_mp = log(mpf(lam_sq))
        eL = exp(L_mp)

        # wr_diag[n] exact values
        omega_0 = mpf(2)
        wr_exact = {}
        for nv in range(N + 1):
            def omega(x, nv=nv):
                return 2 * (1 - x / L_mp) * cos(2 * pi * nv * x / L_mp)
            w_const = (omega_0 / 2) * (euler + log(4 * pi * (eL - 1) / (eL + 1)))
            dx = L_mp / 10000
            integral = mpf(0)
            for k in range(10000):
                x = dx * (k + mpf(1) / 2)
                numer = exp(x / 2) * omega(x) - omega_0
                denom = exp(x) - exp(-x)
                if abs(denom) > mpf(10)**(-40):
                    integral += numer / denom
            integral *= dx
            wr_exact[nv] = float(w_const + integral)

        # alpha[n] exact values
        alpha_exact = {}
        for n in range(N + 1):
            if n == 0:
                alpha_exact[n] = 0.0
            else:
                z = exp(-2 * L_mp)
                a = pi * mpc(0, n) / L_mp + mpf(1) / 4
                h = hyp2f1(1, a, a + 1, z)
                f1 = exp(-L_mp / 2) * (2 * L_mp / (L_mp + 4 * pi * mpc(0, n)) * h).imag
                d = digamma(a).imag / 2
                alpha_exact[n] = float((f1 + d) / pi)

        # Display wr_diag profile
        print(f"\n  wr_diag[n] profile (exact values):")
        print(f"  {'n':>4} {'wr_diag[n]':>14} {'asymptotic':>14} {'ratio':>10}")

        # Asymptotic: for large n, the integral's cosine oscillation averages out.
        # Leading term: wr_diag[n] ≈ w_const + C_0 (DC contribution)
        # But more precisely: the DC part of the integrand for n=0 gives wr_diag[0]
        # For n != 0: oscillation kills most of the integral
        # Riemann-Lebesgue: integral decays as 1/|n| (integrand is smooth)

        w_const_f = float((omega_0 / 2) * (euler + log(4 * pi * (eL - 1) / (eL + 1))))

        for n in range(min(N + 1, 40)):
            # Asymptotic model: wr_diag[n] ≈ w_const + A/L * something that decays with n
            # For now just show exact values
            rl_asymp = wr_exact[0] / (1 + (2 * np.pi * n / L)**2) if n > 0 else wr_exact[0]
            ratio = wr_exact[n] / rl_asymp if abs(rl_asymp) > 1e-15 else float('inf')
            print(f"  {n:>4} {wr_exact[n]:>+14.8f} {rl_asymp:>+14.8f} {ratio:>10.4f}")

        # Display alpha[n] profile
        print(f"\n  alpha[n] profile:")
        print(f"  {'n':>4} {'alpha[n]':>14} {'digamma_asymp':>14} {'error':>12}")

        for n in range(1, min(N + 1, 40)):
            # Asymptotic: digamma(pi*i*n/L + 1/4) ≈ log(pi*n/L) + i*pi/2 for n >> L/pi
            # So Im(digamma)/2 ≈ pi/4
            # And the hyp2f1 term decays exponentially for large L
            asymp_alpha = np.pi / 4 / np.pi  # = 1/4 for large n
            err = alpha_exact[n] - asymp_alpha
            print(f"  {n:>4} {alpha_exact[n]:>+14.8f} {asymp_alpha:>+14.8f} {err:>+12.4e}")

    return wr_exact, alpha_exact


def phase2b_wr_diag_large_n_formula(lam_sq):
    """
    Derive the EXACT asymptotic behavior of wr_diag[n] for large |n|.

    The integral is:
        I(n) = integral_0^L [exp(x/2) * 2(1-x/L)cos(2*pi*n*x/L) - 2] / [exp(x)-exp(-x)] dx

    For large n: use integration by parts / stationary phase.
    The cos(2*pi*n*x/L) oscillates with period L/n.
    The rest of the integrand is smooth on [0, L].
    By Riemann-Lebesgue: I(n) = O(1/n) as n -> infinity.

    More precisely, integrate by parts once:
    integral f(x) cos(kx) dx = [f(x)sin(kx)/k]_0^L + integral f'(x)sin(kx)/k dx
    where k = 2*pi*n/L.

    Boundary terms vanish if f(0)=f(L)=0 (they don't quite).
    """
    L = float(np.log(lam_sq))
    N = max(15, round(6 * L))

    print(f"\n{'-' * 70}")
    print(f"PHASE 2b: wr_diag[n] LARGE-n FORMULA, lam^2={lam_sq}")
    print(f"{'-' * 70}")

    # Compute wr_diag[n] at high precision for validation
    mp.dps = 50
    L_mp = log(mpf(lam_sq))
    eL = exp(L_mp)
    omega_0 = mpf(2)
    w_const = float((omega_0 / 2) * (euler + log(4 * pi * (eL - 1) / (eL + 1))))

    # The integrand for the oscillatory part:
    # g(x) = [exp(x/2) * 2(1-x/L)] / [exp(x) - exp(-x)]
    # I(n) = integral_0^L g(x) cos(2*pi*n*x/L) dx - (DC integral)
    #
    # g(x) = exp(x/2) * (1-x/L) / sinh(x)  (since exp(x)-exp(-x) = 2sinh(x))
    # g(0+) = lim x->0 exp(x/2)(1-x/L)/(2sinh(x)) -> 1/(2x) * 1 ...
    # Actually g(x) = exp(x/2)(1-x/L)/sinh(x) has a singularity at x=0.
    # Near x=0: sinh(x) ~ x, so g(x) ~ (1/x)(1-x/L) -> 1/x.
    # The integrand is not smooth at x=0!

    # This means standard IBP doesn't directly give 1/n decay.
    # The singularity at x=0 generates a LOGARITHMIC contribution.

    # Split: I(n) = integral_0^delta + integral_delta^L
    # Near 0: g(x) ~ 1/x - 1/(2L) + (1/2 - 1/x + ...)
    # integral_0^delta (1/x) cos(kx) dx = -Ci(k*delta) + ...
    # where Ci is the cosine integral. For large k: Ci(k*delta) ~ sin(k*delta)/(k*delta)

    # Let's just measure the ACTUAL decay rate numerically.
    wr_vals = []
    for nv in range(N + 1):
        def omega(x, nv=nv):
            return 2 * (1 - x / L_mp) * cos(2 * pi * nv * x / L_mp)
        dx = L_mp / 10000
        integral = mpf(0)
        for k in range(10000):
            x = dx * (k + mpf(1) / 2)
            numer = exp(x / 2) * omega(x) - omega_0
            denom = exp(x) - exp(-x)
            if abs(denom) > mpf(10)**(-40):
                integral += numer / denom
        integral *= dx
        wr_vals.append(float(w_const + integral))

    # Fit decay: wr_diag[n] - wr_diag_inf ≈ A * n^(-alpha)
    # First, what's the limiting value?
    wr_inf = w_const  # Because integral -> 0 for large n by R-L

    print(f"\n  w_const = {w_const:.8f}")
    print(f"  wr_diag[0] = {wr_vals[0]:.8f}")
    print(f"  Oscillatory integral at n=0: {wr_vals[0] - w_const:.8f}")

    # Log-log regression for decay
    ns = np.arange(2, min(N + 1, 40))
    vals = np.array([abs(wr_vals[n] - w_const) for n in ns])
    valid = vals > 1e-15
    if np.sum(valid) > 3:
        log_n = np.log(ns[valid])
        log_v = np.log(vals[valid])
        # Linear fit
        coeffs = np.polyfit(log_n, log_v, 1)
        alpha_decay = -coeffs[0]
        A_decay = np.exp(coeffs[1])

        print(f"\n  Decay fit: |wr_diag[n] - w_const| ~ {A_decay:.4f} * n^(-{alpha_decay:.3f})")
        print(f"  (Expecting alpha=1 from R-L for 1/x singularity, or alpha=2 if smooth)")

        print(f"\n  {'n':>4} {'wr_diag[n]':>14} {'residual':>14} {'fit':>14} {'ratio':>10}")
        for n in ns[:20]:
            res = wr_vals[n] - w_const
            fit = A_decay * n**(-alpha_decay) * np.sign(res)
            ratio = res / fit if abs(fit) > 1e-15 else float('inf')
            print(f"  {n:>4} {wr_vals[n]:>+14.8f} {res:>+14.4e} {fit:>+14.4e} {ratio:>10.4f}")

    return wr_vals


def phase3_deformation_flow(lam_sq_values):
    """
    Grok's idea: Q_t = W02 - t*M for t in [0,1].
    Track eigenvalues on null(W02) as t increases from 0 to 1.
    If they never cross zero, Q_W >= 0 follows from continuity.
    """
    print("\n" + "=" * 80)
    print("PHASE 3: DEFORMATION FLOW Q_t = W02 - t*M")
    print("=" * 80)

    for lam_sq in lam_sq_values:
        L_f = np.log(lam_sq)
        N = max(15, round(6 * L_f))
        dim = 2 * N + 1

        print(f"\nlam^2 = {lam_sq}, dim = {dim}")

        W02, M, QW = build_all(lam_sq, N)

        # Get null(W02) projector
        ew, ev = np.linalg.eigh(W02)
        thresh = np.max(np.abs(ew)) * 1e-10
        P_null = ev[:, np.abs(ew) <= thresh]
        d_null = P_null.shape[1]

        # Track eigenvalues of Q_t restricted to null(W02)
        # On null(W02): Q_t = -t * M_null
        M_null = P_null.T @ M @ P_null
        evals_M_null = np.linalg.eigvalsh(M_null)

        t_vals = np.linspace(0, 1, 21)
        print(f"  null dim = {d_null}")
        print(f"  {'t':>6} {'min_eig(Q_t|null)':>18} {'max_eig(Q_t|null)':>18} {'n_pos':>6}")

        crossing_t = None
        for t in t_vals:
            evals_Qt = -t * evals_M_null  # Q_t|null = -t*M|null
            n_pos = np.sum(evals_Qt > 1e-12)
            min_e = np.min(evals_Qt)
            max_e = np.max(evals_Qt)
            flag = " <<<" if n_pos < d_null and t > 0 else ""
            print(f"  {t:>6.2f} {min_e:>+18.6f} {max_e:>+18.6f} {n_pos:>6}{flag}")

            if t > 0 and np.min(evals_Qt) < -1e-10 and crossing_t is None:
                crossing_t = t

        # The derivative: dQ_t/dt|null = -M|null
        # Eigenvalues of -M|null = -evals_M_null
        neg_M_evals = -evals_M_null
        print(f"\n  dQ_t/dt|null eigenvalues (= -M|null):")
        print(f"    min: {np.min(neg_M_evals):>+.6f}")
        print(f"    max: {np.max(neg_M_evals):>+.6f}")
        print(f"    All positive (M<=0 on null): {np.min(neg_M_evals) > -1e-10}")

        # ALSO: do the FULL deformation including range
        print(f"\n  Full Q_t spectrum tracking:")
        print(f"  {'t':>6} {'min_eig(Q_t)':>14} {'n_neg':>6}")
        for t in [0, 0.25, 0.5, 0.75, 1.0]:
            Qt = W02 - t * M
            Qt = (Qt + Qt.T) / 2
            evals_Qt_full = np.linalg.eigvalsh(Qt)
            n_neg = np.sum(evals_Qt_full < -1e-10)
            print(f"  {t:>6.2f} {np.min(evals_Qt_full):>+14.6f} {n_neg:>6}")


def phase3b_analytic_vs_prime_deformation(lam_sq):
    """
    Separate deformation: first add M_analytic, then M_prime.
    Q = W02 - M_analytic - M_prime

    Step 1: Q_a = W02 - M_analytic (analytic part only)
    Step 2: Q = Q_a - M_prime

    If Q_a is already "strongly positive" (large margin), and M_prime is small,
    this proves Q >= 0.
    """
    L_f = np.log(lam_sq)
    N = max(15, round(6 * L_f))
    dim = 2 * N + 1

    print(f"\n{'-' * 70}")
    print(f"PHASE 3b: ANALYTIC-THEN-PRIME DEFORMATION, lam^2={lam_sq}")
    print(f"{'-' * 70}")

    W02, M, QW = build_all(lam_sq, N)
    M_diag, M_alpha, M_prime, M_full, primes = compute_M_decomposition(lam_sq, N)
    M_analytic = M_diag + M_alpha

    # On the full space
    Q_analytic = W02 - M_analytic
    Q_analytic = (Q_analytic + Q_analytic.T) / 2
    Q_full = W02 - M_full
    Q_full = (Q_full + Q_full.T) / 2

    evals_Qa = np.linalg.eigvalsh(Q_analytic)
    evals_Qf = np.linalg.eigvalsh(Q_full)
    evals_Mp = np.linalg.eigvalsh(M_prime)

    min_Qa = np.min(evals_Qa)
    max_Mp = np.max(np.abs(evals_Mp))

    print(f"  Q_analytic = W02 - M_analytic:")
    print(f"    min eigenvalue: {min_Qa:>+.6f}")
    print(f"    n_negative:     {np.sum(evals_Qa < -1e-10)}")
    print(f"")
    print(f"  ||M_prime||_op: {max_Mp:.6f}")
    print(f"")
    print(f"  Margin: min(Q_analytic) - ||M_prime|| = {min_Qa - max_Mp:>+.6f}")
    print(f"  Q_full min eigenvalue:                  {np.min(evals_Qf):>+.6f}")

    if min_Qa > max_Mp:
        print(f"\n  *** ANALYTIC MARGIN EXCEEDS PRIME NORM ***")
        print(f"  *** Q_W = Q_analytic - M_prime >= {min_Qa - max_Mp:.6f} > 0 ***")
        print(f"  *** THIS IS A NON-CIRCULAR PROOF PATH ***")
    else:
        deficit = max_Mp - min_Qa
        print(f"\n  Deficit: {deficit:.6f} (prime norm exceeds analytic margin)")
        print(f"  Need to work in orth(v_+) to recover margin...")

        # Try on orth(v_+)
        v_plus, _, P_orth = get_v_plus_and_projector(M, dim)

        evals_Qa_orth = restrict_to_orth(Q_analytic, P_orth)
        evals_Mp_orth = restrict_to_orth(M_prime, P_orth)

        min_Qa_orth = np.min(evals_Qa_orth)
        max_Mp_orth = np.max(np.abs(evals_Mp_orth))

        print(f"\n  On orth(v_+):")
        print(f"    min(Q_analytic): {min_Qa_orth:>+.6f}")
        print(f"    ||M_prime||:     {max_Mp_orth:.6f}")
        print(f"    Margin:          {min_Qa_orth - max_Mp_orth:>+.6f}")

        if min_Qa_orth > max_Mp_orth:
            print(f"\n  *** ON ORTH(v_+): ANALYTIC MARGIN EXCEEDS PRIME NORM ***")
            print(f"  *** Combined with 2x2 range proof → Q_W >= 0 ***")


def phase4_random_test_vectors(lam_sq, n_samples=500):
    """
    Grok's immediate experiment: sample random phi perp v_+ and compute <phi, M phi>.
    Plot histogram. All should be negative.
    """
    L_f = np.log(lam_sq)
    N = max(15, round(6 * L_f))
    dim = 2 * N + 1

    print(f"\n{'-' * 70}")
    print(f"PHASE 4: RANDOM TEST VECTORS IN orth(v_+), lam^2={lam_sq}")
    print(f"{'-' * 70}")

    W02, M, QW = build_all(lam_sq, N)
    v_plus, lambda_plus, P_orth = get_v_plus_and_projector(M, dim)

    quadratic_forms = []
    for _ in range(n_samples):
        phi = np.random.randn(dim)
        phi = P_orth @ phi  # project to orth(v_+)
        phi = phi / np.linalg.norm(phi)
        qf = np.dot(phi, M @ phi)
        quadratic_forms.append(qf)

    qf = np.array(quadratic_forms)
    n_pos = np.sum(qf > 0)

    print(f"  Samples: {n_samples}")
    print(f"  Positive: {n_pos}/{n_samples}")
    print(f"  Max <phi, M phi>:  {np.max(qf):>+.6e}")
    print(f"  Min <phi, M phi>:  {np.min(qf):>+.6e}")
    print(f"  Mean <phi, M phi>: {np.mean(qf):>+.6e}")
    print(f"  Std:               {np.std(qf):.6e}")

    if n_pos == 0:
        margin = -np.max(qf)
        print(f"\n  *** ALL {n_samples} RANDOM VECTORS GIVE NEGATIVE QUADRATIC FORM ***")
        print(f"  *** Margin: {margin:.6e} ***")

    return qf


if __name__ == "__main__":
    print("SESSION 35 — M_diag DOMINATION PROOF")
    print("=" * 80)

    lam_sq_values = [50, 200, 500, 1000]

    # Phase 1: Numerical characterization
    results = phase1_numerical_characterization(lam_sq_values)

    # Phase 1b: Full eigenvalue profiles (smaller set for readability)
    phase1b_eigenvalue_profiles([50, 200])

    # Phase 2: Digamma asymptotics
    wr_vals, alpha_vals = phase2_digamma_asymptotics([200])

    # Phase 2b: Large-n formula
    phase2b_wr_diag_large_n_formula(200)

    # Phase 3: Deformation flow
    phase3_deformation_flow([50, 200, 1000])

    # Phase 3b: Analytic-then-prime deformation
    for ls in lam_sq_values:
        phase3b_analytic_vs_prime_deformation(ls)

    # Phase 4: Random test vectors
    for ls in lam_sq_values:
        phase4_random_test_vectors(ls)

    # Save results
    with open('session35_mdiag_domination.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to session35_mdiag_domination.json")
