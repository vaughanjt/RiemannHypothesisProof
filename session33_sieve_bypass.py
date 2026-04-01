"""
SESSION 33 — DIRECTION A: SIEVE THEORY BYPASS FOR CONNES eps_0 > 0

THE IDEA:
  Q_W = W_{0,2} - M  where M involves sums over prime powers p^k <= lambda^2.
  eps_0 > 0 iff Q_W is positive semidefinite.

  STRATEGY:
  1. Bound ||M||_op from above using prime number theorem + sieve theory
  2. Bound lambda_min(W_{0,2}) from below (it's rank 2, explicitly computable)
  3. If lambda_min(W_{0,2}) > ||M||_op, then Q_W > 0

  WHY SIEVE THEORY?
  The matrix M has entries involving sum_{p^k <= lam^2} log(p)/p^{k/2} * q(n,m,log(p^k)).
  The q function is oscillatory with frequency ~ n/L and m/L.
  Sieve theory (Selberg upper bound sieve) gives:
    sum_{p <= x} f(p) <= (2+o(1)) * integral / log(x)
  This is STRONGER than PNT for bounding oscillatory prime sums.

  THE CRITICAL TEST:
  Compute ratio = lambda_min(W_{0,2}) / ||M||_op for growing lambda.
  If ratio > 1 and STABLE (or growing), the sieve bypass works.
  If ratio -> 0, the bypass fails and we need a different approach.

  NEW IDEA — SELBERG SIEVE BOUND ON M:
  For the operator norm of M, we can use:
    ||M||_op <= ||M_diag||_op + ||M_offdiag||_op
  where M_diag involves wr_diag (Weil-explicit diagonal) and M_offdiag involves
  the alpha terms and prime sum cross-terms.

  The prime sum part of M_offdiag has entries:
    M_prime[n,m] = sum_{p^k <= lam^2} (log p / p^{k/2}) * q(n,m, log(p^k))

  By Selberg sieve: |sum_{p<=x} g(p)| <= C * x/log(x) * sup|g|
  This gives ||M_prime||_F <= C * pi(lam^2) * max|q| * sqrt(dim)
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, log, pi, exp, cos, sin, sinh, euler, digamma, hyp2f1
import time
import json
import sys
sys.path.insert(0, '.')
from connes_crossterm import build_all

mp.dps = 50


def selberg_upper_bound(x):
    """
    Selberg sieve upper bound: sum_{p <= x} 1 <= 2x/log(x) for x >= 2.
    More precisely: pi(x) <= (2 + epsilon) * x / log(x) for large x.
    For our purposes: sum_{p <= x} log(p)/p^{1/2} <= 2*sqrt(x) (by PNT + partial summation).
    """
    if x < 2:
        return 0
    return 2.0 * np.sqrt(x)


def prime_sum_bound_selberg(lam_sq, power=0.5):
    """
    Selberg-type upper bound for sum_{p^k <= lam^2} log(p) / p^{k*power}.

    By partial summation with PNT: theta(x) = sum_{p<=x} log(p) ~ x
    So sum_{p<=x} log(p)/p^a ~ x^{1-a}/(1-a) for a < 1.

    For prime powers: contribution from p^k with k >= 2 is O(sqrt(x)).
    """
    # Main term: primes (k=1)
    if power < 1:
        main = lam_sq**(1 - power) / (1 - power)
    else:
        main = np.log(lam_sq)

    # Higher powers: p^2 <= lam^2 means p <= lam, contribution O(lam^{1-2*power})
    higher = np.sqrt(lam_sq)**(1 - power) if power < 1 else np.log(np.sqrt(lam_sq))

    return main + higher


def compute_M_decomposition(lam_sq, N_val, n_quad=10000):
    """
    Build M and decompose into:
    - M_diag: diagonal part (Weil-explicit integral terms)
    - M_alpha: off-diagonal alpha terms
    - M_prime: pure prime power contribution

    Returns each component separately for analysis.
    """
    L = log(mpf(lam_sq))
    eL = exp(L)
    L_f = float(L)
    dim = 2 * N_val + 1

    # Compute prime powers
    limit = min(lam_sq, 10000)
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 2):
        if i <= limit and sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    vM = []
    for p in range(2, limit + 1):
        if sieve[p] and p <= lam_sq:
            pk = p
            while pk <= lam_sq:
                vM.append((pk, np.log(p), np.log(pk)))
                pk *= p

    def q_func(n, m, y):
        if n != m:
            return (np.sin(2*np.pi*m*y/L_f) - np.sin(2*np.pi*n*y/L_f)) / (np.pi*(n-m))
        else:
            return 2*(L_f - y)/L_f * np.cos(2*np.pi*n*y/L_f)

    # Compute alpha coefficients
    alpha = {}
    for n in range(-N_val, N_val + 1):
        if n == 0:
            alpha[n] = 0.0
        else:
            z = exp(-2*L)
            a = pi*mpc(0, abs(n))/L + mpf(1)/4
            h = hyp2f1(1, a, a+1, z)
            f1 = exp(-L/2) * (2*L/(L + 4*pi*mpc(0, abs(n)))*h).imag
            d = digamma(a).imag / 2
            val = float((f1 + d) / pi)
            alpha[n] = val if n > 0 else -val

    # Compute diagonal wr terms
    omega_0 = mpf(2)
    wr_diag = {}
    for nv in range(N_val + 1):
        def omega(x, nv=nv):
            return 2*(1 - x/L)*cos(2*pi*nv*x/L)
        w_const = (omega_0/2)*(euler + log(4*pi*(eL - 1)/(eL + 1)))
        dx = L/n_quad
        integral = mpf(0)
        for k in range(n_quad):
            x = dx*(k + mpf(1)/2)
            numer = exp(x/2)*omega(x) - omega_0
            denom = exp(x) - exp(-x)
            if abs(denom) > mpf(10)**(-40):
                integral += numer/denom
        integral *= dx
        wr_diag[nv] = float(w_const + integral)
        wr_diag[-nv] = wr_diag[nv]

    # Build three components of M
    M_diag_mat = np.zeros((dim, dim))  # diagonal wr terms
    M_alpha_mat = np.zeros((dim, dim))  # alpha off-diagonal
    M_prime_mat = np.zeros((dim, dim))  # prime power sums

    for i in range(dim):
        n = i - N_val
        M_diag_mat[i, i] = wr_diag[n]
        for j in range(dim):
            m = j - N_val
            if n != m:
                M_alpha_mat[i, j] = (alpha[m] - alpha[n]) / (n - m)
            M_prime_mat[i, j] = sum(
                lk * k**(-0.5) * q_func(n, m, logk)
                for k, lk, logk in vM
            )

    # Symmetrize
    M_alpha_mat = (M_alpha_mat + M_alpha_mat.T) / 2
    M_prime_mat = (M_prime_mat + M_prime_mat.T) / 2

    M_full = M_diag_mat + M_alpha_mat + M_prime_mat
    M_full = (M_full + M_full.T) / 2

    return M_diag_mat, M_alpha_mat, M_prime_mat, M_full, vM


def analyze_sieve_bypass(lam_sq_values):
    """
    For each lambda, compute:
    1. lambda_min(W_{0,2})  — the floor we need
    2. ||M||_op             — the ceiling we need to beat
    3. ||M_prime||_op       — the prime contribution alone
    4. Selberg bound on ||M_prime||_op
    5. The ratio and feasibility
    """
    results = []

    for lam_sq in lam_sq_values:
        L_f = np.log(lam_sq)
        N = max(21, round(8 * L_f))
        dim = 2 * N + 1

        t0 = time.time()

        # Build full matrices
        W02, M_full_check, QW = build_all(lam_sq, N)

        # Build decomposed M
        M_diag, M_alpha, M_prime, M_full, primes_used = compute_M_decomposition(lam_sq, N)

        # Eigenvalues
        evals_w02 = np.linalg.eigvalsh(W02)
        evals_m = np.linalg.eigvalsh(M_full)
        evals_qw = np.linalg.eigvalsh(QW)
        evals_mprime = np.linalg.eigvalsh(M_prime)

        # Key quantities
        w02_min = np.min(evals_w02[evals_w02 > 1e-15])  # smallest nonzero eigenvalue
        w02_max = np.max(evals_w02)
        w02_rank = np.sum(np.abs(evals_w02) > 1e-10)

        m_op_norm = np.max(np.abs(evals_m))
        m_prime_op = np.max(np.abs(evals_mprime))
        m_diag_op = np.max(np.abs(np.diag(M_diag)))
        m_alpha_op = np.max(np.abs(np.linalg.eigvalsh(M_alpha)))

        eps_0 = evals_qw[0]

        # Selberg bound on prime contribution
        selberg_bound = selberg_upper_bound(lam_sq)
        actual_prime_sum = sum(lk * k**(-0.5) for k, lk, _ in primes_used)

        # Frobenius norm of M_prime (easier to bound than operator norm)
        m_prime_frob = np.linalg.norm(M_prime, 'fro')

        # THE KEY RATIO: can W_{0,2} dominate M?
        # Since W_{0,2} has rank 2, its positive eigenspace is 2-dimensional
        # M acts on the full dim-dimensional space
        # For Q_W > 0, we need: on the null space of W_{0,2}, M must be negative semidefinite
        # On the range of W_{0,2}, we need W_{0,2} - M > 0

        # Project M onto null(W_{0,2}) and range(W_{0,2})
        w02_threshold = np.max(np.abs(evals_w02)) * 1e-10
        range_idx = np.where(np.abs(evals_w02) > w02_threshold)[0]
        null_idx = np.where(np.abs(evals_w02) <= w02_threshold)[0]

        evecs_w02 = np.linalg.eigh(W02)[1]
        P_range = evecs_w02[:, range_idx]
        P_null = evecs_w02[:, null_idx]

        # M restricted to null(W_{0,2})
        M_on_null_w02 = P_null.T @ M_full @ P_null
        evals_m_null = np.linalg.eigvalsh(M_on_null_w02)
        m_null_max = np.max(evals_m_null)  # must be <= 0 for Q_W >= 0 on this subspace
        m_null_min = np.min(evals_m_null)

        # M restricted to range(W_{0,2})
        M_on_range = P_range.T @ M_full @ P_range
        W02_on_range = P_range.T @ W02 @ P_range
        QW_on_range = W02_on_range - M_on_range
        evals_qw_range = np.linalg.eigvalsh(QW_on_range)

        # SIEVE BOUND TEST: M_prime on null(W_{0,2})
        M_prime_null = P_null.T @ M_prime @ P_null
        evals_mprime_null = np.linalg.eigvalsh(M_prime_null)

        elapsed = time.time() - t0

        r = {
            'lam_sq': lam_sq,
            'dim': dim,
            'N': N,
            'eps_0': float(eps_0),
            'w02_min_nonzero': float(w02_min),
            'w02_max': float(w02_max),
            'w02_rank': int(w02_rank),
            'm_op_norm': float(m_op_norm),
            'm_prime_op': float(m_prime_op),
            'm_diag_op': float(m_diag_op),
            'm_alpha_op': float(m_alpha_op),
            'm_prime_frob': float(m_prime_frob),
            'actual_prime_sum': float(actual_prime_sum),
            'selberg_bound': float(selberg_bound),
            'n_prime_powers': len(primes_used),
            'm_null_max': float(m_null_max),  # CRITICAL: must be <= 0
            'm_null_min': float(m_null_min),
            'qw_range_min': float(evals_qw_range[0]),  # Q_W on range(W02)
            'mprime_null_max': float(np.max(evals_mprime_null)),
            'mprime_null_min': float(np.min(evals_mprime_null)),
            'elapsed': elapsed
        }
        results.append(r)

        print(f"\nlam^2={lam_sq:>5} (dim={dim}, {elapsed:.1f}s)")
        print(f"  W02: rank={w02_rank}, min_nz={w02_min:.4e}, max={w02_max:.4e}")
        print(f"  M:   ||M||={m_op_norm:.4e}  (diag={m_diag_op:.4e} alpha={m_alpha_op:.4e} prime={m_prime_op:.4e})")
        print(f"  eps_0 = {eps_0:.4e}")
        print(f"  ---")
        print(f"  M on null(W02): max_eig={m_null_max:.4e} min_eig={m_null_min:.4e}")
        if m_null_max > 1e-10:
            print(f"  *** M POSITIVE on null(W02) — this is the obstruction! ***")
            print(f"      (need M <= 0 on null(W02) for Q_W >= 0 there)")
        else:
            print(f"  M <= 0 on null(W02) — GOOD (sieve bypass possible here)")
        print(f"  Q_W on range(W02): min_eig={evals_qw_range[0]:.4e}")
        print(f"  ---")
        print(f"  Prime sums: actual={actual_prime_sum:.4f}, Selberg bound={selberg_bound:.4f}")
        print(f"  M_prime on null(W02): [{np.min(evals_mprime_null):.4e}, {np.max(evals_mprime_null):.4e}]")

    return results


def deep_null_space_analysis(lam_sq, N=None):
    """
    Deep dive into M restricted to null(W_{0,2}).

    Key question: WHY is M positive on null(W_{0,2})?
    Can sieve bounds on the prime sum make it negative?

    The matrix M = M_diag + M_alpha + M_prime.
    On null(W_{0,2}):
    - M_diag contribution: the wr_diag values projected onto null space
    - M_alpha contribution: the alpha cross-terms
    - M_prime contribution: the prime sum — THIS is what sieve theory bounds

    If M_diag + M_alpha is already positive on null(W_{0,2}),
    the prime sum can't save us (sieve gives upper bounds, not sign changes).

    But if M_diag + M_alpha < 0 on null(W_{0,2}), we need:
    ||M_prime_null|| < |lambda_min(M_diag_null + M_alpha_null)|
    """
    if N is None:
        L_f = np.log(lam_sq)
        N = max(21, round(8 * L_f))
    dim = 2 * N + 1

    print(f"\nDEEP NULL SPACE ANALYSIS: lam^2={lam_sq}, N={N}, dim={dim}")
    print("=" * 70)

    W02, M_check, QW = build_all(lam_sq, N)
    M_diag, M_alpha, M_prime, M_full, primes_used = compute_M_decomposition(lam_sq, N)

    # Get null space of W_{0,2}
    evals_w02, evecs_w02 = np.linalg.eigh(W02)
    threshold = np.max(np.abs(evals_w02)) * 1e-10
    null_idx = np.where(np.abs(evals_w02) <= threshold)[0]
    P_null = evecs_w02[:, null_idx]
    D_null = len(null_idx)

    print(f"  null(W02) dimension: {D_null} of {dim}")

    # Project each component onto null(W02)
    M_diag_null = P_null.T @ M_diag @ P_null
    M_alpha_null = P_null.T @ M_alpha @ P_null
    M_prime_null = P_null.T @ M_prime @ P_null
    M_full_null = P_null.T @ M_full @ P_null

    ev_diag = np.linalg.eigvalsh(M_diag_null)
    ev_alpha = np.linalg.eigvalsh(M_alpha_null)
    ev_prime = np.linalg.eigvalsh(M_prime_null)
    ev_full = np.linalg.eigvalsh(M_full_null)
    ev_nopr = np.linalg.eigvalsh(M_diag_null + M_alpha_null)

    print(f"\n  Component spectra on null(W02):")
    print(f"    M_diag:         [{ev_diag[0]:.4e}, {ev_diag[-1]:.4e}]  trace={np.trace(M_diag_null):.4e}")
    print(f"    M_alpha:        [{ev_alpha[0]:.4e}, {ev_alpha[-1]:.4e}]  trace={np.trace(M_alpha_null):.4e}")
    print(f"    M_prime:        [{ev_prime[0]:.4e}, {ev_prime[-1]:.4e}]  trace={np.trace(M_prime_null):.4e}")
    print(f"    M_diag+alpha:   [{ev_nopr[0]:.4e}, {ev_nopr[-1]:.4e}]")
    print(f"    M_full:         [{ev_full[0]:.4e}, {ev_full[-1]:.4e}]")

    # THE KEY TEST: is M_diag + M_alpha already positive?
    if ev_nopr[-1] > 1e-10:
        print(f"\n  *** M_diag + M_alpha has POSITIVE eigenvalues on null(W02) ***")
        print(f"      Max eigenvalue: {ev_nopr[-1]:.4e}")
        print(f"      This means even WITHOUT primes, M can be positive here.")
        print(f"      Sieve bounds on M_prime CANNOT fix this alone.")
        print(f"      Need: M_prime must be sufficiently NEGATIVE to compensate.")

        # Check if M_prime is negative enough
        needed = ev_nopr[-1]
        available = -ev_prime[0]  # most negative eigenvalue of M_prime
        print(f"      Need M_prime min_eig < -{needed:.4e}")
        print(f"      Have M_prime min_eig = {ev_prime[0]:.4e}")
        if available > needed:
            print(f"      *** POSSIBLE: prime cancellation compensates! ***")
        else:
            print(f"      Gap: need {needed:.4e} more negative contribution")
    elif ev_nopr[-1] < -1e-10:
        print(f"\n  M_diag + M_alpha is NEGATIVE on null(W02) — sieve bypass viable!")
        print(f"  Need ||M_prime_null|| < {abs(ev_nopr[-1]):.4e}")
        print(f"  Have ||M_prime_null|| = {np.max(np.abs(ev_prime)):.4e}")

    # Trace analysis: where does the positivity come from?
    print(f"\n  Trace decomposition on null(W02):")
    print(f"    tr(M_diag_null) = {np.trace(M_diag_null):.6e}")
    print(f"    tr(M_alpha_null) = {np.trace(M_alpha_null):.6e}")
    print(f"    tr(M_prime_null) = {np.trace(M_prime_null):.6e}")
    print(f"    tr(M_full_null) = {np.trace(M_full_null):.6e}")
    print(f"    tr(Q_W_null) = {np.trace(P_null.T @ QW @ P_null):.6e}")

    # Per-prime-power contribution analysis
    print(f"\n  Individual prime power contributions to tr(M_prime_null):")
    total_trace = 0
    prime_contribs = []
    for pk, logp, logpk in primes_used[:20]:  # first 20
        # Build the rank-1 (or low-rank) contribution from this prime power
        M_pk = np.zeros((dim, dim))
        for i in range(dim):
            n = i - N
            for j in range(dim):
                m = j - N
                if n != m:
                    q = (np.sin(2*np.pi*m*logpk/np.log(lam_sq)) -
                         np.sin(2*np.pi*n*logpk/np.log(lam_sq))) / (np.pi*(n-m))
                else:
                    q = 2*(np.log(lam_sq) - logpk)/np.log(lam_sq) * np.cos(2*np.pi*n*logpk/np.log(lam_sq))
                M_pk[i, j] = logp * pk**(-0.5) * q
        M_pk = (M_pk + M_pk.T) / 2
        M_pk_null = P_null.T @ M_pk @ P_null
        tr_pk = np.trace(M_pk_null)
        total_trace += tr_pk
        prime_contribs.append((pk, logp, tr_pk))
        if pk <= 30 or abs(tr_pk) > 0.01:
            print(f"    p^k={pk:>5} (log p={logp:.3f}): tr contribution = {tr_pk:+.6e}")

    print(f"    ... total from all: {np.trace(M_prime_null):.6e}")

    return {
        'ev_diag': ev_diag,
        'ev_alpha': ev_alpha,
        'ev_prime': ev_prime,
        'ev_nopr': ev_nopr,
        'ev_full': ev_full,
        'prime_contribs': prime_contribs
    }


def test_modified_sieve_approach(lam_sq_values):
    """
    ALTERNATIVE SIEVE IDEA:

    Instead of bounding ||M|| directly, use the Selberg sieve to construct
    a MINORANT for the Weil distribution.

    The Weil explicit formula: for f >= 0 with compact support,
      sum_rho f_hat(rho) = f_hat(0) + f_hat(1) - sum_{p^k} Lambda(p^k)/p^{k/2} * (f(klogp) + f(-klogp)) + ...

    Selberg sieve: for theta(x) = sum_{p<=x} log(p),
      theta(x) <= (1+eps) * x  for x > x_0(eps)

    This means the prime sum is bounded:
      |sum_{p^k <= X} Lambda(p^k) * g(p^k)| <= (1+eps) * integral of |g| + O(sqrt(X))

    If the integral of |g| over the sieve range is smaller than f_hat(0) + f_hat(1),
    then the Weil distribution is positive on f.

    COMPUTE: for various lambda, what's the ratio
      (f_hat(0) + f_hat(1)) / integral_bound ?
    """
    print("\n\n" + "=" * 75)
    print("MODIFIED SIEVE: WEIL POSITIVITY VIA PRIME SUM BOUNDING")
    print("=" * 75)
    print()
    print("For f(x) = cos(2*pi*n*x/L)^2 (bandwidth-limited test function):")
    print("  Weil(f) = analytic_terms - prime_sum")
    print("  If |prime_sum| < analytic_terms, Weil(f) > 0.")
    print()

    for lam_sq in lam_sq_values:
        L = np.log(lam_sq)

        # The analytic part: W_{0,2} contribution for a generic test function
        # For f(x) = cos(2*pi*x/L), the analytic contribution is:
        # W_{0,2}(f,f) ~ 32*L*sinh^2(L/4) * L^2 / (L^2 + 4*pi^2)^2
        sinh_term = float(mpmath.sinh(mpf(L)/4))**2
        w02_contribution = 32 * L * sinh_term * L**2 / (L**2 + (2*np.pi)**2)**2

        # The prime sum part: sum_{p^k <= lam^2} log(p)/p^{k/2} * |q(1,0,logpk)|
        # q(1,0,y) = sin(2*pi*y/L) / pi  for n=1, m=0
        prime_sum = 0
        limit = min(lam_sq, 10000)
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 2):
            if i <= limit and sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        for p in range(2, limit + 1):
            if sieve[p] and p <= lam_sq:
                pk = p
                while pk <= lam_sq:
                    logpk = np.log(pk)
                    logp = np.log(p)
                    q_val = np.sin(2*np.pi*logpk/L) / np.pi
                    prime_sum += logp * pk**(-0.5) * abs(q_val)
                    pk *= p

        # Selberg bound: sum_{p^k} Lambda(p^k)/p^{k/2} * |sin(...)| <= 2*sqrt(lam^2)/(pi) + O(...)
        selberg = 2 * np.sqrt(lam_sq) / np.pi

        # PNT bound: theta(x) ~ x, so sum log(p)/sqrt(p) ~ 2*sqrt(x)
        pnt_bound = 2 * np.sqrt(lam_sq)

        ratio_actual = w02_contribution / prime_sum if prime_sum > 0 else float('inf')
        ratio_selberg = w02_contribution / selberg if selberg > 0 else float('inf')

        print(f"  lam^2={lam_sq:>5}: W02={w02_contribution:.4e}  "
              f"prime_sum={prime_sum:.4e}  selberg={selberg:.4e}")
        print(f"    ratio(actual)={ratio_actual:.4f}  ratio(selberg)={ratio_selberg:.4f}")
        if ratio_actual > 1:
            print(f"    W02 DOMINATES (actual) -- Weil positive on this test function")
        if ratio_selberg > 1:
            print(f"    *** W02 DOMINATES (Selberg bound) -- PROVABLE! ***")

    return


def spectral_gap_sieve_connection(lam_sq_values):
    """
    THIRD IDEA: Connect eps_0 directly to prime distribution gaps.

    The minimal eigenvector xi_0 of Q_W satisfies:
      eps_0 = <xi_0, W_{0,2} xi_0> - <xi_0, M xi_0>

    The second term decomposes as:
      <xi_0, M xi_0> = sum_{p^k} Lambda(p^k)/p^{k/2} * |sum_n xi_0[n] * q(n, logpk)|^2

    This is a POSITIVE sum (each term is a square!).
    So eps_0 = w02_part - (positive prime-weighted sum of squares).

    KEY: The prime-weighted sum is bounded by Selberg sieve.
    If we can show w02_part > Selberg_bound for ALL unit vectors xi_0,
    then eps_0 > 0.

    But w02_part depends on xi_0 through a rank-2 quadratic form,
    so this is only large when xi_0 has significant overlap with range(W02).
    For generic xi_0 in null(W02), w02_part = 0.

    THIS IS THE FUNDAMENTAL OBSTRUCTION:
    W_{0,2} has rank 2, M has full rank.
    On the (dim-2)-dimensional null space of W_{0,2}, eps_0 > 0 requires M <= 0.
    M <= 0 on null(W_{0,2}) is a statement about prime distribution.
    """
    print("\n\n" + "=" * 75)
    print("SPECTRAL GAP — PRIME DISTRIBUTION CONNECTION")
    print("=" * 75)

    for lam_sq in lam_sq_values:
        L_f = np.log(lam_sq)
        N = max(21, round(8 * L_f))
        dim = 2 * N + 1

        W02, M, QW = build_all(lam_sq, N)
        evals_qw, evecs_qw = np.linalg.eigh(QW)
        xi_0 = evecs_qw[:, 0]
        eps_0 = evals_qw[0]

        # Decompose eps_0 = w02_part - m_part
        w02_part = xi_0 @ W02 @ xi_0
        m_part = xi_0 @ M @ xi_0

        # Further decompose m_part into individual prime power contributions
        M_diag, M_alpha, M_prime, M_full, primes_used = compute_M_decomposition(lam_sq, N)

        m_diag_part = xi_0 @ M_diag @ xi_0
        m_alpha_part = xi_0 @ M_alpha @ xi_0
        m_prime_part = xi_0 @ M_prime @ xi_0

        # The prime part: is it a sum of positive terms?
        # M_prime[i,j] = sum_pk Lambda(pk)/pk^{1/2} * q(n,m,logpk)
        # <xi, M_prime xi> = sum_pk Lambda(pk)/pk^{1/2} * sum_{n,m} xi[n]*xi[m]*q(n,m,logpk)
        # The inner sum is NOT necessarily |...|^2 because q(n,m,y) is not a rank-1 kernel

        # Compute per-prime contribution to <xi_0, M_prime xi_0>
        prime_contributions = []
        for pk, logp, logpk in primes_used:
            contrib = 0
            for i in range(dim):
                n = i - N
                for j in range(dim):
                    m = j - N
                    if n != m:
                        q = (np.sin(2*np.pi*m*logpk/L_f) -
                             np.sin(2*np.pi*n*logpk/L_f)) / (np.pi*(n-m))
                    else:
                        q = 2*(L_f - logpk)/L_f * np.cos(2*np.pi*n*logpk/L_f)
                    contrib += xi_0[i] * xi_0[j] * logp * pk**(-0.5) * q
            prime_contributions.append((pk, contrib))

        total_prime = sum(c for _, c in prime_contributions)
        positive_primes = sum(c for _, c in prime_contributions if c > 0)
        negative_primes = sum(c for _, c in prime_contributions if c < 0)

        print(f"\nlam^2={lam_sq}: eps_0 = {eps_0:.6e}")
        print(f"  = W02_part({w02_part:.6e}) - M_part({m_part:.6e})")
        print(f"  M decomposition:")
        print(f"    diag:  {m_diag_part:.6e}")
        print(f"    alpha: {m_alpha_part:.6e}")
        print(f"    prime: {m_prime_part:.6e} (check: {total_prime:.6e})")
        print(f"  Prime contributions: +{positive_primes:.4e} / {negative_primes:.4e}")
        print(f"    Net prime: {total_prime:.4e}")
        print(f"    Sign pattern: {sum(1 for _,c in prime_contributions if c > 0)} pos, "
              f"{sum(1 for _,c in prime_contributions if c < 0)} neg")

        # Show largest individual contributions
        sorted_contribs = sorted(prime_contributions, key=lambda x: abs(x[1]), reverse=True)
        print(f"  Top 5 prime contributions:")
        for pk, c in sorted_contribs[:5]:
            print(f"    p^k={pk:>5}: {c:+.6e}")


if __name__ == "__main__":
    print("SESSION 33 — DIRECTION A: SIEVE THEORY BYPASS")
    print("=" * 75)
    print()
    print("Can we prove Connes eps_0 > 0 using sieve bounds on prime sums?")
    print("Key: Q_W = W_{0,2} - M, need Q_W >= 0.")
    print("W_{0,2} has rank 2. On null(W_{0,2}), need M <= 0.")
    print()

    lam_sq_values = [50, 100, 200, 500, 1000, 2000]

    # Part 1: Main analysis
    print("\n" + "=" * 75)
    print("PART 1: OPERATOR NORM COMPARISON")
    print("=" * 75)
    results = analyze_sieve_bypass(lam_sq_values)

    # Part 2: Deep null space analysis at key lambda values
    print("\n\n" + "=" * 75)
    print("PART 2: DEEP NULL SPACE DECOMPOSITION")
    print("=" * 75)
    for lam_sq in [200, 1000]:
        deep_null_space_analysis(lam_sq)

    # Part 3: Modified sieve via Weil positivity
    test_modified_sieve_approach(lam_sq_values)

    # Part 4: Per-prime spectral decomposition
    spectral_gap_sieve_connection([200, 1000])

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n\n" + "=" * 75)
    print("SESSION 33A SUMMARY: SIEVE BYPASS FEASIBILITY")
    print("=" * 75)

    print("\nKey findings:")
    print()

    # Check the main diagnostic: is M positive on null(W02)?
    all_m_null_positive = all(r['m_null_max'] > 1e-10 for r in results)
    if all_m_null_positive:
        print("  OBSTRUCTION IDENTIFIED:")
        print("  M has POSITIVE eigenvalues on null(W_{0,2}) for all tested lambda.")
        print("  This means W_{0,2} alone cannot dominate M — the positivity of Q_W")
        print("  relies on CANCELLATION between M components, not W_{0,2} dominance.")
        print()
        print("  Simple sieve bypass (bound ||M|| < lambda_min(W02)) FAILS")
        print("  because W02 has rank 2 while M is full-rank.")
        print()
        print("  HOWEVER: the question becomes whether the PRIME part of M")
        print("  is what makes M positive on null(W02), or if it's the")
        print("  analytic parts (diag + alpha) that are responsible.")
    else:
        print("  M is NEGATIVE on null(W_{0,2}) — sieve bypass may work!")

    print()
    for r in results:
        print(f"  lam^2={r['lam_sq']:>5}: eps_0={r['eps_0']:.3e}  "
              f"M_null_max={r['m_null_max']:.3e}  M_null_min={r['m_null_min']:.3e}")

    # Save results
    with open('session33_sieve_bypass.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to session33_sieve_bypass.json")
