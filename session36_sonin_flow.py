"""
SESSION 36 -- SONIN MONOTONICITY FLOW

Grok's new path: As bandwidth L = log(lambda^2) increases, the compressed
quadratic form Q_L^perp on the discrete Sonin slice (null(W02) intersect
orth(v_+)) should flow monotonically downward (stay <= 0).

KEY DISTINCTION from the cumulative build (which failed):
- Cumulative build: fixed lambda, added primes by weight order
- Sonin flow: lambda GROWS, so W02, v_+, null space, kernel shape ALL co-evolve
  with the prime content. The geometry and arithmetic change together.

EXPERIMENT:
1. Sweep lambda^2 from 4 to 2000
2. At each lambda^2, compute:
   - max eigenvalue of M restricted to null(W02) (= -min eigenvalue of Q_W on null)
   - max eigenvalue of M restricted to orth(v_+)
   - These should be <= 0
3. Track the FLOW: does the max eigenvalue stay negative?
4. At each prime power threshold, compute the DERIVATIVE (finite difference)
5. Check sign of the incremental update
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, log, pi, exp, cos, sin, euler, digamma, hyp2f1, sinh
import time
import json
import sys
sys.path.insert(0, '.')


def build_all_fast(lam_sq, N_val, n_quad=5000):
    """Build W02, M, QW — faster version with reduced quadrature."""
    mp.dps = 30
    L = log(mpf(lam_sq))
    eL = exp(L)
    L_f = float(L)
    dim = 2 * N_val + 1

    # Prime powers up to lam_sq
    limit = min(int(lam_sq), 10000)
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

    # W02
    L2_f = L_f**2
    p2_f = (4*np.pi)**2
    pf_f = 32*L_f*float(sinh(L/4))**2

    W02 = np.zeros((dim, dim))
    for i in range(dim):
        n = i - N_val
        for j in range(dim):
            m = j - N_val
            W02[i, j] = pf_f*(L2_f - p2_f*m*n) / ((L2_f + p2_f*m**2)*(L2_f + p2_f*n**2))

    # Alpha coefficients
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

    # Diagonal wr terms
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

    # Build M
    M = np.zeros((dim, dim))
    for i in range(dim):
        n = i - N_val
        M[i, i] = wr_diag[n]
        for j in range(dim):
            m = j - N_val
            if n != m:
                M[i, j] += (alpha[m] - alpha[n]) / (n - m)
            M[i, j] += sum(lk * k**(-0.5) * q_func(n, m, logk) for k, lk, logk in vM)
    M = (M + M.T) / 2

    QW = W02 - M
    QW = (QW + QW.T) / 2

    return W02, M, QW, len(vM)


def sonin_flow(lam_sq_values):
    """
    THE EXPERIMENT: track max eigenvalue of M on null(W02) as lambda grows.

    At each lambda^2:
    - Build W02, M, QW with appropriate N
    - Compute null(W02) projector
    - Compute max eigenvalue of M restricted to null(W02)
    - Also compute v_+ and max eigenvalue of M on orth(v_+) intersect null(W02)
    """
    print("SONIN MONOTONICITY FLOW")
    print("=" * 80)
    print(f"  {'lam^2':>8} {'L':>6} {'dim':>4} {'#pk':>4} "
          f"{'max_eig(M|null)':>16} {'max_eig(M|orth)':>16} {'QW>=0?':>7} {'time':>6}")

    results = []
    prev_max = None

    for lam_sq in lam_sq_values:
        L_f = np.log(lam_sq)
        N = max(5, min(round(4 * L_f), 30))  # Smaller N for speed in the flow
        dim = 2 * N + 1

        t0 = time.time()
        W02, M, QW, n_primes = build_all_fast(lam_sq, N)
        elapsed = time.time() - t0

        # null(W02)
        ew, ev = np.linalg.eigh(W02)
        thresh = np.max(np.abs(ew)) * 1e-10
        P_null = ev[:, np.abs(ew) <= thresh]
        d_null = P_null.shape[1]

        if d_null == 0:
            print(f"  {lam_sq:>8.1f} {L_f:>6.3f} {dim:>4} {n_primes:>4} "
                  f"{'N/A':>16} {'N/A':>16} {'N/A':>7} {elapsed:>5.1f}s")
            continue

        # Max eigenvalue of M on null(W02)
        M_null = P_null.T @ M @ P_null
        evals_null = np.linalg.eigvalsh(M_null)
        max_null = np.max(evals_null)

        # v_+ and orth projector
        evals_M, evecs_M = np.linalg.eigh(M)
        v_plus = evecs_M[:, -1]
        P_orth = np.eye(dim) - np.outer(v_plus, v_plus)

        # Max eigenvalue of M on orth(v_+)
        M_orth = P_orth @ M @ P_orth
        evals_orth = np.linalg.eigvalsh(M_orth)
        evals_orth_nz = evals_orth[np.abs(evals_orth) > 1e-12]
        max_orth = np.max(evals_orth_nz) if len(evals_orth_nz) > 0 else 0

        qw_ok = max_null < 1e-8

        # Direction of flow
        direction = ""
        if prev_max is not None:
            delta = max_null - prev_max
            if delta < -1e-12:
                direction = " v"  # flowing down
            elif delta > 1e-12:
                direction = " ^"  # flowing UP (bad)
            else:
                direction = " ="

        print(f"  {lam_sq:>8.1f} {L_f:>6.3f} {dim:>4} {n_primes:>4} "
              f"{max_null:>+16.6e} {max_orth:>+16.6e} {'YES' if qw_ok else 'NO':>7} "
              f"{elapsed:>5.1f}s{direction}")

        prev_max = max_null

        results.append({
            'lam_sq': float(lam_sq),
            'L': float(L_f),
            'dim': dim,
            'n_primes': n_primes,
            'max_eig_null': float(max_null),
            'max_eig_orth': float(max_orth),
            'qw_ok': bool(qw_ok),
        })

    return results


def prime_power_thresholds(lam_sq_max):
    """Generate lambda^2 values at prime power boundaries."""
    # All prime powers up to lam_sq_max
    limit = int(lam_sq_max)
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 2):
        if i <= limit and sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False

    pps = set()
    for p in range(2, limit + 1):
        if sieve[p]:
            pk = p
            while pk <= lam_sq_max:
                pps.add(pk)
                pk *= p

    # Just after each prime power threshold
    thresholds = sorted(pps)
    return [pp + 0.5 for pp in thresholds]


def derivative_at_thresholds(lam_sq_max=500):
    """
    Compute the derivative dQ_L^perp / dL at each prime power threshold.

    When lambda^2 crosses a prime power p^k, a new T(p^k) term enters M.
    The change in M is delta_M = log(p)/sqrt(p^k) * T(p^k).
    The change in max eigenvalue of M|null(W02) should be <= 0
    (i.e., the new prime power pushes M further negative on the null block).
    """
    thresholds = prime_power_thresholds(lam_sq_max)

    print(f"\nDERIVATIVE AT PRIME POWER THRESHOLDS")
    print(f"  Testing lambda^2 just before and after each prime power p^k")
    print(f"  {'p^k':>6} {'max_before':>14} {'max_after':>14} {'delta':>12} {'sign':>6}")

    n_neg = 0
    n_pos = 0
    n_total = 0

    # We compute at lambda^2 = pp - 0.5 and pp + 0.5
    prev_lam = None
    prev_max = None

    for pp_val in thresholds:
        if pp_val < 4:
            continue
        if pp_val > lam_sq_max:
            break

        lam_before = pp_val - 0.5
        lam_after = pp_val + 0.5

        L_f = np.log(lam_after)
        N = max(5, min(round(4 * L_f), 25))

        try:
            _, M_before, _, _ = build_all_fast(lam_before, N, n_quad=3000)
            _, M_after, _, _ = build_all_fast(lam_after, N, n_quad=3000)

            # Use W02 from after (they're nearly identical)
            W02_after, _, _, _ = build_all_fast(lam_after, N, n_quad=3000)

            ew, ev = np.linalg.eigh(W02_after)
            thresh_w = np.max(np.abs(ew)) * 1e-10
            P_null = ev[:, np.abs(ew) <= thresh_w]

            if P_null.shape[1] == 0:
                continue

            M_null_before = P_null.T @ M_before @ P_null
            M_null_after = P_null.T @ M_after @ P_null

            max_before = np.max(np.linalg.eigvalsh(M_null_before))
            max_after = np.max(np.linalg.eigvalsh(M_null_after))
            delta = max_after - max_before

            n_total += 1
            if delta < 1e-10:
                n_neg += 1
                sign = "NEG"
            else:
                n_pos += 1
                sign = "POS"

            # Find which prime power this is
            pk_int = round(pp_val - 0.5)
            print(f"  {pk_int:>6} {max_before:>+14.6e} {max_after:>+14.6e} {delta:>+12.4e} {sign:>6}")

        except Exception as e:
            print(f"  {round(pp_val-0.5):>6} ERROR: {e}")

    print(f"\n  MONOTONICITY: {n_neg}/{n_total} thresholds have negative delta")
    print(f"  Non-monotone: {n_pos}/{n_total}")

    if n_pos == 0:
        print(f"\n  *** PERFECT MONOTONICITY: EVERY PRIME POWER PUSHES M FURTHER NEGATIVE ***")
        print(f"  *** ON null(W02) ***")
    else:
        print(f"\n  Monotonicity FAILS at {n_pos} thresholds")


def continuous_flow(lam_sq_min=4, lam_sq_max=500, n_points=100):
    """
    Track max eigenvalue of M|null(W02) on a dense grid of lambda^2.

    This captures both the discrete prime power jumps AND the continuous
    evolution of the kernel shape with L.
    """
    # Dense grid
    lam_sq_vals = np.linspace(lam_sq_min, lam_sq_max, n_points)

    # Also include points just after each prime power
    pps = prime_power_thresholds(lam_sq_max)
    extra = [pp for pp in pps if lam_sq_min < pp < lam_sq_max]
    all_vals = sorted(set(list(lam_sq_vals) + extra))

    print(f"\nCONTINUOUS FLOW: {len(all_vals)} points from lam^2={lam_sq_min} to {lam_sq_max}")

    return sonin_flow(all_vals)


def base_case_verification(max_lam_sq=30):
    """
    Verify the base case: at small lambda^2 (few primes), Q_W >= 0 directly.

    For lambda^2 = 4: only primes 2, 3 and prime powers 4
    For lambda^2 = 10: primes 2,3,4,5,7,8,9
    etc.

    At each small lambda, verify Q_W >= 0 on null(W02) with full precision.
    """
    print(f"\nBASE CASE VERIFICATION: lam^2 from 4 to {max_lam_sq}")
    print(f"  {'lam^2':>6} {'primes':>30} {'max_eig(M|null)':>16} {'QW>=0?':>7}")

    for lam_sq in range(4, max_lam_sq + 1):
        L_f = np.log(lam_sq)
        N = max(5, min(round(4 * L_f), 20))

        W02, M, QW, n_pk = build_all_fast(lam_sq, N, n_quad=5000)

        ew, ev = np.linalg.eigh(W02)
        thresh = np.max(np.abs(ew)) * 1e-10
        P_null = ev[:, np.abs(ew) <= thresh]

        if P_null.shape[1] == 0:
            continue

        M_null = P_null.T @ M @ P_null
        evals = np.linalg.eigvalsh(M_null)
        max_ev = np.max(evals)
        ok = max_ev < 1e-8

        # List prime powers
        limit = lam_sq
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 2):
            if i <= limit and sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        pps = []
        for p in range(2, limit + 1):
            if sieve[p]:
                pk = p
                while pk <= lam_sq:
                    pps.append(pk)
                    pk *= p
        pps.sort()
        pp_str = ','.join(str(x) for x in pps[:10])
        if len(pps) > 10:
            pp_str += f"...({len(pps)})"

        flag = "" if ok else " ***FAIL***"
        print(f"  {lam_sq:>6} {pp_str:>30} {max_ev:>+16.6e} {'YES' if ok else 'NO':>7}{flag}")


if __name__ == "__main__":
    print("SESSION 36 -- SONIN MONOTONICITY FLOW")
    print("=" * 80)

    # 1. Base case: small lambda verification
    base_case_verification(30)

    # 2. Derivative at prime power thresholds (the key test)
    derivative_at_thresholds(200)

    # 3. Continuous flow
    results = continuous_flow(lam_sq_min=4, lam_sq_max=300, n_points=50)

    with open('session36_sonin_flow.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to session36_sonin_flow.json")
