"""
SESSION 81 -- FACTORING IN ZERO-COORDINATE SPACE

New direction: represent integers by their zeta-zero signatures
and explore whether factoring becomes structurally different.

Every integer n has a zero-signature:
  z_k(n) = cos(gamma_k * log(n))  for k = 1, 2, ...

Key properties:
  - Multiplication is phase addition: z_k(ab) = cos(gamma_k*(log a + log b))
  - Primes have unique signatures (precomputable)
  - The explicit formula says these signatures encode ALL multiplicative structure

QUESTION: Does working in zero-space reveal structure that makes
factoring easier?

PROBES:
  1. Build the zero-signature table for small primes and composites
  2. Visualize: do composites cluster near their factor pairs?
  3. Given n=pq, how many zeros K are needed to uniquely identify (p,q)?
  4. Build a zero-space factoring algorithm and test it
  5. Compare resolution: K zeros vs trial division
  6. The "inner product" approach: <sig(n), sig(p)> as a factoring test
"""

import sys
import numpy as np
import mpmath
import time

mpmath.mp.dps = 30


def load_zeros(K):
    """Load first K zeta zero ordinates."""
    return [float(mpmath.zetazero(k).imag) for k in range(1, K + 1)]


def zero_signature(n, zeros):
    """Compute the zero-coordinate signature of integer n."""
    log_n = np.log(float(n))
    return np.array([np.cos(g * log_n) for g in zeros])


def sig_distance(sig1, sig2):
    """Euclidean distance between two signatures."""
    return np.linalg.norm(sig1 - sig2)


def sig_inner(sig1, sig2):
    """Normalized inner product between two signatures."""
    n1 = np.linalg.norm(sig1)
    n2 = np.linalg.norm(sig2)
    if n1 < 1e-15 or n2 < 1e-15:
        return 0
    return float(sig1 @ sig2) / (n1 * n2)


def run():
    print()
    print('#' * 76)
    print('  SESSION 81 -- FACTORING IN ZERO-COORDINATE SPACE')
    print('#' * 76)

    # ======================================================================
    # PROBE 1: Zero-signature table
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 1: ZERO-SIGNATURE TABLE')
    print(f'{"="*76}\n')

    K = 20  # number of zeros to use
    zeros = load_zeros(K)
    print(f'  Using first {K} zeta zeros')
    print(f'  gamma_1 = {zeros[0]:.6f}, gamma_{K} = {zeros[-1]:.6f}')
    print()

    # Signatures of small primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    print(f'  Prime signatures (first 5 components):')
    print(f'  {"p":>4} {"z_1":>10} {"z_2":>10} {"z_3":>10} {"z_4":>10} {"z_5":>10}')
    print('  ' + '-' * 56)

    prime_sigs = {}
    for p in primes:
        sig = zero_signature(p, zeros)
        prime_sigs[p] = sig
        print(f'  {p:>4d} {sig[0]:>+10.6f} {sig[1]:>+10.6f} {sig[2]:>+10.6f} '
              f'{sig[3]:>+10.6f} {sig[4]:>+10.6f}')

    # Signatures of small composites
    print(f'\n  Composite signatures:')
    print(f'  {"n":>6} {"factors":>12} {"z_1":>10} {"z_2":>10} {"z_3":>10}')
    print('  ' + '-' * 50)

    composites = [(6, 2, 3), (10, 2, 5), (15, 3, 5), (21, 3, 7),
                  (35, 5, 7), (77, 7, 11), (143, 11, 13), (221, 13, 17)]
    for n, p, q in composites:
        sig = zero_signature(n, zeros)
        print(f'  {n:>6d} {f"{p}*{q}":>12} {sig[0]:>+10.6f} {sig[1]:>+10.6f} '
              f'{sig[2]:>+10.6f}')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 2: Multiplication = phase addition (verify)
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 2: MULTIPLICATION IS PHASE ADDITION')
    print(f'{"="*76}\n')

    # z_k(pq) should equal cos(gamma_k * (log p + log q))
    # = cos(gamma_k * log p) * cos(gamma_k * log q)
    #   - sin(gamma_k * log p) * sin(gamma_k * log q)
    # So: z_k(pq) = z_k(p)*z_k(q) - s_k(p)*s_k(q)
    # where s_k(n) = sin(gamma_k * log n)

    print(f'  Verify: sig(p*q) = cos-product of sig(p) and sig(q)')
    print(f'  {"p*q":>8} {"direct":>12} {"product":>12} {"error":>12}')
    print('  ' + '-' * 48)

    for p, q in [(2, 3), (5, 7), (11, 13), (23, 29)]:
        n = p * q
        sig_n = zero_signature(n, zeros)
        # Phase addition: cos(a+b) = cos(a)cos(b) - sin(a)sin(b)
        log_p, log_q = np.log(float(p)), np.log(float(q))
        sig_product = np.array([
            np.cos(g * log_p) * np.cos(g * log_q) -
            np.sin(g * log_p) * np.sin(g * log_q)
            for g in zeros
        ])
        err = np.max(np.abs(sig_n - sig_product))
        print(f'  {p}*{q}={n:>4d} {sig_n[0]:>+12.8f} {sig_product[0]:>+12.8f} '
              f'{err:>12.2e}')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 3: Inner product factoring test
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 3: INNER PRODUCT FACTORING')
    print(f'{"="*76}\n')

    # Hypothesis: <sig(n), sig(p)> is large when p divides n.
    # If sig(n) = sig(p) "phase-shifted" by sig(q), then the
    # inner product picks up the correlation.

    # Actually, a better test: compute the "deconvolution"
    # sig_quotient_k(n, p) = cos(gamma_k * (log n - log p))
    # = cos(gamma_k * log(n/p))
    # If p divides n, then n/p is an integer, and sig_quotient is
    # the signature of n/p. If not, n/p is not an integer and the
    # signature won't match any integer.

    print(f'  Deconvolution test: sig(n/p) for n=221=13*17')
    print(f'  {"test p":>8} {"n/p":>8} {"integer?":>10} {"sig match":>12}')
    print('  ' + '-' * 42)

    n = 221
    sig_n = zero_signature(n, zeros)
    log_n = np.log(float(n))

    for p_test in [2, 3, 5, 7, 11, 13, 17, 19]:
        log_p = np.log(float(p_test))
        # Deconvolved signature: cos(gamma_k * (log_n - log_p))
        sig_deconv = np.array([np.cos(g * (log_n - log_p)) for g in zeros])
        quotient = n / p_test
        is_int = abs(quotient - round(quotient)) < 1e-10

        # Check if deconvolved signature matches any prime
        best_match = 0
        best_prime = 0
        for p2 in primes:
            match = sig_inner(sig_deconv, prime_sigs[p2])
            if match > best_match:
                best_match = match
                best_prime = p2

        print(f'  {p_test:>8d} {quotient:>8.1f} {"YES" if is_int else "no":>10} '
              f'{best_match:>10.6f} (p={best_prime})')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 4: Resolution — how many zeros to factor?
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 4: RESOLUTION vs NUMBER OF ZEROS')
    print(f'{"="*76}\n')

    # For n = p*q, try to factor by deconvolving with each candidate prime
    # and checking if the result matches an integer signature.
    # How many zeros K do we need?

    test_cases = [
        (15, 3, 5),
        (221, 13, 17),
        (1073, 29, 37),
        (10403, 101, 103),
        (25319, 149, 170),  # not a semiprime — control
    ]

    for K_test in [3, 5, 10, 20, 50]:
        zeros_k = zeros[:K_test] if K_test <= K else load_zeros(K_test)
        print(f'  K = {K_test} zeros:')

        for n, p_true, q_true in test_cases:
            # Try all primes up to sqrt(n)
            best_score = -1
            best_p = 0
            for p_cand in range(2, int(np.sqrt(n)) + 1):
                log_p = np.log(float(p_cand))
                log_n = np.log(float(n))
                sig_deconv = np.array([np.cos(g * (log_n - log_p))
                                       for g in zeros_k])
                # Score: how close is deconv to an INTEGER signature?
                # An integer q = n/p_cand would have sig(q) = sig_deconv.
                # Check if deconv matches sig(round(n/p_cand))
                q_cand = n / p_cand
                q_round = round(q_cand)
                if q_round < 2 or q_round > n:
                    continue
                sig_q = np.array([np.cos(g * np.log(float(q_round)))
                                  for g in zeros_k])
                score = sig_inner(sig_deconv, sig_q)
                if score > best_score:
                    best_score = score
                    best_p = p_cand

            q_found = n // best_p if best_p > 0 and n % best_p == 0 else 0
            correct = (best_p == p_true or best_p == q_true) if q_found > 0 else False
            print(f'    n={n:>6d}: best p={best_p:>4d}, score={best_score:.6f}, '
                  f'{"CORRECT" if correct else "wrong"}')
        print()
    sys.stdout.flush()

    # ======================================================================
    # PROBE 5: The spectral sieve — can zeros filter candidates?
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 5: SPECTRAL SIEVE — ZEROS AS A FILTER')
    print(f'{"="*76}\n')

    # Instead of factoring directly, use zeros as a SIEVE:
    # For each zero gamma_k, cos(gamma_k * log n) constrains the
    # possible factor pairs. Each zero eliminates some candidates.
    # How fast does the candidate set shrink?

    K_sieve = 50
    zeros_sieve = load_zeros(K_sieve)

    n = 10403  # = 101 * 103
    log_n = np.log(float(n))
    sig_n = np.array([np.cos(g * log_n) for g in zeros_sieve])

    # For each candidate p, compute the "match score" using K zeros
    candidates = list(range(2, int(np.sqrt(n)) + 1))
    print(f'  n = {n} = 101 * 103')
    print(f'  {len(candidates)} initial candidates (2 to {candidates[-1]})')
    print()

    print(f'  {"K zeros":>8} {"candidates left":>16} {"includes 101?":>14} '
          f'{"top candidate":>14}')
    print('  ' + '-' * 56)

    for K_use in [1, 2, 3, 5, 10, 20, 50]:
        zeros_use = zeros_sieve[:K_use]
        scores = []
        for p_cand in candidates:
            log_p = np.log(float(p_cand))
            # Score: match of deconvolved signature to nearest integer
            sig_deconv = np.array([np.cos(g * (log_n - log_p))
                                   for g in zeros_use])
            q_cand = round(n / p_cand)
            if q_cand < 2:
                scores.append(-1)
                continue
            sig_q = np.array([np.cos(g * np.log(float(q_cand)))
                              for g in zeros_use])
            scores.append(sig_inner(sig_deconv, sig_q))

        scores = np.array(scores)
        # Count candidates with score > 0.99
        threshold = 0.99
        n_surviving = np.sum(scores > threshold)
        includes_101 = scores[candidates.index(101)] > threshold if 101 in candidates else False
        top_idx = np.argmax(scores)
        top_p = candidates[top_idx]

        print(f'  {K_use:>8d} {n_surviving:>16d} '
              f'{"YES" if includes_101 else "no":>14} '
              f'{top_p:>14d}')
    sys.stdout.flush()

    # ======================================================================
    # PROBE 6: Timing comparison — zero-sieve vs trial division
    # ======================================================================
    print(f'\n{"="*76}')
    print(f'  PROBE 6: TIMING — ZERO-SIEVE vs TRIAL DIVISION')
    print(f'{"="*76}\n')

    # Factor several semiprimes both ways
    semiprimes = [
        (143, 11, 13),
        (10403, 101, 103),
        (120377, 337, 357),  # not semiprime — 337 is prime, 357=3*7*17
        (1018081, 1009, 1009),  # perfect square of prime
    ]

    # Recompute: actually use proper semiprimes
    semiprimes = [
        (143, 11, 13),
        (10403, 101, 103),
        (96727, 293, 330),  # check: 293*330 != 96727. Use real ones.
    ]
    # Generate proper semiprimes
    from session41g_uncapped_barrier import sieve_primes
    small_primes = list(sieve_primes(500))
    semiprimes = []
    for i in range(0, len(small_primes), 20):
        p = small_primes[i]
        q = small_primes[min(i + 1, len(small_primes) - 1)]
        semiprimes.append((p * q, p, q))
    semiprimes = semiprimes[:6]

    K_factor = 20
    zeros_factor = zeros[:K_factor]

    print(f'  {"n":>10} {"p":>6} {"q":>6} {"trial div (us)":>14} '
          f'{"zero-sieve (us)":>16} {"ratio":>8}')
    print('  ' + '-' * 64)

    for n, p_true, q_true in semiprimes:
        # Trial division
        t0 = time.perf_counter()
        for _ in range(100):
            for p_try in range(2, int(np.sqrt(n)) + 1):
                if n % p_try == 0:
                    break
        td_time = (time.perf_counter() - t0) / 100 * 1e6

        # Zero-sieve
        t0 = time.perf_counter()
        log_n = np.log(float(n))
        for _ in range(100):
            best_score = -1
            best_p = 0
            for p_cand in range(2, int(np.sqrt(n)) + 1):
                sig_d = np.array([np.cos(g * (log_n - np.log(float(p_cand))))
                                  for g in zeros_factor])
                q_r = round(n / p_cand)
                if q_r < 2:
                    continue
                sig_q = np.array([np.cos(g * np.log(float(q_r)))
                                  for g in zeros_factor])
                sc = np.dot(sig_d, sig_q) / (np.linalg.norm(sig_d) *
                                              np.linalg.norm(sig_q) + 1e-30)
                if sc > best_score:
                    best_score = sc
                    best_p = p_cand
        zs_time = (time.perf_counter() - t0) / 100 * 1e6

        ratio = zs_time / td_time if td_time > 0 else 0
        print(f'  {n:>10d} {p_true:>6d} {q_true:>6d} {td_time:>14.1f} '
              f'{zs_time:>16.1f} {ratio:>8.1f}x')
    sys.stdout.flush()

    # ======================================================================
    # VERDICT
    # ======================================================================
    print()
    print('=' * 76)
    print('  SESSION 81 VERDICT')
    print('=' * 76)


if __name__ == '__main__':
    run()
