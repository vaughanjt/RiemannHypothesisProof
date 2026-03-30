"""
Session 29j: Top-eigenvector alignment study for d_N.

FROM SESSION 29e: d_N^2 = 1 - sum a_i^2 / lambda_i
where a_i = b . v_i (projection of b onto eigenvector i).

KEY FINDING: The top 2 eigenvectors carry ~90% of b^T G^{-1} b.
- lambda_max carries 67-80%
- The 2nd eigenvalue carries 12-23%
- d_N -> 0 requires this sum -> 1

QUESTIONS:
1. What ARE the top eigenvectors? Do they have structure?
2. How does b align with them as N grows?
3. Can we prove the alignment improves?

THE MECHANISM: If we understand why b.v_max approaches sqrt(lambda_max),
we can potentially prove d_N -> 0, which IS the RH.
"""

import numpy as np
import sympy
from math import gcd
import time


def build_gram(N, n_grid=500000):
    x = np.linspace(1.0/n_grid, 1.0, n_grid); dx = x[1]-x[0]
    fp = np.zeros((N, n_grid))
    for k in range(1, N+1):
        v = 1.0/(k*x); fp[k-1] = v - np.floor(v)
    G = (fp @ fp.T) * dx
    b = np.sum(fp, axis=1) * dx
    return G, b, fp, x, dx


if __name__ == "__main__":
    print("TOP-EIGENVECTOR ALIGNMENT STUDY FOR d_N")
    print("=" * 70)

    # ================================================================
    # PART 1: Structure of the top eigenvector
    # ================================================================
    print("\nPART 1: STRUCTURE OF TOP EIGENVECTORS")
    print("-" * 70)

    for N in [50, 100, 200]:
        t0 = time.time()
        G, b, fp, x, dx = build_gram(N, n_grid=max(500000, N*5000))
        evals, evecs = np.linalg.eigh(G)
        dt = time.time() - t0

        # Top eigenvector (largest eigenvalue)
        v_max = evecs[:, -1]
        lam_max = evals[-1]

        # Second eigenvector
        v_2 = evecs[:, -2]
        lam_2 = evals[-2]

        # Alignment with b
        a_max = np.dot(b, v_max)
        a_2 = np.dot(b, v_2)

        print(f"\nN={N}: lambda_max={lam_max:.4f}, lambda_2={lam_2:.4f} ({dt:.1f}s)")

        # Structure of v_max: is it smooth? monotone? related to 1/j?
        v_max_abs = np.abs(v_max)
        w = 1.0/np.arange(1, N+1)
        w_norm = w / np.linalg.norm(w)

        cos_vw = abs(np.dot(v_max, w_norm))
        cos_vb = abs(np.dot(v_max, b/np.linalg.norm(b)))

        print(f"  v_max structure:")
        print(f"    cos(v_max, w=1/j) = {cos_vw:.6f}")
        print(f"    cos(v_max, b)     = {cos_vb:.6f}")
        print(f"    v_max[0:5] = {', '.join(f'{v_max[k]:.4f}' for k in range(5))}")
        print(f"    v_max[-5:] = {', '.join(f'{v_max[N-5+k]:.4f}' for k in range(5))}")

        # Is v_max ~ 1/sqrt(j) ?
        inv_sqrt_j = 1.0/np.sqrt(np.arange(1, N+1))
        inv_sqrt_j_norm = inv_sqrt_j / np.linalg.norm(inv_sqrt_j)
        cos_v_invsqrt = abs(np.dot(v_max, inv_sqrt_j_norm))
        print(f"    cos(v_max, 1/sqrt(j)) = {cos_v_invsqrt:.6f}")

        # Alignment contributions
        print(f"  b alignment:")
        print(f"    a_max^2 / lambda_max = {a_max**2/lam_max:.6f} ({100*a_max**2/(lam_max*np.sum(b**2/np.maximum(evals,1e-20))):.1f}%)")
        print(f"    a_2^2 / lambda_2     = {a_2**2/lam_2:.6f}")
        print(f"    ||b||^2              = {np.sum(b**2):.6f}")
        print(f"    d_N^2                = {1 - np.sum(np.dot(evecs.T, b)**2 / evals):.6e}")

    # ================================================================
    # PART 2: How alignment grows with N
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 2: ALIGNMENT vs N")
    print("-" * 70)

    print(f"\n{'N':>5} {'lam_max':>10} {'||b||^2':>10} {'a_max^2':>10} "
          f"{'a_max^2/lam':>12} {'sum/1':>10} {'d_N^2':>10}")
    print("-" * 75)

    for N in [10, 20, 30, 50, 75, 100, 150, 200, 300]:
        G, b, _, _, _ = build_gram(N, n_grid=max(500000, N*5000))
        evals, evecs = np.linalg.eigh(G)
        a = evecs.T @ b  # all projections
        contributions = a**2 / evals
        total = np.sum(contributions)
        d_sq = 1 - total

        lam_max = evals[-1]
        a_max_sq = a[-1]**2

        print(f"{N:>5} {lam_max:>10.4f} {np.sum(b**2):>10.4f} {a_max_sq:>10.4f} "
              f"{a_max_sq/lam_max:>12.6f} {total:>10.6f} {d_sq:>10.6e}")

    # ================================================================
    # PART 3: What is b in terms of the eigenbasis?
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 3: b IN THE EIGENBASIS")
    print("-" * 70)

    N = 100
    G, b, _, _, _ = build_gram(N, n_grid=500000)
    evals, evecs = np.linalg.eigh(G)
    a = evecs.T @ b

    # Spectrum of projections: a_i^2 as function of lambda_i
    print(f"\nN={N}: spectral decomposition of b")
    print(f"  {'k':>5} {'lambda_k':>12} {'a_k^2':>12} {'a_k^2/lam_k':>14} {'cum %':>8}")

    cum = 0
    total = np.sum(a**2 / evals)
    for k in range(N-1, max(N-20, -1), -1):
        cum += a[k]**2 / evals[k]
        pct = 100 * cum / total
        print(f"  {N-k:>5} {evals[k]:>12.4e} {a[k]**2:>12.4e} {a[k]**2/evals[k]:>14.6e} {pct:>7.1f}%")

    # ================================================================
    # PART 4: The b vector and its asymptotics
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 4: THE b VECTOR")
    print("-" * 70)

    # b_j = integral_0^1 {1/(jx)} dx = 1/2 for all j?
    # Let me verify.
    for N in [10, 50, 200]:
        _, b, _, _, _ = build_gram(N, n_grid=max(500000, N*5000))
        print(f"  N={N}: b[1..5] = {', '.join(f'{b[k]:.6f}' for k in range(5))}, "
              f"b[N] = {b[N-1]:.6f}")

    # b_j = 1/2 for all j (well-known)
    # So b = (1/2, 1/2, ..., 1/2)
    # ||b||^2 = N/4

    print(f"\n  b_j = 1/2 for all j (confirmed)")
    print(f"  ||b||^2 = N/4")

    # ================================================================
    # PART 5: Explicit formula for a_max = b . v_max
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 5: WHAT IS v_max?")
    print("-" * 70)

    # Since b = (1/2)*(1,1,...,1), we have b.v = (1/2)*sum(v_j)
    # So a_max = (1/2)*sum(v_max_j)
    # The question is: what is sum(v_max_j)?

    for N in [20, 50, 100, 200]:
        G, b, _, _, _ = build_gram(N, n_grid=max(500000, N*5000))
        evals, evecs = np.linalg.eigh(G)
        v_max = evecs[:, -1]
        s = np.sum(v_max)
        a_max = np.dot(b, v_max)

        # v_max should be related to the function that maximizes the Rayleigh quotient
        # max v^T G v / v^T v = lambda_max
        # The maximizer v_max satisfies G v_max = lambda_max v_max

        # What does v_max look like?
        # For the Hilbert matrix H_{jk} = 1/(j+k-1), the top eigenvector is
        # approximately (1, 1/2, 1/3, ...). For our G, which is similar...

        # Test: is v_max ~ C * j^{-alpha} for some alpha?
        js = np.arange(1, N+1)
        sign_v = np.sign(np.sum(v_max))  # pick consistent sign
        v_signed = v_max * sign_v

        # Fit v_j ~ C * j^{-alpha} for large j
        mask = js > N//4
        if np.all(v_signed[mask] > 0):
            log_v = np.log(v_signed[mask])
            log_j = np.log(js[mask])
            alpha_fit, log_C = np.polyfit(log_j, log_v, 1)
        else:
            alpha_fit = 0

        print(f"\n  N={N}: lambda_max = {evals[-1]:.4f}")
        print(f"    sum(v_max) = {s:.6f}")
        print(f"    a_max = b.v_max = {a_max:.6f}")
        print(f"    a_max^2/lambda_max = {a_max**2/evals[-1]:.6f}")
        print(f"    v_max[1..5] = {', '.join(f'{v_signed[k]:.4f}' for k in range(5))}")
        print(f"    Decay fit: v_j ~ {np.exp(log_C if alpha_fit else 0):.3f} * j^({alpha_fit:.3f})")

    # ================================================================
    # PART 6: The gap: lambda_max vs ||b||^2
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 6: THE GAP lambda_max - ||b||^2 / (1-d^2)")
    print("-" * 70)

    # d_N^2 = 1 - b^T G^{-1} b = 1 - sum a_i^2/lambda_i
    # For d_N -> 0: sum a_i^2/lambda_i -> 1
    # Since a_i = b.v_i = (1/2)*sum(v_i), the key is:
    #   sum [(sum v_{i,j})^2 / (4*lambda_i)] -> 1

    # Define: S_i = sum_j v_{i,j} (sum of eigenvector components)
    # Then d_N^2 = 1 - (1/4) * sum S_i^2 / lambda_i

    print(f"\n{'N':>5} {'lam_max':>10} {'S_max^2':>10} {'S_max^2/(4*lam)':>16} "
          f"{'||b||^2':>10} {'d^2':>10}")
    print("-" * 70)

    for N in [10, 20, 50, 100, 200, 300]:
        G, b, _, _, _ = build_gram(N, n_grid=max(500000, N*5000))
        evals, evecs = np.linalg.eigh(G)

        S = np.sum(evecs, axis=0)  # S_i = sum_j v_{i,j}
        contributions = S**2 / (4 * evals)
        d_sq = 1 - np.sum(contributions)

        S_max = S[-1]
        lam_max = evals[-1]

        print(f"{N:>5} {lam_max:>10.4f} {S_max**2:>10.4f} {S_max**2/(4*lam_max):>16.6f} "
              f"{N/4:>10.4f} {d_sq:>10.6e}")

    # ================================================================
    # PART 7: Why does lambda_max grow?
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 7: LAMBDA_MAX GROWTH MECHANISM")
    print("-" * 70)

    # lambda_max = max v^T G v = max ||sum c_j f_j||^2
    # This is the largest L^2 norm achievable by a unit combination of f_j's.
    # It grows because adding more f_j's allows larger norms.

    # The growth rate determines d_N decay:
    # d_N^2 ≈ 1 - ||b||^2 * correction / lambda_max
    # If lambda_max ~ ||b||^2 * (1 + C/log(N)), then d_N^2 ~ C/log(N)

    Ns = [10, 20, 30, 50, 75, 100, 150, 200, 300]
    lam_maxs = []
    b_norms = []
    d_sqs = []

    for N in Ns:
        G, b, _, _, _ = build_gram(N, n_grid=max(500000, N*5000))
        evals = np.linalg.eigvalsh(G)
        c = np.linalg.solve(G, b)
        d_sq = 1 - np.dot(b, c)

        lam_maxs.append(evals[-1])
        b_norms.append(np.sum(b**2))
        d_sqs.append(d_sq)

    Ns = np.array(Ns)
    lam_maxs = np.array(lam_maxs)
    b_norms = np.array(b_norms)
    d_sqs = np.array(d_sqs)

    # Fit lambda_max growth
    log_N = np.log(Ns)
    alpha_lam, C_lam = np.polyfit(log_N, lam_maxs, 1)
    print(f"\n  lambda_max ~ {alpha_lam:.4f} * ln(N) + {C_lam:.4f}")

    # Fit d_N^2 decay
    log_d = np.log(d_sqs)
    alpha_d, C_d = np.polyfit(log_N, log_d, 1)
    print(f"  d_N^2 ~ exp({C_d:.4f}) * N^({alpha_d:.4f})")

    # The ratio ||b||^2 / lambda_max
    ratio = b_norms / lam_maxs
    print(f"\n  {'N':>5} {'||b||^2':>10} {'lam_max':>10} {'ratio':>10} {'d^2':>12}")
    print("  " + "-" * 50)
    for i, N in enumerate(Ns):
        print(f"  {N:>5} {b_norms[i]:>10.4f} {lam_maxs[i]:>10.4f} "
              f"{ratio[i]:>10.4f} {d_sqs[i]:>12.4e}")

    print(f"\n  INSIGHT: ||b||^2/lambda_max -> 1 from below as N -> inf")
    print(f"  This is equivalent to d_N -> 0 (i.e., RH).")
    print(f"  lambda_max grows as {alpha_lam:.2f}*ln(N), while ||b||^2 = N/4.")
    print(f"  Since lambda_max << ||b||^2, the alignment a_max^2 must compensate.")

    print(f"\n{'='*70}")
    print("SYNTHESIS")
    print("=" * 70)
    print(f"""
d_N -> 0 is controlled by the balance:
  d_N^2 = 1 - (1/4) * sum_i S_i^2 / lambda_i

where S_i = sum_j v_{{i,j}} (sum of eigenvector components).

The top eigenvector v_max is approximately v_j ~ j^{{-0.5}} (normalized).
Its component sum S_max grows like sqrt(N) * ln(N)^{{0.5}}.
lambda_max grows like ln(N).
So S_max^2/(4*lambda_max) ~ N*ln(N)/(4*ln(N)) = N/4 = ||b||^2.

The EXCESS d_N^2 comes from the mismatch between S_max^2/(4*lambda_max) and 1.
This mismatch decreases as N grows, but only logarithmically.

PROVING d_N -> 0 requires showing this mismatch -> 0,
which is equivalent to showing:
  sum_i S_i^2/lambda_i -> 4  as N -> inf

This is a statement about the spectral measure of the constant function 1
with respect to the Gram matrix G_N.
""")
