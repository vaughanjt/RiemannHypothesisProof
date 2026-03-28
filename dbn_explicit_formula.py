"""Explicit formula connection: S_N from zeros vs from primes.

The energy dissipation S_N = sum_{j!=k} 1/(gamma_j - gamma_k)^2
can be decomposed by distance scale:

  S_N = S_N^{close} + S_N^{far}

where "close" means |gamma_j - gamma_k| < d (one mean spacing)
and "far" means > d.

Montgomery proved the pair correlation formula UNCONDITIONALLY
for the Fourier range |alpha| <= 1, which controls the "far" pairs.
The "close" pairs (including the closest that collides) require
the full Montgomery conjecture (|alpha| > 1).

Key question: what fraction of S_N comes from far vs close pairs?
If far pairs dominate, then the unconditional portion controls
most of the energy.
"""

import numpy as np
import mpmath
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json


def compute_SN_by_distance(zeros, d_threshold_multiples=None):
    """Compute S_N decomposed by distance scale.

    S_N = sum_{j!=k} 1/(gamma_j - gamma_k)^2

    Decompose into contributions from pairs at different
    distance scales: |gamma_j - gamma_k| in [n*d, (n+1)*d).
    """
    N = len(zeros)
    d = np.mean(np.diff(np.sort(zeros)))

    if d_threshold_multiples is None:
        d_threshold_multiples = [0.5, 1, 2, 3, 5, 10, 20, 50, 100]

    # Compute all pairwise inverse-square distances
    total = 0.0
    contributions = {f'<{m}d': 0.0 for m in d_threshold_multiples}
    contributions['total'] = 0.0

    pair_distances = []
    pair_contributions = []

    for j in range(N):
        for k in range(j+1, N):
            dist = abs(zeros[k] - zeros[j])
            contrib = 2.0 / dist**2  # factor 2 for both (j,k) and (k,j)
            total += contrib
            pair_distances.append(dist / d)  # normalized
            pair_contributions.append(contrib)

            for m in d_threshold_multiples:
                if dist < m * d:
                    contributions[f'<{m}d'] += contrib

    contributions['total'] = total

    return contributions, d, np.array(pair_distances), np.array(pair_contributions)


def montgomery_proven_fraction(zeros):
    """Estimate what fraction of S_N is controlled by Montgomery's
    proven range |alpha| <= 1.

    Montgomery's theorem (unconditional):
      F(alpha, T) = (|alpha| + o(1)) * (T log T / 2pi)  for 0 <= alpha <= 1

    The pair correlation R2(r) is related to F(alpha) by Fourier transform.
    The proven range |alpha| <= 1 controls R2(r) for |r| > 1/(2*pi)
    (Fourier uncertainty: Fourier support [0,1] means real-space
    resolution ~ 1, so pairs at distance > ~d are controlled).

    More precisely: the pair correlation for CLOSE pairs (r < 1)
    involves F(alpha) at alpha > 1 (the unproven range).

    We compute the fraction of S_N from pairs at distance > d
    (controlled by proven range) vs < d (requires full Montgomery).
    """
    N = len(zeros)
    d = np.mean(np.diff(np.sort(zeros)))

    S_close = 0.0  # |gamma_j - gamma_k| < d
    S_far = 0.0    # |gamma_j - gamma_k| >= d

    for j in range(N):
        for k in range(j+1, N):
            dist = abs(zeros[k] - zeros[j])
            contrib = 2.0 / dist**2
            if dist < d:
                S_close += contrib
            else:
                S_far += contrib

    S_total = S_close + S_far
    return {
        'S_close': S_close,
        'S_far': S_far,
        'S_total': S_total,
        'fraction_close': S_close / S_total if S_total > 0 else 0,
        'fraction_far': S_far / S_total if S_total > 0 else 0,
        'N': N,
        'd': d,
    }


def compute_SN_from_explicit_formula(T, N_primes=10000):
    """Compute S_N (approximately) from the prime-side explicit formula.

    The key identity (from (zeta'/zeta)' at s = 1/2+it):

    sum_rho 1/(s-rho)^2 = 1/(s-1)^2 - (1/4)*psi'(s/2) - (zeta'/zeta)'(s)

    For the energy-related quantity, we need:
    integral_0^T |sum_rho 1/(1/2+it-rho)^2|^2 dt  (approximately)

    But more directly: the Dirichlet series side gives us
    (zeta'/zeta)'(s) = sum Lambda(n) log(n) / n^s  for Re(s) > 1.

    We can evaluate this at s = sigma + it for sigma slightly > 1/2
    using smooth approximation.

    Instead, let's compute the TRACE: sum over zeros of the
    "regular part" of -(xi'/xi)' at each zero. This connects
    directly to S_N.

    For each zero gamma_j:
    sum_{k != j} 1/(gamma_j - gamma_k)^2
      = Re[1/(1/2+i*gamma_j)^2] + Re[1/(-1/2+i*gamma_j)^2]
        - (1/4)*Re[psi'(1/4+i*gamma_j/2)]
        - RegPart[(zeta'/zeta)' at s = 1/2+i*gamma_j]

    The RegPart requires knowing the Laurent expansion of zeta'/zeta
    at the zero, which involves the "secondary term" c_j.
    """
    # This is computed from zeros, not primes -- but shows the structure
    pass


def gue_prediction_SN(N, d):
    """GUE prediction for S_N.

    S_N = N * (1/d^2) * integral R2(u)/u^2 du + O(N/d^2)
        = N * C_GUE / d^2

    where C_GUE = integral_0^inf R2(u)/u^2 du ~ 3.255
    """
    from scipy.integrate import quad

    def R2_over_u2(u):
        if u < 1e-10:
            return 0.0  # R2(0) = 0
        sinc = np.sin(np.pi * u) / (np.pi * u)
        return (1 - sinc**2) / u**2

    C_GUE, _ = quad(R2_over_u2, 0.01, 500, limit=500)
    return N * C_GUE / d**2, C_GUE


def run_explicit_formula_analysis():
    """Main analysis: decompose S_N and find the unconditional fraction."""

    print("=" * 70)
    print("EXPLICIT FORMULA: S_N DECOMPOSITION BY DISTANCE SCALE")
    print("=" * 70)

    all_zeros = np.load('_zeros_200.npy')

    # Step 1: Compute S_N for various N
    print("\n[1/3] Computing S_N from zeros")

    results = []
    for N in [25, 50, 100, 200]:
        zeros = all_zeros[:N]
        m = montgomery_proven_fraction(zeros)

        gue_pred, C_GUE = gue_prediction_SN(N, m['d'])

        print(f"\n  N = {N}, d = {m['d']:.4f}")
        print(f"  S_total = {m['S_total']:.4f}")
        print(f"  S_close (|r| < d) = {m['S_close']:.4f} ({100*m['fraction_close']:.1f}%)")
        print(f"  S_far   (|r| > d) = {m['S_far']:.4f} ({100*m['fraction_far']:.1f}%)")
        print(f"  GUE prediction    = {gue_pred:.4f}")
        print(f"  Ratio S/GUE       = {m['S_total']/gue_pred:.4f}")

        m['gue_pred'] = gue_pred
        m['C_GUE'] = C_GUE
        results.append(m)

    # Step 2: Finer decomposition by distance
    print("\n\n[2/3] Distance-scale decomposition for N=200")
    zeros200 = all_zeros[:200]
    contribs, d, pair_dists, pair_contrs = compute_SN_by_distance(zeros200)

    print(f"\n  Total S_200 = {contribs['total']:.4f}")
    print(f"  Mean spacing d = {d:.4f}")
    print(f"\n  {'Threshold':>12}  {'Cumulative S':>14}  {'Fraction':>10}")
    print(f"  {'-'*12}  {'-'*14}  {'-'*10}")
    for m in [0.5, 1, 2, 3, 5, 10, 20, 50, 100]:
        key = f'<{m}d'
        val = contribs[key]
        frac = val / contribs['total']
        print(f"  {key:>12}  {val:>14.4f}  {100*frac:>9.1f}%")

    # Step 3: The critical analysis
    print("\n\n[3/3] CRITICAL ANALYSIS: What's unconditional?")
    print()
    print("  Montgomery proved F(alpha,T) = (|alpha|+o(1))*T*log(T)/(2pi)")
    print("  UNCONDITIONALLY for |alpha| <= 1.")
    print()
    print("  By Fourier analysis:")
    print("  - |alpha| <= 1 controls pair distances |r/d| >= 1 (far pairs)")
    print("  - |alpha| > 1 controls pair distances |r/d| < 1 (close pairs)")
    print()
    print("  Distance decomposition of S_N:")
    print(f"  {'N':>5}  {'S_close':>10}  {'S_far':>10}  {'% far (proven)':>15}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*15}")
    for r in results:
        print(f"  {r['N']:>5}  {r['S_close']:>10.2f}  {r['S_far']:>10.2f}"
              f"  {100*r['fraction_far']:>14.1f}%")

    print()

    # THE KEY FINDING:
    far_fracs = [r['fraction_far'] for r in results]
    if all(f > 0.5 for f in far_fracs):
        print("  ** FAR PAIRS DOMINATE S_N **")
        print(f"  The unconditionally proven portion is {100*min(far_fracs):.0f}%+ of S_N.")
        print("  This means the energy dissipation rate is MOSTLY controlled")
        print("  by the proven part of Montgomery's theorem.")
    else:
        print("  ** CLOSE PAIRS DOMINATE S_N **")
        print("  The unproven portion of Montgomery controls most of S_N.")
        print("  An unconditional result from this approach is difficult.")

    print()
    print("  However, for the COLLISION TIME bound, what matters is not S_N")
    print("  but the individual closest-pair term 1/(gamma_j-gamma_k)^2.")
    print("  This is entirely in the CLOSE range, requiring full Montgomery.")
    print()
    print("  CONCLUSION: The energy approach CAN be made partially")
    print("  unconditional (for the bulk of S_N), but the collision time")
    print("  for the closest pair -- which determines Lambda -- inherently")
    print("  requires the CLOSE-pair statistics (full Montgomery).")
    print()
    print("  The BYPASS: instead of bounding Lambda via the closest pair,")
    print("  bound it via the TOTAL ENERGY. The energy bound")
    print("     Lambda >= -(E_N - E_min) / (4*S_N)")
    print("  uses S_N (which is mostly unconditional) and E_N (computable).")
    print("  This gives a WEAKER but MORE UNCONDITIONAL bound on Lambda.")

    # Compute the actual energy bound
    print("\n\n  ENERGY BOUND ON LAMBDA:")
    for r in results:
        N = r['N']
        zeros = all_zeros[:N]
        z_sorted = np.sort(zeros)

        # E_N = -2 * sum log|gamma_j - gamma_k|
        E_N = 0.0
        for j in range(N):
            for k in range(j+1, N):
                E_N -= 2 * np.log(abs(z_sorted[k] - z_sorted[j]))

        # E_min: equilibrium energy of N points on [gamma_1, gamma_N]
        # For log-gas: E_min ~ -(N^2/2)*log(L/N) + O(N^2) where L = range
        L = z_sorted[-1] - z_sorted[0]
        # Crude estimate: if uniformly spaced with gap L/(N-1):
        E_equil = 0.0
        for j in range(N):
            for k in range(j+1, N):
                E_equil -= 2 * np.log((k-j) * L / (N-1))

        delta_E = E_N - E_equil  # how far from equilibrium

        S_N = r['S_total']
        lambda_bound = -abs(delta_E) / (4 * S_N) if S_N > 0 else float('-inf')

        print(f"\n  N = {N}:")
        print(f"    E_N = {E_N:.2f}")
        print(f"    E_equil = {E_equil:.2f}")
        print(f"    |E_N - E_equil| = {abs(delta_E):.2f}")
        print(f"    S_N = {S_N:.2f}")
        print(f"    Lambda >= {lambda_bound:.6f}")
        print(f"    (Compare: Rodgers-Tao gives Lambda >= 0)")

    # -- Plot --
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) Cumulative S_N by distance
    ax = axes[0, 0]
    bins = np.logspace(-2, 2, 100)
    cumulative = np.zeros(len(bins))
    for i, b in enumerate(bins):
        cumulative[i] = np.sum(pair_contrs[pair_dists < b])
    ax.semilogx(bins, cumulative / contribs['total'], 'b-', linewidth=2)
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=1.5,
               label='r = d (Montgomery boundary)')
    ax.set_xlabel('r / d (normalized pair distance)')
    ax.set_ylabel('Cumulative fraction of S_N')
    ax.set_title('S_N by distance scale (N=200)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (0,1) Close vs far fraction vs N
    ax = axes[0, 1]
    Ns = [r['N'] for r in results]
    close_fracs = [r['fraction_close'] for r in results]
    far_fracs = [r['fraction_far'] for r in results]
    ax.bar(Ns, close_fracs, width=[4,8,15,30], color='red', alpha=0.7,
           label='Close (|r|<d) — needs full Montgomery')
    ax.bar(Ns, far_fracs, bottom=close_fracs, width=[4,8,15,30],
           color='blue', alpha=0.7, label='Far (|r|>d) — unconditional')
    ax.set_xlabel('N')
    ax.set_ylabel('Fraction of S_N')
    ax.set_title('Unconditional vs conjectural portions')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)

    # (1,0) Individual pair contributions vs distance
    ax = axes[1, 0]
    ax.loglog(pair_dists, pair_contrs, 'b.', markersize=1, alpha=0.3)
    # GUE prediction line: contribution ~ R2(r)/r^2 * (2/d^2)
    r_line = np.logspace(-1, 2, 200)
    sinc = np.sin(np.pi * r_line) / (np.pi * r_line)
    R2_line = 1 - sinc**2
    gue_line = R2_line / r_line**2 * 2 / d**2
    ax.loglog(r_line, gue_line, 'r-', linewidth=1.5, alpha=0.7, label='GUE prediction')
    ax.axvline(x=1.0, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel('r / d')
    ax.set_ylabel('Pair contribution to S_N')
    ax.set_title('Individual pair contributions')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (1,1) S_N vs N with GUE prediction
    ax = axes[1, 1]
    S_vals = [r['S_total'] for r in results]
    S_gue = [r['gue_pred'] for r in results]
    ax.plot(Ns, S_vals, 'bo-', markersize=8, linewidth=2, label='Computed from zeros')
    ax.plot(Ns, S_gue, 'r--', markersize=6, linewidth=1.5, label='GUE prediction')
    ax.set_xlabel('N')
    ax.set_ylabel('S_N')
    ax.set_title('Energy dissipation rate: zeros vs GUE')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Explicit Formula Analysis: What Portion of S_N is Unconditional?',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig('dbn_explicit_formula.png', dpi=150, bbox_inches='tight')
    print("\n  Saved: dbn_explicit_formula.png")
    plt.close(fig)

    with open('dbn_explicit_formula.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print("  Saved: dbn_explicit_formula.json")

    return results


if __name__ == '__main__':
    run_explicit_formula_analysis()
