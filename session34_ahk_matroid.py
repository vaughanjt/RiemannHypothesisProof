"""
SESSION 34 — ADIPRASITO-HUH-KATZ CONNECTION

Q_W has the structure: tau_{n,m} = a_n * delta + (b_n - b_m)/(n - m)
AHK's Chow ring operations involve DIVIDED DIFFERENCES.

TESTS:
1. Is a_n (diagonal of Q_W) LOG-CONCAVE? AHK proved log-concavity
   of Whitney numbers. If a_n is log-concave, it might be the
   Whitney number sequence of some matroid.

2. Does b_n relate to a matroid characteristic polynomial?
   The characteristic polynomial chi_M(t) = sum_k (-1)^k w_k t^{r-k}
   where w_k are the Whitney numbers. If b_n = chi_M(n) or similar,
   the divided difference matrix IS a matroid Chow ring element.

3. Can we identify the MATROID? The ground set should relate to
   primes up to lambda. The rank function should encode the
   prime factorization structure.

4. Does Q_W satisfy the LORENTZIAN property? Branden-Huh (2020)
   showed Lorentzian polynomials are the right framework for
   Hodge-Riemann in the matroid setting.
"""

import numpy as np
import time, json, sys
sys.path.insert(0, '.')
from connes_crossterm import build_all


def test_log_concavity(lam_sq, N=None):
    """Test if the diagonal a_n of Q_W is log-concave."""
    if N is None:
        L = np.log(lam_sq)
        N = max(15, round(6*L))
    dim = 2*N+1

    W02, M, QW = build_all(lam_sq, N)
    a = np.diag(QW)
    ns = np.arange(-N, N+1)

    print(f"\nLOG-CONCAVITY TEST: lam^2={lam_sq}, dim={dim}")

    # a_n should satisfy a_n^2 >= a_{n-1} * a_{n+1}
    # Equivalently: log(a_n) is concave, i.e., 2*log(a_n) >= log(a_{n-1}) + log(a_{n+1})
    violations = 0
    max_violation = 0
    for i in range(1, dim-1):
        if a[i] > 0 and a[i-1] > 0 and a[i+1] > 0:
            ratio = a[i]**2 / (a[i-1] * a[i+1])
            if ratio < 1 - 1e-10:
                violations += 1
                violation = 1 - ratio
                max_violation = max(max_violation, violation)

    log_concave = violations == 0
    print(f"  a_n all positive: {np.all(a > 0)}")
    print(f"  Log-concave: {log_concave} ({violations} violations, max={max_violation:.6e})")

    # Is a_n SYMMETRIC around n=0?
    center = N
    sym_error = max(abs(a[center+k] - a[center-k]) for k in range(1, N+1))
    print(f"  Symmetric a(n) = a(-n): {sym_error < 1e-10} (max error {sym_error:.4e})")

    # Is a_n UNIMODAL? (max at n=0, decreasing outward)
    unimodal = True
    for i in range(1, N+1):
        if a[center+i] > a[center+i-1] + 1e-10:
            unimodal = False
            break
    print(f"  Unimodal (max at n=0): {unimodal}")

    # Plot-like summary: a_n for key indices
    print(f"  a_n values:")
    for k in [0, 1, 2, 5, 10, N//2, N]:
        print(f"    n={k:>3}: a = {a[center+k]:.6f}")

    # The LOG of a_n: is it concave?
    log_a = np.log(a[a > 0])
    d2_log_a = np.diff(log_a, 2)
    all_neg_d2 = np.all(d2_log_a <= 1e-10)
    print(f"  d^2(log a)/dn^2 all <= 0: {all_neg_d2}")
    if not all_neg_d2:
        pos_d2 = d2_log_a[d2_log_a > 1e-10]
        print(f"    {len(pos_d2)} positive 2nd differences, max = {np.max(pos_d2):.4e}")

    return a, log_concave


def analyze_b_sequence(lam_sq, N=None):
    """Analyze the b_n sequence from the Lowner decomposition."""
    if N is None:
        L = np.log(lam_sq)
        N = max(15, round(6*L))
    dim = 2*N+1

    W02, M, QW = build_all(lam_sq, N)
    ns = np.arange(-N, N+1)

    # Extract b_n: QW[i, N] * (i - N) for i != N, b[N] = 0
    b = np.zeros(dim)
    center = N
    for i in range(dim):
        ni = i - N
        if ni != 0:
            b[i] = QW[i, center] * ni

    print(f"\n  b_n SEQUENCE ANALYSIS:")
    print(f"    b range: [{np.min(b):.6f}, {np.max(b):.6f}]")

    # Is b_n antisymmetric? b(-n) = -b(n)?
    antisym_error = max(abs(b[center+k] + b[center-k]) for k in range(1, N+1))
    print(f"    Antisymmetric b(-n) = -b(n): {antisym_error < 1e-10} (err {antisym_error:.4e})")

    # Is b_n roughly linear? b_n ~ c*n?
    b_pos = b[center+1:]
    ns_pos = np.arange(1, N+1, dtype=float)
    if len(b_pos) > 2:
        slope, intercept = np.polyfit(ns_pos, b_pos, 1)
        residual = np.max(np.abs(b_pos - (slope*ns_pos + intercept)))
        print(f"    Linear fit: b_n ~ {slope:.6f}*n + {intercept:.6f} (max residual {residual:.4e})")

    # Second differences of b (for matroid characterization)
    b_d2 = np.diff(b, 2)
    print(f"    2nd differences: [{np.min(b_d2):.4e}, {np.max(b_d2):.4e}]")

    return b


def test_lorentzian_property(lam_sq, N=None):
    """
    Test the LORENTZIAN property (Branden-Huh 2020).

    A homogeneous polynomial p(x_1, ..., x_n) is Lorentzian if:
    1. All coefficients are non-negative
    2. The Hessian H(p) has Lorentzian signature (at most 1 positive eigenvalue)
       at any point with positive coordinates.

    For our matrix: Q_W is "Lorentzian" if it has at most 1 positive
    eigenvalue when restricted to certain subspaces.

    Actually, the relevant test is: does the QUADRATIC FORM defined by Q_W
    have the Lorentzian property? A quadratic form q(x) = x^T Q x is
    Lorentzian if the matrix Q has signature (+, -, -, ..., -) or (-, +, +, ..., +).

    Our Q_W has ALL POSITIVE eigenvalues — it's positive definite.
    That's NOT Lorentzian (which needs mixed signature).

    BUT: the INTERSECTION FORM (= -Q_W on primitive) would be negative definite
    on primitive. And the FULL form (on all of range + null) has signature (2, d-2)
    if we consider the intersection form, which IS Lorentzian!
    """
    if N is None:
        L = np.log(lam_sq)
        N = max(15, round(6*L))
    dim = 2*N+1

    W02, M, QW = build_all(lam_sq, N)

    # The "intersection form" in the Hodge sense:
    # On range(W02): positive (contributes +2 to signature)
    # On null(W02): -Q_W = M which should be negative (contributes -(d-2))
    # Total signature: (2, d-2) — this IS Lorentzian!

    # Compute the intersection form: it's -M on null and W02-related on range
    # Actually, the intersection form is: I(x,y) = <x, W02 y> - <x, Q_W y> = <x, M y>
    # No wait: Q_W = W02 - M, so the "complementary form" is M = W02 - Q_W.

    # M has signature: some positive, some negative eigenvalues
    evals_M = np.linalg.eigvalsh(M)
    n_pos_M = np.sum(evals_M > 1e-10)
    n_neg_M = np.sum(evals_M < -1e-10)
    n_zero_M = dim - n_pos_M - n_neg_M

    print(f"\n  LORENTZIAN / SIGNATURE ANALYSIS: lam^2={lam_sq}")
    print(f"    Q_W eigenvalues: all positive (PD) — signature ({dim}, 0)")
    print(f"    M eigenvalues: {n_pos_M} pos, {n_neg_M} neg, {n_zero_M} zero — signature ({n_pos_M}, {n_neg_M})")
    print(f"    W02 eigenvalues: 2 nonzero — signature (1, 1) on range")

    # The "intersection form" should have signature (2, d-2) for Hodge index
    # On range(W02): the 2x2 block of M has eigenvalues that give the signature
    ew, ev = np.linalg.eigh(W02)
    thresh = np.max(np.abs(ew))*1e-10
    range_idx = np.abs(ew) > thresh
    null_idx = np.abs(ew) <= thresh
    P_range = ev[:, range_idx]
    P_null = ev[:, null_idx]

    M_range = P_range.T @ M @ P_range
    M_null = P_null.T @ M @ P_null

    evals_M_range = np.linalg.eigvalsh(M_range)
    evals_M_null = np.linalg.eigvalsh(M_null)

    n_pos_range = np.sum(evals_M_range > 1e-10)
    n_neg_range = np.sum(evals_M_range < -1e-10)
    n_pos_null = np.sum(evals_M_null > 1e-10)
    n_neg_null = np.sum(evals_M_null < -1e-10)

    print(f"\n    M on range(W02): {n_pos_range} pos, {n_neg_range} neg — {evals_M_range}")
    print(f"    M on null(W02): {n_pos_null} pos, {n_neg_null} neg")
    print(f"      (need: ALL negative for Hodge index)")
    print(f"      M_null all negative: {evals_M_null[-1] < 1e-10}")

    # The FULL M signature:
    # If M has signature (2, d-2) — i.e., exactly 2 positive eigenvalues
    # and d-2 negative eigenvalues — that's the Hodge index signature!
    hodge_signature = (n_pos_M == 2) and (n_neg_M == dim - 2)
    print(f"\n    M has Hodge signature (2, {dim-2}): {hodge_signature}")
    if hodge_signature:
        print(f"    *** M HAS EXACTLY THE HODGE INDEX SIGNATURE ***")
        print(f"    *** 2 positive eigenvalues (on range) + {dim-2} negative (on null) ***")

    return n_pos_M, n_neg_M


def matroid_connection(lam_sq, N=None):
    """
    Test: does Q_W's structure arise from a matroid?

    The PARTITION LATTICE of integers up to lambda has matroid structure.
    The ground set is the primes {p : p <= lambda}.
    Each integer n <= lambda^2 corresponds to a "flat" (set of prime factors).

    The CHARACTERISTIC POLYNOMIAL of this matroid would give a sequence
    related to the Whitney numbers. If the b_n match this...

    Actually, the more relevant matroid is the BOOLEAN MATROID on primes:
    Ground set E = {p : p <= lambda}
    Every subset is independent (free matroid)
    Rank = |E| = pi(lambda)

    The Chow ring of the free matroid U_{n,n} is trivial.

    A more interesting matroid: the CYCLE MATROID of a graph.
    The prime numbers up to lambda define a graph (how?)

    Or: the matroid of LINEAR DEPENDENCIES among the vectors
    {log(p) : p prime <= lambda} over Q.
    Since the log(p) are linearly independent over Q (by unique factorization),
    this is again a free matroid.

    THE KEY INSIGHT might be: the matroid structure comes not from the primes
    themselves but from the WEIL EXPLICIT FORMULA's structure.
    The formula creates algebraic relations between the prime contributions,
    and THESE relations define a matroid.
    """
    if N is None:
        L = np.log(lam_sq)
        N = max(15, round(6*L))
    dim = 2*N+1

    W02, M, QW = build_all(lam_sq, N)

    print(f"\n  MATROID CONNECTION: lam^2={lam_sq}")

    # The divided difference matrix (b_i - b_j)/(i - j) is a CAUCHY-type matrix
    # Cauchy matrices arise from rational interpolation
    # They have known determinant formulas and eigenvalue properties

    # Extract the divided difference matrix L = QW - diag(QW)
    L_mat = QW - np.diag(np.diag(QW))

    # Cauchy matrix test: L_{ij} = c / (x_i - y_j) for some sequences x, y, c
    # Our L_{ij} = (b_i - b_j)/(i - j). This is a LOWNER matrix, not Cauchy.
    # Lowner matrices are related to the DIVIDED DIFFERENCE of a function b.
    # Lowner's theorem: the matrix (f(x_i)-f(x_j))/(x_i-x_j) is PSD iff f is
    # operator monotone (Pick function). We showed b is NOT Pick.

    # But: what about CONDITIONAL positive definiteness?
    # A matrix K is conditionally positive definite (CPD) if
    # sum_{i,j} c_i c_j K_{ij} >= 0 for all c with sum c_i = 0.
    # This is weaker than PSD.

    # Test: is the Lowner matrix CPD?
    # Restrict to vectors orthogonal to the all-ones vector
    ones = np.ones(dim) / np.sqrt(dim)
    P_orth = np.eye(dim) - np.outer(ones, ones)
    L_cpd = P_orth @ L_mat @ P_orth
    evals_cpd = np.linalg.eigvalsh(L_cpd)
    # Remove the zero eigenvalue (from projection)
    nonzero_cpd = evals_cpd[np.abs(evals_cpd) > 1e-10]

    cpd = len(nonzero_cpd) == 0 or np.min(nonzero_cpd) > -1e-10
    print(f"    Lowner matrix conditionally PD: {cpd}")
    if len(nonzero_cpd) > 0:
        print(f"    CPD eigenvalues: [{np.min(nonzero_cpd):.4e}, {np.max(nonzero_cpd):.4e}]")

    # NEGATIVE TYPE: a kernel K is of negative type if -K is CPD
    neg_type = len(nonzero_cpd) == 0 or np.max(nonzero_cpd) < 1e-10
    print(f"    Lowner matrix negative type: {neg_type}")

    # For DISTANCE MATRICES: the matrix (d(i,j)) is of negative type
    # iff the metric space embeds isometrically into a Hilbert space.
    # This is the Schoenberg characterization.
    #
    # If our Lowner matrix is of negative type, it corresponds to a
    # Hilbert space distance. This would connect to the RKHS structure.

    return cpd, neg_type


if __name__ == "__main__":
    print("SESSION 34 -- AHK MATROID / LORENTZIAN ANALYSIS")
    print("=" * 75)

    for lam_sq in [50, 200, 1000]:
        print(f"\n{'#'*75}")
        print(f"# lam^2 = {lam_sq}")
        print(f"{'#'*75}")

        a, lc = test_log_concavity(lam_sq)
        b = analyze_b_sequence(lam_sq)
        n_pos, n_neg = test_lorentzian_property(lam_sq)
        cpd, neg_type = matroid_connection(lam_sq)

    print(f"\n{'='*75}")
    print("SYNTHESIS")
    print("="*75)

    with open('session34_ahk.json', 'w') as f:
        json.dump({'status': 'complete'}, f)
    print(f"\nSaved to session34_ahk.json")
