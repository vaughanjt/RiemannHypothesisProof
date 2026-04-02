"""
SESSION 39b — THE HILBERT TRANSFORM AS HODGE STAR

The Hilbert transform J: phi_n -> -sign(n)*phi_{-n} (antisymmetric n <-> -n)
gives J^2 = -Id on the n != 0 subspace. On null(W02), this is almost exact
(error 0.022, from the n=0 mode leaking through).

Refined test:
1. Project out the n=0 direction from null(W02)
2. Verify J^2 = -Id exactly on the 44-dim subspace
3. Decompose into H^{1,0} (eigenvalue +i) and H^{0,1} (eigenvalue -i)
4. Test: is M definite on each half?
5. Test: is the Hodge-Riemann form h = J*M positive semidefinite?
"""

import numpy as np
import sys
sys.path.insert(0, '.')
from connes_crossterm import build_all


def refined_hilbert_star(lam_sq):
    L_f = np.log(lam_sq)
    N = max(15, round(6 * L_f))
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)

    W02, M, QW = build_all(lam_sq, N, n_quad=10000)

    ew, ev = np.linalg.eigh(W02)
    thresh = np.max(np.abs(ew)) * 1e-10
    P_null = ev[:, np.abs(ew) <= thresh]
    d_null = P_null.shape[1]
    M_null = P_null.T @ M @ P_null

    print(f"REFINED HILBERT STAR: lam^2={lam_sq}, dim={dim}, null_dim={d_null}", flush=True)

    # Build J: antisymmetric n <-> -n pairing
    J_full = np.zeros((dim, dim))
    for i in range(dim):
        n = int(ns[i])
        j = dim - 1 - i  # index of -n
        if n > 0:
            J_full[i, j] = -1
        elif n < 0:
            J_full[i, j] = 1
    J_full = (J_full - J_full.T) / 2

    # Project J to null(W02)
    J_null = P_null.T @ J_full @ P_null

    # J^2 on null
    J_sq = J_null @ J_null
    err_full = np.linalg.norm(J_sq + np.eye(d_null), 'fro') / d_null
    print(f"\n  J^2 + Id error on full null: {err_full:.6f}", flush=True)

    # Eigenvalues of J_null
    evals_J = np.linalg.eigvals(J_null)
    print(f"  J eigenvalues: {np.sum(np.abs(evals_J - 1j) < 0.1)} near +i, "
          f"{np.sum(np.abs(evals_J + 1j) < 0.1)} near -i, "
          f"{np.sum(np.abs(evals_J) < 0.1)} near 0", flush=True)

    # Project out the "bad" direction (near-zero eigenvalue of J)
    # The bad direction is the n=0 component in null(W02)
    # First: find the kernel of J_null (the near-zero eigenvalue direction)
    evals_J_real = np.linalg.eigvalsh(J_null @ J_null + np.eye(d_null))
    bad_idx = np.where(evals_J_real > 0.5)[0]  # Where J^2 != -Id
    print(f"  Bad directions (J^2 + Id has eigenvalue > 0.5): {len(bad_idx)}", flush=True)

    # Alternative: use SVD of (J^2 + Id) to find the bad directions
    U, S, Vt = np.linalg.svd(J_sq + np.eye(d_null))
    n_bad = np.sum(S > 0.1)
    print(f"  SVD of (J^2+Id): {n_bad} significant singular values", flush=True)
    print(f"  Singular values: {S[:5]}", flush=True)

    # The GOOD subspace: where J^2 = -Id exactly
    good_basis = U[:, S < 0.1]  # Columns where J^2+Id ≈ 0
    d_good = good_basis.shape[1]
    print(f"  Good subspace dim: {d_good} (should be {d_null} - {n_bad} = {d_null - n_bad})", flush=True)

    # J restricted to good subspace
    J_good = good_basis.T @ J_null @ good_basis
    J_sq_good = J_good @ J_good
    err_good = np.linalg.norm(J_sq_good + np.eye(d_good), 'fro') / d_good
    print(f"  J^2 + Id error on good subspace: {err_good:.6e}", flush=True)

    # M restricted to good subspace
    M_good = good_basis.T @ M_null @ good_basis

    # Now: eigenvalues of J_good should be exactly +i and -i
    evals_Jg = np.linalg.eigvals(J_good)
    n_plus_i = np.sum(np.abs(evals_Jg - 1j) < 0.1)
    n_minus_i = np.sum(np.abs(evals_Jg + 1j) < 0.1)
    print(f"\n  J_good eigenvalues: {n_plus_i} at +i, {n_minus_i} at -i", flush=True)

    # Decompose into H^{1,0} (eigenvalue +i) and H^{0,1} (eigenvalue -i)
    # Projectors: P_+ = (I - iJ)/2, P_- = (I + iJ)/2
    P_plus = (np.eye(d_good) - 1j * J_good) / 2
    P_minus = (np.eye(d_good) + 1j * J_good) / 2

    rank_plus = round(np.real(np.trace(P_plus)))
    rank_minus = round(np.real(np.trace(P_minus)))
    print(f"  dim(H^{{1,0}}) = {rank_plus}", flush=True)
    print(f"  dim(H^{{0,1}}) = {rank_minus}", flush=True)

    # =====================================================
    # THE HODGE-RIEMANN TEST
    # =====================================================
    # For a curve: the Hodge-Riemann bilinear relation says
    # h(a) = i * <a, *bar(a)> > 0 on P^1
    # where * is the Hodge star.
    #
    # In our real setting: J plays the role of *.
    # The Hodge-Riemann form: h(phi) = <phi, J phi>
    # But <phi, J phi> is antisymmetric (J^T = -J), so <phi, J phi> = 0 for real phi.
    #
    # The CORRECT form: h involves BOTH the star AND the cup product (M).
    # h(phi) = <phi, J M phi> (the composition of star and cup)
    #
    # Or in the standard formulation:
    # The Hodge-Riemann form on P^k is: Q(a,b) = (-1)^{k(k-1)/2} <a, L^{n-k} *b>
    # For a curve (n=1, k=1): Q(a,b) = <a, *b> = <a, Jb>
    # But this is antisymmetric...
    #
    # The HERMITIAN form is: h(a,b) = i^k * Q(a, bar(b))
    # For k=1: h(a,b) = i * <a, J bar(b)>
    #
    # For REAL vectors phi: h(phi, phi) = i * <phi, J phi> = 0 (antisymmetric).
    #
    # THE RESOLUTION: Work with COMPLEX vectors.
    # Decompose phi into H^{1,0} and H^{0,1} parts:
    # phi = phi_+ + phi_- where phi_+ in H^{1,0}, phi_- in H^{0,1}
    #
    # Then: h(phi, phi) = i * [<phi_+, J phi_+> + <phi_-, J phi_->
    #                         + <phi_+, J phi_-> + <phi_-, J phi_+>]
    # = i * [i*||phi_+||^2 + (-i)*||phi_-||^2 + cross terms]
    # = -||phi_+||^2 + ||phi_-||^2 + i*(cross terms)
    #
    # Hmm, this gives an indefinite form, not positive-definite.
    #
    # Let me reconsider. The ACTUAL Hodge-Riemann form involves
    # the INTERSECTION PAIRING, not just the inner product.
    # On H^1 of a curve: the intersection pairing is
    # (a, b) = integral a ^ b (the cup product, antisymmetric)
    #
    # The Hodge-Riemann form: Q(a) = i * (a, bar(a)) = i * integral a ^ bar(a)
    # For a in H^{1,0}: bar(a) in H^{0,1}, and a ^ bar(a) is a (1,1)-form.
    # Q(a) = i * integral a ^ bar(a) > 0 (positive!).
    #
    # In our setting: the "cup product" is NOT the inner product.
    # It might be the antisymmetric form omega from Candidate 4.
    # Or it might be related to M.

    # Actually, let me try a DIFFERENT formulation.
    # The Weil operator C acts as i^{p-q} on H^{p,q}.
    # For H^{1,0}: p=1, q=0, so C = i.
    # For H^{0,1}: p=0, q=1, so C = -i.
    # In real form: C = J (the complex structure itself!).
    #
    # Hodge-Riemann says: the form <a, C*a> = <a, J*a> is positive (or negative)
    # definite on the primitive part, where * is the Hodge star.
    # But C = J and * = J (for a curve), so <a, J*a> = <a, J^2 a> = <a, -a> = -||a||^2.
    # That's negative-definite. Not useful (it's always -||a||^2).
    #
    # I'm confusing the two roles of J. Let me be more careful.
    #
    # On a Riemann surface: the Hodge * maps H^{1,0} -> H^{0,1} by
    # *(f dz) = -i f d(bar(z)). This is -i on H^{1,0} and +i on H^{0,1}.
    # The Weil operator C on H^1 is: C = i*(P_{1,0} - P_{0,1}) where P are projectors.
    # So C = i*(-iJ)/... hmm.
    #
    # Let me just COMPUTE the key forms and check signs.

    print(f"\n  ===== HODGE-RIEMANN TESTS =====", flush=True)

    # Test 1: <phi, J*M phi> for real phi in the good subspace
    JM = J_good @ M_good
    # Symmetrize: the form h(phi, psi) = (JM + (JM)^T)/2
    JM_sym = (JM + JM.T) / 2
    evals_JM = np.linalg.eigvalsh(JM_sym)
    n_pos = np.sum(evals_JM > 1e-6)
    n_neg = np.sum(evals_JM < -1e-6)
    n_zero = d_good - n_pos - n_neg

    print(f"\n  Form h1 = sym(J*M):", flush=True)
    print(f"  Eigenvalues: [{np.min(evals_JM):+.6f}, {np.max(evals_JM):+.6f}]", flush=True)
    print(f"  Positive: {n_pos}, Negative: {n_neg}, Zero: {n_zero}", flush=True)

    # Test 2: <phi, M*J phi>
    MJ = M_good @ J_good
    MJ_sym = (MJ + MJ.T) / 2
    evals_MJ = np.linalg.eigvalsh(MJ_sym)
    n_pos2 = np.sum(evals_MJ > 1e-6)
    n_neg2 = np.sum(evals_MJ < -1e-6)

    print(f"\n  Form h2 = sym(M*J):", flush=True)
    print(f"  Eigenvalues: [{np.min(evals_MJ):+.6f}, {np.max(evals_MJ):+.6f}]", flush=True)
    print(f"  Positive: {n_pos2}, Negative: {n_neg2}, Zero: {d_good - n_pos2 - n_neg2}", flush=True)

    # Test 3: the ANTI-commutator {J, M} = JM + MJ
    anti_comm = J_good @ M_good + M_good @ J_good
    evals_ac = np.linalg.eigvalsh(anti_comm)
    print(f"\n  Anti-commutator {{J, M}}:", flush=True)
    print(f"  Eigenvalues: [{np.min(evals_ac):+.6f}, {np.max(evals_ac):+.6f}]", flush=True)
    print(f"  ||{{J,M}}||_F = {np.linalg.norm(anti_comm, 'fro'):.4f}", flush=True)
    print(f"  (If {{J,M}}=0, then J and M anti-commute: Kahler condition)", flush=True)

    # Test 4: the COMMUTATOR [J, M]
    comm = J_good @ M_good - M_good @ J_good
    evals_comm = np.linalg.eigvalsh((comm + comm.T)/2)  # sym part
    print(f"\n  Commutator [J, M]:", flush=True)
    print(f"  ||[J, M]||_F = {np.linalg.norm(comm, 'fro'):.4f}", flush=True)
    print(f"  (If [J,M]=0, then J and M commute)", flush=True)

    # Test 5: does J PRESERVE the seeing/silent decomposition?
    evals_M, evecs_M = np.linalg.eigh(M_good)
    seeing_idx = np.where(np.abs(evals_M) > 0.001)[0]
    silent_idx = np.where(np.abs(evals_M) <= 0.001)[0]

    if len(seeing_idx) > 0:
        V_see = evecs_M[:, seeing_idx]
        V_sil = evecs_M[:, silent_idx]

        # Does J map seeing to seeing and silent to silent?
        J_see_to_sil = V_sil.T @ J_good @ V_see  # should be ~0 if J preserves
        J_sil_to_see = V_see.T @ J_good @ V_sil  # should be ~0 if J preserves

        cross_norm = np.linalg.norm(J_see_to_sil, 'fro')
        total_norm = np.linalg.norm(J_good, 'fro')

        print(f"\n  Does J preserve seeing/silent decomposition?", flush=True)
        print(f"  ||J(seeing -> silent)||_F = {cross_norm:.4f}", flush=True)
        print(f"  ||J||_F = {total_norm:.4f}", flush=True)
        print(f"  Cross-talk: {cross_norm/total_norm*100:.1f}%", flush=True)

        if cross_norm < 0.1 * total_norm:
            print(f"  J approximately preserves the decomposition!", flush=True)

            # M on seeing subspace, in J-eigenspaces
            M_see = V_see.T @ M_good @ V_see
            J_see = V_see.T @ J_good @ V_see

            # Check J_see^2 = -Id on seeing
            J_see_sq_err = np.linalg.norm(J_see @ J_see + np.eye(len(seeing_idx)), 'fro') / len(seeing_idx)
            print(f"  J^2 + Id error on seeing: {J_see_sq_err:.4e}", flush=True)

            # J*M on seeing
            JM_see = J_see @ M_see
            JM_see_sym = (JM_see + JM_see.T) / 2
            evals_JMs = np.linalg.eigvalsh(JM_see_sym)
            print(f"\n  sym(J*M) on SEEING subspace:", flush=True)
            print(f"  Eigenvalues: [{np.min(evals_JMs):+.4f}, {np.max(evals_JMs):+.4f}]", flush=True)
            print(f"  All positive: {np.min(evals_JMs) > -1e-6}", flush=True)
            print(f"  All negative: {np.max(evals_JMs) < 1e-6}", flush=True)
        else:
            print(f"  J MIXES seeing and silent modes (cross-talk {cross_norm/total_norm*100:.1f}%)", flush=True)

    # =====================================================
    # CRITICAL: Test at a second bandwidth for consistency
    # =====================================================
    return J_good, M_good, d_good


if __name__ == "__main__":
    print("SESSION 39b -- HILBERT STAR REFINED", flush=True)
    print("=" * 80, flush=True)

    for lam_sq in [50, 200, 500]:
        print(f"\n{'#' * 80}", flush=True)
        print(f"# lam^2 = {lam_sq}", flush=True)
        print(f"{'#' * 80}", flush=True)
        refined_hilbert_star(lam_sq)

    print(f"\nDone.", flush=True)
