"""
SESSION 40b — LEFSCHETZ OPERATOR SEARCH

The ±lambda pairing theorem ({J,[J,M]}=0) means [J,M] is NEVER definite
on the full good subspace. But in Hodge theory, definiteness is only
required on the PRIMITIVE part P = ker(L^k) for some Lefschetz operator L.

Question: does there exist a natural operator L such that [J,M] restricted
to some L-defined subspace IS definite?

Strategy: test candidate L operators from the arithmetic/geometric structure.
For each L, check whether restricting [J,M] to eigenspaces of L gives
definiteness.

Key insight: since [J,M] has ±lambda pairs mediated by J (v -> Jv swaps
+lambda to -lambda), L must NOT commute with J to separate the pairs.
If [L,J] = 0, then L preserves H^{1,0} and H^{0,1}, and the ±lambda
pairs stay together in any L-eigenspace.

Usage:
    python session40b_lefschetz_search.py [--lambda-sq 50]
"""

import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from connes_crossterm import build_all
from session33_sieve_bypass import compute_M_decomposition


def build_everything(lam_sq):
    """Build all operators on the good subspace."""
    L_f = np.log(lam_sq)
    N = max(15, round(6 * L_f))
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)

    # Full matrices
    W02, M, QW = build_all(lam_sq, N, n_quad=10000)

    # M decomposition
    M_diag, M_alpha, M_prime, M_full, vM = compute_M_decomposition(lam_sq, N, n_quad=10000)

    # Null(W02)
    ew, ev = np.linalg.eigh(W02)
    thresh = np.max(np.abs(ew)) * 1e-10
    P_null = ev[:, np.abs(ew) <= thresh]
    d_null = P_null.shape[1]

    # Hilbert J
    J_full = np.zeros((dim, dim))
    idx = np.arange(dim)
    j_idx = dim - 1 - idx
    J_full[idx[ns > 0], j_idx[ns > 0]] = -1
    J_full[idx[ns < 0], j_idx[ns < 0]] = 1
    J_full = (J_full - J_full.T) / 2

    # Good subspace
    J_null = P_null.T @ J_full @ P_null
    J_sq = J_null @ J_null
    U, S, _ = np.linalg.svd(J_sq + np.eye(d_null))
    good = U[:, S < 0.1]
    d = good.shape[1]
    if d % 2 != 0:
        good = good[:, :-1]
        d = good.shape[1]

    def proj(A):
        """Project full-space operator to good subspace."""
        return good.T @ (P_null.T @ A @ P_null) @ good

    J0 = proj(J_full)
    M0 = proj(M)

    # ── Candidate Lefschetz operators (in full space) ──

    candidates = {}

    # 1. Number operator: N_op[i,j] = n * delta_{ij}
    N_op = np.diag(ns)
    candidates['N (mode number)'] = proj(N_op)

    # 2. |N| operator: absolute mode number (frequency)
    absN = np.diag(np.abs(ns))
    candidates['|N| (frequency)'] = proj(absN)

    # 3. N^2 operator
    N2 = np.diag(ns**2)
    candidates['N^2'] = proj(N2)

    # 4. Position operator x in Fourier basis
    # x_{nm} = L/2 if n=m, else L/(2*pi*i*(m-n))
    x_op = np.zeros((dim, dim))
    for i in range(dim):
        n = ns[i]
        x_op[i, i] = L_f / 2
        for j in range(dim):
            m = ns[j]
            if n != m:
                x_op[i, j] = L_f / (2 * np.pi * (m - n))
                # Note: this is the imaginary part of the complex matrix element
                # The real position operator has: x_{nm} = L_f * sin(pi*(m-n)) / (pi*(m-n))
                # For integer m-n: sin(pi*k) = 0, so off-diagonal = 0
                # Actually for functions on [0,L] with Fourier basis e^{2*pi*i*n*x/L}:
                # <e_n, x * e_m> = integral_0^L e^{-2pi*i*n*x/L} * x * e^{2pi*i*m*x/L} dx / L
                # = integral_0^L x * e^{2pi*i*(m-n)*x/L} dx / L
    # Recompute properly
    x_op = np.zeros((dim, dim))
    for i in range(dim):
        ni = int(ns[i])
        for j in range(dim):
            nj = int(ns[j])
            diff = nj - ni
            if diff == 0:
                x_op[i, j] = L_f / 2  # <e_n, x*e_n> = L/2
            else:
                # integral_0^L x * e^{2pi*i*diff*x/L} dx / L
                # = L/(2*pi*i*diff) * [x*e^{...}]_0^L - L/(2*pi*i*diff) * integral e^{...} dx / L
                # = L/(2*pi*i*diff) - 0 = L/(2*pi*i*diff)
                x_op[i, j] = L_f / (2 * np.pi * 1j * diff)
    # Take real part (the operator should be self-adjoint)
    x_op = np.real(x_op)
    x_op = (x_op + x_op.T) / 2
    candidates['x (position)'] = proj(x_op)

    # 5. Scaling generator: x*d/dx in Fourier basis
    # (x*d/dx)_{nm} = (2*pi*i*m/L) * x_{nm}
    xddx = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        ni = int(ns[i])
        for j in range(dim):
            nj = int(ns[j])
            diff = nj - ni
            if diff == 0:
                xddx[i, j] = 2 * np.pi * 1j * nj / L_f * L_f / 2  # = pi*i*nj
            else:
                xddx[i, j] = 2 * np.pi * 1j * nj / L_f * L_f / (2 * np.pi * 1j * diff)
                # = nj / diff
    # Symmetrize (take Hermitian part for self-adjoint operator)
    xddx_sa = (xddx + xddx.conj().T) / 2
    candidates['x*d/dx (scaling)'] = proj(np.real(xddx_sa))

    # 6. M_diag (analytic part)
    candidates['M_diag (analytic)'] = proj(M_diag)

    # 7. M_prime (arithmetic part)
    candidates['M_prime (primes)'] = proj(M_prime)

    # 8. M_alpha (off-diagonal analytic)
    candidates['M_alpha'] = proj(M_alpha)

    # 9. W02 on null space (should be ~0 but let's check)
    candidates['W02 (check)'] = proj(W02)

    # 10. [M_diag, M_prime] — interaction of analytic and arithmetic
    comm_dp = M_diag @ M_prime - M_prime @ M_diag
    candidates['[M_diag, M_prime]'] = proj(comm_dp)

    # 11. Antisymmetric N: imaginary part of mode coupling
    # J itself as a candidate (the complex structure)
    candidates['J (complex str.)'] = J0

    # 12. QW = W02 - M (the "signal" matrix)
    candidates['QW'] = proj(QW)

    return J0, M0, d, candidates, ns, N, dim


def test_lefschetz(L, J, comm, label, d):
    """Test whether L defines a primitive subspace where [J,M] is definite.

    For each eigenspace of L (or union of eigenspaces), restrict [J,M]
    and check definiteness.
    """
    # Eigendecompose L
    L_sym = (L + L.T) / 2  # ensure symmetric
    evals_L, evecs_L = np.linalg.eigh(L_sym)

    # Does L commute with J? (crucial: if [L,J]=0, L can't separate ±lambda pairs)
    LJ_comm = np.linalg.norm(L @ J - J @ L, 'fro')
    LJ_rel = LJ_comm / (np.linalg.norm(L, 'fro') * np.linalg.norm(J, 'fro') + 1e-20)

    results = {
        'label': label,
        'LJ_comm': LJ_comm,
        'LJ_rel': LJ_rel,
        'evals_L_range': (evals_L[0], evals_L[-1]),
        'tests': [],
    }

    # Strategy 1: ker(L) — eigenvalues near zero
    for thresh_frac in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
        thresh = thresh_frac * np.max(np.abs(evals_L))
        if thresh < 1e-12:
            thresh = 1e-12
        mask = np.abs(evals_L) <= thresh
        dim_sub = int(np.sum(mask))
        if dim_sub < 2 or dim_sub >= d:
            continue
        V = evecs_L[:, mask]
        comm_sub = V.T @ comm @ V
        comm_sub = (comm_sub + comm_sub.T) / 2
        ev_sub = np.linalg.eigvalsh(comm_sub)
        n_pos = int(np.sum(ev_sub > 1e-8))
        n_neg = int(np.sum(ev_sub < -1e-8))
        definite = (n_pos == 0) or (n_neg == 0)
        results['tests'].append({
            'type': f'ker(L, {thresh_frac:.0%})',
            'dim': dim_sub,
            'n_pos': n_pos, 'n_neg': n_neg,
            'definite': definite,
            'ev_range': (ev_sub[0], ev_sub[-1]),
        })

    # Strategy 2: negative eigenspace of L
    mask = evals_L < -1e-8
    dim_sub = int(np.sum(mask))
    if 2 <= dim_sub < d:
        V = evecs_L[:, mask]
        comm_sub = V.T @ comm @ V
        comm_sub = (comm_sub + comm_sub.T) / 2
        ev_sub = np.linalg.eigvalsh(comm_sub)
        n_pos = int(np.sum(ev_sub > 1e-8))
        n_neg = int(np.sum(ev_sub < -1e-8))
        results['tests'].append({
            'type': 'L < 0',
            'dim': dim_sub,
            'n_pos': n_pos, 'n_neg': n_neg,
            'definite': (n_pos == 0) or (n_neg == 0),
            'ev_range': (ev_sub[0], ev_sub[-1]),
        })

    # Strategy 3: positive eigenspace of L
    mask = evals_L > 1e-8
    dim_sub = int(np.sum(mask))
    if 2 <= dim_sub < d:
        V = evecs_L[:, mask]
        comm_sub = V.T @ comm @ V
        comm_sub = (comm_sub + comm_sub.T) / 2
        ev_sub = np.linalg.eigvalsh(comm_sub)
        n_pos = int(np.sum(ev_sub > 1e-8))
        n_neg = int(np.sum(ev_sub < -1e-8))
        results['tests'].append({
            'type': 'L > 0',
            'dim': dim_sub,
            'n_pos': n_pos, 'n_neg': n_neg,
            'definite': (n_pos == 0) or (n_neg == 0),
            'ev_range': (ev_sub[0], ev_sub[-1]),
        })

    # Strategy 4: bottom-k eigenspace (smallest eigenvalues of L)
    for k_frac in [0.25, 0.5]:
        k = max(2, int(d * k_frac))
        if k >= d:
            continue
        V = evecs_L[:, :k]
        comm_sub = V.T @ comm @ V
        comm_sub = (comm_sub + comm_sub.T) / 2
        ev_sub = np.linalg.eigvalsh(comm_sub)
        n_pos = int(np.sum(ev_sub > 1e-8))
        n_neg = int(np.sum(ev_sub < -1e-8))
        results['tests'].append({
            'type': f'bottom-{k_frac:.0%}',
            'dim': k,
            'n_pos': n_pos, 'n_neg': n_neg,
            'definite': (n_pos == 0) or (n_neg == 0),
            'ev_range': (ev_sub[0], ev_sub[-1]),
        })

    # Strategy 5: top-k eigenspace
    for k_frac in [0.25, 0.5]:
        k = max(2, int(d * k_frac))
        if k >= d:
            continue
        V = evecs_L[:, -k:]
        comm_sub = V.T @ comm @ V
        comm_sub = (comm_sub + comm_sub.T) / 2
        ev_sub = np.linalg.eigvalsh(comm_sub)
        n_pos = int(np.sum(ev_sub > 1e-8))
        n_neg = int(np.sum(ev_sub < -1e-8))
        results['tests'].append({
            'type': f'top-{k_frac:.0%}',
            'dim': k,
            'n_pos': n_pos, 'n_neg': n_neg,
            'definite': (n_pos == 0) or (n_neg == 0),
            'ev_range': (ev_sub[0], ev_sub[-1]),
        })

    return results


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda-sq', type=int, default=50)
    args = parser.parse_args()

    print(f"\n{'#'*60}", flush=True)
    print(f"  LEFSCHETZ OPERATOR SEARCH", flush=True)
    print(f"  lam^2 = {args.lambda_sq}", flush=True)
    print(f"{'#'*60}\n", flush=True)

    print("  Building matrices...", flush=True)
    t0 = time.time()
    J, M, d, candidates, ns, N, dim = build_everything(args.lambda_sq)
    print(f"  Built in {time.time()-t0:.1f}s: dim={dim}, good={d}\n", flush=True)

    # Commutator [J,M]
    from session40_star_optimizer import project_to_complex_structure
    J = project_to_complex_structure(J)
    comm = J @ M - M @ J
    evals_comm = np.linalg.eigvalsh(comm)

    print(f"  [J,M] baseline: [{evals_comm[0]:+.4f} ... {evals_comm[-1]:+.4f}]", flush=True)
    print(f"  Positive: {np.sum(evals_comm > 1e-8)}, "
          f"Negative: {np.sum(evals_comm < -1e-8)}\n", flush=True)

    # Test each candidate
    hits = []
    for label, L in candidates.items():
        res = test_lefschetz(L, J, comm, label, d)

        # Print results
        definite_found = any(t['definite'] for t in res['tests'])
        marker = " <<<" if definite_found else ""
        print(f"  {label}{marker}", flush=True)
        print(f"    ||[L,J]|| = {res['LJ_comm']:.4f} "
              f"(rel: {res['LJ_rel']:.4f})", flush=True)
        print(f"    L eigenvalues: [{res['evals_L_range'][0]:+.4f} ... "
              f"{res['evals_L_range'][1]:+.4f}]", flush=True)

        for t in res['tests']:
            def_str = "DEFINITE!" if t['definite'] else f"+:{t['n_pos']} -:{t['n_neg']}"
            print(f"    {t['type']:20s} dim={t['dim']:3d}  "
                  f"[J,M]=[{t['ev_range'][0]:+.4f},{t['ev_range'][1]:+.4f}]  "
                  f"{def_str}", flush=True)

        if definite_found:
            hits.append(res)
        print(flush=True)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print(f"  SUMMARY", flush=True)
    print(f"{'='*60}\n", flush=True)

    if hits:
        print(f"  DEFINITE SUBSPACES FOUND: {len(hits)} candidates\n", flush=True)
        for h in hits:
            print(f"  ** {h['label']} **", flush=True)
            for t in h['tests']:
                if t['definite']:
                    sign = "all negative" if t['n_neg'] > 0 else "all positive"
                    print(f"     {t['type']}: dim={t['dim']}, {sign}, "
                          f"range=[{t['ev_range'][0]:+.6f}, {t['ev_range'][1]:+.6f}]",
                          flush=True)
            print(flush=True)
    else:
        print("  No definite subspaces found with any candidate.", flush=True)
        print("  The obstruction may require a fundamentally different L.", flush=True)
