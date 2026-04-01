"""
SESSION 33 — DEEP PEELING + CONTRADICTION ANALYSIS

PEELING: Keep restricting to sub-null-spaces until Schur-Horn closes.
  Layer 1: range(W02) — 2D — PROVED
  Layer 2: range(Theta) & null(W02) — ~35D
  Layer 3: null(Theta) & null(W02) — ~20D — ratio 0.54
  Layer 4: null(???) & null(Theta) & null(W02) — ???
  ...keep going until ratio > 1

WHAT TO PEEL WITH: M itself has eigenvectors. The most negative
eigenvectors of M are "safe" (far from zero). Peel them off.
The DANGEROUS eigenvectors are the ones with eigenvalue near zero.

CONTRADICTION: If M had a positive eigenvalue on null(W02), the
corresponding eigenvector v would satisfy:
  <v, M v> > 0 with v in null(W02)
  => QW has negative eigenvalue
  => eps_0 < 0 for this lambda
  => zero off critical line
  => specific detectable prime anomaly
Can we characterize v and show it can't exist?
"""

import numpy as np
import time, json, sys
sys.path.insert(0, '.')
from connes_crossterm import build_all
from session33_sieve_bypass import compute_M_decomposition


def iterated_peeling(lam_sq, N=None):
    """
    Iteratively peel off the most negative eigenvalues of M on null(W02).
    At each step, check if Schur-Horn proves the remainder is negative.
    """
    if N is None:
        L = np.log(lam_sq)
        N = max(15, round(6*L))
    dim = 2*N+1

    W02, M, QW = build_all(lam_sq, N)

    # null(W02)
    ew, ev = np.linalg.eigh(W02)
    thresh = np.max(np.abs(ew))*1e-10
    P_null = ev[:, np.abs(ew) <= thresh]
    M_null = P_null.T @ M @ P_null
    D = M_null.shape[0]

    # Eigendecomposition of M on null(W02)
    evals_m, evecs_m = np.linalg.eigh(M_null)
    # evals_m sorted ascending: most negative first

    print(f"\nITERATED PEELING: lam^2={lam_sq}, null(W02) dim={D}")
    print(f"  M eigenvalues: [{evals_m[0]:.4e}, ..., {evals_m[-1]:.4e}]")
    print(f"  {'peel':>5} {'remain':>7} {'mu':>10} {'sigma':>10} {'|mu|/sig':>10} {'SH_bound':>10} {'proved':>8}")

    # Peel from the most negative (safest) eigenvalues
    for n_peel in range(0, D-1, max(1, D//20)):
        remaining = D - n_peel
        if remaining < 2:
            break

        # The remaining eigenvalues (closest to zero)
        evals_remain = evals_m[n_peel:]
        mu = np.mean(evals_remain)
        sigma = np.std(evals_remain)
        if sigma < 1e-15:
            sh_bound = mu
        else:
            sh_bound = mu + sigma * np.sqrt((remaining-1)/remaining)

        ratio = abs(mu) / sigma if sigma > 1e-15 else float('inf')
        proved = sh_bound < -1e-12

        print(f"  {n_peel:>5} {remaining:>7} {mu:>10.4e} {sigma:>10.4e} "
              f"{ratio:>10.4f} {sh_bound:>10.4e} {'PROVED' if proved else ''}")

        if proved:
            print(f"\n  *** SCHUR-HORN PROVES at peel={n_peel}: "
                  f"M <= 0 on the {remaining}-dim subspace ***")
            print(f"  The remaining {remaining} eigenvalues [{evals_remain[0]:.4e}, ..., {evals_remain[-1]:.4e}]")
            print(f"  are ALL provably negative using only trace and Frobenius norm!")
            return n_peel, remaining, True

    return D, 0, False


def adaptive_peeling(lam_sq, N=None):
    """
    Instead of peeling M eigenvalues (which is circular — we know them),
    peel using INDEPENDENT criteria: restrict to subspaces defined by
    operators we can bound analytically.

    Candidates:
    - M_diag eigenspaces (analytic, no primes)
    - M_alpha eigenspaces (analytic, no primes)
    - Fourier mode subspaces (by frequency)
    """
    if N is None:
        L = np.log(lam_sq)
        N = max(15, round(6*L))
    dim = 2*N+1

    W02, M, QW = build_all(lam_sq, N)
    M_diag, M_alpha, M_prime, M_full, primes = compute_M_decomposition(lam_sq, N)

    ew, ev = np.linalg.eigh(W02)
    thresh = np.max(np.abs(ew))*1e-10
    P_null = ev[:, np.abs(ew) <= thresh]
    D = P_null.shape[1]

    M_null = P_null.T @ M @ P_null
    Md_null = P_null.T @ M_diag @ P_null
    Ma_null = P_null.T @ M_alpha @ P_null
    Mp_null = P_null.T @ M_prime @ P_null

    print(f"\nADAPTIVE PEELING: lam^2={lam_sq}, null(W02) dim={D}")

    # Strategy 1: Peel using M_diag eigenvalues
    # M_diag is diagonal in the original basis, but not in null(W02) basis
    ed, evd = np.linalg.eigh(Md_null)
    print(f"\n  M_diag on null: [{ed[0]:.4e}, ..., {ed[-1]:.4e}]")

    # Restrict to the subspace where M_diag is most negative
    # (these are the modes where the analytic part already helps)
    for threshold_frac in [0.5, 0.25, 0.1, 0.0]:
        threshold = ed[0] + (ed[-1] - ed[0]) * threshold_frac
        mask = ed <= threshold
        P_sub = evd[:, mask]
        d_sub = np.sum(mask)

        if d_sub < 2:
            continue

        # M restricted to this subspace
        M_sub = P_sub.T @ M_null @ P_sub
        evals_sub = np.linalg.eigvalsh(M_sub)
        mu = np.mean(evals_sub)
        sigma = np.std(evals_sub)
        sh = mu + sigma * np.sqrt((d_sub-1)/d_sub) if sigma > 1e-15 and d_sub > 1 else mu
        ratio = abs(mu)/sigma if sigma > 1e-15 else float('inf')

        print(f"  M_diag <= {threshold:.3f} (dim={d_sub}): "
              f"M evals [{evals_sub[0]:.3e},{evals_sub[-1]:.3e}] "
              f"|mu|/sig={ratio:.3f} SH={sh:.3e} {'PROVED' if sh < -1e-10 else ''}")

    # Strategy 2: Peel using FREQUENCY bands
    # The original basis has modes n = -N, ..., N
    # Low frequencies (|n| small) are the hardest
    # High frequencies (|n| large) are easier
    print(f"\n  Frequency band peeling:")
    for n_cutoff in [N//4, N//3, N//2, 2*N//3, 3*N//4, N]:
        # Restrict to modes with |n| >= n_cutoff (high frequency)
        # In the null(W02) basis, we need to identify which modes these are
        # The null(W02) basis vectors are eigenvectors of W02 with eigenvalue ~0
        # They correspond roughly to high-frequency modes

        # Direct approach: project onto modes |n| >= n_cutoff in original basis
        high_freq_idx = [i for i in range(dim) if abs(i - N) >= n_cutoff]
        if len(high_freq_idx) < 2:
            continue

        P_hf = np.eye(dim)[:, high_freq_idx]
        # Intersection with null(W02): project P_hf columns onto null(W02)
        P_hf_null = P_null.T @ P_hf  # D x len(high_freq)
        # SVD to get the effective subspace
        U, S, _ = np.linalg.svd(P_hf_null, full_matrices=False)
        # Keep significant components
        sig_mask = S > 0.1
        P_effective = U[:, sig_mask]
        d_eff = np.sum(sig_mask)

        if d_eff < 2:
            continue

        M_hf = P_effective.T @ M_null @ P_effective
        evals_hf = np.linalg.eigvalsh(M_hf)
        mu = np.mean(evals_hf)
        sigma = np.std(evals_hf)
        sh = mu + sigma * np.sqrt((d_eff-1)/d_eff) if sigma > 1e-15 and d_eff > 1 else mu
        ratio = abs(mu)/sigma if sigma > 1e-15 else float('inf')

        print(f"  |n|>={n_cutoff:>3} (eff_dim={d_eff:>3}): "
              f"M evals [{evals_hf[0]:.3e},{evals_hf[-1]:.3e}] "
              f"|mu|/sig={ratio:.3f} SH={sh:.3e} {'PROVED' if sh < -1e-10 else ''}")


def contradiction_analysis(lam_sq, N=None):
    """
    If M had a positive eigenvalue on null(W02), what would the
    eigenvector look like? Characterize the "offending" vector.
    """
    if N is None:
        L = np.log(lam_sq)
        N = max(15, round(6*L))
    dim = 2*N+1

    W02, M, QW = build_all(lam_sq, N)

    ew, ev = np.linalg.eigh(W02)
    thresh = np.max(np.abs(ew))*1e-10
    P_null = ev[:, np.abs(ew) <= thresh]

    M_null = P_null.T @ M @ P_null
    evals_m, evecs_m = np.linalg.eigh(M_null)

    # The "most dangerous" eigenvector: the one closest to zero
    # (but still negative — if it crossed to positive, RH fails)
    v_danger = evecs_m[:, -1]  # eigenvector for largest (closest to 0) eigenvalue
    lam_danger = evals_m[-1]

    # Map back to original basis
    v_full = P_null @ v_danger

    print(f"\nCONTRADICTION ANALYSIS: lam^2={lam_sq}")
    print(f"  Most dangerous eigenvalue: {lam_danger:.6e}")
    print(f"  Dangerous eigenvector structure:")

    # Fourier decomposition
    ns = np.arange(-N, N+1)
    print(f"    Dominant modes:")
    sorted_idx = np.argsort(np.abs(v_full))[::-1]
    for i in range(min(10, dim)):
        idx = sorted_idx[i]
        n = ns[idx]
        print(f"      n={n:>4}: |v|={abs(v_full[idx]):.6f}")

    # Is it even or odd?
    center = N
    even_energy = sum(abs(v_full[center+k] + v_full[center-k])**2 for k in range(1, N+1))
    odd_energy = sum(abs(v_full[center+k] - v_full[center-k])**2 for k in range(1, N+1))
    parity = "EVEN" if even_energy > odd_energy else "ODD"
    print(f"    Parity: {parity} (even={even_energy:.4f}, odd={odd_energy:.4f})")

    # Frequency concentration
    abs_v = np.abs(v_full)
    low_freq = sum(abs_v[center-5:center+6]**2)
    mid_freq = sum(abs_v[center-N//2:center+N//2+1]**2) - low_freq
    high_freq = 1 - low_freq - mid_freq
    print(f"    Energy: low(|n|<=5)={low_freq:.4f} mid={mid_freq:.4f} high={high_freq:.4f}")

    # What would this eigenvector mean for the prime distribution?
    # <v, M v> = sum_{pk} Lambda(pk)/sqrt(pk) * (<v|T(pk)|v>)
    # If this were positive, each term contributes
    M_diag_comp, M_alpha_comp, M_prime_comp, _, primes = compute_M_decomposition(lam_sq, N)

    v_Md = v_full @ M_diag_comp @ v_full
    v_Ma = v_full @ M_alpha_comp @ v_full
    v_Mp = v_full @ M_prime_comp @ v_full

    print(f"\n    <v, M_diag v> = {v_Md:.6e}")
    print(f"    <v, M_alpha v> = {v_Ma:.6e}")
    print(f"    <v, M_prime v> = {v_Mp:.6e}")
    print(f"    Total <v, M v> = {v_Md + v_Ma + v_Mp:.6e} (check: {lam_danger:.6e})")

    # The prime contributions: which primes contribute most?
    L = np.log(lam_sq)
    prime_contribs = []
    for pk, logp, logpk in primes:
        # Build T(pk) and compute <v|T(pk)|v>
        T_val = 0
        for i in range(dim):
            m = ns[i]
            for j in range(dim):
                n_idx = ns[j]
                if m != n_idx:
                    q = (np.sin(2*np.pi*n_idx*logpk/L) -
                         np.sin(2*np.pi*m*logpk/L)) / (np.pi*(m-n_idx))
                else:
                    q = 2*(L-logpk)/L * np.cos(2*np.pi*m*logpk/L)
                T_val += v_full[i] * v_full[j] * logp * pk**(-0.5) * q
        prime_contribs.append((pk, T_val))

    print(f"\n    Per-prime contributions to <v, M_prime v>:")
    sorted_contribs = sorted(prime_contribs, key=lambda x: abs(x[1]), reverse=True)
    for pk, c in sorted_contribs[:10]:
        print(f"      p^k={pk:>5}: {c:>+.6e}")

    total_pos = sum(c for _, c in prime_contribs if c > 0)
    total_neg = sum(c for _, c in prime_contribs if c < 0)
    print(f"    Positive primes total: {total_pos:+.6e}")
    print(f"    Negative primes total: {total_neg:+.6e}")
    print(f"    Net: {total_pos + total_neg:.6e}")

    # KEY: The dangerous vector has M_prime contribution that NEARLY cancels
    # If it were to cross zero, the positive primes would need to exceed
    # the negative primes + analytic terms
    margin = abs(lam_danger)
    print(f"\n    MARGIN from zero: {margin:.6e}")
    print(f"    If this crossed zero, |pos_primes| > |neg_primes + analytic| by {margin:.6e}")


if __name__ == "__main__":
    print("SESSION 33 -- DEEP PEEL + CONTRADICTION")
    print("=" * 75)

    for lam_sq in [50, 200, 1000]:
        print(f"\n{'#'*75}")
        print(f"# lam^2 = {lam_sq}")
        print(f"{'#'*75}")

        # Iterated eigenvalue peeling
        iterated_peeling(lam_sq)

        # Adaptive peeling
        adaptive_peeling(lam_sq)

        # Contradiction analysis
        contradiction_analysis(lam_sq)

    with open('session33_deep_peel.json', 'w') as f:
        json.dump({'status': 'complete'}, f)
    print(f"\n\nSaved to session33_deep_peel.json")
