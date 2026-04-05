"""
SESSION 48c -- THREE VIABILITY CHECKS FOR TRACE FORMULA APPROACH

Check 1: Can we prove |M| > |W02| from matrix structure alone?
  - Eigenvalue interlacing between M and W02
  - Trace comparison: Tr(M) vs Tr(W02)
  - Operator norm comparison

Check 2: Does the margin B/|W02| have a lower bound?
  - Compute at many lambda^2 values
  - Fit decay/growth model
  - Check if margin stabilizes or decays to 0

Check 3: Is there a non-circular route?
  - M is built from primes (log(p)/p^k terms)
  - W02 is built from analytic functions (no primes)
  - Can their difference be bounded using PNT-type results?
"""

import numpy as np
import sys
import time

sys.path.insert(0, '.')
from connes_crossterm import build_all


def full_decomposition(lam_sq):
    """Complete matrix decomposition at a given lambda^2."""
    L_f = np.log(lam_sq)
    N = max(15, round(6 * L_f))
    dim = 2 * N + 1

    W02, M, QW = build_all(lam_sq, N)

    ns = np.arange(-N, N + 1, dtype=float)
    w = ns / (L_f**2 + (4*np.pi)**2 * ns**2)
    w[N] = 0.0
    w_hat = w / np.linalg.norm(w)

    # Quadratic forms
    b = float(w_hat @ QW @ w_hat)
    w02_qf = float(w_hat @ W02 @ w_hat)
    m_qf = float(w_hat @ M @ w_hat)

    # Eigenvalues
    w02_eigs = np.linalg.eigvalsh(W02)
    m_eigs = np.linalg.eigvalsh(M)
    qw_eigs = np.linalg.eigvalsh(QW)

    # Traces
    tr_w02 = np.trace(W02)
    tr_m = np.trace(M)
    tr_qw = np.trace(QW)

    # Frobenius norms
    fn_w02 = np.linalg.norm(W02, 'fro')
    fn_m = np.linalg.norm(M, 'fro')

    # Operator norms (largest singular value)
    on_w02 = np.linalg.norm(W02, 2)
    on_m = np.linalg.norm(M, 2)

    return {
        'lam_sq': lam_sq, 'L': L_f, 'N': N, 'dim': dim,
        'barrier': b, 'w02_qf': w02_qf, 'm_qf': m_qf,
        'w02_eigs': w02_eigs, 'm_eigs': m_eigs, 'qw_eigs': qw_eigs,
        'tr_w02': tr_w02, 'tr_m': tr_m, 'tr_qw': tr_qw,
        'fn_w02': fn_w02, 'fn_m': fn_m,
        'on_w02': on_w02, 'on_m': on_m,
        'w_hat': w_hat, 'W02': W02, 'M': M, 'QW': QW,
    }


if __name__ == '__main__':
    print()
    print('#' * 72)
    print('  SESSION 48c -- VIABILITY CHECKS')
    print('#' * 72)

    # Compute at multiple scales
    lam_sq_values = [20, 50, 100, 200, 500]
    results = []

    print('\n  Computing decompositions...')
    for lam_sq in lam_sq_values:
        t0 = time.time()
        r = full_decomposition(lam_sq)
        results.append(r)
        print(f'    lam^2={lam_sq:6.0f}  dim={r["dim"]:3d}  B={r["barrier"]:.6f}  ({time.time()-t0:.1f}s)')

    # =============================================================
    # CHECK 1: Matrix structure -- can we prove |M| > |W02|?
    # =============================================================
    print('\n\n' + '=' * 72)
    print('  CHECK 1: MATRIX STRUCTURE -- |M| vs |W02|')
    print('=' * 72)

    # 1a. Trace comparison
    print('\n  -- 1a. TRACE COMPARISON --')
    print(f'  {"lam^2":>8} {"Tr(W02)":>12} {"Tr(M)":>12} {"Tr(QW)":>12} {"Tr(M)-Tr(W02)":>14}')
    print('  ' + '-' * 62)
    for r in results:
        diff = r['tr_m'] - r['tr_w02']
        print(f'  {r["lam_sq"]:8.0f} {r["tr_w02"]:12.4f} {r["tr_m"]:12.4f} '
              f'{r["tr_qw"]:12.4f} {diff:14.4f}')

    # 1b. Eigenvalue comparison
    print('\n  -- 1b. EIGENVALUE COMPARISON --')
    print(f'  {"lam^2":>8} {"W02 min":>10} {"W02 max":>10} {"M min":>10} {"M max":>10} '
          f'{"QW min":>10} {"QW max":>10}')
    print('  ' + '-' * 72)
    for r in results:
        print(f'  {r["lam_sq"]:8.0f} {r["w02_eigs"][0]:10.4f} {r["w02_eigs"][-1]:10.4f} '
              f'{r["m_eigs"][0]:10.4f} {r["m_eigs"][-1]:10.4f} '
              f'{r["qw_eigs"][0]:10.6f} {r["qw_eigs"][-1]:10.4f}')

    # 1c. How many positive eigenvalues does QW have?
    print('\n  -- 1c. QW EIGENVALUE SIGN DISTRIBUTION --')
    print(f'  {"lam^2":>8} {"dim":>5} {"n_pos":>6} {"n_neg":>6} {"n_zero":>6} {"min_eig":>12} {"2nd_min":>12}')
    print('  ' + '-' * 62)
    for r in results:
        eigs = r['qw_eigs']
        n_pos = np.sum(eigs > 1e-10)
        n_neg = np.sum(eigs < -1e-10)
        n_zero = len(eigs) - n_pos - n_neg
        second = eigs[1] if len(eigs) > 1 else 0
        print(f'  {r["lam_sq"]:8.0f} {r["dim"]:5d} {n_pos:6d} {n_neg:6d} {n_zero:6d} '
              f'{eigs[0]:12.6f} {second:12.6f}')

    # 1d. Operator norm comparison
    print('\n  -- 1d. OPERATOR NORM COMPARISON --')
    print(f'  {"lam^2":>8} {"||W02||_op":>12} {"||M||_op":>12} {"||M||/||W02||":>14} '
          f'{"||W02||_F":>12} {"||M||_F":>12}')
    print('  ' + '-' * 72)
    for r in results:
        ratio = r['on_m'] / r['on_w02'] if r['on_w02'] > 0 else float('inf')
        print(f'  {r["lam_sq"]:8.0f} {r["on_w02"]:12.4f} {r["on_m"]:12.4f} {ratio:14.6f} '
              f'{r["fn_w02"]:12.4f} {r["fn_m"]:12.4f}')

    # =============================================================
    # CHECK 2: Does B/|W02| have a lower bound?
    # =============================================================
    print('\n\n' + '=' * 72)
    print('  CHECK 2: MARGIN STABILITY -- does B/|W02| converge?')
    print('=' * 72)

    print(f'\n  {"lam^2":>8} {"L":>8} {"B(L)":>10} {"|W02|":>10} {"B/|W02|":>10} '
          f'{"B*L":>10} {"B*L^2":>10}')
    print('  ' + '-' * 72)
    for r in results:
        ratio = r['barrier'] / abs(r['w02_qf']) if r['w02_qf'] != 0 else 0
        print(f'  {r["lam_sq"]:8.0f} {r["L"]:8.3f} {r["barrier"]:10.6f} '
              f'{abs(r["w02_qf"]):10.4f} {ratio:10.6f} '
              f'{r["barrier"]*r["L"]:10.6f} {r["barrier"]*r["L"]**2:10.4f}')

    # Fit B ~ c * L^alpha
    if len(results) >= 3:
        Ls = np.array([r['L'] for r in results])
        Bs = np.array([r['barrier'] for r in results])
        # Log-log fit
        valid = Bs > 0
        if np.sum(valid) >= 2:
            log_L = np.log(Ls[valid])
            log_B = np.log(Bs[valid])
            alpha, log_c = np.polyfit(log_L, log_B, 1)
            c = np.exp(log_c)
            print(f'\n  Fit: B(L) ~ {c:.4f} * L^({alpha:.3f})')
            if alpha < 0:
                print(f'  B DECAYS as L grows (exponent {alpha:.3f})')
                print(f'  At L=100 (lam^2~2.7e43): B ~ {c * 100**alpha:.2e}')
            else:
                print(f'  B GROWS as L grows (exponent {alpha:.3f})')

        # Also fit B/|W02| ~ c * L^alpha
        ratios = np.array([r['barrier']/abs(r['w02_qf']) for r in results])
        log_ratio = np.log(ratios[valid])
        alpha2, log_c2 = np.polyfit(log_L[valid], log_ratio[valid], 1)
        c2 = np.exp(log_c2)
        print(f'\n  Fit: B/|W02| ~ {c2:.4f} * L^({alpha2:.3f})')
        if alpha2 < -1:
            print(f'  Margin decays FASTER than 1/L -- DANGEROUS')
        elif alpha2 < 0:
            print(f'  Margin decays but slower than 1/L')
        else:
            print(f'  Margin GROWS -- favorable')

    # =============================================================
    # CHECK 3: Non-circular route?
    # =============================================================
    print('\n\n' + '=' * 72)
    print('  CHECK 3: NON-CIRCULAR ROUTE?')
    print('=' * 72)

    # Analyze what makes M more negative than W02
    # M has prime terms, W02 is purely analytic
    # The difference B = W02 - M = -(|M| - |W02|)... wait:
    # W02 < 0, M < 0, |M| > |W02|, so M < W02 < 0
    # B = W02 - M > 0 because subtracting a more-negative number

    for r in results:
        lam_sq = r['lam_sq']
        W02 = r['W02']
        M = r['M']
        w_hat = r['w_hat']
        N = r['N']

        # Decompose M further
        # M_prime has the prime contributions: M[m,n] ~ sum_p log(p)/p^{|m-n|/2}
        # Extract the prime matrix structure
        M_diag = np.diag(np.diag(M))
        M_off = M - M_diag

        # What is W02? It encodes the archimedean (Gamma function) contribution
        W02_diag = np.diag(np.diag(W02))
        W02_off = W02 - W02_diag

        w02_d = float(w_hat @ W02_diag @ w_hat)
        w02_o = float(w_hat @ W02_off @ w_hat)
        m_d = float(w_hat @ M_diag @ w_hat)
        m_o = float(w_hat @ M_off @ w_hat)

        print(f'\n  lam^2 = {lam_sq}:')
        print(f'    W02_diag = {w02_d:.6f}    W02_off = {w02_o:.6f}')
        print(f'    M_diag   = {m_d:.6f}    M_off   = {m_o:.6f}')
        print(f'    diag_diff = {w02_d - m_d:.6f}  (W02_diag - M_diag)')
        print(f'    off_diff  = {w02_o - m_o:.6f}  (W02_off - M_off)')
        print(f'    B(L)      = {r["barrier"]:.6f}')

        # Which component drives positivity?
        diag_part = w02_d - m_d
        off_part = w02_o - m_o
        print(f'    Diag contributes: {diag_part:.6f} ({diag_part/r["barrier"]*100:.1f}% of B)')
        print(f'    Off-diag contributes: {off_part:.6f} ({off_part/r["barrier"]*100:.1f}% of B)')

    print('\n\n  -- CIRCULARITY ASSESSMENT --')
    print("""
  W02 encodes: Gamma function, conductor, archimedean local factors
    -> Built from analytic number theory, NO primes
    -> Computable without knowing zero locations

  M encodes: log(p) * p^{-|m-n|/2} summed over primes
    -> Built from the prime distribution
    -> Computable without knowing zero locations

  B = W02 - M:
    -> Both sides computable without RH
    -> The IDENTITY B(L) = W02 - M is unconditional
    -> But PROVING B > 0 requires showing |M| > |W02|

  The question: does |M| > |W02| follow from PNT alone?
  Or does it require deeper information about prime distribution
  that is equivalent to RH?

  Key test: replace primes with Cramer random model.
  If random primes also give |M| > |W02|, then PNT suffices.
  If random primes give |M| < |W02|, then the actual prime
  distribution is essential -- and we're back to RH territory.
  """)

    # Quick Cramer test at lam^2 = 50
    print('  -- CRAMER MODEL TEST --')
    lam_sq = 50
    L_f = np.log(lam_sq)
    N = max(15, round(6 * L_f))
    dim = 2 * N + 1

    # Real barrier
    W02_real, M_real, QW_real = build_all(lam_sq, N)
    ns = np.arange(-N, N + 1, dtype=float)
    w = ns / (L_f**2 + (4*np.pi)**2 * ns**2)
    w[N] = 0.0
    w_hat = w / np.linalg.norm(w)
    b_real = float(w_hat @ QW_real @ w_hat)

    # Cramer model: replace primes with random integers having density 1/log(n)
    np.random.seed(42)
    n_trials = 20
    cramer_barriers = []
    for trial in range(n_trials):
        # Generate Cramer random primes up to some limit
        limit = 10000
        is_prime = np.random.random(limit) < 1.0 / np.maximum(np.log(np.arange(2, limit + 2)), 1)
        cramer_primes = np.where(is_prime)[0] + 2

        # Build M with Cramer primes instead of real primes
        M_cramer = np.zeros((dim, dim))
        for p in cramer_primes:
            if p < 2:
                continue
            log_p = np.log(p)
            for m_idx in range(dim):
                for n_idx in range(dim):
                    m_val = m_idx - N
                    n_val = n_idx - N
                    diff = abs(m_val - n_val)
                    if diff > 0:
                        M_cramer[m_idx, n_idx] += log_p / (p ** (diff / 2.0))

        QW_cramer = W02_real - M_cramer
        b_cramer = float(w_hat @ QW_cramer @ w_hat)
        cramer_barriers.append(b_cramer)

    cramer_mean = np.mean(cramer_barriers)
    cramer_std = np.std(cramer_barriers)
    cramer_neg = np.sum(np.array(cramer_barriers) < 0)

    print(f'\n  Real primes: B(L) = {b_real:.6f}')
    print(f'  Cramer model ({n_trials} trials):')
    print(f'    Mean B = {cramer_mean:.6f}')
    print(f'    Std  B = {cramer_std:.6f}')
    print(f'    Negative: {cramer_neg}/{n_trials} trials')
    print(f'    Range: [{min(cramer_barriers):.6f}, {max(cramer_barriers):.6f}]')

    if cramer_neg > 0:
        print(f'\n  *** CRAMER PRIMES GIVE NEGATIVE BARRIER ***')
        print(f'  The actual prime distribution is ESSENTIAL.')
        print(f'  PNT alone does NOT suffice -- this is RH territory.')
    else:
        print(f'\n  Cramer primes also give positive barrier.')
        print(f'  PNT-level randomness may suffice.')

    print()
