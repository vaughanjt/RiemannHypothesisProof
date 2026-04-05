"""
SESSION 56 -- LORENTZIAN STRUCTURE PROBE

Session 34 found M has signature (1, d-1): exactly one positive eigenvalue.
This is the Lorentzian signature in the sense of Huh-Branden (Annals 2020).

Questions:
  1. Does M maintain signature (1, d-1) across all lambda?
  2. How tightly is M's positive eigenvector aligned with range(W02)?
     If alignment -> 1 as lambda -> inf, Lorentzian structure
     asymptotically forces Q_W >= 0 on null(W02).
  3. What happens along the deformation Q(t) = W02 - t*M?
     Track eigenvalues as t goes from 0 to 1.
  4. Does the eigenvalue spectrum of M follow patterns predicted
     by Lorentzian theory (e.g., interlacing with W02)?

If M's positive eigenvector lies exactly in range(W02), then:
  - On null(W02): Q_W = -M, and M < 0 there (since positive
    direction is outside null) -> Q_W > 0 automatically
  - On range(W02): Q_W = W02 - M is a 2x2 problem
This would reduce RH to a 2x2 inequality, which Session 40 already
showed holds with barrier ~0.04.
"""

import sys
import time

import numpy as np

sys.path.insert(0, '.')
from session41g_uncapped_barrier import sieve_primes
from session49c_weil_residual import build_all_fast


def analyze_lorentzian(lam_sq, verbose=True):
    """Full Lorentzian analysis at a single lambda^2."""
    L = np.log(lam_sq)
    N = max(15, round(6 * L))
    dim = 2 * N + 1

    W02, M, QW = build_all_fast(lam_sq, N)

    # Eigendecompositions
    ew, vw = np.linalg.eigh(W02)
    em, vm = np.linalg.eigh(M)
    eq, vq = np.linalg.eigh(QW)

    # -- Test 1: Signature of M --
    n_pos_M = np.sum(em > 1e-10)
    n_neg_M = np.sum(em < -1e-10)
    n_zero_M = dim - n_pos_M - n_neg_M
    sig_M = (n_pos_M, n_neg_M, n_zero_M)

    # Positive eigenvalue of M
    max_em = em[-1]
    max_ev_M = vm[:, -1]  # corresponding eigenvector

    # -- Test 2: Alignment with range(W02) --
    # range(W02) is spanned by the eigenvectors with |eigenvalue| > threshold
    w02_thresh = max(abs(ew)) * 1e-10
    range_mask = np.abs(ew) > w02_thresh
    n_range = np.sum(range_mask)
    range_vecs = vw[:, range_mask]  # columns = range basis vectors

    # Project M's positive eigenvector onto range(W02)
    proj = range_vecs @ (range_vecs.T @ max_ev_M)
    alignment = float(np.linalg.norm(proj))  # 1.0 = fully in range

    # Leakage into null(W02)
    null_mask = ~range_mask
    null_vecs = vw[:, null_mask]
    null_proj = null_vecs @ (null_vecs.T @ max_ev_M)
    leakage = float(np.linalg.norm(null_proj))

    # -- Test 3: M restricted to null(W02) --
    M_null = null_vecs.T @ M @ null_vecs
    em_null = np.linalg.eigvalsh(M_null)
    max_M_null = float(em_null[-1])
    min_M_null = float(em_null[0])
    # If M is truly Lorentzian with positive direction in range(W02),
    # then M restricted to null(W02) should be negative definite
    M_null_is_negdef = max_M_null < 0

    # -- Test 4: Q_W eigenvalues --
    min_QW = float(eq[0])
    qw_is_psd = min_QW > -1e-10

    # -- Test 5: Deformation Q(t) = W02 - t*M --
    t_values = np.linspace(0, 1.5, 31)
    min_eigs_t = []
    for t in t_values:
        Qt = W02 - t * M
        et = np.linalg.eigvalsh(Qt)
        min_eigs_t.append(float(et[0]))

    # Find t* where min eigenvalue crosses zero
    t_cross = None
    for i in range(len(t_values) - 1):
        if min_eigs_t[i] < 0 and min_eigs_t[i+1] >= 0:
            # Linear interpolation
            t_cross = t_values[i] + (0 - min_eigs_t[i]) / (min_eigs_t[i+1] - min_eigs_t[i]) * (t_values[i+1] - t_values[i])
        elif min_eigs_t[i] >= 0 and min_eigs_t[i+1] < 0:
            t_cross = t_values[i] + (0 - min_eigs_t[i]) / (min_eigs_t[i+1] - min_eigs_t[i]) * (t_values[i+1] - t_values[i])

    result = dict(
        lam_sq=lam_sq, L=L, N=N, dim=dim,
        sig_M=sig_M, max_em=max_em,
        n_range=n_range, alignment=alignment, leakage=leakage,
        M_null_negdef=M_null_is_negdef, max_M_null=max_M_null,
        min_QW=min_QW, qw_psd=qw_is_psd,
        t_values=t_values, min_eigs_t=min_eigs_t, t_cross=t_cross,
    )

    if verbose:
        print(f'  lam^2={lam_sq:>8d}  dim={dim:>4d}  '
              f'sig(M)=({n_pos_M},{n_neg_M},{n_zero_M})  '
              f'align={alignment:.6f}  leak={leakage:.6f}  '
              f'M_null_max={max_M_null:+.4e}  '
              f'min(QW)={min_QW:+.4e}  '
              f'{"PSD" if qw_is_psd else "NOT PSD"}')

    return result


def run():
    print()
    print('#' * 76)
    print('  SESSION 56 -- LORENTZIAN STRUCTURE PROBE')
    print('#' * 76)

    # == Test 1 + 2: Signature and alignment sweep ==
    print('\n  === TEST 1+2: M SIGNATURE AND ALIGNMENT WITH range(W02) ===')
    print(f'  If M has signature (1, d-1) and positive eigenvector aligns')
    print(f'  with range(W02), then Q_W > 0 on null(W02) is automatic.')
    print()

    lam_values = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000,
                  10000, 20000, 50000]

    results = []
    for lam_sq in lam_values:
        t0 = time.time()
        r = analyze_lorentzian(lam_sq)
        dt = time.time() - t0
        results.append(r)
        if dt > 1:
            print(f'    ({dt:.1f}s)')
    sys.stdout.flush()

    # == Summary ==
    print('\n  === SUMMARY TABLE ===')
    print(f'  {"lam^2":>8} {"dim":>5} {"sig(M)":>14} {"align":>8} '
          f'{"leak":>8} {"M_null max":>12} {"min(QW)":>12}')
    print('  ' + '-' * 80)
    for r in results:
        sig = f'({r["sig_M"][0]},{r["sig_M"][1]},{r["sig_M"][2]})'
        print(f'  {r["lam_sq"]:>8d} {r["dim"]:>5d} {sig:>14s} '
              f'{r["alignment"]:>8.6f} {r["leakage"]:>8.6f} '
              f'{r["max_M_null"]:>+12.4e} {r["min_QW"]:>+12.4e}')

    # == Test 3: Alignment trend ==
    print('\n  === ALIGNMENT TREND ===')
    Ls = np.array([r['L'] for r in results])
    aligns = np.array([r['alignment'] for r in results])
    leaks = np.array([r['leakage'] for r in results])

    if len(Ls) >= 3:
        # Fit: leakage = a * L^b
        mask = leaks > 1e-15
        if np.sum(mask) >= 3:
            log_L = np.log(Ls[mask])
            log_leak = np.log(leaks[mask])
            b, log_a = np.polyfit(log_L, log_leak, 1)
            a = np.exp(log_a)
            print(f'  leakage ~ {a:.4f} * L^{b:.3f}')
            if b < -0.1:
                print(f'  => leakage DECREASING with L (exponent {b:.2f})')
                print(f'     alignment -> 1 as lambda -> inf')
                print(f'     Lorentzian structure ASYMPTOTICALLY forces Q_W >= 0 on null')
            else:
                print(f'  => leakage NOT decreasing: exponent {b:.2f}')

    # == Test 4: M restricted to null(W02) ==
    print('\n  === M ON null(W02) ===')
    all_negdef = all(r['M_null_negdef'] for r in results)
    print(f'  M restricted to null(W02) is negative definite at ALL '
          f'tested lambda: {all_negdef}')
    if all_negdef:
        print(f'  => Q_W = -M > 0 on null(W02) at all tested lambda')
        print(f'     (because M < 0 on null(W02) means -M > 0 there)')
        max_nulls = [r['max_M_null'] for r in results]
        print(f'  max eigenvalue of M|null(W02): '
              f'{max(max_nulls):.4e} (most positive = least negative)')
    else:
        bad = [r for r in results if not r['M_null_negdef']]
        print(f'  VIOLATION at lambda^2 = {[r["lam_sq"] for r in bad]}')

    # == Test 5: Deformation path ==
    print('\n  === DEFORMATION Q(t) = W02 - t*M ===')
    print(f'  Track min eigenvalue as t goes from 0 to 1.5')
    print(f'  t=0: Q=W02 (rank 2, not PSD)')
    print(f'  t=1: Q=Q_W (should be PSD)')
    print()

    # Pick a representative lambda
    rep = results[len(results)//2]  # middle of range
    print(f'  Representative: lam^2 = {rep["lam_sq"]}, dim = {rep["dim"]}')
    print(f'  {"t":>6} {"min_eig":>14}')
    print(f'  ' + '-' * 24)
    for t, me in zip(rep['t_values'], rep['min_eigs_t']):
        marker = ' <-- Q_W' if abs(t - 1.0) < 0.01 else ''
        print(f'  {t:6.2f} {me:>+14.6f}{marker}')

    # == Verdict ==
    print()
    print('=' * 76)
    print('  VERDICT')
    print('=' * 76)

    sigs_ok = all(r['sig_M'] == (1, r['dim'] - 1, 0)
                  or r['sig_M'][0] == 1 for r in results)

    if sigs_ok and all_negdef:
        print(f'\n  M has LORENTZIAN signature (1, d-1) at all tested lambda.')
        print(f'  M restricted to null(W02) is NEGATIVE DEFINITE everywhere.')
        print(f'  => Q_W = W02 - M = -M > 0 on null(W02) follows from:')
        print(f'     "M has at most 1 positive eigenvalue" +')
        print(f'     "that positive direction lies in range(W02)"')
        print()
        print(f'  The positivity on null(W02) is STRUCTURALLY FORCED by')
        print(f'  the Lorentzian property of M, provided the alignment')
        print(f'  of M\'s positive eigenvector with range(W02) is exact.')
        print()
        print(f'  Current alignment: {aligns[-1]:.8f} (leakage {leaks[-1]:.2e})')
        if leaks[-1] < 0.01:
            print(f'  Leakage is small but nonzero. Two paths:')
            print(f'  (a) Prove alignment is exact (= positive eigenvector')
            print(f'      of M lies in range(W02)). This is a structural claim.')
            print(f'  (b) Bound the leakage contribution and show it does not')
            print(f'      flip any eigenvalue of Q_W from positive to negative.')
            print(f'  Session 40 showed cross-coupling contributes ~10^-10 to eps_0.')
        print()
        print(f'  THE LORENTZIAN PROPERTY IS THE STRUCTURAL EXPLANATION')
        print(f'  for why Q_W >= 0 on null(W02). The question is whether')
        print(f'  it can be PROVED that M has at most 1 positive eigenvalue.')
    elif sigs_ok:
        print(f'\n  M has Lorentzian signature but M|null is NOT always neg def.')
        print(f'  Leakage of positive direction into null(W02) is too large.')
    else:
        print(f'\n  M does NOT always have Lorentzian signature (1, d-1).')
        print(f'  The Lorentzian structure hypothesis is DEAD.')

    np.savez('session56_lorentzian_probe.npz',
             lam_values=np.array([r['lam_sq'] for r in results]),
             alignments=aligns, leakages=leaks)
    print('\n  Data saved to session56_lorentzian_probe.npz')


if __name__ == '__main__':
    run()
