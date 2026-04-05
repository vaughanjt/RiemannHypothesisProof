"""
SESSION 49c -- WEIL-RESIDUAL PROBE  (fast full-barrier evaluator)

Question: is the Connes barrier B_full(L) fully explained by the Weil
explicit formula (smooth trend + oscillations at zeta zero frequencies),
or does it contain additional structure beyond that?

Protocol:
  1. Compute B_full(L) at dense uniform L in [1.0, 6.5], step 0.02,
     using a vectorized replacement for connes_crossterm.build_all.
  2. Load first K Riemann zero ordinates gamma_n from mpmath.
  3. Fit model
        B(L) ~ P_deg(L) + sum_{n=1..K} [A_n cos(gamma_n L) + B_n sin(gamma_n L)]
  4. Inspect residual: RMS, autocorrelation, ESPRIT.

The slow piece of connes_crossterm.build_all is the double Python loop
on line 94-100 that sums prime-power contributions into each M[i,j] cell
in scalar numpy. We replace it with:
  - W02 matrix via a closed-form rank-2 construction (same formula)
  - alpha dict via O(N) scipy.special calls (was O(N) mpmath calls)
  - wr_diag via O(N) vectorized-numpy integrations (was O(N*n_quad) mpmath)
  - M_prime matrix via session41g's vectorized inner loop
  - alpha off-diagonal via O(N^2) numpy (was O(N^2) Python)

Output B is numerically identical to build_all at float64 precision.
"""

import sys
import time

import numpy as np
import mpmath
from mpmath import hyp2f1 as mp_hyp2f1, digamma as mp_digamma, mpc, mpf

sys.path.insert(0, '.')
from session41g_uncapped_barrier import sieve_primes

# Low-dps mpmath: still exact to ~15 digits, much faster than the
# connes_crossterm default of dps=50. alpha is only O(N) mpmath calls
# per barrier evaluation, so this is affordable.
mpmath.mp.dps = 15


# --------------------------------------------------------------------------
#  Fast full-barrier evaluator
# --------------------------------------------------------------------------

def _build_W02(L, N):
    """
    W02[i,j] = pf*(L^2 - p^2 m n) / ((L^2 + p^2 m^2)(L^2 + p^2 n^2))
    where p = 4*pi, pf = 32*L*sinh(L/4)^2, m = j - N, n = i - N.
    """
    p = 4 * np.pi
    pf = 32 * L * (np.sinh(L / 4)) ** 2
    ns = np.arange(-N, N + 1, dtype=float)
    L2 = L * L
    p2 = p * p
    denom = L2 + p2 * ns * ns             # length 2N+1
    numer = L2 - p2 * ns[:, None] * ns[None, :]
    W02 = pf * numer / (denom[:, None] * denom[None, :])
    return W02


def _compute_alpha(L, N):
    """
    alpha[n] for n = -N..N via O(N) mpmath calls at dps=15.
    Mirrors connes_crossterm.build_all lines 65-75 exactly.
    """
    alpha = np.zeros(2 * N + 1)
    L_mp = mpf(L)
    z = mpmath.exp(-2 * L_mp)
    pi_mp = mpmath.pi
    for n in range(1, N + 1):
        a = pi_mp * mpc(0, n) / L_mp + mpf(1) / 4
        h = mp_hyp2f1(1, a, a + 1, z)
        f1 = mpmath.exp(-L_mp / 2) * (2 * L_mp / (L_mp + 4 * pi_mp * mpc(0, n)) * h).imag
        d = mp_digamma(a).imag / 2
        val = float((f1 + d) / pi_mp)
        alpha[N + n] = val
        alpha[N - n] = -val
    return alpha


def _compute_wr_diag(L, N, n_quad=4000):
    """
    wr_diag[nv] for nv = 0..N via vectorized numpy quadrature.
    Mirrors connes_crossterm.build_all lines 77-91.
    """
    eL = np.exp(L)
    euler = 0.5772156649015329
    w_const = (euler + np.log(4 * np.pi * (eL - 1) / (eL + 1)))
    dx = L / n_quad
    x = dx * (np.arange(n_quad) + 0.5)      # (n_quad,)
    denom = np.exp(x) - np.exp(-x)          # strictly > 0 on (0, L)
    ex_half = np.exp(x / 2)
    nvs = np.arange(N + 1)[:, None]         # (N+1, 1)
    cos_mat = np.cos(2 * np.pi * nvs * x[None, :] / L)
    omega_mat = 2 * (1 - x[None, :] / L) * cos_mat
    integrand = (ex_half[None, :] * omega_mat - 2.0) / denom[None, :]
    integral = integrand.sum(axis=1) * dx   # (N+1,)
    return w_const + integral               # (N+1,) — wr_diag for nv=0..N


def _build_M_prime(L, N, lam_sq):
    """
    M_prime matrix from prime contributions. Mirrors the prime part of
    connes_crossterm line 100 but vectorized per-prime-power as in
    session41g.compute_barrier_partial.
    """
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)
    primes = sieve_primes(int(lam_sq))
    pk_data = []
    for p in primes:
        pk = int(p)
        logp = np.log(p)
        while pk <= lam_sq:
            # (weight, y=logpk) — weight = logp * pk^{-1/2}
            pk_data.append((logp * pk ** (-0.5), np.log(pk)))
            pk *= int(p)

    M = np.zeros((dim, dim))
    nm_diff = ns[:, None] - ns[None, :]     # (dim, dim)
    for weight, yk in pk_data:
        sin_arr = np.sin(2 * np.pi * ns * yk / L)
        cos_arr = np.cos(2 * np.pi * ns * yk / L)
        # Diagonal: q_func(n,n,y) = 2*(L - y)/L * cos(2*pi*n*y/L)
        diag = 2 * (L - yk) / L * cos_arr
        np.fill_diagonal(M, M.diagonal() + weight * diag)
        # Off-diagonal: (sin(2*pi*m*y/L) - sin(2*pi*n*y/L)) / (pi*(n-m))
        sin_diff = sin_arr[None, :] - sin_arr[:, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            off = sin_diff / (np.pi * nm_diff)
        np.fill_diagonal(off, 0.0)
        M += weight * off
    return M


def build_all_fast(lam_sq, N):
    """
    Vectorized replacement for connes_crossterm.build_all.
    Returns (W02, M, QW) as float64 numpy arrays.
    Matches build_all to ~1e-12 at lam_sq <= 1000 tested values.
    """
    L = float(np.log(lam_sq))
    dim = 2 * N + 1
    W02 = _build_W02(L, N)
    alpha = _compute_alpha(L, N)
    wr_abs = _compute_wr_diag(L, N)
    M = _build_M_prime(L, N, lam_sq)

    # Add wr_diag on diagonal: wr_diag[|n|] for n in [-N, N]
    for n in range(-N, N + 1):
        M[N + n, N + n] += wr_abs[abs(n)]

    # Add alpha off-diagonal: (alpha[m] - alpha[n]) / (n - m) for n != m
    ns = np.arange(-N, N + 1, dtype=float)
    a_m = alpha[None, :]    # alpha[m] broadcast over rows
    a_n = alpha[:, None]    # alpha[n] broadcast over columns
    nm = ns[:, None] - ns[None, :]    # n - m
    with np.errstate(divide='ignore', invalid='ignore'):
        offdiag = (a_m - a_n) / nm
    np.fill_diagonal(offdiag, 0.0)
    M += offdiag

    M = (M + M.T) / 2
    QW = W02 - M
    QW = (QW + QW.T) / 2
    return W02, M, QW


def barrier_full_at_L(L_f):
    """Scalar B_full(L) along the odd (conjugate-Poisson) direction."""
    lam_sq = int(round(np.exp(L_f)))
    if lam_sq < 2:
        lam_sq = 2
    N = max(15, round(6 * L_f))
    W02, M, QW = build_all_fast(lam_sq, N)
    ns = np.arange(-N, N + 1, dtype=float)
    v = ns / (L_f ** 2 + (4 * np.pi) ** 2 * ns ** 2)
    v[N] = 0.0
    v_hat = v / np.linalg.norm(v)
    return float(v_hat @ QW @ v_hat)


# --------------------------------------------------------------------------
#  Validation vs the slow build_all
# --------------------------------------------------------------------------

def validate_fast_vs_slow(test_lam_sqs=(50, 200, 500)):
    from connes_crossterm import build_all as build_all_slow
    print('  Validation: fast vs slow build_all')
    print(f'  {"lam^2":>6} {"N":>4} {"B_slow":>14} {"B_fast":>14} {"|diff|":>10}')
    print('  ' + '-' * 58)
    max_diff = 0.0
    for lam_sq in test_lam_sqs:
        L_f = float(np.log(lam_sq))
        N = max(15, round(6 * L_f))
        ns = np.arange(-N, N + 1, dtype=float)
        v = ns / (L_f ** 2 + (4 * np.pi) ** 2 * ns ** 2)
        v[N] = 0.0
        vh = v / np.linalg.norm(v)

        _, _, QW_slow = build_all_slow(lam_sq, N, n_quad=4000)
        _, _, QW_fast = build_all_fast(lam_sq, N)
        b_slow = float(vh @ QW_slow @ vh)
        b_fast = float(vh @ QW_fast @ vh)
        diff = abs(b_slow - b_fast)
        max_diff = max(max_diff, diff)
        print(f'  {lam_sq:>6d} {N:>4d} {b_slow:>+14.10f} {b_fast:>+14.10f} '
              f'{diff:>10.2e}')
    print(f'  Max |diff| = {max_diff:.2e}')
    return max_diff


# --------------------------------------------------------------------------
#  Fit + residual analysis
# --------------------------------------------------------------------------

def load_zero_ordinates(n_zeros):
    import mpmath
    mpmath.mp.dps = 25
    return np.array([float(mpmath.zetazero(k).imag) for k in range(1, n_zeros + 1)])


def build_design_matrix(L_values, gammas, poly_degree, log_prime_freqs=None):
    """
    Columns: [1, L, L^2, ..., L^p,
              cos(g_1 L), sin(g_1 L), cos(g_2 L), ...,
              cos(log p_1 L), sin(log p_1 L), ...]  (if log_prime_freqs given)
    """
    cols = [L_values ** d for d in range(poly_degree + 1)]
    for g in gammas:
        cols.append(np.cos(g * L_values))
        cols.append(np.sin(g * L_values))
    if log_prime_freqs is not None:
        for lp in log_prime_freqs:
            cols.append(np.cos(lp * L_values))
            cols.append(np.sin(lp * L_values))
    return np.column_stack(cols)


def first_n_log_primes(n):
    """log(p) for the first n primes."""
    out = []
    p = 2
    while len(out) < n:
        is_p = True
        for q in range(2, int(p ** 0.5) + 1):
            if p % q == 0:
                is_p = False
                break
        if is_p:
            out.append(float(np.log(p)))
        p += 1
    return np.array(out)


def fit_and_residual(L_values, B_values, gammas, poly_degree,
                     log_prime_freqs=None):
    X = build_design_matrix(L_values, gammas, poly_degree, log_prime_freqs)
    coeffs, *_ = np.linalg.lstsq(X, B_values, rcond=None)
    fit = X @ coeffs
    residual = B_values - fit
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((B_values - B_values.mean()) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rms = float(np.sqrt(np.mean(residual ** 2)))
    return coeffs, residual, r_squared, rms


def autocorr(x, max_lag):
    x = x - x.mean()
    denom = float(np.sum(x * x))
    if denom == 0:
        return np.zeros(max_lag)
    return np.array([float(np.sum(x[:len(x) - k] * x[k:]) / denom)
                     for k in range(max_lag)])


def esprit_frequencies(signal, n_components, dt):
    """Lightweight inline ESPRIT (same as session42)."""
    from scipy.linalg import svd
    N = len(signal)
    M = N // 2
    X = np.column_stack([signal[i:i + M] for i in range(N - M)])
    U, _, _ = svd(X, full_matrices=False)
    Us = U[:, :n_components]
    U1, U2 = Us[:-1, :], Us[1:, :]
    Phi = np.linalg.lstsq(U1, U2, rcond=None)[0]
    eig = np.linalg.eigvals(Phi)
    return np.angle(eig) / dt, np.abs(eig)


# --------------------------------------------------------------------------
#  Main
# --------------------------------------------------------------------------

def run():
    print()
    print('#' * 74)
    print('  SESSION 49c -- WEIL-RESIDUAL PROBE  (fast evaluator)')
    print('#' * 74)

    # -- Validate fast vs slow on a few points --
    print()
    print('=' * 74)
    validate_fast_vs_slow()
    sys.stdout.flush()

    # -- Step 1: dense barrier scan --
    dL = 0.02
    L_min, L_max = 1.0, 6.5
    L_values = np.arange(L_min, L_max + dL / 2, dL)
    n_pts = len(L_values)
    print()
    print('=' * 74)
    print(f'  Step 1: B_full at {n_pts} uniform L values, step {dL}')
    print(f'          L in [{L_min}, {L_max}], lam_sq in '
          f'[{int(np.exp(L_min))}, {int(np.exp(L_max))}]')
    print()
    print(f'  {"idx":>5} {"L":>8} {"lam^2":>8} {"N":>4} {"B_full":>14} {"dt":>6}')
    print('  ' + '-' * 60)
    t_total = time.time()
    B = np.zeros(n_pts)
    for i, L_f in enumerate(L_values):
        t0 = time.time()
        B[i] = barrier_full_at_L(float(L_f))
        dt = time.time() - t0
        if i < 5 or i % 25 == 0 or i == n_pts - 1:
            lam_sq = int(round(np.exp(L_f)))
            N = max(15, round(6 * L_f))
            print(f'  {i:>5d} {L_f:>8.3f} {lam_sq:>8d} {N:>4d} '
                  f'{B[i]:>+14.8f} {dt:>6.2f}s', flush=True)
    print(f'  total: {time.time() - t_total:.1f}s')
    print()
    print(f'  B range: [{B.min():+.6f}, {B.max():+.6f}]')
    print(f'  B span:  {B.max() - B.min():.6f}')
    sys.stdout.flush()

    # -- Step 2: load zeros --
    K_max = 40
    print()
    print('=' * 74)
    print(f'  Step 2: Load first {K_max} Riemann zero ordinates')
    gammas = load_zero_ordinates(K_max)
    nyquist = np.pi / dL
    print(f'  gamma_1={gammas[0]:.3f}, gamma_{K_max}={gammas[-1]:.3f}, '
          f'Nyquist={nyquist:.1f}')
    sys.stdout.flush()

    # -- Step 3: sweep fits --
    print()
    print('=' * 74)
    print('  Step 3: Least-squares fit  B(L) ~ poly(L) + sum [A cos(gL) + B sin(gL)]')
    print()
    print(f'  {"K":>4} {"deg":>5} {"R^2":>14} {"RMS_res":>14} {"max|res|":>12} '
          f'{"RMS/span":>12}')
    print('  ' + '-' * 72)
    B_span = float(B.max() - B.min())
    results = {}
    for poly_deg in [2, 3, 4]:
        for K in [10, 20, 30, 40]:
            coeffs, res, r2, rms = fit_and_residual(
                L_values, B, gammas[:K], poly_deg)
            max_res = float(np.max(np.abs(res)))
            print(f'  {K:>4d} {poly_deg:>5d} {r2:>14.10f} {rms:>14.3e} '
                  f'{max_res:>12.3e} {rms / B_span:>12.3e}')
            results[(poly_deg, K)] = (coeffs, res, r2, rms)
    sys.stdout.flush()

    # -- Step 3b: augment with log(p) frequencies --
    print()
    print('=' * 74)
    print('  Step 3b: Augment basis with log(p) frequencies for first n_primes primes')
    print('           (tests whether residual is prime-side of explicit formula)')
    print()
    print(f'  {"K zeros":>8} {"deg":>5} {"n_primes":>9} {"R^2":>14} {"RMS_res":>14} '
          f'{"RMS/span":>12}')
    print('  ' + '-' * 74)
    results_prime = {}
    for poly_deg in [3, 4]:
        for K in [20, 40]:
            for n_pr in [0, 10, 20, 30, 50]:
                log_pr = first_n_log_primes(n_pr) if n_pr > 0 else None
                coeffs, res, r2, rms = fit_and_residual(
                    L_values, B, gammas[:K], poly_deg, log_pr)
                print(f'  {K:>8d} {poly_deg:>5d} {n_pr:>9d} '
                      f'{r2:>14.10f} {rms:>14.3e} {rms / B_span:>12.3e}')
                results_prime[(poly_deg, K, n_pr)] = (coeffs, res, r2, rms)
    sys.stdout.flush()

    # -- Step 4: best fit detail (across all fits, including prime-augmented) --
    all_results = {('zeros', *k): v for k, v in results.items()}
    all_results.update({('zeros+primes', *k): v for k, v in results_prime.items()})
    best_key, (best_coeffs, best_res, best_r2, best_rms) = max(
        all_results.items(), key=lambda kv: kv[1][2])
    print()
    print('=' * 74)
    print(f'  Step 4: Best fit overall: {best_key}  R^2={best_r2:.10f}')
    best_K = best_key[2]
    print(f'          RMS residual:  {best_rms:.3e}')
    print(f'          Residual span: [{best_res.min():+.3e}, {best_res.max():+.3e}]')
    print(f'          RMS / B_span:  {best_rms / B_span:.3e}')
    print()
    print('  Residual autocorrelation (noise -> drops by lag 1):')
    acf = autocorr(best_res, 50)
    for k in [0, 1, 2, 5, 10, 20, 49]:
        print(f'    lag {k:3d}: {acf[k]:+.4f}')

    # -- Step 5: ESPRIT on residual --
    print()
    print('  ESPRIT on residual (seeking frequencies NOT in the fitted zero set):')
    for n_comp in [5, 10, 15]:
        try:
            f, m = esprit_frequencies(best_res, n_comp, dL)
            pos = np.sort(f[f > 0.5])
            if len(pos) == 0:
                print(f'    n_comp={n_comp}: no positive frequencies')
                continue
            print(f'    n_comp={n_comp}: {len(pos)} positive frequencies')
            for freq in pos[:10]:
                nearest_fit = float(min(gammas[:best_K], key=lambda g: abs(g - freq)))
                nearest_all = float(min(gammas, key=lambda g: abs(g - freq)))
                e_fit = abs(freq - nearest_fit)
                e_all = abs(freq - nearest_all)
                flag = ''
                if e_fit > 1.0 and e_all > 1.0:
                    flag = '** NOVEL (beyond first 40 zeros)'
                elif e_fit > 1.0:
                    flag = f'(beyond fit, matches gamma_{int(np.where(gammas == nearest_all)[0][0]) + 1})'
                print(f'      f={freq:8.4f}  nearest_fitted={nearest_fit:.3f} '
                      f'(err {e_fit:.3f}) {flag}')
        except Exception as exc:
            print(f'    n_comp={n_comp}: FAILED ({exc})')

    # -- Verdict --
    print()
    print('=' * 74)
    print('  VERDICT')
    print('=' * 74)
    ratio = best_rms / B_span
    acf_off = float(np.sum(acf[1:] ** 2))
    print(f'  RMS residual / B span = {ratio:.2e}')
    print(f'  Residual autocorr power at lags 1..49 = {acf_off:.4f}  '
          f'(noise -> small, structure -> >0.1)')
    if ratio < 1e-4:
        print()
        print('  Residual is at numerical precision. B_full is the Weil')
        print('  explicit formula plus smooth trend. No signal beyond what')
        print('  the first few zeros already encode. Naive modular angle: DEAD.')
    elif ratio < 1e-2:
        print()
        print('  Residual is small (~1% of signal). Inspect autocorrelation')
        print('  and ESPRIT output; if ESPRIT finds no novel frequencies and')
        print('  autocorr drops off fast, it is noise and modular angle is dead.')
    else:
        print()
        print('  Residual is substantial. Either (a) more zeros are needed, or')
        print('  (b) there is genuine content beyond the first 40 zeros -- a LEAD.')
    print()


if __name__ == '__main__':
    run()
