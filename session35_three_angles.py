"""
SESSION 35d -- THREE SURVIVING ANGLES

After killing M_diag domination and tropical lift:
1. FUNCTIONAL EQUATION: What does the spectral representation via zeta zeros tell us?
2. ZERO-FREE REGION: Can Vinogradov-Korobov bootstrap Q_W >= 0 for finite lambda?
3. TOEPLITZ SYMBOL: Is M approximately Toeplitz with non-positive symbol?
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, log, pi, exp, cos, sin, euler, digamma, hyp2f1, sinh, zetazero
import time
import json
import sys
sys.path.insert(0, '.')
from connes_crossterm import build_all
from session33_sieve_bypass import compute_M_decomposition

# ============================================================================
# ANGLE 1: FUNCTIONAL EQUATION / SPECTRAL REPRESENTATION
# ============================================================================

def spectral_representation(lam_sq, N=None, n_zeros=100):
    """
    Connes' explicit formula connects <phi, Q_W phi> to a sum over zeta zeros.

    For test function g supported in [-L, L]:
      <g, Q_W g> = sum_rho |g-hat(rho)|^2   (if RH holds, each term >= 0)

    Our test functions are: g(x) = sum_n phi_n * omega_n(x)
    where omega_n(x) = 2(1 - |x|/L) cos(2*pi*n*x/L) for |x| <= L

    The Fourier-Laplace transform: g-hat(s) = integral g(x) e^{(s-1/2)x} dx
    For omega_n at s = 1/2 + it:
      omega_n-hat(t) = integral_0^L 2(1-x/L) cos(2*pi*n*x/L) e^{itx} dx

    Compute this numerically and verify against known zeros.
    """
    if N is None:
        L_val = np.log(lam_sq)
        N = max(15, round(6 * L_val))
    dim = 2 * N + 1

    W02, M, QW = build_all(lam_sq, N)

    # Get null(W02) eigenvectors for test vectors
    ew, ev = np.linalg.eigh(W02)
    thresh = np.max(np.abs(ew)) * 1e-10
    P_null = ev[:, np.abs(ew) <= thresh]

    mp.dps = 30
    L = float(log(mpf(lam_sq)))

    print(f"\nANGLE 1: SPECTRAL REPRESENTATION via zeta zeros")
    print(f"  lam^2={lam_sq}, L={L:.4f}, dim={dim}, n_zeros={n_zeros}")

    # Compute the transform omega_n-hat(t) for each basis function
    # omega_n-hat(t) = integral_0^L 2(1-x/L) cos(2*pi*n*x/L) * exp(i*t*x) dx
    # = integral_0^L (1-x/L) [exp(i*(t+2*pi*n/L)*x) + exp(i*(t-2*pi*n/L)*x)] dx
    #
    # integral_0^L (1-x/L) exp(i*k*x) dx = (1 - exp(i*k*L))/(i*k*L) + exp(i*k*L)/(i*k)
    #                                        ... let me compute numerically

    def omega_hat(n_val, t, L_val):
        """Compute omega_n-hat(t) = integral_0^L 2(1-x/L)cos(2*pi*n*x/L)*e^{itx} dx."""
        # Numerical integration (quadrature)
        n_pts = 2000
        dx = L_val / n_pts
        result = 0.0 + 0.0j
        for k in range(n_pts):
            x = dx * (k + 0.5)
            integrand = 2 * (1 - x / L_val) * np.cos(2 * np.pi * n_val * x / L_val) * np.exp(1j * t * x)
            result += integrand * dx
        return result

    # Get first n_zeros zeta zeros (imaginary parts)
    print(f"  Computing first {n_zeros} zeta zeros...")
    gammas = []
    for j in range(1, n_zeros + 1):
        z = zetazero(j)
        gammas.append(float(z.imag))

    # Test with several null vectors
    ns = np.arange(-N, N + 1)

    print(f"\n  Testing spectral formula for null(W02) vectors:")
    print(f"  {'vec':>4} {'<phi,QW phi>':>14} {'sum |g-hat|^2':>14} {'ratio':>8} {'residual':>12}")

    for v_idx in range(min(5, P_null.shape[1])):
        phi = P_null[:, v_idx]

        # Compute <phi, Q_W phi> directly
        qw_direct = np.dot(phi, QW @ phi)

        # Compute sum over zeros: sum_gamma |g-hat(gamma)|^2
        # g-hat(gamma) = sum_n phi_n * omega_n-hat(gamma)
        spectral_sum = 0.0
        for gamma in gammas:
            g_hat = 0.0 + 0.0j
            for i in range(dim):
                n_val = ns[i]
                g_hat += phi[i] * omega_hat(n_val, gamma, L)
            spectral_sum += abs(g_hat) ** 2

        # The spectral sum should approximate <phi, Q_W phi>
        # (Not exact because we only use finite zeros)
        ratio = spectral_sum / qw_direct if abs(qw_direct) > 1e-15 else float('inf')
        residual = qw_direct - spectral_sum

        print(f"  {v_idx:>4} {qw_direct:>+14.6e} {spectral_sum:>14.6e} {ratio:>8.4f} {residual:>+12.4e}")

    # Per-zero contributions for the MOST NEGATIVE eigenvector
    M_null = P_null.T @ M @ P_null
    e_null, v_null = np.linalg.eigh(M_null)
    # Most negative M eigenvalue = most positive Q_W eigenvalue
    phi_test = P_null @ v_null[:, 0]  # most negative M eigenvector
    qw_test = np.dot(phi_test, QW @ phi_test)

    print(f"\n  Per-zero contributions for most-positive Q_W eigenvector:")
    print(f"  <phi, Q_W phi> = {qw_test:.6f}")
    print(f"  {'j':>4} {'gamma_j':>12} {'|g-hat|^2':>14} {'cumul_sum':>14} {'% of total':>10}")

    cumul = 0.0
    for j, gamma in enumerate(gammas[:30]):
        g_hat = 0.0 + 0.0j
        for i in range(dim):
            g_hat += phi_test[i] * omega_hat(ns[i], gamma, L)
        contrib = abs(g_hat) ** 2
        cumul += contrib
        pct = 100 * cumul / qw_test if abs(qw_test) > 1e-15 else 0
        print(f"  {j+1:>4} {gamma:>12.4f} {contrib:>14.6e} {cumul:>14.6e} {pct:>9.1f}%")

    return gammas


# ============================================================================
# ANGLE 2: ZERO-FREE REGION BOOTSTRAP
# ============================================================================

def zero_free_region_bootstrap(lam_sq, N=None):
    """
    Use the Vinogradov-Korobov zero-free region to bootstrap Q_W >= 0.

    Zero-free region: zeta(s) != 0 for sigma > 1 - c/(log t)^{2/3}(log log t)^{1/3}
    Best known c ~ 1/57.54 (Ford 2002).

    For a hypothetical zero at rho = beta + i*gamma with beta > 1/2:
    The error in <phi, Q_W phi> from this zero is:
      delta = 2 * Re[g-hat(rho) * conj(g-hat(1-rho))] - |g-hat(rho)|^2 - |g-hat(1-rho)|^2
    which depends on (beta - 1/2).

    For band-limited g with support [-L, L]:
    |g-hat(beta + i*gamma)| <= ||g|| * e^{(beta-1/2)*L}
    So the error from off-line zeros is bounded by e^{2*(beta-1/2)*L}.

    With the zero-free region: beta - 1/2 <= 1/2 - c/(log gamma)^{2/3}(log log gamma)^{1/3}
    For gamma <= L (the interesting range): error <= e^{2*(1/2 - c/(log L)^{2/3+})*L}

    This GROWS with L unless beta is very close to 1/2.
    So the zero-free region alone cannot prove Q_W >= 0 for large lambda.
    But for SMALL lambda, it might work.
    """
    if N is None:
        L_val = np.log(lam_sq)
        N = max(15, round(6 * L_val))
    dim = 2 * N + 1
    L = np.log(lam_sq)

    W02, M, QW = build_all(lam_sq, N)

    print(f"\nANGLE 2: ZERO-FREE REGION BOOTSTRAP")
    print(f"  lam^2={lam_sq}, L={L:.4f}")

    # Vinogradov-Korobov zero-free region constant
    c_vk = 1 / 57.54  # Ford 2002

    # For a zero at height gamma, max real part is:
    # beta_max(gamma) = 1 - c_vk / (log(gamma))^{2/3} * (log(log(gamma)))^{1/3}
    # for gamma >= 3

    def beta_max(gamma):
        if gamma < 3:
            return 0.5  # Below height 3, RH is verified
        lt = np.log(gamma)
        llt = np.log(lt) if lt > 1 else 0.1
        return 1 - c_vk / (lt ** (2/3) * llt ** (1/3))

    # For our band-limited test functions, zeros with gamma > C*lambda contribute negligibly
    # The effective cutoff is gamma ~ pi * dim / L (Nyquist)
    gamma_max = np.pi * dim / L
    print(f"  Effective gamma cutoff: {gamma_max:.1f}")
    print(f"  beta_max at gamma_max: {beta_max(gamma_max):.6f}")
    print(f"  Max deviation from 1/2: {beta_max(gamma_max) - 0.5:.6f}")

    # The error from a single off-line zero at (beta, gamma):
    # |error| ~ e^{2*(beta - 1/2)*L} * (something bounded)
    # At the boundary of zero-free region:
    delta_beta = beta_max(gamma_max) - 0.5
    error_bound = np.exp(2 * delta_beta * L)
    print(f"  Error amplification factor: e^{{2*delta*L}} = {error_bound:.4e}")

    # How many zeros below gamma_max?
    n_zeros_est = gamma_max * np.log(gamma_max) / (2 * np.pi)
    print(f"  Estimated zeros below gamma_max: {n_zeros_est:.0f}")

    # Total error from all potential off-line zeros
    # (Very rough: n_zeros * error_per_zero * typical |g-hat|^2)
    # This is a very conservative bound

    print(f"\n  ZERO-FREE REGION QUALITY at different heights:")
    print(f"  {'gamma':>8} {'beta_max':>10} {'delta':>10} {'e^(2*delta*L)':>14}")
    for gamma in [14.13, 21.02, 25.01, 50, 100, 200, 500, 1000, gamma_max]:
        bm = beta_max(gamma)
        delta = bm - 0.5
        amp = np.exp(2 * delta * L)
        print(f"  {gamma:>8.2f} {bm:>10.6f} {delta:>10.6f} {amp:>14.4e}")

    # KEY QUESTION: is the margin large enough?
    # <phi, Q_W phi> on null(W02) is ~ dim * |average eigenvalue|
    # The smallest positive eigenvalue of Q_W is the max negative eigenvalue of M on null
    # which is ~1e-8 (very small!)
    evals_M_null_approx = np.linalg.eigvalsh(
        ev[:, np.abs(ew) <= thresh].T @ M @ ev[:, np.abs(ew) <= thresh]
    ) if False else None

    ew_w, ev_w = np.linalg.eigh(W02)
    thresh_w = np.max(np.abs(ew_w)) * 1e-10
    P_null = ev_w[:, np.abs(ew_w) <= thresh_w]
    M_null = P_null.T @ M @ P_null
    evals_null = np.linalg.eigvalsh(M_null)
    margin = -np.max(evals_null)

    print(f"\n  Margin: min eigenvalue gap of Q_W on null = {margin:.6e}")
    print(f"  Error bound at boundary: {error_bound:.4e}")
    if margin > error_bound:
        print(f"  *** MARGIN EXCEEDS ERROR: zero-free region SUFFICIENT ***")
    else:
        print(f"  *** MARGIN < ERROR: zero-free region INSUFFICIENT ***")
        print(f"  Need margin/error = {margin/error_bound:.2e}")
        print(f"  Would need delta < {np.log(margin)/(2*L):.6f} (vs actual {delta_beta:.6f})")
        needed_zfr = -np.log(margin) / (2 * L) + 0.5
        print(f"  Would need beta < {0.5 + np.log(margin)/(2*L):.6f}")
        print(f"  Equivalently: zeros within {0.5 - (0.5 + np.log(margin)/(2*L)):.6f} of critical line")


# ============================================================================
# ANGLE 3: TOEPLITZ SYMBOL
# ============================================================================

def toeplitz_symbol(lam_sq, N=None):
    """
    Analyze the Toeplitz structure of M on null(W02).

    A Toeplitz matrix T has entries T[i,j] = t(i-j).
    The symbol is f(theta) = sum_k t(k) e^{ik*theta}.
    Eigenvalues approximate f(2*pi*j/dim).

    M_diag is diagonal (= Toeplitz with only t(0)).
    M_alpha has entries (alpha[m]-alpha[n])/(n-m) which depends on both n and m.
    M_prime has entries that are oscillatory in both n and m.

    For M restricted to null(W02), the basis change complicates things.
    Instead, work in the ORIGINAL Fourier basis and check near-Toeplitz structure.

    KEY: For a Toeplitz matrix, all anti-diagonals are constant.
    Measure how constant the anti-diagonals of M are.
    """
    if N is None:
        L_val = np.log(lam_sq)
        N = max(15, round(6 * L_val))
    dim = 2 * N + 1
    L_f = np.log(lam_sq)

    W02, M, QW = build_all(lam_sq, N)
    M_diag, M_alpha, M_prime, M_full, primes = compute_M_decomposition(lam_sq, N)

    print(f"\nANGLE 3: TOEPLITZ SYMBOL ANALYSIS")
    print(f"  lam^2={lam_sq}, dim={dim}")

    # Extract the Toeplitz part of each component
    # T_toeplitz[i,j] = average of M[i+k, j+k] over valid k
    def extract_toeplitz(A):
        """Extract the Toeplitz part: average along each diagonal."""
        d = A.shape[0]
        t = np.zeros(2 * d - 1)
        counts = np.zeros(2 * d - 1)
        for i in range(d):
            for j in range(d):
                k = i - j + d - 1
                t[k] += A[i, j]
                counts[k] += 1
        t = t / np.maximum(counts, 1)
        return t

    def toeplitz_from_seq(t, d):
        """Build Toeplitz matrix from sequence."""
        T = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                T[i, j] = t[i - j + d - 1]
        return T

    # Toeplitz decomposition of each component
    for name, A in [("M_diag", M_diag), ("M_alpha", M_alpha),
                    ("M_prime", M_prime), ("M_full", M)]:
        t = extract_toeplitz(A)
        T = toeplitz_from_seq(t, dim)
        residual = A - T
        frob_A = np.linalg.norm(A, 'fro')
        frob_R = np.linalg.norm(residual, 'fro')
        pct = frob_R / frob_A * 100 if frob_A > 1e-15 else 0

        print(f"\n  {name}:")
        print(f"    ||A||_F = {frob_A:.4f}")
        print(f"    ||A - Toeplitz(A)||_F = {frob_R:.4f} ({pct:.1f}% non-Toeplitz)")

    # Compute the symbol of M's Toeplitz part
    t_M = extract_toeplitz(M)
    # Symbol: f(theta) = sum_k t_k e^{ik*theta}
    # where t_k is the k-th diagonal value
    # t[d-1] = main diagonal, t[d-1+k] = k-th super-diagonal, t[d-1-k] = k-th sub-diagonal

    # Since M is symmetric: t[d-1+k] = t[d-1-k]
    # Symbol is real: f(theta) = t_0 + 2*sum_{k>0} t_k cos(k*theta)

    t_0 = t_M[dim - 1]  # main diagonal average
    theta_vals = np.linspace(-np.pi, np.pi, 500)
    symbol = np.zeros(len(theta_vals))

    for i, theta in enumerate(theta_vals):
        s = t_0
        for k in range(1, dim):
            s += 2 * t_M[dim - 1 + k] * np.cos(k * theta)
        symbol[i] = s

    max_symbol = np.max(symbol)
    min_symbol = np.min(symbol)

    print(f"\n  TOEPLITZ SYMBOL of M:")
    print(f"    f(theta) range: [{min_symbol:+.6f}, {max_symbol:+.6f}]")
    print(f"    f(0) = {t_0 + 2*sum(t_M[dim:]):.6f}")
    print(f"    Symbol non-positive: {max_symbol < 1e-10}")

    if max_symbol > 1e-10:
        # Where is the symbol positive?
        pos_thetas = theta_vals[symbol > 1e-10]
        print(f"    Symbol positive for theta in [{pos_thetas[0]:.4f}, {pos_thetas[-1]:.4f}]")
        print(f"    Max positive value: {max_symbol:.6f} at theta={theta_vals[np.argmax(symbol)]:.4f}")

    # Compare actual eigenvalues of M with symbol values
    evals_M = np.sort(np.linalg.eigvalsh(M))
    symbol_sorted = np.sort(symbol)
    # Resample to same size
    symbol_sampled = np.sort(np.array([
        t_0 + 2 * sum(t_M[dim - 1 + k] * np.cos(k * 2 * np.pi * j / dim) for k in range(1, dim))
        for j in range(dim)
    ]))

    print(f"\n  Eigenvalue vs Symbol comparison:")
    print(f"    Actual M eigenvalues:  [{evals_M[0]:+.4f}, {evals_M[-1]:+.4f}]")
    print(f"    Symbol sample values:  [{symbol_sampled[0]:+.4f}, {symbol_sampled[-1]:+.4f}]")
    corr = np.corrcoef(evals_M, symbol_sampled)[0, 1]
    print(f"    Correlation: {corr:.6f}")

    # Now check: M restricted to null(W02)
    # The null(W02) constraint eliminates 2 directions.
    # After projection, is the RESTRICTED matrix more Toeplitz?
    ew, ev = np.linalg.eigh(W02)
    thresh = np.max(np.abs(ew)) * 1e-10
    P_null = ev[:, np.abs(ew) <= thresh]

    M_null = P_null.T @ M @ P_null
    d_null = M_null.shape[0]
    t_null = extract_toeplitz(M_null)
    T_null = toeplitz_from_seq(t_null, d_null)
    res_null = M_null - T_null

    frob_Mn = np.linalg.norm(M_null, 'fro')
    frob_Rn = np.linalg.norm(res_null, 'fro')

    print(f"\n  M restricted to null(W02) (dim={d_null}):")
    print(f"    ||M_null||_F = {frob_Mn:.4f}")
    print(f"    Non-Toeplitz: {frob_Rn:.4f} ({frob_Rn/frob_Mn*100:.1f}%)")

    # Symbol of M_null
    t_0n = t_null[d_null - 1]
    symbol_null = np.zeros(len(theta_vals))
    for i, theta in enumerate(theta_vals):
        s = t_0n
        for k in range(1, d_null):
            s += 2 * t_null[d_null - 1 + k] * np.cos(k * theta)
        symbol_null[i] = s

    print(f"    Symbol range: [{np.min(symbol_null):+.6f}, {np.max(symbol_null):+.6f}]")
    print(f"    Symbol non-positive: {np.max(symbol_null) < 1e-10}")

    return symbol, symbol_null, t_M


def toeplitz_deeper(lam_sq, N=None):
    """
    Deeper Toeplitz analysis: use the GENERATING FUNCTION interpretation.

    For the prime sum part: M_prime[n,m] = sum_{p^k} w(p^k) * q(n,m,log(p^k))

    The q function for n != m:
    q(n,m,y) = [sin(2*pi*m*y/L) - sin(2*pi*n*y/L)] / [pi*(n-m)]
             = 2*cos(pi*(n+m)*y/L) * sin(pi*(n-m)*y/L) / [pi*(n-m)]

    The first factor cos(pi*(n+m)*y/L) depends on (n+m) -- this is the NON-Toeplitz part!
    The second factor sin(pi*(n-m)*y/L) / [pi*(n-m)] depends only on (n-m) -- Toeplitz.

    So: q(n,m,y) = D(n+m, y) * S(n-m, y)
    where D depends on the sum and S is the sinc-like Toeplitz part.

    The Toeplitz part S(k, y) = sin(pi*k*y/L) / (pi*k) = (1/L) sinc(k*y/L) roughly.
    Its Fourier transform (symbol) is a rectangular window: S-hat(theta) = 1 for |theta| < y/L.

    The non-Toeplitz part D(s, y) = cos(pi*s*y/L) modulates each anti-diagonal.

    For the DIAGONAL (n=m): q(n,n,y) = 2(L-y)/L * cos(2*pi*n*y/L)
    This IS n-dependent (not constant on the diagonal).

    KEY INSIGHT: The non-Toeplitz correction comes from the (n+m) dependence,
    which is a SLOWLY VARYING modulation. For indices near the center (n,m ~ 0),
    the modulation is ~1. For indices far from center, it oscillates.
    """
    if N is None:
        L_val = np.log(lam_sq)
        N = max(15, round(6 * L_val))
    dim = 2 * N + 1
    L_f = np.log(lam_sq)

    _, _, M_prime, _, primes = compute_M_decomposition(lam_sq, N)

    print(f"\nTOEPLITZ DEEPER: lam^2={lam_sq}")
    print(f"  q(n,m,y) = D(n+m,y) * S(n-m,y)")
    print(f"  D(s,y) = cos(pi*s*y/L) -- non-Toeplitz modulation")
    print(f"  S(k,y) = sin(pi*k*y/L) / (pi*k) -- Toeplitz sinc kernel")

    # Build the PURE TOEPLITZ part of M_prime
    # by averaging over the (n+m) modulation
    ns = np.arange(-N, N + 1, dtype=float)
    M_prime_toeplitz = np.zeros((dim, dim))

    for pk, logp, logpk in primes:
        w = logp * pk**(-0.5)
        y = logpk
        # Toeplitz part: for each k = n-m, average D(n+m, y) over valid (n,m)
        for i in range(dim):
            n_val = ns[i]
            for j in range(dim):
                m_val = ns[j]
                if n_val != m_val:
                    # S factor (Toeplitz)
                    S = np.sin(np.pi * (n_val - m_val) * y / L_f) / (np.pi * (n_val - m_val))
                    # D factor (non-Toeplitz): average cos(pi*(n+m)*y/L) -> 0 for random n+m
                    # But actually we want the EXACT Toeplitz approximation
                    # which replaces D(n+m) with its average over the diagonal
                    # For now, just compute the full thing
                    pass

    # Actually, let's directly measure the (n+m) modulation
    # For each prime power, compute the "Toeplitz defect"
    print(f"\n  Toeplitz defect per prime power:")
    print(f"  {'p^k':>5} {'||T||':>8} {'||T-Toep(T)||':>14} {'defect%':>8}")

    for pk, logp, logpk in primes[:15]:
        w = logp * pk**(-0.5)
        y = logpk
        T = np.zeros((dim, dim))
        for i in range(dim):
            n_val = ns[i]
            for j in range(dim):
                m_val = ns[j]
                if n_val != m_val:
                    T[i, j] = w * (np.sin(2*np.pi*m_val*y/L_f) -
                                   np.sin(2*np.pi*n_val*y/L_f)) / (np.pi*(n_val-m_val))
                else:
                    T[i, j] = w * 2*(L_f-y)/L_f * np.cos(2*np.pi*n_val*y/L_f)
        T = (T + T.T) / 2

        # Extract Toeplitz part
        t_seq = np.zeros(2*dim-1)
        counts = np.zeros(2*dim-1)
        for i in range(dim):
            for j in range(dim):
                k = i - j + dim - 1
                t_seq[k] += T[i,j]
                counts[k] += 1
        t_seq /= np.maximum(counts, 1)

        T_toep = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                T_toep[i,j] = t_seq[i-j+dim-1]

        defect = np.linalg.norm(T - T_toep, 'fro')
        norm_T = np.linalg.norm(T, 'fro')
        pct = defect / norm_T * 100 if norm_T > 1e-15 else 0

        print(f"  {pk:>5} {norm_T:>8.4f} {defect:>14.4f} {pct:>7.1f}%")

    # The symbol of the full M_prime Toeplitz part
    # and comparison with actual eigenvalues
    return


if __name__ == "__main__":
    print("SESSION 35d -- THREE SURVIVING ANGLES")
    print("=" * 80)

    lam_sq = 50  # Start small for speed, especially angle 1

    # Angle 1: Spectral representation (computationally heavy)
    print(f"\n{'#' * 80}")
    print(f"# lam^2 = {lam_sq}")
    print(f"{'#' * 80}")
    spectral_representation(lam_sq, n_zeros=50)

    # Angle 2: Zero-free region
    for ls in [50, 200, 1000]:
        zero_free_region_bootstrap(ls)

    # Angle 3: Toeplitz symbol
    for ls in [50, 200]:
        toeplitz_symbol(ls)
        toeplitz_deeper(ls)

    with open('session35_three_angles.json', 'w') as f:
        json.dump({'status': 'complete'}, f)
    print(f"\nDone.")
