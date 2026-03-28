"""Jacobi matrix of zeta zeros — v2: unfolded, stabilized.

Improvements over v1:
1. Unfold zeros using smooth N(T) counting function
2. Full Lanczos reorthogonalization (store all vectors)
3. Multiple starting vectors compared
4. Reconstruction verified to machine precision
5. Prime signature analysis in b_k fluctuations
"""

import numpy as np
from scipy.linalg import eigh_tridiagonal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json


# -- Unfolding --------------------------------------------------------

def smooth_counting(T):
    """Smooth approximation to N(T), the zero counting function.

    N(T) ~ (T/2pi) * log(T/2pi) - T/2pi + 7/8
    """
    if T <= 0:
        return 0.0
    x = T / (2 * np.pi)
    return x * np.log(x) - x + 7.0/8


def unfold_zeros(gammas):
    """Map gamma_n -> theta_n = N(gamma_n).

    The unfolded zeros theta_n have mean spacing ~1.
    """
    return np.array([smooth_counting(g) for g in gammas])


# -- Stable Lanczos --------------------------------------------------

def lanczos_full_reorth(eigenvalues, v0=None):
    """Lanczos with full reorthogonalization.

    Stores all Lanczos vectors and orthogonalizes each new vector
    against ALL previous ones. This prevents loss of orthogonality
    and gives machine-precision Jacobi parameters.

    Args:
        eigenvalues: the diagonal of D (we compute D*v = lambda.*v)
        v0: starting vector (default: uniform)

    Returns:
        a: diagonal [a_1, ..., a_N]
        b: off-diagonal [b_1, ..., b_{N-1}]
    """
    N = len(eigenvalues)
    lam = np.array(eigenvalues, dtype=np.float64)

    if v0 is None:
        v0 = np.ones(N) / np.sqrt(N)
    v0 = v0 / np.linalg.norm(v0)

    a = np.zeros(N)
    b = np.zeros(N - 1)

    # Store all Lanczos vectors for reorthogonalization
    Q = np.zeros((N, N))
    Q[:, 0] = v0

    for k in range(N):
        v = Q[:, k]

        # w = D * v (componentwise multiply in eigenvalue basis)
        w = lam * v

        # a_k = <v, w>
        a[k] = np.dot(v, w)

        # w = w - a_k * v
        w -= a[k] * v

        # Subtract projection onto previous vector (3-term recurrence)
        if k > 0:
            w -= b[k-1] * Q[:, k-1]

        # FULL reorthogonalization: subtract projections onto ALL previous vectors
        for j in range(k + 1):
            w -= np.dot(Q[:, j], w) * Q[:, j]

        # Second pass (for numerical stability)
        for j in range(k + 1):
            w -= np.dot(Q[:, j], w) * Q[:, j]

        b_k = np.linalg.norm(w)

        if k < N - 1:
            b[k] = b_k
            if b_k > 1e-15:
                Q[:, k+1] = w / b_k
            else:
                # Invariant subspace found — use random restart
                r = np.random.randn(N)
                for j in range(k + 1):
                    r -= np.dot(Q[:, j], r) * Q[:, j]
                for j in range(k + 1):
                    r -= np.dot(Q[:, j], r) * Q[:, j]
                Q[:, k+1] = r / np.linalg.norm(r)
                b[k] = 0.0

    return a, b


def verify_jacobi(a, b, eigenvalues_sorted):
    """Verify Jacobi matrix has the correct eigenvalues."""
    eigs = eigh_tridiagonal(a, b, eigvals_only=True)
    return np.max(np.abs(np.sort(eigs) - eigenvalues_sorted))


# -- Prime analysis ---------------------------------------------------

def prime_sieve(n_max):
    """Sieve of Eratosthenes."""
    sieve = np.ones(n_max + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n_max**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return np.where(sieve)[0]


def test_prime_signature(b_residual, d_mean=1.0, n_primes=20):
    """Test if b_k fluctuations correlate with prime-related frequencies.

    The explicit formula connects zero sums to prime sums.
    If the Jacobi parameters encode prime information, the b_k
    fluctuations should have spectral peaks at frequencies
    related to log(p) / (2*pi).

    For unfolded zeros with mean spacing 1, the natural frequency
    unit is 1/(2*pi). Prime p contributes oscillations at frequency
    log(p)/(2*pi*d) in the original zero sequence. After unfolding,
    this becomes log(p)/(2*pi).
    """
    N = len(b_residual)
    primes = prime_sieve(200)[:n_primes]

    # FFT of b_k residual
    fft = np.fft.fft(b_residual)
    power = np.abs(fft[:N//2])**2
    freqs = np.fft.fftfreq(N)[:N//2]

    # Expected prime frequencies: log(p) / (2*pi) (in unfolded units)
    prime_freqs = np.log(primes) / (2 * np.pi)
    # Wrap to [0, 0.5] (Nyquist)
    prime_freqs_wrapped = prime_freqs % 0.5

    # For each prime, find the nearest FFT bin and measure power
    results = []
    for i, p in enumerate(primes):
        f_p = prime_freqs_wrapped[i]
        # Find nearest frequency bin
        idx = np.argmin(np.abs(freqs - f_p))
        # Also check neighboring bins
        local_power = max(power[max(0,idx-1):min(len(power),idx+2)])
        # Compare to median power (background)
        median_power = np.median(power[1:])  # skip DC
        snr = local_power / median_power if median_power > 0 else 0

        results.append({
            'prime': int(p),
            'log_p': float(np.log(p)),
            'freq': float(f_p),
            'nearest_fft_freq': float(freqs[idx]),
            'power': float(local_power),
            'snr': float(snr),
        })

    return results, power, freqs


# -- Main analysis ----------------------------------------------------

def run_jacobi_v2():
    """Full stabilized Jacobi analysis."""

    print("=" * 70)
    print("JACOBI MATRIX v2: UNFOLDED, STABILIZED")
    print("=" * 70)

    all_zeros = np.load('_zeros_200.npy')

    # Step 1: Unfold
    print("\n[1/5] Unfolding zeros")
    for N in [50, 100, 200]:
        gammas = all_zeros[:N]
        thetas = unfold_zeros(gammas)
        spacings = np.diff(thetas)
        print(f"  N={N}: unfolded spacing mean={np.mean(spacings):.4f},"
              f" std={np.std(spacings):.4f},"
              f" min={np.min(spacings):.4f}, max={np.max(spacings):.4f}")

    # Step 2: Compute Jacobi parameters with full reorthogonalization
    print("\n[2/5] Computing Jacobi parameters (full reorthogonalization)")

    results = {}
    for N in [50, 100, 200]:
        gammas = all_zeros[:N]
        thetas = unfold_zeros(gammas)

        # Center: subtract mean so eigenvalues are ~symmetric around 0
        theta_mean = np.mean(thetas)
        thetas_c = thetas - theta_mean

        a, b = lanczos_full_reorth(np.sort(thetas_c))
        err = verify_jacobi(a, b, np.sort(thetas_c))

        print(f"\n  N={N}: reconstruction error = {err:.2e}")
        print(f"    a: mean={np.mean(a):.6f}, std={np.std(a):.4f},"
              f" range=[{np.min(a):.4f}, {np.max(a):.4f}]")
        print(f"    b: mean={np.mean(b):.6f}, std={np.std(b):.4f},"
              f" range=[{np.min(b):.4f}, {np.max(b):.4f}]")

        results[N] = {'a': a, 'b': b, 'err': err, 'thetas_c': thetas_c}

    # Step 3: Compare starting vectors
    print("\n[3/5] Starting vector comparison (N=200)")
    thetas_200 = results[200]['thetas_c']
    thetas_sorted = np.sort(thetas_200)

    vectors = {
        'uniform': np.ones(200) / np.sqrt(200),
        'sqrt_weight': np.sqrt(np.arange(1, 201, dtype=float)),
        'chebyshev': np.cos(np.pi * np.arange(200) / 199),
    }

    for name, v0 in vectors.items():
        v0 = v0 / np.linalg.norm(v0)
        a_v, b_v = lanczos_full_reorth(thetas_sorted, v0)
        err_v = verify_jacobi(a_v, b_v, thetas_sorted)
        print(f"  {name:>15}: err={err_v:.2e},"
              f" b_mean={np.mean(b_v):.4f}, b_std={np.std(b_v):.4f}")

    # Step 4: Pattern analysis on b_k (using uniform vector, N=200)
    print("\n[4/5] Pattern analysis (N=200, uniform vector)")
    a200 = results[200]['a']
    b200 = results[200]['b']
    N = 200

    # Smooth trend of b_k
    ks = np.arange(len(b200))
    x = ks / len(b200)

    # Fit smooth trend (polynomial degree 6)
    coeffs = np.polyfit(x, b200, 6)
    b_smooth = np.polyval(coeffs, x)
    b_residual = b200 - b_smooth

    print(f"  b_k smooth trend: degree-6 polynomial fit")
    print(f"  Residual: rms={np.sqrt(np.mean(b_residual**2)):.4f},"
          f" max={np.max(np.abs(b_residual)):.4f}")

    # b_k autocorrelation
    b_normed = (b_residual - np.mean(b_residual)) / (np.std(b_residual) + 1e-15)
    autocorr = np.correlate(b_normed, b_normed, mode='full')
    autocorr = autocorr[len(b_normed)-1:] / len(b_normed)

    print(f"  Autocorrelation at lag 1: {autocorr[1]:.4f}")
    print(f"  Autocorrelation at lag 2: {autocorr[2]:.4f}")
    print(f"  Autocorrelation at lag 5: {autocorr[5]:.4f}")

    # Step 5: Prime signature test
    print("\n[5/5] Prime signature test")
    prime_results, fft_power, fft_freqs = test_prime_signature(b_residual)

    print(f"\n  {'prime':>6}  {'log(p)':>8}  {'freq':>8}  {'power':>10}  {'SNR':>8}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*8}")
    for pr in prime_results[:15]:
        marker = " ***" if pr['snr'] > 3.0 else ""
        print(f"  {pr['prime']:>6}  {pr['log_p']:>8.4f}  {pr['freq']:>8.4f}"
              f"  {pr['power']:>10.2f}  {pr['snr']:>8.2f}{marker}")

    # Also test: does b_k correlate with the prime counting function?
    # pi(k) = number of primes <= k
    primes = prime_sieve(N + 50)
    pi_k = np.array([np.sum(primes <= k) for k in range(1, len(b200)+1)], dtype=float)
    # Detrend pi_k (subtract k/log(k))
    pi_smooth = np.array([k / np.log(k) if k > 1 else 0 for k in range(1, len(b200)+1)])
    pi_residual = pi_k - pi_smooth

    corr_b_pi = np.corrcoef(b_residual, pi_residual[:len(b_residual)])[0, 1]
    print(f"\n  Correlation of b_k residual with pi(k) residual: r = {corr_b_pi:.4f}")

    # Test: b_k vs von Mangoldt function
    Lambda_k = np.zeros(len(b200))
    for p in primes:
        pk = p
        while pk <= len(b200):
            Lambda_k[pk - 1] += np.log(p)
            pk *= p
    corr_b_Lambda = np.corrcoef(b_residual, Lambda_k[:len(b_residual)])[0, 1]
    print(f"  Correlation of b_k residual with Lambda(k): r = {corr_b_Lambda:.4f}")

    # -- Plots --
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    # (0,0) Unfolded a_k
    ax = axes[0, 0]
    ax.plot(np.arange(len(a200)), a200, 'b-', linewidth=0.6)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('k')
    ax.set_ylabel('a_k')
    ax.set_title(f'Diagonal Jacobi (unfolded, N={N})')
    ax.grid(True, alpha=0.3)

    # (0,1) Unfolded b_k with smooth trend
    ax = axes[0, 1]
    ax.plot(ks, b200, 'b-', linewidth=0.6, label='b_k')
    ax.plot(ks, b_smooth, 'r-', linewidth=1.5, label='smooth trend')
    ax.set_xlabel('k')
    ax.set_ylabel('b_k')
    ax.set_title(f'Off-diagonal Jacobi (unfolded, N={N})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (1,0) b_k residual
    ax = axes[1, 0]
    ax.plot(ks, b_residual, 'g-', linewidth=0.6)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('k')
    ax.set_ylabel('b_k - trend')
    ax.set_title('b_k residual (arithmetic fluctuation)')
    ax.grid(True, alpha=0.3)

    # (1,1) FFT with prime frequencies marked
    ax = axes[1, 1]
    ax.plot(fft_freqs[1:], fft_power[1:], 'k-', linewidth=0.6)
    primes_small = prime_sieve(30)
    colors = plt.cm.tab10(np.linspace(0, 1, len(primes_small)))
    for i, p in enumerate(primes_small):
        f_p = (np.log(p) / (2 * np.pi)) % 0.5
        ax.axvline(x=f_p, color=colors[i], alpha=0.5, linewidth=1.5,
                   label=f'p={p}')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Power')
    ax.set_title('FFT of b_k residual with prime frequencies')
    ax.legend(fontsize=7, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)

    # (2,0) Autocorrelation of b_k residual
    ax = axes[2, 0]
    lags = np.arange(min(50, len(autocorr)))
    ax.stem(lags, autocorr[:len(lags)], linefmt='b-', markerfmt='bo',
            basefmt='gray')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('b_k residual autocorrelation')
    ax.grid(True, alpha=0.3)

    # (2,1) b_k residual vs pi(k) residual scatter
    ax = axes[2, 1]
    ax.scatter(pi_residual[:len(b_residual)], b_residual,
               s=10, alpha=0.5, c='blue')
    ax.set_xlabel('pi(k) residual')
    ax.set_ylabel('b_k residual')
    ax.set_title(f'b_k vs prime counting (r = {corr_b_pi:.3f})')
    ax.grid(True, alpha=0.3)

    fig.suptitle('Jacobi Matrix v2: Unfolded Zeros, Prime Signature Hunt',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig('jacobi_zeta_v2.png', dpi=150, bbox_inches='tight')
    print("\n  Saved: jacobi_zeta_v2.png")
    plt.close(fig)

    save_data = {
        'a_200': a200.tolist(),
        'b_200': b200.tolist(),
        'b_smooth_coeffs': coeffs.tolist(),
        'b_residual_rms': float(np.sqrt(np.mean(b_residual**2))),
        'reconstruction_errors': {str(N): results[N]['err'] for N in results},
        'prime_signature': prime_results,
        'corr_b_pi': float(corr_b_pi),
        'corr_b_Lambda': float(corr_b_Lambda),
        'autocorr_lag1': float(autocorr[1]),
    }
    with open('jacobi_zeta_v2.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    print("  Saved: jacobi_zeta_v2.json")

    return results


if __name__ == '__main__':
    run_jacobi_v2()
