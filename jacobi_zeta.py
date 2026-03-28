"""Jacobi matrix of the zeta zeros.

Given eigenvalues gamma_1, ..., gamma_N, construct the unique N x N
tridiagonal (Jacobi) matrix:

    J = | a_1  b_1                    |
        | b_1  a_2  b_2               |
        |      b_2  a_3  b_3          |
        |            ...  ...  ...     |
        |                 b_{N-1}  a_N |

The sequences {a_k} and {b_k} encode the complete spectral information.

For GUE(N) eigenvalues:
  a_k ~ 0 (centered)
  b_k ~ sqrt(k(N-k))/N  (Marchenko-Pastur-like)

For zeta zeros: UNKNOWN. That's what we're here to find out.

Method: Lanczos tridiagonalization applied to the diagonal matrix
D = diag(gamma_1, ..., gamma_N) with a random starting vector.
The result is basis-independent (up to sign choices of b_k).

Connection to primes: the moments m_k = (1/N) sum gamma_j^k are
related to prime sums via the explicit formula. The Jacobi parameters
are determined by the moments through the Stieltjes procedure.
"""

import numpy as np
from scipy.linalg import eigh_tridiagonal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json


def lanczos_from_eigenvalues(eigenvalues, v0=None):
    """Lanczos tridiagonalization for a diagonal matrix.

    Given eigenvalues lambda_1, ..., lambda_N of a diagonal matrix D,
    and a starting vector v0, compute the Jacobi parameters (a, b)
    such that Q^T D Q = J (tridiagonal).

    The starting vector determines which Jacobi matrix we get.
    The most natural choice: v0 = (1,1,...,1)/sqrt(N) (uniform).

    Returns:
        a: diagonal entries [a_1, ..., a_N]
        b: off-diagonal entries [b_1, ..., b_{N-1}]
    """
    N = len(eigenvalues)
    lam = np.array(eigenvalues, dtype=np.float64)

    if v0 is None:
        v0 = np.ones(N) / np.sqrt(N)
    else:
        v0 = v0 / np.linalg.norm(v0)

    # Lanczos iteration
    # In eigenvalue basis, D*v = lam * v (componentwise)
    a = np.zeros(N)
    b = np.zeros(N - 1)

    v_prev = np.zeros(N)
    v_curr = v0.copy()

    for k in range(N):
        # w = D * v_curr (in eigenvalue basis: multiply componentwise)
        w = lam * v_curr

        # a_k = v_curr^T * w
        a[k] = np.dot(v_curr, w)

        # w = w - a_k * v_curr - b_{k-1} * v_prev
        w = w - a[k] * v_curr
        if k > 0:
            w = w - b[k-1] * v_prev

        # Reorthogonalize (important for numerical stability)
        # Full reorthogonalization against all previous vectors
        # (we'd need to store them, but for moderate N this is fine)

        # b_k = ||w||
        b_k = np.linalg.norm(w)

        if k < N - 1:
            b[k] = b_k
            if b_k < 1e-14:
                # Breakdown: w is in the span of previous vectors
                # Use a random restart
                w = np.random.randn(N)
                for _ in range(3):  # reorthogonalize
                    w -= np.dot(w, v_curr) * v_curr
                    if k > 0:
                        w -= np.dot(w, v_prev) * v_prev
                b_k = np.linalg.norm(w)
                b[k] = b_k

            v_prev = v_curr.copy()
            v_curr = w / b_k

    return a, b


def stieltjes_from_moments(moments, N):
    """Stieltjes procedure: moments -> Jacobi parameters.

    Given the moments m_0, m_1, ..., m_{2N-1} of a measure,
    compute the Jacobi parameters (a, b) of the associated
    orthogonal polynomials.

    The moments of the zeta zero measure:
    m_k = (1/N) * sum_{j=1}^N gamma_j^k
    """
    # Build the Hankel matrix H_{ij} = m_{i+j}
    H = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i + j < len(moments):
                H[i, j] = moments[i + j]

    # The Jacobi parameters can be extracted from the Cholesky-like
    # decomposition of H, but this is numerically unstable.
    # Use the Lanczos approach instead (more stable).
    return None  # Use lanczos_from_eigenvalues instead


def jacobi_from_zeros(zeros, starting_vector='uniform'):
    """Compute Jacobi parameters from zeta zeros.

    Args:
        zeros: array of zero ordinates gamma_1, ..., gamma_N
        starting_vector: 'uniform', 'counting', or 'random'

    Returns:
        a, b arrays
    """
    N = len(zeros)

    if starting_vector == 'uniform':
        v0 = np.ones(N) / np.sqrt(N)
    elif starting_vector == 'counting':
        # Weight by position in the counting function
        # v0_j proportional to 1/sqrt(gamma_j)
        v0 = 1.0 / np.sqrt(np.abs(zeros) + 1)
        v0 /= np.linalg.norm(v0)
    elif starting_vector == 'random':
        v0 = np.random.randn(N)
        v0 /= np.linalg.norm(v0)
    else:
        v0 = starting_vector
        v0 = v0 / np.linalg.norm(v0)

    a, b = lanczos_from_eigenvalues(zeros, v0)

    # Verify: reconstruct eigenvalues from Jacobi matrix
    eigs_reconstructed = eigh_tridiagonal(a, b, eigvals_only=True)
    eigs_original = np.sort(zeros)
    max_error = np.max(np.abs(eigs_reconstructed - eigs_original))

    return a, b, max_error


def gue_jacobi_prediction(N):
    """Theoretical Jacobi parameters for GUE(N).

    For GUE, in the "bulk" (away from edges), the Jacobi parameters
    of the unfolded eigenvalues are approximately:
      a_k ~ mean (centering)
      b_k ~ d/2 * sqrt(1 - ((2k-N)/N)^2)  (semicircle scaling)

    But for the ORDERED eigenvalues with uniform starting vector,
    the Lanczos process gives something different.
    We compute it empirically from GUE samples.
    """
    n_samples = 200
    a_samples = np.zeros((n_samples, N))
    b_samples = np.zeros((n_samples, N - 1))

    for i in range(n_samples):
        # Generate GUE eigenvalues
        M = np.random.randn(N, N) + 1j * np.random.randn(N, N)
        M = (M + M.conj().T) / (2 * np.sqrt(N))
        eigs = np.sort(np.linalg.eigvalsh(M))

        a, b, _ = jacobi_from_zeros(eigs)
        a_samples[i] = a
        b_samples[i] = b

    return {
        'a_mean': np.mean(a_samples, axis=0),
        'a_std': np.std(a_samples, axis=0),
        'b_mean': np.mean(b_samples, axis=0),
        'b_std': np.std(b_samples, axis=0),
    }


def run_jacobi_analysis():
    """Full Jacobi matrix analysis of zeta zeros."""

    print("=" * 70)
    print("JACOBI MATRIX OF THE ZETA ZEROS")
    print("=" * 70)

    all_zeros = np.load('_zeros_200.npy')

    # Step 1: Compute Jacobi parameters for various N
    print("\n[1/4] Jacobi parameters from zeta zeros")

    results = {}
    for N in [25, 50, 100, 200]:
        zeros = all_zeros[:N]

        # Center the zeros (subtract mean) for cleaner analysis
        mean_z = np.mean(zeros)
        zeros_centered = zeros - mean_z
        d = np.mean(np.diff(np.sort(zeros)))

        # Normalize by mean spacing
        zeros_normalized = zeros_centered / d

        a, b, err = jacobi_from_zeros(zeros_normalized)

        print(f"\n  N = {N}, d = {d:.4f}, reconstruction error = {err:.2e}")
        print(f"  a range: [{a.min():.4f}, {a.max():.4f}]")
        print(f"  b range: [{b.min():.4f}, {b.max():.4f}]")
        print(f"  a mean: {np.mean(a):.4f}, a std: {np.std(a):.4f}")
        print(f"  b mean: {np.mean(b):.4f}, b std: {np.std(b):.4f}")

        results[N] = {
            'a': a, 'b': b, 'err': err, 'd': d,
            'mean_z': mean_z,
        }

    # Step 2: GUE comparison
    print("\n[2/4] GUE comparison (N=100)")
    N_compare = 100

    gue = gue_jacobi_prediction(N_compare)
    zeta_a = results[N_compare]['a']
    zeta_b = results[N_compare]['b']

    print(f"  GUE a_k: mean={np.mean(gue['a_mean']):.4f}, std={np.mean(gue['a_std']):.4f}")
    print(f"  Zeta a_k: mean={np.mean(zeta_a):.4f}, std={np.std(zeta_a):.4f}")
    print(f"  GUE b_k: mean={np.mean(gue['b_mean']):.4f}, std={np.mean(gue['b_std']):.4f}")
    print(f"  Zeta b_k: mean={np.mean(zeta_b):.4f}, std={np.std(zeta_b):.4f}")

    # Step 3: Look for patterns in a_k and b_k
    print("\n[3/4] Pattern analysis for N=200")
    a200 = results[200]['a']
    b200 = results[200]['b']
    ks = np.arange(len(a200))
    ks_b = np.arange(len(b200))

    # Check: is b_k smooth? Does it follow a simple curve?
    # For the semicircle law: b_k ~ (N/2pi) * sqrt(1 - (2k/N - 1)^2) (approximately)
    # For zeta zeros: ???

    # Fit b_k to polynomial
    from numpy.polynomial import polynomial as P
    # Fit b_k vs k/N
    x = (ks_b + 0.5) / len(a200)  # normalized position in [0, 1]
    coeffs_b = np.polyfit(x, b200, 4)
    b_fit = np.polyval(coeffs_b, x)
    b_residual = b200 - b_fit
    print(f"  b_k polynomial fit (degree 4): max residual = {np.max(np.abs(b_residual)):.4f}")
    print(f"  b_k polynomial coefficients: {coeffs_b}")

    # Check: is a_k linear?
    coeffs_a = np.polyfit(ks, a200, 1)
    a_fit = np.polyval(coeffs_a, ks)
    a_residual = a200 - a_fit
    print(f"  a_k linear fit: slope={coeffs_a[0]:.6f}, intercept={coeffs_a[1]:.4f}")
    print(f"  a_k linear residual: max={np.max(np.abs(a_residual)):.4f}, rms={np.sqrt(np.mean(a_residual**2)):.4f}")

    # Step 4: The key question — does b_k encode prime information?
    print("\n[4/4] Looking for arithmetic structure in b_k")

    # Compute the "fluctuation" of b_k around its smooth trend
    b_fluct = b_residual / np.std(b_residual) if np.std(b_residual) > 1e-10 else b_residual

    # Compute FFT of b_k fluctuations to look for periodic structure
    fft_b = np.fft.fft(b_fluct)
    power = np.abs(fft_b[:len(fft_b)//2])**2
    freqs = np.fft.fftfreq(len(b_fluct))[:len(fft_b)//2]

    top_freq_idx = np.argsort(power[1:])[-5:] + 1  # top 5 frequencies (skip DC)
    print(f"  Top 5 frequencies in b_k fluctuations:")
    for idx in sorted(top_freq_idx, key=lambda i: -power[i]):
        period = 1.0 / freqs[idx] if freqs[idx] > 0 else float('inf')
        print(f"    freq={freqs[idx]:.4f}, period={period:.1f}, power={power[idx]:.2f}")

    # -- Plots --
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    # (0,0) a_k for zeta zeros (N=200)
    ax = axes[0, 0]
    ax.plot(ks, a200, 'b-', linewidth=0.8, label='Zeta zeros')
    ax.plot(ks, a_fit, 'r--', linewidth=1, label=f'Linear fit (slope={coeffs_a[0]:.4f})')
    ax.set_xlabel('k')
    ax.set_ylabel('a_k')
    ax.set_title('Diagonal Jacobi parameters (N=200)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (0,1) b_k for zeta zeros (N=200)
    ax = axes[0, 1]
    ax.plot(ks_b, b200, 'b-', linewidth=0.8, label='Zeta zeros')
    ax.plot(ks_b, b_fit, 'r--', linewidth=1, label='Polynomial fit (deg 4)')
    ax.set_xlabel('k')
    ax.set_ylabel('b_k')
    ax.set_title('Off-diagonal Jacobi parameters (N=200)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (1,0) Zeta vs GUE: a_k (N=100)
    ax = axes[1, 0]
    ks100 = np.arange(N_compare)
    ax.plot(ks100, zeta_a, 'b-', linewidth=1, label='Zeta')
    ax.fill_between(ks100,
                    gue['a_mean'] - 2*gue['a_std'],
                    gue['a_mean'] + 2*gue['a_std'],
                    alpha=0.3, color='red', label='GUE 2-sigma')
    ax.plot(ks100, gue['a_mean'], 'r--', linewidth=1)
    ax.set_xlabel('k')
    ax.set_ylabel('a_k')
    ax.set_title('Diagonal: Zeta vs GUE (N=100)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (1,1) Zeta vs GUE: b_k (N=100)
    ax = axes[1, 1]
    ks100b = np.arange(N_compare - 1)
    ax.plot(ks100b, zeta_b, 'b-', linewidth=1, label='Zeta')
    ax.fill_between(ks100b,
                    gue['b_mean'] - 2*gue['b_std'],
                    gue['b_mean'] + 2*gue['b_std'],
                    alpha=0.3, color='red', label='GUE 2-sigma')
    ax.plot(ks100b, gue['b_mean'], 'r--', linewidth=1)
    ax.set_xlabel('k')
    ax.set_ylabel('b_k')
    ax.set_title('Off-diagonal: Zeta vs GUE (N=100)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (2,0) b_k residuals (fluctuation around smooth trend)
    ax = axes[2, 0]
    ax.plot(ks_b, b_residual, 'g-', linewidth=0.8)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('k')
    ax.set_ylabel('b_k - fit')
    ax.set_title('b_k residual (arithmetic structure?)')
    ax.grid(True, alpha=0.3)

    # (2,1) FFT power spectrum of b_k fluctuations
    ax = axes[2, 1]
    ax.plot(freqs[1:len(power)], power[1:], 'k-', linewidth=0.8)
    for idx in top_freq_idx:
        ax.axvline(x=freqs[idx], color='red', alpha=0.3, linewidth=1)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Power')
    ax.set_title('FFT of b_k fluctuations')
    ax.grid(True, alpha=0.3)

    fig.suptitle('Jacobi Matrix of Zeta Zeros: A Constructive Hilbert-Polya?',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig('jacobi_zeta.png', dpi=150, bbox_inches='tight')
    print("\n  Saved: jacobi_zeta.png")
    plt.close(fig)

    # Save
    save_data = {
        'N_values': [25, 50, 100, 200],
        'a_200': a200.tolist(),
        'b_200': b200.tolist(),
        'a_linear_fit': coeffs_a.tolist(),
        'b_poly_fit': coeffs_b.tolist(),
    }
    with open('jacobi_zeta.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    print("  Saved: jacobi_zeta.json")

    return results, gue


if __name__ == '__main__':
    results, gue = run_jacobi_analysis()
