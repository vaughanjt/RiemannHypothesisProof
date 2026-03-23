"""Berry-Keating xp operator in spectral basis — the infinite-dimensional approach.

THE KEY INSIGHT: Previous attempts failed because we truncated the INTEGER
LATTICE l^2({1,...,N}), losing the analytic continuation. The correct approach
is to truncate a SPECTRAL BASIS of the continuous Hilbert space L^2(R+).

The Berry-Keating Hamiltonian:
    H = (xp + px) / 2 = -i(x d/dx + 1/2)

On L^2(R+, dx/x) with measure dx/x, this is self-adjoint with
continuous spectrum (-inf, +inf). The zeta zeros appear when we
add a POTENTIAL that encodes the functional equation.

SIERRA-TOWNSEND (2008): H = xp + V(x) where V imposes the
functional equation as a boundary condition. Their candidate:
    H_ST = -i(x d/dx + 1/2) + V(x)
with V chosen so the eigenvalue condition is xi(1/2 + iE) = 0.

BENDER-BRODY-MULLER (2017): H_BBM = (1-e^{-ip})xpe^{ip} formally
has eigenvalues at zeta zeros, but boundary conditions are problematic.

OUR APPROACH: Discretize in Laguerre basis {L_n^(alpha)(x)}.
On L^2(R+, x^alpha * e^{-x} dx):
    - x is a well-defined operator: <m|x|n> computable
    - p = -i d/dx: <m|p|n> computable
    - xp = -ix d/dx: combine the above

Matrix elements in the generalized Laguerre basis:
    <L_m | x | L_n> = -(2n + alpha + 1) delta_{mn} + ...
    <L_m | d/dx | L_n> = recursive formula from Laguerre recurrence

We truncate to the first N basis functions, giving an N x N matrix
whose eigenvalues approximate the continuous spectrum.
"""
import sys
import time

sys.path.insert(0, "src")
import numpy as np
from scipy.special import factorial, gammaln
from scipy.stats import pearsonr, kstest
import mpmath

t0 = time.time()
mpmath.mp.dps = 20


# ============================================================
# Laguerre basis matrix elements
# ============================================================

def laguerre_x_matrix(N, alpha=0):
    """Matrix elements of x in the Laguerre basis L_n^(alpha)(x).

    Using the three-term recurrence for x * L_n^(alpha):
    x * L_n^(alpha) = -(n+1)*L_{n+1} + (2n+alpha+1)*L_n - (n+alpha)*L_{n-1}

    So <m|x|n> in the orthonormal basis phi_n = L_n * sqrt(n! / Gamma(n+alpha+1)):
    """
    # In the ORTHONORMAL Laguerre basis:
    # <m|x|n> = (2n + alpha + 1) * delta_{mn}
    #         - sqrt((n+1)(n+alpha+1)) * delta_{m,n+1}
    #         - sqrt(n(n+alpha)) * delta_{m,n-1}
    X = np.zeros((N, N))
    for n in range(N):
        X[n, n] = 2 * n + alpha + 1
        if n + 1 < N:
            off = np.sqrt((n + 1) * (n + alpha + 1))
            X[n, n + 1] = -off
            X[n + 1, n] = -off
    return X


def laguerre_derivative_matrix(N, alpha=0):
    """Matrix elements of d/dx in the Laguerre basis.

    d/dx L_n^(alpha)(x) = -sum_{k=0}^{n-1} L_k^(alpha)(x)  (for alpha=0)

    More generally, in the orthonormal basis:
    <m|d/dx|n> = -sqrt(n!/m!) / sqrt(Gamma ratios) for m < n

    For alpha=0, orthonormal L_n = L_n / 1 (already orthonormal with weight e^{-x}):
    <m|d/dx|n> = -1 for m < n, 0 otherwise (in NON-orthonormal basis)

    In orthonormal basis phi_n = L_n (norm = 1 for alpha=0):
    <phi_m|d/dx|phi_n> = -1 for m < n
                       = -1/2 for m = n (from boundary contribution)

    Actually, let me use the numerical approach: build from the
    recurrence x * L_n' = n * L_n - (n) * L_{n-1}
    """
    # Cleaner approach: use x * d/dx directly
    # x * d/dx L_n^(0) = n * L_n - n * L_{n-1}  (for alpha=0)
    # Actually: (x * d/dx) L_n^(alpha) = n * L_n^(alpha) - (n + alpha) * L_{n-1}^(alpha)
    #
    # So <m|x d/dx|n> = n * delta_{mn} - (n+alpha) * <m|L_{n-1}>
    #                  = n * delta_{mn} - (n+alpha) * delta_{m,n-1} (in orthonormal basis with correction)
    #
    # In orthonormal basis phi_n = L_n * sqrt(n! / Gamma(n+alpha+1)):
    # <phi_m | x d/dx | phi_n> = n * delta_{mn} - sqrt(n(n+alpha)) * delta_{m,n-1}

    XD = np.zeros((N, N))
    for n in range(N):
        XD[n, n] = n
        if n > 0:
            off = np.sqrt(n * (n + alpha))
            XD[n - 1, n] = -off  # <n-1| x d/dx |n>
    return XD


def build_xp_operator(N, alpha=0):
    """Build H = -i(x d/dx + 1/2) in Laguerre basis.

    This is the symmetrized Berry-Keating operator (xp + px)/2.
    """
    XD = laguerre_derivative_matrix(N, alpha)
    # H = -i * (x d/dx + 1/2)
    H = -1j * (XD + 0.5 * np.eye(N))
    return H


def build_xp_hermitian(N, alpha=0):
    """Build the HERMITIAN part of xp.

    H = (xp + (xp)^*) / 2 = Re(-i * x d/dx) - 1/2
    """
    XD = laguerre_derivative_matrix(N, alpha)
    H_full = -1j * (XD + 0.5 * np.eye(N))
    # Hermitian part
    H = (H_full + H_full.conj().T) / 2
    return H


def build_sierra_townsend(N, alpha=0, V_coeff=-0.25):
    """Sierra-Townsend: H = xp + V(x) where V(x) = V_coeff / x.

    The -i/(4x) potential encodes the functional equation.
    We need matrix elements of 1/x in Laguerre basis.
    """
    H = build_xp_operator(N, alpha)

    # Matrix elements of 1/x in Laguerre basis
    # <L_m^(0) | 1/x | L_n^(0)> = 1 for all m, n with m <= n
    # (This is for the NON-orthonormal basis with weight e^{-x})
    #
    # Actually, for orthonormal Laguerre with alpha=0:
    # <phi_m | 1/x | phi_n> = min(m,n) + 1... no, that's not right.
    #
    # The integral: integral_0^inf L_m(x) * L_n(x) * (1/x) * e^{-x} dx
    # For alpha=0 Laguerre: = 1 for m <= n (can be shown by recurrence)
    # So <m|1/x|n> = 1 for all m, n (all entries = 1!?)
    # That doesn't seem right. Let me compute numerically.

    # Numerical computation of <m|1/x|n>
    inv_x = np.zeros((N, N))
    from numpy.polynomial.laguerre import lagval
    # Use quadrature
    # For Laguerre polynomials with weight e^{-x} on [0, inf):
    # Use Gauss-Laguerre quadrature
    from numpy.polynomial.laguerre import laggauss
    n_quad = max(3 * N, 200)
    x_quad, w_quad = laggauss(n_quad)

    # Evaluate orthonormal Laguerre functions at quadrature points
    # L_n(x) (standard Laguerre, orthonormal w.r.t. e^{-x} dx)
    L_vals = np.zeros((N, n_quad))
    for n in range(N):
        coeffs = np.zeros(n + 1)
        coeffs[n] = 1.0
        L_vals[n, :] = lagval(x_quad, coeffs)

    # <m|1/x|n> = integral L_m(x) * (1/x) * L_n(x) * e^{-x} dx
    # With Gauss-Laguerre: sum_i w_i * L_m(x_i) * L_n(x_i) / x_i
    for m in range(N):
        for n in range(m, N):
            val = np.sum(w_quad * L_vals[m, :] * L_vals[n, :] / x_quad)
            inv_x[m, n] = val
            inv_x[n, m] = val

    H += V_coeff * 1j * inv_x  # V(x) = V_coeff * i / x
    return H


def build_bender_brody_muller(N, alpha=0):
    """Bender-Brody-Muller operator (simplified version).

    H_BBM = (1 - e^{-ip}) * xp * e^{ip}

    This is PT-symmetric. In practice, we build:
    H = xp - e^{-ip} * xp * e^{ip}

    The momentum shift operator e^{ip} translates x -> x+1.
    In Laguerre basis, this is the TRANSLATION operator.
    """
    # e^{ip} |f(x)> = |f(x+1)>
    # In Laguerre basis: <m|e^{ip}|n> = <L_m(x)|L_n(x+1)>
    # This can be computed from the generating function of Laguerre polynomials.

    # For simplicity, use numerical quadrature
    from numpy.polynomial.laguerre import lagval, laggauss
    n_quad = max(3 * N, 200)
    x_quad, w_quad = laggauss(n_quad)

    L_vals = np.zeros((N, n_quad))
    L_shifted = np.zeros((N, n_quad))
    for n in range(N):
        coeffs = np.zeros(n + 1)
        coeffs[n] = 1.0
        L_vals[n, :] = lagval(x_quad, coeffs)
        # L_n(x+1) — evaluate at shifted points
        L_shifted[n, :] = lagval(x_quad + 1, coeffs)

    # Translation matrix: T_{mn} = <L_m(x) | L_n(x+1)> w.r.t. e^{-x}dx
    T = np.zeros((N, N))
    for m in range(N):
        for n in range(N):
            T[m, n] = np.sum(w_quad * L_vals[m, :] * L_shifted[n, :])

    # xp matrix
    XD = laguerre_derivative_matrix(N, alpha)
    XP = -1j * (XD + 0.5 * np.eye(N))

    # H = XP - T^{-1} @ XP @ T  (the BBM construction)
    # Or simpler: H = (I - T^{-1}) @ XP @ T ... various forms exist
    # Let's use: H = XP - T.conj().T @ XP @ T
    H = XP - T.T @ XP @ T
    return H


# ============================================================
# TEST 1: Basic xp spectrum in Laguerre basis
# ============================================================
print("=" * 70)
print("TEST 1: BERRY-KEATING xp IN LAGUERRE BASIS")
print("=" * 70)

known_zeros_t = [14.1347, 21.0220, 25.0109, 30.4249, 32.9351,
                  37.5862, 40.9187, 43.3271, 48.0052, 49.7738]

for N in [50, 100, 200]:
    H = build_xp_operator(N)
    eigs = np.linalg.eigvals(H)
    eigs_sorted = np.sort(eigs.real)

    # The xp operator should have eigenvalues at i*(n+1/2) on the imaginary axis
    # Its Hermitian part should have spectrum related to...
    H_herm = build_xp_hermitian(N)
    eigs_herm = np.linalg.eigvalsh(H_herm)

    print(f"\n  N={N}:")
    print(f"    xp eigenvalues (first 10 real parts): {eigs_sorted[:10].round(4)}")
    print(f"    xp eigenvalues (last 10 real parts):  {eigs_sorted[-10:].round(4)}")
    print(f"    Hermitian part eigenvalues: [{eigs_herm[0]:.4f}, ..., {eigs_herm[-1]:.4f}]")
    print(f"    Imaginary parts range: [{np.min(eigs.imag):.4f}, {np.max(eigs.imag):.4f}]")


# ============================================================
# TEST 2: Sierra-Townsend with V = -i/(4x)
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: SIERRA-TOWNSEND H = xp - i/(4x)")
print("=" * 70)

for N in [50, 100, 200, 400]:
    t_start = time.time()
    H = build_sierra_townsend(N, V_coeff=-0.25)
    eigs = np.linalg.eigvals(H)
    elapsed = time.time() - t_start

    # Sort by imaginary part (the "energy")
    eigs_by_imag = eigs[np.argsort(eigs.imag)]

    # The eigenvalues should be near s = 1/2 + it where t are zeta zeros
    # So look for eigenvalues with Re ~ 0 and Im ~ zeta zero locations
    positive_imag = eigs_by_imag[eigs_by_imag.imag > 5]

    print(f"\n  N={N} ({elapsed:.1f}s):")
    print(f"    Total eigenvalues: {len(eigs)}")
    print(f"    With Im > 5: {len(positive_imag)}")

    if len(positive_imag) > 0:
        print(f"    {'Eigenvalue':>30} {'|Re|':>8} {'Im':>10} {'Nearest zero':>14} {'Dist':>8}")
        for e in positive_imag[:15]:
            dists = [abs(e.imag - z) for z in known_zeros_t]
            best = np.argmin(dists)
            tag = " <<<" if dists[best] < 1.0 else ""
            print(f"    {e.real:>+14.6f}{e.imag:>+14.6f}i {abs(e.real):>8.4f} "
                  f"{e.imag:>10.4f} {known_zeros_t[best]:>14.4f} {dists[best]:>8.4f}{tag}")


# ============================================================
# TEST 3: Vary the potential strength
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: POTENTIAL STRENGTH SWEEP V = c*i/x")
print("=" * 70)

N_sweep = 100
print(f"\n  N={N_sweep}")
print(f"  {'V_coeff':>10} {'n_eigs(Im>5)':>14} {'nearest_zero_dist':>20}")
print(f"  {'-'*48}")

for V_c in [-1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5]:
    H = build_sierra_townsend(N_sweep, V_coeff=V_c)
    eigs = np.linalg.eigvals(H)
    pos = eigs[eigs.imag > 5]

    if len(pos) > 0:
        # Average distance to nearest zeta zero
        min_dists = []
        for e in pos[:20]:
            dists = [abs(e.imag - z) for z in known_zeros_t]
            min_dists.append(min(dists))
        mean_dist = np.mean(min_dists)
    else:
        mean_dist = float("inf")

    print(f"  {V_c:>+10.2f} {len(pos):>14} {mean_dist:>20.4f}")


# ============================================================
# TEST 4: Bender-Brody-Muller
# ============================================================
print("\n" + "=" * 70)
print("TEST 4: BENDER-BRODY-MULLER OPERATOR")
print("=" * 70)

for N in [50, 100]:
    t_start = time.time()
    H = build_bender_brody_muller(N)
    eigs = np.linalg.eigvals(H)
    elapsed = time.time() - t_start

    eigs_by_imag = eigs[np.argsort(eigs.imag)]
    positive_imag = eigs_by_imag[eigs_by_imag.imag > 5]

    print(f"\n  N={N} ({elapsed:.1f}s):")
    print(f"    Total eigenvalues: {len(eigs)}")
    print(f"    Real part range: [{np.min(eigs.real):.4f}, {np.max(eigs.real):.4f}]")
    print(f"    Imag part range: [{np.min(eigs.imag):.4f}, {np.max(eigs.imag):.4f}]")

    if len(positive_imag) > 0:
        print(f"    {'Eigenvalue':>30} {'Nearest zero':>14} {'Dist':>8}")
        for e in positive_imag[:10]:
            dists = [abs(e.imag - z) for z in known_zeros_t]
            best = np.argmin(dists)
            tag = " <<<" if dists[best] < 1.0 else ""
            print(f"    {e.real:>+14.6f}{e.imag:>+14.6f}i "
                  f"{known_zeros_t[best]:>14.4f} {dists[best]:>8.4f}{tag}")
    else:
        print(f"    No eigenvalues with Im > 5")


# ============================================================
# TEST 5: Direct discretization of xp on a grid
# ============================================================
print("\n" + "=" * 70)
print("TEST 5: DIRECT GRID DISCRETIZATION OF xp")
print("=" * 70)

# On a logarithmic grid x_j = e^{j*h}, the operator xp = -ix d/dx
# becomes -i d/du where u = log(x). This is just the momentum operator
# on a uniform grid! Its eigenvalues are k*h for integer k.
#
# The INTERESTING part is the boundary conditions:
# - Periodic BC: eigenvalues = 2*pi*n/(N*h)
# - Dirichlet BC: eigenvalues = pi*n/(N*h)
# - Modified BC encoding functional equation: ???

# Let's try: x_j = j*h on [0, L], p = -i * central differences / h
# xp_{jk} = x_j * p_{jk}

for N in [100, 200, 500]:
    h = 0.1
    x = np.arange(1, N + 1) * h  # x_j = j*h, j=1,...,N (avoid x=0)

    # Central difference d/dx
    D = np.zeros((N, N))
    for j in range(N):
        if j > 0:
            D[j, j - 1] = -1 / (2 * h)
        if j < N - 1:
            D[j, j + 1] = 1 / (2 * h)

    # xp = -i * diag(x) @ D
    H = -1j * np.diag(x) @ D

    # Symmetrize: (xp + px) / 2 = -i(x d/dx + 1/2)
    H_sym = (H + H.conj().T) / 2

    eigs = np.linalg.eigvalsh(H_sym)

    print(f"\n  Grid N={N}, h={h}, L={N*h:.1f}")
    print(f"    Eigenvalue range: [{eigs[0]:.4f}, {eigs[-1]:.4f}]")
    print(f"    Spacing near 0: {np.diff(eigs[N//2-3:N//2+3]).round(4)}")

    # Compare to pi/(N*h) spacing
    expected_spacing = np.pi / (N * h)
    actual_spacing = np.mean(np.diff(eigs[N//4:3*N//4]))
    print(f"    Expected spacing pi/(Nh): {expected_spacing:.6f}")
    print(f"    Actual mean spacing:      {actual_spacing:.6f}")


# ============================================================
# VERDICT
# ============================================================
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

print(f"\nTotal time: {time.time() - t0:.1f}s")
