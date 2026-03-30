"""
Session 24: Compute ACTUAL prolate spheroidal wave functions (not Gaussian
approximations) and measure overlap with the Weil eigenvector xi_lambda.

The prolate integral operator for the Connes construction:
  (K*psi)(t) = integral sin(c*(t-s))/(pi*(t-s)) * psi(s) ds  on [-T, T]

In the paper's framework:
  T = L/2 = log(lambda)   (half the log-interval)
  c = L/2                  (Mellin bandwidth on critical line)
  Time-bandwidth product: c*T = L^2/4

The eigenfunctions h_{n,lambda} of this operator, projected onto V_n basis,
give the prolate functions. The combination k_lambda = a*h_0 + b*h_4
(with integral vanishing condition) is the paper's target for xi_lambda.
"""

import numpy as np
import mpmath
from mpmath import (mp, mpf, mpc, matrix as mpmatrix, log, pi, euler,
                    exp, cos, sin, hyp2f1, digamma, sinh, eig, quad)
import sympy, time

mp.dps = 50

def primes_up_to(n):
    return list(sympy.primerange(2, n + 1))

def build_xi(lam_sq, N=30):
    L = log(mpf(lam_sq)); eL = exp(L)
    vM = []
    for p in primes_up_to(lam_sq):
        lp = log(mpf(p)); pk = mpf(p)
        while pk <= mpf(lam_sq):
            vM.append((pk, lp, log(pk))); pk *= p
    dim = 2 * N + 1; al = {}
    for n in range(-N, N + 1):
        nn = abs(n)
        if nn == 0: al[n] = mpf(0); continue
        z = exp(-2 * L); a = pi * mpc(0, nn) / L + mpf(1) / 4
        h = hyp2f1(1, a, a + 1, z)
        al[n] = (exp(-L / 2) * (2 * L / (L + 4 * pi * mpc(0, nn)) * h).imag
                 + digamma(a).imag / 2) / pi
        if n < 0: al[n] = -al[n]
    wr_d = {}
    for nv in range(N + 1):
        w_c = euler + log(4 * pi * (eL - 1) / (eL + 1))
        def ig(x, nv=nv):
            return (exp(x / 2) * 2 * (1 - x / L) * cos(2 * pi * nv * x / L) - 2) / (exp(x) - exp(-x))
        wr_d[nv] = w_c + quad(ig, [mpf(0), L]); wr_d[-nv] = wr_d[nv]
    tau = mpmatrix(dim, dim)
    L2 = L * L; p2 = 16 * pi * pi; pf = 32 * L * sinh(L / 4) ** 2
    def q_mp(n, m, y):
        if n != m:
            return (sin(2 * pi * m * y / L) - sin(2 * pi * n * y / L)) / (pi * (n - m))
        else:
            return 2 * (L - y) / L * cos(2 * pi * n * y / L)
    for i in range(dim):
        n = i - N
        for j in range(i, dim):
            m = j - N
            w02 = pf * (L2 - p2 * m * n) / ((L2 + p2 * m ** 2) * (L2 + p2 * n ** 2))
            wp = sum(lk * pk ** (-mpf(1) / 2) * q_mp(n, m, logk) for pk, lk, logk in vM)
            wr = wr_d[n] if n == m else (al[m] - al[n]) / (n - m)
            tau[i, j] = w02 - wr - wp; tau[j, i] = tau[i, j]
    E, ER = eig(tau, left=False, right=True)
    evals = sorted([(E[i].real, i) for i in range(dim)], key=lambda x: float(x[0]))
    eps = evals[0][0]; idx = evals[0][1]
    xi = [float(ER[j, idx].real) for j in range(dim)]
    xs = sum(xi); sqL = float(mpmath.sqrt(L))
    if abs(xs) > 1e-20: xi = [x * sqL / xs for x in xi]
    return np.array(xi), float(eps), float(L), N


def compute_prolate_eigenfunctions(L_f, N, n_grid=200):
    """Compute prolate eigenfunctions in the V_n basis.
    
    The concentration operator in the V_n basis:
    C_{jk} = (1/L) * integral_{-L/2}^{L/2} integral_{-L/2}^{L/2}
             K(t,s) * e^{2*pi*i*j*t/L} * e^{-2*pi*i*k*s/L} dt ds
    
    where K(t,s) = sin(c*(t-s)) / (pi*(t-s)) with c = L/2.
    
    In the V_n basis: C_{jk} = (1/L) * integral K(t,s) V_j(t) conj(V_k(s)) dt ds
    
    For equally spaced frequencies w_j = 2*pi*j/L:
    C_{jk} = sinc(c*(w_j - w_k)/pi) ... but this simplifies.
    
    Actually: the sinc kernel K(t,s) = sin(c(t-s))/(pi(t-s)) with c=L/2
    projected onto the V_n basis gives:
    
    C_{jk} = (1/L^2) integral integral sin(L(t-s)/4)/(pi(t-s)) * e^{2*pi*i*(j*t-k*s)/L} dt ds
    
    This is a convolution operator. For the V_n basis, it becomes:
    C_{jk} = delta_{jk} * mu_j  where mu_j = max(0, 1 - |2*pi*j/L| / c)
    
    Wait, that's the triangular window. Actually for the sinc kernel with
    bandwidth c on interval [-T, T], the eigenvalues in the Fourier basis
    depend on the time-bandwidth product c*T.
    
    Let me just discretize the operator on a spatial grid and project.
    """
    dim = 2 * N + 1
    
    # Discretize the sinc kernel on a grid in [-L/2, L/2]
    c = L_f / 2  # bandwidth parameter (Mellin)
    T = L_f / 2  # half-interval
    
    # Grid points in [-T, T]
    t_grid = np.linspace(-T + T/n_grid, T - T/n_grid, n_grid)
    dt = t_grid[1] - t_grid[0]
    
    # Build the sinc kernel matrix
    K_mat = np.zeros((n_grid, n_grid))
    for i in range(n_grid):
        for j in range(n_grid):
            if i == j:
                K_mat[i, j] = c / np.pi  # limit of sinc
            else:
                K_mat[i, j] = np.sin(c * (t_grid[i] - t_grid[j])) / (np.pi * (t_grid[i] - t_grid[j]))
    K_mat *= dt  # quadrature weight
    
    # Eigenvectors of the sinc kernel (prolate functions on the grid)
    mu, psi = np.linalg.eigh(K_mat)
    # Sort by eigenvalue (largest first = most concentrated)
    idx_sort = np.argsort(-mu)
    mu = mu[idx_sort]
    psi = psi[:, idx_sort]
    
    # Project the even prolate eigenfunctions onto the V_n basis
    # h_{n,lambda}(y) on the grid -> c_j = (1/L) sum h(t_i) exp(-2*pi*i*j*t_i/L) * dt
    prolate_Vn = []
    labels = []
    even_count = 0
    for p_idx in range(min(20, n_grid)):
        psi_p = psi[:, p_idx]
        # Check if even: psi(-t) = psi(t)
        psi_flip = psi_p[::-1]
        even_score = np.linalg.norm(psi_p - psi_flip)
        odd_score = np.linalg.norm(psi_p + psi_flip)
        is_even = even_score < odd_score
        
        if is_even:
            # Project onto V_n basis
            coeffs = np.zeros(dim)
            for j_idx in range(dim):
                j_val = j_idx - N
                # c_j = (1/L) integral h(t) exp(-2*pi*i*j*t/L) dt
                integrand = psi_p * np.exp(-2j * np.pi * j_val * t_grid / L_f)
                coeffs[j_idx] = np.real(np.sum(integrand) * dt / L_f)
            
            prolate_Vn.append(coeffs / np.linalg.norm(coeffs))
            labels.append(f"h_{2*even_count},lam (mu={mu[p_idx]:.6f})")
            even_count += 1
            if even_count >= 3:
                break
    
    return prolate_Vn, labels


print("ACTUAL PROLATE OVERLAP (Session 24)", flush=True)
print("=" * 70, flush=True)
print("", flush=True)

for lam_sq in [14, 50, 100, 200, 500, 1000]:
    t0 = time.time()
    xi, eps, L_f, N = build_xi(lam_sq, 30)
    dim = 2 * N + 1
    xi_norm = xi / np.linalg.norm(xi)
    
    # Compute actual prolate eigenfunctions
    prolates, labels = compute_prolate_eigenfunctions(L_f, N, n_grid=300)
    
    print(f"lam^2={lam_sq}, L={L_f:.3f}, eps={eps:.2e}", flush=True)
    
    for p_idx, (p_coeffs, label) in enumerate(zip(prolates, labels)):
        ov = abs(np.dot(xi_norm, p_coeffs))
        print(f"  overlap(xi, {label}) = {ov:.6f}", flush=True)
    
    # Build k_lambda: linear combination of h_0 and h_4 with integral vanishing
    # k = a*h_0 + b*h_4 such that integral k(u) du/u = 0
    # In V_n basis: sum(k_j) = 0 (since integral V_0 du/u = L, others average to 0)
    # So: a*h0[N] + b*h4[N] = 0 => b = -a*h0[N]/h4[N]
    if len(prolates) >= 2:
        h0 = prolates[0]
        h4 = prolates[1]  # second even = h_4
        if abs(h4[N]) > 1e-15:
            b_over_a = -h0[N] / h4[N]
            k_lambda = h0 + b_over_a * h4
            k_lambda = k_lambda / np.linalg.norm(k_lambda)
            ov_k = abs(np.dot(xi_norm, k_lambda))
            print(f"  overlap(xi, k_lambda) = {ov_k:.6f}  <-- THE KEY NUMBER", flush=True)
        else:
            print(f"  k_lambda: h4[0] too small, skipping", flush=True)
    
    print(f"  ({time.time()-t0:.0f}s)", flush=True)
    print("", flush=True)

print("If overlap(xi, k_lambda) -> 1: educated guess verified.", flush=True)
