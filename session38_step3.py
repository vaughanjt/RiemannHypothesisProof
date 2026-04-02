"""
SESSION 38 — TACKLING STEP 3: Off-diagonal decay and strong convergence

FIRST: A critical re-examination of the "tautology" claim.

Session 36 claimed Q_W >= 0 at finite lambda is a tautology. But IS it?
The explicit formula says:
  <phi, Q_W phi> = sum_rho g-hat(rho) * conj(g-hat(1-bar(rho)))

Under RH: each term is |g-hat(rho)|^2 >= 0. Sum is non-negative.
WITHOUT RH: some terms can be negative (off-line zeros give cross-products).

So Q_W >= 0 at finite lambda MIGHT require RH, not be a tautology!

The "exactly zero" eigenvalue (the silent modes) IS tautological — those modes
avoid all zeros, giving zero spectral sum regardless of zero locations.

But the SEEING modes (eigenvalue -4.8 to -0.1 for M) have positive Q_W
eigenvalues that MIGHT depend on the zeros being on the critical line.

TEST: Can we detect whether Q_W >= 0 is truly unconditional by checking
if it would survive a hypothetical off-line zero?

THEN: The actual Step 3 estimates in whatever eigenbasis is appropriate.
"""

import numpy as np
import mpmath
from mpmath import mp, mpf, mpc, log, pi, exp, cos, sin, euler, digamma, hyp2f1, sinh
import time
import sys
sys.path.insert(0, '.')
from connes_crossterm import build_all


def test_tautology_claim(lam_sq, N=None):
    """
    CRITICAL TEST: Is Q_W >= 0 truly unconditional (a tautology), or does it
    require the zeros to be on the critical line?

    Method: Inject a FAKE off-line zero into the spectral sum and see if
    Q_W becomes indefinite.

    If Q_W stays PSD even with a fake off-line zero: truly unconditional.
    If Q_W becomes indefinite: the positivity USES RH (not a tautology).

    The fake zero injection: add a rank-2 perturbation to M that simulates
    an off-line zero at rho = beta + i*gamma with beta != 1/2.
    """
    if N is None:
        L_f = np.log(lam_sq)
        N = max(15, round(6 * L_f))
    dim = 2 * N + 1
    L_f = np.log(lam_sq)
    ns = np.arange(-N, N + 1, dtype=float)

    W02, M, QW = build_all(lam_sq, N)

    ew, ev = np.linalg.eigh(W02)
    thresh = np.max(np.abs(ew)) * 1e-10
    P_null = ev[:, np.abs(ew) <= thresh]

    M_null = P_null.T @ M @ P_null
    evals_M = np.linalg.eigvalsh(M_null)
    QW_null = -M_null  # Q_W on null(W02)

    print(f"TAUTOLOGY TEST: lam^2={lam_sq}, dim={dim}", flush=True)
    print(f"  Q_W eigenvalues on null(W02): [{np.min(-evals_M):.4f}, {np.max(-evals_M):.6e}]", flush=True)
    print(f"  Q_W PSD: {np.min(-evals_M) > -1e-6}", flush=True)

    # The "seeing modes" are eigenvectors of M with genuinely negative eigenvalues.
    # Their Q_W eigenvalues are positive (e.g., +4.8).
    # If these positive values come from on-line zeros, adding an off-line zero
    # would REDUCE them (or flip them negative).

    # Simulate: compute the Mellin-like transform of each null eigenvector
    # at a hypothetical off-line zero rho = 0.6 + i*gamma (beta = 0.6, not 0.5)

    # The basis functions are omega_n(x) = 2(1-x/L)cos(2*pi*n*x/L) for x in [0,L]
    # The Mellin transform at s = sigma + i*t:
    # g-hat(s) = integral_0^L omega_n(x) * exp((s-1/2)*x) dx
    #          = integral_0^L 2(1-x/L)cos(2*pi*n*x/L) * exp((sigma-1/2)*x + i*t*x) dx

    def mellin_transform(phi_vec, sigma, t, L, ns):
        """Compute g-hat(sigma + i*t) for test function g = sum phi_n * omega_n."""
        # Numerical integration
        n_pts = 2000
        dx = L / n_pts
        result = 0.0 + 0.0j
        for k in range(n_pts):
            x = dx * (k + 0.5)
            # g(x) = sum_n phi_n * omega_n(x)
            g_x = 0.0
            for idx, n in enumerate(ns):
                g_x += phi_vec[idx] * 2 * (1 - x/L) * np.cos(2*np.pi*n*x/L)
            result += g_x * np.exp((sigma - 0.5) * x + 1j * t * x) * dx
        return result

    # For each null eigenvector, compute the spectral contribution
    # of a hypothetical off-line zero at rho = 0.6 + 14.13i
    # (just off the critical line, near the first real zero)
    beta_fake = 0.6  # Off-line: real part 0.6 instead of 0.5
    gamma_fake = 14.134725  # Imaginary part of first zeta zero

    print(f"\n  Hypothetical off-line zero at rho = {beta_fake} + {gamma_fake:.4f}i", flush=True)
    print(f"  (First zero shifted from Re=0.5 to Re=0.6)", flush=True)

    # The contribution of this zero pair (rho, 1-bar(rho)):
    # g-hat(rho) * conj(g-hat(1-bar(rho))) + g-hat(1-bar(rho)) * conj(g-hat(rho))
    # = 2 * Re[g-hat(rho) * conj(g-hat(1-bar(rho)))]
    # where 1-bar(rho) = 1 - beta + i*gamma = 0.4 + 14.13i

    # For on-line zero (beta=0.5): this is 2|g-hat(0.5+i*gamma)|^2 >= 0
    # For off-line (beta=0.6): this is 2*Re[g-hat(0.6+ig)*conj(g-hat(0.4+ig))]
    # which can be negative!

    # The perturbation from moving the zero off-line:
    # delta = 2*Re[g-hat(beta+ig)*conj(g-hat(1-beta+ig))] - 2|g-hat(0.5+ig)|^2
    # This is the change in the spectral contribution.

    # Compute for a few eigenvectors
    evals_null, evecs_null = np.linalg.eigh(M_null)

    print(f"\n  {'mode':>5} {'eig(M)':>10} {'eig(QW)':>10} {'|g(0.5+ig)|^2':>14} "
          f"{'Re[g(.6)*g(.4)]':>16} {'delta':>12} {'QW+delta':>10}", flush=True)

    for idx in range(min(8, len(evals_null))):
        phi_null = evecs_null[:, idx]
        phi_full = P_null @ phi_null  # back to full space

        # Compute Mellin transforms
        g_on = mellin_transform(phi_full, 0.5, gamma_fake, L_f, ns)
        g_off_rho = mellin_transform(phi_full, beta_fake, gamma_fake, L_f, ns)
        g_off_conj = mellin_transform(phi_full, 1 - beta_fake, gamma_fake, L_f, ns)

        on_line_contrib = 2 * abs(g_on)**2
        off_line_contrib = 2 * np.real(g_off_rho * np.conj(g_off_conj))
        delta = off_line_contrib - on_line_contrib
        qw_eig = -evals_null[idx]
        qw_perturbed = qw_eig + delta

        print(f"  {idx:>5} {evals_null[idx]:>+10.4f} {qw_eig:>+10.4f} {on_line_contrib:>14.6f} "
              f"{off_line_contrib:>+16.6f} {delta:>+12.4e} {qw_perturbed:>+10.4f}", flush=True)

    # Now test: what if we REPLACE the on-line contribution with off-line?
    # Build the perturbation matrix
    print(f"\n  Building off-line perturbation matrix...", flush=True)

    # For each pair (m,n) in null(W02), compute the Mellin transform
    # and build the rank-2 perturbation
    d_null = P_null.shape[1]
    g_on_vec = np.zeros(d_null, dtype=complex)
    g_off_rho_vec = np.zeros(d_null, dtype=complex)
    g_off_conj_vec = np.zeros(d_null, dtype=complex)

    for i in range(d_null):
        phi = P_null[:, i]
        g_on_vec[i] = mellin_transform(phi, 0.5, gamma_fake, L_f, ns)
        g_off_rho_vec[i] = mellin_transform(phi, beta_fake, gamma_fake, L_f, ns)
        g_off_conj_vec[i] = mellin_transform(phi, 1 - beta_fake, gamma_fake, L_f, ns)

    # On-line contribution matrix: g_on * g_on^H (rank 1, PSD)
    M_on = 2 * np.real(np.outer(g_on_vec, np.conj(g_on_vec)))

    # Off-line contribution: g_off_rho * g_off_conj^H + conj (rank 2, indefinite)
    M_off = np.real(np.outer(g_off_rho_vec, np.conj(g_off_conj_vec)) +
                    np.outer(g_off_conj_vec, np.conj(g_off_rho_vec)))

    # The perturbation: replace on-line with off-line
    Delta = M_off - M_on

    # Perturbed Q_W on null(W02)
    QW_perturbed = QW_null + Delta
    evals_perturbed = np.linalg.eigvalsh(QW_perturbed)

    print(f"\n  RESULT:", flush=True)
    print(f"  Original Q_W on null: [{np.min(-evals_M):.6f}, {np.max(-evals_M):.6e}]", flush=True)
    print(f"  Perturbed Q_W on null: [{np.min(evals_perturbed):.6f}, {np.max(evals_perturbed):.6f}]", flush=True)
    print(f"  Original PSD: {np.min(-evals_M) > -1e-6}", flush=True)
    print(f"  Perturbed PSD: {np.min(evals_perturbed) > -1e-6}", flush=True)

    if np.min(evals_perturbed) < -1e-6:
        print(f"\n  *** Q_W BECOMES INDEFINITE WITH OFF-LINE ZERO ***", flush=True)
        print(f"  *** THE 'TAUTOLOGY' CLAIM IS WRONG ***", flush=True)
        print(f"  *** Q_W >= 0 at finite lambda REQUIRES zeros on the critical line ***", flush=True)
        print(f"  *** (i.e., it requires verified RH at the relevant height) ***", flush=True)
    else:
        print(f"\n  Q_W stays PSD even with off-line zero.", flush=True)
        print(f"  The tautology claim survives this test.", flush=True)

    return evals_perturbed


def step3_decay_estimates(lam_sq, N=None):
    """
    The actual Step 3: measure the off-diagonal decay of M in the
    eigenbasis of M itself on null(W02).

    Since we don't have a clean "prolate basis," use the eigenbasis of
    the null-block M matrix. In this basis, M is diagonal. The question
    becomes: how does M_lambda change when lambda increases? Does the
    change (the "update" from new primes and kernel change) have
    controlled off-diagonal structure?
    """
    if N is None:
        L_f = np.log(lam_sq)
        N = max(15, round(6 * L_f))
    dim = 2 * N + 1

    W02, M, QW = build_all(lam_sq, N)

    ew, ev = np.linalg.eigh(W02)
    thresh = np.max(np.abs(ew)) * 1e-10
    P_null = ev[:, np.abs(ew) <= thresh]

    M_null = P_null.T @ M @ P_null
    d = M_null.shape[0]

    # Diagonalize M_null
    evals, evecs = np.linalg.eigh(M_null)

    print(f"\nSTEP 3: DECAY ESTIMATES at lam^2={lam_sq}", flush=True)
    print(f"  dim={dim}, null_dim={d}", flush=True)

    # Now compute M at a slightly larger lambda and express the CHANGE
    # in the M eigenbasis
    lam_sq2 = int(lam_sq * 1.1)  # 10% increase
    N2 = N  # Keep same N for comparison
    W02_2, M_2, QW_2 = build_all(lam_sq2, N2)

    ew2, ev2 = np.linalg.eigh(W02_2)
    thresh2 = np.max(np.abs(ew2)) * 1e-10
    P_null2 = ev2[:, np.abs(ew2) <= thresh2]

    M_null2 = P_null2.T @ M_2 @ P_null2

    # Express M_null2 in the eigenbasis of M_null
    # First, align the null spaces (they're similar but not identical)
    # Use the overlap matrix
    overlap = P_null.T @ P_null2  # dim_null1 x dim_null2
    # Project M_null2 onto the original null space basis
    M_null2_proj = overlap @ M_null2 @ overlap.T

    # Express in the eigenbasis of M_null
    Delta = evecs.T @ (M_null2_proj - M_null) @ evecs

    print(f"\n  Change Delta = M(1.1*lam^2) - M(lam^2) in M-eigenbasis:", flush=True)
    print(f"  ||Delta||_F = {np.linalg.norm(Delta, 'fro'):.6f}", flush=True)
    print(f"  ||Delta||_op = {np.linalg.norm(Delta, 2):.6f}", flush=True)

    # Off-diagonal decay
    print(f"\n  Off-diagonal structure of Delta:", flush=True)
    print(f"  {'|i-j|':>6} {'mean |Delta_ij|':>16} {'max |Delta_ij|':>16} {'count':>6}", flush=True)

    for dist in range(min(d, 20)):
        vals = []
        for i in range(d):
            for j in range(d):
                if abs(i - j) == dist:
                    vals.append(abs(Delta[i, j]))
        if vals:
            print(f"  {dist:>6} {np.mean(vals):>16.6e} {np.max(vals):>16.6e} {len(vals):>6}", flush=True)

    # The DIAGONAL of Delta tells us how eigenvalues shift
    print(f"\n  Diagonal of Delta (eigenvalue shifts):", flush=True)
    print(f"  {'i':>4} {'eig(M)':>10} {'shift':>12} {'new_eig':>12}", flush=True)
    for i in range(min(d, 15)):
        print(f"  {i:>4} {evals[i]:>+10.4f} {Delta[i,i]:>+12.6e} {evals[i]+Delta[i,i]:>+12.4f}", flush=True)
    if d > 15:
        print(f"  ...", flush=True)
        for i in range(max(d-5, 15), d):
            print(f"  {i:>4} {evals[i]:>+10.4f} {Delta[i,i]:>+12.6e} {evals[i]+Delta[i,i]:>+12.6e}", flush=True)


if __name__ == "__main__":
    print("SESSION 38 -- STEP 3 AND TAUTOLOGY RE-EXAMINATION", flush=True)
    print("=" * 80, flush=True)

    # CRITICAL: Test whether Q_W >= 0 is truly unconditional
    test_tautology_claim(50)

    # Step 3 decay estimates
    step3_decay_estimates(50)
    step3_decay_estimates(200)

    print(f"\nDone.", flush=True)
