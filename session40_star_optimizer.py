"""
SESSION 40b — ARITHMETIC HODGE STAR OPTIMIZER

Find J on the complex structure manifold O(2n)/U(n) such that [J, M] is
negative definite on the good subspace of null(W02).

Key identity: the Hodge-Riemann form h = sym(J*M) = [J, M] / 2
(because J^T = -J, M^T = M).

So RH (on this subspace, at this lambda^2) <==> all eigenvalues of [J, M] < 0.

Optimization problem:
    min  max_eigenvalue([J, M])
    s.t. J^T = -J,  J^2 = -I

The feasible set is the manifold of orthogonal complex structures,
dim = n(2n-1) - n^2 = n(n-1) for d=2n.  For d=44: 462 degrees of freedom.

We use Riemannian gradient descent on this manifold:
    - Tangent space at J: {dJ : dJ^T = -dJ, dJ*J + J*dJ = 0}
    - Retraction: Cayley transform or geodesic
    - Gradient: from the max eigenvector of [J, M]

Usage:
    python session40_star_optimizer.py [--lambda-sq 200] [--steps 2000]
"""

import numpy as np
from scipy.linalg import expm
import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from connes_crossterm import build_all


# ═══════════════════════════════════════════════════════════════
# MATRIX SETUP (reused from explorer)
# ═══════════════════════════════════════════════════════════════

def build_subspace(lam_sq):
    """Build M on the good subspace of null(W02). Returns (J0, M0, d)."""
    L_f = np.log(lam_sq)
    N = max(15, round(6 * L_f))
    dim = 2 * N + 1
    ns = np.arange(-N, N + 1, dtype=float)

    W02, M, QW = build_all(lam_sq, N, n_quad=10000)

    # Null(W02)
    ew, ev = np.linalg.eigh(W02)
    thresh = np.max(np.abs(ew)) * 1e-10
    P_null = ev[:, np.abs(ew) <= thresh]
    d_null = P_null.shape[1]
    M_null = P_null.T @ M @ P_null

    # Hilbert J
    J_full = np.zeros((dim, dim))
    idx = np.arange(dim)
    j_idx = dim - 1 - idx
    J_full[idx[ns > 0], j_idx[ns > 0]] = -1
    J_full[idx[ns < 0], j_idx[ns < 0]] = 1
    J_full = (J_full - J_full.T) / 2

    # Project to null, extract good subspace
    J_null = P_null.T @ J_full @ P_null
    J_sq = J_null @ J_null
    U, S, _ = np.linalg.svd(J_sq + np.eye(d_null))
    good = U[:, S < 0.1]
    d = good.shape[1]

    J0 = good.T @ J_null @ good
    M0 = good.T @ M_null @ good

    # Ensure d is even (required for complex structure)
    if d % 2 != 0:
        # Drop the last basis vector to make it even
        good = good[:, :-1]
        d = good.shape[1]
        J0 = good.T @ J_null @ good
        M0 = good.T @ M_null @ good

    return J0, M0, d, N, dim, d_null


# ═══════════════════════════════════════════════════════════════
# MANIFOLD OPERATIONS: Complex structures O(2n)/U(n)
# ═══════════════════════════════════════════════════════════════

def project_to_complex_structure(J):
    """Project an antisymmetric matrix to nearest J with J^2 = -I.

    Uses real Schur decomposition: for antisymmetric J, the Schur form
    has 2x2 blocks [[0, s], [-s, 0]]. Setting each s=1 gives J^2=-I.
    """
    from scipy.linalg import schur
    T, Q = schur(J, output='real')
    d = J.shape[0]
    for i in range(0, d - 1, 2):
        # Each 2x2 block: [[~0, s], [-s, ~0]] -> [[0, 1], [-1, 0]]
        T[i, i] = 0.0
        T[i+1, i+1] = 0.0
        # Preserve sign of the off-diagonal
        sign = 1.0 if T[i, i+1] >= 0 else -1.0
        T[i, i+1] = sign
        T[i+1, i] = -sign
    J_proj = Q @ T @ Q.T
    J_proj = (J_proj - J_proj.T) / 2  # enforce exact antisymmetry
    return J_proj


def project_tangent(dJ, J):
    """Project dJ onto tangent space of complex structure manifold at J.

    Tangent space: {dJ : dJ^T = -dJ, dJ*J + J*dJ = 0}
    The second condition means dJ anti-commutes with J.

    Given arbitrary antisymmetric dJ, project out the component that
    commutes with J:
        dJ_tangent = (dJ - J @ dJ @ J) / 2
    This satisfies: dJ_tangent * J + J * dJ_tangent = 0.
    """
    dJ_anti = (dJ - dJ.T) / 2  # ensure antisymmetric
    # Project: remove the part that commutes with J
    dJ_tan = (dJ_anti - J @ dJ_anti @ J) / 2
    return dJ_tan


def retract(J, dJ, step):
    """Retract along tangent direction via exponential map.

    J_new = exp(s*A) @ J @ exp(-s*A)  where A = -dJ @ J (skew-symmetric).

    Preserves J^2 = -I and J^T = -J exactly:
      J_new^2 = R J R^T R J R^T = R J^2 R^T = -I
      J_new^T = R^{-T} J^T R^T = -R^{-T} J R^T = -(R J R^T)... = -J_new
    """
    A = -dJ @ J  # skew-symmetric generator
    R = expm(step * A)
    J_new = R @ J @ R.T
    J_new = (J_new - J_new.T) / 2  # numerical antisymmetry
    return J_new


# ═══════════════════════════════════════════════════════════════
# LOSS AND GRADIENT
# ═══════════════════════════════════════════════════════════════

def loss_and_grad(J, M):
    """Compute max eigenvalue of [J, M] and its Riemannian gradient.

    Loss: L(J) = lambda_max([J, M])
    We want L < 0 (all eigenvalues of [J,M] negative).

    Gradient: dL/dJ projected onto the tangent space at J.
    """
    comm = J @ M - M @ J  # [J, M], symmetric since J^T=-J, M^T=M
    evals, evecs = np.linalg.eigh(comm)

    lam_max = evals[-1]
    v = evecs[:, -1]  # eigenvector for max eigenvalue

    # Gradient of lambda_max w.r.t. J:
    # d(v^T [J,M] v)/dJ = d(v^T JM v - v^T MJ v)/dJ
    # = v (Mv)^T - (Mv) v^T
    # (derivative of v^T J w w.r.t. J is v w^T)
    Mv = M @ v
    grad_J = np.outer(v, Mv) - np.outer(Mv, v)  # antisymmetric automatically

    # Project to tangent space of manifold at J
    grad_tan = project_tangent(grad_J, J)

    return lam_max, evals, grad_tan


def loss_softmax(J, M, temperature=0.1):
    """Smooth approximation to max eigenvalue using log-sum-exp.

    L_soft = T * log(sum exp(lambda_i / T))
    Gradient is softmax-weighted sum of per-eigenvalue gradients.
    """
    comm = J @ M - M @ J
    evals, evecs = np.linalg.eigh(comm)

    shifted = evals / temperature
    max_shifted = np.max(shifted)
    lse = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    loss = temperature * lse

    weights = np.exp(shifted - max_shifted)
    weights /= np.sum(weights)

    grad_J = np.zeros_like(J)
    for i, w in enumerate(weights):
        if w > 1e-10:
            v = evecs[:, i]
            Mv = M @ v
            grad_J += w * (np.outer(v, Mv) - np.outer(Mv, v))

    grad_tan = project_tangent(grad_J, J)
    return loss, evals, grad_tan


def loss_sum_positive(J, M):
    """Sum of positive eigenvalues of [J, M].

    L = sum max(0, lambda_i)

    Penalizes ALL positive eigenvalues, not just the largest.
    Gradient: sum over positive eigenvectors.
    """
    comm = J @ M - M @ J
    evals, evecs = np.linalg.eigh(comm)

    pos_mask = evals > 0
    loss = np.sum(evals[pos_mask])

    grad_J = np.zeros_like(J)
    for i in np.where(pos_mask)[0]:
        v = evecs[:, i]
        Mv = M @ v
        grad_J += np.outer(v, Mv) - np.outer(Mv, v)

    grad_tan = project_tangent(grad_J, J)
    return loss, evals, grad_tan


# ═══════════════════════════════════════════════════════════════
# OPTIMIZER
# ═══════════════════════════════════════════════════════════════

def optimize_star(J0, M, n_steps=2000, lr=0.01, report_every=100,
                  loss_type='sum_positive', temperature=0.1, patience=300):
    """Riemannian gradient descent to find J with [J,M] negative definite.

    loss_type: 'sum_positive' (penalizes all positive evals),
               'softmax' (smooth max), 'max' (hard max eigenvalue)
    """
    d = J0.shape[0]
    J = project_to_complex_structure(J0)

    best_lam_max = np.inf
    best_J = J.copy()
    best_evals = None
    no_improve = 0
    current_lr = lr

    history = {'step': [], 'loss': [], 'lam_max': [], 'lam_min': [],
               'n_pos': [], 'n_neg': [], 'j2_err': [], 'lr': []}

    for step in range(n_steps):
        if loss_type == 'sum_positive':
            loss, evals, grad = loss_sum_positive(J, M)
        elif loss_type == 'softmax':
            loss, evals, grad = loss_softmax(J, M, temperature)
        else:
            loss, evals, grad = loss_and_grad(J, M)

        lam_max = evals[-1]
        lam_min = evals[0]
        n_pos = int(np.sum(evals > 1e-8))
        n_neg = int(np.sum(evals < -1e-8))

        if lam_max < best_lam_max:
            best_lam_max = lam_max
            best_J = J.copy()
            best_evals = evals.copy()
            no_improve = 0
        else:
            no_improve += 1

        j2_err = np.linalg.norm(J @ J + np.eye(d), 'fro') / d
        history['step'].append(step)
        history['loss'].append(float(loss))
        history['lam_max'].append(float(lam_max))
        history['lam_min'].append(float(lam_min))
        history['n_pos'].append(n_pos)
        history['n_neg'].append(n_neg)
        history['j2_err'].append(float(j2_err))
        history['lr'].append(float(current_lr))

        if step % report_every == 0 or step == n_steps - 1:
            marker = " ***" if n_pos == 0 else ""
            print(f"  [{step:5d}] loss={loss:+.6f}  lam_max={lam_max:+.6f}  "
                  f"+:{n_pos} -:{n_neg}  "
                  f"J^2={j2_err:.1e}  lr={current_lr:.4f}{marker}", flush=True)

        if lam_max < -1e-6:
            print(f"\n  *** DEFINITE at step {step}! "
                  f"lam_max = {lam_max:.6e} ***\n", flush=True)
            break

        # Reduce lr on plateau
        if no_improve >= patience:
            current_lr *= 0.5
            no_improve = 0
            if current_lr < 1e-8:
                print(f"  Learning rate bottomed out at step {step}", flush=True)
                break

        grad_norm = np.linalg.norm(grad, 'fro')
        if grad_norm < 1e-12:
            print(f"  Gradient vanished at step {step}", flush=True)
            break

        dJ = -current_lr * grad / grad_norm
        J = retract(J, dJ, 1.0)

    for k in history:
        history[k] = np.array(history[k])
    return best_J, best_evals, history


# ═══════════════════════════════════════════════════════════════
# RANDOM RESTARTS
# ═══════════════════════════════════════════════════════════════

def random_complex_structure(d):
    """Generate a random J with J^2 = -I, J^T = -J.

    Method: take a random antisymmetric matrix, project to
    the complex structure manifold.
    """
    A = np.random.randn(d, d)
    A = (A - A.T) / 2  # antisymmetric
    return project_to_complex_structure(A)


def multi_start_optimize(M, d, n_starts=5, n_steps=2000, lr=0.01,
                         start_from_hilbert=None):
    """Run optimizer from multiple starting points."""
    results = []

    for i in range(n_starts):
        if i == 0 and start_from_hilbert is not None:
            J_init = start_from_hilbert.copy()
            label = "Hilbert"
        else:
            J_init = random_complex_structure(d)
            label = f"Random-{i}"

        print(f"\n{'='*60}", flush=True)
        print(f"  START {i+1}/{n_starts}: {label}", flush=True)
        print(f"{'='*60}", flush=True)

        J_best, evals, hist = optimize_star(
            J_init, M, n_steps=n_steps, lr=lr, report_every=200,
            loss_type='sum_positive')

        lam_max = evals[-1]
        n_pos = int(np.sum(evals > 1e-8))
        results.append({
            'label': label, 'J': J_best, 'evals': evals,
            'lam_max': lam_max, 'n_pos': n_pos, 'history': hist,
        })
        print(f"  Result: lam_max={lam_max:+.6f}  positive_evals={n_pos}", flush=True)

    results.sort(key=lambda r: r['lam_max'])
    return results


# ═══════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_result(J, M, J_hilbert):
    """Analyze the optimized J: what structure does it have?"""
    d = J.shape[0]
    comm = J @ M - M @ J
    evals_comm = np.linalg.eigvalsh(comm)

    print(f"\n  === ANALYSIS OF OPTIMIZED J ===", flush=True)
    print(f"  Dimension: {d}x{d}", flush=True)

    # J^2 quality
    j2_err = np.linalg.norm(J @ J + np.eye(d), 'fro') / d
    print(f"  J^2 + I error: {j2_err:.2e}", flush=True)

    # [J,M] spectrum
    print(f"  [J,M] eigenvalues:", flush=True)
    print(f"    min = {evals_comm[0]:+.6f}", flush=True)
    print(f"    max = {evals_comm[-1]:+.6f}", flush=True)
    print(f"    positive: {np.sum(evals_comm > 1e-8)}", flush=True)
    print(f"    negative: {np.sum(evals_comm < -1e-8)}", flush=True)
    print(f"    near-zero: {np.sum(np.abs(evals_comm) <= 1e-8)}", flush=True)

    # Distance from Hilbert
    J_h = project_to_complex_structure(J_hilbert)
    dist = np.linalg.norm(J - J_h, 'fro') / np.linalg.norm(J_h, 'fro')
    print(f"  Distance from Hilbert: {dist:.4f} (relative Frobenius)", flush=True)

    # How much does J commute with M vs anti-commute?
    comm_n = np.linalg.norm(J @ M - M @ J, 'fro')
    anti_n = np.linalg.norm(J @ M + M @ J, 'fro')
    print(f"  ||[J,M]|| = {comm_n:.4f}", flush=True)
    print(f"  ||{{J,M}}|| = {anti_n:.4f}", flush=True)
    print(f"  Comm ratio: {comm_n/(comm_n+anti_n)*100:.1f}%", flush=True)

    # Eigenvalues of J (should be +/- i)
    evals_J = np.linalg.eigvals(J)
    imag_parts = np.sort(evals_J.imag)
    print(f"  J eigenvalues (imag): [{imag_parts[0]:.3f} ... {imag_parts[-1]:.3f}]",
          flush=True)

    # Frobenius inner product with Hilbert (alignment)
    alignment = np.trace(J.T @ J_h) / (np.linalg.norm(J, 'fro') *
                                         np.linalg.norm(J_h, 'fro'))
    print(f"  Alignment with Hilbert: {alignment:.4f} (1.0 = identical)", flush=True)

    return evals_comm


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hodge Star Optimizer')
    parser.add_argument('--lambda-sq', type=int, default=200)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--starts', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    print(f"\n{'#'*60}", flush=True)
    print(f"  ARITHMETIC HODGE STAR OPTIMIZER", flush=True)
    print(f"  lam^2={args.lambda_sq}  steps={args.steps}  "
          f"starts={args.starts}  lr={args.lr}", flush=True)
    print(f"{'#'*60}\n", flush=True)

    # Build matrices
    print("  Building matrices...", flush=True)
    t0 = time.time()
    J0, M0, d, N, dim, d_null = build_subspace(args.lambda_sq)
    print(f"  Built in {time.time()-t0:.1f}s: dim={dim}, null={d_null}, "
          f"good={d}", flush=True)

    # Check starting point
    comm0 = J0 @ M0 - M0 @ J0
    evals0 = np.linalg.eigvalsh(comm0)
    print(f"\n  Hilbert baseline:", flush=True)
    print(f"    [J_h, M] eigenvalues: [{evals0[0]:+.4f} ... {evals0[-1]:+.4f}]",
          flush=True)
    print(f"    Positive: {np.sum(evals0 > 1e-8)}, "
          f"Negative: {np.sum(evals0 < -1e-8)}", flush=True)

    # M eigenvalues (context)
    evals_M = np.linalg.eigvalsh(M0)
    print(f"  M eigenvalues: [{evals_M[0]:+.4f} ... {evals_M[-1]:+.4f}]", flush=True)
    print(f"    M negative: {np.sum(evals_M < -1e-8)}, "
          f"M near-zero: {np.sum(np.abs(evals_M) <= 1e-8)}", flush=True)

    # Manifold dimension
    n = d // 2
    manifold_dim = n * (n - 1)
    print(f"\n  Complex structure manifold: O({d})/U({n}), "
          f"dim = {manifold_dim}", flush=True)
    print(f"  Question: does J exist with [J,M] < 0?\n", flush=True)

    # Optimize
    results = multi_start_optimize(
        M0, d, n_starts=args.starts, n_steps=args.steps, lr=args.lr,
        start_from_hilbert=J0)

    # Summary
    print(f"\n{'#'*60}", flush=True)
    print(f"  SUMMARY", flush=True)
    print(f"{'#'*60}\n", flush=True)

    for r in results:
        marker = "DEFINITE!" if r['n_pos'] == 0 else f"{r['n_pos']} positive"
        print(f"  {r['label']:12s}: lam_max = {r['lam_max']:+.6f}  ({marker})",
              flush=True)

    # Analyze best
    best = results[0]
    print(f"\n  Best result: {best['label']}", flush=True)
    analyze_result(best['J'], M0, J0)

    # The big question
    if best['n_pos'] == 0:
        print(f"\n  {'='*60}", flush=True)
        print(f"  DEFINITE [J,M] FOUND.", flush=True)
        print(f"  All eigenvalues of [J,M] are negative.", flush=True)
        print(f"  This means: at lam^2={args.lambda_sq}, there EXISTS a", flush=True)
        print(f"  complex structure J making the Hodge-Riemann form definite.",
              flush=True)
        print(f"  {'='*60}", flush=True)
        print(f"\n  Next: test at other lambda^2 values.", flush=True)
        print(f"  If it works everywhere, reverse-engineer the geometry.", flush=True)
    else:
        print(f"\n  No definite solution found in {args.starts} starts.", flush=True)
        print(f"  Best lam_max = {best['lam_max']:+.6f} "
              f"({best['n_pos']} positive eigenvalues remain)", flush=True)
        print(f"  Options:", flush=True)
        print(f"    - More starts (--starts 20)", flush=True)
        print(f"    - More steps (--steps 5000)", flush=True)
        print(f"    - Lower learning rate (--lr 0.001)", flush=True)
        print(f"    - Or: the obstruction may be real.", flush=True)
