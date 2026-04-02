"""Quick verification: compare exact vs approximate at lam^2=50."""
import numpy as np
import sys, time
sys.path.insert(0, '.')
from connes_crossterm import build_all

print("VERIFICATION: exact mpmath at lam^2=50", flush=True)

for N in [8, 12, 15, 18, 21, 23]:
    t0 = time.time()
    W02, M, QW = build_all(50, N)
    dim = 2*N+1

    ew, ev = np.linalg.eigh(W02)
    thresh = np.max(np.abs(ew)) * 1e-10
    P_null = ev[:, np.abs(ew) <= thresh]
    M_null = P_null.T @ M @ P_null
    evals = np.linalg.eigvalsh(M_null)
    max_ev = np.max(evals)
    elapsed = time.time() - t0

    print(f"  N={N:>2} dim={dim:>3} null_dim={P_null.shape[1]:>3} "
          f"max_eig(M|null)={max_ev:>+14.6e} QW>0={'YES' if max_ev < 1e-6 else 'NO':>3} "
          f"({elapsed:.1f}s)", flush=True)

print("\nNow sweep lam^2 with N=15 (exact mpmath):", flush=True)
for lam_sq in [4, 5, 7, 10, 15, 20, 30, 50, 100, 200, 500]:
    L_f = np.log(lam_sq)
    N = max(15, round(6*L_f))
    t0 = time.time()
    W02, M, QW = build_all(lam_sq, N)
    dim = 2*N+1

    ew, ev = np.linalg.eigh(W02)
    thresh = np.max(np.abs(ew)) * 1e-10
    P_null = ev[:, np.abs(ew) <= thresh]
    M_null = P_null.T @ M @ P_null
    evals = np.linalg.eigvalsh(M_null)
    max_ev = np.max(evals)
    elapsed = time.time() - t0

    flag = "" if max_ev < 1e-6 else " ***FAIL***"
    print(f"  lam^2={lam_sq:>5} N={N:>2} dim={dim:>3} #null={P_null.shape[1]:>3} "
          f"max_eig(M|null)={max_ev:>+14.6e} ({elapsed:.1f}s){flag}", flush=True)
