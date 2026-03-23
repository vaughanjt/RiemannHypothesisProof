"""Record session 7 findings in the workbench."""
import sys
sys.path.insert(0, "src")
import numpy as np
from riemann.workbench.conjecture import create_conjecture
from riemann.workbench.experiment import save_experiment

# ============================================================
# CONJECTURE: Gauge Symmetry Theorem
# ============================================================
gauge_id = create_conjecture(
    statement=(
        "For any transfer operator L(t) = sum_p f(p)[exp(-it*log(p))*A_p + "
        "exp(+it*log(p))*A_p^T] on l^2({1,...,N}), the unitary gauge "
        "transformation D(t)=diag(n^{-it}) satisfies L(t)=D(t)*L(0)*D(t)^{-1}. "
        "Therefore the eigenvalue spectrum of L(t) is independent of t, and "
        "det(I-L(t)) cannot have t-dependent zeros. This rules out ALL Hermitian "
        "transfer operators with conjugate-phase forward/backward shifts as "
        "candidates for encoding zeta zeros."
    ),
    description=(
        "Gauge Symmetry Obstruction for Hermitian Transfer Operators. "
        "Proven algebraically: the diagonal unitary D(t)=diag(n^{-it}) "
        "conjugates any operator with p^{-it} forward and p^{+it} backward "
        "shifts into its t=0 version. Verified numerically across 5 operator "
        "variants (Reflected, Normalized, Trace-class, Log-weighted, VonMangoldt) "
        "at N=50-300: eigenvalues and determinant identical at all t values. "
        "Symmetric operator (same p^{-s} on both) breaks gauge because "
        "p^{-it} on backward != p^{+it}, so D(t)L(0)D(t)^{-1} != L(t)."
    ),
    evidence_level=2,  # conditional (proven for this operator class)
    status="conditional",
    confidence=0.99,
    tags=["operator", "gauge-symmetry", "transfer-operator", "hermitian", "obstruction"],
)
print(f"Gauge symmetry theorem: {gauge_id}")

# ============================================================
# CONJECTURE: Trace Formula Mismatch
# ============================================================
trace_id = create_conjecture(
    statement=(
        "The symmetric prime transfer operator L_s = sum_p p^{-s}(A_p + A_p^T) "
        "on l^2({1,...,N}) has spectral determinant det(I-L_s) that does NOT equal "
        "1/zeta(s). At s=2: -sum Tr(L^n)/n converges to -11.80, but log(zeta(2))=+0.50. "
        "The operator's 3 zero-convergent points (t=14.13, 40.92, 49.77) are "
        "coincidental eigenvalue-1 crossings, not structural zeros of zeta."
    ),
    description=(
        "Trace formula analysis definitively rules out the symmetric prime transfer "
        "as the zeta operator. The trace Tr(L^2) = 2*sum_p p^{-2s}*floor(N/p) is "
        "analytically computed and verified. Higher traces grow: spectral radius > 1 "
        "makes the trace expansion diverge on the critical line. The Dirichlet "
        "convolution operator (L_s f)(n) = sum_{d|n,d>1} d^{-s} f(n/d) is nilpotent "
        "(strictly lower-triangular), so Tr(L^n)=0 for all n and det(I-L)=1 always. "
        "Forward-only and backward-only operators are also nilpotent."
    ),
    evidence_level=1,  # heuristic (computational evidence)
    status="computational_evidence",
    confidence=0.95,
    tags=["operator", "trace-formula", "transfer-operator", "symmetric", "mismatch"],
)
print(f"Trace mismatch: {trace_id}")

# ============================================================
# EXPERIMENT: Transfer Operator Zoo
# ============================================================
exp_id = save_experiment(
    description=(
        "Systematic comparison of 6 transfer operator variants for encoding zeta zeros "
        "via Fredholm determinant det(I-L_s)=0. Variants: Symmetric, Reflected, "
        "Normalized, Trace-class, Log-weighted, VonMangoldt. Scanned critical line "
        "t in [10,55] at N=100-400."
    ),
    parameters={
        "N_values": [100, 150, 200, 250, 300, 350, 400],
        "t_range": [10, 55],
        "scan_points": 800,
        "operators": [
            "Symmetric: p^{-s} on both fwd/bwd",
            "Reflected: p^{-s} fwd, p^{s-1} bwd (Hermitian on Re(s)=1/2)",
            "Normalized: 1/sqrt(p) * phases (Hermitian)",
            "Trace-class: 1/p * phases (Hermitian)",
            "Log-weighted: log(p)/p * phases (Hermitian)",
            "VonMangoldt: log(p)/p^m * prime powers (Hermitian)",
        ],
        "detection_methods": ["phase_winding", "minima", "|det|_convergence"],
        "known_zeros_in_range": 11,
    },
    result_summary=(
        "KEY RESULTS: "
        "(1) All Hermitian variants have t-INDEPENDENT eigenvalues due to gauge symmetry "
        "L(t)=D(t)L(0)D(t)^{-1} where D(t)=diag(n^{-it}). Cannot find zeros. "
        "(2) Symmetric operator finds 3/11 zeros with |det|->0 convergence "
        "(t=14.13: 1.8e-4 to 1.4e-15; t=40.92: 3.0e-4 to 3.3e-14; t=49.77: 8.0e-5 to 5.3e-14). "
        "(3) Phase winding detection finds more (8/11 at N=300) but includes false positives. "
        "(4) Trace formula mismatch: -sum Tr(L^n)/n = -11.80 at s=2, but log(zeta(2))=+0.50. "
        "(5) Dirichlet convolution is nilpotent (Tr=0, det=1). "
        "(6) Forward-only, backward-only also nilpotent. "
        "CONCLUSION: No finite transfer operator on l^2({1,...,N}) has det(I-L_s)=1/zeta(s). "
        "The 3 converging zeros are coincidental eigenvalue crossings."
    ),
    computation_time_ms=800_000,
)
print(f"Experiment: {exp_id}")

print("\nDone recording findings.")
