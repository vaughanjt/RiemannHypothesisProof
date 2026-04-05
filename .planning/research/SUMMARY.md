# Project Research Summary

**Project:** Riemann v2.0 — The Modular Barrier
**Domain:** Computational number theory / heat kernel proof strategy for the Riemann Hypothesis
**Researched:** 2026-04-04
**Confidence:** MEDIUM

## Executive Summary

The v2.0 milestone attempts to prove B(L) > 0 for all L by connecting the Connes barrier to the heat kernel trace on the modular surface SL(2,Z)\H. The core hypothesis, crystallized in Session 47, is that the barrier is "morally" a heat kernel trace — and since each term in the heat kernel spectral expansion is non-negative (each Maass form contribution is |phi_j(z)|^2 * e^{-lambda_j*t} >= 0), positivity might follow structurally rather than through direct analytic bounds. This would circumvent the circularity trap that destroyed 40+ prior approaches: every direct analytic decomposition of B(L) eventually requires bounding a sum over primes or zeros in a way that is equivalent to the statement being proved. The modular surface approach replaces "prove the balance" with "identify B(L) as a manifestly-positive object minus a bounded correction."

The recommended implementation builds six new Python modules in the existing `src/riemann/analysis/` hierarchy, adding only one new dependency (python-flint 0.8.0 for certified modular form arithmetic). The mathematical chain has three links that must be addressed in strict order: (1) establish the precise decomposition B(L) = K(t(L)) - epsilon(t,L) with all terms computed independently, including the non-trivial continuous spectrum (Eisenstein series) contribution; (2) bound the correction epsilon rigorously using explicit constants, not asymptotic estimates — the margin-drain gap is only ~0.036 and any O() term without an explicit constant will destroy the proof; (3) conclude B(L) > 0 from heat kernel positivity plus bounded correction. The Rankin-Selberg alternative — identifying B(L) = L(1, f x f_bar) for some Maass form f, making positivity automatic from Petersson norm — should run in parallel as a fallback path.

The fundamental risk is circularity in link (2). The scattering determinant phi(s) = Gamma(s-1/2) * zeta(2s-1) / (Gamma(s) * zeta(2s)) appears directly in the Eisenstein continuous spectrum integral, meaning zeta zeros enter the computation through the back door. If bounding epsilon rigorously requires controlling phi'/phi in a way that encodes zero-location information, the proof reduces to yet another reformulation of RH. This risk must be explicitly assessed at the end of Phase 1 — before any correction-bound work begins — by writing the complete logical dependency graph and testing whether the bound would still hold for "random primes" (a test that killed five approaches in Sessions 35–42).

## Key Findings

### Recommended Stack

The existing stack (mpmath, numpy, scipy, sympy, gmpy2) already provides everything needed for heat kernel evaluation, Selberg trace formula computation, and Rankin-Selberg L-function series. The sole new dependency is python-flint 0.8.0, which wraps FLINT/Arb ball arithmetic and returns interval-valued modular form evaluations with machine-certified error bounds. This is required for the correction bounds phase, where floating-point approximations are insufficient: the total error budget is 0.036 and any unverified rounding error can exhaust it. Windows x86-64 wheels are on PyPI; installation is `pip install python-flint>=0.7.0`.

**Core technologies:**

- **mpmath (>=1.3.0):** Arbitrary-precision special functions (besselk, legenp/q, gammainc, quad, nsum, pslq, findpoly, identify) — handles all heat kernel orbital integrals, CM point algebraic recognition, and Rankin-Selberg Dirichlet series
- **python-flint 0.8.0 (NEW):** Ball arithmetic for certified modular form evaluation — acb.modular_j, acb.eisenstein, acb.modular_eta, acb.modular_delta with rigorous interval bounds; use mpmath for exploration, python-flint for certification
- **scipy.special:** Machine-precision Bessel K functions (kv) for bulk heat kernel parameter sweeps where 50-digit precision is not needed
- **numpy/scipy.linalg:** Matrix operations and eigenvalue storage for Selberg trace formula spectral data
- **Existing lmfdb_client.py:** Pre-computed Maass form spectral parameters (r_j values, first 100+) from LMFDB — do not reimplement Hejhal's algorithm; LMFDB has eigenvalues to 30+ digits

SageMath is explicitly rejected: 2+ GB install, incompatible with the project's pip/venv setup, and all needed functions (j-invariant, Eisenstein series, eta function) are available in python-flint at a fraction of the footprint.

### Expected Features

The research identifies six table-stakes features (T1–T6) and six differentiating features (D1–D6). The critical path is T2 + T3 -> T1 -> T4 -> T5 -> proof. The bottleneck is T3 (Eisenstein continuous spectrum): the non-compact surface SL(2,Z)\H has both a discrete spectrum (Maass cusp forms) and a continuous spectrum starting at lambda = 1/4, and both must be included in the heat kernel trace. Ignoring the continuous spectrum is never acceptable, even for exploratory computation.

**Must have (table stakes):**

- T1: Heat kernel trace on SL(2,Z)\H — discrete Maass form sum plus continuous Eisenstein integral; neither alone gives correct values
- T2: Maass form eigenvalue database — first 100+ eigenvalues from LMFDB (r_1 ~ 9.534, r_2 ~ 12.173); stored in data/maass_forms.json
- T3: Eisenstein continuous spectrum contribution — integral of exp(-(1/4+r^2)*t) * (-phi'/phi)(1/2+ir) dr; involves zeta'/zeta explicitly; this is the highest-risk component
- T4: Barrier-to-heat-kernel comparison engine — parameter identification t = t(L); four candidate mappings exist, none confirmed; Phase 1 must resolve this
- T5: Correction bound computation — bound |K(t) - B(L)|; error budget is 0.036 total; explicit constants required throughout
- T6: Selberg trace formula implementation — both spectral and geometric sides for SL(2,Z) with test function h(r) = 1/(L^2+r^2)

**Should have (differentiators):**

- D1: GL(1)->GL(2) lift — translates the Weil explicit formula (GL(1) barrier) into the Selberg trace formula (GL(2) heat kernel) framework; introduces identity, hyperbolic, elliptic, and parabolic geometric contributions that must all be computed
- D2: Rankin-Selberg L-value check — parallel proof path: if B(L) = L(1, f x f_bar) for some Maass form f, positivity is automatic from Petersson norm; check L(1, f x f_bar) for low-lying Maass forms
- D6: Spectral-geometric duality verification — both sides of the Selberg trace formula agree to 10+ digits at 20+ parameter values; a correctness check, not a proof

**Defer (post-v2.0):**

- D3: q-series fitting — diagnostic only; pursue only if D2 Rankin-Selberg check finds a partial match
- D4: CM point evaluation — speculative; only worth pursuing after modular form connection is established by D3
- D5: Laplacian eigenvalue computation from scratch — avoid reimplementing Hejhal's algorithm; use LMFDB data

**Anti-features to avoid:**

- Direct analytic proof of B(L) > 0 — proven circular in 40+ sessions; Session 42 audit documented this exhaustively
- Full Hejhal algorithm implementation — use LMFDB data; Hejhal is a research-grade eigenvalue solver requiring careful K-Bessel evaluation at large arguments
- Lean 4 formalization before numerical validation — the heat kernel interpretation is still a hypothesis; formalizing it prematurely wastes effort
- Symbolic heat kernel manipulation via sympy — integrals involve delicate special function cancellations; numerical-first is the only viable strategy

### Architecture Approach

All new v2.0 code lives in `src/riemann/analysis/` as six new flat modules, following the existing function-based, dataclass-returning pattern established in v1.0. No new packages, no new abstractions — extend the existing structure. The modules form a strict dependency DAG: heat_kernel.py and selberg_trace.py are independent foundations; rankin_selberg.py and cm_evaluation.py consume heat_kernel.py; barrier_bridge.py consumes both foundation modules; proof_assembly.py is the terminal orchestrator.

**Major components:**

1. **heat_kernel.py** — Compute K_t(z,z) on SL(2,Z)\H: constant term + Maass cusp form spectral sum + Eisenstein continuous integral; returns `HeatKernelResult`; depends only on existing modular_forms.py and data/maass_forms.json; no v2.0 dependencies
2. **selberg_trace.py** — Selberg trace formula (spectral side vs. all four geometric contributions) for test function h(r) = 1/(L^2+r^2); GL(1)->GL(2) lift of the Weil explicit formula; returns `SelbergTraceResult`; depends only on existing trace_formula.py and engine/zeros.py
3. **rankin_selberg.py** — Compute L(s, f x f_bar) for cusp forms via Hecke eigenvalue Dirichlet series; check Petersson norm identity at s=1 (L(1,fxf) = (4pi)^k * ||f||^2 / (k-1)!); returns `RankinSelbergResult`
4. **cm_evaluation.py** — Evaluate barrier and heat kernel at the nine Heegner CM points using python-flint for certified j-invariant values; algebraic recognition via mpmath.findpoly; returns `CMEvaluationResult`; depends on heat_kernel.py
5. **barrier_bridge.py** — Bound |K_t(trace) - B(L)|; the make-or-break correction between heat kernel trace and actual barrier; uses validated_computation with P-vs-2P precision validation throughout; returns `CorrectionBoundResult` with `proof_viable` flag; depends on heat_kernel.py and selberg_trace.py
6. **proof_assembly.py** — Orchestrate the full proof chain; register three conjectures in the workbench (heat positivity, correction bound, barrier positivity), link experiments as evidence, trigger Lean 4 generation only after evidence_level >= 2

**Key patterns throughout:** validated_computation with P-vs-2P validation for all spectral sums; convergence monitoring on every truncated infinite sum (tail_variation metric); heat_trace and correction_bound computed at matching precision; Lean 4 generation gated behind numerical evidence.

### Critical Pitfalls

The pitfalls research distills 40+ sessions of failed approaches and literature review into 13 pitfalls. These five are most likely to invalidate v2.0:

1. **Circularity in correction bounds (CRITICAL)** — The heat kernel trace is trivially positive; the barrier is not the heat kernel trace; the gap between them is the entire mathematical content of the proof; bounding that gap likely requires controlling prime distribution information that is equivalent to zero-location information. Prevention: write the complete logical dependency graph before writing correction-bound code; apply the "random primes" test (would the bound still hold if primes were replaced by random integers of the same density?); ban big-O notation — every bound must have an explicit constant.

2. **Continuous spectrum trap (CRITICAL)** — The scattering matrix determinant phi(s) for SL(2,Z) contains zeta(s) explicitly; the Eisenstein integral involves phi'/phi, which involves zeta'/zeta; the zeros of zeta appear directly in the continuous spectrum contribution; this contribution has no definite sign and may individually exceed the 0.036 error budget. Prevention: compute the Eisenstein contribution numerically at every L value of interest in Phase 1; if its magnitude exceeds 0.036, the "manifestly positive heat kernel" argument is vacuous.

3. **Heat kernel vs. barrier confusion** — The connection between B(L) and K(t) is "moral" (Session 47's word), not a proven identity; the exact equation B(L) = Z(L) + E(L) + R(L) with explicitly computed Z (discrete spectrum), E (Eisenstein integral), and R (GL(1)->GL(2) lift remainder) must be derived before any positivity argument. Prevention: compute each term independently; if E(L) is comparable in magnitude to B(L), the approach is not valid.

4. **GL(1)->GL(2) lift errors** — The lift introduces four geometric contributions (identity term proportional to the fundamental domain area, hyperbolic term summing over prime geodesics with norms log(N(gamma)) — NOT log(p) for rational primes, elliptic term from fixed points at i and rho, parabolic term from the cusp); failing to compute all four and treating the lift as a simple identity will produce incorrect bounds. Prevention: compute all four contributions numerically; compare to barrier; read Wong's CUNY thesis on the precise connection.

5. **Insufficient rigor in bounds (the 0.036 problem)** — The margin-drain gap is ~0.036; any O() notation without explicit constants, any tail bound verified numerically but not proved analytically, or any asymptotic estimate valid only for L >= L_0 without handling smaller L will fail to close the proof. Prevention: establish an error budget document before Phase 3; allocate the 0.036 budget across independent error terms explicitly; use python-flint interval arithmetic for all bounds that feed into proof_assembly.py.

## Implications for Roadmap

The mathematical dependencies are strict and dictate phase order: you cannot bound corrections without first knowing what the corrections are, and you cannot know the corrections without establishing the parameter mapping t(L) and computing the continuous spectrum contribution. The circularity risk assessment must happen at the boundary between Phase 1 and Phase 3 — this is a logical gate, not just a schedule checkpoint.

### Phase 1: Heat Kernel Interpretation and Parameter Identification

**Rationale:** The foundational hypothesis has not been numerically confirmed with full rigor. Phase 1 answers: "Is B(L) approximately K(t(L))? What is t(L)? How large is the continuous spectrum contribution?" Without these answers, nothing downstream has direction.

**Delivers:** A numerically confirmed decomposition B(L) = K(t(L)) + epsilon(L) with independent computation of all three terms (discrete Maass sum, Eisenstein continuous integral, and their sum vs. the barrier); the parameter mapping t = t(L) identified to 6+ significant figures at 50+ L values; a characterization of the Eisenstein contribution's magnitude relative to the 0.036 threshold; a three-regime map (small L, medium L, large L) showing different dominant behaviors.

**Addresses features:** T2 (Maass eigenvalue database), T3 (Eisenstein continuous spectrum), T1 (heat kernel trace assembly), T4 (barrier comparison)

**Avoids:** Pitfall 2 (heat kernel vs. barrier confusion — exact equation written term-by-term before any claim), Pitfall 3 (continuous spectrum never ignored), Pitfall 10 (three-regime map established here)

**Implements:** heat_kernel.py (no v2.0 dependencies, the mathematical foundation)

**Exit criterion:** K(t(L)) and B(L) agree to 6+ significant figures for 50+ values of L; Eisenstein contribution computed and compared to 0.036 threshold; if Eisenstein contribution exceeds 0.036 at any L, Phase 3 approach must be redesigned before proceeding.

**Standard patterns:** LMFDB data fetching, Maass eigenvalue loading, heat kernel special functions (besselk, incomplete gamma, quad) all follow established patterns; no additional research needed for infrastructure.

### Phase 2: Selberg Trace Formula and GL(1)->GL(2) Bridge

**Rationale:** The Selberg trace formula provides the theoretical grounding for why K(t) relates to B(L) and decomposes the correction into identifiable geometric terms. It must follow Phase 1 (the parameter mapping is needed to specify the test function) and must precede Phase 3 (geometric decomposition reveals which correction terms need bounding).

**Delivers:** A verified numerical implementation of both sides of the Selberg trace formula for h(r) = 1/(L^2+r^2); decomposition of epsilon(L) into identity, hyperbolic, elliptic, and parabolic contributions with magnitudes established numerically; spectral-geometric duality verified to 10+ digits at 20+ parameter values; explicit confirmation that the Lorentzian test function satisfies admissibility conditions (or identification of a smoothed variant that does).

**Addresses features:** T6 (Selberg trace formula), D1 (GL(1)->GL(2) lift), D6 (spectral-geometric verification)

**Avoids:** Pitfall 4 (all four geometric contributions computed), Pitfall 8 (test function decay rate verified — the Lorentzian's r^{-2} decay is borderline and may require a smooth cutoff), Pitfall 11 (Maass forms vs. holomorphic modular forms kept strictly distinct)

**Implements:** selberg_trace.py

**Needs research:** The admissibility of the Lorentzian test function under the Selberg trace formula's decay condition (borderline r^{-2} case); whether a smooth cutoff is required and how it modifies both sides of the formula. Wong's CUNY thesis ("Explicit Formulae and Trace Formulae") is the primary reference for the GL(1)->GL(2) connection and should be read carefully before implementation.

### Phase 3: Correction Bound Estimation and Circularity Gate

**Rationale:** This is the make-or-break phase. The circularity risk must be assessed explicitly before any bounding code is written — this is a logical gate, not just documentation. Phase 3 runs two parallel tracks: correction bounds (T5) and Rankin-Selberg identification (D2) as a fallback.

**Delivers:** Either (a) a rigorous bound |epsilon(L)| < K(t(L)) - delta for some delta > 0, proven with explicit constants and no circularity — establishing B(L) > 0; or (b) an identification B(L) = L(1, f x f_bar) for a specific Maass form f, with positivity from the Petersson norm; or (c) a documented obstruction stating exactly which step requires unconditional zero-location information and why it cannot be avoided.

**Addresses features:** T5 (correction bounds), D2 (Rankin-Selberg L-value check), D3 (q-series fitting as diagnostic for D2)

**Avoids:** Pitfall 1 (circularity — dependency graph reviewed and peer-tested before code; "random primes" test applied), Pitfall 5 (explicit constants throughout; error budget allocated), Pitfall 6 (Rankin-Selberg identification must be explicit — find the form f, do not assume it exists)

**Implements:** rankin_selberg.py, barrier_bridge.py

**Critical gate:** Before any correction-bound code is written, produce a written dependency graph from "heat kernel trace > 0" to "B(L) > 0" with every bound step identified. Test each step: does it invoke zero-location information, zero-free regions, or GRH-conditional results? If any path from the conclusion leads back to RH, the approach must be restructured or documented as a reformulation, not a proof.

**Needs research:** Whether the correction epsilon(L) has any component provably bounded by purely geometric (prime geodesic) information without invoking zeta zeros; whether B(L) has an Euler product structure consistent with a Rankin-Selberg L-function.

### Phase 4: CM Point Evaluation (Conditional)

**Rationale:** CM point evaluation is diagnostic, not a primary proof path. It is only worth pursuing if Phase 3's q-series fitting (D3) finds that B(L) has a clean modular parametrization. If B(L) is not a modular function (no q-series structure), evaluation at Heegner points is numerically meaningless — Pitfall 7 explicitly.

**Delivers:** If gated: algebraic values of the barrier at the nine Heegner CM points (D = -3,-4,-7,-8,-11,-19,-43,-67,-163) using python-flint for certified j-invariant evaluation; confirmation or refutation of modular algebraic structure.

**Addresses features:** D4 (CM point evaluation)

**Avoids:** Pitfall 7 (false algebraicity — only pursued after modular form connection is established; q-series test must pass first)

**Implements:** cm_evaluation.py (uses python-flint acb.modular_j for certified values; mpmath.findpoly for algebraic recognition)

**Gate:** Skip entirely if Phase 3 D3 q-series fitting finds no clean modular parametrization. This saves implementation effort with no mathematical cost.

### Phase 5: Proof Assembly and Lean 4 Formalization

**Rationale:** Formalization should follow a validated proof, not anticipate one. The workbench infrastructure is well-established; the new content is the heat_kernel Lean 4 domain. Mathlib has limited analytic number theory coverage — the Selberg trace formula, Rankin-Selberg integrals, and heat kernels on hyperbolic surfaces likely require building new foundational material.

**Delivers:** proof_assembly.py orchestrating the full proof chain with workbench conjecture/experiment/evidence entries; if proof path succeeded: Lean 4 files (HeatKernelPositivity.lean, CorrectionBound.lean, ModularBarrier.lean) generated by the existing translator with new "heat_kernel" domain imports; if Lean 4 is blocked by Mathlib gaps: a rigorous informal proof manuscript.

**Addresses features:** Phase 4 features from FEATURES.md (formal proof assembly)

**Avoids:** Pitfall 13 (Lean 4 formalization gap — two-tier approach: informal proof first, Lean 4 second if Mathlib coverage allows), Pitfall 4 anti-pattern (premature formalization — proof_assembly.py only triggers Lean generation after evidence_level >= 2 and stress tests pass)

**Implements:** proof_assembly.py; new Lean domain "heat_kernel" in formalization/translator.py

**Standard patterns:** Workbench integration and Lean 4 file generation follow v1.0 patterns. The new Mathlib imports (NumberTheory.LSeries.RiemannZeta, NumberTheory.ModularForms.JacobiTheta.Basic) need one-time lookup but no novel research.

### Phase Ordering Rationale

- Phase 1 before Phase 2: the parameter mapping t(L) must be established before the Selberg trace formula can be applied to the specific test function corresponding to the barrier
- Phase 2 before Phase 3: the geometric decomposition from the Selberg trace formula reveals which correction components need bounding; starting bounds without this decomposition means bounding opaque quantities
- Circularity gate between Phases 2 and 3: this is the single highest-risk checkpoint in the project; it is a logical gate, not a schedule date
- Phase 4 is conditional on Phase 3 D3 output: if no modular structure is found in B(L), Phase 4 is skipped, preserving several sessions of effort
- Phase 5 is terminal: formalization adds no mathematical content; it records what has been established and should not be started until the proof path is clear

### Research Flags

Phases needing deeper research during planning:

- **Phase 2:** The admissibility of h(r) ~ 1/(L^2+r^2) under the Selberg trace formula requires mathematical clarification before implementation. The borderline r^{-2} decay may require a smooth cutoff, which modifies both sides of the trace formula by a controllable amount; this must be worked out before the implementation begins.
- **Phase 3:** The central open question — whether epsilon(L) can be bounded without circularly invoking zeta zeros — cannot be resolved by literature search alone; it requires working through the mathematics of the correction term and checking the dependency graph. Plan a dedicated pre-coding mathematical analysis step before Phase 3 implementation begins.

Phases with standard patterns (skip additional research):

- **Phase 1 infrastructure:** LMFDB data fetching, Maass form eigenvalue loading, and heat kernel special function evaluation (Bessel K, incomplete gamma) all follow established codebase patterns; no additional research needed.
- **Phase 5:** Workbench integration and Lean 4 pipeline follow v1.0 established patterns; the new "heat_kernel" Mathlib import list needs a one-time lookup.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | python-flint 0.8.0 verified on PyPI with Windows x86-64 wheels (wraps FLINT 3.3.1); all other libraries already in pyproject.toml and in production use; no version conflicts |
| Features | MEDIUM | Mathematical theory for T1–T6 is well-established; the novel integration path connecting B(L) to K(t) is the project's central unresolved hypothesis; feature scope is accurate but feasibility of T5 (correction bound) is genuinely uncertain |
| Architecture | MEDIUM | Module boundaries and data flow follow established v1.0 patterns and are well-reasoned; implementation of the mathematical content (heat kernel, Selberg trace, Rankin-Selberg) is inferred from literature and codebase analysis rather than prior implementation experience |
| Pitfalls | HIGH | Pitfall catalogue is grounded in direct project history (Sessions 35–43) plus peer-reviewed academic sources; circularity risk and continuous spectrum trap are documented from multiple angles; 0.036 margin-drain gap is experimentally verified at 800+ points |

**Overall confidence:** MEDIUM

### Gaps to Address

- **Parameter identification t = t(L):** The precise mapping between the barrier parameter L and the heat kernel time t is the first unknown to be resolved in Phase 1; four candidate mappings are documented in FEATURES.md and none has been confirmed. This is not resolvable by research — it requires computation.
- **Eisenstein contribution magnitude:** Whether the continuous spectrum integral is within the 0.036 error budget at proof-relevant L values is not known. Computing it numerically in Phase 1 will determine whether the entire approach is viable before Phase 2-3 investment. This is the highest-priority Phase 1 diagnostic.
- **Correction bound non-circularity:** Whether bounding epsilon(t,L) rigorously requires unconditional zero-location information is the central mathematical uncertainty of v2.0. This cannot be resolved by research; it requires working through the math in Phase 3. If it turns out to be circular, v2.0 becomes a reformulation of RH, not a proof — valuable, but a different outcome.
- **Mathlib coverage for Selberg trace and heat kernels:** Before committing to Lean 4 formalization in Phase 5, the extent of Mathlib's analytic number theory coverage must be audited. Likely coverage: partial for spectral theory, minimal for Selberg trace formula, near-zero for heat kernels on hyperbolic surfaces. This could extend Phase 5 substantially.

## Sources

### Primary (HIGH confidence)
- [mpmath docs: identification, Bessel, Legendre, quad, nsum](https://mpmath.org/doc/current/) — PSLQ/findpoly/identify, besselk, legenp/q, quad, nsum capabilities verified against stable library documentation
- [python-flint PyPI and FLINT acb_modular docs](https://pypi.org/project/python-flint/) — v0.8.0 Windows x86-64 wheels confirmed; acb.modular_j, acb.eisenstein, acb.modular_delta stable API
- [LMFDB Maass forms database](https://www.lmfdb.org/ModularForm/GL2/Q/Maass/) — spectral parameters for first 100+ Maass forms with 30+ digit precision; already integrated via lmfdb_client.py
- [Marklof: Selberg's Trace Formula, An Introduction (Bristol)](https://people.maths.bris.ac.uk/~majm/bib/selberg.pdf) — detailed SL(2,Z) formulas, test function conditions, scattering matrix
- [Zagier: Eisenstein Series and the Selberg Trace Formula I and II](https://people.mpim-bonn.mpg.de/zagier/) — authoritative source for continuous spectrum contribution to trace formula
- Session 35–47 project history — direct source for pitfall catalogue, margin-drain gap (0.036), and killed-approach inventory

### Secondary (MEDIUM confidence)
- [Booker-Strombergsson: Numerical computations with the trace formula](https://www2.math.uu.se/~ast10761/papers/stfz14march06.pdf) — rigorous eigenvalue verification methodology; Maass form tables cross-checked
- [Pollack: Rankin-Selberg Method User's Guide (UCSD)](https://mathweb.ucsd.edu/~apollack/rankin-selberg.pdf) — L(s, f x f_bar) integral representation, convergence conditions, unfolding step
- [Lapid: Nonnegativity of Rankin-Selberg L-functions at center of symmetry](https://academic.oup.com/imrn/article-abstract/2003/2/65/660341) — conditions for central value positivity; clarifies when Petersson norm positivity transfers
- [Kim-Sarnak: On Selberg's Eigenvalue Conjecture](https://www.researchgate.net/publication/227031617) — lambda_1 >= 975/4096 for SL(2,Z) proved unconditionally; relevant for Phase 3 spectral bounds
- [Boulanger: Heat kernel and Selberg pre-trace formula](https://arxiv.org/abs/1902.06580v2) — orbital decomposition methodology for the heat kernel on SL(2,Z)\H

### Tertiary (MEDIUM-LOW confidence, verify before use)
- [Tian An Wong: Explicit Formulae and Trace Formulae (CUNY thesis)](https://academicworks.cuny.edu/gc_etds/1542/) — most detailed source for the precise GL(1)->GL(2) connection; key reference for Phase 2; must be read carefully before implementation
- [Grigor'yan: Heat kernel on hyperbolic space](https://www.math.uni-bielefeld.de/~grigor/nog.pdf) — closed-form heat kernel formula reference; PDF not machine-readable but mathematics is standard
- [Borthwick: Spectral geometry lectures (Dartmouth)](https://math.dartmouth.edu/~specgeom/Borthwick_slides.pdf) — Selberg trace formula computational framework; MEDIUM confidence on completeness
- [Heat kernel on Cayley graph of PSL2Z (2025)](https://arxiv.org/html/2506.02340) — recent work on heat kernel on the modular group; may have relevant asymptotic bounds

---
*Research completed: 2026-04-04*
*Ready for roadmap: yes*
