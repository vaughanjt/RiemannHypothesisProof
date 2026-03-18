# Domain Pitfalls

**Domain:** Hybrid computational research platform and formal proof workbench for the Riemann Hypothesis
**Researched:** 2026-03-18
**Source basis:** Training data (web search unavailable). Confidence adjusted accordingly.

---

## Critical Pitfalls

Mistakes that cause rewrites, months of wasted effort, or fundamentally invalid results.

---

### Pitfall 1: Silent Precision Collapse in Zeta Function Evaluation

**What goes wrong:** The Riemann zeta function evaluated near the critical strip (Re(s) near 1/2) involves massive cancellation between large terms. IEEE 754 double-precision (64-bit float, ~15 decimal digits) silently loses all significant digits for large imaginary parts. A call like `scipy.special.zeta(0.5 + 1000000j)` returns garbage with no warning. The result looks like a number -- it is a number -- it is just completely wrong. You build visualizations, detect "patterns," and draw conclusions from noise.

**Why it happens:** The zeta function for large Im(s) is computed via the Riemann-Siegel formula or Euler-Maclaurin summation, both involving sums of oscillatory terms whose magnitudes vastly exceed the final result. If you have N digits of working precision and the intermediate terms are 10^M times larger than the result, you need N > M digits or the answer is pure rounding error. For Im(s) ~ 10^6, you need hundreds of digits of precision. Standard numpy/scipy use hardware floats with ~15 digits.

**Consequences:**
- "Discovered" zeros that do not exist
- "Patterns" in zero distributions that are artifacts of rounding
- Visualizations showing structure where there is only noise
- Weeks of investigation into phantom phenomena
- Erosion of trust in the entire computational pipeline

**Prevention:**
1. **Never use float64 for zeta evaluation.** Start every computation with `mpmath` and `mp.dps` set appropriately.
2. **Calibrate precision per-region.** Rule of thumb: for Im(s) ~ T, you need at least `log10(T) * K` extra digits where K depends on the algorithm (K ~ 0.5 for Riemann-Siegel, more for direct summation). For T = 10^6, use at least 50+ digits.
3. **Build a precision validation layer.** Compute every critical result at precision P and again at precision 2P. If they disagree beyond a tolerance, the result at precision P is unreliable.
4. **Test against known zeros.** The first 10 billion zeros of the zeta function are known. Any computational pipeline must exactly reproduce known zero locations (e.g., first zero at t ~ 14.134725...) before being trusted for exploration.
5. **Wrap all zeta calls behind an abstraction** that enforces minimum precision and logs precision metadata with every result.

**Warning signs:**
- Zeros appearing at non-symmetric positions (all non-trivial zeros come in conjugate pairs)
- Results changing when you increase `mp.dps` by a factor of 2
- Suspiciously "interesting" patterns that vanish at higher precision
- Zeta values near zeros that are exactly zero (they should be approximately zero to the working precision, never exactly zero)

**Detection:** Build automated "precision canary" tests that run at startup and before any exploration session.

**Phase to address:** Phase 1 (Computational Engine). This must be the absolute bedrock before any exploration begins.

**Confidence:** HIGH -- catastrophic cancellation in zeta evaluation is extremely well-documented in the numerical analysis literature (Brent, Odlyzko, Rubinstein, et al.).

---

### Pitfall 2: Confusing Numerical Evidence with Mathematical Proof

**What goes wrong:** The platform generates compelling computational evidence -- "every zero I checked lies on the critical line," "this spectral operator's eigenvalues match zero spacings to 50 digits" -- and the user begins treating numerical patterns as established facts. Cross-disciplinary connections that "look right" computationally get woven into a proof narrative without rigorous justification. The gap between "true for the first million zeros" and "true for all infinitely many zeros" is the entire content of the Riemann Hypothesis.

**Why it happens:** Human pattern recognition is powerful but not a proof. The Riemann Hypothesis has been verified for the first 10^13 zeros. That is not a proof. There are conjectures in number theory (e.g., Mertens' conjecture, Polya's conjecture, Skewes' number phenomena) where numerical evidence supported the conjecture for astronomically large ranges before failing. The first counterexample to some conjectures appears only at numbers exceeding 10^300.

**Consequences:**
- Building a "proof" on an unproven computational observation
- Wasting months formalizing a statement that is not actually true
- Confirmation bias: selectively exploring regions that support your hypothesis
- Missing genuine insights because you stopped looking after "confirming" a pattern

**Prevention:**
1. **Maintain a strict epistemological hierarchy in the research workbench:**
   - Level 0: Computational observation (X appears to hold for tested cases)
   - Level 1: Heuristic argument (here is a plausible reason X might be true)
   - Level 2: Conditional result (X is true IF Y and Z, which are also unproven)
   - Level 3: Rigorous proof (X is formally verified in Lean 4)
2. **Every conjecture in the workbench must be tagged with its level.** Visualizations should display this level. Never let Level 0 evidence flow into Level 2+ reasoning without explicit flagging.
3. **Actively seek counterexamples.** For every pattern observed, spend time trying to break it before investing in understanding why it holds.
4. **Log the parameter range tested** for every numerical observation. "This holds for |t| < 10^4 with 100-digit precision" is a meaningful statement. "This is true" is not.

**Warning signs:**
- Phrases like "clearly true" or "obviously holds" in research notes without proof references
- Skipping counterexample searches because "it always works"
- Excitement about a pattern leading immediately to formalization attempts
- Proof sketches that say "by computation" for non-trivial steps

**Detection:** Research workbench should require explicit evidence-level tagging for every claim.

**Phase to address:** Phase 2 (Research Workbench). Must be designed into the workbench from day one, not bolted on.

**Confidence:** HIGH -- this is a foundational principle of mathematical research, and the specific examples (Mertens' conjecture, etc.) are well-established.

---

### Pitfall 3: Building Infrastructure Instead of Doing Mathematics

**What goes wrong:** You spend six months building a beautiful, extensible, perfectly-architected computational framework with plugin systems, configurable pipelines, abstract base classes for every mathematical object -- and never actually explore the zeta function in any depth. The platform becomes the project instead of the mathematics becoming the project. This is particularly dangerous for a non-mathematician user because infrastructure work feels productive and controllable, while mathematical exploration feels uncertain and uncomfortable.

**Why it happens:** Software engineering is a known skill set; mathematical research is unfamiliar territory. Building tools has clear progress metrics (features shipped, tests passing, code coverage). Mathematical exploration has no such metrics -- you can spend a week thinking and have nothing to show for it. The temptation to retreat into building "one more feature" before starting exploration is overwhelming.

**Consequences:**
- A polished tool with no mathematical insights
- Over-engineered abstractions that do not match how exploration actually works (because you have not explored yet)
- Architecture decisions made without understanding the actual computational patterns needed
- Loss of motivation when the "real work" (math) never begins

**Prevention:**
1. **Phase 1 must include actual mathematical exploration.** Do not wait for a "complete" engine. Compute zeta zeros on day one with mpmath in a Jupyter notebook. Build the platform around patterns you discover in actual use.
2. **Apply a 70/30 rule in every phase.** 70% exploration/math, 30% infrastructure. If a sprint is 100% infrastructure, something is wrong.
3. **Define "mathematical milestones" not just "engineering milestones."** Example mathematical milestones: "reproduce the known zero distribution statistics," "compute the pair correlation of zeros and compare to GUE," "evaluate a candidate spectral operator on known zeros."
4. **Time-box infrastructure work.** If a feature is taking more than 2 days to build, ask: "Can I do the math I want with a simpler version of this?"

**Warning signs:**
- No mathematical computation has been run in the last week
- Refactoring code that has only been used once
- Adding configuration options for scenarios that have not arisen
- The words "we need to build X before we can explore Y" appearing frequently

**Detection:** Track the ratio of git commits that touch mathematical exploration code vs. infrastructure code.

**Phase to address:** All phases, but especially Phase 1. Start exploring immediately, even with crude tools.

**Confidence:** HIGH -- this is a universal pattern in research software projects, well-documented in scientific computing literature and researcher experience reports.

---

### Pitfall 4: Premature Lean 4 Formalization

**What goes wrong:** A promising computational pattern is observed, and the team immediately begins formalizing it in Lean 4. Lean 4 formalization is extremely labor-intensive -- even simple mathematical statements can take days or weeks to formalize. The computational insight turns out to be an artifact, the approach does not generalize, or the mathematical framework shifts, and the formalization work is wasted.

**Why it happens:** Formalization feels like "real progress" toward a proof. There is also an underestimation of how hard Lean 4 formalization is -- people expect it to be like writing pseudocode with type checking, but it is more like building a skyscraper from individual bricks where every brick must fit perfectly.

**Consequences:**
- Weeks of formalization work discarded when the mathematical direction shifts
- Frustration and burnout from Lean 4's steep learning curve applied to uncertain mathematics
- Premature commitment to a mathematical framework that constrains exploration
- Formalization of trivial lemmas while critical insights remain unformalized

**Prevention:**
1. **Gate formalization behind a maturity threshold.** A result should be at evidence Level 2 (conditional result with heuristic justification) before formalization begins.
2. **Start Lean 4 formalization with known, established results** -- formalize the functional equation of zeta, the Euler product, basic properties of the critical strip. This builds Lean 4 skill on known-correct mathematics.
3. **Budget formalization time explicitly.** It is typically 10-100x more effort to formalize a result than to state it informally. Plan accordingly.
4. **Build a "formalization readiness checklist":** Has the result been stable for 2+ weeks? Has it survived counterexample searches? Is the proof strategy clear at the informal level? Can Claude outline a complete proof?

**Warning signs:**
- Formalizing results discovered in the last 48 hours
- Lean 4 compilation errors consuming more time than mathematical thinking
- Formalizing helper lemmas without a clear path to what they support
- The Lean 4 codebase has more `sorry` (unproven goals) than proven theorems

**Detection:** Track the ratio of `sorry` to complete proofs. Track the age of results before formalization begins.

**Phase to address:** Phase 3 or later (Lean 4 Pipeline). The formalization phase should explicitly begin with "practice formalization" of known results.

**Confidence:** HIGH -- the difficulty of theorem prover formalization is extensively documented by the Lean community, Mathlib contributors, and projects like Formal Abstractions.

---

### Pitfall 5: Visualization Artifacts Mistaken for Mathematical Structure

**What goes wrong:** A beautiful pattern appears in a visualization of zeta zeros, spectral eigenvalues, or a projected high-dimensional structure. The user investigates it, builds theories around it, and it turns out to be an aliasing artifact, a Moire pattern from the plotting resolution, a projection artifact from dimensionality reduction, or a color map discontinuity.

**Why it happens:** The human visual system is extraordinarily good at detecting patterns -- including patterns that do not exist in the data. When projecting from high-dimensional spaces to 2D/3D (a core feature of this project), information is necessarily lost and distorted. Different projection methods (PCA, t-SNE, UMAP, random projection) can create wildly different apparent structures from the same data. Color maps with perceptual nonlinearities (looking at you, jet/rainbow) create artificial boundaries.

**Consequences:**
- Days investigating a projection artifact
- False confidence from "seeing" a structure that is not there
- Correct structures being dismissed because they look similar to known artifacts
- Over-reliance on one projection method that happens to show what you expect

**Prevention:**
1. **Always use perceptually uniform colormaps** (viridis, inferno, cividis). Never use rainbow/jet.
2. **Every visualization must be reproducible at different resolutions.** If a pattern disappears when you double the sampling density, it is an artifact.
3. **High-dimensional projections must be shown with at least two independent methods.** If a structure appears in PCA but not in a random projection, it may be a PCA artifact (or it may be real but subtle -- this is information either way).
4. **Display quantitative metrics alongside visualizations.** Do not just show a plot of zero spacings -- show the computed pair correlation function, the nearest-neighbor spacing distribution, and their numerical comparison to GUE predictions.
5. **Build "artifact awareness" into the visualization module.** Every projection should display: the projection method, the fraction of variance preserved, the effective dimensionality, and a warning when the data has been substantially distorted.

**Warning signs:**
- A pattern that is only visible at one zoom level or resolution
- Structure that depends strongly on the projection method
- Color-boundary-aligned features
- Patterns that look "too clean" for mathematical data

**Detection:** Implement automated artifact detection: render at 2 resolutions and flag regions where structure differs.

**Phase to address:** Phase 2 (Visualization). Must be built into the visualization framework from the start.

**Confidence:** HIGH -- visualization artifacts are well-documented in scientific visualization literature (Borland & Taylor on rainbow colormaps, Wattenberg on dimensionality reduction distortions).

---

### Pitfall 6: Underestimating the Lean 4 + Mathlib Learning Curve

**What goes wrong:** The user (who is not a mathematician by training) or the AI assistant attempts to formalize results in Lean 4 without adequate preparation. Lean 4 is not just a programming language -- it is a dependent type theory with its own idioms, tactics, and ecosystem. Mathlib (the mathematical library for Lean 4) has its own conventions, naming schemes, and organizational principles that take significant time to learn.

**Why it happens:** Lean 4 syntax looks superficially like a functional programming language, giving the false impression that programming experience transfers directly. In reality, the skills required are: understanding dependent type theory, navigating Mathlib's 500,000+ lines of code, knowing which tactics to apply when, and thinking in terms of type-theoretic proof terms rather than computational steps.

**Consequences:**
- Simple lemmas taking days to formalize
- Inability to find relevant Mathlib theorems (they exist but under unintuitive names)
- Writing 100-line proofs for things that are 3-line proofs with the right tactic
- Giving up on formalization entirely due to frustration
- Lean 4 version/Mathlib version incompatibilities causing mysterious breakage

**Prevention:**
1. **Allocate explicit Lean 4 learning time** before attempting any project-specific formalization. Budget at minimum 2-4 weeks of focused learning.
2. **Start with the Natural Number Game and Mathematics in Lean 4** tutorials before touching project code.
3. **Formalize known simple results first:** the triangle inequality, basic properties of complex numbers, convergence of simple series. Graduate to: the functional equation of zeta (which is in Mathlib).
4. **Pin Lean 4 and Mathlib versions** and do not upgrade mid-project without explicit planning. Mathlib moves fast and breaking changes are common.
5. **Leverage Claude for Lean 4 heavily.** Claude can write and debug Lean 4 tactics, search Mathlib, and translate informal proofs to formal ones. The user should direct what to formalize; Claude should do the typing.
6. **Use `lake` correctly.** Lean 4 build tool (`lake`) has its own conventions. Set up the project structure once, correctly, following Mathlib's template.

**Warning signs:**
- More than 2 hours stuck on a single tactic application
- `sorry` count increasing faster than proven-theorem count
- Mathlib version drift causing compilation failures
- Attempting to formalize advanced results before basic ones compile

**Detection:** Track time-per-lemma and sorry-to-proof ratios.

**Phase to address:** Phase 3 (Lean 4 Pipeline), but learning should begin in Phase 2.

**Confidence:** HIGH -- Lean 4 difficulty is universally acknowledged by the Lean community. Mathlib documentation explicitly warns about the learning curve.

---

## Moderate Pitfalls

Mistakes that cause significant delays or rework but are recoverable.

---

### Pitfall 7: Scope Explosion in Cross-Disciplinary Approaches

**What goes wrong:** The project connects the Riemann Hypothesis to spectral theory, random matrix theory, quantum chaos, information theory, modular forms, adelic spaces, and hyperbolic geometry. Each of these is a deep field with its own literature, computational tools, and open problems. The user gets pulled into exploring each connection, each spawning new sub-investigations, and the project becomes an unfocused survey of interesting mathematics rather than a directed proof attempt.

**Why it happens:** Cross-disciplinary exploration is genuinely exciting and every connection feels like it could be "the one." The Riemann Hypothesis genuinely does connect to all of these fields. The problem is not that the connections are uninteresting -- they are fascinating -- but that investigating all of them simultaneously means investigating none of them deeply enough.

**Prevention:**
1. **Maintain an active "investigation stack"** with a maximum depth of 3. When a new connection is discovered, it goes on a backlog, not into active investigation.
2. **Time-box exploratory tangents.** "I will spend 2 hours exploring this spectral connection. If it does not yield a concrete next step, it goes to the backlog."
3. **Define success criteria before starting each exploration.** "This investigation succeeds if: I can compute eigenvalues of operator X and show they correlate with zeta zeros to 20 digits."
4. **Quarterly pruning of the investigation backlog.** If an idea has been on the backlog for 3 months without rising to the top, archive it.

**Warning signs:**
- More than 5 active investigation threads
- Unable to articulate the current primary hypothesis
- Every session starts a new topic rather than deepening a current one
- Research notes reference 10+ distinct mathematical frameworks

**Phase to address:** All phases, but formalize this discipline in Phase 2 (Research Workbench).

**Confidence:** HIGH -- scope explosion is the most common failure mode in exploratory research projects.

---

### Pitfall 8: Neglecting the Functional Equation and Known Symmetries

**What goes wrong:** The computational engine correctly evaluates zeta(s) but does not build in the deep structural properties of the zeta function -- the functional equation relating zeta(s) to zeta(1-s), the Euler product over primes, the relationship to the xi function which is entire and has better symmetry properties. Computations that violate these symmetries go undetected because the engine does not check them.

**Why it happens:** Implementing zeta(s) as a "black box" that returns a number is the natural first step. Building in structural awareness requires mathematical sophistication that goes beyond just computing values.

**Prevention:**
1. **Implement symmetry checks as assertions.** Every zeta evaluation should be verifiable against the functional equation (at least in a test/debug mode).
2. **Work with the xi function** (or the Riemann xi function, or the Z-function on the critical line) when possible -- these have better symmetry properties than raw zeta.
3. **Build the Euler product as an independent verification path** for Re(s) > 1.
4. **Test the functional equation numerically** at every precision level you plan to use.

**Warning signs:**
- zeta(s) and zeta(1-s) do not satisfy the functional equation to working precision
- Zeros computed off the critical line when working with the Z-function (Z-function zeros on the real axis correspond to zeta zeros on the critical line)
- Euler product and direct evaluation disagree for Re(s) > 1

**Phase to address:** Phase 1 (Computational Engine).

**Confidence:** HIGH -- structural properties of the zeta function are textbook material.

---

### Pitfall 9: Performance Death Spiral in High-Dimensional Computation

**What goes wrong:** Operating in N-dimensional spaces (a core requirement) with arbitrary-precision arithmetic creates a computational cost that scales horrifyingly. Arbitrary precision is O(n * log(n)) per multiplication where n is the number of digits. N-dimensional computations often involve O(N^2) to O(N^3) matrix operations. Combined: a 100-dimensional computation at 200-digit precision can be 10,000x slower than the same computation at machine precision. The platform becomes unusable for interactive exploration.

**Why it happens:** Each factor seems manageable in isolation. "100 dimensions? That is just a matrix." "200 digits? mpmath handles that." The product of these costs is the killer.

**Prevention:**
1. **Profile aggressively from day one.** Every computation should log its wall-clock time and precision settings.
2. **Use a precision hierarchy:** Explore at moderate precision (50-100 digits), confirm at high precision (200+ digits), visualize at the lowest precision that preserves structure.
3. **Cache aggressively.** Zeta function evaluations at high precision are expensive. Cache results keyed by (s, precision). Invalidate only on precision upgrade.
4. **Use numpy/scipy for preliminary work** where machine precision suffices (e.g., matrix decompositions for structure detection), then re-verify interesting findings at arbitrary precision.
5. **Parallelize where possible.** Zeta evaluations at independent points are embarrassingly parallel. Use multiprocessing (not threading -- GIL).
6. **Consider ARB (Arb library, now FLINT)** for interval arithmetic with rigorous error bounds -- it can be faster than mpmath for some operations and gives guaranteed error bounds.

**Warning signs:**
- A single computation taking more than 60 seconds in an interactive session
- Memory usage growing unboundedly during exploration
- The user waiting for results instead of exploring
- Precision set globally rather than per-computation

**Phase to address:** Phase 1 (Computational Engine), with ongoing optimization.

**Confidence:** MEDIUM -- specific performance characteristics depend on exact algorithms and hardware. The general principle (exponential cost multiplication) is HIGH confidence.

---

### Pitfall 10: Research Progress Amnesia

**What goes wrong:** After weeks of exploration, the user cannot reconstruct why they pursued a particular direction, what they tried that did not work, or what the key insight from last month's investigation was. The research workbench accumulates disconnected observations without a coherent narrative. Previous dead ends are re-explored because no one remembers they were dead ends.

**Why it happens:** Mathematical research is inherently nonlinear. Insights arrive out of order, failed attempts are psychologically discarded, and the connection between observations is often only clear in retrospect. Without disciplined logging, the exploratory process becomes a random walk that revisits the same territory.

**Prevention:**
1. **Require structured logging for every exploration session.** Template: Date, Hypothesis, What was tried, What was observed, What it means, Next steps.
2. **Maintain a "dead ends" registry** with as much care as the "promising leads" list. "We tried X, it did not work because Y" is extremely valuable information.
3. **Weekly synthesis notes** that attempt to connect the week's observations into a narrative.
4. **Tag all computational results** with the investigation context that produced them.
5. **Build a "previously investigated" search** into the workbench so the user can check "have I looked at this before?"

**Warning signs:**
- Asking "didn't we try this already?"
- Research notes that say "interesting" without explaining why
- Inability to explain the current proof strategy to a new audience
- Computational results saved without context

**Phase to address:** Phase 2 (Research Workbench). This is a core workbench feature, not an afterthought.

**Confidence:** HIGH -- this is a universal problem in research, particularly acute in solo/small-team projects without external accountability structures.

---

## Minor Pitfalls

Mistakes that cause annoyance or minor delays.

---

### Pitfall 11: Lean 4 / Python Ecosystem Mismatch

**What goes wrong:** Results are computed in Python and then need to be stated in Lean 4. The translation is lossy and error-prone. Python uses floating-point approximations; Lean 4 works with exact mathematical objects. A computed "zero at 14.134725..." must become a formal statement about the existence of a zero in an interval, which requires interval arithmetic that was not part of the Python pipeline.

**Prevention:**
1. Design the Python-to-Lean pipeline to output interval-arithmetic results (a zero exists in [14.13472, 14.13473]) rather than point estimates.
2. Use the ARB/FLINT library for computations intended for formalization -- it produces rigorous enclosures.
3. Build a translation layer that converts Python computational results into Lean 4-compatible formal statements.

**Phase to address:** Phase 3 (Lean 4 Pipeline).

**Confidence:** MEDIUM -- the specific integration challenges depend on what Mathlib provides for analytic number theory (coverage is growing but may have gaps).

---

### Pitfall 12: Overreliance on AI for Mathematical Reasoning

**What goes wrong:** Claude is used not just for formalism and computation but as the primary source of mathematical insight. Claude can be wrong about mathematics -- it can state incorrect theorems, produce plausible but invalid proofs, and hallucinate mathematical results. If the user does not have the mathematical background to verify Claude's claims, incorrect mathematics can propagate through the entire project.

**Prevention:**
1. Every mathematical claim from Claude must be verifiable: either computationally (does the formula produce correct values?) or formally (does it compile in Lean 4?).
2. Build "Claude verification checkpoints" into the workflow. Periodically have Claude check its own earlier work with fresh context.
3. Cross-reference Claude's claims against established references (DLMF, LMFDB, Mathworld).
4. Never trust Claude's claim that something is "well-known" or "follows easily" without verification.

**Warning signs:**
- Mathematical results that cannot be independently checked
- "Claude said so" as justification in research notes
- Proofs that "feel right" but have not been verified

**Phase to address:** All phases.

**Confidence:** HIGH -- LLM mathematical reasoning limitations are well-documented.

---

### Pitfall 13: Ignoring Existing Computational Infrastructure

**What goes wrong:** Building custom implementations of algorithms that already exist in well-tested, optimized libraries. Reimplementing zero-finding algorithms, special functions, or matrix decompositions from scratch when mpmath, scipy, FLINT, or PARI/GP already provide them.

**Prevention:**
1. Before implementing any mathematical algorithm, check: mpmath, scipy, FLINT/ARB, PARI/GP, and SageMath.
2. The LMFDB (L-functions and Modular Forms Database) has extensive precomputed data on zeros of L-functions.
3. Andrew Odlyzko has published tables of zeta zeros to extremely high precision -- use these for verification.

**Phase to address:** Phase 1 (Computational Engine).

**Confidence:** HIGH -- these resources are well-established and publicly available.

---

## Technical Debt Patterns

Patterns that seem harmless initially but accumulate into serious problems.

---

### Debt 1: Global Precision State

**What it is:** Using `mpmath.mp.dps = 50` as a global setting and forgetting about it. Different computations require different precision levels, and a global setting means either every computation is too slow (precision too high) or some computations are wrong (precision too low).

**How it accumulates:** Early code sets precision globally. Later code assumes that precision. Changing it breaks earlier code. Nobody knows what precision any particular result was computed at.

**Prevention:** Wrap mpmath in a context manager that sets precision per-computation and logs it. Never expose global `mp.dps` to user code.

```python
# BAD
mp.dps = 100
result = zeta(s)

# GOOD
with precision_context(dps=100, label="zeta evaluation for zero-finding"):
    result = zeta(s)
# result carries metadata: {precision: 100, algorithm: "mpmath.zeta", timestamp: ...}
```

---

### Debt 2: Unversioned Mathematical Conjectures

**What it is:** Conjectures and hypotheses in the research workbench evolve over time but previous versions are overwritten rather than versioned. When an approach fails, you cannot go back to the version of the conjecture that seemed promising.

**Prevention:** Every conjecture should be versioned (like git for ideas). The research workbench should never allow destructive updates to conjectures -- only append new versions.

---

### Debt 3: Visualization Code Entangled with Computation Code

**What it is:** Plotting code mixed into computational functions. When you want to compute without plotting (e.g., in a batch job or formalization pipeline), you have to hack around the visualization. When you want to change how something is plotted, you risk breaking the computation.

**Prevention:** Strict separation: computation functions return data objects; visualization functions consume data objects and produce plots. Never call `plt.plot()` from inside a mathematical function.

---

### Debt 4: Hardcoded Mathematical Constants

**What it is:** Sprinkling `14.134725...` (first zeta zero), `0.5` (critical line), or `2 * pi` throughout the codebase without named references or precision-appropriate computation.

**Prevention:** Define all mathematical constants through mpmath at the required precision. Use named constants: `CRITICAL_LINE = mpf('0.5')`, `FIRST_ZERO = ... computed to working precision ...`.

---

## Performance Traps

---

### Trap 1: mpmath's Hidden O(n^2) in String Conversion

**What it is:** Converting very high-precision mpmath numbers to strings (for display, logging, or serialization) can be surprisingly slow because of the base conversion algorithm. At 10,000+ digits, this becomes noticeable.

**Prevention:** For logging and display, truncate to reasonable display precision. For serialization, use mpmath's native binary format or pickle.

---

### Trap 2: Matplotlib Cannot Handle Large Point Counts

**What it is:** Plotting 10^6+ points in matplotlib causes the GUI to become unresponsive. This is easy to hit when plotting zeta function evaluations along the critical line at high density.

**Prevention:** Use decimation/level-of-detail for interactive plots. For publication-quality static plots, render to PNG directly without GUI. Consider vispy or datashader for large point clouds.

---

### Trap 3: Python's GIL Blocks True Parallelism

**What it is:** Threading in Python does not give parallel computation due to the Global Interpreter Lock. For CPU-bound work like zeta evaluation, threads provide no speedup.

**Prevention:** Use `multiprocessing` (not `threading`) for parallel zeta evaluations. Or use subprocess-based parallelism. Consider `gmpy2` backend for mpmath (significantly faster than pure Python).

---

### Trap 4: Memory Exhaustion from Caching Without Eviction

**What it is:** Caching zeta evaluations (recommended above) without a cache eviction policy leads to unbounded memory growth during long exploration sessions.

**Prevention:** Use an LRU cache with a configurable size limit. For persistent caching across sessions, use a disk-backed cache (e.g., `diskcache` or SQLite).

---

## "Looks Done But Isn't" Checklist

Items that superficially appear complete but harbor hidden incompleteness.

| Component | Looks Done When... | Actually Done When... |
|-----------|-------------------|----------------------|
| Zeta evaluation | Returns a number for zeta(s) | Matches known zeros to full working precision, satisfies functional equation, precision-validated |
| Zero finder | Finds zeros near known locations | Finds zeros in unexplored regions, reports isolation (no closer zero exists), handles high-lying zeros (Im > 10^6) |
| High-dim visualization | Renders a 3D projection | Shows artifacts warnings, supports multiple projection methods, includes quantitative metrics alongside visual |
| Research workbench | Can record notes | Enforces evidence levels, versions conjectures, tracks dead ends, supports search/cross-reference |
| Lean 4 pipeline | Compiles a simple proof | Has Mathlib integration working, can formalize statements about zeta, builds reproducibly on clean machine |
| Spectral analysis | Computes eigenvalues of a matrix | Validates against known spectra, handles operator discretization errors, reports spectral convergence |
| Pattern detection | Flags statistical outliers | Accounts for multiple comparison correction, distinguishes mathematical structure from numerical artifact, reports effect size and significance |
| Cross-disciplinary module | Implements formula from a paper | Verified against paper's own numerical examples, boundary cases handled, parameter ranges documented |

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Phase 1: Computational Engine | Precision collapse (Pitfall 1) | Build precision validation from day one; test against known zeros before any exploration |
| Phase 1: Computational Engine | Performance death spiral (Pitfall 9) | Profile from day one; use precision hierarchy; cache aggressively |
| Phase 1: Computational Engine | Rebuilding existing wheels (Pitfall 13) | Audit mpmath, scipy, FLINT, PARI/GP before writing anything custom |
| Phase 2: Visualization | Artifact blindness (Pitfall 5) | Multiple projection methods; resolution-independence testing; perceptual colormaps |
| Phase 2: Research Workbench | Evidence level confusion (Pitfall 2) | Build epistemological hierarchy into workbench schema from day one |
| Phase 2: Research Workbench | Progress amnesia (Pitfall 10) | Structured session logging; dead-end registry; weekly synthesis |
| Phase 3: Cross-Disciplinary Modules | Scope explosion (Pitfall 7) | Investigation stack with depth limit; time-boxed tangents; success criteria before exploration |
| Phase 3: Lean 4 Pipeline | Premature formalization (Pitfall 4) | Maturity threshold gate; start with known results; track sorry-to-proof ratio |
| Phase 3: Lean 4 Pipeline | Learning curve underestimation (Pitfall 6) | Explicit learning period; tutorials before project code; Claude handles Lean 4 writing |
| Phase 3: Lean 4 Pipeline | Python-Lean mismatch (Pitfall 11) | Interval arithmetic in Python; ARB/FLINT for formalization-bound computations |
| All Phases | Infrastructure over mathematics (Pitfall 3) | 70/30 exploration-to-infrastructure ratio; mathematical milestones in every sprint |
| All Phases | AI overreliance (Pitfall 12) | Verification checkpoints; computational checks for every claim; cross-reference established sources |

---

## Sources

- Training data only (web search, Brave Search, and WebFetch were unavailable during research)
- Domain knowledge derived from: numerical analysis literature (Brent, Odlyzko, Rubinstein on zeta computation), Lean 4 community documentation and Mathlib contributor experience reports, scientific visualization best practices (Borland & Taylor), computational mathematics software engineering patterns
- Known zero tables: Andrew Odlyzko's computations, LMFDB database
- Confidence levels adjusted downward due to inability to verify against current documentation

**Note:** All findings are based on training data (cutoff: early-mid 2025). Specific version numbers, API details, and current ecosystem state should be verified against live documentation before implementation decisions. The general principles and pitfall patterns described here are stable and well-established, lending HIGH confidence despite the inability to perform live verification.
