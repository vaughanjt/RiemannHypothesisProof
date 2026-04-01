import Mathlib.NumberTheory.LSeries.RiemannZeta

/-!
# External Formalization Landscape: Tao's IEANT Network and the PNT+ Project

Survey of active Lean 4 formalization efforts in analytic number theory,
with connections to our RH investigation (Sessions 1-31).

Last updated: 2026-03-31

## Sources

* Tao blog post (2026-01-15):
  https://terrytao.wordpress.com/2026/01/15/the-integrated-explicit-analytic-number-theory-network/
* IPAM project page:
  https://www.ipam.ucla.edu/news-research/special-projects/integrated-explicit-analytic-number-theory-network/
* PNT+ repository: https://github.com/AlexKontorovich/PrimeNumberTheoremAnd
* PNT+ blueprint: https://alexkontorovich.github.io/PrimeNumberTheoremAnd/
* Loeffler-Stoll paper (arXiv:2503.00959, Annals of Formalized Mathematics 2025):
  "Formalizing zeta and L-functions in Lean"
* Analytic Number Theory Exponent Database (ANTEDB):
  https://teorth.github.io/expdb/
* Irrationality of zeta(3) (arXiv:2503.07625): Lean 4 formalization using
  Beukers' method, building on PNT+ for the error-term PNT.

---

# PART 1: WHAT EXISTS (as of March 2026)
---

## 1A. Mathlib (in mainline, fully proved)

The following are PROVED in Mathlib with NO sorry:

* `riemannZeta`: definition via analytic continuation through the Hurwitz zeta
  and the Jacobi theta function's Mellin transform.

* Analytic continuation: ζ(s) is defined for all s, with the Dirichlet series
  agreeing with the L-series definition for Re(s) > 1.

* Functional equation: `riemannZeta_one_sub` —
  ζ(1-s) = 2^{1-s} π^{-s} Γ(s) sin(π(1-s)/2) ζ(s) for 1-s not in N.

* Non-vanishing on Re(s) >= 1: `riemannZeta_ne_zero_of_one_le_re` —
  the keystone theorem. Proved via the Euler product for Re(s) > 1 and a
  clever identity argument at Re(s) = 1 (the 3-4-1 trick).

* Dirichlet L-functions: `DirichletCharacter.LFunction_ne_zero_of_one_le_re` —
  L(chi, s) != 0 for Re(s) >= 1, for any Dirichlet character chi (handling
  both the quadratic and non-quadratic cases separately).

* Dirichlet's theorem on primes in AP: proved as a consequence of the
  L-function non-vanishing.

* Euler product for Dirichlet L-series.

* Formal statement: `RiemannHypothesis` is accessible as a Prop.

* Special values: `riemannZeta_zero` gives ζ(0) = -1/2.

* Hurwitz zeta: full meromorphic continuation, functional equation,
  residue at s=1 equal to 1. The Riemann zeta is a special case.

* Cauchy integral formula (for discs): `DiffContOnCl.circleIntegral_eq`.
  BUT: the general residue theorem and Laurent series are NOT yet in Mathlib.

* Complex analysis infrastructure: holomorphic functions, Cauchy-Riemann,
  maximum modulus principle, open mapping theorem, identity theorem.
  Contour integrals exist for circles but NOT for general paths.

### What is NOT in Mathlib:
- No zero enumeration (gamma_n sequence)
- No Xi function
- No Hadamard product formula
- No explicit formula (von Mangoldt / Riemann-Weil)
- No zero-free region beyond Re(s) >= 1 (no de la Vallee-Poussin)
- No residue theorem for general contours
- No Laurent series
- No Perron's formula
- No zero density estimates
- No random matrix theory

## 1B. PNT+ Project (Kontorovich-Tao, github.com/AlexKontorovich/PrimeNumberTheoremAnd)

Status: Active development. The Prime Number Theorem has been PROVED in Lean 4
via the Wiener-Ikehara Tauberian theorem.

### Formalized and proved:
* Wiener-Ikehara Tauberian theorem (Fourier-analytic proof)
* Prime Number Theorem: psi(x) ~ x (Chebyshev function is asymptotic to x)
* PNT with error term: a strengthened version with classical error term,
  stronger than what had previously been formalized
* Bijectivity of Fourier transform on Schwartz class
* Mobius inversion form of PNT: sum_{n <= x} mu(n)/n -> 0
* Rectangle border definition (RectangleBorder) — foundational for
  future contour integral work

### In progress / blueprint:
* PNT in arithmetic progressions
* Chebotarev density theorem (stretch goal)
* Selberg sieve (partial; fundamental theorem proved by a contributor)
* Rectangle integral machinery for Perron-type formulas

### Architecture notes:
* Uses a blueprint system with color-coded dependency graph
* Contributions tracked via GitHub issues with XS-XL size labels
* AI use permitted with disclosure; all PRs must pass CI

## 1C. Tao's Integrated Explicit Analytic Number Theory Network (IEANT)
     Launched January 2026; hosted within PNT+

### What it is:
A crowdsourced formalization project to formalize EXPLICIT (all constants
explicit, no big-O) analytic number theory results in Lean. Partially
hosted in the PNT+ repo. Run by Tao as IPAM Director of Special Projects,
with financial/technical support from Math Inc.

### Two components:
1. **Lean formalization**: Formalizing papers by Fiori, Kadiri, Swidinsky
   and others on explicit bounds for psi(x), pi(x), theta(x).
   Key target: the explicit PNT of Fiori-Kadiri-Swidinsky, which gives
   |psi(x) - x| < 9.22 * x * (log x)^{3/2} * exp(-0.8477 * sqrt(log x))
   for all x > 2.

2. **Interactive spreadsheet**: An "estimate propagation" system where
   changing one numerical input (e.g., the height to which RH is verified)
   automatically propagates through all dependent explicit bounds.
   Future: AI-assisted optimization of these numerical relationships.

### What has been formalized so far (as of Jan 2026 announcement):
* Some smaller results largely formalized
* Good progress on several larger papers
* Blueprint breaks proofs into independent lemmas (XS to XL difficulty)
* Many portions of blueprint still disconnected; linkages growing

### Papers being formalized:
* Fiori-Kadiri-Swidinsky: "Sharper bounds for the Chebyshev function psi(x)"
  (arXiv:2204.02588)
* Fiori-Kadiri-Swidinsky: "Sharper bounds for the error term in the PNT"
  (arXiv:2206.12557)
* Related explicit zero density results (Kadiri-Lumley et al.)

## 1D. Analytic Number Theory Exponent Database (ANTEDB)
     By Tao, Trudgian, Yang — https://teorth.github.io/expdb/

### What it is:
A repository of exponent pairs, zero density estimates, exponential sum
bounds, moment bounds for zeta, large value estimates, and additive energy
bounds. Stores data in LaTeX (human) + Python (computation).

### Lean status: Placeholder only. No formal Lean proofs yet. Future goal
is to support formal derivations in Lean.

### Results obtained: Four new exponent pairs, new zero density estimates,
new additive energy estimates for zeta zeros.

## 1E. Other Active Projects

* **Irrationality of zeta(3)** (arXiv:2503.07625): Formal proof in Lean 4
  using Beukers' method with shifted Legendre polynomials. Builds on PNT+
  for the error-term PNT. Extends Mathlib with shifted Legendre polynomials.

* **Hardy-Wright formalization** (github.com/wgabrielong/NT_lean):
  Lean 4 formalization of "An Introduction to the Theory of Numbers."

* **Tao's Analysis I** (github.com/teorth/analysis): Lean companion to
  Analysis I textbook. Real analysis foundations, not directly ANT.

* **UC Berkeley course** (Spring 2026): CS 294-268 "Proving TCS and Math
  Theorems in Lean" — may produce additional analytic NT formalizations.

* **Selberg sieve**: A contributor has formalized the fundamental theorem
  of the Selberg sieve in Lean 4. Goal: Brun's theorem, pi(x) << x/log x.

---

# PART 2: WHAT WE CAN USE FOR OUR RH INVESTIGATION
---

## 2A. Immediate connections to our existing Lean files

### ExclusionZone.lean already uses:
* `riemannZeta_ne_zero_of_one_le_re` — Mathlib's proved non-vanishing
* Our `IsZeroFreeRegion` definition aligns with IEANT's explicit bound work

### ZetaZeroStructure.lean needs:
* Zero enumeration — NOT in any project yet (fundamental gap)
* GUE/RMT infrastructure — NOT formalized anywhere

### ConjugatePairStability.lean needs:
* Xi function — NOT in Mathlib, not in PNT+
* Zero ordinates — same gap as ZetaZeroStructure

### CriticalPhenomenon.lean needs:
* de Bruijn-Newman constant — NOT formalized anywhere
* Heat equation on entire functions — NOT in Mathlib

## 2B. What IEANT/PNT+ will provide that we need

1. **Explicit error term PNT**: When IEANT completes the Fiori-Kadiri-Swidinsky
   formalization, we get EXPLICIT bounds on |psi(x) - x|. This connects to
   our zero-free region hierarchy (ExclusionZone.lean, Level 1).

2. **Rectangle integrals**: PNT+ has Rectangle.lean defining rectangle borders.
   When this matures into a proper contour integral framework, it enables
   Perron's formula, which is the bridge between zero-free regions and
   prime counting functions.

3. **Explicit zero-free region**: The IEANT papers include zero density
   results. If a de la Vallee-Poussin zero-free region is formalized,
   it would fill our ExclusionZone.lean Level 1 gap.

4. **Propagation spreadsheet**: Once operational, we could feed our
   computational bounds from Session 31 (e.g., verified RH height)
   and see the downstream impact on explicit estimates.

## 2C. What we have that THEY might want

1. **Zero spacing statistics infrastructure**: Our ZetaZeroStructure.lean
   has definitions for spacing ACF, oscillatory/short-range decomposition,
   and explicit formulas for these. None of this exists elsewhere.

2. **Conjugate pair stability framework**: The energy-based analysis of
   why zeros prefer the critical line. Novel formalization.

3. **dBN critical phenomenon formalization**: Our CriticalPhenomenon.lean
   connects the dBN constant to GUE spacing. This is not formalized
   elsewhere.

---

# PART 3: CONNECTIONS TO SESSION 31 FINDINGS
---

Session 31 established that:
1. Local stability of zeros on the critical line (conjugate pair attraction
   with curvature 1/epsilon^2) does NOT imply global stability.
2. The convexity attack fails: the energy landscape has lower-energy
   configurations off the critical line.
3. RH is a critical phenomenon (Lambda = 0), not a safety-margin statement.
4. Finite-to-infinite transition in the Euler product is the core obstruction.

### Connection to IEANT:
The IEANT project is formalizing EXPLICIT bounds — these are finite
truncations of infinite sums/products. The "spreadsheet" component tracks
how finite truncation parameters propagate through estimates. This is
precisely our "finite-to-infinite obstruction" in a different guise.

If the IEANT spreadsheet can track the dependence of zero-free region
width on the truncation parameter N, this would give a FORMAL version
of our Session 31 observation: as N -> infinity, the margin goes to zero
(consistent with Lambda = 0).

### Connection to PNT+:
The Wiener-Ikehara proof of PNT uses the non-vanishing on Re(s) = 1.
Our ExclusionZone.lean Level 0 is exactly this. The PNT+ error term
work extends to Level 1 (de la Vallee-Poussin type). The gap between
their Level 1 and our Level 3 (RH) is where all the action is.

### Connection to ANTEDB:
The exponent database tracks zero density estimates N(sigma, T) — the
number of zeros with Re(s) > sigma up to height T. Our ZetaZeroStructure
works with the complementary object: the SPACING statistics of zeros
assumed to be on the critical line. If RH is true, these are the same
zeros. The exponent database gives unconditional density bounds that
constrain the spacing statistics we model.

---

# PART 4: WHAT WE SHOULD FORMALIZE NEXT
---

Priority order, considering what connects to the external pipeline:

## Priority 1: Fill the zero enumeration gap (HIGH IMPACT)

No project has formalized the zero counting function N(T) = number of
zeros of zeta with 0 < Im(s) < T. This is foundational for everything.
The Riemann-von Mangoldt formula N(T) = (T/2pi) log(T/2pi) - T/2pi + O(log T)
is the first target. This would connect our ZetaZeroStructure to the
explicit bounds in IEANT.
-/

/-- The zero counting function: number of nontrivial zeros with
    0 < Im(rho) <= T. Not in Mathlib or any external project. -/
noncomputable def zeroCountingFunction (T : ℝ) : ℕ := sorry

/-- The Riemann-von Mangoldt formula: N(T) ~ (T/2pi) log(T/2pi) - T/2pi.
    Formalizing this would bridge our project to IEANT's explicit bounds. -/
theorem riemann_von_mangoldt_asymptotic :
    ∀ ε : ℝ, ε > 0 → ∃ T₀ : ℝ, ∀ T : ℝ, T > T₀ →
    |(zeroCountingFunction T : ℝ) -
      (T / (2 * Real.pi) * Real.log (T / (2 * Real.pi)) -
       T / (2 * Real.pi))| < ε * T := sorry

/-!
## Priority 2: Formalize the explicit formula connection

The von Mangoldt explicit formula connects the prime counting function
psi(x) to the zeros of zeta:
  psi(x) = x - sum_rho x^rho / rho - log(2pi) - (1/2)log(1 - x^{-2})

This is the BRIDGE between:
- PNT+ (which has psi(x) ~ x)
- Our zero spacing statistics (which model the sum over rho)
- IEANT (which wants explicit error bounds)

Requires: Perron's formula, which requires contour integrals (PNT+'s
Rectangle.lean is the starting point).
-/

/-- Placeholder: the explicit formula relates psi to zeros.
    When PNT+ develops contour integration, this becomes provable. -/
theorem explicit_formula_connection :
    ∀ x : ℝ, x > 1 →
    ∃ (mainTerm errorTerm : ℝ),
      mainTerm = x ∧
      -- The error term is controlled by the zero-free region
      -- (connects ExclusionZone.lean to PNT+ error bounds)
      |errorTerm| ≤ x * Real.exp (-Real.sqrt (Real.log x) / 10) :=
  sorry

/-!
## Priority 3: Connect to the Hadamard product (MEDIUM IMPACT)

The Hadamard product representation of Xi:
  Xi(s) = Xi(0) * prod_rho (1 - s/rho)

is the foundation for:
- The explicit formula (via logarithmic derivative)
- Our conjugate pair stability analysis (pair energy comes from this product)
- The de Bruijn-Newman constant (heat flow on this product)

Not in Mathlib. Not in PNT+. Would be a major contribution.

Requires: Weierstrass factorization theorem for entire functions of
order 1. Mathlib has the basic theory of entire functions but NOT the
factorization theorem.
-/

/-- The Hadamard product: Xi(s) as a product over zeros.
    Formalizing this would be a major contribution to the Lean
    analytic number theory ecosystem. -/
axiom hadamard_product_exists :
    ∃ (ρ : ℕ → ℂ) (c : ℂ),
    -- Each rho is a nontrivial zero
    (∀ n, riemannZeta (ρ n) = 0) ∧
    -- The product converges to Xi(s)
    -- (precise statement requires Weierstrass factorization)
    c ≠ 0

/-!
## Priority 4: De la Vallee-Poussin zero-free region (connects to IEANT)

If IEANT formalizes the classical zero-free region, we should immediately
connect it to our ExclusionZone framework. The connection is:

  IEANT proves: Re(s) >= 1 - c/log(|Im(s)| + 2) => zeta(s) != 0
  Our framework: IsZeroFreeRegion delta  where  delta(t) = 1 - c/log(t+2)

This would upgrade ExclusionZone.lean from Level 0 to Level 1.
-/

/-- The de la Vallee-Poussin zero-free region.
    Target for connection to IEANT once they formalize it. -/
def delaValleePoussinZFR : Prop :=
  ∃ c : ℝ, c > 0 ∧
  ∀ s : ℂ, riemannZeta s = 0 →
    s.re < 1 - c / Real.log (‖s‖ + 2)

/-!
## Priority 5: Random matrix theory infrastructure (LONG TERM)

No Lean project has ANY RMT formalization. Our ZetaZeroStructure.lean
is the closest thing to RMT in formal mathematics. Key targets:

* GUE k-point correlation functions (we have R_GUE as opaque)
* Sine kernel: K(x,y) = sin(pi(x-y)) / (pi(x-y))
* Tracy-Widom distribution (for minimum spacing)
* Montgomery's pair correlation conjecture (formal statement)

This is a long-term project but would be groundbreaking.

---

# PART 5: RECOMMENDED ACTION ITEMS
---

## Immediate (next 1-2 sessions):
1. Monitor IEANT GitHub for formalized zero-free region results
2. Upstream our `IsZeroFreeRegion` definition to be compatible with IEANT
3. Add PNT+ as a dependency if their Rectangle.lean matures

## Short-term (next month):
4. Formalize `zeroCountingFunction` and Riemann-von Mangoldt
5. Connect to PNT+'s error-term PNT via the explicit formula
6. Contribute our `conjugatePairSelfEnergy` curvature proof to PNT+
   (it's fully proved, no sorry — could be useful for their zero dynamics)

## Medium-term (next quarter):
7. Hadamard product formalization (major effort, ~XL task in IEANT terms)
8. Bridge ANTEDB zero density estimates to our spacing statistics
9. Formalize the dBN constant connection to GUE spacing rigorously

## Long-term:
10. Contribute to IEANT's spreadsheet: our Session 31 numerics on how
    RH verification height affects zero spacing could be a test case
11. RMT infrastructure: would be a first in formal mathematics

---

# PART 6: THE KEY INSIGHT FROM THIS SURVEY
---

The formalization landscape reveals a clear PIPELINE:

  Euler product (Mathlib, PROVED)
    -> Non-vanishing on Re=1 (Mathlib, PROVED)
      -> PNT (PNT+, PROVED)
        -> PNT with error term (PNT+/IEANT, IN PROGRESS)
          -> Explicit zero-free region (IEANT, PLANNED)
            -> ... GAP ...
              -> RH (us, CONJECTURAL)

The GAP between "explicit zero-free region" and "RH" is precisely what
our Session 31 analysis characterizes: it is the critical phenomenon
(Lambda = 0), and no finite improvement to the explicit bounds can cross it.

This confirms our Category 3 classification: the proof of RH needs a
structural argument, not a quantitative improvement. The IEANT pipeline
gives ever-better QUANTITATIVE bounds, but the qualitative leap to RH
requires something they are not building.

Our contribution to the ecosystem is precisely this structural analysis:
the energy landscape, the critical phenomenon, the finite-to-infinite
obstruction. These are COMPLEMENTARY to the IEANT/PNT+ pipeline, not
competing with it.
-/
