import Mathlib.NumberTheory.LSeries.RiemannZeta

/-!
# Basic Lean 4 + Mathlib Validation
Validates that:
1. Lean 4 compiles inside WSL2
2. Mathlib imports resolve
3. `lake build` succeeds from Python subprocess
-/

open Complex

/-- Validate Mathlib's formalized zeta: zeta(0) = -1/2. -/
example : riemannZeta 0 = -1 / 2 := riemannZeta_zero

/-- Validate RiemannHypothesis is accessible as a Prop. -/
example : Prop := RiemannHypothesis
