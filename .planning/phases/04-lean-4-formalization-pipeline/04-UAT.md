---
status: complete
phase: 04-lean-4-formalization-pipeline
source: [04-01-SUMMARY.md, 04-02-SUMMARY.md, 04-03-SUMMARY.md]
started: 2026-03-19T21:00:00Z
updated: 2026-03-19T21:15:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Lean 4 Builds from Python
expected: run_lake_build() returns LakeBuildResult with success=True, sorry_count=0
result: pass

### 2. Hello-World Theorem Validates Zeta and RH
expected: Basic.lean contains a theorem referencing riemannZeta_zero and RiemannHypothesis compiles without error
result: pass

### 3. Output Parser Extracts Structured Messages
expected: parse_lean_output on Lean compiler output returns list of LeanMessage with file/line/severity fields, and sorry count
result: pass

### 4. Formalization State Machine Tracks 4 States
expected: Creating a formalization starts at not_formalized, can transition through statement_formalized -> proof_attempted -> proof_complete, invalid transitions raise ValueError
result: pass

### 5. Auto-Promotion on Clean Build
expected: Recording a build with sorry_count=0 and success=True auto-promotes conjecture evidence_level to 3 (FORMAL_PROOF)
result: pass

### 6. Translator Generates Lean 4 from Conjecture
expected: translate_conjecture produces Lean 4 source with Mathlib imports, evidence-mapping docstring, and sorry-placeholder theorem body
result: pass

### 7. Triage Scores Conjectures by Viability
expected: triage_conjectures returns TriageEntry list sorted by score, with confidence/proximity/continuation/novelty factors
result: pass

### 8. Formalization Package Exports Complete API
expected: from riemann.formalization imports at least 15 public names including run_lake_build, FormalizationState, generate_lean_file, triage_conjectures, run_formalization_assault
result: pass

## Summary

total: 8
passed: 8
issues: 0
pending: 0
skipped: 0

## Gaps

[none]
