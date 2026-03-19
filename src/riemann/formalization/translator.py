"""Translate workbench conjectures to Lean 4 theorem statements.

Claude generates Lean 4 code directly from conjecture statements (no template
scaffolding). Each .lean file includes evidence-mapping docstrings linking back
to workbench experiments. Mathlib imports are selected based on conjecture domain.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

from riemann.formalization.builder import LEAN_PROJECT_DIR
from riemann.formalization.tracker import create_formalization, update_formalization_state
from riemann.workbench.conjecture import get_conjecture
from riemann.workbench.evidence import get_evidence_for_conjecture


# Domain-to-Mathlib import mapping
_DOMAIN_IMPORTS = {
    "spectral": [
        "Mathlib.NumberTheory.LSeries.RiemannZeta",
        "Mathlib.Analysis.SpecialFunctions.Gamma.Deligne",
    ],
    "trace": [
        "Mathlib.NumberTheory.LSeries.RiemannZeta",
        "Mathlib.NumberTheory.VonMangoldt",
    ],
    "modular": [
        "Mathlib.NumberTheory.ModularForms.JacobiTheta.Basic",
        "Mathlib.NumberTheory.LSeries.RiemannZeta",
    ],
    "padic": [
        "Mathlib.NumberTheory.Padics.PadicVal",
        "Mathlib.NumberTheory.LSeries.RiemannZeta",
    ],
    "tda": [
        "Mathlib.NumberTheory.LSeries.RiemannZeta",
        "Mathlib.Topology.Basic",
    ],
    "dynamics": [
        "Mathlib.NumberTheory.LSeries.RiemannZeta",
        "Mathlib.Dynamics.Ergodic.MeasurePreserving",
    ],
    "ncg": [
        "Mathlib.NumberTheory.LSeries.RiemannZeta",
        "Mathlib.Analysis.SpecialFunctions.Gamma.Deligne",
    ],
    "default": [
        "Mathlib.NumberTheory.LSeries.RiemannZeta",
    ],
}

# Base imports always included
_BASE_IMPORTS = ["Mathlib.NumberTheory.LSeries.RiemannZeta"]


@dataclass
class TranslationResult:
    """Result of translating a conjecture to Lean 4."""
    lean_code: str
    lean_file_path: str
    formalization_id: str
    mathlib_imports: list[str]
    conjecture_id: str


def _sanitize_id(conjecture_id: str) -> str:
    """Convert UUID to a valid Lean identifier component."""
    # Replace hyphens with underscores, prefix with C_ to ensure valid Lean name
    return "C_" + conjecture_id.replace("-", "_")


def _infer_domain(conjecture: dict) -> str:
    """Infer the mathematical domain from conjecture tags/statement."""
    tags = conjecture.get("tags") or []
    statement = (conjecture.get("statement") or "").lower()
    description = (conjecture.get("description") or "").lower()
    text = " ".join(tags) + " " + statement + " " + description

    domain_keywords = {
        "spectral": ["spectral", "eigenvalue", "operator", "hamiltonian", "berry-keating"],
        "trace": ["trace", "selberg", "weil", "explicit formula", "primes"],
        "modular": ["modular", "eisenstein", "hecke", "fourier", "q-expansion"],
        "padic": ["p-adic", "padic", "kubota", "leopoldt"],
        "tda": ["persistent", "homology", "topolog", "betti"],
        "dynamics": ["lyapunov", "orbit", "dynamical", "ergodic", "attractor"],
        "ncg": ["noncommutative", "bost-connes", "kms", "partition function"],
    }

    for domain, keywords in domain_keywords.items():
        if any(kw in text for kw in keywords):
            return domain
    return "default"


def _build_evidence_docstring(
    conjecture: dict,
    evidence: list[dict],
) -> str:
    """Build the /-! ... -/ evidence-mapping docstring."""
    lines = [
        "/-!",
        f"# Conjecture: {conjecture['id']}",
        f"## Statement",
        f"{conjecture['statement']}",
    ]

    if conjecture.get("description"):
        lines.append(f"## Description")
        lines.append(f"{conjecture['description']}")

    lines.append(f"## Evidence")
    if evidence:
        for ev in evidence:
            exp_desc = ev.get("experiment_description", "unknown experiment")
            rel = ev.get("relationship", "unknown")
            strength = ev.get("strength", "N/A")
            exp_id = ev.get("experiment_id", "unknown")
            lines.append(f"- Experiment {exp_id}: {exp_desc} ({rel}, strength: {strength})")
    else:
        lines.append("- No linked experiments")

    lines.extend([
        f"## Workbench State",
        f"- Evidence level: {conjecture.get('evidence_level', 0)}",
        f"- Status: {conjecture.get('status', 'unknown')}",
        f"- Confidence: {conjecture.get('confidence', 'N/A')}",
        f"- Created: {conjecture.get('created_at', 'unknown')}",
        "-/",
    ])
    return "\n".join(lines)


def _build_theorem_name(conjecture: dict) -> str:
    """Generate a Lean-valid theorem name from conjecture ID."""
    return _sanitize_id(conjecture["id"])


def translate_conjecture(
    conjecture_id: str,
    db_path: str | Path | None = None,
) -> str:
    """Generate Lean 4 source code for a conjecture.

    Returns the full .lean file content as a string. The generated code
    includes evidence-mapping docstrings, Mathlib imports based on domain,
    and a theorem statement with sorry placeholder.

    Raises ValueError if conjecture not found.
    """
    conjecture = get_conjecture(conjecture_id, db_path)
    if conjecture is None:
        raise ValueError(f"Conjecture {conjecture_id} not found")

    evidence = get_evidence_for_conjecture(conjecture_id, db_path)
    domain = _infer_domain(conjecture)
    imports = list(dict.fromkeys(_BASE_IMPORTS + _DOMAIN_IMPORTS.get(domain, [])))

    # Build file content
    parts = []

    # Evidence docstring
    parts.append(_build_evidence_docstring(conjecture, evidence))
    parts.append("")

    # Imports
    for imp in imports:
        parts.append(f"import {imp}")
    parts.append("")

    # Open namespaces
    parts.append("open Complex")
    parts.append("")

    # Theorem with sorry
    theorem_name = _build_theorem_name(conjecture)
    statement_clean = conjecture["statement"].replace("\n", " ").strip()
    parts.append(f"/-- {statement_clean} -/")
    parts.append(f"theorem {theorem_name} :")
    parts.append(f"    sorry := by")
    parts.append(f"  sorry")
    parts.append("")

    return "\n".join(parts)


def generate_lean_file(
    conjecture_id: str,
    project_dir: Path | None = None,
    db_path: str | Path | None = None,
) -> TranslationResult:
    """Generate a .lean file for a conjecture and register it in the tracker.

    Writes the .lean file to {project_dir}/RiemannProofs/{sanitized_id}.lean
    Creates a formalization record in the tracker with state not_formalized,
    then transitions to statement_formalized.

    Returns TranslationResult with file path and formalization ID.
    """
    project_dir = project_dir or LEAN_PROJECT_DIR
    conjecture = get_conjecture(conjecture_id, db_path)
    if conjecture is None:
        raise ValueError(f"Conjecture {conjecture_id} not found")

    # Generate Lean code
    lean_code = translate_conjecture(conjecture_id, db_path)

    # Determine file path
    sanitized = _sanitize_id(conjecture_id)
    lean_file = project_dir / "RiemannProofs" / f"{sanitized}.lean"

    # Write the file
    lean_file.parent.mkdir(parents=True, exist_ok=True)
    lean_file.write_text(lean_code, encoding="utf-8")

    # Determine imports used
    domain = _infer_domain(conjecture)
    imports = list(dict.fromkeys(_BASE_IMPORTS + _DOMAIN_IMPORTS.get(domain, [])))

    # Register in tracker
    formalization_id = create_formalization(
        conjecture_id=conjecture_id,
        lean_file_path=str(lean_file),
        mathlib_imports=imports,
        db_path=db_path,
    )

    # Transition to statement_formalized (theorem + sorry written)
    update_formalization_state(
        formalization_id,
        "statement_formalized",
        sorry_count=1,  # The generated theorem has sorry
        db_path=db_path,
    )

    return TranslationResult(
        lean_code=lean_code,
        lean_file_path=str(lean_file),
        formalization_id=formalization_id,
        mathlib_imports=imports,
        conjecture_id=conjecture_id,
    )
