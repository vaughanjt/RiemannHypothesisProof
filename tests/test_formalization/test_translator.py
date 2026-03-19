"""Tests for the conjecture-to-Lean 4 translator.

Covers: translation output, evidence mapping, domain inference,
file generation, formalization registration, and edge cases.
"""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from riemann.workbench.conjecture import create_conjecture, get_conjecture
from riemann.workbench.db import get_connection, init_db
from riemann.workbench.evidence import link_evidence


@pytest.fixture
def tmp_db(temp_db):
    """Initialize a temp DB with all tables."""
    init_db(temp_db)
    return temp_db


def _create_experiment(db_path, description="test experiment", params=None):
    """Helper to create an experiment record directly."""
    exp_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        conn.execute("""
            INSERT INTO experiments (id, description, parameters, created_at)
            VALUES (?, ?, ?, ?)
        """, (exp_id, description, json.dumps(params or {}), now))
    return exp_id


def test_translate_conjecture_basic(tmp_db):
    """Translate a conjecture and verify output structure."""
    from riemann.formalization.translator import translate_conjecture

    conj_id = create_conjecture(
        "The zeros of the zeta function lie on the critical line",
        description="Riemann Hypothesis",
        db_path=tmp_db,
    )
    lean_code = translate_conjecture(conj_id, db_path=tmp_db)

    # Must contain evidence docstring
    assert "/-!" in lean_code
    assert "-/" in lean_code
    assert conj_id in lean_code

    # Must contain import
    assert "import Mathlib.NumberTheory.LSeries.RiemannZeta" in lean_code

    # Must contain theorem with sorry
    assert "theorem" in lean_code
    assert "sorry" in lean_code


def test_translate_conjecture_not_found(tmp_db):
    """Translating a non-existent conjecture raises ValueError."""
    from riemann.formalization.translator import translate_conjecture

    with pytest.raises(ValueError, match="not found"):
        translate_conjecture("nonexistent-id", db_path=tmp_db)


def test_evidence_mapping_in_docstring(tmp_db):
    """Evidence from linked experiments appears in the docstring."""
    from riemann.formalization.translator import translate_conjecture

    conj_id = create_conjecture(
        "Spectral operator eigenvalues correlate with zeros",
        db_path=tmp_db,
    )
    exp_id = _create_experiment(
        tmp_db, description="Eigenvalue computation for N=1000",
    )
    link_evidence(conj_id, exp_id, "supports", strength=0.85, db_path=tmp_db)

    lean_code = translate_conjecture(conj_id, db_path=tmp_db)

    # Experiment ID and description should appear in docstring
    assert exp_id in lean_code
    assert "Eigenvalue computation" in lean_code
    assert "supports" in lean_code


def test_domain_inference_spectral(tmp_db):
    """Conjecture with spectral keywords gets spectral imports."""
    from riemann.formalization.translator import translate_conjecture

    conj_id = create_conjecture(
        "The eigenvalue distribution of the operator matches GUE",
        db_path=tmp_db,
    )
    lean_code = translate_conjecture(conj_id, db_path=tmp_db)

    assert "import Mathlib.Analysis.SpecialFunctions.Gamma.Deligne" in lean_code


def test_domain_inference_modular(tmp_db):
    """Conjecture with modular tag gets modular form imports."""
    from riemann.formalization.translator import translate_conjecture

    conj_id = create_conjecture(
        "The Hecke eigenforms satisfy the expected bound",
        tags=["modular"],
        db_path=tmp_db,
    )
    lean_code = translate_conjecture(conj_id, db_path=tmp_db)

    assert "import Mathlib.NumberTheory.ModularForms.JacobiTheta.Basic" in lean_code


def test_domain_inference_default(tmp_db):
    """Generic conjecture gets base imports only."""
    from riemann.formalization.translator import translate_conjecture

    conj_id = create_conjecture(
        "Some generic mathematical statement",
        db_path=tmp_db,
    )
    lean_code = translate_conjecture(conj_id, db_path=tmp_db)

    assert "import Mathlib.NumberTheory.LSeries.RiemannZeta" in lean_code
    # Should NOT have domain-specific imports
    assert "Gamma.Deligne" not in lean_code
    assert "JacobiTheta" not in lean_code
    assert "PadicVal" not in lean_code


def test_generate_lean_file(tmp_db, tmp_path):
    """generate_lean_file creates file on disk and returns TranslationResult."""
    from riemann.formalization.translator import TranslationResult, generate_lean_file

    conj_id = create_conjecture(
        "Test theorem statement",
        db_path=tmp_db,
    )
    result = generate_lean_file(conj_id, project_dir=tmp_path, db_path=tmp_db)

    assert isinstance(result, TranslationResult)
    assert result.conjecture_id == conj_id
    assert result.formalization_id is not None
    assert result.lean_code is not None
    assert len(result.mathlib_imports) >= 1

    # File should exist on disk
    lean_file = Path(result.lean_file_path)
    assert lean_file.exists()
    assert lean_file.read_text(encoding="utf-8") == result.lean_code


def test_generate_lean_file_creates_formalization(tmp_db, tmp_path):
    """generate_lean_file registers a formalization record in the DB."""
    from riemann.formalization.tracker import get_formalization
    from riemann.formalization.translator import generate_lean_file

    conj_id = create_conjecture(
        "Test theorem for formalization record",
        db_path=tmp_db,
    )
    result = generate_lean_file(conj_id, project_dir=tmp_path, db_path=tmp_db)

    form = get_formalization(result.formalization_id, db_path=tmp_db)
    assert form is not None
    assert form["conjecture_id"] == conj_id
    assert form["lean_file_path"] == result.lean_file_path


def test_generate_lean_file_state_is_statement_formalized(tmp_db, tmp_path):
    """After generate_lean_file, formalization state is statement_formalized."""
    from riemann.formalization.tracker import get_formalization
    from riemann.formalization.translator import generate_lean_file

    conj_id = create_conjecture(
        "Test theorem for state check",
        db_path=tmp_db,
    )
    result = generate_lean_file(conj_id, project_dir=tmp_path, db_path=tmp_db)

    form = get_formalization(result.formalization_id, db_path=tmp_db)
    assert form["formalization_state"] == "statement_formalized"
    assert form["sorry_count"] == 1  # Generated theorem has sorry


def test_sanitize_id(tmp_db):
    """UUID with hyphens becomes valid Lean identifier with C_ prefix."""
    from riemann.formalization.translator import _sanitize_id

    result = _sanitize_id("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
    assert result == "C_a1b2c3d4_e5f6_7890_abcd_ef1234567890"
    assert "-" not in result
    assert result.startswith("C_")


def test_mathlib_imports_in_output(tmp_db):
    """Generated code contains the base Mathlib import."""
    from riemann.formalization.translator import translate_conjecture

    conj_id = create_conjecture(
        "Any mathematical conjecture",
        db_path=tmp_db,
    )
    lean_code = translate_conjecture(conj_id, db_path=tmp_db)

    assert "import Mathlib.NumberTheory.LSeries.RiemannZeta" in lean_code
