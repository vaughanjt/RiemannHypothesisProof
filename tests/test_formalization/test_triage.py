"""Tests for conjecture triage and formalization assault runner."""
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from riemann.workbench.conjecture import create_conjecture
from riemann.workbench.db import init_db


@pytest.fixture
def tmp_db():
    """Temporary SQLite database for triage tests."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    init_db(db_path)
    try:
        yield db_path
    finally:
        import gc
        gc.collect()
        try:
            os.unlink(db_path)
        except PermissionError:
            pass


# ---------- triage_conjectures tests ----------


def test_triage_empty_workbench(tmp_db):
    """Empty workbench returns empty triage list."""
    from riemann.formalization.triage import triage_conjectures

    entries = triage_conjectures(db_path=tmp_db)
    assert entries == []


def test_triage_excludes_proved(tmp_db):
    """Conjectures with evidence_level >= 3 are excluded from triage."""
    from riemann.formalization.triage import triage_conjectures

    # evidence_level=3 => FORMAL_PROOF => excluded
    create_conjecture(
        statement="Already proved conjecture",
        evidence_level=3,
        status="proved",
        confidence=0.99,
        db_path=tmp_db,
    )
    # evidence_level=1 => should appear
    create_conjecture(
        statement="Heuristic conjecture",
        evidence_level=1,
        status="speculative",
        confidence=0.5,
        db_path=tmp_db,
    )

    entries = triage_conjectures(db_path=tmp_db)
    assert len(entries) == 1
    assert entries[0].confidence == 0.5


def test_triage_scores_high_confidence_higher(tmp_db):
    """Higher confidence conjectures score higher."""
    from riemann.formalization.triage import triage_conjectures

    create_conjecture(
        statement="Low confidence eigenvalue gap",
        evidence_level=0,
        status="speculative",
        confidence=0.3,
        tags=["eigenvalue"],
        db_path=tmp_db,
    )
    create_conjecture(
        statement="High confidence eigenvalue result",
        evidence_level=0,
        status="speculative",
        confidence=0.9,
        tags=["eigenvalue"],
        db_path=tmp_db,
    )

    entries = triage_conjectures(db_path=tmp_db)
    assert len(entries) == 2
    assert entries[0].confidence == 0.9
    assert entries[1].confidence == 0.3
    assert entries[0].score > entries[1].score


def test_triage_spectral_scores_above_tda(tmp_db):
    """Spectral domain (high Mathlib coverage) scores above TDA (low coverage)."""
    from riemann.formalization.triage import triage_conjectures

    create_conjecture(
        statement="Persistent homology of zero gaps",
        evidence_level=0,
        status="speculative",
        confidence=0.5,
        tags=["persistent homology"],
        db_path=tmp_db,
    )
    create_conjecture(
        statement="Eigenvalue spacing distribution",
        evidence_level=0,
        status="speculative",
        confidence=0.5,
        tags=["eigenvalue"],
        db_path=tmp_db,
    )

    entries = triage_conjectures(db_path=tmp_db)
    assert len(entries) == 2
    # Spectral (eigenvalue) should rank higher than TDA (persistent homology)
    spectral = [e for e in entries if e.domain == "spectral"]
    tda = [e for e in entries if e.domain == "tda"]
    assert len(spectral) == 1 and len(tda) == 1
    assert spectral[0].score > tda[0].score


def test_triage_continuation_bonus(tmp_db):
    """Conjectures in proof_attempted state get continuation bonus."""
    from riemann.formalization.tracker import create_formalization, update_formalization_state
    from riemann.formalization.triage import triage_conjectures

    # Conjecture A: has formalization in proof_attempted
    cid_a = create_conjecture(
        statement="Conjecture A with ongoing proof",
        evidence_level=0,
        status="speculative",
        confidence=0.5,
        tags=["eigenvalue"],
        db_path=tmp_db,
    )
    fid = create_formalization(cid_a, "/fake/path.lean", db_path=tmp_db)
    update_formalization_state(fid, "statement_formalized", db_path=tmp_db)
    update_formalization_state(fid, "proof_attempted", db_path=tmp_db)

    # Conjecture B: untouched, same confidence
    create_conjecture(
        statement="Conjecture B untouched",
        evidence_level=0,
        status="speculative",
        confidence=0.5,
        tags=["eigenvalue"],
        db_path=tmp_db,
    )

    entries = triage_conjectures(db_path=tmp_db)
    assert len(entries) == 2
    # A should rank higher due to continuation bonus
    assert entries[0].conjecture_id == cid_a
    assert entries[0].formalization_state == "proof_attempted"


def test_triage_entry_fields(tmp_db):
    """TriageEntry has all expected fields populated."""
    from riemann.formalization.triage import TriageEntry, triage_conjectures

    create_conjecture(
        statement="Test conjecture for field check",
        evidence_level=0,
        status="speculative",
        confidence=0.6,
        tags=["eigenvalue"],
        db_path=tmp_db,
    )

    entries = triage_conjectures(db_path=tmp_db)
    assert len(entries) == 1
    entry = entries[0]

    assert isinstance(entry, TriageEntry)
    assert entry.conjecture_id  # non-empty string
    assert isinstance(entry.score, float)
    assert entry.confidence == 0.6
    assert isinstance(entry.domain, str) and entry.domain
    assert isinstance(entry.reason, str)
    # formalization_state is None for un-formalized conjecture
    assert entry.formalization_state is None


# ---------- run_formalization_assault tests ----------


def test_run_formalization_assault_empty(tmp_db):
    """Empty workbench returns AssaultResult with total=0."""
    from riemann.formalization.triage import run_formalization_assault

    result = run_formalization_assault(db_path=tmp_db)
    assert result.total_conjectures == 0
    assert result.attempted == 0
    assert result.outcomes == []


@patch("riemann.formalization.triage.run_lake_build")
@patch("riemann.formalization.triage.generate_lean_file")
def test_run_formalization_assault_with_mock_build(mock_gen, mock_build, tmp_db):
    """Assault with a mocked build creates outcomes."""
    from riemann.formalization.builder import LakeBuildResult
    from riemann.formalization.tracker import create_formalization, update_formalization_state
    from riemann.formalization.triage import run_formalization_assault

    cid = create_conjecture(
        statement="Mock assault target",
        evidence_level=0,
        status="speculative",
        confidence=0.7,
        db_path=tmp_db,
    )

    # Mock generate_lean_file to create a real formalization record
    from riemann.formalization.translator import TranslationResult
    fid = create_formalization(cid, "/fake/path.lean", db_path=tmp_db)
    update_formalization_state(fid, "statement_formalized", sorry_count=1, db_path=tmp_db)

    mock_gen.return_value = TranslationResult(
        lean_code="sorry",
        lean_file_path="/fake/path.lean",
        formalization_id=fid,
        mathlib_imports=[],
        conjecture_id=cid,
    )

    mock_build.return_value = LakeBuildResult(
        success=True, returncode=0, output="ok",
        sorry_count=1, error_count=0, warning_count=0, duration_ms=100,
    )

    result = run_formalization_assault(max_conjectures=1, max_attempts_per=3, db_path=tmp_db)
    assert result.attempted >= 1
    assert len(result.outcomes) >= 1
    outcome = result.outcomes[0]
    assert outcome.conjecture_id == cid
    assert outcome.formalization_id is not None
    assert outcome.builds_attempted >= 1


@patch("riemann.formalization.triage.run_lake_build")
def test_assault_skips_proof_complete(mock_build, tmp_db):
    """Conjectures already at proof_complete are skipped."""
    from riemann.formalization.tracker import create_formalization, update_formalization_state
    from riemann.formalization.triage import run_formalization_assault

    cid = create_conjecture(
        statement="Already complete",
        evidence_level=0,
        status="speculative",
        confidence=0.8,
        db_path=tmp_db,
    )

    fid = create_formalization(cid, "/fake/path.lean", db_path=tmp_db)
    update_formalization_state(fid, "statement_formalized", db_path=tmp_db)
    update_formalization_state(fid, "proof_attempted", db_path=tmp_db)
    update_formalization_state(fid, "proof_complete", db_path=tmp_db)

    result = run_formalization_assault(max_conjectures=1, db_path=tmp_db)
    # Should be excluded from triage (proof_complete) and skipped
    assert result.attempted == 0
    mock_build.assert_not_called()


@patch("riemann.formalization.triage.run_lake_build")
@patch("riemann.formalization.triage.generate_lean_file")
def test_assault_timebox(mock_gen, mock_build, tmp_db):
    """Assault stops after max_attempts_per when sorry count doesn't decrease."""
    from riemann.formalization.builder import LakeBuildResult
    from riemann.formalization.tracker import create_formalization, update_formalization_state
    from riemann.formalization.triage import run_formalization_assault

    cid = create_conjecture(
        statement="Stuck conjecture",
        evidence_level=0,
        status="speculative",
        confidence=0.7,
        db_path=tmp_db,
    )

    from riemann.formalization.translator import TranslationResult
    fid = create_formalization(cid, "/fake/path.lean", db_path=tmp_db)
    update_formalization_state(fid, "statement_formalized", sorry_count=5, db_path=tmp_db)

    mock_gen.return_value = TranslationResult(
        lean_code="sorry",
        lean_file_path="/fake/path.lean",
        formalization_id=fid,
        mathlib_imports=[],
        conjecture_id=cid,
    )

    # Always returns sorry_count=5 (never decreasing)
    mock_build.return_value = LakeBuildResult(
        success=False, returncode=1, output="sorry",
        sorry_count=5, error_count=0, warning_count=0, duration_ms=100,
    )

    result = run_formalization_assault(
        max_conjectures=1, max_attempts_per=3, db_path=tmp_db,
    )
    outcome = result.outcomes[0]
    # Should stop after 2 builds (first build sets last_sorry=5, second build sees 5 >= 5 and breaks)
    assert outcome.builds_attempted <= 3
    assert outcome.sorry_count == 5


@patch("riemann.formalization.triage.run_lake_build")
@patch("riemann.formalization.triage.generate_lean_file")
def test_assault_advances_state_to_proof_attempted(mock_gen, mock_build, tmp_db):
    """CRITICAL: Assault advances state from statement_formalized to proof_attempted before building."""
    from riemann.formalization.builder import LakeBuildResult
    from riemann.formalization.tracker import (
        create_formalization,
        get_formalization,
        update_formalization_state,
    )
    from riemann.formalization.triage import run_formalization_assault

    cid = create_conjecture(
        statement="State machine test",
        evidence_level=0,
        status="speculative",
        confidence=0.8,
        db_path=tmp_db,
    )

    from riemann.formalization.translator import TranslationResult
    fid = create_formalization(cid, "/fake/path.lean", db_path=tmp_db)
    update_formalization_state(fid, "statement_formalized", sorry_count=1, db_path=tmp_db)

    mock_gen.return_value = TranslationResult(
        lean_code="sorry",
        lean_file_path="/fake/path.lean",
        formalization_id=fid,
        mathlib_imports=[],
        conjecture_id=cid,
    )

    # Clean build: success=True, sorry_count=0
    mock_build.return_value = LakeBuildResult(
        success=True, returncode=0, output="ok",
        sorry_count=0, error_count=0, warning_count=0, duration_ms=100,
    )

    result = run_formalization_assault(
        max_conjectures=1, max_attempts_per=3, db_path=tmp_db,
    )
    outcome = result.outcomes[0]

    # Full traversal: statement_formalized -> proof_attempted -> proof_complete
    assert outcome.final_state == "proof_complete"

    # Verify the formalization is actually at proof_complete in DB
    form = get_formalization(fid, db_path=tmp_db)
    assert form["formalization_state"] == "proof_complete"


@patch("riemann.formalization.triage.run_lake_build")
@patch("riemann.formalization.triage.generate_lean_file")
def test_assault_promotes_on_clean_build(mock_gen, mock_build, tmp_db):
    """End-to-end: clean build promotes to proof_complete and updates evidence_level to 3."""
    from riemann.formalization.builder import LakeBuildResult
    from riemann.formalization.tracker import create_formalization, update_formalization_state
    from riemann.formalization.triage import run_formalization_assault
    from riemann.workbench.conjecture import get_conjecture

    cid = create_conjecture(
        statement="Auto-promotion test",
        evidence_level=0,
        status="speculative",
        confidence=0.9,
        db_path=tmp_db,
    )

    from riemann.formalization.translator import TranslationResult
    fid = create_formalization(cid, "/fake/path.lean", db_path=tmp_db)
    update_formalization_state(fid, "statement_formalized", sorry_count=1, db_path=tmp_db)

    mock_gen.return_value = TranslationResult(
        lean_code="sorry",
        lean_file_path="/fake/path.lean",
        formalization_id=fid,
        mathlib_imports=[],
        conjecture_id=cid,
    )

    mock_build.return_value = LakeBuildResult(
        success=True, returncode=0, output="ok",
        sorry_count=0, error_count=0, warning_count=0, duration_ms=100,
    )

    result = run_formalization_assault(
        max_conjectures=1, max_attempts_per=3, db_path=tmp_db,
    )
    outcome = result.outcomes[0]
    assert outcome.final_state == "proof_complete"

    # Verify conjecture evidence_level was promoted to 3 (FORMAL_PROOF)
    conj = get_conjecture(cid, db_path=tmp_db)
    assert conj["evidence_level"] == 3
    assert conj["status"] == "proved"
