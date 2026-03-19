"""Tests for the formalization lifecycle tracker.

Covers: state machine transitions, build history, auto-promotion,
filtering, and error handling.
"""
import pytest

from riemann.formalization.builder import LakeBuildResult
from riemann.formalization.parser import LeanMessage
from riemann.formalization.tracker import (
    FormalizationState,
    auto_promote_if_clean,
    create_formalization,
    get_build_history,
    get_formalization,
    list_formalizations,
    record_build,
    update_formalization_state,
)
from riemann.workbench.conjecture import create_conjecture, get_conjecture
from riemann.workbench.db import init_db


@pytest.fixture
def tmp_db(temp_db):
    """Initialize a temp DB with all tables (including formalizations)."""
    init_db(temp_db)
    return temp_db


def test_create_formalization(tmp_db):
    """Create a formalization record and verify fields via get_formalization."""
    form_id = create_formalization(
        "conj-123", "/path/to/file.lean", db_path=tmp_db,
    )
    assert form_id is not None
    form = get_formalization(form_id, db_path=tmp_db)
    assert form is not None
    assert form["conjecture_id"] == "conj-123"
    assert form["lean_file_path"] == "/path/to/file.lean"
    assert form["formalization_state"] == "not_formalized"
    assert form["sorry_count"] == 0
    assert form["error_count"] == 0
    assert form["created_at"] is not None
    assert form["updated_at"] is not None


def test_valid_state_transitions(tmp_db):
    """Walk the full happy path: not_formalized -> statement_formalized -> proof_attempted -> proof_complete."""
    form_id = create_formalization("conj-1", "/f.lean", db_path=tmp_db)

    # not_formalized -> statement_formalized
    update_formalization_state(form_id, "statement_formalized", sorry_count=3, db_path=tmp_db)
    form = get_formalization(form_id, db_path=tmp_db)
    assert form["formalization_state"] == "statement_formalized"
    assert form["sorry_count"] == 3

    # statement_formalized -> proof_attempted
    update_formalization_state(form_id, "proof_attempted", sorry_count=2, db_path=tmp_db)
    form = get_formalization(form_id, db_path=tmp_db)
    assert form["formalization_state"] == "proof_attempted"
    assert form["sorry_count"] == 2

    # proof_attempted -> proof_complete
    update_formalization_state(form_id, "proof_complete", sorry_count=0, db_path=tmp_db)
    form = get_formalization(form_id, db_path=tmp_db)
    assert form["formalization_state"] == "proof_complete"
    assert form["sorry_count"] == 0


def test_proof_attempted_self_transition(tmp_db):
    """proof_attempted -> proof_attempted (iterating on proof) is allowed."""
    form_id = create_formalization("conj-1", "/f.lean", db_path=tmp_db)
    update_formalization_state(form_id, "statement_formalized", db_path=tmp_db)
    update_formalization_state(form_id, "proof_attempted", sorry_count=5, db_path=tmp_db)

    # Self-transition with reduced sorry count
    update_formalization_state(form_id, "proof_attempted", sorry_count=2, db_path=tmp_db)
    form = get_formalization(form_id, db_path=tmp_db)
    assert form["formalization_state"] == "proof_attempted"
    assert form["sorry_count"] == 2


def test_invalid_state_transition_not_formalized_to_proof_attempted(tmp_db):
    """Cannot skip from not_formalized directly to proof_attempted."""
    form_id = create_formalization("conj-1", "/f.lean", db_path=tmp_db)
    with pytest.raises(ValueError, match="Invalid transition"):
        update_formalization_state(form_id, "proof_attempted", db_path=tmp_db)


def test_invalid_state_transition_not_formalized_to_proof_complete(tmp_db):
    """Cannot skip from not_formalized directly to proof_complete."""
    form_id = create_formalization("conj-1", "/f.lean", db_path=tmp_db)
    with pytest.raises(ValueError, match="Invalid transition"):
        update_formalization_state(form_id, "proof_complete", db_path=tmp_db)


def test_record_build(tmp_db):
    """Record a build and verify it appears in build history."""
    form_id = create_formalization("conj-1", "/f.lean", db_path=tmp_db)
    build_result = LakeBuildResult(
        success=False,
        returncode=1,
        output="error output",
        messages=[
            LeanMessage("file.lean", 10, 4, "error", "unknown identifier"),
        ],
        sorry_count=2,
        error_count=1,
        warning_count=1,
    )
    build_id = record_build(form_id, build_result, db_path=tmp_db)
    assert build_id is not None

    history = get_build_history(form_id, db_path=tmp_db)
    assert len(history) == 1
    assert history[0]["sorry_count"] == 2
    assert history[0]["error_count"] == 1
    assert history[0]["success"] == 0  # SQLite stores booleans as ints

    # Verify formalization's last_build fields are updated
    form = get_formalization(form_id, db_path=tmp_db)
    assert form["sorry_count"] == 2
    assert form["error_count"] == 1


def test_multiple_builds_tracked(tmp_db):
    """Record 3 builds and verify all appear in history, newest first."""
    form_id = create_formalization("conj-1", "/f.lean", db_path=tmp_db)

    for i, (sorry, success) in enumerate([(5, False), (2, False), (0, True)]):
        record_build(
            form_id,
            LakeBuildResult(
                success=success, returncode=0 if success else 1,
                output=f"build {i}", sorry_count=sorry,
                error_count=0, warning_count=sorry,
            ),
            db_path=tmp_db,
        )

    history = get_build_history(form_id, db_path=tmp_db)
    assert len(history) == 3
    # Newest first
    assert history[0]["sorry_count"] == 0
    assert history[1]["sorry_count"] == 2
    assert history[2]["sorry_count"] == 5


def test_auto_promote_clean_build(tmp_db):
    """Auto-promote: zero sorry + clean build -> proof_complete + evidence_level=3."""
    # Create conjecture in workbench
    conj_id = create_conjecture(
        "Test conjecture", evidence_level=1, status="computational_evidence",
        db_path=tmp_db,
    )
    form_id = create_formalization(conj_id, "/f.lean", db_path=tmp_db)

    # Walk to proof_attempted state
    update_formalization_state(form_id, "statement_formalized", db_path=tmp_db)
    update_formalization_state(form_id, "proof_attempted", db_path=tmp_db)

    # Record a clean build
    record_build(
        form_id,
        LakeBuildResult(success=True, returncode=0, output="ok", sorry_count=0, error_count=0, warning_count=0),
        db_path=tmp_db,
    )

    # Auto-promote
    result = auto_promote_if_clean(form_id, db_path=tmp_db)
    assert result is True

    # Verify formalization state
    form = get_formalization(form_id, db_path=tmp_db)
    assert form["formalization_state"] == "proof_complete"

    # Verify conjecture updated
    conj = get_conjecture(conj_id, db_path=tmp_db)
    assert conj["evidence_level"] == 3
    assert conj["status"] == "proved"


def test_auto_promote_with_sorry_does_nothing(tmp_db):
    """Auto-promote returns False when sorry_count > 0."""
    conj_id = create_conjecture("Test", db_path=tmp_db)
    form_id = create_formalization(conj_id, "/f.lean", db_path=tmp_db)
    update_formalization_state(form_id, "statement_formalized", db_path=tmp_db)
    update_formalization_state(form_id, "proof_attempted", sorry_count=3, db_path=tmp_db)

    # Record a build with sorry
    record_build(
        form_id,
        LakeBuildResult(success=True, returncode=0, output="ok", sorry_count=3, error_count=0, warning_count=1),
        db_path=tmp_db,
    )

    result = auto_promote_if_clean(form_id, db_path=tmp_db)
    assert result is False

    # State unchanged
    form = get_formalization(form_id, db_path=tmp_db)
    assert form["formalization_state"] == "proof_attempted"


def test_list_formalizations_filter_by_state(tmp_db):
    """List formalizations with state filter."""
    form_id1 = create_formalization("conj-1", "/a.lean", db_path=tmp_db)
    form_id2 = create_formalization("conj-2", "/b.lean", db_path=tmp_db)

    update_formalization_state(form_id1, "statement_formalized", db_path=tmp_db)

    all_forms = list_formalizations(db_path=tmp_db)
    assert len(all_forms) == 2

    formalized = list_formalizations(state="statement_formalized", db_path=tmp_db)
    assert len(formalized) == 1
    assert formalized[0]["id"] == form_id1

    not_formalized = list_formalizations(state="not_formalized", db_path=tmp_db)
    assert len(not_formalized) == 1
    assert not_formalized[0]["id"] == form_id2


def test_formalization_not_found_raises(tmp_db):
    """get_formalization returns None; update raises ValueError for non-existent ID."""
    assert get_formalization("nonexistent", db_path=tmp_db) is None

    with pytest.raises(ValueError, match="not found"):
        update_formalization_state("nonexistent", "statement_formalized", db_path=tmp_db)

    with pytest.raises(ValueError, match="not found"):
        auto_promote_if_clean("nonexistent", db_path=tmp_db)
