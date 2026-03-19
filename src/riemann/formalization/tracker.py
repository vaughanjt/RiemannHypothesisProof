"""Formalization lifecycle tracker: state machine, sorry tracking, auto-promotion.

4-state pipeline per conjecture:
  not_formalized -> statement_formalized -> proof_attempted -> proof_complete

On proof_complete: auto-promote conjecture evidence_level to FORMAL_PROOF (3).
"""
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from riemann.formalization.builder import LakeBuildResult
from riemann.workbench.conjecture import get_conjecture, update_conjecture
from riemann.workbench.db import get_connection, init_db


class FormalizationState(str, Enum):
    NOT_FORMALIZED = "not_formalized"
    STATEMENT_FORMALIZED = "statement_formalized"
    PROOF_ATTEMPTED = "proof_attempted"
    PROOF_COMPLETE = "proof_complete"


# Valid state transitions (from -> set of allowed to-states)
_VALID_TRANSITIONS = {
    FormalizationState.NOT_FORMALIZED: {FormalizationState.STATEMENT_FORMALIZED},
    FormalizationState.STATEMENT_FORMALIZED: {FormalizationState.PROOF_ATTEMPTED},
    FormalizationState.PROOF_ATTEMPTED: {
        FormalizationState.PROOF_ATTEMPTED,  # iterating on proof
        FormalizationState.PROOF_COMPLETE,
    },
    FormalizationState.PROOF_COMPLETE: set(),  # terminal state
}


def create_formalization(
    conjecture_id: str,
    lean_file_path: str,
    mathlib_imports: list[str] | None = None,
    db_path: str | Path | None = None,
) -> str:
    """Create a new formalization record for a conjecture.

    Returns the formalization ID (UUID).
    """
    formalization_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    imports_json = json.dumps(mathlib_imports) if mathlib_imports else None

    init_db(db_path)
    with get_connection(db_path) as conn:
        conn.execute("""
            INSERT INTO formalizations
            (id, conjecture_id, lean_file_path, formalization_state,
             sorry_count, error_count, last_build_success,
             mathlib_imports, created_at, updated_at)
            VALUES (?, ?, ?, 'not_formalized', 0, 0, 0, ?, ?, ?)
        """, (formalization_id, conjecture_id, lean_file_path,
              imports_json, now, now))

    return formalization_id


def get_formalization(
    formalization_id: str,
    db_path: str | Path | None = None,
) -> dict | None:
    """Get a formalization record by ID."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM formalizations WHERE id = ?",
            (formalization_id,),
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    if result.get("mathlib_imports"):
        result["mathlib_imports"] = json.loads(result["mathlib_imports"])
    return result


def list_formalizations(
    state: str | None = None,
    conjecture_id: str | None = None,
    db_path: str | Path | None = None,
) -> list[dict]:
    """List formalizations with optional filtering."""
    query = "SELECT * FROM formalizations WHERE 1=1"
    params: list = []
    if state is not None:
        query += " AND formalization_state = ?"
        params.append(state)
    if conjecture_id is not None:
        query += " AND conjecture_id = ?"
        params.append(conjecture_id)
    query += " ORDER BY updated_at DESC"

    with get_connection(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def update_formalization_state(
    formalization_id: str,
    new_state: str,
    sorry_count: int | None = None,
    error_count: int | None = None,
    last_build_success: bool | None = None,
    last_build_output: str | None = None,
    db_path: str | Path | None = None,
) -> str:
    """Transition formalization to a new state.

    Validates the transition is legal per the state machine.
    Raises ValueError for invalid transitions.
    """
    current = get_formalization(formalization_id, db_path)
    if current is None:
        raise ValueError(f"Formalization {formalization_id} not found")

    current_state = FormalizationState(current["formalization_state"])
    target_state = FormalizationState(new_state)

    if target_state not in _VALID_TRANSITIONS.get(current_state, set()):
        raise ValueError(
            f"Invalid transition: {current_state.value} -> {target_state.value}. "
            f"Allowed: {[s.value for s in _VALID_TRANSITIONS[current_state]]}"
        )

    now = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        conn.execute("""
            UPDATE formalizations SET
                formalization_state = ?,
                sorry_count = COALESCE(?, sorry_count),
                error_count = COALESCE(?, error_count),
                last_build_success = COALESCE(?, last_build_success),
                last_build_output = COALESCE(?, last_build_output),
                updated_at = ?
            WHERE id = ?
        """, (new_state, sorry_count, error_count,
              last_build_success, last_build_output, now,
              formalization_id))

    return formalization_id


def record_build(
    formalization_id: str,
    build_result: LakeBuildResult,
    db_path: str | Path | None = None,
) -> int:
    """Record a build attempt in build_history.

    Also updates the formalization's last_build_* fields.
    Returns the build_history row ID.
    """
    now = datetime.now(timezone.utc).isoformat()
    errors_json = json.dumps([
        {"file": m.file, "line": m.line, "col": m.col,
         "severity": m.severity, "message": m.message}
        for m in build_result.messages if m.severity == "error"
    ])

    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.execute("""
            INSERT INTO build_history
            (formalization_id, build_timestamp, success,
             sorry_count, error_count, warning_count,
             build_duration_ms, output, errors_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (formalization_id, now, build_result.success,
              build_result.sorry_count, build_result.error_count,
              build_result.warning_count, build_result.duration_ms,
              build_result.output, errors_json, now))
        build_id = cursor.lastrowid

        # Update formalization's last_build fields
        conn.execute("""
            UPDATE formalizations SET
                sorry_count = ?,
                error_count = ?,
                last_build_success = ?,
                last_build_output = ?,
                updated_at = ?
            WHERE id = ?
        """, (build_result.sorry_count, build_result.error_count,
              build_result.success, build_result.output, now,
              formalization_id))

    return build_id


def get_build_history(
    formalization_id: str,
    db_path: str | Path | None = None,
) -> list[dict]:
    """Get build history for a formalization, newest first."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM build_history WHERE formalization_id = ? ORDER BY build_timestamp DESC",
            (formalization_id,),
        ).fetchall()
    results = []
    for row in rows:
        r = dict(row)
        if r.get("errors_json"):
            r["errors_json"] = json.loads(r["errors_json"])
        results.append(r)
    return results


def auto_promote_if_clean(
    formalization_id: str,
    db_path: str | Path | None = None,
) -> bool:
    """Auto-promote to proof_complete + FORMAL_PROOF if build is clean.

    Checks: sorry_count == 0 AND last_build_success == True.
    If so: transitions state to proof_complete, updates conjecture
    evidence_level to 3 (FORMAL_PROOF) and status to "proved".

    Returns True if promotion happened, False otherwise.
    """
    form = get_formalization(formalization_id, db_path)
    if form is None:
        raise ValueError(f"Formalization {formalization_id} not found")

    if form["sorry_count"] != 0 or not form["last_build_success"]:
        return False

    # Must be in a state that can transition to proof_complete
    current_state = FormalizationState(form["formalization_state"])
    if FormalizationState.PROOF_COMPLETE not in _VALID_TRANSITIONS.get(current_state, set()):
        return False

    # Transition state
    update_formalization_state(
        formalization_id, "proof_complete", db_path=db_path,
    )

    # Auto-promote conjecture evidence level to FORMAL_PROOF
    update_conjecture(
        form["conjecture_id"],
        evidence_level=3,  # EvidenceLevel.FORMAL_PROOF
        status="proved",
        db_path=db_path,
    )

    return True
