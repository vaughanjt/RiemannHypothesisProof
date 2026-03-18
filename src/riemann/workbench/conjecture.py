"""Conjecture CRUD with strict evidence hierarchy enforcement.

NON-NEGOTIABLE per user decision: every finding tagged as
observation / heuristic / conditional / formal proof.
Never overwrite conjectures -- append new versions.
"""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from riemann.types import EvidenceLevel
from riemann.workbench.db import (
    VALID_STATUSES,
    get_connection,
    init_db,
)


def create_conjecture(
    statement: str,
    description: str = "",
    evidence_level: int = 0,
    status: str = "speculative",
    confidence: float | None = None,
    tags: list[str] | None = None,
    db_path: str | Path | None = None,
) -> str:
    """Create a new conjecture record.

    Args:
        statement: Formal mathematical statement.
        description: Human-readable explanation.
        evidence_level: 0=observation, 1=heuristic, 2=conditional, 3=formal_proof.
        status: One of VALID_STATUSES.
        confidence: Optional confidence score 0.0-1.0.
        tags: Optional list of string tags.
        db_path: Database path (default: production DB).

    Returns:
        UUID string of the created conjecture.

    Raises:
        ValueError: If evidence_level not in 0-3 or status not in VALID_STATUSES.
    """
    # Strict evidence level enforcement
    if evidence_level not in (0, 1, 2, 3):
        raise ValueError(
            f"evidence_level must be 0-3 (EvidenceLevel enum), got {evidence_level}. "
            f"0=OBSERVATION, 1=HEURISTIC, 2=CONDITIONAL, 3=FORMAL_PROOF"
        )
    if status not in VALID_STATUSES:
        raise ValueError(
            f"status must be one of {VALID_STATUSES}, got '{status}'"
        )
    if confidence is not None and not (0.0 <= confidence <= 1.0):
        raise ValueError(f"confidence must be 0.0-1.0, got {confidence}")

    conjecture_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    tags_json = json.dumps(tags) if tags else None

    init_db(db_path)
    with get_connection(db_path) as conn:
        conn.execute("""
            INSERT INTO conjectures
            (id, version, statement, description, status, evidence_level,
             confidence, created_at, updated_at, tags)
            VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            conjecture_id, statement, description, status,
            evidence_level, confidence, now, now, tags_json,
        ))

    return conjecture_id


def get_conjecture(
    conjecture_id: str,
    db_path: str | Path | None = None,
) -> dict | None:
    """Retrieve a conjecture by ID.

    Returns dict with all fields, or None if not found.
    """
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM conjectures WHERE id = ?",
            (conjecture_id,)
        ).fetchone()

    if row is None:
        return None

    result = dict(row)
    if result.get("tags"):
        result["tags"] = json.loads(result["tags"])
    return result


def update_conjecture(
    conjecture_id: str,
    statement: str | None = None,
    description: str | None = None,
    evidence_level: int | None = None,
    status: str | None = None,
    confidence: float | None = None,
    tags: list[str] | None = None,
    db_path: str | Path | None = None,
) -> str:
    """Update a conjecture, preserving the old version in history.

    NEVER overwrites -- archives old version to conjecture_history first.

    Returns:
        The conjecture_id (same as input).

    Raises:
        ValueError: If conjecture not found or invalid field values.
    """
    if evidence_level is not None and evidence_level not in (0, 1, 2, 3):
        raise ValueError(f"evidence_level must be 0-3, got {evidence_level}")
    if status is not None and status not in VALID_STATUSES:
        raise ValueError(f"status must be one of {VALID_STATUSES}, got '{status}'")

    with get_connection(db_path) as conn:
        # Get current version
        current = conn.execute(
            "SELECT * FROM conjectures WHERE id = ?",
            (conjecture_id,)
        ).fetchone()

        if current is None:
            raise ValueError(f"Conjecture {conjecture_id} not found")

        current = dict(current)

        # Archive current version to history
        history_id = f"{conjecture_id}_v{current['version']}"
        conn.execute("""
            INSERT INTO conjecture_history
            (id, original_id, version, statement, description, status,
             evidence_level, confidence, created_at, updated_at,
             parent_version_id, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            history_id, conjecture_id, current['version'],
            current['statement'], current['description'],
            current['status'], current['evidence_level'],
            current['confidence'], current['created_at'],
            current['updated_at'], current['parent_version_id'],
            current['tags'],
        ))

        # Update current with new values
        now = datetime.now(timezone.utc).isoformat()
        new_version = current['version'] + 1
        tags_json = json.dumps(tags) if tags is not None else current['tags']

        conn.execute("""
            UPDATE conjectures SET
                version = ?,
                statement = ?,
                description = ?,
                status = ?,
                evidence_level = ?,
                confidence = ?,
                updated_at = ?,
                parent_version_id = ?,
                tags = ?
            WHERE id = ?
        """, (
            new_version,
            statement if statement is not None else current['statement'],
            description if description is not None else current['description'],
            status if status is not None else current['status'],
            evidence_level if evidence_level is not None else current['evidence_level'],
            confidence if confidence is not None else current['confidence'],
            now,
            history_id,
            tags_json,
            conjecture_id,
        ))

    return conjecture_id


def list_conjectures(
    status: str | None = None,
    evidence_level: int | None = None,
    db_path: str | Path | None = None,
) -> list[dict]:
    """List conjectures with optional filtering.

    Args:
        status: Filter by status (e.g., 'speculative').
        evidence_level: Filter by evidence level (0-3).
        db_path: Database path.

    Returns:
        List of conjecture dicts.
    """
    query = "SELECT * FROM conjectures WHERE 1=1"
    params: list = []

    if status is not None:
        query += " AND status = ?"
        params.append(status)
    if evidence_level is not None:
        query += " AND evidence_level = ?"
        params.append(evidence_level)

    query += " ORDER BY updated_at DESC"

    with get_connection(db_path) as conn:
        rows = conn.execute(query, params).fetchall()

    results = []
    for row in rows:
        r = dict(row)
        if r.get("tags"):
            r["tags"] = json.loads(r["tags"])
        results.append(r)
    return results


def get_conjecture_history(
    conjecture_id: str,
    db_path: str | Path | None = None,
) -> list[dict]:
    """Get version history for a conjecture.

    Returns list of archived versions, oldest first.
    """
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM conjecture_history WHERE original_id = ? ORDER BY version ASC",
            (conjecture_id,)
        ).fetchall()
    return [dict(row) for row in rows]
