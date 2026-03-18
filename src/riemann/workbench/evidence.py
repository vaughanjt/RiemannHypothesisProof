"""Evidence chain management: linking experiments to conjectures.

Supports four relationship types: supports, contradicts, neutral, extends.
"""
from datetime import datetime, timezone
from pathlib import Path

from riemann.workbench.db import VALID_RELATIONSHIPS, get_connection


def link_evidence(
    conjecture_id: str,
    experiment_id: str,
    relationship: str,
    strength: float | None = None,
    notes: str = "",
    db_path: str | Path | None = None,
) -> int:
    """Link an experiment to a conjecture as evidence.

    Args:
        conjecture_id: UUID of the conjecture.
        experiment_id: UUID of the experiment.
        relationship: One of 'supports', 'contradicts', 'neutral', 'extends'.
        strength: Optional strength score 0.0-1.0.
        notes: Optional notes about the evidence link.
        db_path: Database path.

    Returns:
        ID of the evidence link record.

    Raises:
        ValueError: If relationship not in VALID_RELATIONSHIPS.
    """
    if relationship not in VALID_RELATIONSHIPS:
        raise ValueError(
            f"relationship must be one of {VALID_RELATIONSHIPS}, got '{relationship}'"
        )
    if strength is not None and not (0.0 <= strength <= 1.0):
        raise ValueError(f"strength must be 0.0-1.0, got {strength}")

    now = datetime.now(timezone.utc).isoformat()

    with get_connection(db_path) as conn:
        cursor = conn.execute("""
            INSERT INTO evidence_links
            (conjecture_id, experiment_id, relationship, strength, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (conjecture_id, experiment_id, relationship, strength, notes, now))
        last_id = cursor.lastrowid
    return last_id


def get_evidence_for_conjecture(
    conjecture_id: str,
    db_path: str | Path | None = None,
) -> list[dict]:
    """Get all evidence links for a conjecture.

    Returns list of evidence link dicts with experiment details.
    """
    with get_connection(db_path) as conn:
        rows = conn.execute("""
            SELECT el.*, e.description as experiment_description,
                   e.parameters as experiment_parameters
            FROM evidence_links el
            LEFT JOIN experiments e ON el.experiment_id = e.id
            WHERE el.conjecture_id = ?
            ORDER BY el.created_at DESC
        """, (conjecture_id,)).fetchall()

    return [dict(row) for row in rows]
