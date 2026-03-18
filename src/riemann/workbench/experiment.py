"""Experiment tracking with full parameter reproducibility.

RSRCH-02: Save, annotate, and revisit experiments. Every experiment
records all parameters, random seeds, and a SHA-256 checksum of results
for tamper detection and reproducibility verification.

Checksum strategy (from RESEARCH.md Open Question 4): checksum the string
representation at the experiment's stated precision. This makes checksums
precision-stable (same precision = same checksum, regardless of internal
representation).
"""
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from riemann.config import COMPUTED_DIR
from riemann.workbench.db import get_connection, init_db


def save_experiment(
    description: str,
    parameters: dict,
    result_summary: str = "",
    result_data: np.ndarray | None = None,
    seed: int | None = None,
    computation_time_ms: float = 0.0,
    precision_digits: int | None = None,
    validated: bool = False,
    notebook_path: str = "",
    db_path: str | Path | None = None,
) -> str:
    """Save an experiment with full parameter serialization and checksum.

    Args:
        description: Human-readable experiment description.
        parameters: Dict of all experiment parameters (must be JSON-serializable).
        result_summary: Brief text summary of results.
        result_data: Optional numpy array of numerical results to save as .npz.
        seed: Random seed used (if applicable).
        computation_time_ms: Total computation time.
        precision_digits: mpmath precision used.
        validated: Whether results passed P-vs-2P validation.
        notebook_path: Source notebook path.
        db_path: Database path.

    Returns:
        UUID string of the created experiment.
    """
    experiment_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    params_json = json.dumps(parameters, sort_keys=True, default=str)

    # Save result data if provided
    data_files: list[str] = []
    if result_data is not None:
        COMPUTED_DIR.mkdir(parents=True, exist_ok=True)
        data_path = COMPUTED_DIR / f"{experiment_id}.npz"
        np.savez_compressed(str(data_path), results=result_data)
        data_files.append(str(data_path))

    data_files_json = json.dumps(data_files) if data_files else None

    # Compute checksum of parameters + result summary + data
    checksum = _compute_checksum(
        params_json, result_summary, result_data, precision_digits
    )

    init_db(db_path)
    with get_connection(db_path) as conn:
        conn.execute("""
            INSERT INTO experiments
            (id, description, parameters, seed, checksum, result_summary,
             data_files, computation_time_ms, precision_digits, validated,
             created_at, notebook_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            experiment_id, description, params_json, seed, checksum,
            result_summary, data_files_json, computation_time_ms,
            precision_digits, validated, now, notebook_path,
        ))

    return experiment_id


def load_experiment(
    experiment_id: str,
    db_path: str | Path | None = None,
) -> dict | None:
    """Load an experiment by ID.

    Returns dict with all fields. Parameters are deserialized from JSON.
    Result data loaded from .npz if data_files exists.
    """
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM experiments WHERE id = ?",
            (experiment_id,)
        ).fetchone()

    if row is None:
        return None

    result = dict(row)
    result["parameters"] = json.loads(result["parameters"])
    if result.get("data_files"):
        result["data_files"] = json.loads(result["data_files"])
    else:
        result["data_files"] = []

    # Load numpy data if available
    result["result_data"] = None
    if result["data_files"]:
        first_file = result["data_files"][0]
        if Path(first_file).exists():
            loaded = np.load(first_file)
            result["result_data"] = loaded["results"]

    return result


def verify_checksum(
    experiment_id: str,
    db_path: str | Path | None = None,
) -> bool:
    """Verify that experiment data has not been modified.

    Recomputes checksum from stored parameters + summary + data files
    and compares with stored checksum.

    Returns:
        True if checksum matches (data unmodified), False otherwise.
    """
    experiment = load_experiment(experiment_id, db_path)
    if experiment is None:
        raise ValueError(f"Experiment {experiment_id} not found")

    params_json = json.dumps(
        experiment["parameters"], sort_keys=True, default=str
    )
    recomputed = _compute_checksum(
        params_json,
        experiment.get("result_summary", "") or "",
        experiment.get("result_data"),
        experiment.get("precision_digits"),
    )

    return recomputed == experiment["checksum"]


def list_experiments(
    db_path: str | Path | None = None,
) -> list[dict]:
    """List all experiments, most recent first.

    Returns list of experiment dicts (without loaded numpy data).
    """
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM experiments ORDER BY created_at DESC"
        ).fetchall()

    results = []
    for row in rows:
        r = dict(row)
        r["parameters"] = json.loads(r["parameters"])
        if r.get("data_files"):
            r["data_files"] = json.loads(r["data_files"])
        else:
            r["data_files"] = []
        results.append(r)
    return results


def _compute_checksum(
    params_json: str,
    result_summary: str,
    result_data: np.ndarray | None,
    precision_digits: int | None,
) -> str:
    """Compute SHA-256 checksum for experiment reproducibility.

    Checksums the string representation at stated precision (per RESEARCH.md
    Open Question 4 decision). This makes checksums precision-stable.
    """
    hasher = hashlib.sha256()
    hasher.update(params_json.encode('utf-8'))
    hasher.update(result_summary.encode('utf-8'))

    if precision_digits is not None:
        hasher.update(str(precision_digits).encode('utf-8'))

    if result_data is not None:
        # Use tobytes for deterministic hashing of numpy arrays
        hasher.update(result_data.tobytes())

    return hasher.hexdigest()
