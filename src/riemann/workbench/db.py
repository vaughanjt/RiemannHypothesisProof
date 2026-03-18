"""SQLite database schema and connection management for the research workbench.

Uses sqlite3 stdlib. No ORM -- raw SQL is sufficient for a single-user
research database with ~10 tables.
"""
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from riemann.config import DB_PATH


VALID_STATUSES = (
    'speculative', 'computational_evidence', 'heuristic_support',
    'conditional', 'formalized', 'proved', 'disproved',
)

VALID_RELATIONSHIPS = ('supports', 'contradicts', 'neutral', 'extends')

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS conjectures (
    id TEXT PRIMARY KEY,
    version INTEGER NOT NULL DEFAULT 1,
    statement TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'speculative'
        CHECK(status IN ('speculative', 'computational_evidence',
                         'heuristic_support', 'conditional', 'formalized',
                         'proved', 'disproved')),
    evidence_level INTEGER NOT NULL DEFAULT 0
        CHECK(evidence_level BETWEEN 0 AND 3),
    confidence REAL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    parent_version_id TEXT,
    tags TEXT
);

CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    parameters TEXT NOT NULL,
    seed INTEGER,
    checksum TEXT,
    result_summary TEXT,
    data_files TEXT,
    computation_time_ms REAL,
    precision_digits INTEGER,
    validated BOOLEAN DEFAULT 0,
    created_at TEXT NOT NULL,
    notebook_path TEXT
);

CREATE TABLE IF NOT EXISTS evidence_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conjecture_id TEXT NOT NULL REFERENCES conjectures(id),
    experiment_id TEXT NOT NULL REFERENCES experiments(id),
    relationship TEXT NOT NULL
        CHECK(relationship IN ('supports', 'contradicts', 'neutral', 'extends')),
    strength REAL,
    notes TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT REFERENCES experiments(id),
    content TEXT NOT NULL,
    evidence_level INTEGER NOT NULL DEFAULT 0
        CHECK(evidence_level BETWEEN 0 AND 3),
    tags TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conjecture_history (
    id TEXT PRIMARY KEY,
    original_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    statement TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL,
    evidence_level INTEGER NOT NULL,
    confidence REAL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    parent_version_id TEXT,
    tags TEXT,
    archived_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def init_db(db_path: str | Path | None = None) -> str:
    """Initialize the research workbench database with all tables.

    Creates tables if they don't exist. Safe to call multiple times.

    Args:
        db_path: Path to SQLite database. Default: config.DB_PATH.

    Returns:
        String path to the database.
    """
    path = str(db_path or DB_PATH)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()
    return path


@contextmanager
def get_connection(db_path: str | Path | None = None):
    """Get a connection to the research database as a context manager.

    Commits on success, rolls back on exception, and always closes the
    connection on exit. This is critical on Windows where unclosed SQLite
    connections hold file locks.

    Usage::

        with get_connection(db_path) as conn:
            conn.execute(...)

    Args:
        db_path: Path to SQLite database. Default: config.DB_PATH.
    """
    path = str(db_path or DB_PATH)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
