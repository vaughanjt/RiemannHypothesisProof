"""LMFDB REST API client with SQLite caching.

Queries the L-functions and Modular Forms Database (https://www.lmfdb.org)
for precomputed L-function data, modular form data, and number field data.
Responses are cached locally in SQLite to avoid repeated API calls.

The LMFDB provides access to thousands of L-functions and modular forms,
enabling cross-referencing with the platform's own computations. This is
essential for the Langlands program connection to the Riemann Hypothesis.

Function-based API with SQLite context manager pattern matching workbench/db.py.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import requests

from riemann.config import DATA_DIR


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LMFDB_BASE = "https://www.lmfdb.org/api"
DEFAULT_CACHE_DB = DATA_DIR / "lmfdb_cache.db"
MAX_PAGES = 10
REQUEST_TIMEOUT = 30


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LMFDBError(Exception):
    """Raised when an LMFDB API request fails."""

    def __init__(self, message: str, status_code: int | None = None, url: str = ""):
        self.status_code = status_code
        self.url = url
        super().__init__(message)


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS lmfdb_cache (
    query_key TEXT PRIMARY KEY,
    response_json TEXT NOT NULL,
    collection TEXT NOT NULL,
    cached_at TEXT NOT NULL
);
"""


@contextmanager
def _get_cache_connection(cache_db: str | Path):
    """Get a connection to the cache database.

    Follows the project SQLite context manager pattern from workbench/db.py.
    """
    path = str(cache_db)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _init_cache(cache_db: str | Path) -> str:
    """Initialize the cache database with the schema.

    Idempotent -- safe to call multiple times.

    Args:
        cache_db: Path to the cache SQLite database.

    Returns:
        String path to the database.
    """
    path = str(cache_db)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(CACHE_SCHEMA)
        conn.commit()
    finally:
        conn.close()
    return path


def _cache_key(collection: str, params: dict, fields: list[str] | None) -> str:
    """Compute a deterministic cache key for a query.

    SHA-256 of collection + sorted params JSON + sorted fields.

    Args:
        collection: LMFDB collection name.
        params: Query parameters dict.
        fields: Optional list of fields to request.

    Returns:
        Hex digest string as cache key.
    """
    key_parts = {
        "collection": collection,
        "params": dict(sorted(params.items())),
        "fields": sorted(fields) if fields else None,
    }
    key_json = json.dumps(key_parts, sort_keys=True, default=str)
    return hashlib.sha256(key_json.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Core query function
# ---------------------------------------------------------------------------

def query_lmfdb(
    collection: str,
    params: dict,
    fields: list[str] | None = None,
    cache_db: str | Path | None = None,
    refresh: bool = False,
) -> list[dict]:
    """Query the LMFDB REST API with SQLite caching.

    Builds a URL like: {LMFDB_BASE}/{collection}/?{params}&_format=json
    Checks the cache first (unless refresh=True). On cache miss, makes an
    HTTP GET request. Handles pagination up to MAX_PAGES pages.

    Args:
        collection: LMFDB collection to query (e.g., 'mf_newforms',
            'lfunc_lfunctions', 'nf_fields').
        params: Query parameters as a dict.
        fields: Optional list of field names to return.
        cache_db: Path to cache database. Default: DATA_DIR/lmfdb_cache.db.
        refresh: If True, bypass cache and make a fresh HTTP request.

    Returns:
        List of result dicts from the LMFDB response 'data' field.

    Raises:
        LMFDBError: On HTTP error or malformed response.
    """
    db_path = cache_db or DEFAULT_CACHE_DB
    _init_cache(db_path)

    key = _cache_key(collection, params, fields)

    # Check cache first (unless refresh requested)
    if not refresh:
        cached = _get_cached(db_path, key)
        if cached is not None:
            return cached

    # Build URL
    query_params = dict(params)
    query_params["_format"] = "json"
    if fields:
        query_params["_fields"] = ",".join(fields)

    url = f"{LMFDB_BASE}/{collection}/?{urlencode(query_params)}"

    # Make HTTP request with pagination
    all_data: list[dict] = []
    current_url = url
    pages_fetched = 0

    while current_url and pages_fetched < MAX_PAGES:
        try:
            response = requests.get(current_url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, "status_code", None) if e.response is not None else None
            raise LMFDBError(
                f"LMFDB API error: {e}",
                status_code=status,
                url=current_url,
            ) from e
        except requests.exceptions.RequestException as e:
            raise LMFDBError(
                f"LMFDB request failed: {e}",
                url=current_url,
            ) from e

        try:
            body = response.json()
        except ValueError as e:
            raise LMFDBError(
                f"Invalid JSON response from LMFDB: {e}",
                url=current_url,
            ) from e

        data = body.get("data", [])
        all_data.extend(data)

        # Check for next page
        current_url = body.get("next")
        pages_fetched += 1

    # Cache the result
    _set_cached(db_path, key, collection, all_data)

    return all_data


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _get_cached(cache_db: str | Path, key: str) -> list[dict] | None:
    """Retrieve a cached response by key.

    Args:
        cache_db: Path to cache database.
        key: Cache key (SHA-256 hex digest).

    Returns:
        Cached list of dicts, or None if not found.
    """
    with _get_cache_connection(cache_db) as conn:
        row = conn.execute(
            "SELECT response_json FROM lmfdb_cache WHERE query_key = ?",
            (key,),
        ).fetchone()
        if row is not None:
            return json.loads(row["response_json"])
    return None


def _set_cached(
    cache_db: str | Path, key: str, collection: str, data: list[dict]
) -> None:
    """Store a response in the cache.

    Args:
        cache_db: Path to cache database.
        key: Cache key (SHA-256 hex digest).
        collection: LMFDB collection name.
        data: List of result dicts to cache.
    """
    now = datetime.now(timezone.utc).isoformat()
    data_json = json.dumps(data, default=str)

    with _get_cache_connection(cache_db) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO lmfdb_cache
               (query_key, response_json, collection, cached_at)
               VALUES (?, ?, ?, ?)""",
            (key, data_json, collection, now),
        )


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def get_lfunction(
    label: str,
    cache_db: str | Path | None = None,
) -> dict | None:
    """Query LMFDB for an L-function by label.

    Args:
        label: LMFDB label string (e.g., '1-1-1.1-r0-0-0').
        cache_db: Path to cache database.

    Returns:
        Dict with L-function data, or None if not found.
    """
    results = query_lmfdb(
        "lfunc_lfunctions",
        {"label": label},
        cache_db=cache_db,
    )
    return results[0] if results else None


def get_modular_form(
    label: str,
    cache_db: str | Path | None = None,
) -> dict | None:
    """Query LMFDB for a modular form by label.

    Args:
        label: LMFDB label string (e.g., '1.12.1.a.a').
        cache_db: Path to cache database.

    Returns:
        Dict with modular form data, or None if not found.
    """
    results = query_lmfdb(
        "mf_newforms",
        {"label": label},
        cache_db=cache_db,
    )
    return results[0] if results else None


def get_number_field(
    label: str,
    cache_db: str | Path | None = None,
) -> dict | None:
    """Query LMFDB for a number field by label.

    Args:
        label: LMFDB label string (e.g., '2.2.5.1').
        cache_db: Path to cache database.

    Returns:
        Dict with number field data, or None if not found.
    """
    results = query_lmfdb(
        "nf_fields",
        {"label": label},
        cache_db=cache_db,
    )
    return results[0] if results else None


def clear_cache(cache_db: str | Path | None = None) -> int:
    """Delete all cached entries from the LMFDB cache.

    Args:
        cache_db: Path to cache database. Default: DATA_DIR/lmfdb_cache.db.

    Returns:
        Number of entries deleted.
    """
    db_path = cache_db or DEFAULT_CACHE_DB
    _init_cache(db_path)

    with _get_cache_connection(db_path) as conn:
        cursor = conn.execute("DELETE FROM lmfdb_cache")
        return cursor.rowcount
