"""Tests for LMFDB REST API client with SQLite caching.

All HTTP calls are mocked -- no real network calls in tests.

Tests cover:
- query_lmfdb: basic queries, field filtering, caching, cache bypass
- get_lfunction: convenience wrapper for L-function queries
- get_modular_form: convenience wrapper for modular form queries
- get_number_field: convenience wrapper for number field queries
- clear_cache: cache entry deletion
- Error handling: HTTP errors raise LMFDBError
- Cache behavior: second call hits cache, not HTTP
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cache_db(tmp_path):
    """Provide a temporary SQLite database path for cache tests."""
    return str(tmp_path / "test_lmfdb_cache.db")


def _mock_response(data: list[dict], status_code: int = 200) -> MagicMock:
    """Create a mock requests.Response with LMFDB-format JSON."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.ok = status_code == 200
    mock.json.return_value = {"data": data}
    mock.raise_for_status = MagicMock()
    if status_code >= 400:
        from requests.exceptions import HTTPError
        mock.raise_for_status.side_effect = HTTPError(
            f"{status_code} Error", response=mock
        )
    return mock


# ---------------------------------------------------------------------------
# query_lmfdb
# ---------------------------------------------------------------------------

class TestQueryLMFDB:
    @patch("riemann.analysis.lmfdb_client.requests.get")
    def test_basic_query_returns_list(self, mock_get, cache_db):
        """query_lmfdb returns a list of dicts from LMFDB response."""
        from riemann.analysis.lmfdb_client import query_lmfdb

        mock_get.return_value = _mock_response([
            {"label": "1.12.1.a.a", "weight": 12, "level": 1},
        ])
        result = query_lmfdb("mf_newforms", {"weight": 12, "level": 1}, cache_db=cache_db)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["weight"] == 12

    @patch("riemann.analysis.lmfdb_client.requests.get")
    def test_query_with_fields_parameter(self, mock_get, cache_db):
        """When fields specified, they are passed to LMFDB API."""
        from riemann.analysis.lmfdb_client import query_lmfdb

        mock_get.return_value = _mock_response([
            {"label": "1.12.1.a.a", "weight": 12},
        ])
        result = query_lmfdb(
            "mf_newforms", {"weight": 12}, fields=["label", "weight"], cache_db=cache_db
        )
        assert len(result) == 1
        # Verify that _fields was included in the URL
        call_url = mock_get.call_args[0][0]
        assert "_fields=" in call_url

    @patch("riemann.analysis.lmfdb_client.requests.get")
    def test_caching_avoids_second_http_call(self, mock_get, cache_db):
        """Second call with same params hits cache, not HTTP."""
        from riemann.analysis.lmfdb_client import query_lmfdb

        mock_get.return_value = _mock_response([
            {"label": "test", "value": 42},
        ])

        # First call: hits HTTP
        result1 = query_lmfdb("test_collection", {"key": "val"}, cache_db=cache_db)
        assert mock_get.call_count == 1

        # Second call: same params, should hit cache
        result2 = query_lmfdb("test_collection", {"key": "val"}, cache_db=cache_db)
        assert mock_get.call_count == 1  # No additional HTTP call
        assert result1 == result2

    @patch("riemann.analysis.lmfdb_client.requests.get")
    def test_refresh_bypasses_cache(self, mock_get, cache_db):
        """refresh=True forces a new HTTP call even if cached."""
        from riemann.analysis.lmfdb_client import query_lmfdb

        mock_get.return_value = _mock_response([{"label": "v1"}])
        query_lmfdb("coll", {"k": "v"}, cache_db=cache_db)
        assert mock_get.call_count == 1

        mock_get.return_value = _mock_response([{"label": "v2"}])
        result = query_lmfdb("coll", {"k": "v"}, cache_db=cache_db, refresh=True)
        assert mock_get.call_count == 2
        assert result[0]["label"] == "v2"

    @patch("riemann.analysis.lmfdb_client.requests.get")
    def test_http_error_raises_lmfdb_error(self, mock_get, cache_db):
        """HTTP errors (404, 500) raise LMFDBError."""
        from riemann.analysis.lmfdb_client import query_lmfdb, LMFDBError

        mock_get.return_value = _mock_response([], status_code=404)
        with pytest.raises(LMFDBError):
            query_lmfdb("nonexistent", {}, cache_db=cache_db)


# ---------------------------------------------------------------------------
# get_lfunction
# ---------------------------------------------------------------------------

class TestGetLFunction:
    @patch("riemann.analysis.lmfdb_client.requests.get")
    def test_returns_dict_for_valid_label(self, mock_get, cache_db):
        """get_lfunction returns a dict with L-function data."""
        from riemann.analysis.lmfdb_client import get_lfunction

        mock_get.return_value = _mock_response([
            {"label": "1-1-1.1-r0-0-0", "degree": 1, "conductor": 1},
        ])
        result = get_lfunction("1-1-1.1-r0-0-0", cache_db=cache_db)
        assert isinstance(result, dict)
        assert result["label"] == "1-1-1.1-r0-0-0"

    @patch("riemann.analysis.lmfdb_client.requests.get")
    def test_returns_none_for_empty_results(self, mock_get, cache_db):
        """get_lfunction returns None if label not found."""
        from riemann.analysis.lmfdb_client import get_lfunction

        mock_get.return_value = _mock_response([])
        result = get_lfunction("nonexistent-label", cache_db=cache_db)
        assert result is None


# ---------------------------------------------------------------------------
# get_modular_form
# ---------------------------------------------------------------------------

class TestGetModularForm:
    @patch("riemann.analysis.lmfdb_client.requests.get")
    def test_returns_dict_for_valid_label(self, mock_get, cache_db):
        """get_modular_form returns a dict with modular form data."""
        from riemann.analysis.lmfdb_client import get_modular_form

        mock_get.return_value = _mock_response([
            {"label": "1.12.1.a.a", "weight": 12, "level": 1},
        ])
        result = get_modular_form("1.12.1.a.a", cache_db=cache_db)
        assert isinstance(result, dict)
        assert result["weight"] == 12


# ---------------------------------------------------------------------------
# get_number_field
# ---------------------------------------------------------------------------

class TestGetNumberField:
    @patch("riemann.analysis.lmfdb_client.requests.get")
    def test_returns_dict_for_valid_label(self, mock_get, cache_db):
        """get_number_field returns a dict with number field data."""
        from riemann.analysis.lmfdb_client import get_number_field

        mock_get.return_value = _mock_response([
            {"label": "2.2.5.1", "degree": 2, "discriminant": 5},
        ])
        result = get_number_field("2.2.5.1", cache_db=cache_db)
        assert isinstance(result, dict)
        assert result["degree"] == 2


# ---------------------------------------------------------------------------
# clear_cache
# ---------------------------------------------------------------------------

class TestClearCache:
    @patch("riemann.analysis.lmfdb_client.requests.get")
    def test_clear_removes_all_entries(self, mock_get, cache_db):
        """clear_cache removes all cached entries."""
        from riemann.analysis.lmfdb_client import query_lmfdb, clear_cache

        mock_get.return_value = _mock_response([{"a": 1}])
        query_lmfdb("coll", {"k": "v"}, cache_db=cache_db)

        deleted = clear_cache(cache_db=cache_db)
        assert deleted >= 1

    @patch("riemann.analysis.lmfdb_client.requests.get")
    def test_clear_forces_re_fetch(self, mock_get, cache_db):
        """After clearing cache, next query hits HTTP again."""
        from riemann.analysis.lmfdb_client import query_lmfdb, clear_cache

        mock_get.return_value = _mock_response([{"a": 1}])
        query_lmfdb("coll", {"k": "v"}, cache_db=cache_db)
        assert mock_get.call_count == 1

        clear_cache(cache_db=cache_db)

        query_lmfdb("coll", {"k": "v"}, cache_db=cache_db)
        assert mock_get.call_count == 2

    def test_clear_empty_cache_returns_zero(self, cache_db):
        """Clearing a fresh/empty cache returns 0."""
        from riemann.analysis.lmfdb_client import clear_cache

        deleted = clear_cache(cache_db=cache_db)
        assert deleted == 0
