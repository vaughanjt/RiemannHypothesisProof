"""Tests for RSRCH-01: Conjecture tracking with evidence hierarchy.

Covers: SQLite schema creation, conjecture CRUD, evidence level enforcement,
version history on update, evidence link management.
"""
import pytest

from riemann.types import EvidenceLevel


class TestInitDb:
    """Schema creation tests."""

    def test_init_db_creates_all_tables(self, temp_db):
        """init_db(path) creates conjectures, experiments, evidence_links, observations tables."""
        import sqlite3
        from riemann.workbench.db import init_db

        init_db(temp_db)
        conn = sqlite3.connect(temp_db)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = sorted(row[0] for row in cursor.fetchall())
        conn.close()

        assert "conjectures" in tables
        assert "experiments" in tables
        assert "evidence_links" in tables
        assert "observations" in tables
        assert "conjecture_history" in tables


class TestCreateConjecture:
    """Conjecture creation and retrieval tests."""

    def test_create_conjecture_returns_uuid(self, temp_db):
        """create_conjecture returns a UUID id string."""
        from riemann.workbench.conjecture import create_conjecture

        cid = create_conjecture(
            statement="All non-trivial zeros lie on Re(s) = 1/2",
            description="The Riemann Hypothesis",
            evidence_level=EvidenceLevel.OBSERVATION.value,
            db_path=temp_db,
        )
        assert cid is not None
        assert isinstance(cid, str)
        assert len(cid) == 36  # UUID format

    def test_get_conjecture_returns_matching_record(self, temp_db):
        """get_conjecture(id) returns conjecture with matching id, statement, evidence_level."""
        from riemann.workbench.conjecture import create_conjecture, get_conjecture

        cid = create_conjecture(
            statement="All non-trivial zeros lie on Re(s) = 1/2",
            description="The Riemann Hypothesis",
            evidence_level=EvidenceLevel.OBSERVATION.value,
            db_path=temp_db,
        )

        conjecture = get_conjecture(cid, db_path=temp_db)
        assert conjecture is not None
        assert conjecture["id"] == cid
        assert conjecture["statement"] == "All non-trivial zeros lie on Re(s) = 1/2"
        assert conjecture["evidence_level"] == 0

    def test_create_conjecture_evidence_level_5_raises(self, temp_db):
        """create_conjecture with evidence_level=5 raises ValueError (only 0-3 valid)."""
        from riemann.workbench.conjecture import create_conjecture

        with pytest.raises(ValueError, match="evidence_level must be 0-3"):
            create_conjecture(
                statement="Invalid conjecture",
                evidence_level=5,
                db_path=temp_db,
            )

    def test_create_conjecture_evidence_level_negative_raises(self, temp_db):
        """create_conjecture with evidence_level=-1 raises ValueError."""
        from riemann.workbench.conjecture import create_conjecture

        with pytest.raises(ValueError, match="evidence_level must be 0-3"):
            create_conjecture(
                statement="Invalid conjecture",
                evidence_level=-1,
                db_path=temp_db,
            )

    def test_create_conjecture_invalid_status_raises(self, temp_db):
        """create_conjecture with invalid status raises ValueError."""
        from riemann.workbench.conjecture import create_conjecture

        with pytest.raises(ValueError, match="status must be one of"):
            create_conjecture(
                statement="Invalid status conjecture",
                evidence_level=0,
                status="invalid_status",
                db_path=temp_db,
            )


class TestUpdateConjecture:
    """Conjecture update and version history tests."""

    def test_update_creates_version_history(self, temp_db):
        """update_conjecture creates version history (old version preserved)."""
        from riemann.workbench.conjecture import (
            create_conjecture,
            get_conjecture,
            get_conjecture_history,
            update_conjecture,
        )

        cid = create_conjecture(
            statement="Initial statement",
            evidence_level=EvidenceLevel.OBSERVATION.value,
            db_path=temp_db,
        )

        update_conjecture(
            cid,
            statement="Revised statement",
            evidence_level=EvidenceLevel.HEURISTIC.value,
            db_path=temp_db,
        )

        # Current version should be updated
        current = get_conjecture(cid, db_path=temp_db)
        assert current["statement"] == "Revised statement"
        assert current["evidence_level"] == 1
        assert current["version"] == 2
        assert current["parent_version_id"] is not None

        # History should contain old version
        history = get_conjecture_history(cid, db_path=temp_db)
        assert len(history) >= 1
        assert history[0]["statement"] == "Initial statement"
        assert history[0]["evidence_level"] == 0


class TestListConjectures:
    """Conjecture listing and filtering tests."""

    def test_list_conjectures_returns_all(self, temp_db):
        """list_conjectures returns all active conjectures."""
        from riemann.workbench.conjecture import create_conjecture, list_conjectures

        create_conjecture(
            statement="Conjecture 1",
            evidence_level=0,
            db_path=temp_db,
        )
        create_conjecture(
            statement="Conjecture 2",
            evidence_level=1,
            db_path=temp_db,
        )

        results = list_conjectures(db_path=temp_db)
        assert len(results) == 2

    def test_list_conjectures_filter_by_status(self, temp_db):
        """list_conjectures with status filter returns only matching conjectures."""
        from riemann.workbench.conjecture import create_conjecture, list_conjectures

        create_conjecture(
            statement="Speculative conjecture",
            evidence_level=0,
            status="speculative",
            db_path=temp_db,
        )
        create_conjecture(
            statement="Proved conjecture",
            evidence_level=3,
            status="proved",
            db_path=temp_db,
        )

        speculative = list_conjectures(status="speculative", db_path=temp_db)
        assert len(speculative) == 1
        assert speculative[0]["statement"] == "Speculative conjecture"

        proved = list_conjectures(status="proved", db_path=temp_db)
        assert len(proved) == 1
        assert proved[0]["statement"] == "Proved conjecture"


class TestEvidenceLinks:
    """Evidence linking tests."""

    def test_link_evidence_connects_experiment_to_conjecture(self, temp_db):
        """link_evidence connects an experiment to a conjecture with relationship type."""
        from riemann.workbench.conjecture import create_conjecture
        from riemann.workbench.db import init_db, get_connection
        from riemann.workbench.evidence import link_evidence, get_evidence_for_conjecture

        import json
        import uuid
        from datetime import datetime, timezone

        init_db(temp_db)

        cid = create_conjecture(
            statement="Test conjecture",
            evidence_level=0,
            db_path=temp_db,
        )

        # Create an experiment directly in the DB for this test
        eid = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with get_connection(temp_db) as conn:
            conn.execute(
                """INSERT INTO experiments
                   (id, description, parameters, created_at)
                   VALUES (?, ?, ?, ?)""",
                (eid, "Test experiment", json.dumps({"n": 100}), now),
            )
            conn.commit()

        link_id = link_evidence(
            conjecture_id=cid,
            experiment_id=eid,
            relationship="supports",
            strength=0.8,
            db_path=temp_db,
        )
        assert link_id is not None

        links = get_evidence_for_conjecture(cid, db_path=temp_db)
        assert len(links) == 1
        assert links[0]["relationship"] == "supports"
        assert links[0]["strength"] == 0.8

    def test_link_evidence_invalid_relationship_raises(self, temp_db):
        """link_evidence with invalid relationship raises ValueError."""
        from riemann.workbench.evidence import link_evidence

        with pytest.raises(ValueError, match="relationship must be one of"):
            link_evidence(
                conjecture_id="fake-id",
                experiment_id="fake-id",
                relationship="invalid_relationship",
                db_path=temp_db,
            )

    def test_get_evidence_for_conjecture_returns_all_links(self, temp_db):
        """get_evidence_for_conjecture returns all linked evidence."""
        from riemann.workbench.conjecture import create_conjecture
        from riemann.workbench.db import init_db, get_connection
        from riemann.workbench.evidence import link_evidence, get_evidence_for_conjecture

        import json
        import uuid
        from datetime import datetime, timezone

        init_db(temp_db)

        cid = create_conjecture(
            statement="Test conjecture",
            evidence_level=0,
            db_path=temp_db,
        )

        now = datetime.now(timezone.utc).isoformat()
        with get_connection(temp_db) as conn:
            for i in range(3):
                eid = str(uuid.uuid4())
                conn.execute(
                    """INSERT INTO experiments
                       (id, description, parameters, created_at)
                       VALUES (?, ?, ?, ?)""",
                    (eid, f"Experiment {i}", json.dumps({"n": i}), now),
                )
                conn.commit()

                link_evidence(
                    conjecture_id=cid,
                    experiment_id=eid,
                    relationship="supports",
                    db_path=temp_db,
                )

        links = get_evidence_for_conjecture(cid, db_path=temp_db)
        assert len(links) == 3
