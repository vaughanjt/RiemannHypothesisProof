"""TDD RED tests for Task 1: types, config, and backend verification."""
import mpmath


def test_import_riemann():
    """Test that riemann package is importable."""
    import riemann
    assert hasattr(riemann, "__version__")
    assert riemann.__version__ == "0.1.0"


def test_import_types():
    """Test that all types are importable."""
    from riemann.types import ZetaZero, ComputationResult, EvidenceLevel, PrecisionError
    assert ZetaZero is not None
    assert ComputationResult is not None
    assert EvidenceLevel is not None
    assert PrecisionError is not None


def test_zeta_zero_is_frozen_dataclass():
    """Test ZetaZero is a frozen dataclass with correct fields."""
    from riemann.types import ZetaZero
    import dataclasses

    assert dataclasses.is_dataclass(ZetaZero)

    # Check frozen
    zero = ZetaZero(
        index=1,
        value=mpmath.mpc(0.5, 14.134725),
        precision_digits=50,
        validated=True,
    )
    try:
        zero.index = 2
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass

    # Check fields
    fields = {f.name: f.type for f in dataclasses.fields(ZetaZero)}
    assert "index" in fields
    assert "value" in fields
    assert "precision_digits" in fields
    assert "validated" in fields
    assert "on_critical_line" in fields
    assert "verified_against_odlyzko" in fields

    # Check defaults
    assert zero.on_critical_line is None
    assert zero.verified_against_odlyzko is False


def test_evidence_level_enum():
    """Test EvidenceLevel enum has correct members."""
    from riemann.types import EvidenceLevel

    assert EvidenceLevel.OBSERVATION.value == 0
    assert EvidenceLevel.HEURISTIC.value == 1
    assert EvidenceLevel.CONDITIONAL.value == 2
    assert EvidenceLevel.FORMAL_PROOF.value == 3


def test_computation_result_fields():
    """Test ComputationResult has all required fields."""
    from riemann.types import ComputationResult
    from datetime import datetime

    result = ComputationResult(
        value=mpmath.mpf("3.14"),
        precision_digits=50,
        validated=True,
        validation_precision=100,
        algorithm="test",
    )

    assert result.precision_digits == 50
    assert result.validated is True
    assert result.validation_precision == 100
    assert result.algorithm == "test"
    assert isinstance(result.timestamp, datetime)
    assert result.computation_time_ms == 0.0


def test_precision_error_is_exception():
    """Test PrecisionError is a subclass of Exception."""
    from riemann.types import PrecisionError

    assert issubclass(PrecisionError, Exception)
    try:
        raise PrecisionError("test error")
    except PrecisionError as e:
        assert str(e) == "test error"


def test_default_dps():
    """Test DEFAULT_DPS is 50."""
    from riemann.config import DEFAULT_DPS
    assert DEFAULT_DPS == 50


def test_gmpy2_backend():
    """Test gmpy2 backend is active for mpmath."""
    assert mpmath.libmp.BACKEND == "gmpy", f"Expected 'gmpy', got '{mpmath.libmp.BACKEND}'"
