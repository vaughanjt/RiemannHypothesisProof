"""Tests for COMP-02: Zero computation, Odlyzko validation, and SQLite catalog.

Tests verify compute_zero, compute_zeros_range, validate_against_odlyzko,
and ZeroCatalog against known zero values and the Odlyzko table.
"""
import mpmath
import pytest

from riemann.types import ZetaZero


def test_compute_zero_first(default_precision):
    """compute_zero(1) returns ZetaZero with index=1, value near 14.134725i, validated=True."""
    from riemann.engine.zeros import compute_zero

    zero = compute_zero(1, dps=50)
    assert isinstance(zero, ZetaZero)
    assert zero.index == 1
    assert zero.validated is True
    assert zero.precision_digits == 50
    # First zero imaginary part near 14.134725
    assert abs(zero.value.imag - mpmath.mpf("14.134725141734693790")) < mpmath.power(10, -10)


def test_zero_on_critical_line(default_precision):
    """Each computed zero has Re(z) within threshold of 0.5."""
    from riemann.engine.zeros import compute_zero

    for n in range(1, 6):
        zero = compute_zero(n, dps=50)
        assert zero.on_critical_line is True, (
            f"Zero {n} not on critical line: Re = {zero.value.real}"
        )
        assert abs(zero.value.real - mpmath.mpf("0.5")) < mpmath.power(10, -45), (
            f"Zero {n} Re part too far from 0.5: {zero.value.real}"
        )


def test_odlyzko_validation(odlyzko_zeros, default_precision):
    """First 10 computed zeros match Odlyzko table to 45 digits."""
    from riemann.engine.zeros import compute_zero

    for n in range(1, 11):
        zero = compute_zero(n, dps=50)
        odlyzko_t = odlyzko_zeros[n - 1]

        # Compare imaginary part (Odlyzko table gives t values)
        computed_t = zero.value.imag
        with mpmath.workdps(60):
            diff = abs(computed_t - odlyzko_t)
        assert diff < mpmath.power(10, -45), (
            f"Zero {n}: computed={mpmath.nstr(computed_t, 20)}, "
            f"odlyzko={mpmath.nstr(odlyzko_t, 20)}, diff={diff}"
        )


def test_validate_against_odlyzko_function(odlyzko_zeros, default_precision):
    """validate_against_odlyzko returns matching results for first 5 zeros."""
    from riemann.engine.zeros import compute_zeros_range, validate_against_odlyzko

    zeros = compute_zeros_range(1, 5, dps=50)
    results = validate_against_odlyzko(zeros, odlyzko_zeros)
    assert len(results) == 5
    for r in results:
        assert r["matches"] is True, (
            f"Zero {r['index']} failed Odlyzko validation: diff={r['diff']}"
        )


def test_compute_zeros_range(default_precision):
    """compute_zeros_range(1, 5) returns list of 5 ZetaZero objects."""
    from riemann.engine.zeros import compute_zeros_range

    zeros = compute_zeros_range(1, 5, dps=50)
    assert len(zeros) == 5
    for i, zero in enumerate(zeros, start=1):
        assert isinstance(zero, ZetaZero)
        assert zero.index == i
        assert zero.validated is True


def test_zero_catalog_store_and_get(temp_db):
    """Store and retrieve zeros from SQLite."""
    from riemann.engine.zeros import ZeroCatalog

    catalog = ZeroCatalog(temp_db)
    zero = ZetaZero(
        index=1,
        value=mpmath.mpc(mpmath.mpf("0.5"), mpmath.mpf("14.134725141734693790")),
        precision_digits=50,
        validated=True,
        on_critical_line=True,
    )

    catalog.store(zero)
    retrieved = catalog.get(1)
    assert retrieved is not None
    assert retrieved.index == 1
    assert abs(retrieved.value.imag - zero.value.imag) < mpmath.power(10, -40)
    assert retrieved.validated is True
    assert retrieved.on_critical_line is True


def test_zero_catalog_get_range(temp_db):
    """ZeroCatalog.get_range returns stored zeros."""
    from riemann.engine.zeros import ZeroCatalog

    catalog = ZeroCatalog(temp_db)
    for n in range(1, 4):
        zero = ZetaZero(
            index=n,
            value=mpmath.mpc(mpmath.mpf("0.5"), mpmath.mpf(str(14.0 + n))),
            precision_digits=50,
            validated=True,
            on_critical_line=True,
        )
        catalog.store(zero)

    retrieved = catalog.get_range(1, 3)
    assert len(retrieved) == 3
    assert [z.index for z in retrieved] == [1, 2, 3]


def test_zero_catalog_count(temp_db):
    """ZeroCatalog.count returns correct count."""
    from riemann.engine.zeros import ZeroCatalog

    catalog = ZeroCatalog(temp_db)
    assert catalog.count() == 0

    zero = ZetaZero(
        index=1,
        value=mpmath.mpc(mpmath.mpf("0.5"), mpmath.mpf("14.134725")),
        precision_digits=50,
        validated=True,
        on_critical_line=True,
    )
    catalog.store(zero)
    assert catalog.count() == 1


def test_zero_catalog_higher_precision_replaces(temp_db):
    """Storing a zero with higher precision replaces the existing one."""
    from riemann.engine.zeros import ZeroCatalog

    catalog = ZeroCatalog(temp_db)
    zero_50 = ZetaZero(
        index=1,
        value=mpmath.mpc(mpmath.mpf("0.5"), mpmath.mpf("14.134725141734693790")),
        precision_digits=50,
        validated=True,
        on_critical_line=True,
    )
    zero_100 = ZetaZero(
        index=1,
        value=mpmath.mpc(mpmath.mpf("0.5"), mpmath.mpf("14.134725141734693790")),
        precision_digits=100,
        validated=True,
        on_critical_line=True,
    )

    catalog.store(zero_50)
    catalog.store(zero_100)
    retrieved = catalog.get(1)
    assert retrieved.precision_digits == 100

    # Lower precision should NOT replace
    catalog.store(zero_50)
    retrieved = catalog.get(1)
    assert retrieved.precision_digits == 100


def test_nzeros_count(default_precision):
    """N(T) from mpmath.nzeros agrees with count of zeros in range."""
    from riemann.engine.zeros import zero_count

    # mpmath.nzeros gives the number of zeros with 0 < Im < T
    for T in [50, 100]:
        count = zero_count(T)
        expected = int(mpmath.nzeros(T))
        assert count == expected, f"N({T}): got {count}, expected {expected}"
