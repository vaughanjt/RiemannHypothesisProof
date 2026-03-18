"""Non-trivial zero computation, validation, and cataloging.

Uses mpmath.zetazero(n) for zero-finding. Never implement custom
Newton/bisection -- mpmath handles Gram point failures and arbitrary precision.
"""
import sqlite3
from pathlib import Path

import mpmath

from riemann.config import DEFAULT_DPS, DB_PATH, ODLYZKO_DIR
from riemann.engine.precision import precision_scope, validated_computation
from riemann.types import ComputationResult, PrecisionError, ZetaZero


def compute_zero(
    n: int,
    *,
    dps: int | None = None,
    validate: bool = True,
) -> ZetaZero:
    """Compute the nth non-trivial zero of the Riemann zeta function.

    Uses mpmath.zetazero(n) with always-validate pattern.
    Returns ZetaZero with validation metadata.

    Args:
        n: Index of the zero (1-based: n=1 is the first zero at ~14.13i).
        dps: Decimal digits of precision. Default: 50.
        validate: If True, run P-vs-2P validation.
    """
    if dps is None:
        dps = DEFAULT_DPS

    result = validated_computation(
        lambda: mpmath.zetazero(n),
        dps=dps,
        validate=validate,
        algorithm=f"mpmath.zetazero({n})",
    )

    zero_value = result.value
    # Check if real part is within threshold of 0.5
    threshold = mpmath.power(10, -(dps - 5))
    half = mpmath.mpf('0.5')
    on_critical_line = abs(zero_value.real - half) < threshold

    return ZetaZero(
        index=n,
        value=zero_value,
        precision_digits=dps,
        validated=result.validated,
        on_critical_line=on_critical_line,
    )


def compute_zeros_range(
    start: int,
    end: int,
    *,
    dps: int | None = None,
    validate: bool = True,
) -> list[ZetaZero]:
    """Compute zeros from index start to end (inclusive).

    Args:
        start: First zero index (1-based).
        end: Last zero index (inclusive).
        dps: Decimal digits of precision.
        validate: Always-validate flag.
    """
    return [
        compute_zero(n, dps=dps, validate=validate)
        for n in range(start, end + 1)
    ]


def load_odlyzko_zeros(
    file_path: Path | None = None,
) -> list[mpmath.mpf]:
    """Load Odlyzko zero table (imaginary parts of zeros on critical line).

    Returns list of mpf values (imaginary parts). Real part is always 0.5.
    """
    if file_path is None:
        file_path = ODLYZKO_DIR / "zeros_100.txt"

    zeros = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                zeros.append(mpmath.mpf(line))
    return zeros


def validate_against_odlyzko(
    computed_zeros: list[ZetaZero],
    odlyzko_zeros: list[mpmath.mpf] | None = None,
    tolerance_digits: int | None = None,
) -> list[dict]:
    """Validate computed zeros against Odlyzko table.

    Args:
        computed_zeros: List of ZetaZero objects to validate.
        odlyzko_zeros: List of mpf imaginary parts from Odlyzko table.
            If None, loads from default file.
        tolerance_digits: Number of digits that must agree. Default: min(computed_dps, 45).

    Returns:
        List of dicts with keys: index, matches, diff, tolerance
    """
    if odlyzko_zeros is None:
        odlyzko_zeros = load_odlyzko_zeros()

    results = []
    for zero in computed_zeros:
        idx = zero.index - 1  # Odlyzko list is 0-indexed
        if idx >= len(odlyzko_zeros):
            results.append({
                "index": zero.index,
                "matches": None,
                "diff": None,
                "tolerance": None,
                "reason": "No Odlyzko reference for this index",
            })
            continue

        tol_digits = tolerance_digits or min(zero.precision_digits - 5, 45)
        odlyzko_t = odlyzko_zeros[idx]
        computed_t = zero.value.imag

        with mpmath.workdps(max(zero.precision_digits, 50) + 10):
            diff = abs(computed_t - odlyzko_t)
            threshold = mpmath.power(10, -tol_digits)
            matches = diff < threshold

        results.append({
            "index": zero.index,
            "matches": bool(matches),
            "diff": float(diff),
            "tolerance": tol_digits,
        })

    return results


def zero_count(T: float | mpmath.mpf) -> int:
    """Count non-trivial zeros with imaginary part between 0 and T.

    Uses mpmath.nzeros(T) which implements the Riemann-von Mangoldt formula.

    Args:
        T: Upper bound for imaginary part.

    Returns:
        Number of zeros with 0 < Im(rho) < T.
    """
    return int(mpmath.nzeros(T))


class ZeroCatalog:
    """SQLite catalog for storing and retrieving computed zeros.

    Never delete zeros -- only add or update with higher precision.
    """

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = str(db_path or DB_PATH)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS zeros (
                    index_n INTEGER PRIMARY KEY,
                    real_part TEXT NOT NULL,
                    imag_part TEXT NOT NULL,
                    precision_digits INTEGER NOT NULL,
                    validated BOOLEAN NOT NULL DEFAULT 0,
                    on_critical_line BOOLEAN,
                    verified_against_odlyzko BOOLEAN DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)
            conn.commit()

    def store(self, zero: ZetaZero) -> None:
        """Store a zero, replacing if new precision is higher."""
        with sqlite3.connect(self.db_path) as conn:
            # Check if existing zero has higher precision
            existing = conn.execute(
                "SELECT precision_digits FROM zeros WHERE index_n = ?",
                (zero.index,)
            ).fetchone()

            if existing and existing[0] >= zero.precision_digits:
                return  # Keep higher-precision version

            conn.execute("""
                INSERT OR REPLACE INTO zeros
                (index_n, real_part, imag_part, precision_digits,
                 validated, on_critical_line, verified_against_odlyzko, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """, (
                zero.index,
                mpmath.nstr(zero.value.real, zero.precision_digits + 5),
                mpmath.nstr(zero.value.imag, zero.precision_digits + 5),
                zero.precision_digits,
                zero.validated,
                zero.on_critical_line,
                zero.verified_against_odlyzko,
            ))
            conn.commit()

    def get(self, index: int) -> ZetaZero | None:
        """Retrieve a zero by index."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM zeros WHERE index_n = ?", (index,)
            ).fetchone()

        if row is None:
            return None

        return ZetaZero(
            index=row[0],
            value=mpmath.mpc(row[1], row[2]),
            precision_digits=row[3],
            validated=bool(row[4]),
            on_critical_line=bool(row[5]) if row[5] is not None else None,
            verified_against_odlyzko=bool(row[6]),
        )

    def get_range(self, start: int, end: int) -> list[ZetaZero]:
        """Retrieve zeros in index range [start, end]."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM zeros WHERE index_n BETWEEN ? AND ? ORDER BY index_n",
                (start, end)
            ).fetchall()

        return [
            ZetaZero(
                index=row[0],
                value=mpmath.mpc(row[1], row[2]),
                precision_digits=row[3],
                validated=bool(row[4]),
                on_critical_line=bool(row[5]) if row[5] is not None else None,
                verified_against_odlyzko=bool(row[6]),
            )
            for row in rows
        ]

    def count(self) -> int:
        """Return total number of stored zeros."""
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM zeros").fetchone()[0]
