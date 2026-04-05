from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from mpmath import mpc


class EvidenceLevel(Enum):
    """Strict evidence hierarchy -- non-negotiable per user decision."""
    OBSERVATION = 0       # "X appears to hold for tested cases"
    HEURISTIC = 1         # "Here is a plausible reason X might be true"
    CONDITIONAL = 2       # "X is true IF Y and Z (also unproven)"
    FORMAL_PROOF = 3      # "X is verified in Lean 4"


class PrecisionError(Exception):
    """Raised when P-vs-2P validation detects precision collapse."""
    pass


@dataclass(frozen=True)
class ZetaZero:
    """A non-trivial zero of the Riemann zeta function."""
    index: int
    value: mpc
    precision_digits: int
    validated: bool
    on_critical_line: bool | None = None
    verified_against_odlyzko: bool = False


@dataclass
class ComputationResult:
    """Result from any computation with full provenance metadata."""
    value: object                  # mpf, mpc, or array
    precision_digits: int
    validated: bool
    validation_precision: int | None = None
    algorithm: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    computation_time_ms: float = 0.0


@dataclass
class DualResult:
    """Result from dual mpmath + python-flint computation (per D-09)."""
    mpmath_value: object          # mpmath mpf or mpc
    flint_value: object           # flint arb or acb
    agreement_digits: float       # -log10(relative difference)
    label: str = ""
    flagged: bool = False         # True if agreement below threshold


@dataclass
class BarrierComparison:
    """Single comparison point between heat kernel trace and barrier."""
    L: float
    t: float
    heat_kernel_value: float
    barrier_value: float
    discrete_sum: float
    eisenstein_contrib: float
    constant_term: float
    digits_of_agreement: float
    n_maass_terms: int
    dual_validated: bool


@dataclass
class ConvergenceDiagnostic:
    """Convergence info for a truncated spectral sum."""
    n_terms_used: int
    n_terms_available: int
    last_term_magnitude: float
    tail_bound: float
    convergence_rate: float
