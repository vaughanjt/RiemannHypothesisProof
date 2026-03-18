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
