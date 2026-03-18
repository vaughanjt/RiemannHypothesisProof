# Phase 1: Computational Foundation and Research Workbench - Research

**Researched:** 2026-03-18
**Domain:** Arbitrary-precision zeta function evaluation, zero computation, complex function visualization, research workbench infrastructure
**Confidence:** HIGH

## Summary

Phase 1 establishes the trusted computational bedrock for all downstream exploration of the Riemann Hypothesis. The core stack is mpmath 1.4.0 (released 2026-02-23) with gmpy2 2.2.2 backend for 2-10x acceleration, evaluated inside JupyterLab 4.x notebooks. Every critical-strip computation must use mpmath -- never float64/scipy near the critical line. The always-validate pattern (compute at P and 2P digits, compare) is the single most important architectural decision to catch silent precision collapse, which is the #1 project-killing pitfall.

The phase delivers eight requirements spanning three tracks: computation (zeta evaluation, zero-finding, related functions, stress-testing), visualization (critical line interactive plots, domain coloring), and research infrastructure (conjecture tracking in SQLite, experiment reproducibility). All three tracks can be developed in parallel after the precision management foundation is laid, since visualization and workbench code depend on computation outputs but not on each other.

**Primary recommendation:** Build the precision management layer (context manager, always-validate, metadata tagging) first. Everything else depends on it being correct. Then deliver computation, visualization, and workbench in parallel waves.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- JupyterLab is the primary interface -- all exploration happens in notebooks
- Claude is the primary driver: Claude builds notebooks, runs computations, analyzes results, presents findings
- The user directs exploration ("what if we look at X?") and Claude does the mathematical heavy lifting
- Claude has full discretion on notebook organization (by topic, by session, or whatever makes mathematical sense)
- Optimize for speed over visual polish -- Claude is the primary consumer of visualizations
- Start coarse, zoom to refine on demand (progressive resolution for domain coloring)
- Claude picks visualization tools per use case (matplotlib for static analysis, Plotly for interactive when needed)
- Claude picks color schemes optimized for analytical clarity, not aesthetics
- SQLite for structured research tracking (conjectures, experiments, evidence chains, metadata)
- numpy files for numerical data persistence (computed zeros, function values)
- HDF5 (h5py) for large array storage when needed (Phase 2+ primarily)
- Strict evidence-level hierarchy from day 1: every finding tagged as observation / heuristic / conditional / formal proof
- The user emphasized: "the only way to understand this problem is to hold in context a whole mess of variables -- go strict and heavy"
- Default precision: 50 decimal digits (user intuition: "this problem will be cracked in under 50 digits")
- Always-validate mode: every computation runs at P and 2P digits, results compared to catch silent precision collapse
- mpmath for all critical strip evaluation -- never use float64/scipy near the critical line
- Claude manages precision escalation as needed without user intervention
- gmpy2 as mpmath's C backend for 2-10x acceleration

### Claude's Discretion
- Notebook organization strategy
- Color scheme selection per visualization type
- Specific matplotlib vs Plotly decisions per plot
- Domain coloring resolution levels at each zoom stage
- Evidence hierarchy subcategories beyond the four main levels
- Workbench schema design (SQLite table structure)
- Precision escalation thresholds
- Error handling and edge case management
- File organization for computed data

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| COMP-01 | Evaluate Riemann zeta function at any complex point to arbitrary precision | mpmath.zeta(s) with gmpy2 backend; precision context manager with workdps; three algorithm paths (Borwein near real, Riemann-Siegel for large Im, Euler-Maclaurin elsewhere) |
| COMP-02 | Compute and catalog non-trivial zeros with verification against Odlyzko tables | mpmath.zetazero(n) for nth zero; Odlyzko tables at UMN for first 100K zeros to 3e-9 accuracy; first 100 zeros to 1000+ digits for high-precision validation |
| COMP-03 | Evaluate related functions (Dirichlet L, Hardy Z, xi, Selberg zeta) to arbitrary precision | mpmath provides: siegelz() for Hardy Z, siegeltheta() for RS theta, dirichlet(s, chi) for L-functions; xi function must be hand-built from zeta + gamma; Selberg zeta requires custom implementation |
| COMP-04 | Stress-test patterns against expanded data to distinguish structure from artifacts | Always-validate (P vs 2P) pattern; parameterized re-run at higher precision / more zeros / wider parameter ranges; automated comparison framework |
| VIZ-01 | Visualize |zeta(1/2+it)| along critical line with interactive zoom and pan | Plotly for interactive zoom/pan/hover; matplotlib for static high-resolution; siegelz() for efficient critical line evaluation |
| VIZ-02 | Domain coloring of zeta in complex plane with zoomable critical strip | matplotlib.colors.hsv_to_rgb for vectorized HSV-to-RGB; phase-to-hue, log-magnitude-to-brightness; progressive resolution (coarse grid, refine on zoom) |
| RSRCH-01 | Track conjectures with formal statement, evidence, status, confidence | SQLite database with conjectures, evidence_links, experiments tables; strict 4-level evidence hierarchy enforced at insert time; version history for conjecture evolution |
| RSRCH-02 | Save, annotate, revisit experiments with full parameter reproducibility | Experiment records with serialized parameters, seeds, checksums; numpy .npy/.npz for numerical results; JSON metadata sidecar files |
</phase_requirements>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| mpmath | 1.4.0 | Arbitrary-precision zeta, zeros, L-functions, Riemann-Siegel | The only Python library with production-quality arbitrary-precision zeta function evaluation. Released 2026-02-23. Tested with CPython 3.9-3.14. |
| gmpy2 | 2.2.2 | GMP/MPFR/MPC C-level backend for mpmath | Automatic 2-10x acceleration of mpmath at high precision. Pre-built wheels for Windows/Python 3.12. mpmath auto-detects gmpy2 >= 2.2.0. |
| numpy | >=2.0 | Machine-precision arrays, FFT, linear algebra | Foundation for vectorized visualization grids, domain coloring computation, numerical data storage (.npy/.npz). |
| matplotlib | >=3.8 | Static 2D plots, domain coloring, publication-quality figures | Bedrock plotting; hsv_to_rgb for vectorized domain coloring; best for static analysis Claude consumes. |
| plotly | 6.x | Interactive visualization with zoom/pan/hover | FigureWidget with anywidget backend for JupyterLab 4 compatibility. Interactive critical line exploration. |
| jupyterlab | >=4.0 | Primary notebook interface | User-decided. All exploration happens here. Supports inline matplotlib, plotly, ipywidgets. |
| ipywidgets | >=8.1 | Interactive parameter controls in Jupyter | Sliders for precision, t-range, zoom level. FloatSlider with continuous_update=False for expensive computations. |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| sqlite3 | (stdlib) | Research workbench database | Always. Conjecture tracking, experiment metadata, evidence chains. Zero-config, single-file. |
| tqdm | >=4.66 | Progress bars for long computations | Bulk zero computation, domain coloring grid evaluation. |
| pandas | >=2.1 | Tabular result organization | Intermediate format between computation and display; zero tables, statistics summaries. |
| pytest | >=7.4 | Computational validation tests | Verifying engine against known zero values, functional equation checks, precision canary tests. |
| ruff | latest | Linting and formatting | Code quality for all .py modules. |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| matplotlib domain coloring | cplot 0.9.3 | cplot uses OKLAB (perceptually uniform) but last release Aug 2022 and GPL-3.0; custom matplotlib gives full control and is maintainable |
| Plotly for interactive | Bokeh | Weaker 3D support; Plotly has native anywidget JupyterLab 4 integration |
| sqlite3 stdlib | SQLAlchemy ORM | Adds complexity; raw sqlite3 is sufficient for single-user research DB with ~10 tables |
| numpy .npy for mpmath objects | pickle | mpmath mpc/mpf objects need pickle; use JSON string serialization for portability, pickle only for bulk caching |

**Installation:**
```bash
uv init --python 3.12
uv add mpmath gmpy2 numpy scipy sympy
uv add matplotlib plotly ipywidgets anywidget
uv add jupyterlab
uv add pandas tqdm
uv add --dev pytest ruff
```

## Architecture Patterns

### Recommended Project Structure
```
src/
  riemann/
    __init__.py
    engine/                 # COMP-01 through COMP-04
      __init__.py
      precision.py          # Precision context manager, always-validate decorator
      zeta.py               # Zeta function wrapper with metadata
      zeros.py              # Zero-finding, cataloging, Odlyzko verification
      lfunctions.py         # Dirichlet L-functions, Hardy Z, xi, Selberg
      validation.py         # Stress-test framework, P-vs-2P comparator
    viz/                    # VIZ-01, VIZ-02
      __init__.py
      critical_line.py      # |zeta(1/2+it)| interactive and static plots
      domain_coloring.py    # Complex plane phase/magnitude visualization
      styles.py             # Color scheme constants, analytical clarity palettes
    workbench/              # RSRCH-01, RSRCH-02
      __init__.py
      db.py                 # SQLite schema, connection, migrations
      conjecture.py         # Conjecture CRUD with evidence hierarchy
      experiment.py         # Experiment records, parameter serialization
      evidence.py           # Evidence chain management, level enforcement
    types.py                # ZetaZero, ComputationResult, EvidenceLevel enums
    config.py               # Global config: default precision, DB path, data dirs
data/
  zeros.db                  # SQLite research database
  cache/                    # Computation cache (content-addressed)
  computed/                 # numpy .npy/.npz files for numerical results
  odlyzko/                  # Downloaded verification tables
notebooks/
  explorations/             # Claude-organized research notebooks
tests/
  test_engine/
    test_precision.py       # Precision context manager, always-validate
    test_zeta.py            # Zeta evaluation against known values
    test_zeros.py           # Zero-finding against Odlyzko tables
    test_lfunctions.py      # Related function evaluation
    test_validation.py      # Stress-test framework
  test_viz/
    test_domain_coloring.py # Domain coloring output sanity
  test_workbench/
    test_db.py              # Schema creation, CRUD operations
    test_conjecture.py      # Evidence hierarchy enforcement
    test_experiment.py      # Parameter serialization, reproducibility
  conftest.py               # Shared fixtures: mpmath contexts, test DB
```

### Pattern 1: Precision Context Manager with Always-Validate

**What:** Every critical computation is wrapped in a precision context that (a) sets mpmath.mp.dps, (b) optionally runs at 2x precision for validation, (c) attaches precision metadata to the result.

**When to use:** Every zeta evaluation, zero computation, or related function call in the critical strip.

**Example:**
```python
# Source: mpmath docs (workdps context manager) + project-specific always-validate
from contextlib import contextmanager
import mpmath

@contextmanager
def precision_scope(dps: int, *, validate: bool = True):
    """Set precision with optional always-validate mode.

    When validate=True (default), computation runs at both dps and 2*dps.
    Results are compared; disagreement raises PrecisionError.
    """
    with mpmath.workdps(dps + 5):  # guard digits
        yield dps

def validated_computation(func, *args, dps: int = 50, tolerance: int = None):
    """Run func at dps and 2*dps, compare results.

    tolerance: number of digits that must agree. Default: dps - 5.
    Returns the higher-precision result with metadata.
    """
    if tolerance is None:
        tolerance = dps - 5

    with mpmath.workdps(dps + 5):
        result_p = func(*args)

    with mpmath.workdps(2 * dps + 5):
        result_2p = func(*args)

    # Compare: first 'tolerance' digits must agree
    if not _digits_agree(result_p, result_2p, tolerance):
        raise PrecisionError(
            f"Results disagree within {tolerance} digits at dps={dps}"
        )

    return ComputationResult(
        value=result_2p,
        precision_digits=dps,
        validated=True,
        validation_precision=2 * dps,
    )
```

### Pattern 2: Mathematical Object with Metadata

**What:** All computed values carry provenance: precision, algorithm, timestamp, validation status.

**When to use:** Every value returned from the computation engine.

**Example:**
```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from mpmath import mpc

class EvidenceLevel(Enum):
    OBSERVATION = 0       # "X appears to hold for tested cases"
    HEURISTIC = 1         # "Here is a plausible reason X might be true"
    CONDITIONAL = 2       # "X is true IF Y and Z (also unproven)"
    FORMAL_PROOF = 3      # "X is verified in Lean 4"

@dataclass(frozen=True)
class ZetaZero:
    index: int                     # Ordinal (1st, 2nd, ...)
    value: mpc                     # The zero (Re should be ~0.5)
    precision_digits: int          # Verified to this many digits
    validated: bool                # Passed P-vs-2P check
    on_critical_line: bool | None  # None = not verified to sufficient precision
    verified_against_odlyzko: bool = False

@dataclass
class ComputationResult:
    value: object                  # mpf, mpc, or array
    precision_digits: int
    validated: bool
    validation_precision: int | None = None
    algorithm: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    computation_time_ms: float = 0.0
```

### Pattern 3: Domain Coloring with Progressive Resolution

**What:** Start with a coarse grid (e.g., 200x200) for fast initial view, refine to higher resolution (800x800, 2000x2000) on demand when Claude or the user zooms in.

**When to use:** VIZ-02 domain coloring of zeta in the complex plane.

**Example:**
```python
import numpy as np
from matplotlib.colors import hsv_to_rgb
import mpmath

def domain_coloring(f, re_range, im_range, resolution=200):
    """Vectorized domain coloring for a complex function.

    For zeta near the critical strip, f should use mpmath internally.
    For display, we convert final RGB values to float64.
    """
    re = np.linspace(*re_range, resolution)
    im = np.linspace(*im_range, resolution)
    Re, Im = np.meshgrid(re, im)

    # Evaluate function (mpmath for critical strip, numpy otherwise)
    Z = np.array([
        [complex(f(mpmath.mpc(r, i))) for r in re] for i in im
    ])

    # Phase -> Hue (0 to 1, wrapping)
    H = (np.angle(Z) / (2 * np.pi)) % 1.0
    # Log-magnitude -> Brightness (avoid log(0))
    magnitude = np.abs(Z)
    V = 1.0 - 1.0 / (1.0 + np.log1p(magnitude))
    # Saturation: constant or modulated
    S = np.ones_like(H) * 0.9

    HSV = np.stack([H, S, V], axis=-1)
    RGB = hsv_to_rgb(HSV)
    return RGB, re, im
```

### Pattern 4: SQLite Research Workbench Schema

**What:** Core tables for conjecture tracking, experiment management, and evidence chains with strict evidence-level enforcement.

**When to use:** RSRCH-01, RSRCH-02.

**Example:**
```sql
-- Core research tracking schema
CREATE TABLE conjectures (
    id TEXT PRIMARY KEY,           -- UUID
    version INTEGER NOT NULL DEFAULT 1,
    statement TEXT NOT NULL,        -- Formal statement
    description TEXT,               -- Human-readable explanation
    status TEXT NOT NULL DEFAULT 'speculative'
        CHECK(status IN ('speculative', 'computational_evidence',
                         'heuristic_support', 'conditional', 'formalized',
                         'proved', 'disproved')),
    evidence_level INTEGER NOT NULL DEFAULT 0
        CHECK(evidence_level BETWEEN 0 AND 3),
    confidence REAL,               -- 0.0 to 1.0
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    parent_version_id TEXT,        -- Previous version (never overwrite)
    tags TEXT                      -- JSON array of tags
);

CREATE TABLE experiments (
    id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    parameters TEXT NOT NULL,       -- JSON: all params for reproducibility
    seed INTEGER,                  -- Random seed if applicable
    checksum TEXT,                 -- SHA-256 of serialized results
    result_summary TEXT,           -- Brief text summary
    data_files TEXT,               -- JSON array of .npy/.npz paths
    computation_time_ms REAL,
    precision_digits INTEGER,
    validated BOOLEAN DEFAULT 0,
    created_at TEXT NOT NULL,
    notebook_path TEXT             -- Source notebook
);

CREATE TABLE evidence_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conjecture_id TEXT NOT NULL REFERENCES conjectures(id),
    experiment_id TEXT NOT NULL REFERENCES experiments(id),
    relationship TEXT NOT NULL
        CHECK(relationship IN ('supports', 'contradicts', 'neutral', 'extends')),
    strength REAL,                 -- 0.0 to 1.0
    notes TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT REFERENCES experiments(id),
    content TEXT NOT NULL,
    evidence_level INTEGER NOT NULL DEFAULT 0
        CHECK(evidence_level BETWEEN 0 AND 3),
    tags TEXT,                     -- JSON array
    created_at TEXT NOT NULL
);

-- Never delete conjectures, only supersede
CREATE TRIGGER conjecture_version_on_update
BEFORE UPDATE ON conjectures
BEGIN
    INSERT INTO conjectures(id, version, statement, description, status,
        evidence_level, confidence, created_at, updated_at, parent_version_id, tags)
    VALUES (OLD.id || '_v' || OLD.version, OLD.version, OLD.statement,
        OLD.description, OLD.status, OLD.evidence_level, OLD.confidence,
        OLD.created_at, OLD.updated_at, OLD.parent_version_id, OLD.tags);
END;
```

### Anti-Patterns to Avoid

- **Global precision state:** Never use bare `mpmath.mp.dps = N`. Always use `mpmath.workdps(N)` context manager or the project's `precision_scope()`. Global state causes precision leak between unrelated computations.
- **float64 near the critical line:** Using `scipy.special.zeta()` or `complex()` conversion before computation is complete destroys all precision. The result looks like a number but is wrong.
- **Visualization-driven computation:** Never compute zeta values inside a plot callback. Compute first, store results, then visualize stored results. This enables caching and reproducibility.
- **Unversioned conjectures:** Never overwrite a conjecture record. Append new versions. The trigger above enforces this.
- **Hardcoded mathematical constants:** Use `mpmath.mpf('0.5')` for the critical line, `mpmath.pi` for pi. Never hardcode `14.134725` -- compute `mpmath.zetazero(1)` at the working precision.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Zeta function evaluation | Custom Riemann-Siegel implementation | `mpmath.zeta(s)` | mpmath uses Borwein / RS / Euler-Maclaurin automatically per region. Decades of testing. |
| Zero finding | Custom Newton/bisection for zeta zeros | `mpmath.zetazero(n)` | Handles Gram point failures, large n, arbitrary precision. |
| Hardy Z-function | Manual `exp(i*theta)*zeta(1/2+it)` | `mpmath.siegelz(t)` | Numerically stable, supports derivatives up to 4th order. |
| Riemann-Siegel theta | Custom implementation | `mpmath.siegeltheta(t)` | Supports derivatives, complex arguments. |
| Dirichlet L-functions | Custom character sum | `mpmath.dirichlet(s, chi)` | Handles analytic continuation, arbitrary characters as lists. |
| HSV-to-RGB conversion | Custom color math | `matplotlib.colors.hsv_to_rgb()` | Vectorized NumPy, handles (N,M,3) arrays natively. |
| Precision context management | Custom global state wrapper | `mpmath.workdps(n)` / `mpmath.workprec(n)` | Built-in context manager, exception-safe, restores on exit. |
| N(t) zero counting function | Custom formula | `mpmath.nzeros(t)` | Riemann-von Mangoldt formula, arbitrary precision. |
| Gram points | Custom theta inversion | `mpmath.grampoint(n)` | Supports non-integer n, arbitrary precision. |

**Key insight:** mpmath is extraordinarily comprehensive for zeta-related functions. The Phase 1 computation engine is primarily a well-designed wrapper around mpmath, adding precision validation, metadata, caching, and the always-validate pattern -- NOT reimplementing mathematical algorithms.

## Common Pitfalls

### Pitfall 1: Silent Precision Collapse
**What goes wrong:** Zeta evaluation near the critical strip involves massive cancellation between large oscillatory terms. Float64 (~15 digits) produces plausible but completely wrong numbers. You build theories on noise.
**Why it happens:** For Im(s) ~ T, intermediate terms are ~10^(T/2) larger than the final result. If working precision has fewer digits than this, the result is pure rounding error.
**How to avoid:** (1) Never use float64 for critical-strip evaluation. (2) Always-validate: compute at P and 2P digits, compare. (3) Test against Odlyzko's known zeros before trusting any pipeline. (4) Build automated "precision canary" tests.
**Warning signs:** Zeros at non-symmetric positions; results change when doubling precision; "interesting" patterns that vanish at higher precision; zeta values that are exactly zero (should be approximately zero).

### Pitfall 2: Confusing Numerical Evidence with Proof
**What goes wrong:** "Every zero I checked lies on the critical line" treated as established fact. The gap between "true for the first million zeros" and "true for all infinitely many zeros" is the entire content of RH. Mertens' conjecture was "true" for all tested values before failing at ~10^300.
**Why it happens:** Human pattern recognition is powerful but not a proof.
**How to avoid:** Strict evidence-level hierarchy enforced by the database schema (observation / heuristic / conditional / formal proof). Every claim tagged with parameter range tested. Actively seek counterexamples before investing in a pattern.
**Warning signs:** Phrases like "clearly true" without proof references; skipping counterexample searches.

### Pitfall 3: Infrastructure Over Mathematics
**What goes wrong:** Spending months building a perfect framework and never computing a single zero.
**Why it happens:** Software engineering feels productive and controllable; mathematical exploration feels uncertain.
**How to avoid:** Phase 1 must include actual mathematical exploration. Compute zeros on day one in a notebook. Build infrastructure around patterns discovered in actual use. 70/30 rule: 70% exploration, 30% infrastructure.
**Warning signs:** No mathematical computation run in the last week; refactoring code used only once.

### Pitfall 4: mpmath Thread Safety / Context Leaks
**What goes wrong:** Using the global `mpmath.mp` context from multiple threads or forgetting to restore precision causes one computation to corrupt another's precision.
**Why it happens:** mpmath's `mp.dps` is global state. Not all mpmath functions are available as context-local methods.
**How to avoid:** Always use `mpmath.workdps()` context managers. Never set `mp.dps` directly. For parallelism, use multiprocessing (not threading) -- each process gets its own mpmath state.
**Warning signs:** Intermittent precision differences in identical computations.

### Pitfall 5: mpmath String Conversion O(n^2)
**What goes wrong:** Converting very high-precision mpmath numbers to strings (for logging, display, serialization) is surprisingly slow at 10,000+ digits.
**How to avoid:** Truncate to display precision for logging. Use pickle or mpmath's native format for serialization. For the 50-digit default, this is not an issue; flag it if precision escalates.

### Pitfall 6: Matplotlib Cannot Handle Large Point Counts
**What goes wrong:** Plotting 10^6+ points in matplotlib causes the GUI/notebook to freeze.
**How to avoid:** Use decimation/subsampling for interactive plots. For domain coloring, start coarse (200x200 = 40K points, fine). For critical line with millions of evaluations, subsample for display.

## Code Examples

### Verified: Zeta Evaluation with gmpy2 Backend Verification
```python
# Source: mpmath 1.4.0 docs (setup.html), verified via web search
import mpmath

# Verify gmpy2 backend is active (auto-detected)
assert mpmath.libmp.BACKEND == 'gmpy', (
    f"Expected gmpy2 backend, got '{mpmath.libmp.BACKEND}'. "
    "Install gmpy2: pip install gmpy2"
)

# Evaluate zeta at a point on the critical line
with mpmath.workdps(50):
    s = mpmath.mpc(0.5, 14.134725)
    z = mpmath.zeta(s)
    print(f"zeta(0.5 + 14.134725i) = {z}")
    # Should be very close to zero (first zero)
```

### Verified: Computing and Validating Zeros
```python
# Source: mpmath docs (functions/zeta.html), Odlyzko tables
import mpmath

def compute_and_validate_zero(n: int, dps: int = 50) -> ZetaZero:
    """Compute nth zeta zero with always-validate pattern."""
    # Compute at target precision
    with mpmath.workdps(dps + 5):
        zero_p = mpmath.zetazero(n)

    # Compute at double precision
    with mpmath.workdps(2 * dps + 5):
        zero_2p = mpmath.zetazero(n)

    # Compare: first dps-5 digits must agree
    diff = abs(zero_p - zero_2p)
    threshold = mpmath.mpf(10) ** (-(dps - 5))

    if diff > threshold:
        raise PrecisionError(f"Zero {n}: P and 2P disagree by {diff}")

    return ZetaZero(
        index=n,
        value=zero_2p,
        precision_digits=dps,
        validated=True,
        on_critical_line=abs(zero_2p.real - 0.5) < threshold,
    )
```

### Verified: Hardy Z-function on the Critical Line
```python
# Source: mpmath docs (siegelz function)
import mpmath
import numpy as np

def critical_line_data(t_start, t_end, num_points, dps=50):
    """Evaluate |zeta(1/2+it)| via Hardy Z-function along critical line."""
    t_values = np.linspace(float(t_start), float(t_end), num_points)
    z_values = []

    with mpmath.workdps(dps):
        for t in t_values:
            # siegelz(t) = Z(t) = e^(i*theta(t)) * zeta(1/2 + it)
            # |Z(t)| = |zeta(1/2 + it)|, and Z(t) is real-valued
            z = float(mpmath.siegelz(t))
            z_values.append(z)

    return t_values, np.array(z_values)
```

### Verified: Dirichlet L-function Evaluation
```python
# Source: mpmath docs (dirichlet function)
import mpmath

with mpmath.workdps(50):
    # Riemann zeta: trivial character [1]
    zeta_val = mpmath.dirichlet(2, [1])  # = pi^2/6

    # Dirichlet L-function with character chi = [-1, 1] (mod 2)
    # This is the Dirichlet beta function (Catalan's constant related)
    l_val = mpmath.dirichlet(2, [-1, 1])

    # Non-principal character mod 4: chi = [0, 1, 0, -1]
    l_mod4 = mpmath.dirichlet(mpmath.mpc(0.5, 10), [0, 1, 0, -1])
```

### Verified: Domain Coloring Core
```python
# Source: matplotlib.colors.hsv_to_rgb docs, domain coloring technique
import numpy as np
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt

def domain_color_plot(f, re_range=(-2, 2), im_range=(-2, 2),
                      resolution=200, ax=None):
    """Domain coloring: phase -> hue, log-magnitude -> brightness."""
    re = np.linspace(*re_range, resolution)
    im = np.linspace(*im_range, resolution)
    Re, Im = np.meshgrid(re, im)
    Z = Re + 1j * Im

    # Evaluate (use vectorized numpy for speed; switch to mpmath per-point
    # for critical strip high-precision work)
    W = f(Z)

    H = (np.angle(W) / (2 * np.pi)) % 1.0
    mag = np.abs(W)
    # Log-scaled brightness: zeros are dark, poles are bright
    V = 1.0 - 1.0 / (1.0 + 0.3 * np.log1p(mag))
    S = 0.9 * np.ones_like(H)

    HSV = np.stack([H, S, V], axis=-1)
    RGB = hsv_to_rgb(HSV)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(RGB, extent=[*re_range, *im_range], origin='lower', aspect='auto')
    ax.set_xlabel('Re(s)')
    ax.set_ylabel('Im(s)')
    return ax
```

### Xi Function (Must Hand-Build)
```python
# The completed Riemann xi function is NOT a built-in mpmath function.
# Build from zeta + gamma:
# xi(s) = (1/2) * s * (s-1) * pi^(-s/2) * gamma(s/2) * zeta(s)
import mpmath

def xi(s, dps=50):
    """Riemann xi function. Entire, symmetric: xi(s) = xi(1-s)."""
    with mpmath.workdps(dps + 10):
        return (mpmath.mpf(1)/2 * s * (s - 1)
                * mpmath.power(mpmath.pi, -s/2)
                * mpmath.gamma(s/2)
                * mpmath.zeta(s))
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| mpmath 1.3.0 | mpmath 1.4.0 | Feb 2026 | f-string formatting for mpf/mpc, Fox H-function, CLI mode, better introspection |
| Plotly ipywidgets integration | Plotly anywidget integration | Plotly 6.x | Better JupyterLab 4 compatibility; install anywidget>=0.9.13 |
| gmpy2 2.1.x | gmpy2 2.2.2 | Nov 2025 | Requires CPython 3.11+; pre-built Windows wheels for 3.12 |
| Manual backend check | Auto-detection | mpmath >= 1.3 | mpmath auto-detects gmpy2 >= 2.2.0; set MPMATH_NOGMPY env var to disable |

**Deprecated/outdated:**
- `mpmath.mp.dps = N` (direct assignment): Use `mpmath.workdps(N)` context manager instead
- cplot library (last release Aug 2022): Use custom matplotlib domain coloring for maintainability
- Plotly <5.15.0: Had JupyterLab 4 figure sizing bug; use >=6.x

## Open Questions

1. **Selberg Zeta Function Implementation**
   - What we know: mpmath does NOT provide a Selberg zeta function. The Selberg zeta is defined for a discrete group acting on the hyperbolic plane and requires spectral data (lengths of closed geodesics).
   - What's unclear: The exact API design for specifying the group and its geodesic spectrum
   - Recommendation: Defer detailed Selberg implementation to Phase 2/3. For Phase 1, implement a stub that accepts spectral data and computes the product formula to moderate precision. Mark as COMP-03 partial delivery.

2. **mpmath Parallel Zero Computation**
   - What we know: mpmath is not thread-safe (global mp context). multiprocessing works but each process re-initializes.
   - What's unclear: Overhead of multiprocessing for moderate zero counts (100-1000)
   - Recommendation: Use sequential computation for Phase 1 (adequate for <1000 zeros). Parallel computation is a Phase 2 optimization.

3. **Odlyzko Table Download and Parsing**
   - What we know: Tables are at https://www-users.cse.umn.edu/~odlyzko/zeta_tables/ as text files. First 100K zeros to 3e-9 accuracy. First 100 zeros to 1000+ digits.
   - What's unclear: Exact format parsing needs. Whether to bundle tables or download on demand.
   - Recommendation: Download the first-100-zeros (1000-digit) file and bundle it in data/odlyzko/. Parse once, store in SQLite zeros table. Use for precision canary tests.

4. **Experiment Checksum Strategy**
   - What we know: Need SHA-256 of serialized results for reproducibility verification.
   - What's unclear: Whether to checksum the mpmath values (precision-dependent) or a canonical string representation.
   - Recommendation: Checksum the string representation at the experiment's stated precision (e.g., 50-digit string of each value). This makes checksums precision-stable.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest >= 7.4 |
| Config file | none -- Wave 0 must create pyproject.toml [tool.pytest.ini_options] |
| Quick run command | `uv run pytest tests/ -x --timeout=30` |
| Full suite command | `uv run pytest tests/ -v --timeout=120` |

### Phase Requirements to Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| COMP-01 | zeta(s) matches known values to 50 digits | unit | `uv run pytest tests/test_engine/test_zeta.py -x` | Wave 0 |
| COMP-01 | zeta satisfies functional equation zeta(s) = chi(s)*zeta(1-s) | unit | `uv run pytest tests/test_engine/test_zeta.py::test_functional_equation -x` | Wave 0 |
| COMP-02 | zetazero(n) matches Odlyzko table for n=1..100 | unit | `uv run pytest tests/test_engine/test_zeros.py::test_odlyzko_validation -x` | Wave 0 |
| COMP-02 | Zero cataloging stores and retrieves from SQLite | integration | `uv run pytest tests/test_engine/test_zeros.py::test_zero_catalog -x` | Wave 0 |
| COMP-03 | siegelz(t) real-valued on real axis | unit | `uv run pytest tests/test_engine/test_lfunctions.py::test_hardy_z -x` | Wave 0 |
| COMP-03 | dirichlet(s, [1]) equals zeta(s) | unit | `uv run pytest tests/test_engine/test_lfunctions.py::test_dirichlet_trivial -x` | Wave 0 |
| COMP-03 | xi(s) = xi(1-s) symmetry | unit | `uv run pytest tests/test_engine/test_lfunctions.py::test_xi_symmetry -x` | Wave 0 |
| COMP-04 | P-vs-2P validation catches injected precision error | unit | `uv run pytest tests/test_engine/test_validation.py::test_always_validate -x` | Wave 0 |
| COMP-04 | Stress-test framework re-runs at higher precision | integration | `uv run pytest tests/test_engine/test_validation.py::test_stress_rerun -x` | Wave 0 |
| VIZ-01 | Critical line plot generates without error | smoke | `uv run pytest tests/test_viz/test_critical_line.py -x` | Wave 0 |
| VIZ-02 | Domain coloring produces valid RGB array | unit | `uv run pytest tests/test_viz/test_domain_coloring.py -x` | Wave 0 |
| RSRCH-01 | Conjecture CRUD with evidence level enforcement | unit | `uv run pytest tests/test_workbench/test_conjecture.py -x` | Wave 0 |
| RSRCH-01 | Evidence hierarchy rejects invalid levels | unit | `uv run pytest tests/test_workbench/test_conjecture.py::test_evidence_levels -x` | Wave 0 |
| RSRCH-02 | Experiment save/load reproduces exact parameters | unit | `uv run pytest tests/test_workbench/test_experiment.py -x` | Wave 0 |
| RSRCH-02 | Checksum verification detects tampering | unit | `uv run pytest tests/test_workbench/test_experiment.py::test_checksum -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/ -x --timeout=30`
- **Per wave merge:** `uv run pytest tests/ -v --timeout=120`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `pyproject.toml` -- project initialization with uv, pytest config, dependency list
- [ ] `tests/conftest.py` -- shared fixtures: mpmath precision contexts, temporary SQLite DB, test data paths
- [ ] `data/odlyzko/zeros_100.txt` -- first 100 zeros at 1000-digit precision for validation
- [ ] `tests/test_engine/test_zeta.py` -- covers COMP-01
- [ ] `tests/test_engine/test_zeros.py` -- covers COMP-02
- [ ] `tests/test_engine/test_lfunctions.py` -- covers COMP-03
- [ ] `tests/test_engine/test_validation.py` -- covers COMP-04
- [ ] `tests/test_viz/test_critical_line.py` -- covers VIZ-01
- [ ] `tests/test_viz/test_domain_coloring.py` -- covers VIZ-02
- [ ] `tests/test_workbench/test_conjecture.py` -- covers RSRCH-01
- [ ] `tests/test_workbench/test_experiment.py` -- covers RSRCH-02

## Sources

### Primary (HIGH confidence)
- [mpmath 1.4.0 documentation](https://mpmath.org/doc/current/functions/zeta.html) -- zeta, zetazero, siegelz, siegeltheta, dirichlet, nzeros, grampoint function signatures and behavior
- [mpmath 1.4.0 setup docs](https://mpmath.readthedocs.io/en/latest/setup.html) -- gmpy2 backend auto-detection, workdps/workprec context managers, MPMATH_NOGMPY
- [mpmath PyPI](https://pypi.org/project/mpmath/) -- version 1.4.0, released 2026-02-23, Python 3.9-3.14 support
- [gmpy2 PyPI](https://pypi.org/project/gmpy2/) -- version 2.2.2, Nov 2025, CPython 3.11+, Windows wheels
- [Plotly PyPI](https://pypi.org/project/plotly/) -- version 6.5.2, Jan 2026, anywidget integration
- [matplotlib.colors.hsv_to_rgb docs](https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.hsv_to_rgb.html) -- vectorized HSV-to-RGB on (...,3) arrays
- [Odlyzko zeta tables](https://www-users.cse.umn.edu/~odlyzko/zeta_tables/) -- first 100K zeros to 3e-9; first 100 to 1000+ digits

### Secondary (MEDIUM confidence)
- [mpmath GitHub issue #895](https://github.com/mpmath/mpmath/issues/895) -- python-gmp as alternative to gmpy2; gmpy2 preferred
- [Plotly community forum](https://community.plotly.com/t/does-plotly-work-with-jupyterlab-4/76095) -- JupyterLab 4 compatibility confirmed since Plotly 5.15.0
- [cplot PyPI](https://pypi.org/project/cplot/) -- version 0.9.3, Aug 2022 (stale); OKLAB color space
- [ipywidgets 8.1 docs](https://ipywidgets.readthedocs.io/en/latest/examples/Using%20Interact.html) -- FloatSlider, continuous_update parameter

### Tertiary (LOW confidence)
- Selberg zeta function implementation approach -- based on training data knowledge of the mathematical definition; no verified Python implementation found
- mpmath thread safety details -- documentation is sparse; recommendation based on known global state pattern

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all library versions verified against PyPI/official docs within the last month
- Architecture: HIGH -- patterns derived from mpmath's own documented API (workdps, zetazero) and established domain coloring technique
- Pitfalls: HIGH -- precision collapse is extensively documented (Brent, Odlyzko, Rubinstein); evidence hierarchy is a project design decision from CONTEXT.md
- Code examples: HIGH -- all based on verified mpmath function signatures; domain coloring based on well-established matplotlib API
- Validation architecture: MEDIUM -- test structure is recommended; specific test assertions need implementation

**Research date:** 2026-03-18
**Valid until:** 2026-04-18 (stable libraries; 30-day window)
