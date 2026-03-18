# Architecture Patterns

**Domain:** Hybrid computational research + formal verification platform for the Riemann Hypothesis
**Researched:** 2026-03-18
**Confidence:** MEDIUM (based on training data; web search unavailable for verification of latest library versions)

## System Overview

```
+------------------------------------------------------------------+
|                     RIEMANN PLATFORM                              |
|                                                                   |
|  +---------------------+     +-----------------------------+     |
|  |  RESEARCH WORKBENCH |     |    VISUALIZATION LAYER      |     |
|  |  (Orchestration)    |<--->|    (Projection Pipeline)     |     |
|  |                     |     |                              |     |
|  | - Session mgmt      |     | - 2D/3D rendering           |     |
|  | - Conjecture tracker |     | - N-dim projection          |     |
|  | - Experiment log     |     | - Interactive exploration    |     |
|  | - Proof state mgmt   |     | - Animation / parameter     |     |
|  +----------+----------+     |   sweeps                    |     |
|             |                +-------------+---------------+     |
|             |                              |                      |
|             v                              v                      |
|  +---------------------+     +-----------------------------+     |
|  | COMPUTATION ENGINE  |<--->|  DATA / OBJECT STORE        |     |
|  | (Core Math)         |     |  (Mathematical Objects)      |     |
|  |                     |     |                              |     |
|  | - Zeta evaluation   |     | - Computed zeros            |     |
|  | - Zero-finding      |     | - Spectral data             |     |
|  | - Spectral operators|     | - Matrices / operators      |     |
|  | - Random matrices   |     | - Conjecture records        |     |
|  | - Modular forms     |     | - Session state             |     |
|  | - Info-theoretic    |     | - Cached computations       |     |
|  +----------+----------+     +-----------------------------+     |
|             |                                                     |
|             v                                                     |
|  +---------------------+     +-----------------------------+     |
|  | ANALYSIS MODULES    |     |  FORMALIZATION PIPELINE     |     |
|  | (Cross-Disciplinary)|     |  (Lean 4)                    |     |
|  |                     |     |                              |     |
|  | - Pattern detection  |---->| - Statement translator      |     |
|  | - Anomaly surfacing  |     | - Proof skeleton generator  |     |
|  | - Correlation finder |     | - Mathlib integration       |     |
|  | - Statistical tests  |     | - Proof state tracker       |     |
|  +---------------------+     +-----------------------------+     |
|                                                                   |
+------------------------------------------------------------------+
```

### Layered Architecture

The platform follows a **layered architecture with plugin modules**, organized bottom-up:

1. **Layer 0 -- Data/Object Store**: Persistence and caching of mathematical objects
2. **Layer 1 -- Computation Engine**: Core mathematical evaluation (zeta, L-functions, operators)
3. **Layer 2 -- Analysis Modules**: Cross-disciplinary analysis (pluggable)
4. **Layer 3 -- Visualization Layer**: Projection pipeline and interactive rendering
5. **Layer 4 -- Research Workbench**: Session orchestration, conjecture tracking, experiment management
6. **Layer 5 -- Formalization Pipeline**: Lean 4 integration (separate process, loosely coupled)

Dependencies flow downward. The Formalization Pipeline sits beside the stack rather than on top -- it consumes outputs from any layer but runs in a separate Lean 4 process.

## Component Responsibilities

### 1. Computation Engine (Core)

**Responsibility:** All numerical evaluation of mathematical functions and objects. This is the foundation everything else depends on.

| Subcomponent | What It Does | Key Library |
|---|---|---|
| Zeta evaluator | Riemann zeta function at arbitrary precision | `mpmath` (zeta, zetazero) |
| Zero finder | Locate non-trivial zeros on the critical strip | `mpmath` + custom Newton/bisection |
| L-function evaluator | Dirichlet L-functions, Dedekind zeta | `mpmath` + custom |
| Spectral operator engine | Construct and diagonalize operators (GUE, quantum graphs) | `numpy`/`scipy` for moderate dim; custom for symbolic |
| Random matrix generator | GUE/GOE ensembles, eigenvalue statistics | `numpy` + custom |
| Modular form evaluator | q-expansions, Hecke operators, modular symbols | Custom (possibly wrapping `sage` components) |
| Information-theoretic measures | Entropy of zero spacings, mutual information, KL divergence | `scipy.stats` + custom |

**Boundary rule:** The computation engine NEVER does visualization. It returns mathematical objects (numbers, arrays, matrices, symbolic expressions). It knows nothing about how results will be displayed.

**Precision contract:** All functions accept a `precision` parameter (number of decimal digits). Default: 50 digits. The engine uses `mpmath.mp.dps` to set working precision. Functions that mix mpmath (arbitrary precision) and numpy (machine precision) must clearly document which regime they operate in.

### 2. Data / Object Store

**Responsibility:** Persist, cache, and retrieve mathematical objects. Avoid recomputation of expensive results.

| Concern | Approach |
|---|---|
| Zero database | SQLite table of known zeros (index, real part, imaginary part, precision verified) |
| Computation cache | Content-addressed store keyed by (function, parameters, precision) |
| Session state | JSON/YAML files per research session |
| Conjecture records | Structured YAML with status, evidence, counterexample attempts |
| Large arrays | NumPy `.npy` / `.npz` files for matrices and spectral data |

**Boundary rule:** The store is a dumb persistence layer. It does not compute, analyze, or interpret. Components request objects by key; the store returns them or signals "not cached."

**Why SQLite (not Postgres, not flat files for zeros):** The project is a local research tool (explicitly out of scope: web deployment). SQLite is zero-config, single-file, supports SQL queries over the zero database (e.g., "all zeros with imaginary part between 1000 and 2000"), and handles millions of rows. No server process needed.

### 3. Analysis Modules (Pluggable)

**Responsibility:** Cross-disciplinary analysis that looks for patterns, anomalies, and correlations in computed data. This is where the unconventional approach lives.

Each module is a **plugin** with a standard interface:

```python
class AnalysisModule(Protocol):
    """Standard interface for cross-disciplinary analysis modules."""

    @property
    def name(self) -> str: ...

    @property
    def domain(self) -> str:
        """e.g., 'spectral_theory', 'random_matrices', 'information_theory'"""
        ...

    def analyze(self, data: MathObject, params: dict) -> AnalysisResult: ...

    def visualizable_outputs(self) -> list[str]:
        """Names of outputs that can be sent to the visualization layer."""
        ...
```

**Planned modules (build order):**

| Module | Domain | What It Does | Depends On |
|---|---|---|---|
| `ZeroSpacingAnalyzer` | Statistics | Nearest-neighbor spacing, pair correlation of zeros | Computation Engine (zeros) |
| `GUEComparator` | Random Matrices | Compare zero statistics to GUE predictions (Montgomery-Odlyzko) | ZeroSpacingAnalyzer + Random Matrix Generator |
| `SpectralOperatorModule` | Spectral Theory | Construct candidate operators whose eigenvalues model zeros | Computation Engine (spectral) |
| `EntropyAnalyzer` | Information Theory | Entropy measures on zero distributions, detect structure | ZeroSpacingAnalyzer |
| `ModularFormBridge` | Number Theory | Connections between modular forms and L-function zeros | Computation Engine (modular forms) |
| `HighDimGeometryModule` | Geometry | Embed zeros / spectral data in higher-dimensional spaces, look for structure | Multiple engines |
| `AdelicAnalyzer` | Algebraic Number Theory | Adelic completions, local-global connections | Advanced -- likely Phase 3+ |

**Boundary rule:** Modules receive data, return results. They do not persist data (the workbench handles that). They do not render visualizations (they declare what outputs are visualizable, and the visualization layer handles rendering).

### 4. Visualization Layer (Projection Pipeline)

**Responsibility:** Render mathematical objects and analysis results as interactive 2D/3D visualizations. Handle projection from N-dimensional spaces.

**Sub-pipeline:**

```
N-dim math object
       |
       v
  [Projection Engine]  -- PCA, t-SNE, UMAP, stereographic, custom
       |
       v
  3D scene graph
       |
       v
  [Renderer]  -- Plotly (interactive 3D), Matplotlib (publication 2D),
       |          or custom WebGL via Panel/Bokeh
       v
  Interactive widget (Pan, zoom, rotate, parameter sliders)
```

| Subcomponent | What It Does | Key Library |
|---|---|---|
| Projection engine | Dimensionality reduction / geometric projection | Custom + `scikit-learn` (PCA, t-SNE, UMAP) |
| 3D renderer | Interactive 3D plots with rotation, zoom | `plotly` (primary), `pyvista` (meshes) |
| 2D renderer | High-quality 2D plots and complex plane views | `matplotlib` |
| Dashboard / widgets | Parameter sliders, animation controls, side-by-side views | `panel` (HoloViz ecosystem) |
| Complex plane viewer | Specialized: domain coloring, phase portraits, zero locations | Custom on `matplotlib` |

**Why Plotly for 3D (not Mayavi, not raw Matplotlib 3d):** Plotly provides interactive 3D in the browser with zoom/rotate/hover out of the box. It works in Jupyter notebooks (the primary interface) without a separate GUI framework. Matplotlib 3D is not truly interactive. Mayavi requires VTK and is heavyweight for a research tool.

**Why Panel for dashboards (not Streamlit, not Dash):** Panel integrates natively with Jupyter, supports Matplotlib and Plotly objects directly, and allows building dashboards that can also run standalone. Streamlit forces a separate server model. Dash is Plotly-specific.

**Boundary rule:** The visualization layer NEVER computes mathematical results. It receives arrays/objects and renders them. If a visualization requires a derived quantity (e.g., nearest-neighbor spacings from a list of zeros), that computation happens in the analysis module, not in the visualization code.

### 5. Research Workbench (Orchestration)

**Responsibility:** The user-facing layer that ties everything together. Manages research sessions, tracks conjectures, logs experiments, and provides the interactive exploration experience.

| Subcomponent | What It Does |
|---|---|
| Session manager | Create/resume research sessions with full state |
| Experiment runner | Execute a defined computation + analysis + visualization pipeline |
| Conjecture tracker | Record conjectures with status (speculative / computational evidence / formalized / proved / disproved) |
| Insight journal | Timestamped log of observations, linked to experiments |
| Notebook integration | Jupyter kernel extensions / magic commands for common operations |
| Claude interface layer | Structured prompts for Claude to perform formalism on demand |

**Primary interface: Jupyter Notebooks.** The user interacts through notebooks. The workbench provides:
- Magic commands: `%riemann.zeros 1000` (compute first 1000 zeros)
- Rich display: Custom `_repr_html_` for mathematical objects
- Session context: Automatic logging of what was computed/observed

**Why Jupyter (not a custom GUI, not a CLI):** The user is technically capable with Python. Jupyter provides the right balance: code when needed, rich output always, reproducible sessions, inline visualization. A custom GUI would be enormous to build and less flexible. A pure CLI cannot show visualizations.

**Boundary rule:** The workbench orchestrates but does not implement math or rendering. It calls the computation engine, passes results to analysis modules, and sends outputs to visualization. It is the "glue" layer.

### 6. Formalization Pipeline (Lean 4)

**Responsibility:** Translate computational findings into formal Lean 4 statements and assist with proof construction. This is a separate, loosely coupled system.

| Subcomponent | What It Does |
|---|---|
| Statement translator | Convert conjectures from workbench format to Lean 4 syntax |
| Proof skeleton generator | Generate Lean 4 proof scaffolding with `sorry` placeholders |
| Mathlib bridge | Interface with Mathlib's existing number theory, analysis, topology |
| Proof state tracker | Track which lemmas are proved, which have `sorry`, overall progress |
| Verification runner | Invoke `lean` CLI to typecheck proofs, report errors back to workbench |

**Integration model:** The Lean 4 pipeline is a **separate process** communicating with the Python platform via:
1. **File-based exchange:** Python writes `.lean` files, invokes `lean` CLI, reads results
2. **Structured output:** Lean errors/warnings parsed back into Python objects
3. **Lake project:** The Lean code lives in its own `lean/` directory with a proper `lakefile.lean`

**Why file-based (not a server, not FFI):** Lean 4 compilation is a batch process. There is no stable Python-Lean FFI. The `lean` CLI and Lake build system are the supported interfaces. File-based exchange is simple, debuggable, and reliable. A Language Server Protocol (LSP) connection is possible for richer interaction (hover info, goal states) but is a later optimization, not essential for MVP.

**Why Lean 4 (confirmed from PROJECT.md):** Already decided. Lean 4 has the most active proof formalization community (Mathlib), strong tooling, and the best ecosystem for new mathematical formalization projects.

## Project Structure

```
riemann/
|-- pyproject.toml                 # Project config, dependencies
|-- src/
|   |-- riemann/
|   |   |-- __init__.py
|   |   |-- engine/                # Layer 1: Computation Engine
|   |   |   |-- __init__.py
|   |   |   |-- zeta.py            # Zeta function evaluation
|   |   |   |-- zeros.py           # Zero-finding algorithms
|   |   |   |-- lfunctions.py      # L-function evaluation
|   |   |   |-- spectral.py        # Spectral operator construction
|   |   |   |-- random_matrix.py   # GUE/GOE generation
|   |   |   |-- modular.py         # Modular form evaluation
|   |   |   |-- precision.py       # Precision management context
|   |   |
|   |   |-- store/                 # Layer 0: Data/Object Store
|   |   |   |-- __init__.py
|   |   |   |-- cache.py           # Computation cache (content-addressed)
|   |   |   |-- zeros_db.py        # SQLite zero database
|   |   |   |-- objects.py         # Math object serialization
|   |   |   |-- sessions.py        # Session state persistence
|   |   |
|   |   |-- analysis/              # Layer 2: Analysis Modules (pluggable)
|   |   |   |-- __init__.py
|   |   |   |-- base.py            # AnalysisModule protocol + AnalysisResult
|   |   |   |-- spacing.py         # Zero spacing statistics
|   |   |   |-- gue_comparison.py  # GUE comparison
|   |   |   |-- spectral_module.py # Spectral theory analysis
|   |   |   |-- entropy.py         # Information-theoretic analysis
|   |   |   |-- modular_bridge.py  # Modular form connections
|   |   |   |-- highdim.py         # Higher-dimensional embedding/analysis
|   |   |
|   |   |-- viz/                   # Layer 3: Visualization Layer
|   |   |   |-- __init__.py
|   |   |   |-- projection.py      # N-dim -> 2D/3D projection engine
|   |   |   |-- complex_plane.py   # Domain coloring, phase portraits
|   |   |   |-- plots_3d.py        # Interactive 3D visualization
|   |   |   |-- plots_2d.py        # 2D plots (zero distribution, spacing histograms)
|   |   |   |-- dashboard.py       # Panel dashboard builder
|   |   |   |-- styles.py          # Consistent visual theming
|   |   |
|   |   |-- workbench/             # Layer 4: Research Workbench
|   |   |   |-- __init__.py
|   |   |   |-- session.py         # Session management
|   |   |   |-- experiment.py      # Experiment runner
|   |   |   |-- conjecture.py      # Conjecture tracker
|   |   |   |-- journal.py         # Insight journal
|   |   |   |-- magics.py          # Jupyter magic commands
|   |   |   |-- display.py         # Rich display formatters
|   |   |
|   |   |-- lean/                  # Layer 5: Lean bridge (Python side)
|   |   |   |-- __init__.py
|   |   |   |-- translator.py      # Python conjecture -> Lean statement
|   |   |   |-- skeleton.py        # Proof skeleton generator
|   |   |   |-- runner.py          # Lean CLI invocation + result parsing
|   |   |   |-- state.py           # Proof state tracking
|   |   |
|   |   |-- types.py               # Shared type definitions (MathObject, etc.)
|   |   |-- config.py              # Platform configuration
|   |
|-- lean/                          # Layer 5: Lean 4 project (separate)
|   |-- lakefile.lean              # Lake build config
|   |-- lean-toolchain             # Lean version pin
|   |-- Riemann/
|   |   |-- Basic.lean             # Basic definitions (zeta, zeros, RH statement)
|   |   |-- Spectral.lean          # Spectral theory formalizations
|   |   |-- Analytic.lean          # Analytic number theory
|   |   |-- Main.lean              # Top-level imports
|
|-- notebooks/                     # Jupyter notebooks (primary interface)
|   |-- 00_getting_started.ipynb
|   |-- 01_exploring_zeros.ipynb
|   |-- 02_gue_comparison.ipynb
|   |-- explorations/              # User's experiment notebooks
|
|-- data/                          # Persistent data
|   |-- zeros.db                   # SQLite zero database
|   |-- cache/                     # Computation cache
|   |-- sessions/                  # Session state files
|
|-- tests/
|   |-- test_engine/
|   |-- test_analysis/
|   |-- test_viz/
|   |-- test_store/
|   |-- test_lean/
```

## Architectural Patterns

### Pattern 1: Mathematical Object as First-Class Citizen

**What:** All mathematical objects (zeros, functions, operators, matrices, analysis results) are represented as typed Python objects with standard interfaces, not raw arrays or dicts.

**Why:** Enables consistent serialization, caching, visualization dispatch, and Lean translation. A `ZetaZero` object knows its index, value, and precision. An `AnalysisResult` knows what visualization types it supports.

**Example:**

```python
from dataclasses import dataclass
from mpmath import mpf, mpc
from typing import Literal

@dataclass(frozen=True)
class ZetaZero:
    """A non-trivial zero of the Riemann zeta function."""
    index: int                    # Ordinal (1st, 2nd, ... zero)
    value: mpc                    # The zero itself (should have real part ~0.5)
    precision_digits: int         # Verified to this many digits
    on_critical_line: bool | None # None = not yet verified to sufficient precision

@dataclass
class AnalysisResult:
    """Output from an analysis module."""
    module_name: str
    description: str
    data: dict                    # Named arrays/values
    visualizable: list[str]       # Keys in data that can be visualized
    metadata: dict                # Computation parameters, timing, etc.
```

### Pattern 2: Precision Context Manager

**What:** A context manager that sets arbitrary-precision arithmetic scope, ensuring precision is explicit and cannot accidentally leak between computations.

**Why:** mpmath uses a global `mp.dps` for precision. Mixing precisions silently causes wrong results. The context manager makes precision explicit and scoped.

**Example:**

```python
from contextlib import contextmanager
import mpmath

@contextmanager
def precision(digits: int):
    """Set mpmath precision for a block of computation."""
    old_dps = mpmath.mp.dps
    mpmath.mp.dps = digits + 10  # Extra guard digits
    try:
        yield
    finally:
        mpmath.mp.dps = old_dps

# Usage:
with precision(100):
    z = mpmath.zetazero(1000)  # 100-digit precision
```

### Pattern 3: Plugin Registry for Analysis Modules

**What:** Analysis modules register themselves with a central registry. The workbench discovers available modules at startup. New modules can be added without modifying existing code.

**Why:** The cross-disciplinary approach means new analysis angles will be added throughout the project. The plugin pattern avoids modifying a central switch statement every time.

**Example:**

```python
# In analysis/__init__.py
_registry: dict[str, type[AnalysisModule]] = {}

def register(cls: type[AnalysisModule]) -> type[AnalysisModule]:
    _registry[cls.name] = cls
    return cls

def get_module(name: str) -> AnalysisModule:
    return _registry[name]()

def list_modules() -> list[str]:
    return list(_registry.keys())

# In analysis/spacing.py
@register
class ZeroSpacingAnalyzer:
    name = "zero_spacing"
    domain = "statistics"
    ...
```

### Pattern 4: Projection Pipeline (N-dim to Visual)

**What:** A composable pipeline that takes N-dimensional mathematical data, applies a projection method, and produces a renderable scene.

**Why:** Higher-dimensional exploration is central to the project's thesis. The projection step must be swappable (PCA today, custom geometric projection tomorrow) without rewriting visualization code.

**Example:**

```python
class ProjectionPipeline:
    def __init__(self):
        self.steps: list[ProjectionStep] = []

    def add(self, step: ProjectionStep) -> "ProjectionPipeline":
        self.steps.append(step)
        return self

    def project(self, data: NDArray, target_dim: int = 3) -> NDArray:
        result = data
        for step in self.steps:
            result = step.transform(result)
        assert result.shape[-1] == target_dim
        return result

# Usage:
pipeline = (ProjectionPipeline()
    .add(CenterAndScale())
    .add(PCAProjection(n_components=3))
)
scene_data = pipeline.project(high_dim_embedding)
```

### Pattern 5: Experiment as Reproducible Unit

**What:** Every exploration the user runs is captured as an `Experiment` -- a reproducible record of computation + analysis + visualization with all parameters.

**Why:** The user is exploring. They will want to re-run experiments with different parameters, compare results, and build on prior work. Without explicit experiment tracking, research becomes unreproducible.

**Example:**

```python
@dataclass
class Experiment:
    id: str                       # UUID
    timestamp: datetime
    description: str              # User's description
    computation: dict             # What was computed, with all params
    analysis: dict                # Which modules ran, with all params
    results: dict                 # References to stored results
    observations: list[str]       # User's notes
    conjecture_refs: list[str]    # Linked conjectures
```

### Pattern 6: Lean File-Exchange Protocol

**What:** A structured protocol for Python-to-Lean communication via files. Python generates `.lean` files in a staging directory, invokes Lake, and parses structured output.

**Why:** Clean separation between the Python exploration world and the Lean formalization world. No fragile FFI. Debuggable (you can open the `.lean` files and read them).

**Example:**

```python
class LeanRunner:
    def __init__(self, lean_project_dir: Path):
        self.project_dir = lean_project_dir
        self.staging_dir = lean_project_dir / "Riemann" / "Generated"

    def submit(self, lean_code: str, filename: str) -> LeanResult:
        filepath = self.staging_dir / filename
        filepath.write_text(lean_code)
        result = subprocess.run(
            ["lake", "build"],
            cwd=self.project_dir,
            capture_output=True, text=True
        )
        return self._parse_result(result, filepath)
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: God Notebook

**What:** A single massive Jupyter notebook that does computation, analysis, visualization, and conjecture tracking all in one.

**Why bad:** Unreproducible, impossible to refactor, merge conflicts, and execution order bugs. The notebook becomes the platform instead of using the platform.

**Instead:** Notebooks import from the `riemann` package. Computation logic lives in `.py` files. Notebooks are thin orchestration layers that call library functions and display results.

### Anti-Pattern 2: Precision Confusion

**What:** Mixing mpmath (arbitrary precision) and numpy (machine float64) without clear boundaries, silently losing precision.

**Why bad:** The entire point of the computation engine is numerical precision. A single accidental `float()` conversion can destroy 100 digits of precision, producing plausible but wrong results.

**Instead:** Functions are tagged as `arbitrary_precision` or `machine_precision`. The type system distinguishes `mpf`/`mpc` from `float`/`complex`. Conversion is always explicit with a `to_machine_float()` function that logs a warning.

### Anti-Pattern 3: Visualization-Driven Computation

**What:** Computing mathematical results inside visualization callbacks or rendering functions.

**Why bad:** Computations become unreproducible (tied to UI state), uncacheable, and untestable. Changing the visualization framework means rewriting computation code.

**Instead:** Strict separation. Computation produces data. Visualization renders data. They never mix.

### Anti-Pattern 4: Monolithic Lean Formalization

**What:** One huge `.lean` file that tries to formalize everything at once.

**Why bad:** Lean compilation is slow. A monolithic file means recompiling everything for every change. It also makes it impossible to track progress on individual lemmas.

**Instead:** Small, focused `.lean` files organized by topic. Each file formalizes one concept or one lemma. The proof state tracker knows which files exist and their status.

## Data Flow

### Flow 1: Exploration Loop (Primary User Flow)

```
User (notebook)
  |
  | "Show me the first 1000 zeros and their spacing distribution"
  v
Research Workbench (experiment runner)
  |
  | 1. Create Experiment record
  | 2. Check cache for existing computation
  v
Computation Engine
  |
  | Compute zeros (mpmath.zetazero), cache results
  v
Data Store (cache zeros, return ZetaZero objects)
  |
  v
Analysis Module (ZeroSpacingAnalyzer)
  |
  | Compute nearest-neighbor spacings, normalize
  | Return AnalysisResult with histogram data
  v
Visualization Layer
  |
  | Render: (1) zeros on critical line, (2) spacing histogram
  | Compare overlay: GUE prediction curve
  v
User sees interactive plots, records observations
```

### Flow 2: Higher-Dimensional Exploration

```
User: "Embed the first 500 zeros in a spectral feature space
       and project to 3D -- do they cluster?"
  |
  v
Computation Engine
  |
  | Compute zeros + derived features:
  |   - spacing to neighbors
  |   - local density
  |   - connection to nearby L-function zeros
  |   Result: 500 x N feature matrix (N could be 10-50 dims)
  v
Analysis Module (HighDimGeometryModule)
  |
  | Analyze structure in N-dim:
  |   - Persistent homology
  |   - Clustering (HDBSCAN)
  |   - Manifold detection
  v
Projection Pipeline
  |
  | Project N-dim -> 3D:
  |   - PCA (preserves variance)
  |   - UMAP (preserves topology)
  |   - Custom stereographic (preserves geometry)
  v
Visualization Layer (3D interactive)
  |
  | Plotly 3D scatter, colored by cluster/feature
  | User rotates, zooms, selects points for detail
  v
User identifies structure, records conjecture
```

### Flow 3: Formalization Pipeline

```
User: "Formalize the conjecture that zero spacings converge
       to GUE in the limit"
  |
  v
Conjecture Tracker (retrieves conjecture record + evidence)
  |
  v
Statement Translator (Python -> Lean 4)
  |
  | Generate Lean 4 statement:
  |   theorem gue_convergence :
  |     ...  := by sorry
  |
  v
Proof Skeleton Generator
  |
  | Add structure: imports, intermediate lemmas with sorry,
  | references to Mathlib theorems
  |
  v
Lean Runner
  |
  | Write .lean files to lean/Riemann/Generated/
  | Run `lake build`
  | Parse output: typechecks? errors? sorry count?
  |
  v
Proof State Tracker
  |
  | Update: "GUE convergence" -- 1 theorem, 4 sorry placeholders
  | Report back to workbench
  |
  v
User + Claude iterate on filling sorry holes
```

### Flow 4: Pattern Detection (Background)

```
Analysis Module (anomaly surfacing, running periodically or on-demand)
  |
  | Scan cached zero data for:
  |   - Deviations from expected distributions
  |   - Unexpected correlations between features
  |   - Clusters in higher-dimensional embeddings
  |
  v
Workbench (alert system)
  |
  | "Anomaly detected: zeros 4500-4600 show unusual
  |  spacing pattern -- 2.3 sigma deviation from GUE"
  |
  v
User investigates with exploration loop (Flow 1)
```

## Build Order and Dependencies

The dependency graph determines what must be built first.

```
Phase 1: Foundation
  [Computation Engine: zeta + zeros] --> [Data Store: cache + zeros_db]
      |
      v
Phase 2: See Results
  [Visualization: 2D complex plane + basic plots]
  [Workbench: basic session + notebook integration]
      |
      v
Phase 3: Cross-Disciplinary Analysis
  [Analysis Modules: spacing, GUE comparison, entropy]
  [Visualization: 3D interactive]
  [Workbench: experiment tracking, conjecture tracker]
      |
      v
Phase 4: Higher Dimensions
  [Computation Engine: spectral operators, modular forms]
  [Analysis Modules: high-dim geometry, spectral module]
  [Visualization: projection pipeline (N-dim -> 3D)]
      |
      v
Phase 5: Formalization
  [Lean 4 project setup + Mathlib integration]
  [Statement translator + skeleton generator]
  [Proof state tracker]
      |
      v
Phase 6: Advanced
  [Adelic analysis, advanced modular form connections]
  [Pattern detection / anomaly surfacing (automated)]
  [Dashboard / publication-quality output]
```

**Why this order:**

1. **Computation Engine first** because nothing else can function without numerical results. You cannot visualize what you have not computed. You cannot analyze what does not exist.

2. **Visualization in Phase 2** because the user's primary interaction mode is visual exploration. Without seeing results, the user cannot direct research. Early visualization also validates that computation results are correct (visual sanity checking).

3. **Analysis Modules in Phase 3** because they require both computed data (Phase 1) and visualization (Phase 2) to be useful. An analysis result you cannot see is not actionable for this user.

4. **Higher-dimensional work in Phase 4** because it is the project's differentiator but requires the foundation (compute, analyze, visualize) to be solid first. Attempting N-dimensional projection before basic 2D/3D works will produce confusing results.

5. **Lean 4 formalization in Phase 5** because formalization only makes sense after the user has found something worth formalizing. Premature formalization is wasted effort. The exploration pipeline (Phases 1-4) must produce insights first.

6. **Advanced features last** because they build on everything else and are the least certain in design. Adelic analysis and automated pattern detection will be shaped by what the user discovers in Phases 1-5.

## Scalability Considerations

| Concern | 100 zeros | 100K zeros | 10M zeros |
|---|---|---|---|
| Computation time | Seconds | Hours (parallelize) | Days (distribute or precompute) |
| Storage | Trivial | ~50MB | ~5GB (SQLite handles this) |
| Visualization | Direct plot | Subsample or density plot | Aggregated statistics only |
| Memory | Trivial | ~1GB (mpmath objects are large) | Out-of-core required (chunked loading) |

**Key scaling decision:** For the MVP, optimize for interactivity with up to ~10K zeros. Beyond that, precompute and cache. The LMFDB project has precomputed billions of zeros -- consider importing rather than recomputing for large-scale analysis.

**Parallelization:** Zero computation is embarrassingly parallel (each zero is independent). Use `multiprocessing` (not `threading` -- GIL) or `joblib` for parallel zero-finding. mpmath is not thread-safe; each worker needs its own mpmath context.

## Technology Compatibility Notes

| Library | Role | Confidence | Notes |
|---|---|---|---|
| `mpmath` | Arbitrary precision zeta, zeros | HIGH | De facto standard; `mpmath.zetazero()` is well-tested |
| `numpy` | Machine-precision arrays, linear algebra | HIGH | Foundation library |
| `scipy` | Statistics, eigenvalue problems, optimization | HIGH | Stable, well-documented |
| `plotly` | Interactive 3D visualization | HIGH | Mature, Jupyter-native |
| `matplotlib` | 2D publication-quality plots | HIGH | Standard |
| `panel` | Dashboard / widget framework | MEDIUM | Good HoloViz ecosystem; verify current Jupyter compatibility |
| `scikit-learn` | PCA, t-SNE, UMAP projections | HIGH | `umap-learn` is separate package |
| `SQLite` (via `sqlite3`) | Zero database | HIGH | Standard library |
| Lean 4 + Mathlib | Formal verification | MEDIUM | Lean 4 stable; Mathlib coverage of analytic number theory is partial |
| `elan` | Lean toolchain management | MEDIUM | Standard Lean installer; verify Windows support |

## Sources

- Project requirements from `.planning/PROJECT.md`
- mpmath documentation (training data -- mpmath has been stable for years; HIGH confidence)
- Lean 4 / Mathlib ecosystem knowledge (training data -- MEDIUM confidence, Mathlib evolves rapidly)
- Plotly / Matplotlib / Panel knowledge (training data -- HIGH confidence for core features)
- Riemann Hypothesis computational approaches (Montgomery-Odlyzko, Keating-Snaith, spectral interpretation) -- domain knowledge
- LMFDB (L-functions and Modular Forms DataBase) as reference architecture for large-scale number theory computation

**Note:** Web search was unavailable during this research. All technology recommendations are based on training data (cutoff: early 2025). Before implementation, verify: (1) current Lean 4 toolchain version, (2) Mathlib coverage of relevant areas, (3) Panel compatibility with current Jupyter. These are the most likely areas where information may be stale.
