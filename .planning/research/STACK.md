# Technology Stack

**Project:** Riemann -- Hybrid Computational Math Research Platform & Formal Proof Workbench
**Researched:** 2026-03-18
**Overall confidence:** MEDIUM (versions based on training data through May 2025; all libraries are mature and stable, but exact latest patch versions should be validated with `pip index versions <pkg>` before installation)

---

## Recommended Stack

### Python Version

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Python | >=3.11, recommend 3.12 | Runtime | 3.12 has significant performance improvements (specializing adaptive interpreter) and better error messages. 3.11+ required by modern NumPy/SciPy. Avoid 3.13 initially -- some scientific libraries lag on bleeding-edge Python support. |

**Confidence:** HIGH -- Python 3.12 is the safe choice for scientific computing as of early 2025; 3.13 may be viable by now but validate library support.

---

### Core Framework: Arbitrary-Precision Arithmetic

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **mpmath** | >=1.3 | Arbitrary-precision floating-point arithmetic, zeta function evaluation, special functions | **The** library for this project. mpmath is the only Python library with production-quality arbitrary-precision implementations of the Riemann zeta function (`mpmath.zeta`), Dirichlet L-functions, the Riemann-Siegel formula, and zero-finding (`mpmath.zetazero`). It supports thousands of digits of precision. No alternative exists in the Python ecosystem with comparable coverage. |
| **gmpy2** | >=2.1 | GMP/MPFR/MPC bindings for accelerated multi-precision arithmetic | gmpy2 provides C-level speed for multi-precision integer and floating-point operations. mpmath can use gmpy2 as its backend (set `mpmath.mp.backend = 'gmpy'`), yielding 2-10x speedups for high-precision computation. Essential when evaluating zeta at thousands of digits. |

**Confidence:** HIGH -- mpmath is the undisputed standard for arbitrary-precision special function evaluation in Python. gmpy2 acceleration is well-documented. These libraries have stable APIs.

**Critical note on mpmath:** The Riemann zeta function at high precision is computationally expensive. For zeros with imaginary part > 10^6, expect evaluation times of seconds per zero even with gmpy2 backend. Plan computation strategies accordingly.

---

### Core Framework: Numerical Computing

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **NumPy** | >=1.26, recommend ~2.0 | N-dimensional arrays, linear algebra, FFT | Foundation of the entire scientific Python ecosystem. NumPy 2.0 brought a cleaned-up API and improved performance. All other scientific libraries depend on it. Required for matrix operations in random matrix theory, higher-dimensional array manipulations, and fast vectorized computation at machine precision. |
| **SciPy** | >=1.12 | Sparse linear algebra, eigenvalue problems, signal processing, optimization, special functions | SciPy's `scipy.linalg.eig`/`eigvals` for random matrix eigenvalue computation, `scipy.signal` for spectral analysis, `scipy.sparse` for large operator matrices, `scipy.special` for machine-precision special functions. The spectral analysis and linear algebra modules are central to the physics/spectral theory approach. |

**Confidence:** HIGH -- NumPy and SciPy are non-negotiable for scientific Python. Version recommendations are conservative and stable.

**Division of labor:** Use NumPy/SciPy for machine-precision bulk computation (random matrix ensembles, spectral analysis, FFTs). Use mpmath when precision beyond float64 (~15 digits) is required (zeta evaluation, zero verification).

---

### Core Framework: Symbolic Mathematics

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **SymPy** | >=1.12, recommend ~1.13 | Symbolic computation, algebraic manipulation, analytic continuation formulas | SymPy handles symbolic differentiation, integration, series expansion, and algebraic manipulation needed for deriving and verifying formulas before numerical evaluation. Its `sympy.functions.special.zeta_functions` module provides symbolic representations. Critical for manipulating functional equations, Euler products, and modular form expressions symbolically before feeding them to mpmath for numerical evaluation. |

**Confidence:** HIGH -- SymPy is the standard Python CAS. No serious alternative for this use case (SageMath is an option but brings enormous dependency weight; see Alternatives Considered).

---

### Core Framework: Visualization

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **Matplotlib** | >=3.8 | Static 2D plots, publication-quality figures, density plots, contour plots | The bedrock plotting library. Best for static analysis: contour plots of |zeta(s)| in the critical strip, zero distribution histograms, heatmaps. Mature, well-documented, and handles complex-valued function visualization well. Use for any plot that will be saved or studied carefully. |
| **Plotly** | >=5.18 | Interactive 3D visualization, higher-dimensional projections | Plotly is essential for the interactive exploration workflow. Its 3D scatter, surface, and mesh plots with rotation/zoom allow the user to explore projections of higher-dimensional structures intuitively. `plotly.graph_objects` for fine control; `plotly.express` for rapid exploration. Renders in Jupyter and standalone HTML. |
| **ipywidgets** | >=8.1 | Interactive parameter controls in Jupyter | Sliders, dropdowns, and toggles for interactively adjusting visualization parameters (precision, dimension, projection angle, number of zeros). Pairs with both Matplotlib and Plotly for real-time "what if" exploration. |

**Confidence:** HIGH for Matplotlib/Plotly. These are mature, dominant choices. ipywidgets is the standard for Jupyter interactivity.

**Why not other visualization libraries:**
- **Mayavi/VTK**: Overkill for this use case, harder to install, poor Jupyter integration
- **PyVista**: Good for 3D meshes but less suited to mathematical function visualization
- **Bokeh**: Competes with Plotly but has weaker 3D support
- **Manim**: Beautiful for animations/presentations but not for interactive exploration

---

### Core Framework: Formal Verification

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **Lean 4** | >=4.x (latest stable via elan) | Theorem prover and programming language | Lean 4 is the modern choice for formal mathematics. It is actively developed, has a growing community, and its dependent type theory is expressive enough for deep mathematical formalization. The tactic language is programmable and extensible. Install via `elan` (the Lean version manager). |
| **Mathlib4** | latest (pinned via lake) | Mathematical library for Lean 4 | Mathlib is the comprehensive, community-maintained mathematical library with 100,000+ theorems. It includes formalized number theory, complex analysis, measure theory, topology, and algebra -- all foundational for any approach to RH. Without Mathlib, you'd be formalizing basic analysis from scratch. Use `lake` (Lean's build tool) to manage the Mathlib dependency. |

**Confidence:** HIGH for Lean 4 as the choice; MEDIUM for specific version numbers (Lean 4 releases frequently via toolchain updates). Always install via `elan` to get the correct toolchain version that matches your Mathlib pin.

**Critical Lean 4 / Mathlib note:** Mathlib is large (~3GB compiled). Initial build takes 30-60 minutes. Use `lake exe cache get` to download pre-built oleans instead of compiling from source. The formalization side of this project should start small -- formalizing lemmas and definitions, not attempting to formalize the full proof pipeline from day one.

---

### Spectral Analysis & Signal Processing

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **SciPy (scipy.signal, scipy.fft)** | (same as above) | Spectral analysis, FFT, periodograms, window functions | Already in the stack. `scipy.signal.periodogram`, `scipy.signal.welch`, and `scipy.fft` cover the spectral analysis needs: detecting patterns in zero spacings, computing pair correlation functions, analyzing spectral statistics of operators. |
| **NumPy (numpy.fft)** | (same as above) | Fast Fourier transforms | Already in the stack. For basic FFT needs; SciPy's `scipy.fft` is preferred for advanced use (better backend support, more transforms). |

**Confidence:** HIGH -- SciPy is the standard for signal processing in Python.

**No additional spectral library needed.** SciPy covers spectral density estimation, windowed FFTs, short-time Fourier transforms, and cross-spectral analysis. For random matrix spectral statistics specifically, custom code on top of NumPy/SciPy is the standard approach (see Architecture).

---

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **JupyterLab** | >=4.0 | Interactive notebook environment | Always. The primary interface for exploration. Supports inline Matplotlib, Plotly, and ipywidgets. The "workbench" is built around Jupyter notebooks. |
| **numba** | >=0.59 | JIT compilation for numerical Python | When machine-precision loops become bottlenecks. Numba JIT-compiles NumPy-based Python to LLVM machine code. Use for tight loops over zero distributions, custom random matrix samplers, and Monte Carlo simulations. Does NOT work with mpmath (arbitrary precision is inherently not JIT-able). |
| **pandas** | >=2.1 | Tabular data management | For organizing computed results: tables of zeros, eigenvalue statistics, correlation measurements. Use DataFrames as the intermediate format between computation and visualization. |
| **h5py** or **zarr** | h5py >=3.10 / zarr >=2.16 | Large dataset storage | When precomputed zero tables or eigenvalue distributions exceed memory. HDF5 (via h5py) is the standard for large numerical datasets in scientific computing. Zarr is the modern alternative with better chunking and cloud-compatible storage. Prefer h5py for simplicity; use zarr if datasets grow very large. |
| **tqdm** | >=4.66 | Progress bars | For long-running computations (bulk zero evaluation, random matrix sampling). Small dependency, large quality-of-life improvement. |
| **pytest** | >=7.4 | Testing | For validating computational modules against known values (e.g., first 100 zeta zeros to 50 decimal places). |
| **python-flint** | >=0.5 | FLINT bindings for fast number theory | Optional. FLINT (Fast Library for Number Theory) provides very fast polynomial arithmetic, integer factoring, and number-theoretic functions at arbitrary precision. Useful for modular forms computations. Less mature Python bindings than gmpy2, so treat as optional acceleration. |

**Confidence:** HIGH for JupyterLab, numba, pandas, pytest, tqdm. MEDIUM for h5py/zarr (depends on data volume). LOW for python-flint (bindings maturity may have changed).

---

### Development Tools

| Tool | Version | Purpose | Why |
|------|---------|---------|-----|
| **uv** | latest | Python package and project management | uv is the modern replacement for pip + venv + pip-tools. It is 10-100x faster than pip, handles dependency resolution correctly, creates virtual environments, and supports lockfiles via `uv.lock`. Use `uv init`, `uv add`, and `uv sync` for all dependency management. |
| **elan** | latest | Lean 4 toolchain manager | The official way to install and manage Lean 4 versions. Similar to rustup for Rust. Ensures the correct Lean toolchain version matches your Mathlib dependency. |
| **VS Code** | latest | IDE | With the Lean 4 extension for interactive theorem proving (goal state display, tactic suggestions) and Python/Jupyter extensions for computation. The Lean 4 VS Code extension is effectively required for productive Lean development. |
| **Ruff** | latest | Python linter and formatter | Fast, comprehensive Python linter/formatter. Replaces flake8 + black + isort in a single tool. Use for consistent code quality across computational modules. |
| **Git** | latest | Version control | Track research progress, experiments, and formalization. Essential for a long-running research project where you need to revisit earlier approaches. |

**Confidence:** HIGH for all. uv has become the standard Python project tool; elan is the only way to manage Lean 4.

---

## What NOT to Use (and Why)

### SageMath -- Avoid as primary tool
**Why not:** SageMath is a massive (~8GB) all-in-one mathematical software system. While it includes mpmath, SymPy, NumPy, and much more, it creates an isolated environment that conflicts with modern Python packaging (uv, virtual environments). Its notebook interface is inferior to JupyterLab. The individual libraries (mpmath, SymPy, NumPy/SciPy) are better used directly -- you get the same mathematical functionality without the dependency bloat, version conflicts, and installation headaches.

**When SageMath would be appropriate:** If you needed its unique capabilities (algebraic geometry via Singular, combinatorics via GAP, number theory via PARI/GP) as primary tools. For this project, mpmath + SymPy covers the needed symbolic/numeric math, and PARI/GP can be called directly via `cypari2` if needed.

### Mathematica / Maple -- Avoid
**Why not:** Proprietary, expensive, cannot be integrated into a Lean 4 formalization pipeline, and Python's ecosystem has caught up for computational math. mpmath matches or exceeds Mathematica's arbitrary-precision special function library for the specific functions relevant to RH.

### Coq / Isabelle -- Avoid for formalization
**Why not:** Lean 4 with Mathlib is the clear modern choice for formalizing mathematics. Coq has a steeper learning curve and a smaller active math formalization community. Isabelle/HOL is powerful but uses a different logical foundation (HOL vs dependent type theory) that is less natural for the kind of mathematics involved in RH. The Lean 4 community (and specifically Mathlib) is where the momentum is in 2025-2026.

### TensorFlow / PyTorch -- Avoid
**Why not:** These are machine learning frameworks, not mathematical computing tools. While they offer GPU-accelerated tensor operations, they operate at machine precision only (float32/float64), have no arbitrary-precision support, and their autodiff is designed for neural network training, not mathematical analysis. NumPy/SciPy cover the needed tensor operations. If GPU acceleration becomes critical, use CuPy (NumPy-compatible GPU arrays) rather than ML frameworks.

### Arb (arblib) directly -- Avoid direct C bindings
**Why not:** Arb is a C library for arbitrary-precision ball arithmetic (rigorous error bounds). It is excellent but the Python bindings are immature. mpmath provides similar functionality with much better Python integration. If you need rigorous interval arithmetic specifically, consider `python-flint` (which wraps Arb) rather than raw C bindings.

### Jupyter Notebook (classic) -- Avoid
**Why not:** JupyterLab is the successor. Classic Notebook is in maintenance mode. JupyterLab has better file management, multiple tab support, and extension ecosystem. Always use JupyterLab.

### F2 / Proof General (Emacs) for Lean -- Avoid
**Why not:** VS Code with the Lean 4 extension is the standard, best-supported development environment for Lean 4. Emacs support exists but is less mature and less documented.

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not Alternative |
|----------|-------------|-------------|---------------------|
| Arbitrary precision | mpmath + gmpy2 | SageMath PARI/GP | Dependency bloat; mpmath sufficient for zeta-specific computation |
| Symbolic math | SymPy | SageMath | Isolated environment; SymPy integrates cleanly with the rest of the stack |
| Visualization (2D) | Matplotlib | Bokeh | Bokeh has weaker publication-quality output and is less standard in math/science |
| Visualization (3D) | Plotly | Mayavi | Mayavi installation is fragile; poor Jupyter integration |
| Interactive notebooks | JupyterLab | VS Code notebooks | JupyterLab has richer widget/interactive support; VS Code notebooks are catching up but not there yet for this use case |
| Formal verification | Lean 4 + Mathlib | Coq + MathComp | Smaller community for pure math formalization; less momentum |
| Formal verification | Lean 4 + Mathlib | Isabelle/HOL | Different logical foundation (HOL); less natural for dependent-type-heavy math |
| Package management | uv | pip + venv | uv is faster, handles resolution better, and supports lockfiles natively |
| GPU acceleration | CuPy (if needed) | PyTorch tensors | CuPy is NumPy-compatible; PyTorch adds ML baggage and has no arbitrary-precision support |
| JIT compilation | Numba | Cython | Numba requires no separate compilation step; decorators are simpler for research code |

---

## Installation

**Note:** Validate version numbers with `pip index versions <pkg>` before running. Versions below are based on training data through May 2025 and may have newer releases.

### Python environment setup (using uv)

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Initialize project
cd Riemann
uv init --python 3.12

# Core computational libraries
uv add mpmath gmpy2 numpy scipy sympy

# Visualization
uv add matplotlib plotly ipywidgets

# Jupyter
uv add jupyterlab

# Data management and utilities
uv add pandas h5py tqdm

# Development tools
uv add --dev pytest ruff numba

# Optional: FLINT bindings for fast number theory
# uv add python-flint
```

### Lean 4 environment setup

```bash
# Install elan (Lean version manager)
curl https://elan-init.github.io/elan/elan-init.sh -sSf | sh

# Create a Lean 4 project with Mathlib dependency
# (Do this in a subdirectory of the project, e.g., Riemann/lean/)
mkdir -p lean && cd lean
lake init RiemannLean math

# Download pre-built Mathlib cache (MUCH faster than compiling)
lake exe cache get

# Build
lake build
```

### VS Code Extensions

```
# Install these extensions:
# - Python (Microsoft)
# - Jupyter (Microsoft)
# - Lean 4 (leanprover)
# - Ruff (Astral Software)
```

---

## Stack Interaction Map

```
                    EXPLORATION LAYER (Python)
    +--------------------------------------------------+
    |  JupyterLab (interactive workbench)               |
    |  +----------------------------------------------+ |
    |  | ipywidgets (interactive controls)             | |
    |  +----------------------------------------------+ |
    |  | Matplotlib (2D)  |  Plotly (3D interactive)   | |
    |  +----------------------------------------------+ |
    |                                                    |
    |  COMPUTATION LAYER                                 |
    |  +----------------------------------------------+ |
    |  | mpmath + gmpy2        | NumPy + SciPy         | |
    |  | (arbitrary precision)  | (machine precision)   | |
    |  | - zeta evaluation     | - random matrices     | |
    |  | - zero finding        | - spectral analysis   | |
    |  | - L-functions         | - FFT / eigenvalues   | |
    |  +----------------------------------------------+ |
    |  | SymPy (symbolic manipulation)                 | |
    |  | - formula derivation, algebraic simplification | |
    |  +----------------------------------------------+ |
    |  | pandas (result organization) | h5py (storage)  | |
    |  +----------------------------------------------+ |
    +--------------------------------------------------+
                          |
                    Export conjectures,
                    verified computations
                          |
                          v
                FORMALIZATION LAYER (Lean 4)
    +--------------------------------------------------+
    |  Lean 4 + Mathlib4                                |
    |  - Formalize definitions (zeta, L-functions)      |
    |  - Prove lemmas supporting the proof strategy      |
    |  - Machine-verified rigor for promising results    |
    +--------------------------------------------------+
```

---

## Version Validation Checklist

Before installing, run these commands to confirm latest versions:

```bash
uv pip index versions mpmath
uv pip index versions gmpy2
uv pip index versions numpy
uv pip index versions scipy
uv pip index versions sympy
uv pip index versions matplotlib
uv pip index versions plotly
uv pip index versions jupyterlab
elan show  # for Lean 4 toolchain version
```

All version numbers in this document were sourced from training data (cutoff: May 2025). The libraries are mature and unlikely to have breaking API changes, but patch/minor versions will have advanced.

---

## Sources

- **mpmath:** Training data knowledge of mpmath through v1.3.0 (May 2025). mpmath.org is the official site. The library has been stable for years with infrequent releases.
- **NumPy:** Training data knowledge through NumPy 2.0.x. numpy.org is authoritative.
- **SciPy:** Training data knowledge through SciPy 1.12-1.13. scipy.org is authoritative.
- **SymPy:** Training data knowledge through SymPy 1.12-1.13. sympy.org is authoritative.
- **Matplotlib:** Training data knowledge through Matplotlib 3.8-3.9. matplotlib.org is authoritative.
- **Plotly:** Training data knowledge through Plotly 5.18-5.22. plotly.com/python is authoritative.
- **Lean 4 / Mathlib:** Training data knowledge through Lean 4 toolchains circa early 2025. leanprover.github.io and leanprover-community.github.io are authoritative.
- **gmpy2:** Training data knowledge through gmpy2 2.1.x. PyPI is authoritative.
- **uv:** Training data knowledge through uv 0.4.x (fast-moving project; likely newer). astral.sh/uv is authoritative.

**Confidence note:** All web-based verification tools (WebSearch, WebFetch, Bash) were unavailable during this research session. All version numbers and library assessments are based on training data with a May 2025 cutoff. The core recommendations (which libraries to use and why) are HIGH confidence because these are mature, well-established tools with stable ecosystems. Specific version numbers are MEDIUM confidence and should be validated before installation.
