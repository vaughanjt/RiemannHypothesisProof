# Phase 2: Higher-Dimensional Analysis - Research

**Researched:** 2026-03-19
**Domain:** N-dimensional embedding, projection methods, zero distribution statistics, random matrix theory, information theory, anomaly detection
**Confidence:** HIGH (core libraries are mature and well-documented; custom mathematical algorithms are domain-standard)

## Summary

Phase 2 is the platform's core differentiator: seeing structure in mathematical objects beyond human spatial intuition. It covers nine requirements spanning five distinct technical domains: (1) zero distribution statistics (spacing, pair correlation, n-level density), (2) N-dimensional embedding with configurable coordinate mappings, (3) projection methods (PCA, t-SNE, UMAP, stereographic, custom), (4) random matrix theory (GUE ensemble generation and comparison), and (5) information-theoretic analysis (entropy, mutual information, compression complexity). An anomaly detection layer (SPC-based, >3-sigma from GUE) ties it all together, auto-logging observations to the workbench.

The existing Phase 1 codebase provides a solid foundation: `ZetaZero` dataclass, `ZeroCatalog` with SQLite bulk retrieval, `validated_computation` for precision-critical paths, `save_experiment` / `create_conjecture` for workbench integration, and established patterns (function-based API, dual-mode computation, strict separation of computation and visualization). Phase 2 builds new modules atop this substrate without modifying Phase 1 code.

**Primary recommendation:** Build bottom-up: zero statistics engine first (data source for everything), then embedding/projection pipeline, then RMT comparison, then information theory, then anomaly detection. Use scikit-learn for PCA/t-SNE, umap-learn for UMAP, numpy/scipy for all statistical computation. Do NOT pull in scikit-rmt -- custom GUE generation with numpy is simpler and avoids a fragile dependency. Store large embedding arrays in HDF5 via h5py (already a project dependency).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Multiple embedding schemes -- maximize diversity of views to avoid bias toward expected outcomes
- Embedding coordinates for zeros: imaginary part, spacing to neighbors (left/right), derivative |zeta'(rho)|, local zero density deviation, pair correlation at multiple scales, Hardy Z sign changes
- Additional embeddings: zeros as points in spectral space (eigenvalue-gap analogy), zeros in information-theoretic space (local entropy, compression distance)
- User directive: "Surprise me" -- cast the widest net, don't bias toward any particular expected structure
- Each embedding is a named, reproducible configuration stored in the workbench
- Implement all projection methods: PCA, t-SNE, UMAP, stereographic, custom mathematical projections (e.g., Hopf fibration for S^3 data)
- Side-by-side comparison: same data, multiple projections, linked highlighting
- Progressive: start with PCA (linear, fast, trustworthy), then nonlinear methods
- Interactive 3D visualization using Plotly with rotation, zoom, parameter controls
- "Projection path" animations -- smoothly interpolate between projection methods
- Dimension slicing: fix some dimensions, project remaining
- Speed over polish: computation speed matters more than frame rate or rendering quality
- Nearest-neighbor spacing (normalized by mean spacing)
- Pair correlation function r_2(x) compared against GUE sine kernel
- n-level density for n=2,3,4
- Number variance and Sigma_2 statistic
- All statistics computed with configurable zero ranges and overlap with Odlyzko-verified zeros
- Generate GUE ensembles at configurable matrix sizes (N=10 to N=1000+)
- Compute eigenvalue statistics matching every zero statistic above
- Linked views: zero statistics and RMT statistics side-by-side, interactive N slider
- Residual analysis: where do zeros deviate from GUE? At what scale?
- Shannon entropy of zero spacing sequences (binned and kernel-density estimated)
- Mutual information between consecutive spacings at multiple lags
- Lempel-Ziv complexity / compression-based distance metrics on zero sequences
- Compare information signatures: zeros vs GUE eigenvalues vs Poisson random points vs primes
- Statistical process control (SPC) applied to zero data streams
- Flag any window where local statistics deviate >3-sigma from GUE prediction
- Anomaly severity levels: info / warning / critical (integrate with evidence hierarchy)
- Every anomaly auto-logged as an "observation" in the research workbench (evidence level 0)
- JupyterLab, Claude-driven exploration
- Speed over visual polish
- SQLite + numpy + HDF5 for data storage
- Strict evidence hierarchy (observation / heuristic / conditional / formal)
- 50-digit default, always-validate precision
- Phase 1 computation engine, visualization layer, and workbench are the integration substrate

### Claude's Discretion
- Exact embedding coordinate engineering and feature scaling
- Projection hyperparameters (t-SNE perplexity, UMAP n_neighbors, etc.)
- Statistical test selection and significance thresholds
- Matrix ensemble sampling strategy
- Information-theoretic estimator choices (binning vs KDE vs k-NN)
- Anomaly detection window sizes and threshold tuning
- Notebook organization for Phase 2 explorations
- HDF5 schema for high-dimensional data
- Performance optimization (vectorization, caching, lazy computation)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| ZERO-01 | User can compute zero distribution statistics (nearest-neighbor spacing, pair correlation, n-level density) and compare against GUE predictions | Zero statistics engine (Section: Architecture Pattern 1), GUE sine kernel comparison (Pattern 4), all formulas documented in Code Examples |
| ZERO-02 | User can detect anomalies in zero structure -- deviations from expected behavior are automatically flagged | SPC anomaly detection module (Pattern 6), >3-sigma threshold from GUE, auto-logging to workbench |
| HDIM-01 | User can represent mathematical objects in N-dimensional spaces with configurable coordinate mappings | Embedding registry pattern (Pattern 2), named/reproducible configs, HDF5 storage |
| HDIM-02 | User can apply multiple projection methods and compare results side-by-side | Projection pipeline (Pattern 3), scikit-learn PCA/t-SNE, umap-learn UMAP, custom stereographic |
| VIZ-03 | User can interactively rotate through higher-dimensional spaces, watching how structures project into different 2D/3D views | Plotly 3D scatter with frames/animation, projection path interpolation, dimension slicing |
| RMT-01 | User can generate GUE/GOE/GSE random matrix ensembles, compute eigenvalue statistics, and overlay with zeta zero statistics in interactive linked views | Custom numpy GUE generation (Pattern 4), eigenvalue statistics matching zero statistics, Plotly linked views |
| RMT-02 | User can vary matrix size and ensemble type and observe how the fit to zero statistics changes | Interactive N slider with ipywidgets, ensemble type selector, linked recomputation |
| INFO-01 | User can apply information-theoretic measures to zero sequences and related data | Shannon entropy via scipy.stats.entropy, MI via k-NN estimator, LZ complexity (Pattern 5) |
| INFO-02 | User can compare information-theoretic signatures across different mathematical objects | Cross-object comparison framework: zeros vs GUE eigenvalues vs Poisson vs primes (Pattern 5) |
</phase_requirements>

## Standard Stack

### Core (New for Phase 2)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scikit-learn | >=1.8.0 | PCA, t-SNE, StandardScaler, preprocessing | The standard for dimensionality reduction in Python; consistent fit/transform API; PCA and TSNE classes are production-quality |
| umap-learn | >=0.5.11 | UMAP projection | The canonical UMAP implementation; sklearn-compatible API; preserves global topology better than t-SNE |
| numba | >=0.59 | JIT compilation for custom statistics loops | Required dependency of umap-learn; also useful for vectorizing pair correlation and spacing computation loops |

### Existing (from Phase 1, used extensively)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | >=2.4.3 | All numerical arrays, eigenvalue computation, random matrix generation | Every computation; `np.linalg.eigvalsh` for Hermitian eigenvalues |
| scipy | >=1.17.1 | `scipy.stats.entropy`, `scipy.spatial.distance`, KDE, statistical tests | Information theory measures, statistical testing, kernel density estimation |
| h5py | >=3.16.0 | HDF5 storage for large embedding arrays | Any embedding array exceeding ~100MB; already a project dependency |
| plotly | >=6.6.0 | Interactive 3D scatter, animation frames, linked views | Projection theater, RMT comparison views, all interactive viz |
| ipywidgets | >=8.1.8 | Sliders, dropdowns for matrix size N, projection params | RMT N slider, projection hyperparameter controls |
| matplotlib | >=3.10.8 | Static 2D plots for statistics (histograms, correlation functions) | Publication-quality spacing distributions, pair correlation plots |
| mpmath | >=1.3.0 | Arbitrary-precision zeta derivatives for embedding coordinates | Computing |zeta'(rho)| at zeros, precision-critical embedding features |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom GUE generation | scikit-rmt | scikit-rmt adds a dependency for something that is 5 lines of numpy; our GUE needs are specific (eigenvalue statistics, not distribution fitting) |
| scipy.stats.entropy | pyitlib | pyitlib has richer info-theory tools but is less maintained; scipy.stats.entropy + custom MI estimator covers our needs |
| Custom LZ complexity | lempel-ziv-complexity package | The package is tiny and unmaintained; 20 lines of Python with numba JIT is more reliable |
| Panel dashboards | ipywidgets alone | Panel was considered in architecture research but ipywidgets + Plotly is sufficient for linked views; avoids another dependency to validate |

**Installation (new Phase 2 dependencies):**
```bash
uv add scikit-learn umap-learn
```

Note: numba is already required by umap-learn. h5py, plotly, ipywidgets, scipy are already in pyproject.toml.

## Architecture Patterns

### Recommended Project Structure (Phase 2 additions)
```
src/riemann/
  analysis/                  # NEW -- Layer 2: Analysis Modules
    __init__.py
    spacing.py               # ZERO-01: spacing statistics, pair correlation, n-level density
    rmt.py                   # RMT-01/02: GUE generation, eigenvalue stats, comparison
    information.py           # INFO-01/02: entropy, MI, LZ complexity, cross-comparison
    anomaly.py               # ZERO-02: SPC anomaly detection, auto-logging

  embedding/                 # NEW -- HDIM-01: N-dimensional embedding
    __init__.py
    registry.py              # Named embedding configurations, storage
    coordinates.py           # Feature extraction from zeros (spacing, derivatives, etc.)
    storage.py               # HDF5 read/write for large embedding arrays

  viz/
    projection.py            # NEW -- HDIM-02: projection pipeline (PCA, t-SNE, UMAP, stereographic)
    theater.py               # NEW -- VIZ-03: interactive projection theater (Plotly 3D + animation)
    comparison.py            # NEW -- RMT-01: side-by-side linked views
    styles.py                # EXISTING -- extend with multi-dim color scales

tests/
  test_analysis/
    test_spacing.py
    test_rmt.py
    test_information.py
    test_anomaly.py
  test_embedding/
    test_coordinates.py
    test_registry.py
    test_storage.py
  test_viz/
    test_projection.py
    test_theater.py
```

### Pattern 1: Zero Statistics Engine (spacing.py)

**What:** A module that takes a list of ZetaZero objects (from ZeroCatalog) and computes all distribution statistics: normalized spacings, pair correlation r_2(x), n-level density, number variance.

**When to use:** Any time zero distribution properties are needed -- for direct analysis, GUE comparison, or anomaly detection input.

**Key algorithms:**

```python
# Nearest-neighbor spacing (normalized by mean spacing)
def normalized_spacings(zeros: list[ZetaZero]) -> np.ndarray:
    """Compute nearest-neighbor spacings normalized by local mean spacing.

    The mean spacing at height T is approximately 2*pi / log(T/(2*pi)).
    Normalizing removes the secular variation so spacings can be compared
    across different height ranges.
    """
    t_values = np.array([float(z.value.imag) for z in zeros])
    t_values.sort()
    raw_spacings = np.diff(t_values)
    # Local mean spacing: 2*pi / log(t/(2*pi))
    midpoints = (t_values[:-1] + t_values[1:]) / 2
    mean_spacings = 2 * np.pi / np.log(midpoints / (2 * np.pi))
    return raw_spacings / mean_spacings

# Pair correlation function r_2(x)
def pair_correlation(spacings: np.ndarray, bins: int = 200,
                     x_range: tuple = (0.0, 4.0)) -> tuple[np.ndarray, np.ndarray]:
    """Compute the pair correlation function from normalized spacings.

    Returns (x_centers, r2_values) where r2 should approach
    1 - (sin(pi*x)/(pi*x))^2 for GUE zeros.
    """
    # Histogram of all pairwise normalized gaps
    ...

# GUE sine kernel prediction for comparison
def gue_pair_correlation(x: np.ndarray) -> np.ndarray:
    """GUE pair correlation: 1 - (sin(pi*x)/(pi*x))^2."""
    result = np.ones_like(x)
    nonzero = x != 0
    sinc = np.sin(np.pi * x[nonzero]) / (np.pi * x[nonzero])
    result[nonzero] = 1 - sinc**2
    return result
```

**Boundary rule:** This module returns numpy arrays, never plots. Visualization is separate.

### Pattern 2: Embedding Registry (registry.py + coordinates.py)

**What:** Named, reproducible embedding configurations. Each embedding defines which features to extract from zeros and how to scale them. Configs stored in workbench as experiments.

**When to use:** Creating any N-dimensional representation of mathematical objects.

**Example:**

```python
@dataclass(frozen=True)
class EmbeddingConfig:
    """A named, reproducible embedding configuration."""
    name: str
    description: str
    feature_names: tuple[str, ...]   # Ordered list of feature extractors
    scaling: str = "standard"         # "standard", "robust", "none"
    zero_range: tuple[int, int] = (1, 1000)  # Zero index range
    dps: int = 50                     # Precision for feature extraction

# Feature extraction functions (coordinates.py)
FEATURE_EXTRACTORS = {
    "imag_part": extract_imaginary_part,
    "spacing_left": extract_left_spacing,
    "spacing_right": extract_right_spacing,
    "zeta_derivative_magnitude": extract_zeta_derivative_mag,
    "local_density_deviation": extract_local_density_deviation,
    "pair_correlation_local": extract_pair_correlation_local,
    "hardy_z_sign_changes": extract_hardy_z_sign_changes,
    "local_entropy": extract_local_entropy,
    "compression_distance": extract_compression_distance,
}

def compute_embedding(config: EmbeddingConfig,
                      zeros: list[ZetaZero]) -> np.ndarray:
    """Compute N-dimensional embedding from zeros using config.

    Returns array of shape (n_zeros, n_features).
    """
    features = []
    for fname in config.feature_names:
        extractor = FEATURE_EXTRACTORS[fname]
        features.append(extractor(zeros, dps=config.dps))

    embedding = np.column_stack(features)

    if config.scaling == "standard":
        from sklearn.preprocessing import StandardScaler
        embedding = StandardScaler().fit_transform(embedding)
    elif config.scaling == "robust":
        from sklearn.preprocessing import RobustScaler
        embedding = RobustScaler().fit_transform(embedding)

    return embedding
```

**Storage:** Save embedding arrays to HDF5 keyed by config hash. Save config as experiment in workbench.

### Pattern 3: Projection Pipeline (projection.py)

**What:** Takes an N-dimensional embedding array and projects it to 2D or 3D using a specified method. Returns projected coordinates plus metadata (variance explained, trustworthiness, etc.).

**When to use:** Every time an embedding needs to be visualized.

**Example:**

```python
@dataclass
class ProjectionResult:
    """Result of projecting N-dim data to lower dimensions."""
    coordinates: np.ndarray         # (n_points, target_dim)
    method: str
    source_dim: int
    target_dim: int
    metadata: dict                  # Method-specific: variance_explained, stress, etc.

def project_pca(data: np.ndarray, n_components: int = 3) -> ProjectionResult:
    """PCA projection -- linear, fast, preserves variance."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(data)
    return ProjectionResult(
        coordinates=coords,
        method="PCA",
        source_dim=data.shape[1],
        target_dim=n_components,
        metadata={
            "variance_explained": pca.explained_variance_ratio_.tolist(),
            "total_variance_explained": float(sum(pca.explained_variance_ratio_)),
            "components": pca.components_.tolist(),
        },
    )

def project_tsne(data: np.ndarray, n_components: int = 3,
                 perplexity: float = 30.0) -> ProjectionResult:
    """t-SNE projection -- nonlinear, preserves local neighborhoods."""
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity,
                random_state=42)
    coords = tsne.fit_transform(data)
    return ProjectionResult(
        coordinates=coords,
        method="t-SNE",
        source_dim=data.shape[1],
        target_dim=n_components,
        metadata={
            "perplexity": perplexity,
            "kl_divergence": float(tsne.kl_divergence_),
        },
    )

def project_umap(data: np.ndarray, n_components: int = 3,
                 n_neighbors: int = 15) -> ProjectionResult:
    """UMAP projection -- preserves global topology better than t-SNE."""
    import umap
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                        random_state=42)
    coords = reducer.fit_transform(data)
    return ProjectionResult(
        coordinates=coords,
        method="UMAP",
        source_dim=data.shape[1],
        target_dim=n_components,
        metadata={"n_neighbors": n_neighbors, "min_dist": reducer.min_dist},
    )

def project_stereographic(data: np.ndarray) -> ProjectionResult:
    """Stereographic projection from S^n to R^n (for data on unit sphere)."""
    # Project from north pole: x_i / (1 - x_n) for i < n
    norm = np.linalg.norm(data, axis=1, keepdims=True)
    normalized = data / norm  # Project onto sphere first
    n = data.shape[1]
    denom = 1.0 - normalized[:, -1:]  # 1 - x_n
    denom = np.clip(denom, 1e-10, None)  # Avoid division by zero
    coords = normalized[:, :-1] / denom
    return ProjectionResult(
        coordinates=coords,
        method="stereographic",
        source_dim=n,
        target_dim=n - 1,
        metadata={"projection_pole": "north"},
    )
```

**Anti-pattern to avoid:** Never compute embeddings inside projection functions. Embedding and projection are separate pipeline stages.

### Pattern 4: RMT Comparison Engine (rmt.py)

**What:** Generate GUE/GOE/GSE ensembles, compute eigenvalue statistics matching the zero statistics API, enable direct comparison.

**When to use:** Any RMT comparison (ZERO-01 comparison, RMT-01/02 linked views, anomaly detection baselines).

**Example:**

```python
def generate_gue(n: int, num_matrices: int = 100,
                 seed: int | None = None) -> list[np.ndarray]:
    """Generate GUE ensemble: Hermitian matrices with Gaussian entries.

    GUE(N): H = (A + A*) / sqrt(2N) where A has i.i.d. complex Gaussian entries.

    Args:
        n: Matrix dimension.
        num_matrices: Number of matrices to sample.
        seed: Random seed for reproducibility.

    Returns:
        List of eigenvalue arrays, each of shape (n,).
    """
    rng = np.random.default_rng(seed)
    eigenvalues_list = []
    for _ in range(num_matrices):
        # Complex Gaussian matrix
        A = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2)
        # Hermitianize
        H = (A + A.conj().T) / (2 * np.sqrt(n))
        # eigvalsh is O(n^3) but returns sorted real eigenvalues
        eigenvalues_list.append(np.linalg.eigvalsh(H))
    return eigenvalues_list

def gue_eigenvalue_spacings(eigenvalues_list: list[np.ndarray]) -> np.ndarray:
    """Compute normalized nearest-neighbor spacings from GUE eigenvalues.

    Unfolding: use the semicircle law to normalize eigenvalue density.
    """
    all_spacings = []
    for eigs in eigenvalues_list:
        # Unfold using Wigner semicircle CDF
        n = len(eigs)
        unfolded = _unfold_semicircle(eigs, n)
        spacings = np.diff(unfolded)
        mean_spacing = np.mean(spacings)
        if mean_spacing > 0:
            all_spacings.extend(spacings / mean_spacing)
    return np.array(all_spacings)

def wigner_surmise(s: np.ndarray, beta: int = 2) -> np.ndarray:
    """Wigner surmise for nearest-neighbor spacing distribution.

    beta=1 (GOE), beta=2 (GUE), beta=4 (GSE).
    For GUE: p(s) = (32/pi^2) * s^2 * exp(-4s^2/pi)
    """
    if beta == 2:
        return (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)
    elif beta == 1:
        return (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4)
    elif beta == 4:
        c = 2**18 / (3**6 * np.pi**3)
        return c * s**4 * np.exp(-64 * s**2 / (9 * np.pi))
    raise ValueError(f"beta must be 1, 2, or 4, got {beta}")
```

**Performance note:** GUE generation at N=1000 with `eigvalsh` takes ~0.5s per matrix (LAPACK). 100 matrices = ~50s. For interactive N slider, precompute at key sizes and interpolate, or cap interactive range at N=200 and batch larger sizes.

### Pattern 5: Information-Theoretic Analysis (information.py)

**What:** Shannon entropy, mutual information, and compression complexity applied to zero spacing sequences and compared across mathematical objects.

**When to use:** INFO-01/02 requirements, and as embedding coordinates for information-theoretic space.

**Example:**

```python
def spacing_entropy(spacings: np.ndarray, method: str = "kde",
                    bins: int = 50) -> float:
    """Shannon entropy of a spacing distribution.

    Args:
        spacings: Normalized spacing values.
        method: "binned" for histogram, "kde" for kernel density estimate.
        bins: Number of bins (for binned method) or evaluation points (for KDE).
    """
    if method == "binned":
        counts, _ = np.histogram(spacings, bins=bins, density=False)
        counts = counts[counts > 0]  # Remove zeros
        probs = counts / counts.sum()
        return float(scipy.stats.entropy(probs))
    elif method == "kde":
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(spacings)
        x = np.linspace(spacings.min(), spacings.max(), bins)
        density = kde(x)
        density = density[density > 0]
        dx = x[1] - x[0]
        return float(-np.sum(density * np.log(density) * dx))

def mutual_information_spacings(spacings: np.ndarray,
                                 lag: int = 1,
                                 k: int = 5) -> float:
    """Mutual information between spacing s_i and s_{i+lag}.

    Uses k-nearest neighbors estimator (Kraskov et al. 2004).
    """
    from sklearn.feature_selection import mutual_info_regression
    x = spacings[:-lag].reshape(-1, 1)
    y = spacings[lag:]
    # mutual_info_regression uses k-NN estimator internally
    mi = mutual_info_regression(x, y, n_neighbors=k, random_state=42)
    return float(mi[0])

def lempel_ziv_complexity(sequence: np.ndarray, threshold: float = None) -> int:
    """Lempel-Ziv complexity of a binarized sequence.

    Binarizes the sequence by thresholding at the median (or given threshold),
    then counts distinct subsequences in the LZ76 sense.
    """
    if threshold is None:
        threshold = np.median(sequence)
    binary = ''.join('1' if x > threshold else '0' for x in sequence)
    # LZ76 algorithm
    n = len(binary)
    complexity = 1
    i = 0
    while i < n:
        l = 1
        found = True
        while found and i + l <= n:
            substr = binary[i:i+l]
            if substr in binary[:i+l-1]:
                l += 1
            else:
                found = False
        complexity += 1
        i += l
    return complexity

def cross_object_comparison(
    zero_spacings: np.ndarray,
    gue_spacings: np.ndarray,
    poisson_spacings: np.ndarray | None = None,
    prime_gaps: np.ndarray | None = None,
) -> dict:
    """Compare information-theoretic signatures across object types.

    Returns dict of {object_name: {metric_name: value}}.
    """
    if poisson_spacings is None:
        rng = np.random.default_rng(42)
        poisson_spacings = rng.exponential(1.0, size=len(zero_spacings))

    objects = {
        "zeta_zeros": zero_spacings,
        "gue_eigenvalues": gue_spacings,
        "poisson": poisson_spacings,
    }
    if prime_gaps is not None:
        objects["primes"] = prime_gaps

    results = {}
    for name, spacings in objects.items():
        results[name] = {
            "entropy_binned": spacing_entropy(spacings, method="binned"),
            "entropy_kde": spacing_entropy(spacings, method="kde"),
            "mi_lag1": mutual_information_spacings(spacings, lag=1),
            "mi_lag2": mutual_information_spacings(spacings, lag=2),
            "lz_complexity": lempel_ziv_complexity(spacings),
        }
    return results
```

### Pattern 6: SPC Anomaly Detection (anomaly.py)

**What:** Sliding-window statistical process control over zero data streams. Flags deviations >3-sigma from GUE predictions. Auto-logs to workbench.

**When to use:** After computing zero statistics, as an automated scan for interesting regions.

**Example:**

```python
@dataclass
class Anomaly:
    """A detected anomaly in zero structure."""
    zero_range: tuple[int, int]     # Index range of anomalous window
    statistic: str                   # Which statistic deviated
    observed_value: float
    expected_value: float
    sigma_deviation: float
    severity: str                    # "info" | "warning" | "critical"
    description: str

def detect_anomalies(
    zeros: list[ZetaZero],
    window_size: int = 50,
    stride: int = 25,
    sigma_thresholds: dict | None = None,
) -> list[Anomaly]:
    """Scan zero windows for deviations from GUE predictions.

    Args:
        zeros: Sorted list of ZetaZero objects.
        window_size: Number of zeros per analysis window.
        stride: Step between windows.
        sigma_thresholds: {severity: sigma_value}. Default: info=2, warning=3, critical=4.
    """
    if sigma_thresholds is None:
        sigma_thresholds = {"info": 2.0, "warning": 3.0, "critical": 4.0}

    anomalies = []
    for start in range(0, len(zeros) - window_size, stride):
        window = zeros[start:start + window_size]
        # Compute local statistics
        spacings = normalized_spacings(window)
        mean_spacing = np.mean(spacings)
        var_spacing = np.var(spacings)

        # Expected GUE values (from theory)
        expected_mean = 1.0
        expected_var = 0.2728...  # GUE variance of normalized spacing

        # Check mean spacing deviation
        se_mean = np.sqrt(expected_var / len(spacings))
        z_score = abs(mean_spacing - expected_mean) / se_mean

        for severity, threshold in sorted(sigma_thresholds.items(),
                                           key=lambda x: x[1], reverse=True):
            if z_score > threshold:
                anomalies.append(Anomaly(
                    zero_range=(window[0].index, window[-1].index),
                    statistic="mean_spacing",
                    observed_value=float(mean_spacing),
                    expected_value=expected_mean,
                    sigma_deviation=float(z_score),
                    severity=severity,
                    description=f"Mean spacing {z_score:.1f}sigma from GUE expectation",
                ))
                break

    return anomalies

def log_anomalies_to_workbench(
    anomalies: list[Anomaly],
    db_path: str | None = None,
) -> list[str]:
    """Auto-log detected anomalies as observations in the workbench."""
    from riemann.workbench.conjecture import create_conjecture
    conjecture_ids = []
    for anomaly in anomalies:
        if anomaly.severity in ("warning", "critical"):
            cid = create_conjecture(
                statement=f"Anomalous {anomaly.statistic} in zeros "
                          f"{anomaly.zero_range[0]}-{anomaly.zero_range[1]}: "
                          f"{anomaly.sigma_deviation:.1f}sigma from GUE",
                description=anomaly.description,
                evidence_level=0,  # OBSERVATION
                status="speculative",
                tags=["anomaly", anomaly.severity, anomaly.statistic],
                db_path=db_path,
            )
            conjecture_ids.append(cid)
    return conjecture_ids
```

### Anti-Patterns to Avoid

- **Computing embeddings inside visualization callbacks:** Embedding is expensive (may require zeta derivative evaluation at arbitrary precision). Compute once, store in HDF5, project from stored data.
- **Using float64 for embedding feature extraction near the critical strip:** Features like |zeta'(rho)| must be computed with mpmath. Convert to float64 only after extraction, before storing in numpy arrays.
- **Global t-SNE/UMAP hyperparameters:** Different embedding dimensionalities and data scales need different perplexity/n_neighbors. Always expose these as parameters, never hardcode.
- **Treating t-SNE distances as meaningful:** t-SNE preserves local structure but distorts global distances. Never interpret inter-cluster distances in t-SNE plots. Always compare with PCA (which preserves variance) and UMAP (which preserves topology).
- **One-shot anomaly detection:** The >3-sigma threshold is a starting point. Anomalies must be confirmed at higher precision and larger context windows before escalating evidence level.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| PCA projection | Custom SVD-based PCA | `sklearn.decomposition.PCA` | Handles edge cases (constant features, n_samples < n_features), auto-selects solver |
| t-SNE embedding | Barnes-Hut t-SNE from scratch | `sklearn.manifold.TSNE` | Barnes-Hut approximation is complex; sklearn handles learning rate, early exaggeration, momentum |
| UMAP projection | Custom simplicial complex construction | `umap.UMAP` | UMAP theory involves Riemannian geometry and fuzzy simplicial sets -- months to implement correctly |
| KDE for entropy | Custom kernel density estimator | `scipy.stats.gaussian_kde` | Bandwidth selection, boundary effects, multivariate support are all handled |
| Hermitian eigenvalues | Custom QR iteration | `np.linalg.eigvalsh` | LAPACK's divide-and-conquer is optimized assembly; 10-100x faster than Python |
| Mutual information | Custom k-NN MI estimator | `sklearn.feature_selection.mutual_info_regression` | Kraskov estimator with bias correction is subtle to implement correctly |
| Feature scaling | Manual mean/std centering | `sklearn.preprocessing.StandardScaler` or `RobustScaler` | Edge cases (zero variance, outliers) handled correctly |

**Key insight:** Phase 2's mathematical novelty is in the CHOICE of embedding coordinates and the INTERPRETATION of projections, not in implementing standard algorithms. Use battle-tested implementations for projection and statistics; invest implementation effort in the domain-specific feature extractors and comparison framework.

## Common Pitfalls

### Pitfall 1: t-SNE / UMAP Artifacts Mistaken for Mathematical Structure
**What goes wrong:** Clusters appear in t-SNE/UMAP projections that don't correspond to genuine mathematical structure. The user investigates phantom patterns.
**Why it happens:** t-SNE and UMAP create local distortions by design. Different perplexity/n_neighbors values can produce wildly different apparent structures from the same data.
**How to avoid:** Always show PCA alongside nonlinear projections. If structure appears in PCA, it's real (linear subspace). If it appears only in t-SNE/UMAP, run at 3+ different hyperparameter settings. Record projection metadata with every visualization.
**Warning signs:** Structure that changes dramatically with perplexity. Clusters with no statistical differentiation when tested numerically.

### Pitfall 2: GUE Comparison at Wrong Scale
**What goes wrong:** Zero statistics are compared against GUE predictions at mismatched matrix sizes. GUE(N) approaches the universal limit (sine kernel) only as N approaches infinity; at finite N, there are significant deviations.
**Why it happens:** The universal limit formulas (Wigner surmise, sine kernel) are only asymptotically correct. For finite samples of zeros (e.g., zeros 1-500), the comparison should be against GUE(N) at a matched effective N, not the N=infinity limit.
**How to avoid:** Always compute GUE statistics at multiple matrix sizes (N=10, 50, 100, 500, 1000). Show how the fit changes with N. The residual analysis is specifically about this: at what effective N do zeros best match GUE?
**Warning signs:** Systematic bias in the comparison that shrinks as N increases.

### Pitfall 3: Unfolding Errors in Spacing Statistics
**What goes wrong:** Raw spacings between zeros are used without normalizing by the local mean spacing, or the wrong mean-spacing formula is used.
**Why it happens:** The density of zeta zeros increases logarithmically with height. Without unfolding (dividing by local mean spacing), spacings systematically decrease, destroying the universality.
**How to avoid:** Always normalize spacings using the known average density: the mean spacing near height T is 2*pi / log(T/(2*pi)). Verify the unfolding by checking that the mean of normalized spacings is approximately 1.0.
**Warning signs:** Mean normalized spacing deviating from 1.0 by more than 1/sqrt(N) where N is the number of spacings.

### Pitfall 4: HDF5 File Locking on Windows
**What goes wrong:** HDF5 files opened for writing in one process cannot be read by another, or vice versa. On Windows, file locks are enforced by the OS.
**Why it happens:** HDF5's default file locking is stricter on Windows than Linux. Multiple Jupyter kernels or processes competing for the same HDF5 file will deadlock.
**How to avoid:** Use a single-writer pattern: compute embedding, write to HDF5, close file, then open read-only for visualization. Consider per-embedding-config HDF5 files rather than one monolithic file. Set `HDF5_USE_FILE_LOCKING=FALSE` environment variable if single-user access is guaranteed.
**Warning signs:** `OSError: Unable to open file` or `IOError: File is locked`.

### Pitfall 5: Performance Death Spiral with Many Zeros
**What goes wrong:** Computing embedding features for 10,000+ zeros becomes prohibitively slow because some features (|zeta'(rho)|) require arbitrary-precision evaluation at each zero.
**Why it happens:** Each zeta derivative evaluation at 50-digit precision takes 10-100ms. For 10,000 zeros, that's 100-1000 seconds just for one embedding coordinate.
**How to avoid:** Cache all derived zero properties in the ZeroCatalog (add columns for derivative magnitude, local density, etc.). Compute expensive features once, store forever. For interactive exploration, start with cheap features (spacing, imaginary part) and add expensive features on demand.
**Warning signs:** Embedding computation taking >30 seconds for interactive use.

## Code Examples

### Computing Zero Statistics End-to-End

```python
# Source: mathematical standard (Odlyzko 1987, Montgomery 1973)
from riemann.engine.zeros import ZeroCatalog
from riemann.analysis.spacing import (
    normalized_spacings, pair_correlation, gue_pair_correlation,
    number_variance, n_level_density,
)

# Load zeros from catalog
catalog = ZeroCatalog()
zeros = catalog.get_range(1, 1000)

# Compute normalized spacings
spacings = normalized_spacings(zeros)
assert abs(np.mean(spacings) - 1.0) < 0.05  # Sanity check unfolding

# Pair correlation
x, r2 = pair_correlation(spacings)
r2_gue = gue_pair_correlation(x)
residual = r2 - r2_gue  # Where do zeros deviate from GUE?
```

### Creating and Projecting an Embedding

```python
from riemann.embedding.registry import EmbeddingConfig
from riemann.embedding.coordinates import compute_embedding
from riemann.viz.projection import project_pca, project_tsne, project_umap

# Define a diverse embedding
config = EmbeddingConfig(
    name="spectral_features_v1",
    description="Zeros in spectral feature space: spacing + derivatives + density",
    feature_names=(
        "imag_part", "spacing_left", "spacing_right",
        "zeta_derivative_magnitude", "local_density_deviation",
    ),
    scaling="standard",
    zero_range=(1, 500),
)

# Compute embedding
embedding = compute_embedding(config, zeros[:500])  # shape: (500, 5)

# Project with multiple methods for comparison
pca_result = project_pca(embedding, n_components=3)
tsne_result = project_tsne(embedding, n_components=3, perplexity=30)
umap_result = project_umap(embedding, n_components=3, n_neighbors=15)

# Store embedding and projections
from riemann.embedding.storage import save_embedding
save_embedding(config, embedding, projections={
    "pca": pca_result, "tsne": tsne_result, "umap": umap_result,
})
```

### GUE Comparison with Interactive N Slider

```python
from riemann.analysis.rmt import generate_gue, gue_eigenvalue_spacings
from riemann.analysis.spacing import normalized_spacings

# Precompute at multiple N values
zero_spacings = normalized_spacings(zeros)
gue_data = {}
for n in [10, 50, 100, 200, 500, 1000]:
    eigvals = generate_gue(n, num_matrices=200, seed=42)
    gue_data[n] = gue_eigenvalue_spacings(eigvals)

# Visualization creates linked Plotly figure with ipywidgets N slider
# (visualization code in viz/comparison.py)
```

### HDF5 Storage for Large Embedding Arrays

```python
# Source: h5py documentation (https://docs.h5py.org/en/stable/)
import h5py
from pathlib import Path
from riemann.config import DATA_DIR

def save_embedding(config: EmbeddingConfig, embedding: np.ndarray,
                   projections: dict | None = None):
    """Save embedding array and projections to HDF5."""
    hdf5_path = DATA_DIR / "embeddings" / f"{config.name}.h5"
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(hdf5_path), 'w') as f:
        # Store embedding with metadata
        ds = f.create_dataset("embedding", data=embedding,
                              chunks=True, compression="gzip",
                              compression_opts=4)
        ds.attrs["config_name"] = config.name
        ds.attrs["feature_names"] = list(config.feature_names)
        ds.attrs["scaling"] = config.scaling
        ds.attrs["zero_range"] = list(config.zero_range)

        # Store projections as separate groups
        if projections:
            proj_group = f.create_group("projections")
            for name, result in projections.items():
                pg = proj_group.create_group(name)
                pg.create_dataset("coordinates", data=result.coordinates)
                pg.attrs["method"] = result.method
                for k, v in result.metadata.items():
                    try:
                        pg.attrs[k] = v
                    except TypeError:
                        pg.attrs[k] = str(v)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual spacing histograms | Automated unfolding + GUE comparison pipeline | Standard since Odlyzko 1987 | Must normalize spacings by local mean density |
| PCA only for projection | PCA + t-SNE + UMAP ensemble | UMAP introduced 2018 | Multiple projections required for trustworthy structure claims |
| Fixed-N GUE comparison | Variable-N with residual analysis | Standard RMT practice | Finite-size effects are where interesting deviations hide |
| Binned entropy only | KDE + k-NN entropy estimators | k-NN MI from Kraskov 2004 | KDE avoids binning artifacts; k-NN is consistent for MI |
| scikit-learn TSNE 2D only | sklearn TSNE supports n_components=3 | sklearn >=1.1 | 3D t-SNE projections now standard |
| umap-learn early versions | umap-learn 0.5.11 (Jan 2026) | Ongoing | densMAP, parametric UMAP, improved stability |

**Deprecated/outdated:**
- **Rainbow/jet colormaps:** Never use. Use viridis, inferno, or cividis for perceptual uniformity.
- **sklearn.manifold.TSNE with method='exact':** Only for very small datasets (<200 points). Always use default Barnes-Hut for >200 points.

## Open Questions

1. **How many embedding features before t-SNE/UMAP becomes unreliable?**
   - What we know: t-SNE/UMAP work well up to ~50 input dimensions. Beyond that, pre-reduce with PCA to 50 dims first (sklearn docs recommend this).
   - What's unclear: The optimal number of embedding coordinates for zeta zeros specifically.
   - Recommendation: Start with 5-10 features, expand incrementally. Always PCA pre-reduce if >20 features.

2. **Effective N for GUE comparison -- how to determine?**
   - What we know: The effective matrix size N that best matches a range of zeros depends on the height range and number of zeros.
   - What's unclear: The exact mapping from zero range to effective N.
   - Recommendation: Fit N empirically by minimizing chi-squared between zero spacing distribution and GUE(N) spacing distribution across a range of N values.

3. **Precision requirements for embedding features**
   - What we know: |zeta'(rho)| needs arbitrary precision (mpmath). Spacings from imaginary parts can be float64 if zeros are already computed to 50 digits.
   - What's unclear: Whether Hardy Z sign change counting needs more than float64.
   - Recommendation: Default to mpmath for all features involving zeta evaluation; convert to float64 only at the embedding storage stage.

4. **Panel vs ipywidgets for linked views**
   - What we know: ARCHITECTURE.md suggested Panel; ipywidgets is already a dependency.
   - What's unclear: Whether Panel + JupyterLab compatibility is solid on Windows Server 2025. STATE.md flagged this as a concern.
   - Recommendation: Use ipywidgets + Plotly for Phase 2. This combination is well-tested. Panel can be evaluated for Phase 3 if richer dashboards are needed.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >=9.0.2 |
| Config file | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run command | `uv run pytest tests/ -x --timeout=30 -q` |
| Full suite command | `uv run pytest tests/ --timeout=120` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ZERO-01 | Normalized spacings from zeros match expected statistics | unit | `uv run pytest tests/test_analysis/test_spacing.py -x` | Wave 0 |
| ZERO-01 | Pair correlation matches GUE sine kernel for Odlyzko zeros | unit | `uv run pytest tests/test_analysis/test_spacing.py::test_pair_correlation_gue -x` | Wave 0 |
| ZERO-02 | Anomaly detection flags injected deviations | unit | `uv run pytest tests/test_analysis/test_anomaly.py -x` | Wave 0 |
| HDIM-01 | Embedding produces correct shape from config | unit | `uv run pytest tests/test_embedding/test_coordinates.py -x` | Wave 0 |
| HDIM-01 | Embedding config round-trips through workbench | unit | `uv run pytest tests/test_embedding/test_registry.py -x` | Wave 0 |
| HDIM-02 | PCA projection preserves correct variance fraction | unit | `uv run pytest tests/test_viz/test_projection.py::test_pca -x` | Wave 0 |
| HDIM-02 | t-SNE and UMAP produce correct output shapes | unit | `uv run pytest tests/test_viz/test_projection.py::test_tsne_umap -x` | Wave 0 |
| VIZ-03 | Plotly 3D figure renders from projection result | smoke | `uv run pytest tests/test_viz/test_theater.py -x` | Wave 0 |
| RMT-01 | GUE eigenvalue spacings follow Wigner surmise | unit | `uv run pytest tests/test_analysis/test_rmt.py::test_gue_wigner -x` | Wave 0 |
| RMT-02 | Varying N changes spacing distribution toward universal limit | unit | `uv run pytest tests/test_analysis/test_rmt.py::test_gue_n_scaling -x` | Wave 0 |
| INFO-01 | Shannon entropy of GUE spacings matches known value | unit | `uv run pytest tests/test_analysis/test_information.py::test_entropy -x` | Wave 0 |
| INFO-02 | Cross-object comparison returns all expected metrics | unit | `uv run pytest tests/test_analysis/test_information.py::test_cross_comparison -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/ -x --timeout=30 -q`
- **Per wave merge:** `uv run pytest tests/ --timeout=120`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_analysis/__init__.py` -- package init
- [ ] `tests/test_analysis/test_spacing.py` -- covers ZERO-01
- [ ] `tests/test_analysis/test_rmt.py` -- covers RMT-01, RMT-02
- [ ] `tests/test_analysis/test_information.py` -- covers INFO-01, INFO-02
- [ ] `tests/test_analysis/test_anomaly.py` -- covers ZERO-02
- [ ] `tests/test_embedding/__init__.py` -- package init
- [ ] `tests/test_embedding/test_coordinates.py` -- covers HDIM-01
- [ ] `tests/test_embedding/test_registry.py` -- covers HDIM-01
- [ ] `tests/test_embedding/test_storage.py` -- covers HDIM-01 HDF5
- [ ] `tests/test_viz/test_projection.py` -- covers HDIM-02
- [ ] `tests/test_viz/test_theater.py` -- covers VIZ-03
- [ ] Dependencies: `uv add scikit-learn umap-learn` -- required before tests can run

## Sources

### Primary (HIGH confidence)
- [scikit-learn 1.8.0 PCA documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) -- PCA API, solver selection
- [scikit-learn 1.8.0 TSNE documentation](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) -- t-SNE API, perplexity, Barnes-Hut
- [umap-learn 0.5.8 documentation](https://umap-learn.readthedocs.io/) -- UMAP API, n_neighbors, min_dist, densMAP
- [umap-learn PyPI](https://pypi.org/project/umap-learn/) -- version 0.5.11, Jan 2026
- [h5py 3.16.0 dataset documentation](https://docs.h5py.org/en/stable/high/dataset.html) -- chunking, compression, attributes
- [scipy.stats.entropy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html) -- Shannon entropy, KL divergence
- [Plotly Scatter3d documentation](https://plotly.github.io/plotly.py-docs/generated/plotly.graph_objects.Scatter3d.html) -- 3D scatter API
- [Plotly animations](https://plotly.com/python/animations/) -- frame-based animation for projection paths
- Existing Phase 1 source code: types.py, zeros.py, zeta.py, precision.py, conjecture.py, experiment.py

### Secondary (MEDIUM confidence)
- [Montgomery's pair correlation conjecture (Wikipedia)](https://en.wikipedia.org/wiki/Montgomery's_pair_correlation_conjecture) -- r_2(x) formula, historical context
- [Montgomery's pair correlation (Wolfram MathWorld)](https://mathworld.wolfram.com/MontgomerysPairCorrelationConjecture.html) -- sine kernel formula
- [scikit-rmt (GitHub)](https://github.com/AlejandroSantorum/scikit-rmt) -- evaluated but not recommended; custom GUE generation preferred
- [Lempel-Ziv complexity (PyPI)](https://pypi.org/project/lempel-ziv-complexity/) -- evaluated but custom implementation preferred

### Tertiary (LOW confidence)
- [pyspc (GitHub)](https://github.com/carlosqsilva/pyspc) -- SPC library evaluated; too narrow for our anomaly detection needs
- Wigner surmise exact formula for beta=4 (GSE) -- verified against Wikipedia but should be double-checked against Mehta's "Random Matrices" textbook

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- scikit-learn, umap-learn, scipy, numpy are all mature, well-documented, widely used
- Architecture: HIGH -- function-based API matches Phase 1 patterns; separation of embedding/projection/visualization follows established architecture
- Zero statistics algorithms: HIGH -- Montgomery-Odlyzko pair correlation, unfolding by mean density, Wigner surmise are textbook standard
- RMT computation: HIGH -- GUE generation is straightforward linear algebra; eigenvalue computation via LAPACK is battle-tested
- Information theory: MEDIUM -- entropy and MI estimators are standard, but optimal hyperparameters (bins, k for k-NN) need empirical tuning
- Anomaly detection: MEDIUM -- SPC concept is well-established but threshold tuning for zeta zero data is novel
- Pitfalls: HIGH -- visualization artifacts (Pitfall 5 from PITFALLS.md), precision issues, t-SNE distortions are well-documented

**Research date:** 2026-03-19
**Valid until:** 2026-04-19 (30 days -- libraries are stable; mathematical algorithms are timeless)
