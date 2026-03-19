"""Analysis modules for zero distribution statistics and random matrix theory."""

from riemann.analysis.rmt import (
    eigenvalue_spacings,
    fit_effective_n,
    generate_goe,
    generate_gse,
    generate_gue,
    wigner_surmise,
)
from riemann.analysis.spacing import (
    gue_pair_correlation,
    n_level_density,
    normalized_spacings,
    number_variance,
    pair_correlation,
)

from riemann.analysis.anomaly import (
    Anomaly,
    detect_anomalies,
    log_anomalies_to_workbench,
)
from riemann.analysis.information import (
    cross_object_comparison,
    lempel_ziv_complexity,
    mutual_information_spacings,
    spacing_entropy,
)
from riemann.analysis.lmfdb_client import (
    LMFDBError,
    clear_cache,
    get_lfunction,
    get_modular_form,
    get_number_field,
    query_lmfdb,
)
from riemann.analysis.modular_forms import (
    ModularFormResult,
    compute_q_expansion,
    eisenstein_series,
    hecke_eigenvalues,
)
from riemann.analysis.ncg import (
    BostConnesResult,
    bost_connes_kms_values,
    bost_connes_partition,
    compute_bost_connes,
    phase_transition_scan,
)
from riemann.analysis.spectral import (
    SpectralResult,
    compute_spectrum,
    construct_berry_keating_box,
    construct_berry_keating_smooth,
    spectral_comparison,
)
from riemann.analysis.trace_formula import (
    TraceFormulaResult,
    chebyshev_psi_exact,
    compute_trace_formula,
    explicit_formula_terms,
    weil_explicit_psi,
)
from riemann.analysis.analogy import (
    AnalogyMapping,
    create_analogy_mapping,
    load_analogy_from_workbench,
    save_analogy_to_workbench,
    test_correspondence,
    update_analogy_confidence,
)
from riemann.analysis.padic import (
    PadicNumber,
    kubota_leopoldt_zeta,
    padic_fractal_tree_data,
    padic_from_rational,
)
from riemann.analysis.tda import (
    PersistenceResult,
    compare_persistence_diagrams,
    compute_persistence,
    persistence_summary,
)
from riemann.analysis.dynamics import (
    DynamicsResult,
    analyze_dynamics,
    compute_orbit,
    find_fixed_points,
    logistic_map,
    lyapunov_exponent,
    zeta_map,
)
from riemann.analysis.conjecture_gen import (
    ExperimentSuggestion,
    analyze_results,
    generate_conjecture,
    suggest_experiments,
)

__all__ = [
    "Anomaly",
    "AnalogyMapping",
    "BostConnesResult",
    "DynamicsResult",
    "ExperimentSuggestion",
    "LMFDBError",
    "ModularFormResult",
    "PadicNumber",
    "PersistenceResult",
    "SpectralResult",
    "TraceFormulaResult",
    "analyze_dynamics",
    "analyze_results",
    "bost_connes_kms_values",
    "bost_connes_partition",
    "chebyshev_psi_exact",
    "clear_cache",
    "compare_persistence_diagrams",
    "compute_bost_connes",
    "compute_orbit",
    "compute_persistence",
    "compute_q_expansion",
    "compute_spectrum",
    "compute_trace_formula",
    "construct_berry_keating_box",
    "construct_berry_keating_smooth",
    "create_analogy_mapping",
    "cross_object_comparison",
    "detect_anomalies",
    "eigenvalue_spacings",
    "eisenstein_series",
    "explicit_formula_terms",
    "find_fixed_points",
    "fit_effective_n",
    "generate_conjecture",
    "generate_goe",
    "generate_gse",
    "generate_gue",
    "get_lfunction",
    "get_modular_form",
    "get_number_field",
    "gue_pair_correlation",
    "hecke_eigenvalues",
    "kubota_leopoldt_zeta",
    "lempel_ziv_complexity",
    "load_analogy_from_workbench",
    "log_anomalies_to_workbench",
    "logistic_map",
    "lyapunov_exponent",
    "mutual_information_spacings",
    "n_level_density",
    "normalized_spacings",
    "number_variance",
    "padic_fractal_tree_data",
    "padic_from_rational",
    "pair_correlation",
    "persistence_summary",
    "phase_transition_scan",
    "query_lmfdb",
    "save_analogy_to_workbench",
    "spacing_entropy",
    "spectral_comparison",
    "suggest_experiments",
    "test_correspondence",
    "update_analogy_confidence",
    "weil_explicit_psi",
    "wigner_surmise",
    "zeta_map",
]
