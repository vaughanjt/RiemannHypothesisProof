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

__all__ = [
    "Anomaly",
    "AnalogyMapping",
    "BostConnesResult",
    "ModularFormResult",
    "SpectralResult",
    "TraceFormulaResult",
    "bost_connes_kms_values",
    "bost_connes_partition",
    "compute_bost_connes",
    "compute_q_expansion",
    "create_analogy_mapping",
    "cross_object_comparison",
    "detect_anomalies",
    "eisenstein_series",
    "eigenvalue_spacings",
    "fit_effective_n",
    "generate_goe",
    "generate_gse",
    "generate_gue",
    "hecke_eigenvalues",
    "clear_cache",
    "get_lfunction",
    "get_modular_form",
    "get_number_field",
    "query_lmfdb",
    "LMFDBError",
    "gue_pair_correlation",
    "lempel_ziv_complexity",
    "load_analogy_from_workbench",
    "log_anomalies_to_workbench",
    "mutual_information_spacings",
    "n_level_density",
    "normalized_spacings",
    "number_variance",
    "pair_correlation",
    "phase_transition_scan",
    "save_analogy_to_workbench",
    "spacing_entropy",
    "test_correspondence",
    "update_analogy_confidence",
    "chebyshev_psi_exact",
    "compute_spectrum",
    "compute_trace_formula",
    "construct_berry_keating_box",
    "construct_berry_keating_smooth",
    "explicit_formula_terms",
    "spectral_comparison",
    "weil_explicit_psi",
    "wigner_surmise",
]
