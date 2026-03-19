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
from riemann.analysis.modular_forms import (
    ModularFormResult,
    compute_q_expansion,
    eisenstein_series,
    hecke_eigenvalues,
)

__all__ = [
    "Anomaly",
    "ModularFormResult",
    "compute_q_expansion",
    "cross_object_comparison",
    "detect_anomalies",
    "eisenstein_series",
    "eigenvalue_spacings",
    "fit_effective_n",
    "generate_goe",
    "generate_gse",
    "generate_gue",
    "hecke_eigenvalues",
    "gue_pair_correlation",
    "lempel_ziv_complexity",
    "log_anomalies_to_workbench",
    "mutual_information_spacings",
    "n_level_density",
    "normalized_spacings",
    "number_variance",
    "pair_correlation",
    "spacing_entropy",
    "wigner_surmise",
]
