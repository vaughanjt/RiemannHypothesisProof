"""Lean 4 formalization pipeline: translate conjectures to proofs.

Modules:
- builder: WSL2 subprocess build runner
- parser: Lean compiler output parser
- tracker: Formalization lifecycle state machine
- translator: Conjecture-to-Lean 4 code generator
- triage: Conjecture ranking and formalization assault
"""
from riemann.formalization.builder import (
    LEAN_PROJECT_DIR,
    LakeBuildResult,
    run_lake_build,
    windows_to_wsl_path,
    wsl_to_windows_path,
)
from riemann.formalization.parser import (
    LeanMessage,
    count_sorry_in_source,
    parse_lean_output,
)
from riemann.formalization.tracker import (
    FormalizationState,
    auto_promote_if_clean,
    create_formalization,
    get_build_history,
    get_formalization,
    list_formalizations,
    record_build,
    update_formalization_state,
)
from riemann.formalization.translator import (
    TranslationResult,
    generate_lean_file,
    translate_conjecture,
)
from riemann.formalization.triage import (
    AssaultOutcome,
    AssaultResult,
    TriageEntry,
    run_formalization_assault,
    triage_conjectures,
)

__all__ = [
    # builder
    "LEAN_PROJECT_DIR",
    "LakeBuildResult",
    "run_lake_build",
    "windows_to_wsl_path",
    "wsl_to_windows_path",
    # parser
    "LeanMessage",
    "parse_lean_output",
    "count_sorry_in_source",
    # tracker
    "FormalizationState",
    "create_formalization",
    "get_formalization",
    "list_formalizations",
    "update_formalization_state",
    "record_build",
    "get_build_history",
    "auto_promote_if_clean",
    # translator
    "TranslationResult",
    "translate_conjecture",
    "generate_lean_file",
    # triage
    "TriageEntry",
    "AssaultResult",
    "AssaultOutcome",
    "triage_conjectures",
    "run_formalization_assault",
]
