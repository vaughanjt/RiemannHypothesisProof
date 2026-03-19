"""Conjecture triage: ranking, time-boxing, and formalization assault.

Claude triages the workbench and picks optimal attack order per conjecture.
Considers confidence, Mathlib proximity, novelty, and prior attempts.
Time-boxes per conjecture: if sorry count isn't decreasing after N attempts, move on.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from riemann.formalization.builder import run_lake_build
from riemann.formalization.tracker import (
    FormalizationState,
    auto_promote_if_clean,
    get_build_history,
    get_formalization,
    list_formalizations,
    record_build,
    update_formalization_state,
)
from riemann.formalization.translator import generate_lean_file
from riemann.workbench.conjecture import get_conjecture, list_conjectures
from riemann.workbench.db import init_db

logger = logging.getLogger(__name__)

# Domain proximity to Mathlib (higher = more Mathlib infrastructure exists)
_MATHLIB_PROXIMITY = {
    "spectral": 0.9,     # Gamma, zeta, functional equation formalized
    "modular": 0.85,     # Jacobi theta, modular forms, Hecke operators in Mathlib
    "trace": 0.8,        # VonMangoldt, Euler products, L-series
    "padic": 0.75,       # PadicVal, PadicInt in Mathlib
    "ncg": 0.5,          # Limited NCG in Mathlib
    "tda": 0.3,          # Minimal topology formalization relevant to TDA
    "dynamics": 0.4,     # Some ergodic theory in Mathlib
    "analogy": 0.2,      # Meta-mathematical, hard to formalize
    "cross_domain": 0.3, # Depends on specific domains
    "default": 0.5,
}


@dataclass
class TriageEntry:
    """A conjecture ranked for formalization attempt."""
    conjecture_id: str
    score: float           # 0.0-1.0, higher = higher priority
    confidence: float | None
    domain: str
    reason: str            # Human-readable explanation of ranking
    formalization_state: str | None  # Current state if any formalization exists
    statement: str = ""


@dataclass
class AssaultOutcome:
    """Result of attempting to formalize one conjecture."""
    conjecture_id: str
    formalization_id: str | None
    initial_state: str
    final_state: str
    sorry_count: int
    builds_attempted: int
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class AssaultResult:
    """Result of the full formalization assault."""
    total_conjectures: int
    attempted: int
    skipped: int
    outcomes: list[AssaultOutcome] = field(default_factory=list)


def _infer_domain_from_conjecture(conjecture: dict) -> str:
    """Infer mathematical domain from conjecture tags/statement."""
    tags = conjecture.get("tags") or []
    statement = (conjecture.get("statement") or "").lower()
    description = (conjecture.get("description") or "").lower()
    text = " ".join(tags).lower() + " " + statement + " " + description

    domain_keywords = {
        "spectral": ["spectral", "eigenvalue", "operator", "hamiltonian", "berry-keating"],
        "trace": ["trace", "selberg", "weil", "explicit formula", "chebyshev", "primes"],
        "modular": ["modular", "eisenstein", "hecke", "fourier", "q-expansion", "ramanujan"],
        "padic": ["p-adic", "padic", "kubota", "leopoldt"],
        "tda": ["persistent", "homology", "topolog", "betti"],
        "dynamics": ["lyapunov", "orbit", "dynamical", "ergodic", "attractor"],
        "ncg": ["noncommutative", "bost-connes", "kms", "partition function"],
        "analogy": ["analogy", "correspondence", "mapping"],
    }
    for domain, keywords in domain_keywords.items():
        if any(kw in text for kw in keywords):
            return domain
    return "default"


def triage_conjectures(
    db_path: str | Path | None = None,
) -> list[TriageEntry]:
    """Rank workbench conjectures by formalization viability.

    Scoring formula:
      score = 0.4 * confidence + 0.3 * mathlib_proximity + 0.2 * continuation_bonus + 0.1 * novelty

    Excludes conjectures with evidence_level >= 3 (already formally proved).
    Excludes conjectures with formalization_state == proof_complete.

    Returns list sorted by score descending.
    """
    init_db(db_path)
    conjectures = list_conjectures(db_path=db_path)

    # Get existing formalizations for continuation scoring
    existing = list_formalizations(db_path=db_path)
    form_by_conj: dict[str, dict] = {}
    for f in existing:
        cid = f["conjecture_id"]
        if cid not in form_by_conj or _state_rank(f["formalization_state"]) > _state_rank(form_by_conj[cid].get("formalization_state", "")):
            form_by_conj[cid] = f

    entries = []
    for conj in conjectures:
        if conj.get("evidence_level", 0) >= 3:
            continue

        cid = conj["id"]
        form = form_by_conj.get(cid)
        form_state = form["formalization_state"] if form else None

        if form_state == "proof_complete":
            continue

        confidence = conj.get("confidence") or 0.0
        domain = _infer_domain_from_conjecture(conj)
        proximity = _MATHLIB_PROXIMITY.get(domain, 0.5)

        continuation = 0.0
        if form_state == "proof_attempted":
            continuation = 1.0
        elif form_state == "statement_formalized":
            continuation = 0.5

        novelty = 1.0 if form is None else 0.3

        score = (
            0.4 * confidence
            + 0.3 * proximity
            + 0.2 * continuation
            + 0.1 * novelty
        )

        reason_parts = []
        if confidence > 0.7:
            reason_parts.append(f"high confidence ({confidence:.2f})")
        if proximity > 0.7:
            reason_parts.append(f"strong Mathlib coverage ({domain})")
        if continuation > 0:
            reason_parts.append(f"continuation ({form_state})")
        if novelty > 0.5:
            reason_parts.append("never attempted")
        reason = "; ".join(reason_parts) if reason_parts else "baseline candidate"

        entries.append(TriageEntry(
            conjecture_id=cid,
            score=round(score, 4),
            confidence=confidence,
            domain=domain,
            reason=reason,
            formalization_state=form_state,
            statement=conj.get("statement", ""),
        ))

    entries.sort(key=lambda e: e.score, reverse=True)
    return entries


def _state_rank(state: str) -> int:
    """Numeric rank of formalization state for comparison."""
    return {
        "not_formalized": 0,
        "statement_formalized": 1,
        "proof_attempted": 2,
        "proof_complete": 3,
    }.get(state, -1)


def run_formalization_assault(
    max_conjectures: int = 10,
    max_attempts_per: int = 3,
    build_timeout: int = 300,
    db_path: str | Path | None = None,
) -> AssaultResult:
    """Execute the full formalization assault.

    Triages conjectures, then for each (up to max_conjectures):
    1. If not yet translated: generate .lean file via translator
       (leaves state at statement_formalized)
    2. Advance state to proof_attempted (required by _VALID_TRANSITIONS
       before auto_promote_if_clean can reach proof_complete)
    3. Run lake build
    4. Record build result
    5. Check auto-promotion (proof_attempted -> proof_complete on clean build)
    6. If sorry count not decreasing after max_attempts_per builds, move on

    Args:
        max_conjectures: Maximum conjectures to attempt.
        max_attempts_per: Maximum build attempts per conjecture before moving on.
        build_timeout: Timeout per lake build in seconds.
        db_path: Database path.

    Returns:
        AssaultResult with per-conjecture outcomes.
    """
    init_db(db_path)
    ranked = triage_conjectures(db_path=db_path)
    total = len(ranked)
    outcomes = []
    attempted = 0
    skipped = 0

    for entry in ranked[:max_conjectures]:
        cid = entry.conjecture_id
        logger.info(f"Assault: conjecture {cid} (score={entry.score}, domain={entry.domain})")

        existing_forms = list_formalizations(conjecture_id=cid, db_path=db_path)
        form = existing_forms[0] if existing_forms else None

        if form and form["formalization_state"] == "proof_complete":
            outcomes.append(AssaultOutcome(
                conjecture_id=cid,
                formalization_id=form["id"],
                initial_state="proof_complete",
                final_state="proof_complete",
                sorry_count=0,
                builds_attempted=0,
                skipped=True,
                skip_reason="already proof_complete",
            ))
            skipped += 1
            continue

        initial_state = form["formalization_state"] if form else "not_formalized"
        formalization_id = form["id"] if form else None

        if form is None:
            try:
                result = generate_lean_file(cid, db_path=db_path)
                formalization_id = result.formalization_id
                logger.info(f"  Translated to {result.lean_file_path}")
            except Exception as e:
                logger.warning(f"  Translation failed for {cid}: {e}")
                outcomes.append(AssaultOutcome(
                    conjecture_id=cid,
                    formalization_id=None,
                    initial_state="not_formalized",
                    final_state="not_formalized",
                    sorry_count=0,
                    builds_attempted=0,
                    skipped=True,
                    skip_reason=f"translation failed: {e}",
                ))
                skipped += 1
                continue

        # CRITICAL: Advance state machine to proof_attempted before build loop.
        # generate_lean_file leaves state at statement_formalized. The tracker's
        # _VALID_TRANSITIONS only allows proof_attempted -> proof_complete, so
        # auto_promote_if_clean would silently fail without this advancement.
        # Also handles existing formalizations stuck at statement_formalized.
        current_form = get_formalization(formalization_id, db_path=db_path)
        if current_form and current_form["formalization_state"] == "statement_formalized":
            update_formalization_state(
                formalization_id,
                "proof_attempted",
                sorry_count=current_form.get("sorry_count", 1),
                last_build_success=False,
                db_path=db_path,
            )
            logger.info(f"  Advanced state: statement_formalized -> proof_attempted")

        attempted += 1
        builds_done = 0
        last_sorry = None
        current_sorry = 0

        for attempt in range(max_attempts_per):
            try:
                build_result = run_lake_build(timeout_seconds=build_timeout)
                record_build(formalization_id, build_result, db_path=db_path)
                builds_done += 1
                current_sorry = build_result.sorry_count

                logger.info(
                    f"  Build {attempt + 1}: success={build_result.success}, "
                    f"sorry={build_result.sorry_count}, errors={build_result.error_count}"
                )

                if build_result.success and build_result.sorry_count == 0:
                    promoted = auto_promote_if_clean(formalization_id, db_path=db_path)
                    if promoted:
                        logger.info(f"  AUTO-PROMOTED to FORMAL_PROOF!")
                    break

                if last_sorry is not None and current_sorry >= last_sorry:
                    logger.info(f"  Sorry count not decreasing ({last_sorry} -> {current_sorry}), moving on")
                    break

                last_sorry = current_sorry

            except Exception as e:
                logger.warning(f"  Build attempt {attempt + 1} failed: {e}")
                builds_done += 1

        final_form = get_formalization(formalization_id, db_path=db_path)
        final_state = final_form["formalization_state"] if final_form else "unknown"

        outcomes.append(AssaultOutcome(
            conjecture_id=cid,
            formalization_id=formalization_id,
            initial_state=initial_state,
            final_state=final_state,
            sorry_count=current_sorry,
            builds_attempted=builds_done,
        ))

    return AssaultResult(
        total_conjectures=total,
        attempted=attempted,
        skipped=skipped,
        outcomes=outcomes,
    )
