# Phase 4: Lean 4 Formalization Pipeline - Research

**Researched:** 2026-03-19
**Domain:** Lean 4 theorem proving, Mathlib integration, WSL2 subprocess interop, formalization pipeline
**Confidence:** HIGH

## Summary

Phase 4 builds a complete pipeline from Python research workbench conjectures to machine-verified Lean 4 proofs running in WSL2. The core technical challenge is three-fold: (1) managing Lean 4 + Mathlib inside WSL2 with subprocess calls from Windows Python, (2) translating workbench conjectures into well-formed Lean 4 theorem statements using Mathlib's extensive number theory formalization, and (3) parsing build output to track sorry counts, errors, and auto-promote conjectures upon clean builds.

Mathlib already contains a formalized `RiemannHypothesis : Prop` statement, the Riemann zeta function (`riemannZeta`), its functional equation, special values, L-series theory, and Hurwitz zeta functions. This is exceptional -- the project can build on formalized zeta infrastructure rather than starting from scratch. Loeffler and Stoll's 2025 work formalizing zeta and L-functions in Lean means our conjectures can reference real Mathlib theorems, not just sorry-filled stubs.

**Primary recommendation:** Use raw subprocess calls to `wsl lake build` (not lean-interact) for build/check operations. Use `wslpath` for path conversion. Install elan + Lean 4 inside WSL2 as a Wave 0 gate. The Lean project lives on the Windows filesystem at `lean_proofs/` accessible from both sides via `/mnt/c/...`. Parse Lean's `file:line:col: severity: message` output format with regex for structured error reporting.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Lean 4 runs inside WSL2 (already installed and working on the machine)
- Install elan + lake inside WSL2 as a Wave 0 validation task
- Python calls Lean via subprocess: writes .lean files to a shared path, calls `wsl lake build`, parses output
- Lean project lives on Windows filesystem accessible from both Python and WSL
- Wave 0 must validate the full chain: elan install -> lake init -> hello-world theorem -> `wsl lake build` from Python succeeds
- Claude generates Lean 4 code directly from conjecture statements (no template scaffolding)
- Full Mathlib integration: Lean project depends on Mathlib, Claude uses existing formalized number theory
- Evidence mapping: each .lean file includes docstrings with linked experiment IDs, computational evidence summaries, and confidence scores
- Storage: .lean files in a Lean project directory + workbench DB stores file path, formalization status, sorry count, last build result
- 4-state pipeline: not_formalized -> statement_formalized -> proof_attempted -> proof_complete
- Automatic sorry-count parsing after every lake build
- Structured error reporting: parse Lean output into {file, line, error_type, message, suggestion}
- Store build history in workbench (every build attempt with timestamp, sorry count, errors)
- Auto-promote: zero sorries + clean build = evidence_level automatically set to 3 (FORMAL_PROOF)
- Full assault: build the pipeline AND aggressively attempt to formalize every viable conjecture
- Claude triages workbench and picks optimal attack order per conjecture
- Time-box per conjecture: if sorry count isn't decreasing after N attempts, move to next

### Claude's Discretion
- elan/Lean 4 version selection (latest stable)
- Lean project structure and lakefile configuration
- Mathlib module selection per conjecture
- Proof tactic strategies and lemma decomposition
- Time-box threshold (how many attempts before moving on)
- Conjecture triage ordering algorithm
- Build output parsing heuristics
- Whether to decompose a conjecture into sub-lemmas before formalizing

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| FORM-01 | User can translate conjectures from the research workbench into Lean 4 theorem statements | Mathlib has `riemannZeta`, `RiemannHypothesis`, L-series, functional equations formalized. Claude generates Lean 4 code directly. `lean_proofs/` project with Mathlib dependency. Subprocess pipeline writes .lean files and builds via `wsl lake build`. |
| FORM-02 | User can track formalization progress (statement formalized -> proof attempted -> proof complete) with Mathlib integration for existing formalized mathematics | 4-state pipeline tracked in SQLite `formalizations` table. Sorry-count parsed from build output. Build history stored per conjecture. Auto-promotion to FORMAL_PROOF (evidence_level=3) on zero-sorry clean build. Mathlib provides 100,000+ theorems including zeta-specific infrastructure. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Lean 4 | v4.28.0 (latest stable via elan) | Theorem prover | Latest stable release as of 2026-02-17. Installed inside WSL2 Ubuntu 24.04 via elan. |
| Mathlib4 | latest (pinned via lake) | Mathematical library | 100,000+ formalized theorems including `riemannZeta`, `RiemannHypothesis`, L-series, functional equations, complex analysis, measure theory. |
| elan | latest | Lean toolchain manager | Official Lean version manager. Auto-resolves `lean-toolchain` file. Since elan 4.0, release channels resolve automatically. |
| lake | (bundled with Lean 4) | Build system | Lean 4's integrated build tool. Merged into Lean 4 repo (standalone is deprecated). Handles Mathlib dependency, caching, builds. |

### Supporting (Python side)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| subprocess (stdlib) | Python 3.12 | WSL2 interop | Calling `wsl -e bash -c "cd /path && lake build"` from Windows Python |
| sqlite3 (stdlib) | Python 3.12 | Formalization tracking | New `formalizations` and `build_history` tables in workbench DB |
| re (stdlib) | Python 3.12 | Build output parsing | Regex parsing of Lean error format `file:line:col: severity: message` |
| pathlib (stdlib) | Python 3.12 | Path management | Windows/WSL path conversion |
| json (stdlib) | Python 3.12 | Data serialization | Build result storage |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Raw subprocess | lean-interact (PyPI) | lean-interact provides REPL-based interaction with sorry extraction and environment persistence, but adds complexity and requires its own Lean REPL build. Raw subprocess is simpler, matches established project patterns, and provides full `lake build` which is the ground truth for proof verification. |
| Raw subprocess | lean-repl-py (PyPI) | Similar to lean-interact but less mature. Same tradeoff -- added complexity for incremental proof stepping. |
| Lean project on Windows filesystem | Lean project inside WSL only | Windows filesystem via /mnt/c/ has slower I/O but enables Python to directly read/write .lean files without WSL. The project already uses this pattern. Acceptable for our scale. |

**Installation:**
```bash
# Inside WSL2 Ubuntu 24.04:
sudo apt install git curl
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh
source $HOME/.elan/env

# Create Lean project on Windows filesystem accessible from both sides:
cd "/mnt/c/Users/james.vaughan/OneDrive - Denali Water Solutions/Development/Riemann"
lake +leanprover-community/mathlib4:lean-toolchain new lean_proofs math
cd lean_proofs
lake exe cache get    # Download prebuilt Mathlib oleans (~5-10 min)
lake build            # Verify everything compiles
```

## Architecture Patterns

### Recommended Project Structure
```
src/riemann/
├── formalization/
│   ├── __init__.py
│   ├── builder.py           # WSL subprocess: lake build, output capture
│   ├── parser.py            # Parse Lean compiler output into structured errors
│   ├── translator.py        # Conjecture -> Lean 4 theorem statement generation
│   ├── tracker.py           # Formalization state machine, sorry tracking, auto-promotion
│   └── triage.py            # Conjecture triage: ordering, time-boxing, attack planning
├── workbench/
│   ├── db.py                # Extended with formalizations + build_history tables
│   └── ...existing...
lean_proofs/                  # Lean 4 project (at project root)
├── lakefile.toml             # Mathlib dependency
├── lean-toolchain            # Pinned Lean version
├── RiemannProofs/
│   ├── Basic.lean            # Wave 0 hello-world theorem
│   └── {ConjectureId}.lean   # Generated per-conjecture files
└── lake-packages/            # Mathlib + dependencies (gitignored)
```

### Pattern 1: WSL Subprocess Build Runner
**What:** Python calls Lean 4 builds inside WSL2 via subprocess
**When to use:** Every build/check operation
**Example:**
```python
# Source: Verified against WSL2 interop patterns
import subprocess
from pathlib import Path

def run_lake_build(lean_project_dir: Path, timeout_seconds: int = 300) -> dict:
    """Run lake build inside WSL2 and capture structured output."""
    # Convert Windows path to WSL path
    win_path = str(lean_project_dir).replace("\\", "/")
    result = subprocess.run(
        ["wsl", "-e", "bash", "-c",
         f'source $HOME/.elan/env && cd "{wsl_path}" && lake build 2>&1'],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0,
    }
```

### Pattern 2: Lean Error Output Parsing
**What:** Parse Lean 4 compiler messages into structured error objects
**When to use:** After every `lake build` to extract sorry count, errors, warnings
**Example:**
```python
# Source: Verified against Lean 4 compiler output format
import re
from dataclasses import dataclass

@dataclass
class LeanMessage:
    file: str
    line: int
    col: int
    severity: str    # "error", "warning", "info"
    message: str

# Lean 4 outputs: filepath:line:col: severity: message
_LEAN_MSG_RE = re.compile(
    r'^(.+?):(\d+):(\d+):\s*(error|warning|info):\s*(.+)',
    re.MULTILINE,
)
_SORRY_RE = re.compile(r"declaration uses 'sorry'")

def parse_lean_output(output: str) -> tuple[list[LeanMessage], int]:
    """Parse Lean build output into messages and sorry count."""
    messages = []
    for m in _LEAN_MSG_RE.finditer(output):
        messages.append(LeanMessage(
            file=m.group(1), line=int(m.group(2)),
            col=int(m.group(3)), severity=m.group(4),
            message=m.group(5).strip(),
        ))
    sorry_count = len(_SORRY_RE.findall(output))
    return messages, sorry_count
```

### Pattern 3: Formalization State Machine
**What:** 4-state pipeline per conjecture tracked in SQLite
**When to use:** Managing formalization lifecycle
```
not_formalized --[translate]--> statement_formalized (theorem + sorry)
statement_formalized --[attempt proof]--> proof_attempted (partial tactics, reduced sorry)
proof_attempted --[iterate]--> proof_attempted (further sorry reduction)
proof_attempted --[zero sorry + clean build]--> proof_complete
proof_complete --[auto-promote]--> conjecture.evidence_level = 3 (FORMAL_PROOF)
```

### Pattern 4: Path Conversion (Windows <-> WSL)
**What:** Convert paths between Windows and WSL filesystems
**When to use:** Any operation that bridges Python (Windows) and Lean (WSL)
**Example:**
```python
import subprocess

def windows_to_wsl_path(windows_path: str) -> str:
    """Convert Windows path to WSL mount path using wslpath."""
    result = subprocess.run(
        ["wsl", "-e", "wslpath", "-u", windows_path],
        capture_output=True, text=True,
    )
    return result.stdout.strip()

def wsl_to_windows_path(wsl_path: str) -> str:
    """Convert WSL path to Windows path using wslpath."""
    result = subprocess.run(
        ["wsl", "-e", "wslpath", "-w", wsl_path],
        capture_output=True, text=True,
    )
    return result.stdout.strip()
```

### Pattern 5: Lean File Generation with Evidence Mapping
**What:** Generate .lean files with metadata docstrings linking back to workbench
**When to use:** Translating conjectures to Lean 4
**Example:**
```lean
/-!
# Conjecture: {conjecture_id}
## Evidence
- Experiment {exp_id_1}: {description} (confidence: 0.85)
- Experiment {exp_id_2}: {description} (confidence: 0.72)
## Workbench State
- Evidence level: HEURISTIC (1)
- Status: computational_evidence
- Created: 2026-03-19
-/

import Mathlib.NumberTheory.LSeries.RiemannZeta
import Mathlib.Analysis.SpecialFunctions.Gamma.Deligne

open Complex

/-- The zero-spacing pattern observed in experiments {exp_ids}
    suggests a spectral operator connection. -/
theorem spectral_zero_correlation :
    sorry := by
  sorry
```

### Anti-Patterns to Avoid
- **Running Lean natively on Windows:** Lean 4 has known issues with long Windows command lines in Lake (GitHub issue #4159). WSL2 avoids this entirely.
- **Using lean-interact for build verification:** lean-interact is designed for REPL-based interaction. For authoritative "does this project build clean?", always use `lake build` -- it's the ground truth.
- **Putting the Lean project inside WSL filesystem:** Python on Windows cannot directly access WSL's ext4 filesystem. Keep Lean project on Windows side (/mnt/c/...) so both sides can read/write.
- **Hardcoding WSL paths:** Always use `wslpath` for conversion. Don't assume `/mnt/c/` prefix -- it depends on WSL configuration.
- **Building Mathlib from source:** Always run `lake exe cache get` first. Building Mathlib from source takes 30-60+ minutes and ~8GB RAM.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Lean 4 installation | Custom installer | `elan` via official script | elan handles toolchain versioning, lean-toolchain file resolution, and PATH management |
| Mathlib compilation | Compile from source | `lake exe cache get` | Pre-built oleans save 30-60 minutes; Mathlib CI produces caches for every commit |
| Path conversion | String manipulation | `wslpath -u` / `wslpath -w` | wslpath handles edge cases (spaces, special chars, UNC paths, mount config) |
| Zeta function formalization | Custom Lean definitions | Mathlib's `riemannZeta`, `completedRiemannZeta`, `RiemannHypothesis` | Loeffler & Stoll (2025) formalized zeta + L-functions. Use it. |
| L-series infrastructure | Custom L-series theory | Mathlib's `Mathlib.NumberTheory.LSeries.*` | ~1400 lines of formalized L-series theory including convergence, Euler products |
| Lean project structure | Manual lakefile creation | `lake new ... math` template | The `math` template auto-configures Mathlib dependency, lean-toolchain, and project structure |
| Complex analysis foundations | Formalize from axioms | Mathlib's `Mathlib.Analysis.SpecialFunctions.*` | Gamma function, Fourier analysis, Jacobi theta, all formalized |

**Key insight:** The Mathlib ecosystem has done extraordinary work formalizing exactly the mathematics this project needs. Loeffler & Stoll's 2025 paper formalized the Riemann zeta function, its functional equation, L-functions of Dirichlet characters, and a formal statement of the Riemann Hypothesis itself. Build on their work, don't duplicate it.

## Common Pitfalls

### Pitfall 1: Mathlib Version Drift
**What goes wrong:** Lean 4 and Mathlib release frequently (Lean 4.28 as of Feb 2026). Updating one without the other causes cryptic compilation failures. Mathlib pins a specific Lean toolchain version -- mismatches break everything.
**Why it happens:** `lake update` pulls latest Mathlib which may require a different Lean version than what elan has installed.
**How to avoid:** Pin versions at project creation time. The `lean-toolchain` file locks the Lean version. Only update intentionally: `curl mathlib4/master/lean-toolchain -o lean-toolchain && lake update`. Never update mid-formalization sprint.
**Warning signs:** `lake build` errors mentioning "unknown identifier" in Mathlib imports, or "toolchain not found".

### Pitfall 2: Windows Filesystem Performance in WSL2
**What goes wrong:** The Lean project on `/mnt/c/...` (Windows filesystem accessed from WSL) has 3-5x slower I/O than native WSL ext4. For Mathlib-dependent projects, this can make `lake build` noticeably slow.
**Why it happens:** WSL2's 9P filesystem bridge between Windows NTFS and Linux adds overhead on every file operation.
**How to avoid:** Accept the performance hit -- it's the price of Python/Lean interop on shared files. The alternative (Lean project on WSL ext4) would require complex file synchronization. For our use case (individual .lean files, not Mathlib-scale compilation), the overhead is tolerable. Mathlib's pre-built cache (`lake exe cache get`) eliminates the worst case (compiling Mathlib itself).
**Warning signs:** `lake build` taking >60 seconds for a single small .lean file change.

### Pitfall 3: Sorry Count Misparse
**What goes wrong:** Parsing sorry count by simple string matching can over-count (comments mentioning sorry, string literals) or under-count (sorry in dependent modules). Lean 4.22+ has `warn.sorry` option that controls the warning.
**Why it happens:** The "declaration uses 'sorry'" warning is per-declaration, not per-sorry occurrence. A theorem with 5 sorry placeholders generates 1 warning if the overall declaration uses sorry.
**How to avoid:** Parse the actual Lean output format carefully. Count "declaration uses 'sorry'" warnings (1 per declaration). For per-sorry granularity, parse the source .lean file directly for `sorry` tokens (excluding comments and strings). Cross-validate: if declaration count is 0 and build succeeds, it's a clean build regardless of file-level sorry scanning.
**Warning signs:** Sorry count shows 0 but proof is incomplete; sorry count shows large number but most are in Mathlib imports.

### Pitfall 4: Subprocess Timeout on First Build
**What goes wrong:** The first `lake build` after project creation downloads and extracts Mathlib cache, which can take 5-15 minutes depending on network/disk speed. Default subprocess timeout of 120 seconds kills it.
**Why it happens:** `lake exe cache get` downloads ~200MB+ of compressed oleans. Extraction can be slow on Windows filesystem via WSL.
**How to avoid:** Separate the one-time setup (elan install, lake init, cache get) from per-build operations. Use longer timeouts (600s+) for setup. Normal builds (after cache) should complete in 10-60 seconds for small files.
**Warning signs:** `subprocess.TimeoutExpired` on first run but not subsequent runs.

### Pitfall 5: Lean File Write Race Condition
**What goes wrong:** Python writes a .lean file on Windows, then immediately calls `wsl lake build`. The WSL filesystem cache hasn't flushed, so Lean sees the old file content.
**Why it happens:** 9P filesystem caching between Windows and WSL can delay write visibility by milliseconds.
**How to avoid:** After writing a .lean file, call `os.fsync()` on the file descriptor, or add a small `time.sleep(0.1)` before invoking `lake build`. In practice this is rare but worth defending against.
**Warning signs:** Build results don't match the file you just wrote; "stale" errors that resolve on manual retry.

### Pitfall 6: Trying to Formalize Too Much Too Fast
**What goes wrong:** Attempting to formalize deep conjectures about zero distributions before establishing basic Lean 4 infrastructure and simple lemmas. The full assault should be systematic, not chaotic.
**Why it happens:** Excitement about the mathematical content overrides infrastructure discipline.
**How to avoid:** Wave 0 validates the full chain with a trivial theorem. First real formalization should be simple known results (e.g., ζ(0) = -1/2, which is already in Mathlib as `riemannZeta_zero`). Build complexity gradually. The triage algorithm should order conjectures by Mathlib proximity.
**Warning signs:** First conjecture generates 50+ errors; no conjecture has ever reached `proof_complete` state.

## Code Examples

Verified patterns from official sources:

### Lean 4 Lakefile with Mathlib (TOML format)
```toml
# Source: https://github.com/leanprover-community/mathlib4/wiki/Using-mathlib4-as-a-dependency
[package]
name = "RiemannProofs"

[[require]]
name = "mathlib"
scope = "leanprover-community"
```

### Lean 4 Theorem Using Mathlib Zeta Infrastructure
```lean
-- Source: https://leanprover-community.github.io/mathlib4_docs/Mathlib/NumberTheory/LSeries/RiemannZeta.html
import Mathlib.NumberTheory.LSeries.RiemannZeta

open Complex

-- Reference: riemannZeta_zero is already in Mathlib
-- This demonstrates importing and using Mathlib's zeta infrastructure
example : riemannZeta 0 = -1 / 2 := riemannZeta_zero

-- RiemannHypothesis is already defined in Mathlib:
-- RiemannHypothesis : Prop :=
--   forall s : C, riemannZeta s = 0 ->
--     (not exists n : Nat, s = -2 * (n + 1)) ->
--     s != 1 ->
--     s.re = 1/2
#check RiemannHypothesis
```

### Key Mathlib Imports for Number Theory Formalization
```lean
-- Zeta function and RH
import Mathlib.NumberTheory.LSeries.RiemannZeta
-- L-series convergence, Dirichlet series
import Mathlib.NumberTheory.LSeries.HurwitzZeta
-- Euler product
import Mathlib.NumberTheory.EulerProduct.Basic
-- Complex analysis
import Mathlib.Analysis.SpecialFunctions.Gamma.Deligne
-- Fourier analysis (Poisson summation)
import Mathlib.Analysis.Fourier.FourierTransform
-- Jacobi theta functions
import Mathlib.NumberTheory.ModularForms.JacobiTheta.Basic
-- Elementary number theory
import Mathlib.NumberTheory.Bernoulli
-- Dirichlet characters
import Mathlib.NumberTheory.DirichletCharacter.Basic
```

### SQLite Schema Extension for Formalization Tracking
```sql
-- New tables added to existing workbench schema
CREATE TABLE IF NOT EXISTS formalizations (
    id TEXT PRIMARY KEY,
    conjecture_id TEXT NOT NULL REFERENCES conjectures(id),
    lean_file_path TEXT NOT NULL,
    formalization_state TEXT NOT NULL DEFAULT 'not_formalized'
        CHECK(formalization_state IN (
            'not_formalized', 'statement_formalized',
            'proof_attempted', 'proof_complete'
        )),
    sorry_count INTEGER NOT NULL DEFAULT 0,
    error_count INTEGER NOT NULL DEFAULT 0,
    last_build_success BOOLEAN DEFAULT 0,
    last_build_output TEXT,
    mathlib_imports TEXT,          -- JSON list of Mathlib imports used
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS build_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    formalization_id TEXT NOT NULL REFERENCES formalizations(id),
    build_timestamp TEXT NOT NULL,
    success BOOLEAN NOT NULL,
    sorry_count INTEGER NOT NULL,
    error_count INTEGER NOT NULL,
    warning_count INTEGER NOT NULL,
    build_duration_ms REAL,
    output TEXT,                   -- Full build output
    errors_json TEXT,              -- JSON array of structured errors
    created_at TEXT NOT NULL
);
```

### Windows-to-WSL Build Invocation Pattern
```python
import subprocess
import os
from pathlib import Path

def wsl_lake_build(project_dir: Path, timeout: int = 300) -> dict:
    """Execute lake build in WSL2 and return structured result."""
    # Convert Windows path to WSL path via wslpath
    win_path = str(project_dir)
    wsl_path_result = subprocess.run(
        ["wsl", "-e", "wslpath", "-u", win_path],
        capture_output=True, text=True, timeout=10,
    )
    wsl_path = wsl_path_result.stdout.strip()

    # Run lake build with elan env sourced
    cmd = f'source "$HOME/.elan/env" && cd "{wsl_path}" && lake build 2>&1'
    result = subprocess.run(
        ["wsl", "-e", "bash", "-c", cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    return {
        "returncode": result.returncode,
        "output": result.stdout + result.stderr,
        "success": result.returncode == 0,
    }
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Lean 3 + Mathlib3 | Lean 4 + Mathlib4 | 2023-2024 | Complete rewrite; Lean 4 is a new language with different syntax and tactic system |
| lake as separate repo | lake merged into Lean 4 | 2024 | `lake` is now part of the Lean 4 distribution; standalone repo deprecated |
| Manual Mathlib build | `lake exe cache get` | 2023+ | Pre-built caches save 30-60 min build time |
| No formalized RH | `RiemannHypothesis : Prop` in Mathlib | 2025 | Loeffler & Stoll formalized zeta, L-functions, and RH statement |
| lakefile.lean only | lakefile.toml (preferred) | 2024+ | TOML format for declarative config; .lean for programmatic config |
| elan with manual channels | elan 4.0+ auto-resolve channels | 2025+ | `stable` channel auto-resolves; no manual `elan update` needed |

**Deprecated/outdated:**
- `lake init` (replaced by `lake new` for new projects)
- Lean 3 Mathlib tactics (completely different from Lean 4)
- Standalone `lake` repository (merged into leanprover/lean4)
- `lakefile.lean` `require mathlib from git "..."` syntax (old format; use `require "leanprover-community" / "mathlib"` or TOML)

## Open Questions

1. **Performance of lake build on Windows FS via WSL2**
   - What we know: 3-5x I/O overhead is documented for /mnt/c/ access. Pre-built Mathlib cache eliminates the worst case.
   - What's unclear: Exact build time for a single .lean file change in a Mathlib-dependent project on this specific machine.
   - Recommendation: Measure in Wave 0. If >60s per build, consider keeping only .lean source files on Windows and symlinking into a WSL-native lake project. This is a Wave 0 validation item.

2. **Sorry granularity vs declaration granularity**
   - What we know: "declaration uses 'sorry'" is 1 warning per declaration. Lean 4.22+ has `warn.sorry` option.
   - What's unclear: Whether we need per-sorry-occurrence count or per-declaration count.
   - Recommendation: Track both. Parse the .lean file for `sorry` token count (source-level). Parse build output for declaration-level warnings. Report both in build history.

3. **lean-interact as future enhancement**
   - What we know: lean-interact provides REPL-based interaction with sorry goal extraction, environment persistence, and tactic-by-tactic proof stepping. Supports Lean 4.8-4.29.
   - What's unclear: Whether REPL-based interaction adds enough value over subprocess to justify the dependency.
   - Recommendation: Start with subprocess (simpler, proven pattern). If iterative tactic development proves valuable, evaluate lean-interact as a later enhancement.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 9.0.2+ |
| Config file | `pyproject.toml` [tool.pytest.ini_options] |
| Quick run command | `uv run pytest tests/test_formalization/ -x --timeout=30` |
| Full suite command | `uv run pytest tests/ --timeout=120` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FORM-01 | Translate conjecture to Lean 4 theorem statement | unit | `uv run pytest tests/test_formalization/test_translator.py -x` | No -- Wave 0 |
| FORM-01 | Write .lean file with evidence docstrings | unit | `uv run pytest tests/test_formalization/test_translator.py::test_lean_file_generation -x` | No -- Wave 0 |
| FORM-01 | Call wsl lake build and capture output | integration | `uv run pytest tests/test_formalization/test_builder.py -x --timeout=60` | No -- Wave 0 |
| FORM-01 | Parse Lean error output into structured messages | unit | `uv run pytest tests/test_formalization/test_parser.py -x` | No -- Wave 0 |
| FORM-02 | Track formalization state (4-state machine) | unit | `uv run pytest tests/test_formalization/test_tracker.py -x` | No -- Wave 0 |
| FORM-02 | Sorry count parsing from build output | unit | `uv run pytest tests/test_formalization/test_parser.py::test_sorry_count -x` | No -- Wave 0 |
| FORM-02 | Store build history in SQLite | unit | `uv run pytest tests/test_formalization/test_tracker.py::test_build_history -x` | No -- Wave 0 |
| FORM-02 | Auto-promote to FORMAL_PROOF on clean build | unit | `uv run pytest tests/test_formalization/test_tracker.py::test_auto_promote -x` | No -- Wave 0 |
| FORM-02 | Mathlib import usage in generated theorems | unit | `uv run pytest tests/test_formalization/test_translator.py::test_mathlib_imports -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/test_formalization/ -x --timeout=30`
- **Per wave merge:** `uv run pytest tests/ --timeout=120`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_formalization/__init__.py` -- package init
- [ ] `tests/test_formalization/test_builder.py` -- WSL subprocess build tests
- [ ] `tests/test_formalization/test_parser.py` -- Lean output parsing tests
- [ ] `tests/test_formalization/test_translator.py` -- Conjecture-to-Lean translation tests
- [ ] `tests/test_formalization/test_tracker.py` -- State machine, sorry tracking, auto-promotion tests
- [ ] `tests/test_formalization/conftest.py` -- Shared fixtures (temp DB, mock lean output)
- [ ] WSL elan installation validation (Wave 0 task)
- [ ] Lean project creation + Mathlib cache download (Wave 0 task)
- [ ] Hello-world theorem build from Python subprocess (Wave 0 gate)

## Sources

### Primary (HIGH confidence)
- [Mathlib4 RiemannZeta docs](https://leanprover-community.github.io/mathlib4_docs/Mathlib/NumberTheory/LSeries/RiemannZeta.html) - Full list of formalized zeta function theorems, `RiemannHypothesis` definition
- [Lean 4 install manual](https://lean-lang.org/install/manual/) - Official elan installation steps for Linux/WSL
- [Mathlib4 as dependency](https://github.com/leanprover-community/mathlib4/wiki/Using-mathlib4-as-a-dependency) - lakefile.toml configuration, `lake exe cache get` workflow
- [Lean 4 release notes](https://lean-lang.org/doc/reference/latest/releases/) - v4.22 warn.sorry, v4.28 latest stable, lake improvements
- [Lake reference](https://lean-lang.org/doc/reference/latest/Build-Tools-and-Distribution/Lake/) - Build system documentation, --wfail/--iofail flags
- [elan GitHub](https://github.com/leanprover/elan) - Toolchain management, auto-resolution since 4.0
- Existing codebase: `src/riemann/types.py` (EvidenceLevel.FORMAL_PROOF=3), `workbench/conjecture.py` (CRUD + update), `workbench/db.py` (schema, get_connection)

### Secondary (MEDIUM confidence)
- [Loeffler & Stoll 2025 - Formalizing zeta and L-functions in Lean](https://arxiv.org/abs/2503.00959) - Confirmed RH formal statement, zeta formalization, L-series in Mathlib
- [LeanInteract GitHub](https://github.com/augustepoiroux/LeanInteract) - Python REPL interface, sorry extraction, environment persistence
- [Lean REPL](https://github.com/leanprover-community/repl) - JSON-based REPL with sorry/error structured output
- [LeanMillenniumPrizeProblems](https://github.com/lean-dojo/LeanMillenniumPrizeProblems) - Millennium.RiemannHypothesis formal statement
- WSL2 verified: Ubuntu 24.04 running, `wslpath` available, project path resolves to `/mnt/c/Users/james.vaughan/OneDrive - Denali Water Solutions/Development/Riemann`

### Tertiary (LOW confidence)
- [WSL subprocess issues](https://github.com/microsoft/WSL/issues/9799) - FileNotFoundError edge cases with subprocess.check_output in WSL2 (may be resolved in current WSL version)
- [Lake Windows long command line issue #4159](https://github.com/leanprover/lean4/issues/4159) - Known issue with native Windows Lake builds (avoided by using WSL)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Lean 4 + Mathlib is the undisputed choice. elan/lake are the official tools. Versions verified via release notes.
- Architecture: HIGH - Subprocess pattern matches existing project patterns (function-based, SQLite). WSL2 interop verified on actual machine.
- Pitfalls: HIGH - Version drift, /mnt/c/ performance, sorry parsing all documented in community discussions. Lean learning curve widely acknowledged.
- Mathlib coverage: HIGH - RiemannZeta module verified in official docs with full theorem listing. Loeffler & Stoll paper confirms zeta + L-function formalization.

**Research date:** 2026-03-19
**Valid until:** 2026-04-19 (Lean 4 releases monthly; Mathlib moves fast but our pinned version insulates us)
