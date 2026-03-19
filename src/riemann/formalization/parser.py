"""Parse Lean 4 compiler output into structured messages."""
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class LeanMessage:
    """A single Lean 4 compiler message."""

    file: str
    line: int
    col: int
    severity: str  # "error", "warning", "info"
    message: str


# Lean 4 output format: filepath:line:col: severity: message
_LEAN_MSG_RE = re.compile(
    r"^(.+?):(\d+):(\d+):\s*(error|warning|info):\s*(.+)",
    re.MULTILINE,
)
_SORRY_DECL_RE = re.compile(r"declaration uses 'sorry'")

# For source-level sorry counting: match `sorry` as a word token,
# but NOT inside line comments (--) or block comments (/- ... -/)
_SORRY_TOKEN_RE = re.compile(r"\bsorry\b")
_LINE_COMMENT_RE = re.compile(r"--.*$", re.MULTILINE)
_BLOCK_COMMENT_RE = re.compile(r"/-.*?-/", re.DOTALL)
_STRING_LITERAL_RE = re.compile(r'"[^"]*"')


def parse_lean_output(output: str) -> tuple[list[LeanMessage], int]:
    """Parse Lean build output into structured messages and sorry count.

    Args:
        output: Raw stdout+stderr from ``lake build``.

    Returns:
        Tuple of (list of LeanMessage, sorry_declaration_count).
        sorry_declaration_count is the number of "declaration uses 'sorry'" warnings.
    """
    messages = []
    for m in _LEAN_MSG_RE.finditer(output):
        messages.append(
            LeanMessage(
                file=m.group(1),
                line=int(m.group(2)),
                col=int(m.group(3)),
                severity=m.group(4),
                message=m.group(5).strip(),
            )
        )
    sorry_count = len(_SORRY_DECL_RE.findall(output))
    return messages, sorry_count


def count_sorry_in_source(source: str) -> int:
    """Count sorry tokens in Lean source, excluding comments and strings.

    Args:
        source: Raw .lean file content.

    Returns:
        Number of sorry tokens found outside comments and string literals.
    """
    # Strip comments and strings before counting
    cleaned = _BLOCK_COMMENT_RE.sub("", source)
    cleaned = _LINE_COMMENT_RE.sub("", cleaned)
    cleaned = _STRING_LITERAL_RE.sub("", cleaned)
    return len(_SORRY_TOKEN_RE.findall(cleaned))
