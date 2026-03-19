"""Tests for Lean 4 compiler output parser."""
import pytest

from riemann.formalization.parser import LeanMessage, parse_lean_output, count_sorry_in_source


class TestParseLeanOutput:
    """Tests for parse_lean_output function."""

    def test_empty_input_returns_empty(self):
        """Empty string produces no messages and zero sorry count."""
        messages, sorry_count = parse_lean_output("")
        assert messages == []
        assert sorry_count == 0

    def test_single_error_parsed(self):
        """A single error line is parsed into a LeanMessage."""
        output = "foo.lean:10:4: error: unknown identifier 'bar'"
        messages, sorry_count = parse_lean_output(output)
        assert len(messages) == 1
        msg = messages[0]
        assert msg.file == "foo.lean"
        assert msg.line == 10
        assert msg.col == 4
        assert msg.severity == "error"
        assert msg.message == "unknown identifier 'bar'"
        assert sorry_count == 0

    def test_sorry_warning_counted(self):
        """Warning about 'sorry' is counted in sorry_count."""
        output = "foo.lean:5:0: warning: declaration uses 'sorry'"
        messages, sorry_count = parse_lean_output(output)
        assert len(messages) == 1
        assert messages[0].severity == "warning"
        assert sorry_count == 1

    def test_multiple_messages_mixed_severity(self, sample_lean_error_output):
        """Multiple lines with mixed severities are all parsed."""
        messages, sorry_count = parse_lean_output(sample_lean_error_output)
        assert len(messages) == 3
        assert messages[0].severity == "error"
        assert messages[1].severity == "warning"
        assert messages[2].severity == "warning"
        assert sorry_count == 2

    def test_clean_output_no_messages(self, sample_lean_clean_output):
        """Clean build output produces no messages."""
        messages, sorry_count = parse_lean_output(sample_lean_clean_output)
        assert messages == []
        assert sorry_count == 0

    def test_info_severity_parsed(self):
        """Info messages are parsed correctly."""
        output = "RiemannProofs/Basic.lean:17:0: info: RiemannHypothesis : Prop"
        messages, sorry_count = parse_lean_output(output)
        assert len(messages) == 1
        assert messages[0].severity == "info"
        assert messages[0].message == "RiemannHypothesis : Prop"
        assert sorry_count == 0

    def test_lean_message_is_frozen(self):
        """LeanMessage dataclass is frozen (immutable)."""
        msg = LeanMessage(file="a.lean", line=1, col=0, severity="error", message="test")
        with pytest.raises(AttributeError):
            msg.line = 2


class TestCountSorryInSource:
    """Tests for count_sorry_in_source function."""

    def test_sorry_in_code_counted(self):
        """Sorry tokens in proof code are counted."""
        source = "theorem t : True := by\n  sorry"
        assert count_sorry_in_source(source) == 1

    def test_sorry_in_line_comment_excluded(self):
        """Sorry in -- line comments is not counted."""
        source = "-- sorry is not real\ntheorem t : True := True.intro"
        assert count_sorry_in_source(source) == 0

    def test_sorry_in_block_comment_excluded(self):
        """Sorry in /- block comments -/ is not counted."""
        source = "/- sorry -/\ntheorem t : True := True.intro"
        assert count_sorry_in_source(source) == 0

    def test_sorry_in_string_excluded(self):
        """Sorry inside string literals is not counted."""
        source = 'def msg := "sorry not sorry"\ntheorem t : True := True.intro'
        assert count_sorry_in_source(source) == 0

    def test_multiple_sorry_counted(self):
        """Multiple sorry tokens in code are counted correctly."""
        source = "sorry\nsorry\nsorry"
        assert count_sorry_in_source(source) == 3

    def test_mixed_sorry_and_comments(self, sample_lean_source_with_sorry):
        """Source with sorry tokens in code returns correct count."""
        assert count_sorry_in_source(sample_lean_source_with_sorry) == 3

    def test_all_sorry_in_comments(self, sample_lean_source_clean):
        """Source with sorry only in comments/strings returns 0."""
        assert count_sorry_in_source(sample_lean_source_clean) == 0
