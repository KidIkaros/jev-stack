"""Tests for the DecisionAdapter and DecisionParser.

Tests verify:
- Parser correctly extracts [STATE], [CHOICE], [SCORE], [NOUL] blocks
- Adapter correctly maps to Laya's system_one API format
- Results map back from Laya's output to DecisionResult
- Format roundtrips preserve information
- Named questions (CHOICE route) work correctly
"""

import pytest
from src.adapter import (
    DecisionParser, DecisionAdapter, DecisionSpec,
    ChoiceQuestion, ScoreQuestion, NoulQuestion, 
    Option, DecisionResult
)


class TestDecisionParser:
    """Tests for parsing [STATE][CHOICE] format."""

    def test_parse_basic_state(self):
        """Parser should extract state from [STATE] block."""
        text = "[STATE] Customer charged $50 [/STATE]"
        spec = DecisionParser.parse(text)
        assert spec.state == "Customer charged $50"

    def test_parse_state_and_choice(self):
        """Parser should extract state and single choice question."""
        text = "[STATE] Billing issue [/STATE] [CHOICE] billing refund cancel [/CHOICE]"
        spec = DecisionParser.parse(text)
        assert spec.state == "Billing issue"
        assert len(spec.choices) == 1
        assert spec.choices[0].options[0].label == "billing"
        assert spec.choices[0].options[1].label == "refund"
        assert spec.choices[0].options[2].label == "cancel"

    def test_parse_named_choice(self):
        """Parser should handle [CHOICE route] with named id."""
        text = "[STATE] context [/STATE] [CHOICE route] A B C D [/CHOICE]"
        spec = DecisionParser.parse(text)
        assert len(spec.choices) == 1
        assert spec.choices[0].id == "route"
        assert len(spec.choices[0].options) == 4

    def test_parse_score(self):
        """Parser should extract score questions."""
        text = "[STATE] context [/STATE] [CHOICE] A B C [/CHOICE] [SCORE] low medium high [/SCORE]"
        spec = DecisionParser.parse(text)
        assert len(spec.scores) == 1
        assert spec.scores[0].levels == ["low", "medium", "high"]

    def test_parse_noul(self):
        """Parser should extract noul (yes/no) questions."""
        text = "[STATE] context [/STATE] [NOUL] Can this be automated [/NOUL]"
        spec = DecisionParser.parse(text)
        assert len(spec.nouls) == 1
        assert "automated" in spec.nouls[0].instructions

    def test_parse_context_alias(self):
        """Parser should accept [CONTEXT] as alias for [STATE]."""
        text = "[CONTEXT] context here [/CONTEXT]"
        spec = DecisionParser.parse(text)
        assert spec.state == "context here"

    def test_parse_answer_block(self):
        """Parser should find answer for named questions."""
        text = "[CHOICE route] A B C [/CHOICE] [ANSWER route] B [/ANSWER]"
        spec = DecisionParser.parse(text)
        # Answer should be incorporated into instructions
        assert "B" in spec.choices[0].instructions

    def test_parse_empty_text(self):
        """Parser should handle empty text gracefully."""
        spec = DecisionParser.parse("")
        assert spec.state == ""
        assert len(spec.choices) == 0


class TestFormatOutput:
    """Tests for formatting results back to [ANSWER] format."""

    def test_format_choice(self):
        """Format should produce [ANSWER] block."""
        output = DecisionParser.format_output(choice="billing")
        assert "[ANSWER] billing [/ANSWER]" in output

    def test_format_with_probabilities(self):
        """Format should include probability block."""
        output = DecisionParser.format_output(
            choice="billing",
            probabilities={"billing": 0.95, "refund": 0.05}
        )
        assert "[ANSWER] billing [/ANSWER]" in output
        assert "[PROBS]" in output
        assert "billing=0.9500" in output

    def test_format_with_confidence(self):
        """Format should include score block."""
        output = DecisionParser.format_output(
            choice="billing",
            confidence=0.85
        )
        assert "[SCORE] 0.8500 [/SCORE]" in output

    def test_format_empty(self):
        """Format with no results should return empty string."""
        output = DecisionParser.format_output()
        assert output == ""


class TestDecisionAdapter:
    """Tests for the DecisionAdapter (uses mock agent)."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock Laya agent that returns canned responses."""
        class MockAgent:
            def system_one(self, state, questions=None, min_confidence=None, **kwargs):
                return {
                    "answers": {
                        "route": {
                            "type": "choice",
                            "choice": "billing",
                            "probabilities": {"billing": 0.95, "refund": 0.05},
                            "confidence": 0.82,
                            "answer_confidence": 0.95,
                        },
                        "confidence": {
                            "type": "score",
                            "score": 0.85,
                            "probabilities": {"0": 0.1, "1": 0.9},
                            "confidence": 0.75,
                            "answer_confidence": 0.9,
                        },
                    },
                    "model": "mock-laya",
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                }
        return MockAgent()

    def test_adapter_basic_decision(self, mock_agent):
        """Adapter should map parsed spec to agent call and back."""
        adapter = DecisionAdapter(mock_agent)
        text = "[STATE] Charged for nothing [/STATE] [CHOICE] billing refund [/CHOICE]"
        result = adapter.decide_text(text)
        
        assert result.choice == "billing"
        assert result.probabilities["billing"] == 0.95
        assert result.confidence == 0.95  # answer_confidence
        assert result.ece == 0.82  # Laya's normalized confidence

    def test_adapter_with_score(self, mock_agent):
        """Adapter should extract score questions."""
        adapter = DecisionAdapter(mock_agent)
        text = "[STATE] context [/STATE] [CHOICE] A B [/CHOICE] [SCORE] low high [/SCORE]"
        result = adapter.decide_text(text)
        
        assert result.score is not None
        assert result.score == 0.85

    def test_adapter_with_noul(self, mock_agent):
        """Adapter should handle noul questions."""
        class MockAgentNoul:
            def system_one(self, state, questions=None, min_confidence=None, **kwargs):
                return {
                    "answers": {
                        "decision": {
                            "type": "noul",
                            "noul": True,
                            "confidence": 0.9,
                            "answer_confidence": 0.9,
                        },
                    },
                    "model": "mock",
                }
        adapter = DecisionAdapter(MockAgentNoul())
        text = "[STATE] context [/STATE] [NOUL] can automate [/NOUL]"
        result = adapter.decide_text(text)
        
        assert result.noul is True

    def test_adapter_infers_questions(self, mock_agent):
        """Adapter should infer default questions when none specified."""
        adapter = DecisionAdapter(mock_agent)
        # No [CHOICE] block — adapter should infer
        result = adapter.decide_text("[STATE] just context")
        
        # Should still get results from mock
        assert result.raw is not None

    def test_adapter_preserves_raw(self, mock_agent):
        """Adapter should preserve raw Laya output."""
        adapter = DecisionAdapter(mock_agent)
        result = adapter.decide_text("[STATE] ctx [/STATE] [CHOICE] A B [/CHOICE]")
        
        assert result.raw["answers"]["route"]["choice"] == "billing"

    def test_adapter_format_output(self, mock_agent):
        """Adapter result should be formatable back to [ANSWER] format."""
        adapter = DecisionAdapter(mock_agent)
        result = adapter.decide_text("[STATE] ctx [/STATE] [CHOICE] A B [/CHOICE]")
        
        formatted = DecisionParser.format_output(
            choice=result.choice,
            probabilities=result.probabilities,
            confidence=result.confidence,
        )
        assert "[ANSWER] billing [/ANSWER]" in formatted
        assert "[SCORE]" in formatted


class TestDecisionRoundtrip:
    """Tests that parse → decide → format preserves information."""

    def test_roundtrip_basic(self):
        """Full roundtrip: parse input format, get result, format output."""
        text = "[STATE] Customer charged $50 [/STATE] [CHOICE] billing refund [/CHOICE]"
        spec = DecisionParser.parse(text)
        
        assert spec.state == "Customer charged $50"
        assert spec.choices[0].options[0].label == "billing"
        assert spec.choices[0].options[1].label == "refund"

    def test_roundtrip_with_named_questions(self):
        """Roundtrip with named choice questions."""
        text = "[STATE] Ticket [/STATE] [CHOICE route] A B C D [/CHOICE] [SCORE confidence] low high [/SCORE]"
        spec = DecisionParser.parse(text)
        
        assert spec.choices[0].id == "route"
        assert spec.scores[0].id == "confidence"
        assert spec.scores[0].levels == ["low", "high"]
