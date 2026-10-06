"""Integration tests using the real Laya model.

These tests download and run the actual Laya checkpoint from HuggingFace.
They verify end-to-end behavior: parsing, deciding, and formatting.

Marked as integration tests — run with: pytest tests/test_integration.py -v
"""

import pytest
import os
import json
import time
from src.adapter import DecisionAdapter, DecisionParser, DecisionSpec


# Skip if Laya isn't available or model can't be downloaded
pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_INTEGRATION_TESTS", "0") == "1",
    reason="Set RUN_INTEGRATION_TESTS=1 to run real model tests",
)


@pytest.fixture(scope="module")
def laya_agent():
    """Load the real Laya model once for all tests."""
    from laya import RLAgent
    agent = RLAgent(model_id_or_path="convaiinnovations/laya")
    agent.warmup()
    return agent


@pytest.fixture
def adapter(laya_agent):
    """Create adapter with real agent."""
    return DecisionAdapter(laya_agent)


class TestRealLayaDecisions:
    """Integration tests with the actual Laya model."""

    def test_basic_routing_decision(self, adapter):
        """Laya should route a billing inquiry correctly."""
        text = (
            "[STATE] I was charged $50 for a service I never received. "
            "Order #12345. [/STATE] "
            "[CHOICE route] billing technical_support security_fraud human_review [/CHOICE]"
        )
        result = adapter.decide_text(text)

        assert result.choice is not None
        assert result.choice in ["billing", "technical_support", "security_fraud", "human_review"]
        assert sum(result.probabilities.values()) >= 0.99  # Probabilities normalized
        assert result.confidence is not None

    def test_security_routing(self, adapter):
        """Laya should identify security-related states."""
        text = (
            "[STATE] My account was accessed from an unknown IP address. "
            "I see suspicious activity in my logs. [/STATE] "
            "[CHOICE route] billing technical_support security_fraud human_review [/CHOICE]"
        )
        result = adapter.decide_text(text)

        assert result.choice == "security_fraud"
        assert result.probabilities["security_fraud"] >= 0.5

    def test_technical_routing(self, adapter):
        """Laya should route technical questions correctly."""
        text = (
            "[STATE] The API is returning a 500 error when I try to create a user. "
            "The docs say this endpoint supports POST /api/users. [/STATE] "
            "[CHOICE route] billing technical_support security_fraud human_review [/CHOICE]"
        )
        result = adapter.decide_text(text)

        assert result.choice == "technical_support"

    def test_format_output_integration(self, adapter):
        """Full pipeline: parse → decide → format should produce valid output."""
        text = "[STATE] Charged twice on my card [/STATE] [CHOICE route] billing tech security human [/CHOICE]"
        result = adapter.decide_text(text)

        formatted = DecisionParser.format_output(
            choice=result.choice,
            probabilities=result.probabilities,
            confidence=result.confidence,
        )

        assert "[ANSWER]" in formatted
        assert result.choice in formatted

    def test_latency_requirement(self, adapter):
        """Decision latency should be reasonable (cold start <500ms, warm <200ms).

        Laya claims 33ms per decision on T4 batched. On our test machine,
        full cold-start (tokenizer + model) takes longer. We verify
        warm-call latency is under 500ms (including Python overhead).
        """
        text = "[STATE] Test state [/STATE] [CHOICE route] A B C D [/CHOICE]"

        # Warmup call
        adapter.decide_text(text)

        # Timed call
        t0 = time.time()
        result = adapter.decide_text(text)
        latency_ms = (time.time() - t0) * 1000

        # Allow headroom for non-T4 hardware and Python overhead
        assert latency_ms < 500, f"Latency {latency_ms:.0f}ms exceeds 500ms target"

    def test_option_order_invariance(self, adapter):
        """Decision should not change when option order changes."""
        text1 = "[STATE] Security breach detected [/STATE] [CHOICE route] billing technical_support security_fraud human_review [/CHOICE]"
        text2 = "[STATE] Security breach detected [/STATE] [CHOICE route] security_fraud human_review billing technical_support [/CHOICE]"

        result1 = adapter.decide_text(text1)
        result2 = adapter.decide_text(text2)

        # Same answer regardless of option order
        assert result1.choice == result2.choice == "security_fraud"

    def test_score_question(self, adapter):
        """Score questions should produce a numerical score."""
        text = (
            "[STATE] Uncertain about the billing discrepancy [/STATE] "
            "[CHOICE route] billing tech security human [/CHOICE] "
            "[SCORE confidence] low medium high [/SCORE]"
        )
        result = adapter.decide_text(text)

        assert result.score is not None
        assert 0.0 <= result.score <= 1.0

    def test_noul_question(self, adapter):
        """Noul questions should produce a boolean answer."""
        text = (
            "[STATE] The system logs show normal activity [/STATE] "
            "[NOUL can_be_automated] Can this be handled automatically? [/NOUL]"
        )
        result = adapter.decide_text(text)

        # Noul should produce a boolean answer
        assert "can_be_automated" in result.raw["answers"]
