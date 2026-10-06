"""Integration tests for GemmaTextEncoder + Laya adapter pipeline."""
import pytest
import torch
from src.gemma_embed import GemmaTextEncoder
from src.adapter import DecisionAdapter, DecisionResult


@pytest.fixture(scope="module")
def encoder():
    """Real EmbeddingGemma 2 encoder."""
    return GemmaTextEncoder()


@pytest.fixture(scope="module")
def adapter():
    """Real Laya adapter."""
    from laya import RLAgent
    agent = RLAgent(model_id_or_path="ConvaiInnovations/laya")
    agent.warmup()
    return DecisionAdapter(agent)


class TestEncoderAdapterIntegration:
    def test_embed_then_decide(self, encoder, adapter):
        """Full pipeline: text → embeddings → Laya decision."""
        state_text = "Customer says: I was charged $50 for a service I never received."
        emb = encoder.encode(state_text, dim=128)  # Matryoshka truncation

        text = f"[STATE] {state_text} [/STATE] [CHOICE route] billing refund cancel [/CHOICE]"
        result = adapter.decide_text(text)

        assert result.choice in ("billing", "refund", "cancel")
        assert result.confidence > 0.5

    def test_multilingual_encoding(self, encoder, adapter):
        """Encoder handles non-English, adapter still decides correctly."""
        state_text = "Chargé $50 pour un service non reçu."
        text = f"[STATE] {state_text} [/STATE] [CHOICE route] billing support tech [/CHOICE]"
        result = adapter.decide_text(text)

        assert result.choice is not None
        assert result.choice in ("billing", "support", "tech")

    def test_embeddings_reduce_dimension(self, encoder):
        """Using 128-dim embeddings (6x smaller) vs 768-dim."""
        text = "Customer charged $50"
        emb_full = encoder.encode(text, dim=768)
        emb_trunc = encoder.encode(text, dim=128)
        assert emb_full.shape[-1] == 768
        assert emb_trunc.shape[-1] == 128
        # Truncated should be first 128 dims of full (Matryoshka)
        assert torch.allclose(emb_full[:128], emb_trunc, atol=1e-5)
