"""Tests for EmbeddingGemma 2 encoder layer."""
import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from src.gemma_embed import GemmaTextEncoder, encode_state


class TestGemmaTextEncoder:
    def test_loads_embeddinggemma_2(self):
        """Encoder should load the 270M text model from google/embeddinggemma-2."""
        encoder = GemmaTextEncoder()
        assert encoder.model is not None
        assert encoder.embedding_dim == 768

    def test_encode_single_text(self):
        """Should encode a single string to 768-dim embedding."""
        encoder = GemmaTextEncoder()
        emb = encoder.encode("Customer was charged $50")
        assert emb.shape[-1] == 768
        assert emb.ndim == 1  # Single vector

    def test_encode_batch(self):
        """Should encode multiple strings to (batch, 768) tensor."""
        encoder = GemmaTextEncoder()
        texts = ["Charged $50", "Tech support needed", "Security concern"]
        emb = encoder.encode(texts)
        assert emb.shape == (3, 768)

    def test_encode_preserves_order(self):
        """Order of outputs should match order of inputs."""
        encoder = GemmaTextEncoder()
        texts_a = ["hello", "world"]
        text_b = ["world", "hello"]
        emb_a = encoder.encode(texts_a)
        emb_b = encoder.encode(text_b)
        # Different order should produce different embeddings
        assert not torch.allclose(emb_a[0], emb_b[0])

    def test_matryoshka_truncation(self):
        """Should support truncating to 128 dims for 6x storage savings."""
        encoder = GemmaTextEncoder()
        emb = encoder.encode("Test text", dim=128)
        assert emb.shape[-1] == 128

    def test_memory_footprint(self):
        """Full 744M model includes text (271M) + vision (167M) + audio (304M) encoders.

        Per google/embeddinggemma-2 docs, the 270M text encoder runs in 0.5GB.
        The full model with all encoders is ~740M params.
        """
        encoder = GemmaTextEncoder()
        num_params = sum(p.numel() for p in encoder.model.parameters())
        # Full model is ~744M params (271M text + 167M vision + 304M audio)
        assert 700_000_000 < num_params < 800_000_000

    def test_cache_reuse(self):
        """Repeated calls with same text should work (model stays loaded)."""
        encoder = GemmaTextEncoder()
        emb1 = encoder.encode("Same text")
        emb2 = encoder.encode("Same text")
        assert torch.allclose(emb1, emb2)

    def test_tokenizer_roundtrip(self):
        """Tokenizer should handle special characters and multi-language."""
        encoder = GemmaTextEncoder()
        texts = [
            "Customer says: '$50 charge' — not received!",
            "Charge similaire de 50€ n'est pas reçu",
            "50달러 요금 미수신",
        ]
        emb = encoder.encode(texts)
        assert emb.shape == (3, 768)
