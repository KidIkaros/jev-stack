"""EmbeddingGemma 2 text encoder layer.

Wraps the google/embeddinggemma-2 model (270M parameters, 768-dim embeddings)
from Google DeepMind. Runs locally in ~0.5GB RAM via Sentence Transformers.

Reference: https://huggingface.co/google/embeddinggemma-2
"""
from __future__ import annotations

from typing import Union

import torch
from sentence_transformers import SentenceTransformer


class GemmaTextEncoder:
    """Text encoder using EmbeddingGemma 2 (270M parameter text encoder).

    Produces 768-dimensional embeddings from text input. Supports Matryoshka
    truncation to 128 dimensions for 6x storage savings.

    Args:
        model_name: HuggingFace model ID. Defaults to google/embeddinggemma-2.
        device: Device to load model on. Defaults to CPU.
    """

    DEFAULT_MODEL = "google/embeddinggemma-2"
    DEFAULT_DIM = 768

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        normalize_embeddings: bool = True,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize_embeddings = normalize_embeddings
        self.model: SentenceTransformer = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.DEFAULT_DIM

    @torch.no_grad()
    def encode(
        self,
        texts: Union[str, list[str]],
        dim: int = 768,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Encode text(s) to embeddings.

        Args:
            texts: String or list of strings to embed.
            dim: Output dimension (Matryoshka truncation). Default 768 (full).
                 128 dims gives ~6x storage savings.
            batch_size: Batch size for encoding multiple texts.

        Returns:
            For single string: 1D tensor of shape (dim,).
            For list of strings: 2D tensor of shape (batch, dim).
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=self.normalize_embeddings,
            batch_size=batch_size,
            device=self.device,
        )

        if dim < self.embedding_dim:
            embeddings = embeddings[:, :dim]

        if single:
            embeddings = embeddings[0]

        return embeddings  # stays on self.device (cpu or cuda)

    def __call__(self, texts: Union[str, list[str]], dim: int = 768) -> torch.Tensor:
        """Alias for encode()."""
        return self.encode(texts, dim=dim)


def encode_state(state_text: str, dim: int = 768, encoder: GemmaTextEncoder | None = None) -> torch.Tensor:
    """Convenience function to encode a state string.

    Creates a temporary encoder if none is provided (for one-off usage).
    For repeated usage, pass a pre-loaded encoder to avoid model reload overhead.

    Args:
        state_text: Text to encode.
        dim: Output embedding dimension.
        encoder: Pre-loaded encoder instance (optional).

    Returns:
        1D tensor of shape (dim,).
    """
    if encoder is None:
        encoder = GemmaTextEncoder()
    return encoder.encode(state_text, dim=dim)
