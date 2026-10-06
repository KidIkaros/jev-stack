# jev-stack

> A Jev-compatible System 1 decision stack using Laya + EmbeddingGemma 2 + OpenJev flywheel.

**Status:** Work in progress — see [ROADMAP.md](ROADMAP.md) for current tasks.

## What It Is

`jev-stack` is an open-source decision engine that maps structured state (`[STATE][CHOICE][ANSWER]` format) to typed, calibrated decisions via Google's Laya model. It runs locally in ~1.5GB of RAM.

## Quick Start

```bash
pip install git+https://github.com/KidIkaros/jev-stack.git

# Interactive mode
python -m src.cli --interactive

# One-off decision
python -m src.cli --choices "billing,refund,cancel" --state "Customer says: I was charged $50"
```

## Architecture

```
[Input State]
      ↓
EmbeddingGemma 2 (270M, 0.5GB)  →  768-dim embeddings
      ↓
Laya Decision Head (421M)       →  {Choice, Noul, Score} with probabilities
      ↓
OpenJev Flywheel                →  System 2→1 self-refinement
```

See [ADR-001](docs/adr/0001-laya-stack.md) for the architecture decision record.

## Development

```bash
# Setup
pip install -e ".[dev]"
pytest tests/                    # Unit tests only (~0.1s)
RUN_INTEGRATION_TESTS=1 pytest tests/  # Includes real model tests (~30s)
```

## References

- [Laya model](https://modelsystem.one/models/laya/) — Jev-compatible System 1 decision model
- [EmbeddingGemma 2](https://unsloth.ai/docs/models/embeddinggemma-2) — Google's open multimodal embedding model
- [OpenJev](https://github.com/zhangcy122/OpenJev) — Self-evolving decision flywheel
